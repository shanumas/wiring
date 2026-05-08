"""
Four-stage estimation pipeline for canalisation drawings.

Stage 1 — Route length    : traced from PDF vector paths near each annotation;
                            falls back to annotation-position span.
Stage 2 — Vertical routing: derived from mounting height deltas (ok/uk).
Stage 3 — Cost mapping    : (system, width_mm, fire_rating) × total length → SEK.
Stage 4 — Confidence      : 0–1 score per line item, with audit notes.
"""
import fitz
import json
import math
from pathlib import Path
from collections import defaultdict

COMPONENTS_JSON = "components.json"
PDF_PATH        = "pdf/2.pdf"
OUTPUT          = "estimate.json"

# ── Scale ─────────────────────────────────────────────────────────────────────
# Drawing: 1:50 on A1 paper.  PyMuPDF page units = PDF points (1 pt = 1/72 in).
MM_PER_PT       = 25.4 / 72       # PDF point → mm on paper
DRAWING_SCALE   = 50              # 1:50
REAL_MM_PER_PT  = MM_PER_PT * DRAWING_SCALE   # ≈ 17.64 mm real per PDF point

# ── Geometry tolerances ────────────────────────────────────────────────────────
MIN_SEG_PT      = 15    # discard noise segments shorter than this (pts)
COLLINEAR_TOL   = 2     # pts — snap x/y to integer grid for grouping
PROXIMITY_PT    = 60    # how far an annotation can sit from a line it labels
ENDPOINT_TOL    = 8     # pts — two endpoints count as coincident within this

# ── Assumed building geometry ─────────────────────────────────────────────────
FLOOR_TO_CEILING_MM = 2700   # standard Swedish office/school floor height

# ── Cost table: (system_code, width_mm, fire_rating) → SEK/m installed ───────
# Indicative rates, Swedish market 2024 (supply + installation, excl. VAT).
# Adjust to match your actual price list.
COST_TABLE = {
    ("KS",     400, None):      620,
    ("KS",     400, "EI 30-C"): 980,
    ("KR",     400, None):      510,
    ("FBK",    100, None):      380,
    ("Tomrör", None, None):      75,
}
COST_FALLBACK_PER_M = 500   # SEK/m when no exact match


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 helpers — route length from drawing paths
# ══════════════════════════════════════════════════════════════════════════════

def _seg_endpoints(page):
    """
    Return all straight line segments from the PDF drawing layer as
    list of (p1, p2) fitz.Point pairs, filtering obvious noise.
    Normalises direction so p1 <= p2 (by x then y) for deduplication.
    """
    # Bounding y of the legend / notes area — skip lines below this
    legend_y = 2050

    raw = set()
    for path in page.get_drawings():
        if path.get("color") is None:
            continue                        # unfilled, unstroked — skip
        # Discard paths entirely within the legend area
        r = path.get("rect")
        if r and r.y0 >= legend_y:
            continue
        items = path.get("items", [])
        cur = None                          # reset per path — must not leak across paths
        for item in items:
            k = item[0]
            if k == "m":
                cur = item[1]               # item[1] is always a Point for 'm'
            elif k == "l":
                # PyMuPDF 'l' item: ('l', p_start, p_end)
                a, b = item[1], item[2]
                cur = b
                L = math.hypot(b.x - a.x, b.y - a.y)
                if L < MIN_SEG_PT:
                    continue
                # Canonicalise direction so (a,b) and (b,a) hash the same
                if (a.x, a.y) > (b.x, b.y):
                    a, b = b, a
                raw.add((round(a.x, 1), round(a.y, 1),
                         round(b.x, 1), round(b.y, 1)))
            elif k == "qu":
                # Quad — treat its bounding rect diagonal as a segment if large enough
                q = item[1]
                # q.ul / q.lr are opposite corners
                try:
                    a, b = q.ul, q.lr
                    cur = b
                    L = math.hypot(b.x - a.x, b.y - a.y)
                    if L >= MIN_SEG_PT:
                        if (a.x, a.y) > (b.x, b.y):
                            a, b = b, a
                        raw.add((round(a.x, 1), round(a.y, 1),
                                 round(b.x, 1), round(b.y, 1)))
                except Exception:
                    cur = None
            elif k == "c":
                # Cubic bezier — skip geometry, just advance current point
                cur = item[3] if len(item) > 3 else None
    return [(fitz.Point(ax, ay), fitz.Point(bx, by))
            for ax, ay, bx, by in raw]


def _build_run_groups(segments):
    """
    Group collinear, axis-aligned segments by their fixed coordinate.

    Returns two dicts:
      verticals   : { round(x) → list of (y_min, y_max) intervals }
      horizontals : { round(y) → list of (x_min, x_max) intervals }
    """
    verticals   = defaultdict(list)   # x-key → [(y0,y1), ...]
    horizontals = defaultdict(list)   # y-key → [(x0,x1), ...]

    for a, b in segments:
        dx = abs(b.x - a.x)
        dy = abs(b.y - a.y)
        if dy > dx and dx <= COLLINEAR_TOL:          # vertical
            x_key = round((a.x + b.x) / 2)
            y0, y1 = min(a.y, b.y), max(a.y, b.y)
            verticals[x_key].append((y0, y1))
        elif dx > dy and dy <= COLLINEAR_TOL:         # horizontal
            y_key = round((a.y + b.y) / 2)
            x0, x1 = min(a.x, b.x), max(a.x, b.x)
            horizontals[y_key].append((x0, x1))

    return verticals, horizontals


def _merge_intervals(intervals):
    """Merge overlapping/adjacent intervals; return list of (lo, hi) + total span."""
    if not intervals:
        return [], 0.0
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append([lo, hi])
    total = sum(hi - lo for lo, hi in merged)
    return merged, total


def _merged_span(ivs):
    _, span = _merge_intervals(ivs)
    return span


def _run_direction(bbox):
    """
    Infer the cable run direction from the annotation bbox aspect ratio.

    In a plan drawing:
    - Label text rotated 90° (tall/narrow bbox) → run extends along the
      PDF y-axis → look for VERTICAL groups.
    - Label text horizontal (wide/flat bbox) → run extends along the
      PDF x-axis → look for HORIZONTAL groups.

    Returns "v", "h", or "both".
    """
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if h > 0 and w > 0:
        ratio = h / w
        if ratio >= 1.3:
            return "v"
        if ratio <= 0.77:        # 1/1.3
            return "h"
    return "both"


def _nearby_group_keys(bbox, verticals, horizontals):
    """
    Return the set of group keys (direction, axis_key) for the cable run
    most likely labelled by this annotation bbox.

    Strategy: among all axis-aligned runs within PROXIMITY_PT of the
    annotation centre that overlap the annotation's perpendicular position,
    keep the one with the LONGEST merged span in each direction.

    Only search the axis that matches the annotation's run direction
    (inferred from bbox aspect ratio) to avoid picking up perpendicular
    structural lines as false matches.
    """
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    direction = _run_direction(bbox)
    keys = set()

    if direction in ("v", "both"):
        # Best vertical run (fixed x, extends in y) near annotation centre
        best_v, best_v_span = None, 0.0
        for x_key, ivs in verticals.items():
            if abs(x_key - cx) > PROXIMITY_PT:
                continue
            if not any(lo - PROXIMITY_PT <= cy <= hi + PROXIMITY_PT for lo, hi in ivs):
                continue
            span = _merged_span(ivs)
            if span > best_v_span:
                best_v, best_v_span = x_key, span
        if best_v is not None:
            keys.add(("v", best_v))

    if direction in ("h", "both"):
        # Best horizontal run (fixed y, extends in x) near annotation centre
        best_h, best_h_span = None, 0.0
        for y_key, ivs in horizontals.items():
            if abs(y_key - cy) > PROXIMITY_PT:
                continue
            if not any(lo - PROXIMITY_PT <= cx <= hi + PROXIMITY_PT for lo, hi in ivs):
                continue
            span = _merged_span(ivs)
            if span > best_h_span:
                best_h, best_h_span = y_key, span
        if best_h is not None:
            keys.add(("h", best_h))

    return keys


def _total_length_for_keys(group_keys, verticals, horizontals):
    """Sum merged interval lengths for a set of unique group keys."""
    total = 0.0
    for direction, key in group_keys:
        ivs = verticals[key] if direction == "v" else horizontals[key]
        _, span = _merge_intervals(ivs)
        total += span
    return total


def _annotation_span_pts(bboxes):
    """
    Fallback: estimate run extent from the min/max position of annotation bboxes.
    Returns the dominant-axis span (the larger of x-span and y-span).
    """
    if not bboxes:
        return 0.0
    xs = [b[0] for b in bboxes] + [b[2] for b in bboxes]
    ys = [b[1] for b in bboxes] + [b[3] for b in bboxes]
    return max(max(xs) - min(xs), max(ys) - min(ys))


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Vertical routing lengths
# ══════════════════════════════════════════════════════════════════════════════

def _vertical_length_mm(item):
    """
    Estimate vertical routing length for one summary item (horizontal tray).

    For each drawn instance that has both ok and uk heights, the vertical
    extent is (ok - uk) mm.  When only uk is present for a ceiling-mounted
    tray (uk ≈ FLOOR_TO_CEILING_MM), the drop to floor is uk itself.
    """
    ok = item.get("ok_ofg_mm")
    uk = item.get("uk_ofg_mm")
    count = item.get("count", 1)
    if not isinstance(count, int):
        count = 0

    if ok is not None and uk is not None:
        drop_per_instance = abs(ok - uk)
    elif uk is not None:
        # ceiling mount: distance from uk down to floor
        drop_per_instance = uk
    elif ok is not None:
        drop_per_instance = FLOOR_TO_CEILING_MM - ok
    else:
        drop_per_instance = 0

    return drop_per_instance * count   # mm


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Cost mapping
# ══════════════════════════════════════════════════════════════════════════════

def _unit_cost(system, width_mm, fire_rating):
    """Return SEK/m for the given specification."""
    key = (system, width_mm if isinstance(width_mm, int) else None, fire_rating)
    return COST_TABLE.get(key, COST_FALLBACK_PER_M)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Confidence scoring
# ══════════════════════════════════════════════════════════════════════════════

def _confidence(item, length_source):
    """
    Score 0.0–1.0 for how reliable the cost estimate is.

    Factors
    -------
    annotation_completeness : all fields present?
    length_source           : "paths" > "annotation_span" > "heuristic" > "legend"
    fire_rating_basis       : explicit annotation vs proximity inference
    count_basis             : multiple confirmed instances
    """
    score = 0.0
    notes = []

    # ── Field completeness ────────────────────────────────────────────────
    has_size   = isinstance(item.get("width_mm"), int)
    has_height = item.get("ok_ofg_mm") or item.get("uk_ofg_mm")
    has_count  = isinstance(item.get("count"), int)

    if has_size:
        score += 0.20
    else:
        notes.append("width unknown (per drawing)")

    if has_height:
        score += 0.15
    else:
        notes.append("mounting height unspecified")

    if has_count and item["count"] > 1:
        score += 0.10
    elif has_count:
        score += 0.05
    else:
        notes.append("count not determined (legend entry)")

    # ── Length source ─────────────────────────────────────────────────────
    if length_source == "paths":
        score += 0.35
    elif length_source == "annotation_span":
        score += 0.20
        notes.append("length lower-bound from annotation positions")
    elif length_source == "heuristic":
        score += 0.10
        notes.append("length estimated from height data only")
    else:  # legend / unknown
        score += 0.0
        notes.append("no length data (legend entry)")

    # ── Fire rating ───────────────────────────────────────────────────────
    if item.get("fire_rating"):
        score += 0.15   # fire-rated items are well-specified
        if length_source not in ("paths", "annotation_span"):
            notes.append("fire rating associated by x-band proximity")
    else:
        score += 0.05

    # ── Diameter known (Tomrör) ───────────────────────────────────────────
    if item.get("diameter_mm"):
        score += 0.05

    return round(min(score, 1.0), 2), notes


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def estimate(pdf_path=PDF_PATH, components_data=None):
    """
    Run all four estimation stages.

    Parameters
    ----------
    pdf_path        : path to the PDF (needed for vector-path tracing)
    components_data : dict returned by extract(); if None, read COMPONENTS_JSON
    """
    if components_data is None:
        components_data = json.loads(Path(COMPONENTS_JSON).read_text(encoding="utf-8"))

    data       = components_data
    summary    = data["summary"]
    components = data["components"]

    doc  = fitz.open(pdf_path)
    page = doc[0]

    segments               = _seg_endpoints(page)
    verticals, horizontals = _build_run_groups(segments)
    doc.close()

    # ── Pre-index drawn instances by (type, size, ok, uk, fire_rating) ───
    # so we can look up annotation bboxes for each summary variant
    def _sig(c):
        return (
            c["type"],
            str(c.get("size") or ""),
            str(c.get("ok_height") or ""),
            str(c.get("uk_height") or ""),
            str(c.get("fire_rating") or ""),
        )

    def _summary_sig(s):
        w = s.get("width_mm")
        return (
            s["system"],
            str(w) if isinstance(w, int) else "",
            str(s.get("ok_ofg_mm") or ""),
            str(s.get("uk_ofg_mm") or ""),
            str(s.get("fire_rating") or ""),
        )

    instances_by_sig = defaultdict(list)
    for c in components:
        if c.get("bbox") is not None and c.get("occurrences") == 1:
            instances_by_sig[_sig(c)].append(c["bbox"])

    items = []

    for s in summary:
        is_legend = not isinstance(s.get("count"), int)

        # ── Stage 1: horizontal run length ───────────────────────────────
        if is_legend:
            horiz_length_mm = 0.0
            length_source   = "legend"
        else:
            sig       = _summary_sig(s)
            bboxes    = instances_by_sig.get(sig, [])

            # Collect all unique run groups near ANY of this variant's annotations,
            # then sum each group's merged length exactly once.
            all_keys = set()
            for bb in bboxes:
                all_keys |= _nearby_group_keys(bb, verticals, horizontals)

            path_total_pts = _total_length_for_keys(all_keys, verticals, horizontals)

            if path_total_pts > 0:
                horiz_length_mm = path_total_pts * REAL_MM_PER_PT
                length_source   = "paths"
            else:
                span_pts        = _annotation_span_pts(bboxes)
                horiz_length_mm = span_pts * REAL_MM_PER_PT
                length_source   = "annotation_span" if span_pts > 0 else "heuristic"

        # ── Stage 2: vertical routing length ─────────────────────────────
        vert_length_mm = _vertical_length_mm(s)

        total_length_mm = horiz_length_mm + vert_length_mm
        total_length_m  = total_length_mm / 1000.0

        # ── Stage 3: cost ─────────────────────────────────────────────────
        w  = s.get("width_mm")
        fr = s.get("fire_rating")
        unit_cost_sek = _unit_cost(
            s["system"],
            w if isinstance(w, int) else None,
            fr,
        )
        total_cost_sek = round(unit_cost_sek * total_length_m)

        # ── Stage 4: confidence ───────────────────────────────────────────
        confidence, conf_notes = _confidence(s, length_source)

        items.append({
            **s,
            # Stage 1
            "horizontal_length_mm": round(horiz_length_mm),
            "length_source":         length_source,
            # Stage 2
            "vertical_length_mm":   round(vert_length_mm),
            # Totals
            "total_length_mm":      round(total_length_mm),
            "total_length_m":       round(total_length_m, 2),
            # Stage 3
            "unit_cost_sek_per_m":  unit_cost_sek,
            "total_cost_sek":       total_cost_sek,
            # Stage 4
            "confidence":           confidence,
            "confidence_notes":     conf_notes,
        })

    # ── Roll-up totals ────────────────────────────────────────────────────
    countable = [i for i in items if isinstance(i.get("count"), int)]

    by_system = defaultdict(lambda: {"total_length_m": 0.0, "total_cost_sek": 0})
    for i in countable:
        g = by_system[i["system"]]
        g["total_length_m"]  += i["total_length_m"]
        g["total_cost_sek"]  += i["total_cost_sek"]

    totals = {
        "total_cost_sek":    sum(i["total_cost_sek"]  for i in countable),
        "total_length_m":    round(sum(i["total_length_m"] for i in countable), 2),
        "scale_mm_per_pt":   round(REAL_MM_PER_PT, 4),
        "by_system": {
            k: {"total_length_m": round(v["total_length_m"], 2),
                "total_cost_sek": v["total_cost_sek"]}
            for k, v in by_system.items()
        },
    }

    result = {"items": items, "totals": totals}
    Path(OUTPUT).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    result = estimate(PDF_PATH)

    print(f"\nScale: {result['totals']['scale_mm_per_pt']} mm/pt\n")
    print(f"{'System':<8} {'Orient':<12} {'W mm':<10} {'UK/ÖK':<18} {'FR':<10} "
          f"{'N':<4} {'H-len m':<9} {'V-len m':<9} {'Tot m':<8} "
          f"{'SEK/m':<7} {'SEK':<8} {'Conf':<6} Source")
    print("─" * 130)

    for i in result["items"]:
        fr  = i.get("fire_rating") or "—"
        w   = i.get("width_mm")
        wstr = str(w) if isinstance(w, int) else "per_drw"
        ok  = i.get("ok_ofg_mm") or "—"
        uk  = i.get("uk_ofg_mm") or "—"
        heights = f"ÖK{ok}/UK{uk}"
        cnt = i.get("count")
        print(f"{i['system']:<8} {i['orientation']:<12} {wstr:<10} {heights:<18} {fr:<10} "
              f"{str(cnt):<4} {i['horizontal_length_mm']/1000:>7.1f}m "
              f"{i['vertical_length_mm']/1000:>7.1f}m "
              f"{i['total_length_m']:>6.1f}m "
              f"{i['unit_cost_sek_per_m']:>6}  "
              f"{i['total_cost_sek']:>7}  "
              f"{i['confidence']:.2f}  "
              f"{i['length_source']}")
        for note in i.get("confidence_notes", []):
            print(f"         ↳ {note}")

    print("─" * 130)
    t = result["totals"]
    print(f"{'TOTAL':<8} {'':12} {'':10} {'':18} {'':10} {'':4} "
          f"{'':9} {'':9} {t['total_length_m']:>6.1f}m "
          f"{'':6}  {t['total_cost_sek']:>7}")
    print(f"\nBreakdown by system:")
    for sys, v in t["by_system"].items():
        print(f"  {sys}: {v['total_length_m']} m  →  {v['total_cost_sek']} SEK")
    print(f"\nEstimate saved → {OUTPUT}")
