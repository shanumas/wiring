"""
Vector-based cable tray extractor.

Instead of reading text annotations, this module reads the actual drawing
geometry (line segments) from the PDF, identifies cable tray runs as pairs
of parallel lines, measures their lengths using the drawing scale, and
associates them with component types via text label proximity.

Returns the same top-level schema as extract.py so the frontend and
estimate.py can consume it unchanged.
"""

import fitz
import math
import re
from collections import defaultdict
from pathlib import Path

# ── Scale extraction ──────────────────────────────────────────────────────────

_SCALE_PAT = re.compile(r'SKALA\s+1\s*[:\s]\s*(\d+)', re.IGNORECASE)
_SCALE_ALT  = re.compile(r'\b1\s*:\s*(\d+)\b')

def _extract_scale(page):
    """Return the drawing scale denominator (e.g. 50 for 1:50)."""
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        spans = [s["text"] for ln in b.get("lines", [])
                 for s in ln.get("spans", []) if s["text"].strip()]
        full = " ".join(spans)
        m = _SCALE_PAT.search(full) or _SCALE_ALT.search(full)
        if m:
            val = int(m.group(1))
            if 10 <= val <= 1000:          # sanity check
                return val
    return 100                             # fallback


# ── Segment extraction ────────────────────────────────────────────────────────

def _get_segments(page, min_len=25, angle_tol=0.08):
    """
    Return all near-horizontal and near-vertical line segments longer than
    min_len, represented in the page's unrotated coordinate space.
    """
    segs = []
    for d in page.get_drawings():
        col = d.get("color")
        w   = d.get("width") or 0
        for item in d.get("items", []):
            if item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length = math.hypot(dx, dy)
            if length < min_len:
                continue
            # classify direction — skip diagonals
            if abs(dy) <= abs(dx) * angle_tol:
                direction = "H"
            elif abs(dx) <= abs(dy) * angle_tol:
                direction = "V"
            else:
                continue
            segs.append({
                "dir":  direction,
                "len":  length,
                "x0":   min(p1.x, p2.x),
                "y0":   min(p1.y, p2.y),
                "x1":   max(p1.x, p2.x),
                "y1":   max(p1.y, p2.y),
                "col":  col,
                "w":    w,
            })
    return segs


# ── Run merging ───────────────────────────────────────────────────────────────

def _cluster_value(values, tol):
    """Cluster a sorted list of floats; return list of (centroid, [values])."""
    if not values:
        return []
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [(sum(c) / len(c), c) for c, in [(c,) for c in clusters]]


def _merge_spans(spans, gap=60):
    """Merge overlapping/nearby 1-D spans [(start, end), ...]."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]


def _build_runs(segs, perp_tol=4, merge_gap=60):
    """
    Group collinear/parallel segments into runs.

    Returns list of dicts:
      dir, perp (position on perpendicular axis), start, end, length, n_segs
    """
    runs = []
    for direction in ("H", "V"):
        d_segs = [s for s in segs if s["dir"] == direction]

        if direction == "H":
            perp_key  = lambda s: (s["y0"] + s["y1"]) / 2
            span_key  = lambda s: (s["x0"], s["x1"])
        else:
            perp_key  = lambda s: (s["x0"] + s["x1"]) / 2
            span_key  = lambda s: (s["y0"], s["y1"])

        # Sort by perpendicular position
        d_segs.sort(key=perp_key)

        # Cluster by perpendicular position
        perp_clusters = defaultdict(list)
        bucket = None
        for s in d_segs:
            p = perp_key(s)
            if bucket is None or abs(p - bucket) > perp_tol:
                bucket = p
            perp_clusters[bucket].append(s)

        for perp, group in perp_clusters.items():
            spans = [span_key(s) for s in group]
            for start, end in _merge_spans(spans, gap=merge_gap):
                runs.append({
                    "dir":    direction,
                    "perp":   perp,
                    "start":  start,
                    "end":    end,
                    "length": end - start,
                    "n_segs": len(group),
                })

    return runs


# ── Cable tray identification ─────────────────────────────────────────────────

def _pair_runs(runs, min_sep=1, max_sep=40, min_overlap=40):
    """
    Find pairs of parallel runs close together — the two walls of a cable tray.
    Returns list of tray_candidate dicts.
    """
    by_dir = defaultdict(list)
    for r in runs:
        by_dir[r["dir"]].append(r)

    candidates = []
    for direction, druns in by_dir.items():
        druns_sorted = sorted(druns, key=lambda r: r["perp"])
        used = set()
        for i, r1 in enumerate(druns_sorted):
            if i in used:
                continue
            best = None
            for j, r2 in enumerate(druns_sorted[i + 1:], i + 1):
                sep = r2["perp"] - r1["perp"]
                if sep > max_sep:
                    break
                if sep < min_sep:
                    continue
                overlap = (min(r1["end"], r2["end"])
                           - max(r1["start"], r2["start"]))
                if overlap < min_overlap:
                    continue
                if best is None or overlap > best["overlap"]:
                    best = {"j": j, "r2": r2, "sep": sep, "overlap": overlap}
            if best:
                r2 = best["r2"]
                perp_c = (r1["perp"] + r2["perp"]) / 2
                start  = max(r1["start"], r2["start"])
                end    = min(r1["end"],   r2["end"])
                # bbox in unrotated page space
                if direction == "H":
                    bbox = (start, r1["perp"], end, r2["perp"])
                else:
                    bbox = (r1["perp"], start, r2["perp"], end)
                candidates.append({
                    "dir":         direction,
                    "perp_center": perp_c,
                    "sep_units":   best["sep"],
                    "start":       start,
                    "end":         end,
                    "length":      best["overlap"],
                    "raw_bbox":    bbox,
                })
                used.add(i)
                used.add(best["j"])

    return candidates


# ── Text label association ────────────────────────────────────────────────────

_COMP_CODE   = re.compile(r'\b(KS|KR|FBK)\b')
_SIZE_PAT    = re.compile(r'\b(?:KS|KR|FBK)\s+(\d+)', re.IGNORECASE)
_OK_PAT      = re.compile(r'ÖK=\s*(\d+)', re.IGNORECASE)
_UK_PAT      = re.compile(r'UK=\s*(\d+)', re.IGNORECASE)
_LEGEND_Y    = 2050   # same boundary as extract.py
_EI_PAT      = re.compile(r'EI\s*30-C', re.IGNORECASE)

COMP_META = {
    "KS":  {"name": "Kabelstege",       "en": "Cable Ladder",        "color": "#1e4d8c"},
    "KR":  {"name": "Kabelränna",        "en": "Cable Tray",          "color": "#1a6b45"},
    "FBK": {"name": "Fönsterbänkskanal","en": "Window Sill Channel",  "color": "#8a6200"},
}


def _collect_labels(page):
    """
    Return list of label dicts from drawing body text that contain KS/KR/FBK.
    Bboxes are in unrotated page space.
    """
    labels = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        bbox = b["bbox"]
        if bbox[1] >= _LEGEND_Y:
            continue
        spans = [s["text"].strip()
                 for ln in b.get("lines", [])
                 for s in ln.get("spans", []) if s["text"].strip()]
        full = " ".join(spans)
        m = _COMP_CODE.search(full)
        if not m:
            continue
        code     = m.group(1)
        size_m   = _SIZE_PAT.search(full)
        ok_m     = _OK_PAT.search(full)
        uk_m     = _UK_PAT.search(full)
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        labels.append({
            "type":      code,
            "size":      size_m.group(1) if size_m else None,
            "ok_height": ok_m.group(1)   if ok_m   else None,
            "uk_height": uk_m.group(1)   if uk_m   else None,
            "cx": cx, "cy": cy,
            "bbox": bbox,
        })
    return labels


def _bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _dist_point_to_bbox(cx, cy, bbox):
    """Minimum distance from point (cx,cy) to rectangle bbox."""
    bx0, by0, bx1, by1 = bbox
    dx = max(bx0 - cx, 0, cx - bx1)
    dy = max(by0 - cy, 0, cy - by1)
    return math.hypot(dx, dy)


def _assign_labels(candidates, labels, max_dist=200):
    """
    For each tray candidate, find the nearest text label and assign its type.
    Candidates with no nearby label get type=None.
    """
    for cand in candidates:
        cx = cand["perp_center"] if cand["dir"] == "V" else (cand["start"] + cand["end"]) / 2
        cy = cand["perp_center"] if cand["dir"] == "H" else (cand["start"] + cand["end"]) / 2

        best_dist  = max_dist
        best_label = None
        for lbl in labels:
            d = math.hypot(cx - lbl["cx"], cy - lbl["cy"])
            if d < best_dist:
                best_dist  = d
                best_label = lbl

        cand["label"]     = best_label
        cand["label_dist"] = best_dist

    return candidates


# ── Display-space coordinate transform (same as extract.py) ──────────────────

def _to_display_bbox(bbox, page):
    x0, y0, x1, y1 = bbox
    rot = page.rotation
    if rot == 270:
        W = page.rect.height
        return (y0, W - x1, y1, W - x0)
    elif rot == 90:
        H = page.rect.width
        return (H - y1, x0, H - y0, x1)
    elif rot == 180:
        W, H = page.rect.width, page.rect.height
        return (W - x1, H - y1, W - x0, H - y0)
    return bbox


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_vectors(pdf_path):
    """
    Vector-based extraction.  Returns the same top-level schema as extract():
      page_width, page_height, drawing_type, components, summary
    plus extra keys:
      scale            — drawing scale denominator (e.g. 50 for 1:50)
      extraction_mode  — "vector"
    """
    doc  = fitz.open(str(pdf_path))
    page = doc[0]

    scale = _extract_scale(page)
    mm_per_unit = scale          # 1 page unit = scale mm at real world

    segs       = _get_segments(page, min_len=25)
    runs       = _build_runs(segs, perp_tol=4, merge_gap=60)
    candidates = _pair_runs(runs, min_sep=1, max_sep=45, min_overlap=40)
    labels     = _collect_labels(page)
    candidates = _assign_labels(candidates, labels, max_dist=200)

    # EI 30-C fire rating blocks
    ei_bboxes = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        spans = [s["text"] for ln in b.get("lines", [])
                 for s in ln.get("spans", []) if s["text"].strip()]
        if _EI_PAT.search(" ".join(spans)):
            ei_bboxes.append(b["bbox"])

    # Build component list — one entry per tray run
    components = []
    for i, cand in enumerate(candidates):
        lbl  = cand["label"]
        if lbl is None:
            continue                   # unidentified — skip
        code  = lbl["type"]
        meta  = COMP_META.get(code, {"name": code, "en": code, "color": "#555555"})

        # Length in metres
        length_m = round(cand["length"] * mm_per_unit / 1000, 2)

        # Display bbox for overlay
        raw  = cand["raw_bbox"]
        disp = _to_display_bbox(raw, page)
        disp = [round(x, 2) for x in disp]

        # Fire rating by x-band overlap (in raw space)
        fire = None
        for ei in ei_bboxes:
            rx0, ry0, rx1, ry1 = raw
            if rx0 <= ei[2] and ei[0] <= rx1:
                fire = "EI 30-C"
                break

        components.append({
            "id":          f"VEC_{code}_{i}",
            "type":        code,
            "name":        meta["name"],
            "en_name":     meta["en"],
            "color":       meta["color"],
            "size":        lbl.get("size"),
            "ok_height":   lbl.get("ok_height"),
            "uk_height":   lbl.get("uk_height"),
            "is_vertical": False,
            "fire_rating": fire,
            "label":       f"{code} — {length_m} m",
            "bbox":        disp,
            "length_m":    length_m,
            "occurrences": 1,
        })

    # Aggregate summary — total length per variant
    from collections import defaultdict as dd
    sig_totals = dd(lambda: {"length_m": 0.0, "count": 0, "proto": None})
    for c in components:
        sig = (c["type"], c.get("size"), c.get("ok_height"),
               c.get("uk_height"), c.get("fire_rating"))
        sig_totals[sig]["length_m"] += c["length_m"]
        sig_totals[sig]["count"]    += 1
        if sig_totals[sig]["proto"] is None:
            sig_totals[sig]["proto"] = c

    summary = []
    for sig, agg in sorted(sig_totals.items(), key=lambda x: x[0][0]):
        p = agg["proto"]
        raw_size = p.get("size")
        summary.append({
            "system":      p["type"],
            "name":        p["name"],
            "orientation": "horizontal",
            "width_mm":    int(raw_size) if raw_size else None,
            "ok_ofg_mm":   int(p["ok_height"]) if p.get("ok_height") else None,
            "uk_ofg_mm":   int(p["uk_height"]) if p.get("uk_height") else None,
            "fire_rating": p.get("fire_rating"),
            "count":       agg["count"],           # number of runs
            "total_length_m": round(agg["length_m"], 2),
        })

    drawing_type = "kanalisation" if components else "unknown"

    w = page.rect.width
    h = page.rect.height
    doc.close()

    return {
        "page_width":      w,
        "page_height":     h,
        "drawing_type":    drawing_type,
        "components":      components,
        "summary":         summary,
        "scale":           scale,
        "extraction_mode": "vector",
    }


if __name__ == "__main__":
    import json
    data = extract_vectors("pdf/2.pdf")
    print(f"Scale 1:{data['scale']}")
    print(f"Runs found: {len(data['components'])}")
    print()
    for s in data["summary"]:
        print(f"  {s['system']} {s['width_mm'] or '?'}mm  "
              f"{s['count']} runs  total={s['total_length_m']} m")
    print()
    for c in data["components"]:
        print(f"  {c['type']} {c['size'] or '?'}mm  {c['length_m']} m  bbox={c['bbox']}")
