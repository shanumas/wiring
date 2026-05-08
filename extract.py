"""
Electrical component extractor for wiring/canalisation drawings.
Extracts KS / KR / FBK annotation instances with bboxes, plus Tomrör and
vertical-system entries from the legend.  Fire rating (EI 30-C) is
associated with components that share an x-band with an EI block.
"""
import fitz
import json
import re
from pathlib import Path
from collections import defaultdict

PDF_PATH = "pdf/2.pdf"
OUTPUT   = "components.json"

# ── Component metadata ────────────────────────────────────────────────────────
COMP_META = {
    "KS":     {"name": "Kabelstege",         "en": "Cable Ladder",        "color": "#3b82f6"},
    "KR":     {"name": "Kabelränna",          "en": "Cable Tray",          "color": "#10b981"},
    "FBK":    {"name": "Fönsterbänkskanal",   "en": "Window Sill Channel", "color": "#f59e0b"},
    "Tomrör": {"name": "Tomrör",              "en": "Conduit",             "color": "#8b5cf6"},
}

# ── Regex patterns ────────────────────────────────────────────────────────────
COMP_CODE    = re.compile(r'\b(KS|KR|FBK)\b')
TOMROR_PAT   = re.compile(r'\bTOMRÖR\b', re.IGNORECASE)
VERTIKAL_PAT = re.compile(r'\bVERTIKAL\b', re.IGNORECASE)
# Heights: "ÖK= 2700 ÖFG", "UK= 2700 ÖFG", "ÖK= 900MM" (note uppercase MM variant)
OK_PAT       = re.compile(r'ÖK=\s*(\d+)',  re.IGNORECASE)
UK_PAT       = re.compile(r'UK=\s*(\d+)',  re.IGNORECASE)
# Width: "<CODE> <number>" e.g. "KS 400", "FBK 100"
SIZE_PAT     = re.compile(r'\b(?:KS|KR|FBK)\s+(\d+)', re.IGNORECASE)
DIAMETER_PAT = re.compile(r'Ø(\d+)')
EI_PAT       = re.compile(r'EI\s*30-C',    re.IGNORECASE)

# y-coordinate above which all blocks are in the legend / notes area
# (observed boundary: drawn components end around y=1931, legend starts ~y=2072)
LEGEND_Y = 2050


# ── Helpers ───────────────────────────────────────────────────────────────────

def tight_bbox(block):
    """Tighten block bbox to start from the first component-code span."""
    spans = [s for line in block.get("lines", [])
               for s in line.get("spans", [])]
    start = next((i for i, s in enumerate(spans)
                  if COMP_CODE.search(s["text"].strip())), None)
    if start is None:
        return block["bbox"]
    pre = [s["text"].strip() for s in spans[:start] if s["text"].strip()]
    if not pre:
        return block["bbox"]
    tail = spans[start:]
    if not tail:
        return block["bbox"]
    return (
        min(s["bbox"][0] for s in tail), min(s["bbox"][1] for s in tail),
        max(s["bbox"][2] for s in tail), max(s["bbox"][3] for s in tail),
    )


def x_bands_overlap(bbox_a, bbox_b):
    """True if two bboxes share any x range."""
    return bbox_a[0] <= bbox_b[2] and bbox_b[0] <= bbox_a[2]


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_standard(text):
    """Parse a KS/KR/FBK annotation block. Returns dict or None."""
    m = COMP_CODE.search(text)
    if not m:
        return None
    code = m.group(1)
    size_m = SIZE_PAT.search(text)
    ok_m   = OK_PAT.search(text)
    uk_m   = UK_PAT.search(text)
    return {
        "type":        code,
        "name":        COMP_META[code]["name"],
        "en_name":     COMP_META[code]["en"],
        "color":       COMP_META[code]["color"],
        "size":        size_m.group(1) if size_m else None,
        "ok_height":   ok_m.group(1)   if ok_m   else None,
        "uk_height":   uk_m.group(1)   if uk_m   else None,
        "is_vertical": False,
        "fire_rating": None,       # filled in later by proximity pass
    }


def parse_tomror_note(text):
    """Extract Tomrör diameter from a text block containing 'TOMRÖR'."""
    d = DIAMETER_PAT.search(text)
    return d.group(1) if d else None


def parse_vertical_legend(text):
    """
    Parse a vertical-system legend entry.
    Returns list of dicts (one entry per system mentioned in the block).
    """
    entries = []
    utext = text.upper()

    systems = []
    if "FÖNSTERBÄNKSKANAL" in utext:
        systems.append("FBK")
    if "KABELRÄNNA" in utext:
        systems.append("KR")
    if "KABELSTEGE" in utext:
        systems.append("KS")

    ok_m = OK_PAT.search(text)
    uk_m = UK_PAT.search(text)

    for code in systems:
        entries.append({
            "type":        code,
            "name":        "Vertikal " + COMP_META[code]["name"],
            "en_name":     "Vertical " + COMP_META[code]["en"],
            "color":       COMP_META[code]["color"],
            "size":        "per_drawing",   # "bredd & höjd enligt ritning"
            "ok_height":   ok_m.group(1) if ok_m else None,
            "uk_height":   uk_m.group(1) if uk_m else None,
            "is_vertical": True,
            "fire_rating": None,
        })
    return entries


# ── Main extractor ────────────────────────────────────────────────────────────

def extract(pdf_path=PDF_PATH):
    doc  = fitz.open(pdf_path)
    page = doc[0]

    # First pass: collect EI 30-C block bboxes
    ei_bboxes = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        spans_text = [s["text"].strip()
                      for line in b.get("lines", [])
                      for s in line.get("spans", [])
                      if s["text"].strip()]
        full = " ".join(spans_text)
        if EI_PAT.search(full):
            ei_bboxes.append(b["bbox"])

    # Track known Tomrör diameter from notes
    tomror_diameter = None

    components   = []
    seen_bboxes  = set()          # dedup drawn instances
    legend_types = set()          # dedup legend entries

    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        spans_text = [s["text"].strip()
                      for line in b.get("lines", [])
                      for s in line.get("spans", [])
                      if s["text"].strip()]
        full = " ".join(spans_text)
        if not full:
            continue

        bbox  = b["bbox"]
        in_legend = bbox[1] >= LEGEND_Y

        # ── Tomrör (both legend header and installation note) ──────────────
        if TOMROR_PAT.search(full):
            d = parse_tomror_note(full)
            if d:
                tomror_diameter = d
            continue   # not a positioned component to overlay

        # ── Vertical system legend entries ─────────────────────────────────
        if in_legend and VERTIKAL_PAT.search(full):
            for entry in parse_vertical_legend(full):
                key = (entry["type"], entry["is_vertical"])
                if key not in legend_types:
                    legend_types.add(key)
                    entry["id"]          = f"LEG_{entry['type']}_V_{len(components)}"
                    entry["label"]       = full
                    entry["bbox"]        = None   # legend, no drawing position
                    entry["occurrences"] = "legend"
                    components.append(entry)
            continue

        # ── Skip remaining legend / note blocks ───────────────────────────
        if in_legend:
            continue

        # ── Standard drawn KS / KR / FBK instances ────────────────────────
        if not COMP_CODE.search(full):
            continue

        parsed = parse_standard(full)
        if not parsed:
            continue

        tight  = tight_bbox(b)
        dedup  = (parsed["type"], parsed["size"],
                  tuple(round(x) for x in tight))
        if dedup in seen_bboxes:
            continue
        seen_bboxes.add(dedup)

        # Associate fire rating: any EI block whose x-band overlaps this one
        for ei in ei_bboxes:
            if x_bands_overlap(tight, ei):
                parsed["fire_rating"] = "EI 30-C"
                break

        parsed["id"]          = f"{parsed['type']}_{len(components)}"
        parsed["label"]       = full
        parsed["bbox"]        = [round(x, 2) for x in tight]
        parsed["occurrences"] = 1    # individual instance
        components.append(parsed)

    # ── Tomrör entry ──────────────────────────────────────────────────────
    if tomror_diameter is not None:
        components.append({
            "id":          "Tomrör_legend",
            "type":        "Tomrör",
            "name":        "Tomrör",
            "en_name":     "Conduit",
            "color":       "#8b5cf6",
            "size":        None,
            "diameter_mm": int(tomror_diameter),
            "ok_height":   None,
            "uk_height":   None,
            "is_vertical": False,
            "fire_rating": None,
            "label":       "Tomrör Ø16",
            "bbox":        None,
            "occurrences": "present",
        })

    # ── Aggregated summary (for cost estimation) ──────────────────────────
    sig_counts = defaultdict(int)
    sig_proto  = {}
    for c in components:
        if c["occurrences"] in (1,):
            sig = (c["type"], c.get("size"), c.get("ok_height"),
                   c.get("uk_height"), c.get("is_vertical"), c.get("fire_rating"))
            sig_counts[sig] += 1
            if sig not in sig_proto:
                sig_proto[sig] = c

    summary = []
    for sig, count in sig_counts.items():
        p      = sig_proto[sig]
        is_vert = p.get("is_vertical", False)
        raw_size = p.get("size")
        summary.append({
            "system":       p["type"],
            "name":         p["name"],
            "orientation":  "vertical" if is_vert else "horizontal",
            "width_mm":     int(raw_size) if raw_size else "per_drawing",
            "ok_ofg_mm":    int(p["ok_height"]) if p.get("ok_height") else None,
            "uk_ofg_mm":    int(p["uk_height"]) if p.get("uk_height") else None,
            "fire_rating":  p.get("fire_rating"),
            "count":        count,
        })

    # Sort: by system code, then fire-rating (rated first), then height signature
    summary.sort(key=lambda s: (
        s["system"],
        0 if s["fire_rating"] else 1,
        s.get("ok_ofg_mm") or 0,
        s.get("uk_ofg_mm") or 0,
    ))

    # Add legend-only types (vertical systems, Tomrör) to summary
    for c in components:
        if c.get("occurrences") in ("legend", "present"):
            raw_size = c.get("size")
            is_vert  = c.get("is_vertical", False)
            ok_val   = int(c["ok_height"]) if c.get("ok_height") else None
            entry = {
                "system":       c["type"],
                "name":         c["name"],
                "orientation":  "vertical" if is_vert else "horizontal",
                "width_mm":     int(raw_size) if raw_size and raw_size != "per_drawing"
                                else ("per_drawing" if is_vert else None),
                "ok_ofg_mm":    ok_val,
                "uk_ofg_mm":    int(c["uk_height"]) if c.get("uk_height") else None,
                "fire_rating":  c.get("fire_rating"),
                "count":        c["occurrences"],   # "legend" or "present"
            }
            # Alias: vertical reference height has a dedicated key for clarity
            if is_vert and ok_val is not None:
                entry["vertical_reference_height_mm"] = ok_val
            if c.get("diameter_mm"):
                entry["diameter_mm"] = c["diameter_mm"]
            summary.append(entry)

    # ── Drawing type detection ────────────────────────────────────────────
    # Scan all text to infer the discipline so the UI can show a helpful
    # message when no canalisation components are found.
    all_text = " ".join(
        " ".join(s["text"] for line in b.get("lines", []) for s in line.get("spans", []))
        for b in page.get_text("dict")["blocks"] if b.get("type") == 0
    ).upper()

    if components:
        drawing_type = "kanalisation"
    elif any(kw in all_text for kw in ["DALI", "STRÖMSTÄLL", "DIMMER", "SENSOR", "ARMAT", "BELYSN"]):
        drawing_type = "belysning"
    elif any(kw in all_text for kw in ["UTTAG", "ELCENTRAL", "SÄKERHETSBRYT", "MOTOR"]):
        drawing_type = "el-kraft"
    elif any(kw in all_text for kw in ["VVS", "RÖRLED", "VENTIL", "SPRINKLER"]):
        drawing_type = "vvs"
    else:
        drawing_type = "unknown"

    w = page.rect.width
    h = page.rect.height
    doc.close()
    return {
        "page_width":   w,
        "page_height":  h,
        "drawing_type": drawing_type,
        "components":   components,
        "summary":      summary,
    }


if __name__ == "__main__":
    data = extract()
    Path(OUTPUT).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    print(f"\nExtracted {len(data['components'])} component entries → {OUTPUT}")
    print("\n── Drawn instances ──────────────────────────────────────────────")
    for c in data["components"]:
        if c.get("occurrences") == 1:
            fr = f"  [{c['fire_rating']}]" if c.get("fire_rating") else ""
            v  = " VERTICAL" if c.get("is_vertical") else ""
            print(f"  [{c['type']}{v}] {c['name']} {c.get('size') or '?'}mm  "
                  f"ÖK={c.get('ok_height')} UK={c.get('uk_height')}{fr}  "
                  f"bbox={[round(x) for x in c['bbox']]}")

    print("\n── Legend / notes entries ───────────────────────────────────────")
    for c in data["components"]:
        if c.get("occurrences") != 1:
            extra = f"  Ø{c['diameter_mm']}mm" if c.get("diameter_mm") else ""
            ok = f"  ÖK={c.get('ok_height')}" if c.get("ok_height") else ""
            print(f"  [{c['type']}] {c['name']}{extra}{ok}  ({c['occurrences']})")

    print("\n── Aggregated summary ───────────────────────────────────────────")
    for s in data["summary"]:
        fr  = f"  [{s['fire_rating']}]" if s.get("fire_rating") else ""
        v   = " VERTICAL" if s.get("orientation") == "vertical" else ""
        vrh = f"  ref={s['vertical_reference_height_mm']}mm" if s.get("vertical_reference_height_mm") else ""
        w = s.get('width_mm')
        wstr = f"{w}mm" if isinstance(w, int) else (w or '—')
        dstr = f" Ø{s['diameter_mm']}mm" if s.get("diameter_mm") else ""
        print(f"  {s['system']}{v} {wstr}{dstr}  "
              f"ÖK={s.get('ok_ofg_mm')} UK={s.get('uk_ofg_mm')}{vrh}{fr}  "
              f"× {s['count']}")
