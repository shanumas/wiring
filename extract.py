"""
Electrical component extractor for wiring/canalisation drawings.
Extracts KS (cable ladder), KR (cable tray), FBK (window sill channel)
annotations with their bounding boxes from the PDF.
"""
import fitz
import json
import re
from pathlib import Path

PDF_PATH  = "pdf/2.pdf"
OUTPUT    = "components.json"

COMP_META = {
    "KS":  {"name": "Kabelstege",        "en": "Cable Ladder",        "color": "#3b82f6"},
    "KR":  {"name": "Kabelränna",         "en": "Cable Tray",          "color": "#10b981"},
    "FBK": {"name": "Fönsterbänkskanal", "en": "Window Sill Channel", "color": "#f59e0b"},
}

COMP_CODE = re.compile(r'\b(KS|KR|FBK)\b')
OK_PAT    = re.compile(r'ÖK=\s*(\d+)')
UK_PAT    = re.compile(r'UK=\s*(\d+)')


def tight_bbox(block):
    """
    Return the bounding box of just the component annotation within a block.
    Most blocks contain only annotation text so the block bbox is used directly.
    For mixed blocks (e.g. route labels A/B followed by KS annotation) we
    compute a tighter rect starting from the component code span.
    """
    spans = [s for line in block.get("lines", [])
               for s in line.get("spans", [])]

    start = next((i for i, s in enumerate(spans)
                  if COMP_CODE.search(s["text"].strip())), None)
    if start is None:
        return block["bbox"]

    # Any real (non-whitespace) content before the code? → tighten
    pre = [s["text"].strip() for s in spans[:start] if s["text"].strip()]
    if not pre:
        return block["bbox"]   # pure annotation block, trust the block bbox

    # Tighter: spans from the component code onwards
    tail = spans[start:]
    if not tail:
        return block["bbox"]
    return (
        min(s["bbox"][0] for s in tail),
        min(s["bbox"][1] for s in tail),
        max(s["bbox"][2] for s in tail),
        max(s["bbox"][3] for s in tail),
    )


def parse_annotation(text):
    m = COMP_CODE.search(text)
    if not m:
        return None
    code = m.group(1)
    size_m = re.search(rf'\b{code}\s*(\d+)', text)
    ok_m   = OK_PAT.search(text)
    uk_m   = UK_PAT.search(text)
    return {
        "type":      code,
        "name":      COMP_META[code]["name"],
        "en_name":   COMP_META[code]["en"],
        "color":     COMP_META[code]["color"],
        "size":      size_m.group(1) if size_m else None,
        "ok_height": ok_m.group(1)   if ok_m   else None,
        "uk_height": uk_m.group(1)   if uk_m   else None,
    }


def extract(pdf_path=PDF_PATH):
    doc  = fitz.open(pdf_path)
    page = doc[0]

    components = []
    seen       = set()

    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue

        # Collapse all text in block
        spans_text = [s["text"].strip()
                      for line in b.get("lines", [])
                      for s in line.get("spans", [])
                      if s["text"].strip()]
        full = " ".join(spans_text)

        if not COMP_CODE.search(full):
            continue

        parsed = parse_annotation(full)
        if not parsed:
            continue

        bbox   = tight_bbox(b)
        dedup  = (parsed["type"], parsed["size"],
                  tuple(round(x) for x in bbox))
        if dedup in seen:
            continue
        seen.add(dedup)

        components.append({
            "id":        f"{parsed['type']}_{len(components)}",
            **parsed,
            "label":     full,
            "bbox":      [round(x, 2) for x in bbox],
        })

    w = page.rect.width
    h = page.rect.height
    doc.close()
    return {
        "page_width":  w,
        "page_height": h,
        "components":  components,
    }


if __name__ == "__main__":
    data = extract()
    Path(OUTPUT).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Extracted {len(data['components'])} components → {OUTPUT}")
    for c in data["components"]:
        print(f"  [{c['type']}] {c['name']} {c['size'] or '?'}mm  "
              f"ÖK={c['ok_height']} UK={c['uk_height']}  "
              f"bbox={[round(x) for x in c['bbox']]}")
