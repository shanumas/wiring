"""
AI-vision extraction pipeline using Claude claude-sonnet-4-6.

Two-stage process
─────────────────
Stage 1 — Component library (run once per project)
  description.pdf → Claude Vision → {code: {name, measurement_type, unit, notes}}

  measurement_type is the key decision:
    "count"  → point-installed items (fixtures, sockets, sensors, panels …)
                AI counts visible instances; length is irrelevant
    "length" → line-installed items (cable trays, conduits, pipes, bus-bars …)
                AI estimates total run length using the drawing scale

Stage 2 — Drawing takeoff (run once per drawing PDF)
  drawing.pdf + component library → Claude Vision
    → per-component quantity (pcs or metres)
    → returned in our standard schema so the frontend works unchanged
"""

import anthropic
import base64
import json
import re
import fitz
from pathlib import Path

_CLIENT = None   # lazy-init so import doesn't fail when key is absent

def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    return _CLIENT


# ── PDF → base64 PNG ──────────────────────────────────────────────────────────

def _page_to_b64(pdf_path: str, page_num: int = 0, max_px: int = 2400) -> str:
    """Render one PDF page to a base64-encoded PNG, capped at max_px on the long side."""
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    longest = max(page.rect.width, page.rect.height)
    scale   = min(max_px / longest, 2.0)
    pix     = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    data    = base64.standard_b64encode(pix.tobytes("png")).decode()
    doc.close()
    return data


# ── JSON extraction helper ────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Pull the first complete JSON object out of a Claude response."""
    # Try fenced code block first
    m = re.search(r'```json\s*([\s\S]+?)\s*```', text)
    candidate = m.group(1) if m else None

    if candidate is None:
        start = text.find('{')
        end   = text.rfind('}') + 1
        if start >= 0 and end > start:
            candidate = text[start:end]

    if candidate is None:
        raise ValueError(f"No JSON in response: {text[:300]}")

    # First try strict parse
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: use json5 / demjson-style repair heuristics inline
    # Remove trailing commas before } or ] (common Claude mistake)
    fixed = re.sub(r',\s*([}\]])', r'\1', candidate)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON: {e}\nRaw (first 500): {candidate[:500]}") from e


# ── Stage 1: component library from description.pdf ──────────────────────────

LIBRARY_CACHE_PATH = Path("component_library.json")

def build_component_library(description_pdf_path: str) -> dict:
    """
    Send every page of description.pdf to Claude and extract the component
    library: what each code means and whether to count or measure it.

    Result is cached to component_library.json.
    """
    doc     = fitz.open(description_pdf_path)
    n_pages = len(doc)
    doc.close()

    content = []
    for i in range(n_pages):
        b64 = _page_to_b64(description_pdf_path, page_num=i, max_px=2400)
        content.append({
            "type":   "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64},
        })

    content.append({"type": "text", "text": """
You are reading a component description / legend document from a building services
drawing set (likely Swedish electrical or multi-discipline drawings).

Extract EVERY component or symbol listed. For each one produce:

  code             – the abbreviation used in drawings (e.g. "P11", "DA", "KS", "NS 3B")
  name             – full description (keep Swedish + add English gloss if clear)
  measurement_type – EXACTLY one of:
      "count"   point-installed item: light fixture, socket, switch, sensor,
                smoke detector, speaker, access point, panel, valve, pump …
                (anything installed at a single location — quantity = number of pieces)
      "length"  line-installed item: cable tray, cable ladder, conduit, pipe,
                duct, bus-bar, wiring system, railing …
                (anything installed along a continuous run — quantity = metres)
  unit             – "pcs" for count, "m" for length
  notes            – key specs (wattage, IP rating, size, colour …) or "" if none

Return ONLY valid JSON, no other text:
{
  "components": [
    {"code": "P11", "name": "...", "measurement_type": "count", "unit": "pcs", "notes": "..."},
    {"code": "KS",  "name": "Kabelstege / Cable Ladder",
     "measurement_type": "length", "unit": "m", "notes": "width per drawing"}
  ]
}
"""})

    resp = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}],
    )
    library = _extract_json(resp.content[0].text)

    # Cache to disk
    LIBRARY_CACHE_PATH.write_text(
        json.dumps(library, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return library


def load_component_library() -> dict | None:
    """Return cached library from disk, or None if not yet built."""
    if LIBRARY_CACHE_PATH.exists():
        return json.loads(LIBRARY_CACHE_PATH.read_text(encoding="utf-8"))
    return None


# ── Stage 2: drawing takeoff ──────────────────────────────────────────────────

def _tile_b64(pdf_path: str, page_num: int, row: int, col: int,
              rows: int, cols: int, max_px: int = 2400) -> str:
    """Render one tile (row, col) of a PDF page as base64 PNG."""
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    w, h = page.rect.width, page.rect.height
    tw, th = w / cols, h / rows
    clip = fitz.Rect(col * tw, row * th, (col + 1) * tw, (row + 1) * th)
    longest = max(clip.width, clip.height)
    scale   = min(max_px / longest, 3.0)
    mat     = fitz.Matrix(scale, scale)
    pix     = page.get_pixmap(matrix=mat, clip=clip)
    data    = base64.standard_b64encode(pix.tobytes("png")).decode()
    doc.close()
    return data


def _ask_full_drawing(b64: str, lib_block: str) -> dict:
    """
    Pass 1 — full drawing at once.
    Identifies all component codes, their types, and gives rough quantities.
    Also extracts scale, drawing_type, and all length-based measurements.
    """
    prompt = f"""You are an expert quantity surveyor performing a drawing takeoff.

{lib_block}

Analyse this drawing image and return a full quantity takeoff.

Instructions:
1. Identify EVERY component visible — read codes from text annotations, symbol
   labels, and any legend shown in the drawing itself.
2. For each code, check the component library above:
   • measurement_type "count"  → count all visible instances.
   • measurement_type "length" → estimate TOTAL run length in metres using the
     scale printed in the title block (e.g. "SKALA 1:50").
   • Code not in library → use your judgement: fixtures/outlets/sensors → count;
     trays/pipes/conduits → length.
3. Read the drawing scale from the title block.
4. Capture width/size annotations (e.g. "KS 400" → width_mm 400).
5. Capture mounting heights (ÖK = top of tray, UK = bottom, in mm ÖFG).
6. Note fire ratings (e.g. "EI 30-C") if shown alongside a component.

Return ONLY valid JSON — no markdown, no explanation:
{{
  "scale": "1:50",
  "drawing_type": "one-line description of what this drawing shows",
  "components": [
    {{"code": "P11", "name": "full name", "measurement_type": "count",
      "quantity": 8, "unit": "pcs", "width_mm": null,
      "ok_height_mm": null, "uk_height_mm": null, "fire_rating": null}},
    {{"code": "KS", "name": "Kabelstege", "measurement_type": "length",
      "quantity": 45.5, "unit": "m", "width_mm": 400,
      "ok_height_mm": 2700, "uk_height_mm": null, "fire_rating": null}}
  ]
}}"""

    resp = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return _extract_json(resp.content[0].text)


def _count_in_tile(b64: str, codes: list[dict]) -> dict[str, int]:
    """
    Pass 2 — count specific symbols in one tile.
    Returns {code: count} for each requested code.
    """
    codes_desc = "\n".join(
        f'  "{c["code"]}": {c["name"]}' for c in codes
    )
    prompt = f"""You are counting electrical/building-services symbols in one section of a drawing.

Count ONLY these specific symbols:
{codes_desc}

Rules:
- Count every visible instance of each symbol in this image tile.
- Do NOT count the same symbol twice.
- If a symbol is partially cut off at the edge, count it only if more than half is visible.
- If you see 0 of a symbol, return 0 — do not omit it.

Return ONLY valid JSON, no other text:
{{{{"P11": 3, "DA": 1, ...}}}}"""

    resp = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    result = _extract_json(resp.content[0].text)
    # Ensure all requested codes are present
    return {c["code"]: int(result.get(c["code"], 0)) for c in codes}


def extract_with_ai(drawing_pdf_path: str, component_library: dict | None) -> dict:
    """
    Two-pass AI extraction:
      Pass 1 — full drawing → identify all components, measure lengths, get rough counts.
      Pass 2 — tile each count-type component through a 2×2 grid → sum tile counts.

    Returns a dict in the same top-level schema as extract().
    """
    doc    = fitz.open(drawing_pdf_path)
    page   = doc[0]
    page_w = page.rect.width
    page_h = page.rect.height
    doc.close()

    if component_library:
        lib_block = (
            "COMPONENT LIBRARY (from the project description document):\n"
            + json.dumps(component_library.get("components", []),
                         ensure_ascii=False, indent=2)
        )
    else:
        lib_block = (
            "No component library available yet — infer each component type "
            "from context and drawing conventions."
        )

    # ── Pass 1: full drawing ───────────────────────────────────────────────────
    full_b64 = _page_to_b64(drawing_pdf_path, max_px=2400)
    raw = _ask_full_drawing(full_b64, lib_block)

    # Separate count vs length components
    count_items  = [c for c in raw.get("components", [])
                    if c.get("measurement_type", "count") == "count"]
    length_items = [c for c in raw.get("components", [])
                    if c.get("measurement_type", "count") != "count"]

    # ── Pass 2: tile counting for count-type components ────────────────────────
    if count_items:
        ROWS, COLS = 2, 2
        tile_totals: dict[str, int] = {c["code"]: 0 for c in count_items}

        for row in range(ROWS):
            for col in range(COLS):
                tile_b64 = _tile_b64(drawing_pdf_path, 0, row, col, ROWS, COLS, max_px=2400)
                tile_counts = _count_in_tile(tile_b64, count_items)
                for code, n in tile_counts.items():
                    tile_totals[code] = tile_totals.get(code, 0) + n

        # Replace quantities in count_items with tiled totals
        for item in count_items:
            item["quantity"] = tile_totals.get(item["code"], item.get("quantity", 0))

    # Reassemble components list
    raw["components"] = count_items + length_items
    return _normalise(raw, page_w, page_h)


# ── Schema normalisation ──────────────────────────────────────────────────────

_COLORS = {
    "KS": "#1e4d8c", "KR": "#1a6b45", "FBK": "#8a6200", "Tomrör": "#5a3a7a",
}
_PALETTE = [
    "#9c2c2c","#1a5c6e","#7a4a00","#2c5a3a",
    "#4a2c7a","#1a4a5c","#6e3a1a","#1a4a2c",
]

def _color(code: str) -> str:
    return _COLORS.get(code, _PALETTE[sum(ord(c) for c in code) % len(_PALETTE)])


def _normalise(raw: dict, page_w: float, page_h: float) -> dict:
    """Convert Claude's raw JSON into the standard extract() schema."""
    components = []
    summary    = []

    for i, item in enumerate(raw.get("components", [])):
        code  = str(item.get("code", "?")).strip()
        name  = item.get("name", code)
        mtype = item.get("measurement_type", "count")
        qty   = item.get("quantity", 0)
        unit  = item.get("unit", "pcs")
        fire  = item.get("fire_rating") or None
        wmm   = item.get("width_mm")
        ok    = item.get("ok_height_mm")
        uk    = item.get("uk_height_mm")

        try:
            qty = float(qty)
        except (TypeError, ValueError):
            qty = 0.0

        comp = {
            "id":               f"AI_{code}_{i}",
            "type":             code,
            "name":             name,
            "en_name":          "",
            "color":            _color(code),
            "size":             str(int(wmm)) if wmm else None,
            "ok_height":        str(int(ok))  if ok  else None,
            "uk_height":        str(int(uk))  if uk  else None,
            "is_vertical":      False,
            "fire_rating":      fire,
            "label":            f"{code} — {qty:g} {unit}",
            "bbox":             None,     # AI gives totals, not per-instance positions
            "occurrences":      1,
            "measurement_type": mtype,
            "quantity":         qty,
            "unit":             unit,
        }
        if mtype == "length":
            comp["length_m"] = qty
        components.append(comp)

        s = {
            "system":           code,
            "name":             name,
            "orientation":      "horizontal",
            "width_mm":         int(wmm) if wmm else None,
            "ok_ofg_mm":        int(ok)  if ok  else None,
            "uk_ofg_mm":        int(uk)  if uk  else None,
            "fire_rating":      fire,
            "measurement_type": mtype,
            "unit":             unit,
        }
        if mtype == "length":
            s["count"]          = 1
            s["total_length_m"] = qty
        else:
            s["count"] = max(0, int(round(qty)))
        summary.append(s)

    return {
        "page_width":      page_w,
        "page_height":     page_h,
        "drawing_type":    raw.get("drawing_type", "unknown"),
        "scale":           raw.get("scale", "unknown"),
        "components":      components,
        "summary":         summary,
        "extraction_mode": "ai",
    }
