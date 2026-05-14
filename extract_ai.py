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
import hashlib
import json
import re
import fitz
from pathlib import Path

_CLIENT = None   # lazy-init so import doesn't fail when key is absent

AI_CACHE_DIR = Path("ai_cache")
AI_CACHE_DIR.mkdir(exist_ok=True)

def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    return _CLIENT


def _pdf_hash(pdf_path: str) -> str:
    """SHA-1 of the PDF file contents — used as cache key."""
    h = hashlib.sha1(Path(pdf_path).read_bytes()).hexdigest()
    return h


def _ai_cache_path(pdf_path: str) -> Path:
    return AI_CACHE_DIR / f"{Path(pdf_path).stem}_{_pdf_hash(pdf_path)[:12]}.json"


def load_ai_cache(pdf_path: str) -> dict | None:
    """Return cached AI extraction result for this PDF, or None."""
    p = _ai_cache_path(pdf_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_ai_cache(pdf_path: str, result: dict) -> None:
    p = _ai_cache_path(pdf_path)
    p.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


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


def _tile_b64_offset(pdf_path: str, page_num: int, row: int, col: int,
                     rows: int, cols: int, offset_x: float, offset_y: float,
                     max_px: int = 2400) -> str:
    """
    Same as _tile_b64 but the grid is shifted by (offset_x, offset_y) as a
    fraction of the tile size — so cut lines fall at different positions than
    the standard grid, avoiding the same boundary symbols.
    """
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    w, h = page.rect.width, page.rect.height
    tw, th = w / cols, h / rows
    x0 = (col * tw + offset_x * tw) % w
    y0 = (row * th + offset_y * th) % h
    x1 = min(x0 + tw, w)
    y1 = min(y0 + th, h)
    clip    = fitz.Rect(x0, y0, x1, y1)
    longest = max(clip.width, clip.height)
    scale   = min(max_px / longest, 3.0)
    pix     = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
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
1. Identify EVERY component code present — read codes from annotations and symbol
   labels in the floor plan. Use the legend / förklaringar box only to understand
   what each code means — do NOT count symbols shown inside the legend box itself.
2. For each code, check the component library above:
   • measurement_type "count"  → count all instances placed in the floor plan
     (rooms, corridors, shafts). Exclude any instance inside the legend /
     förklaringar table, title block, or revision table.
   • measurement_type "length" → estimate TOTAL run length in metres using the
     scale printed in the title block (e.g. "SKALA 1:50").
   • Code not in library → use your judgement: fixtures/outlets/sensors → count;
     trays/pipes/conduits → length.
3. Read the drawing scale from the title block.
4. Capture width/size annotations (e.g. "KS 400" → width_mm 400).
5. Capture mounting heights (ÖK = top of tray, UK = bottom, in mm ÖFG).
6. Note fire ratings (e.g. "EI 30-C") if shown alongside a component.

YOUR RESPONSE MUST BE A RAW JSON OBJECT WITH NO TEXT BEFORE OR AFTER IT.
No markdown fences, no explanation, no commentary. The very first character must be {{ and the last must be }}.

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
    """Count specific symbols in one tile. Returns {code: count}."""
    codes_desc = "\n".join(
        f'  "{c["code"]}": {c["name"]}' for c in codes
    )
    prompt = f"""You are counting electrical/building-services symbols in one section of a drawing.

Count ONLY these specific symbols:
{codes_desc}

Rules:
- Count every visible instance of each symbol in this image tile.
- Do NOT count the same symbol twice.
- IMPORTANT: If a symbol is partially cut off at any edge of the image, do NOT count it.
  Only count symbols that are fully visible within the tile boundary.
- If you see 0 of a symbol, return 0 — do not omit it.

IMPORTANT: Your entire response must be a single JSON object with no text before or after it.
Do not write any explanation. Do not use markdown. Just output the raw JSON object.
{{{{"P11": 3, "DA": 1, "P12": 0}}}}"""

    resp = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    result = _extract_json(resp.content[0].text)
    return {c["code"]: int(result.get(c["code"], 0)) for c in codes}


def _count_one_symbol(b64: str, code: str, name: str, hint: int | None = None) -> int:
    """
    Count a single symbol in the floor plan, ignoring legend.
    hint: Pass 1 rough count — used as example value so Claude anchors near the right range.
    """
    example_val = hint if hint is not None else 0
    prompt = f"""You are counting electrical/building-services symbols installed in a building floor plan.

CRITICAL — before you count anything:
  Locate the FÖRKLARINGAR / LEGEND / BETECKNINGAR table in the drawing.
  This is a bordered box (often bottom-right or top-right) that lists what each
  symbol looks like. Every symbol shown INSIDE that box is just an example —
  it is NOT installed in the building. Do not count any symbol inside that box.

Count only this symbol, and only where it appears INSIDE rooms, corridors,
or building spaces of the floor plan:
  "{code}": {name}

Rules:
- If a symbol is in the legend/förklaringar box → do NOT count it.
- If a symbol is in the title block or revision table → do NOT count it.
- If you are unsure whether something is really the symbol → do NOT count it.
- Count each physical instance exactly once.

YOUR ENTIRE RESPONSE MUST BE ONLY A RAW JSON OBJECT — no explanation, no markdown.
{{"{code}": {example_val}}}"""

    resp = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    result = _extract_json(resp.content[0].text)
    return int(result.get(code, 0))


def _count_full_image(b64: str, codes: list[dict], hints: dict[str, int] | None = None) -> dict[str, int]:
    """
    Pass A/C — count each symbol individually using the legend-aware prompt.
    hints parameter kept for signature compatibility but no longer used.
    """
    return {c["code"]: _count_one_symbol_legend(b64, c["code"], c["name"]) for c in codes}


def _count_one_symbol_legend(b64: str, code: str, name: str) -> int:
    """Pass B variant — same single-symbol approach but different framing."""
    prompt = f"""You are a quantity surveyor counting symbols in a building services floor plan.

The drawing contains a "FÖRKLARINGAR", "LEGEND", or "BETECKNINGAR" box
(a bordered table, often in a corner) that shows what each symbol means.
Symbols inside that box are just examples — they must NOT be counted.

Count how many "{code}" ({name}) symbols appear in the actual floor plan
(rooms, corridors, shafts) — never in the legend, title block, or revision table.

YOUR ENTIRE RESPONSE MUST BE ONLY A RAW JSON OBJECT — no explanation, no markdown.
Format: {{"{code}": <your_count>}}"""

    resp = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    result = _extract_json(resp.content[0].text)
    return int(result.get(code, 0))


def _count_legend_aware(b64: str, codes: list[dict]) -> dict[str, int]:
    """Pass B — individual calls with legend-aware framing."""
    return {c["code"]: _count_one_symbol_legend(b64, c["code"], c["name"]) for c in codes}


def _count_systematic_scan(b64: str, codes: list[dict]) -> dict[str, int]:
    """
    Pass C — spatial-quadrant count.
    Claude mentally counts per quadrant and sums them, skipping annotation areas.
    Different decomposition from Passes A and B.
    """
    codes_desc = "\n".join(f'  "{c["code"]}": {c["name"]}' for c in codes)
    prompt = f"""You are a quantity surveyor counting symbols in a building services floor plan.

Mentally divide the floor plan into four quadrants (top-left, top-right, bottom-left,
bottom-right) and count the symbols in each quadrant. Then sum them for the total.

Do NOT count any symbol inside a legend table, förklaringar box, title block,
or any bordered annotation area — those are diagram examples, not real installed items.

Symbols to count:
{codes_desc}

YOUR ENTIRE RESPONSE MUST BE ONLY A FLAT RAW JSON OBJECT — one key per symbol code,
value is the TOTAL across all quadrants. No per-quadrant breakdown, no explanation,
no markdown. The first character must be {{ and the last must be }}.
{{{{"P11": 27, "DA": 3}}}}"""

    resp = _client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    result = _extract_json(resp.content[0].text)
    # Guard: if Claude still returns nested dict, fall back to 0 for that code
    flat = {}
    for c in codes:
        val = result.get(c["code"], 0)
        flat[c["code"]] = int(val) if isinstance(val, (int, float)) else 0
    return flat


def extract_with_ai(drawing_pdf_path: str, component_library: dict | None) -> dict:
    """
    Two-pass AI extraction:
      Pass 1 — full drawing → identify all components, measure lengths, get rough counts.
      Pass 2 — tile each count-type component through a 2×2 grid → sum tile counts.

    Results are cached to ai_cache/ by PDF hash so hot-reloads skip Claude calls.
    Returns a dict in the same top-level schema as extract().
    """
    cached = load_ai_cache(drawing_pdf_path)
    if cached is not None:
        print(f"  [AI] loaded from cache: {Path(drawing_pdf_path).name}")
        return cached

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

    # ── Passes A/B — two independent counts ────────────────────────────────────
    # Pass A: reuse the quantity already produced by Pass 1 (_ask_full_drawing).
    #         It reasons holistically about the whole drawing — no extra tokens.
    # Pass B: dedicated count with explicit legend-exclusion instruction.
    #         Different prompt context → genuinely independent result.
    # Pass C: legend-aware count from a different angle (exclude förklaringar).
    if count_items:
        pdf_stem = Path(drawing_pdf_path).stem

        print(f"\n  [AI] Counting (3 passes) for {pdf_stem}:")

        # Three independent runs of the legend-aware per-symbol prompt.
        # Same prompt, stochastic variation → genuine independent counts.
        grid_a = _count_full_image(full_b64, count_items)
        print(f"  Pass A: { {c['code']: grid_a.get(c['code'],0) for c in count_items} }")

        grid_b = _count_legend_aware(full_b64, count_items)
        print(f"  Pass B: { {c['code']: grid_b.get(c['code'],0) for c in count_items} }")

        grid_c = _count_full_image(full_b64, count_items)
        print(f"  Pass C: { {c['code']: grid_c.get(c['code'],0) for c in count_items} }")

        print(f"\n  {'Code':<12} {'A':>4} {'B':>4} {'C':>4}  result")
        for item in count_items:
            code = item["code"]
            a, b, c = grid_a.get(code, 0), grid_b.get(code, 0), grid_c.get(code, 0)

            if a == b == c:
                final, confidence = a, "high"
            elif a == b:
                final, confidence = a, "medium"
            elif a == c:
                final, confidence = a, "medium"
            elif b == c:
                final, confidence = b, "medium"
            else:
                # All three differ — trust Pass A (holistic pass1 analysis)
                final, confidence = a, "low"

            item["count_grid_a"]     = a
            item["count_grid_b"]     = b
            item["count_grid_c"]     = c
            item["count_confidence"] = confidence
            item["quantity"]         = final
            icon = {"high": "✓", "medium": "⚠", "low": "✗"}[confidence]
            print(f"  {code:<12} {a:>4} {b:>4} {c:>4}  {final} {icon}")

    # Reassemble components list
    raw["components"] = count_items + length_items
    result = _normalise(raw, page_w, page_h)
    save_ai_cache(drawing_pdf_path, result)
    return result


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
            # Propagate per-grid validator counts so the UI can display them
            if "count_grid_a" in item:
                s["count_grid_a"]    = item["count_grid_a"]
                s["count_grid_b"]    = item["count_grid_b"]
                s["count_grid_c"]    = item["count_grid_c"]
                s["count_confidence"] = item.get("count_confidence", "unknown")
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
