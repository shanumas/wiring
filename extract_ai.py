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

_CLIENT = None        # lazy-init so import doesn't fail when key is absent
_CLIENT_HAIKU = None  # cheaper model for counting passes

AI_CACHE_DIR = Path("ai_cache")
AI_CACHE_DIR.mkdir(exist_ok=True)

def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic()
    return _CLIENT

def _client_haiku():
    global _CLIENT_HAIKU
    if _CLIENT_HAIKU is None:
        _CLIENT_HAIKU = anthropic.Anthropic()
    return _CLIENT_HAIKU

SONNET = "claude-sonnet-4-6"
HAIKU  = "claude-haiku-4-5-20251001"


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


# ── PDF text-based symbol counting ───────────────────────────────────────────

def _count_from_pdf_text(pdf_path: str, codes: list[dict],
                          legend_bbox: dict | None = None) -> dict[str, int]:
    """
    Count symbol occurrences by extracting text from the PDF page.
    Excludes any text in the FÖRKLARINGAR / legend section.

    Strategy (zero tokens):
    1. Scan the PDF text for a FÖRKLARINGAR / LEGEND header to find where
       the symbol legend section starts.  All text whose y-coordinate is
       at or above that header (minus a 120-pt buffer to catch the symbol
       code column that sits above the title row) is excluded.
    2. Optionally also exclude text inside a Claude-supplied legend_bbox.

    legend_bbox: {"x1": %, "y1": %, "x2": %, "y2": %} as percentages of
                 page.rect dimensions, or None.
    """
    doc  = fitz.open(pdf_path)
    page = doc[0]
    w, h = page.rect.width, page.rect.height

    # ── 1. Locate the FÖRKLARINGAR label in the PDF text stream ──────────────
    # In rotated PDFs the y-axis from get_text("dict") runs in a transformed
    # coordinate space; we find the label and use y1 - 120 pts as a cut-off
    # so that symbol codes listed in rows just above the header are excluded too.
    legend_y_cut = None
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            for sp in ln["spans"]:
                if re.search(r'F.RKLARINGAR|FÖRKLARINGAR|FORKLARINGAR|LEGEND',
                             sp["text"], re.IGNORECASE):
                    legend_y_cut = sp["bbox"][1] - 120   # include a safe buffer
                    break
            if legend_y_cut is not None:
                break
        if legend_y_cut is not None:
            break

    # ── 2. Build exclusion rect from Claude-supplied legend_bbox (if any) ────
    excl = None
    if legend_bbox and legend_bbox.get("x2", 0) > 0:
        excl = fitz.Rect(
            w * legend_bbox["x1"] / 100,
            h * legend_bbox["y1"] / 100,
            w * legend_bbox["x2"] / 100,
            h * legend_bbox["y2"] / 100,
        )

    # ── 3. Collect text spans that are in the drawing area ───────────────────
    # Spans are joined with "|" so that letters at the end of one span cannot
    # bleed into the start of a code in the next span (e.g. "A|P12" not "AP12").
    parts = []
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            for sp in ln["spans"]:
                bbox = sp["bbox"]
                # Exclude text at or past the FÖRKLARINGAR section
                if legend_y_cut is not None and bbox[1] >= legend_y_cut:
                    continue
                # Also exclude by Claude-identified bbox rect (if available)
                if excl and fitz.Rect(bbox).intersects(excl):
                    continue
                parts.append(sp["text"])

    full_text = "|".join(parts)
    doc.close()

    counts = {}
    for c in codes:
        code = c["code"]
        # Three boundary guards, in order of precedence:
        # • Lookbehind (?<![A-Za-z…]) — code must not be preceded by a letter.
        #   Digits are allowed before (packed spans: "P11P11N1" → N1 matches).
        #   Spans are joined with "|" so cross-span letter bleed is prevented.
        # • Lookahead (?!\d) — code must not be followed by a digit, so "D1"
        #   won't match inside "D10" or "SLÖJD15" (the 1 is followed by 5).
        # • Lookahead (?!-[A-Z0-9]) — code must not be followed by a hyphen
        #   then alphanumeric, so "N1" won't match inside "N1-R".
        #   A plain letter after the code IS still allowed, which handles packed
        #   spans like "N1F2" where F starts the next code.
        pattern = (
            r"(?<![A-Za-z\u00C0-\u024F])"
            + re.escape(code)
            + r"(?!\d)(?!-[A-Z0-9])"
        )
        counts[code] = len(re.findall(pattern, full_text))
    return counts


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


def _find_legend_bbox(b64: str) -> dict | None:
    """
    Ask Claude where the FÖRKLARINGAR / legend box is located.
    Returns {"x1": %, "y1": %, "x2": %, "y2": %} as percentages of image dimensions,
    or None if no legend box found.
    """
    prompt = """This is a Swedish building services drawing.
Locate the FÖRKLARINGAR, LEGEND, or BETECKNINGAR box — a bordered table that
lists and explains what each symbol means. It is usually in a corner of the drawing.

Return its location as percentage of the full image width/height:
{"x1": <left%>, "y1": <top%>, "x2": <right%>, "y2": <bottom%>}

If no such box exists, return: {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

ONLY output the raw JSON object, nothing else."""

    resp = _client().messages.create(
        model=SONNET,
        max_tokens=64,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    try:
        bbox = _extract_json(resp.content[0].text)
        if bbox.get("x2", 0) == 0 and bbox.get("y2", 0) == 0:
            return None
        return bbox
    except Exception:
        return None


def _page_to_b64_masked(pdf_path: str, legend_bbox: dict, page_num: int = 0, max_px: int = 2400) -> str:
    """
    Render the PDF page with the legend area whited out so Claude cannot see it.
    legend_bbox: {"x1": %, "y1": %, "x2": %, "y2": %} as percentages.
    """
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    w, h = page.rect.width, page.rect.height

    x1 = w * legend_bbox["x1"] / 100
    y1 = h * legend_bbox["y1"] / 100
    x2 = w * legend_bbox["x2"] / 100
    y2 = h * legend_bbox["y2"] / 100

    # Draw a white rectangle over the legend area
    rect = fitz.Rect(x1, y1, x2, y2)
    page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))

    longest = max(w, h)
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
        model=SONNET,
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
1. Identify EVERY component code present — read codes EXACTLY as printed in the
   drawing (e.g. "V18" is not "V11", "D2" is not "D7"). Read each digit carefully.
   Use the legend / förklaringar box only to understand what each code means —
   do NOT count symbols shown inside the legend box itself.
   IMPORTANT: Only use codes that actually appear in the component library above
   or that you can clearly read from the drawing labels. Do not invent codes.
2. For each code, check the component library above:
   • measurement_type "count"  → count all instances placed in the floor plan
     (rooms, corridors, shafts). Exclude any instance inside the legend /
     förklaringar table, title block, or revision table.
   • measurement_type "length" → estimate TOTAL run length in metres using the
     scale printed in the title block (e.g. "SKALA 1:50").
   • Code not in library → use your judgement: fixtures/outlets/sensors → count;
     trays/pipes/conduits → length.
   IMPORTANT: Prefabricated / plug-in wiring systems (e.g. WAGO Sladdställ,
   snabbkopplingssystem, pendelupphängning) are delivered from the factory in
   fixed lengths. They must ALWAYS be classified as "count" (antal), NOT "length".
   Even though they involve cables, they are ordered by piece, not by metre.
3. Read the drawing scale from the title block.
4. Capture width/size annotations.  The code field must ONLY contain the bare
   letter code (e.g. "KS", "FBK", "KR") — never include the width number or
   mounting height in the code.  Extract those separately:
     "KS 400"          → code "KS",  width_mm 400
     "KS 400 UK=2700"  → code "KS",  width_mm 400, uk_height_mm 2700
     "FBK 100"         → code "FBK", width_mm 100
   If the same code appears at multiple mounting heights, return ONE entry per
   distinct height combination, each with the correct ok_height_mm / uk_height_mm.
5. Capture mounting heights (ÖK = top of tray, UK = bottom, in mm ÖFG).
   IMPORTANT: each height annotation belongs to the component in the SAME legend row.
   Never carry a height from one legend row to an adjacent row.
   ÖK means the TOP edge of the item; UK means the BOTTOM edge. Do not swap them.
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
        model=SONNET,
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
        model=SONNET,
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
        model=SONNET,
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
        model=SONNET,
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


def _verify_presence(b64: str, codes: list[dict]) -> set[str]:
    """
    Ask Claude which codes are actually present in the floor plan.
    Returns a set of codes that exist. Codes not in the set should be counted as 0.
    Uses a single batched call for efficiency.
    """
    codes_desc = "\n".join(f'  "{c["code"]}": {c["name"]}' for c in codes)
    prompt = f"""You are reviewing a building services floor plan.

For each symbol code below, answer whether it actually appears in the FLOOR PLAN
(not in the legend/förklaringar box, not in the title block — only in the actual plan).

Symbols to check:
{codes_desc}

Return a JSON object where the value is true if the symbol exists in the floor plan,
false if it does not exist at all.
YOUR ENTIRE RESPONSE MUST BE ONLY A RAW JSON OBJECT — no explanation, no markdown.
{{"P11": true, "D3": false}}"""

    resp = _client().messages.create(
        model=SONNET,
        max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    try:
        result = _extract_json(resp.content[0].text)
        return {c["code"] for c in codes if result.get(c["code"], True)}
    except Exception:
        # On parse failure, assume all present (safe fallback)
        return {c["code"] for c in codes}


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
        model=SONNET,
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


def _inject_variant_codes(pdf_path: str, count_items: list[dict],
                           length_codes: set[str] | None = None) -> list[dict]:
    """
    Supplement the Pass-1 component list by scanning the PDF drawing-body text
    for component codes that Claude missed.

    Two discovery passes (zero tokens each):

    1. Hyphen-suffix variants — for every known code X, find any "X-Y" in the
       text and inject it (e.g. "N1" → discovers "N1-R").

    2. General code discovery — scan the whole drawing body for tokens that
       look like component codes (1–4 uppercase letters + 1–2 digits, optional
       hyphen suffix) and inject any not already in the list.  Tokens with 3+
       digit suffixes are skipped because those are typically room labels
       (A113, A117 …) not components.
    """
    # ── Build drawing-body text ───────────────────────────────────────────────
    doc  = fitz.open(pdf_path)
    page = doc[0]
    legend_y_cut: float | None = None
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            for sp in ln["spans"]:
                if re.search(r'F.RKLARINGAR|FÖRKLARINGAR', sp["text"], re.IGNORECASE):
                    legend_y_cut = sp["bbox"][1] - 120
                    break
            if legend_y_cut is not None:
                break
        if legend_y_cut is not None:
            break

    # Two text buffers:
    # • full_text  — all spans; used for hyphen-variant discovery (variants of
    #               known codes must be counted regardless of text colour).
    # • dark_text  — only spans whose text colour is dark (≤ 0x404040); used
    #               for general code discovery to skip room labels / grid text
    #               that is deliberately printed in a light grey font.
    _DARK_THRESHOLD = 0x404040   # colours above this are "light" (grey, etc.)

    parts_all: list[str] = []
    parts_dark: list[str] = []
    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            for sp in ln["spans"]:
                if legend_y_cut is not None and sp["bbox"][1] >= legend_y_cut:
                    continue
                parts_all.append(sp["text"])
                if sp.get("color", 0) <= _DARK_THRESHOLD:
                    parts_dark.append(sp["text"])

    full_text = "|".join(parts_all)
    dark_text = "|".join(parts_dark)
    doc.close()

    known_codes = {c["code"] for c in count_items}
    extras: list[dict] = []

    def _add(code: str, name: str, notes: str = "") -> None:
        if (length_codes or set()) and code in (length_codes or set()):
            return  # skip — already measured as a length item
        if code not in known_codes:
            known_codes.add(code)
            extras.append({
                "code":             code,
                "name":             name,
                "measurement_type": "count",
                "unit":             "pcs",
                "notes":            notes,
                "quantity":         0,
            })
            print(f"  [auto-discover] {code}: {name}")

    # ── Pass 1: hyphen-suffix variants of known codes (all text) ─────────────
    for item in list(count_items):
        base = re.escape(item["code"])
        for m in re.finditer(
            r"(?<![A-Za-z\u00C0-\u024F])" + base + r"-([A-Z0-9]+)(?!\d)(?!-[A-Z0-9])",
            full_text,
        ):
            variant = item["code"] + "-" + m.group(1)
            _add(variant, item.get("name", variant) + f" (variant of {item['code']})",
                 item.get("notes", ""))

    # ── Pass 2: general code scan (dark text only) ────────────────────────────
    # Only scans dark-coloured spans so that room labels / annotations printed
    # in light grey (colour > 0x404040) are not mistaken for components.
    # Matches 1–4 uppercase letters + 1–2 digits (+ optional hyphen suffix).
    general_pat = re.compile(
        r"(?<![A-Za-z\u00C0-\u024F])"
        r"([A-ZÅÄÖ]{1,4}\d{1,2}(?:-[A-Z0-9]{1,3})?)"
        r"(?!\d)(?!-[A-Z0-9])"
    )
    for m in general_pat.finditer(dark_text):
        _add(m.group(1), m.group(1))

    return count_items + extras


def _is_text_countable(code: str) -> bool:
    """
    Return True when the symbol code is safe to count via PDF text extraction.

    A code is text-safe when it is at least 2 characters long and contains at
    least one digit.  This ensures it is specific enough to avoid false matches
    against common words or single Swedish letters (Å, A, T …) that would
    produce thousands of spurious hits.

    Examples:
      "P11"  → True   (letter + digits)
      "D1"   → True   (letter + digit)
      "V17"  → True
      "F1"   → True
      "Å"    → False  (single letter — use vision)
      "A"    → False  (too common)
      "OT"   → False  (no digit — could appear in Swedish text)
    """
    return len(code) >= 2 and bool(re.search(r"\d", code))


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
    # Also strip codes that are purely letters with no digits — these are
    # architectural grid references (A, B, AA …) not real component codes.
    _GRID_REF = re.compile(r'^[A-ZÅÄÖ]{1,3}$')
    raw_comps = [c for c in raw.get("components", [])
                 if not _GRID_REF.match(str(c.get("code", "")).strip())]

    count_items  = [c for c in raw_comps
                    if c.get("measurement_type", "count") == "count"]
    length_items = [c for c in raw_comps
                    if c.get("measurement_type", "count") != "count"]

    # ── Discover hyphen-suffix variants missed by Pass 1 ──────────────────────
    # Claude often identifies "N1" but not "N1-R" as a separate component.
    # Scan the PDF text for any "{known_code}-{suffix}" patterns and inject
    # them as extra count items so they get properly counted.
    # Pass length_codes so that auto-discovery doesn't re-add a code that Pass 1
    # already correctly classified as a length item (e.g. KS, KR, FBK …).
    length_codes = {c["code"] for c in length_items}
    count_items = _inject_variant_codes(drawing_pdf_path, count_items, length_codes)

    # ── Counting passes ────────────────────────────────────────────────────────
    if count_items:
        pdf_stem = Path(drawing_pdf_path).stem
        print(f"\n  [AI] Counting for {pdf_stem}:")

        # Find legend bbox once (used by text extraction and masked image)
        legend_bbox = _find_legend_bbox(full_b64)
        if legend_bbox:
            print(f"  Legend box found: {legend_bbox}")
        else:
            print(f"  No legend box found — using FÖRKLARINGAR text detection only")

        # ── Split: text-safe vs vision-only symbols ────────────────────────────
        # A code is text-safe when it contains at least one digit and is ≥2 chars:
        # that makes it unique enough that PDF text search won't produce false
        # matches against common words or single letters.
        text_items  = [c for c in count_items if _is_text_countable(c["code"])]
        vision_items = [c for c in count_items if not _is_text_countable(c["code"])]

        # Text extraction — exact, zero tokens
        if text_items:
            text_counts = _count_from_pdf_text(drawing_pdf_path, text_items, legend_bbox)
            print(f"  Text counts (exact): { {c['code']: text_counts.get(c['code'], 0) for c in text_items} }")
        else:
            text_counts = {}

        # Vision counting — 3 independent passes for majority-vote confidence.
        # Uses the full unmasked image so Claude can see what the symbol looks
        # like in the legend before searching for it in the floor plan.
        vision_a: dict[str, int] = {}
        vision_b: dict[str, int] = {}
        vision_c: dict[str, int] = {}
        if vision_items:
            print(f"  Vision counting for {len(vision_items)} graphical symbol(s): "
                  f"{[c['code'] for c in vision_items]}")
            for c in vision_items:
                code_v, name_v = c["code"], c["name"]
                va = _count_one_symbol(full_b64, code_v, name_v)
                vb = _count_one_symbol(full_b64, code_v, name_v)
                vc = _count_one_symbol(full_b64, code_v, name_v)
                vision_a[code_v] = va
                vision_b[code_v] = vb
                vision_c[code_v] = vc
                print(f"    {code_v}: A={va} B={vb} C={vc}")

        # Merge into three grids (text items are identical across all three)
        grid_a = {**text_counts, **vision_a}
        grid_b = {**text_counts, **vision_b}
        grid_c = {**text_counts, **vision_c}

        print(f"\n  {'Code':<12} {'A':>4} {'B':>4} {'C':>4}  result  method")
        for item in count_items:
            code = item["code"]
            is_text = _is_text_countable(code)

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
                final, confidence = a, "low"

            item["count_grid_a"]     = a
            item["count_grid_b"]     = b
            item["count_grid_c"]     = c
            item["count_confidence"] = confidence
            item["quantity"]         = final
            method = "text" if is_text else "vision"
            icon = {"high": "✓", "medium": "⚠", "low": "✗"}[confidence]
            print(f"  {code:<12} {a:>4} {b:>4} {c:>4}  {final} {icon}  [{method}]")

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
