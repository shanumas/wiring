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
import os
import re
import struct
import fitz
from pathlib import Path

_CLIENT = None        # lazy-init so import doesn't fail when key is absent
_CLIENT_HAIKU = None  # cheaper model for counting passes
_CLIENT_QWEN  = None  # httpx.Client for OpenRouter (Qwen pass D)
_QWEN_KEY     = None  # OpenRouter API key (cached alongside client)

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

def _client_qwen():
    """
    Lazy-init a plain httpx.Client for calling OpenRouter directly.
    Returns None (instead of raising) if OPENROUTER_API_KEY is not set,
    so the rest of the pipeline degrades gracefully.
    Using httpx directly avoids openai-SDK version skew with OpenRouter.
    """
    global _CLIENT_QWEN, _QWEN_KEY
    if _CLIENT_QWEN is None:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            return None
        try:
            import ssl
            import httpx
            # Use the OS native trust store (macOS Keychain / Windows cert store)
            # so that system-trusted CA roots are available to httpx.
            # Falls back to certifi if truststore is not installed.
            try:
                import truststore
                _ssl_ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            except ImportError:
                import certifi
                _ssl_ctx = certifi.where()
            _CLIENT_QWEN = httpx.Client(
                verify=_ssl_ctx,
                follow_redirects=True,
                timeout=90.0,
            )
            _QWEN_KEY = key
        except Exception:
            return None
    return _CLIENT_QWEN

SONNET = "claude-sonnet-4-6"
HAIKU  = "claude-haiku-4-5-20251001"

# Qwen model via OpenRouter. Override with $env:QWEN_MODEL if needed.
# Candidates: qwen/qwen3-vl-32b-instruct (default, fast), qwen/qwen3-vl-8b-thinking (accurate, slow)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen/qwen3-vl-32b-instruct")


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

    # ── 1. Locate the FÖRKLARINGAR label and the title block ─────────────────
    # We compute two independent cut-offs:
    #
    # • legend_y_cut   — full-width horizontal cut: anything at y ≥ this value
    #                    is in or below the FÖRKLARINGAR symbol table and must
    #                    be excluded entirely.
    # • titleblk_y_cut — partial-width cut: the drawing stamp lives in the left
    #                    margin (x < TITLEBLK_X_MAX).  Spans are only excluded
    #                    when BOTH y ≥ titleblk_y_cut AND x < TITLEBLK_X_MAX.
    #                    This preserves component labels in the drawing body that
    #                    share the same y-band as the title block strip.
    #
    # Swedish engineering drawings:
    #   title block strip  x ≈ 0 – 250  (RITAD AV, FÖRFRÅGNINGSUNDERLAG, …)
    #   drawing body       x > 600      (floor plan, labels, legend)
    TITLEBLK_X_MAX = 300   # anything left of this column is title-block territory

    _TITLEBLK_RE = re.compile(
        r'RITAD|FÖRFRÅGNINGSUNDERLAG|RELATIONSRITNING|BYGGHANDLING'
        r'|BYGGLOVSRITNING|UTREDNINGSHANDLING|FÖRSLAGSHANDLING'
        r'|HANDLÄGGARE|ANSVARIG',
        re.IGNORECASE,
    )

    legend_y_cut   = None
    titleblk_y_cut = None

    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            for sp in ln["spans"]:
                txt = sp["text"]
                y   = sp["bbox"][1]
                if re.search(r'F.RKLARINGAR|FÖRKLARINGAR|FORKLARINGAR|LEGEND',
                             txt, re.IGNORECASE):
                    if legend_y_cut is None or y < legend_y_cut:
                        legend_y_cut = y - 120   # safe buffer above header row

                if _TITLEBLK_RE.search(txt):
                    if titleblk_y_cut is None or y < titleblk_y_cut:
                        titleblk_y_cut = y

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
                x0, y0 = bbox[0], bbox[1]
                # Exclude text at or past the FÖRKLARINGAR section (full-width cut)
                if legend_y_cut is not None and y0 >= legend_y_cut:
                    continue
                # Exclude title block strip: only left-margin spans at the stamp row
                if (titleblk_y_cut is not None
                        and y0 >= titleblk_y_cut
                        and x0 < TITLEBLK_X_MAX):
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
        # Choose lookbehind based on whether the code starts with a digit.
        #
        # Mixed codes (P11, D1, N1-R): allow a digit immediately before so
        # packed spans like "P11P11N1" are counted correctly.  Only block
        # a preceding letter (which would mean the code is a substring of a
        # longer token).
        #
        # Pure-digit codes (4, 12, …): ALSO block a preceding digit, otherwise
        # "4" would match the trailing digit in "14" or "24".
        if re.match(r'^\d', code):
            lookbehind = r"(?<![A-Za-z\u00C0-\u024F\d])"   # no letter OR digit before
        else:
            lookbehind = r"(?<![A-Za-z\u00C0-\u024F])"     # no letter before (digits OK)

        pattern = lookbehind + re.escape(code) + r"(?!\d)(?!-[A-Z0-9])"
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
        max_tokens=8192,
        messages=[{"role": "user", "content": content}],
    )
    text = resp.content[0].text
    try:
        library = _extract_json(text)
    except ValueError:
        # Response truncated mid-JSON — recover completed component objects
        components = []
        for m in re.finditer(r'\{[^{}]+\}', text, re.DOTALL):
            try:
                obj = json.loads(m.group())
                if "code" in obj and "measurement_type" in obj:
                    components.append(obj)
            except json.JSONDecodeError:
                pass
        if not components:
            raise
        print(f"  [AI] Component library JSON truncated — recovered {len(components)} component(s)")
        library = {"components": components}

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
   IMPORTANT — reading legend rows: In the FÖRKLARINGAR legend, each row has a
   short code label at the left, followed by a graphical symbol, then the
   description text. The description belongs ONLY to that row's code — never
   carry a description across rows. In particular:
   • A "D" appearing inside a description word (e.g. "DALI", "MED DALI") is NOT
     a component code — it is part of the description text. Do NOT confuse it
     with a D-code (D1, D2…) from a different legend row.
   • Codes that are a single letter or Å/Ä/Ö character (e.g. "Å", "A", "RA")
     are valid component codes when they appear as the leftmost label in a legend
     row, next to a graphical symbol. Read them as separate codes with their own
     descriptions — never merge them into an adjacent numbered code (D1, D2…).
   IMPORTANT: Only use codes that actually appear in the component library above
   or that you can clearly read from the drawing labels. Do not invent codes.
   Component codes may be a single digit (e.g. "4" for a 4-gang wall outlet,
   "2" for a 2-gang outlet). If you see a bare digit repeated many times at
   outlet/socket positions in the floor plan, it IS a component code — include it.
   Component codes may also be a single letter or short letter combination
   (e.g. "A" for a spring-return dimmer, "R", "T" for timer variants).
   If the FÖRKLARINGAR legend defines a graphical symbol with a short code label,
   look for that same graphical symbol in the drawing body and count it — even if
   the code is a single letter.
   Do NOT include as codes:
   • Cable specification labels (e.g. "FRHF 3G1,5", "5G2,5", "3×2,5") — these
     describe the cable type on a route, not an installed component.
   • Mounting height annotations ending in ÖFG or ÖFK (e.g. "1000ÖFG", "1900ÖFG").
   • Multi-word annotation phrases (e.g. "VIA NÖDSTOPP", "MED STANDARDCYLINDER").
   • Architectural drawing grid references — these appear ONLY as column/row labels
     in the outermost border/margin of the sheet (far outside the floor plan area).
     Do NOT apply this exclusion to codes inside the drawing body or legend: a
     single letter such as "A", "R", or "T" next to a graphical component symbol
     IS a valid component code, not a grid reference.
   • Swedish description words inside a legend entry text, such as "1-VÄGS",
     "2-VÄGS", "3-POL", "1-fas" — these describe the component type (1-way,
     2-way, 3-pole) and are NOT codes. The code is the label BEFORE the dash-word,
     e.g. in "DM 1-VÄGS UTTAG DISKMASKIN" the code is "DM", not "1".
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


def _count_one_symbol(b64: str, code: str, name: str, hint: int | None = None,
                       exclude_codes: list[dict] | None = None) -> int:
    """
    Count a single symbol in the floor plan, ignoring legend.
    hint: Pass 1 rough count — used as example value so Claude anchors near the right range.
    exclude_codes: other vision items that look similar — Claude must NOT count these here.
    """
    example_val = hint if hint is not None else 0

    excl_block = ""
    if exclude_codes:
        lines = "\n".join(
            f'  "{c["code"]}": {c["name"]}  ← do NOT count this here'
            for c in exclude_codes
        )
        excl_block = f"""
IMPORTANT — similar symbols that must NOT be counted in this call:
{lines}
Count ONLY "{code}" ({name}). Even if the symbols look similar, only count the ones
that match the exact variant above. Each variant will be counted separately in its own call.
"""

    prompt = f"""You are counting electrical/building-services symbols installed in a building floor plan.

CRITICAL — before you count anything:
  Locate the FÖRKLARINGAR / LEGEND / BETECKNINGAR table in the drawing.
  This is a bordered box (often bottom-right or top-right) that lists what each
  symbol looks like. Every symbol shown INSIDE that box is just an example —
  it is NOT installed in the building. Do not count any symbol inside that box.

Count only this symbol, and only where it appears INSIDE rooms, corridors,
or building spaces of the floor plan:
  "{code}": {name}
{excl_block}
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
    return {
        c["code"]: _count_one_symbol_legend(
            b64, c["code"], c["name"],
            exclude_codes=[x for x in codes if x["code"] != c["code"]],
        )
        for c in codes
    }


def _count_one_symbol_legend(b64: str, code: str, name: str,
                              exclude_codes: list[dict] | None = None) -> int:
    """Pass B variant — same single-symbol approach but different framing."""
    excl_block = ""
    if exclude_codes:
        lines = "\n".join(
            f'  "{c["code"]}": {c["name"]}  ← do NOT count this here'
            for c in exclude_codes
        )
        excl_block = f"""
IMPORTANT — similar symbols that must NOT be counted in this call:
{lines}
Count ONLY "{code}" ({name}). Each variant is counted separately in its own call.
"""
    prompt = f"""You are a quantity surveyor counting symbols in a building services floor plan.

The drawing contains a "FÖRKLARINGAR", "LEGEND", or "BETECKNINGAR" box
(a bordered table, often in a corner) that shows what each symbol means.
Symbols inside that box are just examples — they must NOT be counted.

Count how many "{code}" ({name}) symbols appear in the actual floor plan
(rooms, corridors, shafts) — never in the legend, title block, or revision table.
{excl_block}
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
    return {
        c["code"]: _count_one_symbol_legend(
            b64, c["code"], c["name"],
            exclude_codes=[x for x in codes if x["code"] != c["code"]],
        )
        for c in codes
    }


def _count_one_symbol_qwen(b64: str, code: str, name: str) -> int | None:
    """
    Pass D — count one symbol using Qwen VL via OpenRouter.

    Returns None if the Qwen client is unavailable (key not set, API error, …)
    so the caller can skip pass D without breaking the majority vote.

    The prompt is intentionally phrased differently from passes A/B/C so that
    Qwen acts as a truly independent validator rather than echoing Claude's
    framing.
    """
    client = _client_qwen()
    if client is None:
        return None

    prompt = f"""You are analysing a Swedish building services floor plan drawing.

Your task: count how many instances of the component "{code}" ({name}) are
installed in the building floor plan.

Rules:
1. The drawing has a legend box (often labelled FÖRKLARINGAR or BETECKNINGAR)
   showing example symbols. Do NOT count anything inside that legend box.
2. Do NOT count symbols in the title block or any revision/annotation table.
3. Count only symbols that are physically placed inside rooms or corridors.
4. If you are not sure whether a mark is this symbol, do not count it.

Reply with ONLY a JSON object and nothing else:
{{"{code}": <integer count>}}"""

    try:
        payload = {
            "model": QWEN_MODEL,
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        http_resp = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {_QWEN_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        body = http_resp.text.strip()
        if http_resp.status_code != 200 or not body:
            print(f"  [Qwen] pass D HTTP {http_resp.status_code} for {code}, body={body[:300]!r}")
            return None
        data = http_resp.json()
        if "error" in data:
            print(f"  [Qwen] pass D API error for {code}: {data['error']}")
            return None
        text = data["choices"][0]["message"]["content"] or ""
        result = _extract_json(text)
        return int(result.get(code, 0))
    except Exception as exc:
        print(f"  [Qwen] pass D failed for {code}: {exc}")
        return None


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

    _TITLEBLK_RE2 = re.compile(
        r'RITAD|FÖRFRÅGNINGSUNDERLAG|RELATIONSRITNING|BYGGHANDLING'
        r'|BYGGLOVSRITNING|UTREDNINGSHANDLING|FÖRSLAGSHANDLING'
        r'|HANDLÄGGARE|ANSVARIG',
        re.IGNORECASE,
    )
    _TITLEBLK_X_MAX = 300   # left-margin x boundary for title block strips
    legend_y_cut:   float | None = None
    titleblk_y_cut: float | None = None

    for blk in page.get_text("dict")["blocks"]:
        if blk.get("type") != 0:
            continue
        for ln in blk["lines"]:
            for sp in ln["spans"]:
                txt, y = sp["text"], sp["bbox"][1]
                if re.search(r'F.RKLARINGAR|FÖRKLARINGAR', txt, re.IGNORECASE):
                    if legend_y_cut is None or y < legend_y_cut:
                        legend_y_cut = y - 120
                if _TITLEBLK_RE2.search(txt):
                    if titleblk_y_cut is None or y < titleblk_y_cut:
                        titleblk_y_cut = y

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
                x0, y0 = sp["bbox"][0], sp["bbox"][1]
                # Full-width cut at the FÖRKLARINGAR / legend section
                if legend_y_cut is not None and y0 >= legend_y_cut:
                    continue
                # Title block strip: left-margin only (x < _TITLEBLK_X_MAX)
                if (titleblk_y_cut is not None
                        and y0 >= titleblk_y_cut
                        and x0 < _TITLEBLK_X_MAX):
                    continue
                parts_all.append(sp["text"])
                if sp.get("color", 0) <= _DARK_THRESHOLD:
                    parts_dark.append(sp["text"])

    full_text = "|".join(parts_all)
    dark_text = "|".join(parts_dark)
    # doc stays open — Pass 4 still needs page.get_text() for legend scanning

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

    # ── Pass 3 + 4 (merged): short codes from the legend section ────────────
    # Pure-digit codes (e.g. "4" = 4-vägguttag) and pure-letter codes (e.g.
    # "A" = återfjädrande dimmer) both share the same problem: they are too
    # short / generic to discover reliably from the drawing-body text.
    #
    # Frequency counting in the drawing body (the old Pass 3 approach) was
    # too noisy — digits like "1", "2", "11" appeared 60+ times as circuit
    # numbers and were wrongly injected as component codes.
    #
    # The correct source is the FÖRKLARINGAR (legend) section itself:
    # each row has a small, isolated code-label cell (width < 80 pt,
    # height < 40 pt, exactly one token) to the left of the graphical symbol.
    # Scanning ONLY that area gives us real codes with no false positives from
    # circuit/breaker annotations in the drawing body.
    #
    # Discovered codes go to vision counting (A/B/C/D) — they are not
    # text-safe because single-letter / single-digit tokens would cause too
    # many false regex matches in the drawing body text.
    if legend_y_cut is not None:
        _DARK_THR = 0x404040
        # Tokens that look like codes but are always legend annotations
        _LEG_SKIP = frozenset([
            'EI', 'IP', 'UK', 'OK', 'SE', 'EJ', 'MED', 'OCH', 'FÖR', 'AV',
            'VIA', 'HUS', 'DEL', 'BLÅ', 'TYP', 'MM', 'ST', 'ÖFG', 'ÖFK',
        ])
        # Accept: 1–3 uppercase letters, OR 1–2 digits (not pure "0")
        _SHORT_CODE = re.compile(r'^([A-ZÅÄÖ]{1,3}|\d{1,2})$')
        for blk in page.get_text("dict")["blocks"]:
            if blk.get("type") != 0:
                continue
            bb = blk["bbox"]
            if bb[1] < legend_y_cut:          # must be inside the legend area
                continue
            if bb[2] - bb[0] > 80:            # narrow block only (code cell)
                continue

            # Collect all dark tokens in this block
            tokens = []
            for ln in blk["lines"]:
                for sp in ln["spans"]:
                    if sp.get("color", 0) <= _DARK_THR:
                        tokens.extend(sp["text"].strip().split())

            if not tokens:
                continue

            # Case A: small isolated block (height < 40pt) — exactly one token
            if bb[3] - bb[1] <= 40:
                if len(tokens) == 1:
                    tok = tokens[0].strip(".,;:()")
                    if _SHORT_CODE.match(tok) and tok not in _LEG_SKIP and tok != "0":
                        _add(tok, tok)
            # Case B: very narrow block (width < 30pt) with description text —
            # the code cell column in the FÖRKLARINGAR table sometimes wraps the
            # full "CODE description…" into one narrow block (e.g. width=11 pt).
            # Only trust the FIRST token; the rest are the component description.
            elif bb[2] - bb[0] < 30:
                raw_first = tokens[0]
                tok = raw_first.strip(".,;:()")
                # Guard: must be a short code, not a description word like "2-VÄGS"
                # or a fraction like "3N/16A".
                # • Trailing period in the raw token → abbreviation in a sentence
                #   (e.g. "KV." = "kvarter", a Swedish building-address prefix),
                #   not a component code.
                # • Second token is a Swedish connective (EJ, SE, MED …) → the
                #   block is narrative text ("DÄR EJ ANNAT", "HL SE HÄNVISNING"),
                #   not a code-label row.  Only the immediately following word is
                #   checked; words further in the description (FÖR, MED, SE …)
                #   are part of valid component names and must not trigger rejection.
                second = tokens[1].strip(".,;:()") if len(tokens) > 1 else ""
                has_connective = second in _LEG_SKIP
                if (_SHORT_CODE.match(tok)
                        and tok not in _LEG_SKIP
                        and tok != "0"
                        and "/" not in tok
                        and not raw_first.endswith(".")
                        and not has_connective):
                    _add(tok, tok)

    doc.close()
    return count_items + extras


def _is_text_countable(code: str) -> bool:
    """
    Return True when the symbol code is safe to count via PDF text extraction.

    A code is text-safe when it meets one of:
    • Pure digit(s), 1–2 chars: "4", "12" — component code like 4-vägguttag.
      The digit-boundary guards in the regex prevent "4" from
      matching inside "400", "2700" etc.
    • At least 2 chars long AND contains at least one digit: "P11", "D1", "V17".

    Single letters without digits ("Å", "A", "OT") are NOT text-safe because
    they appear too often in Swedish prose and produce false matches.

    Examples:
      "4"    → True   (pure digit — 4-vägguttag)
      "P11"  → True   (letter + digits)
      "D1"   → True
      "V17"  → True
      "F1"   → True
      "Å"    → False  (single letter, no digit — use vision)
      "A"    → False
      "OT"   → False  (no digit)
    """
    if re.match(r'^\d{1,2}$', code):
        return True   # pure-digit code like "4"
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

    # Separate count vs length components.
    # Strip anything that is clearly not a component code: codes containing
    # spaces (annotation phrases like "VIA NÖDSTOPP"), height-only tokens
    # (ends with ÖFG/ÖFK), and cable spec labels (contains "G" between digits,
    # e.g. "3G1,5").
    _NOT_CODE = re.compile(
        r'\s'                        # any whitespace → phrase, not a code
        r'|ÖFG$|ÖFK$'               # mounting height annotations
        r'|\d+G\d'                   # cable spec: 3G1,5 / 5G2,5
        r'|\d{3,}'                   # 3+ consecutive digits (room labels A112)
        r'|\.\w+\.'                  # dot-separated tokens: TEXT.SLO.10
    )
    raw_comps = [c for c in raw.get("components", [])
                 if not _NOT_CODE.search(str(c.get("code", "")).strip())]

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

    # ── Build full PDF text for phantom-hallucination guard ───────────────────
    # All spans from all zones (no y-cut filtering) joined into one string.
    # Used later to detect vision-counted items whose code appears NOWHERE in
    # the PDF text — a strong signal of Pass-1 hallucination.
    _ph_doc  = fitz.open(drawing_pdf_path)
    _ph_page = _ph_doc[0]
    _full_pdf_text = " ".join(
        sp["text"]
        for blk in _ph_page.get_text("dict")["blocks"]
        if blk.get("type") == 0
        for ln in blk["lines"]
        for sp in ln["spans"]
    )
    _ph_doc.close()

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

        # ── TEXT items: exact PDF text extraction + one vision sanity check ──────
        # Text extraction is deterministic and exact (zero tokens), so we use it
        # as the authoritative count for codes that are safe to count from text.
        #
        # We also run ONE independent vision pass per text item as a sanity check.
        # If the vision result differs from the text count by more than 10% the
        # confidence is flagged — this surfaces bugs like the title-block y-cut
        # regression where text returns a wrong number with false "high confidence".
        #
        # Text items store:  count_method="text", count_text=N, count_vision=M
        # Vision items store: count_method="vision", count_grid_a/b/c/d

        if text_items:
            text_counts = _count_from_pdf_text(drawing_pdf_path, text_items, legend_bbox)
            print(f"  Text counts (exact): { {c['code']: text_counts.get(c['code'], 0) for c in text_items} }")
        else:
            text_counts = {}

        # One vision sanity-check pass for each text item
        vision_check: dict[str, int] = {}
        if text_items:
            print(f"  Vision sanity-check for {len(text_items)} text item(s):")
            for c in text_items:
                code_t, name_t = c["code"], c["name"]
                vc = _count_one_symbol(full_b64, code_t, name_t)
                vision_check[code_t] = vc
                tc = text_counts.get(code_t, 0)
                delta = abs(tc - vc)
                ratio = delta / tc if tc > 0 else (1.0 if vc > 0 else 0.0)
                flag = " ⚠ MISMATCH" if ratio > 0.10 else ""
                print(f"    {code_t}: text={tc}  vision={vc}{flag}")

        # ── VISION items: 3 Claude passes + 1 Qwen pass ───────────────────────
        qwen_available = _client_qwen() is not None
        if qwen_available:
            print(f"  Qwen pass D enabled (model: {QWEN_MODEL})")
        else:
            print(f"  Qwen pass D disabled — set OPENROUTER_API_KEY to enable")

        vision_a: dict[str, int] = {}
        vision_b: dict[str, int] = {}
        vision_c: dict[str, int] = {}
        vision_d: dict[str, int | None] = {}
        if vision_items:
            print(f"  Vision counting for {len(vision_items)} graphical symbol(s): "
                  f"{[c['code'] for c in vision_items]}")
            for c in vision_items:
                code_v, name_v = c["code"], c["name"]
                # Build sibling exclusion list: all other vision items
                # so Claude counts each variant exclusively.
                excl = [x for x in vision_items if x["code"] != code_v]
                va = _count_one_symbol(full_b64, code_v, name_v, exclude_codes=excl)
                vb = _count_one_symbol(full_b64, code_v, name_v, exclude_codes=excl)
                vc = _count_one_symbol(full_b64, code_v, name_v, exclude_codes=excl)
                vd = _count_one_symbol_qwen(full_b64, code_v, name_v)
                vision_a[code_v] = va
                vision_b[code_v] = vb
                vision_c[code_v] = vc
                vision_d[code_v] = vd
                d_str = str(vd) if vd is not None else "—"
                print(f"    {code_v}: A={va} B={vb} C={vc} D={d_str}")

        # ── Final count + confidence per item ─────────────────────────────────
        print(f"\n  {'Code':<12} {'result':>8}  confidence  method")
        for item in count_items:
            code    = item["code"]
            is_text = _is_text_countable(code)

            if is_text:
                # Authoritative = text count; vision is a sanity check only
                tc    = text_counts.get(code, 0)
                vc    = vision_check.get(code, tc)
                delta = abs(tc - vc)
                ratio = delta / tc if tc > 0 else (1.0 if vc > 0 else 0.0)

                if ratio <= 0.10:
                    confidence = "high"
                elif ratio <= 0.25:
                    confidence = "medium"
                else:
                    confidence = "low"

                final = tc          # text count is authoritative

                item["count_method"]  = "text"
                item["count_text"]    = tc
                item["count_vision"]  = vc
                # Keep grid slots null — they are NOT independent validators for text items
                item["count_grid_a"]  = None
                item["count_grid_b"]  = None
                item["count_grid_c"]  = None
                item["count_grid_d"]  = None

            else:
                # Vision items: majority vote across A/B/C (+ D if available)
                a = vision_a.get(code, 0)
                b = vision_b.get(code, 0)
                c = vision_c.get(code, 0)
                d = vision_d.get(code)
                votes = [a, b, c] + ([d] if d is not None else [])
                freq: dict[int, int] = {}
                for v in votes:
                    freq[v] = freq.get(v, 0) + 1
                best_val   = max(freq, key=lambda v: (freq[v], -v))
                best_count = freq[best_val]
                n_votes    = len(votes)

                if best_count == n_votes:
                    confidence = "high"
                elif best_count >= n_votes - 1:
                    confidence = "medium"
                else:
                    confidence = "low"

                final = best_val

                # ── Phantom-hallucination guard ──────────────────────────────
                # If all vision passes returned a non-zero count but the code
                # string appears NOWHERE in the full PDF text (body + legend +
                # title block), the component is very likely a hallucination.
                # Single-letter codes (A, B, Å …) are excluded from this check
                # because they appear constantly in Swedish prose.
                # Only applied to codes ≥ 2 chars with a non-zero final count.
                if final > 0 and len(code) >= 2:
                    # Word-boundary search: code must appear as a standalone
                    # token (not buried inside a longer word).
                    _pat = r'(?<![A-Za-z\u00C0-\u024F])' + re.escape(code) + r'(?![A-Za-z\u00C0-\u024F\d])'
                    if not re.search(_pat, _full_pdf_text):
                        print(f"  ⚠ PHANTOM? {code}: vision={final} but code appears 0 times in PDF text → confidence forced to low")
                        confidence = "low"
                        final = 0   # zero out — no text evidence this component exists

                item["count_method"]  = "vision"
                item["count_text"]    = None
                item["count_vision"]  = None
                item["count_grid_a"]  = a
                item["count_grid_b"]  = b
                item["count_grid_c"]  = c
                item["count_grid_d"]  = d

            item["count_confidence"] = confidence
            item["quantity"]         = final
            icon = {"high": "✓", "medium": "⚠", "low": "✗"}[confidence]
            print(f"  {code:<12} {final:>8}  {confidence:<10}  {item['count_method']}  {icon}")

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


# ── Image-based extraction (pre-split legend + body PNGs) ────────────────────

def _images_cache_key(legend_path: str, body_path: str) -> str:
    h = hashlib.sha1()
    h.update(Path(legend_path).read_bytes())
    h.update(Path(body_path).read_bytes())
    return h.hexdigest()[:12]


def _png_wh(path: str) -> tuple[int, int]:
    """Read width, height from PNG file header (no extra deps)."""
    with open(path, 'rb') as f:
        f.read(16)  # PNG sig (8) + IHDR chunk length (4) + "IHDR" (4)
        w = struct.unpack('>I', f.read(4))[0]
        h = struct.unpack('>I', f.read(4))[0]
    return w, h


def _image_file_to_b64(image_path: str) -> str:
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode()


def _tile_png_b64(body_path: str, row: int, col: int,
                  rows: int = 3, cols: int = 3,
                  offset_x: float = 0.0, offset_y: float = 0.0,
                  max_px: int = 2400) -> str:
    """
    Extract one tile from a PNG as base64, zoomed to max_px on the long side.
    offset_x / offset_y shift the grid by a fraction of tile size (0.0–1.0)
    so consecutive passes hit different cut lines and catch boundary symbols.
    """
    doc  = fitz.open(body_path)
    page = doc[0]
    w, h = page.rect.width, page.rect.height
    tw, th = w / cols, h / rows
    x0 = min((col + offset_x) * tw, w - tw)
    y0 = min((row + offset_y) * th, h - th)
    clip  = fitz.Rect(x0, y0, x0 + tw, y0 + th)
    scale = min(max_px / max(clip.width, clip.height), 3.0)
    pix   = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
    data  = base64.standard_b64encode(pix.tobytes("png")).decode()
    doc.close()
    return data


def _count_all_variants_in_tile(tile_b64: str, symbols: list[dict]) -> dict[str, int]:
    """
    Count ALL visual variants in one tile with a single API call.
    Only counts symbols fully visible (not cut off at tile edges).
    Returns {visual_id: count}.
    """
    syms_desc = "\n".join(
        f'  "{s["visual_id"]}" ({s["code"]}): {s["name"]} — {s.get("description", "")}'
        for s in symbols
    )
    prompt = f"""Count building-services symbols in this floor plan tile.
Only count symbols that are FULLY visible — do NOT count any symbol cut off at the image edge.

Symbols to count:
{syms_desc}

YOUR ENTIRE RESPONSE MUST BE ONLY A RAW JSON OBJECT — no explanation, no markdown.
The very first character must be {{ and the last must be }}.
Example: {{"{symbols[0]['visual_id']}": 2}}"""

    resp = _client().messages.create(
        model=SONNET, max_tokens=256,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": tile_b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    try:
        result = _extract_json(resp.content[0].text)
        return {s["visual_id"]: int(result.get(s["visual_id"], 0)) for s in symbols}
    except Exception:
        return {s["visual_id"]: 0 for s in symbols}


def _analyse_legend_image(legend_b64: str, component_library: dict | None) -> list[dict]:
    """
    Send legend.png to Claude to identify all visual symbol variants.

    Returns list of dicts: {visual_id, code, name, description, measurement_type}
    visual_id is unique — if the same code label appears with two different graphics
    (e.g. "A" with 1 leg vs 2 legs) they get separate entries: "A_v1", "A_v2".
    """
    lib_section = ""
    if component_library:
        lib_section = (
            "\nKnown components from the project description:\n"
            + json.dumps(component_library.get("components", []), ensure_ascii=False, indent=2)
            + "\n"
        )

    prompt = f"""You are reading the FÖRKLARINGAR (legend) section of a Swedish building services drawing.
{lib_section}
List EVERY symbol shown in this legend.

CRITICAL: If the same code label (e.g. "A") appears with TWO VISUALLY DIFFERENT graphic symbols
(e.g. one with 2 legs and one with 1 leg), list them as SEPARATE entries with distinct visual_id
values: "A_v1" and "A_v2". Describe what makes each visually distinct.

For each symbol return:
  visual_id        — unique identifier: use the code itself if it appears once (e.g. "KS"),
                     or add "_v1", "_v2" suffix for multiple visual variants of the same code
  code             — the text label exactly as printed (e.g. "A", "KS", "P11")
  name             — full description from the legend text
  description      — 1-2 sentences describing the graphic shape: number of legs/lines,
                     circle/square/triangle, filled/outline, size relative to others, etc.
  measurement_type — "count" for point-installed items (fixtures, sockets, sensors …)
                     "length" for line-installed runs (trays, conduits, pipes …)

YOUR RESPONSE MUST BE A RAW JSON OBJECT WITH NO TEXT BEFORE OR AFTER IT.
No markdown fences, no explanation, no description of what you see. The very first character
must be {{ and the last must be }}.

{{
  "symbols": [
    {{"visual_id": "A_v1", "code": "A", "name": "Armatur 2-rör",
      "description": "Circle with two downward lines (legs)", "measurement_type": "count"}},
    {{"visual_id": "A_v2", "code": "A", "name": "Armatur 1-rör",
      "description": "Circle with one downward line (leg)", "measurement_type": "count"}},
    {{"visual_id": "KS", "code": "KS", "name": "Kabelstege",
      "description": "Two parallel horizontal lines with cross-bars", "measurement_type": "length"}}
  ]
}}"""

    resp = _client().messages.create(
        model=SONNET,
        max_tokens=4096,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": legend_b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    text = resp.content[0].text
    try:
        result = _extract_json(text)
        return result.get("symbols", [])
    except ValueError:
        # Response was truncated mid-JSON — recover completed symbol objects
        symbols = []
        for m in re.finditer(r'\{[^{}]+\}', text, re.DOTALL):
            try:
                obj = json.loads(m.group())
                if "visual_id" in obj and "code" in obj:
                    symbols.append(obj)
            except json.JSONDecodeError:
                pass
        if symbols:
            print(f"  [AI] Legend JSON truncated — recovered {len(symbols)} symbol(s) from partial response")
        return symbols


def _count_variant_body(body_b64: str, visual_id: str, code: str, name: str,
                        description: str, exclude_variants: list[dict]) -> int:
    """Count one visual variant in the body image (legend already removed)."""
    excl_block = ""
    if exclude_variants:
        lines = "\n".join(
            f'  "{v["visual_id"]}" ({v["name"]}): {v.get("description", "")}  ← do NOT count this'
            for v in exclude_variants
        )
        excl_block = (
            f'\nIMPORTANT — these other variants share the "{code}" label but look different. '
            f'Do NOT count them here:\n{lines}\n'
            f'Count ONLY the variant described above.\n'
        )

    prompt = f"""You are counting electrical/building-services symbols in a building floor plan.
The legend has been removed from this image — everything visible is the actual floor plan.

Count how many instances of this specific symbol appear:
  Code: "{code}"
  Name: {name}
  Visual appearance: {description}
{excl_block}
Rules:
- Count every instance that matches the visual description above.
- Do not double-count (each physical location = 1 instance).
- If uncertain whether a mark matches → do NOT count it.

YOUR ENTIRE RESPONSE MUST BE ONLY A RAW JSON OBJECT:
{{"{visual_id}": <integer count>}}"""

    resp = _client().messages.create(
        model=SONNET,
        max_tokens=256,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": body_b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    text = resp.content[0].text
    try:
        result = _extract_json(text)
        return int(result.get(visual_id, 0))
    except (ValueError, KeyError):
        # Claude returned prose — try to pull a number from patterns like "D1" label - that's 1 instance
        m = re.search(r'\b(\d+)\s+instance', text) or re.search(r'total[^\d]*(\d+)', text, re.I)
        if m:
            return int(m.group(1))
        return 0


def _count_variant_qwen(body_b64: str, visual_id: str, code: str, name: str,
                        description: str) -> int | None:
    """Qwen pass D — count one visual variant in the body image."""
    client = _client_qwen()
    if client is None:
        return None

    prompt = f"""You are counting symbols in a Swedish building services floor plan.
The legend box has already been removed from this image — count only real installed items.

Count instances of this specific symbol:
  Code: "{code}"
  Name: {name}
  Visual appearance: {description}

Reply ONLY with a JSON object, nothing else:
{{"{visual_id}": <integer>}}"""

    try:
        payload = {
            "model": QWEN_MODEL,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{body_b64}"}},
                {"type": "text", "text": prompt},
            ]}],
        }
        http_resp = _CLIENT_QWEN.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {_QWEN_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
        body_text = http_resp.text.strip()
        if http_resp.status_code != 200 or not body_text:
            print(f"  [Qwen] HTTP {http_resp.status_code} for {visual_id}")
            return None
        data = http_resp.json()
        if "error" in data:
            print(f"  [Qwen] error for {visual_id}: {data['error']}")
            return None
        text = data["choices"][0]["message"]["content"] or ""
        result = _extract_json(text)
        return int(result.get(visual_id, 0))
    except Exception as exc:
        print(f"  [Qwen] failed for {visual_id}: {exc}")
        return None


def _estimate_lengths_from_body(body_b64: str, length_syms: list[dict]) -> dict[str, float]:
    """Ask Claude to estimate total run lengths for line-installed items from the body image."""
    if not length_syms:
        return {}
    syms_desc = "\n".join(
        f'  "{s["visual_id"]}" ({s["code"]}): {s["name"]} — {s.get("description", "")}'
        for s in length_syms
    )
    prompt = f"""You are estimating total run lengths of line-installed items in a building floor plan.
The legend has been removed from this image.

For each item below, estimate the TOTAL installed run length in metres.
Use the drawing scale printed in the title block (e.g. "SKALA 1:50").

Items to measure:
{syms_desc}

Return ONLY a JSON object mapping visual_id to total metres (float):
{{{{"KS": 45.5, "KR": 12.0}}}}"""

    resp = _client().messages.create(
        model=SONNET,
        max_tokens=512,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": body_b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    try:
        result = _extract_json(resp.content[0].text)
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def load_images_cache(legend_path: str, body_path: str) -> dict | None:
    key = _images_cache_key(legend_path, body_path)
    p = AI_CACHE_DIR / f"images_{key}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def extract_with_images(legend_path: str, body_path: str,
                        component_library: dict | None = None,
                        progress_cb=None) -> dict:
    """
    Image-based extraction pipeline using pre-split legend and body PNGs.

    Step 1 — legend.png → Claude identifies all visual symbol variants.
              Two "A" symbols with different shapes become "A_v1" and "A_v2".
    Step 2 — body.png → count each variant (3 Claude passes + optional Qwen).
    Step 3 — body.png → estimate lengths for line-installed items.
    """
    def _emit(event):
        if progress_cb:
            try:
                progress_cb(event)
            except Exception:
                pass

    cache_key  = _images_cache_key(legend_path, body_path)
    cache_file = AI_CACHE_DIR / f"images_{cache_key}.json"
    if cache_file.exists():
        print(f"  [AI] loaded from image cache ({cache_key})")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    page_w, page_h = _png_wh(body_path)
    legend_b64 = _image_file_to_b64(legend_path)
    body_b64   = _image_file_to_b64(body_path)

    # ── Step 1: legend analysis ──────────────────────────────────────────────
    _emit({"type": "phase", "phase": "legend", "msg": "Analysing legend…"})
    print("  [AI] Analysing legend image …")
    try:
        symbols = _analyse_legend_image(legend_b64, component_library)
    except Exception as exc:
        print(f"  [AI] Legend analysis failed: {exc}")
        symbols = []
    print(f"  [AI] {len(symbols)} symbol variant(s) found:")
    for s in symbols:
        print(f"    {s['visual_id']!r:16} code={s['code']!r}  [{s['measurement_type']}]  {s['name']}")

    count_syms  = [s for s in symbols if s.get("measurement_type") == "count"]
    length_syms = [s for s in symbols if s.get("measurement_type") != "count"]

    # ── Step 2: count each visual variant ────────────────────────────────────
    qwen_ok = _client_qwen() is not None
    print(f"  [AI] Qwen pass D {'enabled' if qwen_ok else 'disabled'}")

    counts: dict[str, dict] = {}

    # ── Passes A / B / C: independent 3×3 tiled counts ──────────────────────
    # Same straight grid (no offsets), three independent API call sets.
    # Model stochasticity means they can diverge — the voting rule below
    # exploits agreement to pick the right answer.
    ROWS, COLS = 3, 3
    tile_passes: list[dict[str, int]] = []
    for pass_label in ["A", "B", "C"]:
        _emit({"type": "phase", "phase": f"pass_{pass_label}",
               "msg": f"Tiling pass {pass_label} ({ROWS}×{COLS})…"})
        pass_sum = {s["visual_id"]: 0 for s in count_syms}
        print(f"\n  Pass {pass_label} — {ROWS}×{COLS} tiled:")
        for r in range(ROWS):
            for c in range(COLS):
                try:
                    tile_b64 = _tile_png_b64(body_path, r, c, ROWS, COLS)
                    tile_cnt = _count_all_variants_in_tile(tile_b64, count_syms)
                    for vid, cnt in tile_cnt.items():
                        pass_sum[vid] += cnt
                except Exception as e:
                    print(f"    tile({r},{c}) failed: {e}")
        tile_passes.append(pass_sum)
        print(f"    { {k: v for k, v in pass_sum.items() if v > 0} }")

    # ── Voting rule ──────────────────────────────────────────────────────────
    #   1. ≥3 of [A,B,C,D] agree        → that value  (high)
    #   2. A or C agrees with D          → D           (high)
    #   3. A == C                        → A           (medium)
    #   4. All other cases               → D           (low)
    def _vote(a: int, b: int, c: int, d: int | None) -> tuple[int, str]:
        vals = [a, b, c] + ([d] if d is not None else [])
        for v in set(vals):
            if vals.count(v) >= 3:
                return v, "high"
        if d is not None and (a == d or c == d):
            return d, "high"
        if a == c:
            return a, "medium"
        if d is not None:
            return d, "low"
        return a, "low"   # D unavailable fallback

    # A symbol is a true visual variant only when multiple legend entries share
    # the same code (e.g. two different "A" symbols). A unique code like "C" that
    # Claude named "circle_C" is NOT a variant even though visual_id != code.
    from collections import Counter as _Counter
    _code_freq = _Counter(s["code"] for s in count_syms)

    # ── Pass D: Qwen full-image count + per-symbol vote (streamed) ───────────
    if qwen_ok:
        _emit({"type": "phase", "phase": "qwen",
               "msg": f"Qwen counting {len(count_syms)} symbol(s)…"})
        print(f"\n  Pass D — Qwen full-image:")
    else:
        _emit({"type": "phase", "phase": "voting", "msg": "Voting…"})

    print(f"\n  {'Code':<20} {'A':>4} {'B':>4} {'C':>4} {'D':>4}  result  conf")
    for sym in count_syms:
        vid        = sym["visual_id"]
        is_variant = _code_freq[sym["code"]] > 1

        a = tile_passes[0].get(vid, 0)
        b = tile_passes[1].get(vid, 0)
        c = tile_passes[2].get(vid, 0)

        if qwen_ok:
            d = _count_variant_qwen(body_b64, vid, sym["code"],
                                    sym["name"], sym.get("description", sym["name"]))
            if d is not None:
                print(f"    {vid}: {d}")
        else:
            d = None

        if is_variant:
            best = d if d is not None else 0
            conf = "high" if d is not None else "low"
        else:
            best, conf = _vote(a, b, c, d)

        d_str  = str(d) if d is not None else "—"
        marker = " [visual-only]" if is_variant else ""
        icon   = {"high": "✓", "medium": "⚠", "low": "✗"}[conf]
        print(f"    {vid:<18} A={a} B={b} C={c} D={d_str}  → {best} ({conf}) {icon}{marker}")
        counts[vid] = {"a": a if not is_variant else None,
                       "b": b if not is_variant else None,
                       "c": c if not is_variant else None,
                       "d": d, "final": best, "confidence": conf}
        _emit({
            "type":       "symbol",
            "visual_id":  vid,
            "code":       sym["code"],
            "name":       sym.get("name", sym["code"]),
            "count":      best,
            "confidence": conf,
            "a":          a if not is_variant else None,
            "b":          b if not is_variant else None,
            "c":          c if not is_variant else None,
            "d":          d,
        })

    # ── Step 3: estimate lengths ─────────────────────────────────────────────
    lengths: dict[str, float] = {}
    if length_syms:
        _emit({"type": "phase", "phase": "lengths",
               "msg": f"Estimating lengths for {len(length_syms)} item(s)…"})
        print(f"\n  Estimating lengths for {len(length_syms)} item(s) …")
        try:
            lengths = _estimate_lengths_from_body(body_b64, length_syms)
            for vid, m in lengths.items():
                print(f"    {vid}: {m:.1f} m")
        except Exception as exc:
            print(f"  [AI] Length estimation failed: {exc}")
            lengths = {}

    # ── Assemble standard schema ─────────────────────────────────────────────
    components: list[dict] = []
    summary:    list[dict] = []

    for i, sym in enumerate(symbols):
        vid   = sym["visual_id"]
        code  = sym["code"]
        name  = sym["name"]
        mtype = sym.get("measurement_type", "count")
        color = _color(vid)

        if mtype == "count":
            ct   = counts.get(vid, {})
            qty  = ct.get("final", 0)
            conf = ct.get("confidence", "unknown")
            components.append({
                "id": f"IMG_{vid}_{i}", "type": vid, "name": name,
                "en_name": "", "color": color, "size": None,
                "ok_height": None, "uk_height": None, "is_vertical": False,
                "fire_rating": None, "label": f"{code} — {qty} pcs",
                "bbox": None, "occurrences": 1,
                "measurement_type": "count", "quantity": qty, "unit": "pcs",
                "original_code": code,
            })
            summary.append({
                "system": vid, "name": name, "orientation": "horizontal",
                "width_mm": None, "ok_ofg_mm": None, "uk_ofg_mm": None,
                "fire_rating": None, "measurement_type": "count", "unit": "pcs",
                "count": qty, "count_method": "vision", "count_confidence": conf,
                "count_grid_a": ct.get("a"), "count_grid_b": ct.get("b"),
                "count_grid_c": ct.get("c"), "count_grid_d": ct.get("d"),
                "count_text": None, "count_vision": None,
                "original_code": code,
            })
        else:
            qty = lengths.get(vid, 0.0)
            components.append({
                "id": f"IMG_{vid}_{i}", "type": vid, "name": name,
                "en_name": "", "color": color, "size": None,
                "ok_height": None, "uk_height": None, "is_vertical": False,
                "fire_rating": None, "label": f"{code} — {qty:.1f} m",
                "bbox": None, "occurrences": 1,
                "measurement_type": "length", "quantity": qty, "unit": "m",
                "original_code": code, "length_m": qty,
            })
            summary.append({
                "system": vid, "name": name, "orientation": "horizontal",
                "width_mm": None, "ok_ofg_mm": None, "uk_ofg_mm": None,
                "fire_rating": None, "measurement_type": "length", "unit": "m",
                "count": 1, "total_length_m": qty,
                "original_code": code,
            })

    result = {
        "page_width":      page_w,
        "page_height":     page_h,
        "drawing_type":    "image-based",
        "scale":           "unknown",
        "components":      components,
        "summary":         summary,
        "extraction_mode": "ai_images",
    }
    cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


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
            # Propagate validation metadata so the UI can display it correctly.
            # count_method distinguishes two very different confidence models:
            #
            #   "text"   — authoritative PDF text extraction + one vision sanity check.
            #              count_text   = exact text count (final answer)
            #              count_vision = independent vision cross-check
            #              confidence   = based on text/vision agreement
            #              count_grid_a/b/c/d are NOT independent validators → stored as null.
            #
            #   "vision" — majority vote of 3 Claude passes + optional Qwen pass D.
            #              count_grid_a/b/c/d = independent visual counts
            #              confidence   = based on vote agreement
            #              count_text/count_vision are null.
            method = item.get("count_method", "unknown")
            s["count_method"]    = method
            s["count_confidence"] = item.get("count_confidence", "unknown")
            if method == "text":
                s["count_text"]   = item.get("count_text")
                s["count_vision"] = item.get("count_vision")
                s["count_grid_a"] = None
                s["count_grid_b"] = None
                s["count_grid_c"] = None
                s["count_grid_d"] = None
            else:
                s["count_text"]   = None
                s["count_vision"] = None
                s["count_grid_a"] = item.get("count_grid_a")
                s["count_grid_b"] = item.get("count_grid_b")
                s["count_grid_c"] = item.get("count_grid_c")
                s["count_grid_d"] = item.get("count_grid_d")
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
