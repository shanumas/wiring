"""
FastAPI server — multi-drawing support with live upload.

Endpoints
---------
GET  /                            → index.html
GET  /drawings                    → list of loaded drawing names
POST /upload                      → upload one or more PDFs; returns list of names
GET  /drawing/{name}/image        → rendered PNG (1.5×)
GET  /drawing/{name}/components   → extract() result as JSON
GET  /drawing/{name}/estimate     → estimate() result as JSON
"""
import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
import fitz
from extract import extract, _to_display_bbox
from extract_vector import extract_vectors
from estimate import estimate, COST_TABLE, COST_FALLBACK_PER_M
from extract_ai import (build_component_library, load_component_library,
                        extract_with_ai, load_ai_cache, _ai_cache_path,
                        extract_with_images, load_images_cache, _images_cache_key)


# ── Startup API key checks ────────────────────────────────────────────────────
def _check_keys() -> None:
    """
    Verify all required API keys are present and reachable before the server
    starts accepting requests.  Exits with a clear error message if any check
    fails so the operator knows exactly what to fix.
    """
    errors: list[str] = []
    skip_anthropic = os.environ.get("SKIP_ANTHROPIC", "").lower() in ("1", "true", "yes")

    # ── 1. Anthropic ─────────────────────────────────────────────────────────
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if skip_anthropic:
        print("  ⚠ ANTHROPIC_API_KEY check skipped (SKIP_ANTHROPIC=1) — AI passes disabled.")
    elif not anthropic_key:
        errors.append("ANTHROPIC_API_KEY is not set.")
    else:
        try:
            import anthropic as _ant
            resp = _ant.Anthropic(api_key=anthropic_key).messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=8,
                messages=[{"role": "user", "content": "Reply with the word OK only."}],
            )
            reply = resp.content[0].text.strip()
            print(f"  ✓ ANTHROPIC_API_KEY — model replied: {reply!r}")
        except _ant.AuthenticationError:
            errors.append("ANTHROPIC_API_KEY is set but rejected by Anthropic (wrong key?).")
        except Exception as exc:
            errors.append(f"ANTHROPIC_API_KEY check failed: {exc}")

    # ── 2. OpenRouter / Vision pass D (required if key is set) ──────────────
    # We do a real model call (tiny 1×1 PNG) so startup fails fast if the
    # model is unavailable, quota-exceeded, or misconfigured.
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    vision_model = (os.environ.get("VISION_MODEL")
                    or os.environ.get("QWEN_MODEL")
                    or "google/gemini-3.5-flash")
    if not or_key:
        print("  ⚠ OPENROUTER_API_KEY not set — Vision pass D will be disabled.")
    else:
        try:
            import ssl as _ssl
            import json as _json
            import httpx as _httpx
            try:
                import truststore as _ts
                _ssl_ctx = _ts.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
            except ImportError:
                import certifi as _certifi
                _ssl_ctx = _certifi.where()

            _http = _httpx.Client(verify=_ssl_ctx, follow_redirects=True, timeout=60.0)

            _probe = _http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"},
                json={
                    "model": vision_model,
                    "max_tokens": 5,
                    "messages": [{
                        "role": "user",
                        "content": "Reply with the single word OK.",
                    }],
                },
            )
            _body = _probe.text.strip()
            if _probe.status_code != 200:
                errors.append(
                    f"Vision model probe failed: HTTP {_probe.status_code}. Body: {_body[:300]!r}"
                )
            elif not _body:
                errors.append(
                    f"Vision model probe failed: HTTP 200 but empty response body "
                    f"(model={vision_model!r}). Check OpenRouter credits / model availability."
                )
            else:
                _data = _json.loads(_body)
                if "error" in _data:
                    errors.append(f"Vision model probe error: {_data['error']}")
                else:
                    _reply = _data["choices"][0]["message"]["content"]
                    print(f"  ✓ Vision pass D — model {vision_model!r} responded: {_reply!r}")
        except _httpx.ConnectError as exc:
            errors.append(f"Vision model probe: cannot reach openrouter.ai — {exc}")
        except Exception as exc:
            errors.append(f"Vision model probe failed: {exc}")

    if errors:
        print("\n" + "─" * 60)
        print("STARTUP FAILED — API key problems:\n")
        for e in errors:
            print(f"  ✗ {e}")
        print("─" * 60 + "\n")
        sys.exit(1)


print("Checking API keys …")
_check_keys()

app    = FastAPI()
PDF_DIR = Path("pdf")
PDF_DIR.mkdir(exist_ok=True)

# ── Per-drawing in-memory cache ────────────────────────────────────────────────
# { filename: {"png": bytes, "components": dict, "estimate": dict, ...} }
_cache: dict = {}

# ── Component library (built from description.pdf) ─────────────────────────────
_component_library: dict | None = None

# ── Tracks which drawings are currently being AI-extracted ─────────────────────
_ai_running: set[str] = set()


def _estimate_vector(vec_data: dict) -> dict:
    """Produce an estimate from vector-extracted data (lengths already measured)."""
    items = []
    total_cost = 0
    total_len  = 0.0

    for s in vec_data["summary"]:
        if not isinstance(s.get("count"), int):
            continue
        w  = s.get("width_mm")
        fr = s.get("fire_rating")
        key = (s["system"], w if isinstance(w, int) else None, fr)
        unit_cost = COST_TABLE.get(key, COST_FALLBACK_PER_M)
        length_m  = s.get("total_length_m", 0.0)
        cost      = round(unit_cost * length_m)
        total_cost += cost
        total_len  += length_m
        items.append({
            **s,
            "total_length_m":       round(length_m, 2),
            "total_length_mm":      round(length_m * 1000),
            "horizontal_length_mm": round(length_m * 1000),
            "vertical_length_mm":   0,
            "length_source":        "vector_paths",
            "unit_cost_sek_per_m":  unit_cost,
            "total_cost_sek":       cost,
            "confidence":           0.85,
            "confidence_notes":     ["length measured from drawing geometry"],
        })

    return {
        "items":  items,
        "totals": {
            "total_cost_sek": total_cost,
            "total_length_m": round(total_len, 2),
        },
    }


def _estimate_ai(ai_comp: dict) -> dict:
    """Build an estimate from AI-extracted components (count and length types)."""
    items = []
    total_cost = 0
    total_len  = 0.0

    for s in ai_comp.get("summary", []):
        mtype = s.get("measurement_type", "count")
        w     = s.get("width_mm")
        fr    = s.get("fire_rating")
        system = s.get("system", "?")

        if mtype == "length":
            length_m  = s.get("total_length_m", 0.0)
            key       = (system, w if isinstance(w, int) else None, fr)
            unit_cost = COST_TABLE.get(key, COST_FALLBACK_PER_M)
            cost      = round(unit_cost * length_m)
            total_cost += cost
            total_len  += length_m
            items.append({
                **s,
                "total_length_m":       round(length_m, 2),
                "total_length_mm":      round(length_m * 1000),
                "horizontal_length_mm": round(length_m * 1000),
                "vertical_length_mm":   0,
                "length_source":        "ai_vision",
                "unit_cost_sek_per_m":  unit_cost,
                "total_cost_sek":       cost,
                "confidence":           0.70,
                "confidence_notes":     ["quantity estimated by AI vision"],
            })
        else:
            # count-based: no cost estimation yet, just pass through
            items.append({
                **s,
                "length_source":    "ai_vision",
                "unit_cost_sek_per_m": None,
                "total_cost_sek":   None,
                "confidence":       0.70,
                "confidence_notes": ["count estimated by AI vision"],
            })

    return {
        "items":  items,
        "totals": {
            "total_cost_sek": total_cost,
            "total_length_m": round(total_len, 2),
        },
    }


def _process_images() -> dict:
    """Cache entry for the image-based project — no PDF required."""
    png     = BODY_IMG.read_bytes() if BODY_IMG.exists() else None
    empty   = {"page_width": 0, "page_height": 0, "drawing_type": "image-based",
                "components": [], "summary": []}
    empty_e = {"items": [], "totals": {"total_cost_sek": 0, "total_length_m": 0}}
    ai_comp = load_images_cache(str(LEGEND_IMG), str(BODY_IMG))
    ai_est  = _estimate_ai(ai_comp) if ai_comp else None
    return {
        "png":               png,
        "components":        empty,
        "estimate":          empty_e,
        "components_vector": empty,
        "estimate_vector":   empty_e,
        "components_ai":     ai_comp,
        "estimate_ai":       ai_est,
    }


def _process(pdf_path: Path) -> dict:
    """Render PNG + run text/vector extract. Never calls Claude — zero tokens."""
    doc  = fitz.open(str(pdf_path))
    page = doc[0]
    pix  = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    png  = pix.tobytes("png")
    doc.close()

    comp     = extract(str(pdf_path))
    est      = estimate(str(pdf_path), comp)
    vec_comp = extract_vectors(str(pdf_path))
    vec_est  = _estimate_vector(vec_comp)

    # Load AI result from disk cache only — never call Claude here
    ai_comp = load_ai_cache(str(pdf_path))
    ai_est  = _estimate_ai(ai_comp) if ai_comp else None

    return {
        "png":               png,
        "components":        comp,
        "estimate":          est,
        "components_vector": vec_comp,
        "estimate_vector":   vec_est,
        "components_ai":     ai_comp,
        "estimate_ai":       ai_est,
    }


LEGEND_IMG = Path("images/legend.png")
BODY_IMG   = Path("images/body.png")


def _run_ai(pdf_path: Path) -> None:
    """Explicitly call Claude for one PDF and update the in-memory cache entry."""
    name = pdf_path.name
    _ai_running.add(name)
    print(f"  [AI] running extraction for {name} …")
    try:
        if LEGEND_IMG.exists() and BODY_IMG.exists():
            print(f"  [AI] using pre-split images (images/legend.png + images/body.png)")
            ai_comp = extract_with_images(str(LEGEND_IMG), str(BODY_IMG), _component_library,
                                          drawing_pdf_path=str(pdf_path))
        else:
            ai_comp = extract_with_ai(str(pdf_path), _component_library)
        ai_est  = _estimate_ai(ai_comp)
        _cache[name]["components_ai"] = ai_comp
        _cache[name]["estimate_ai"]   = ai_est
        print(f"  [AI] done: {name}")
    except Exception as exc:
        print(f"  [AI] failed for {name}: {exc}")
        raise
    finally:
        _ai_running.discard(name)


# ── Pre-load component library and all existing PDFs at startup ───────────────
_component_library = load_component_library()
if _component_library:
    print(f"Component library loaded ({len(_component_library.get('components', []))} components).")
else:
    # Auto-build from images/description.pdf if present
    _desc_img = Path("images/description.pdf")
    if _desc_img.exists():
        print("Building component library from images/description.pdf …")
        _component_library = build_component_library(str(_desc_img))
        print(f"  → {len(_component_library.get('components', []))} components extracted.")
    else:
        print("No component library found — upload description.pdf to build one.")

if LEGEND_IMG.exists() and BODY_IMG.exists():
    print("Loading project from images/legend.png + images/body.png …")
    _cache["project"] = _process_images()
else:
    for _p in sorted(PDF_DIR.glob("*.pdf")):
        if _p.name.lower() == "description.pdf":
            continue
        print(f"Loading {_p.name} …")
        _cache[_p.name] = _process(_p)
print(f"Ready — {len(_cache)} drawing(s) loaded.")
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/drawings")
def list_drawings():
    return {"drawings": sorted(_cache.keys())}


@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    """
    Accept one or more PDF files.
    • description.pdf → builds / rebuilds the component library (not added as drawing)
    • any other PDF    → processed as a drawing and cached
    Returns: {"loaded": [...], "errors": [...], "library_built": bool}
    """
    global _component_library
    loaded = []
    errors = []
    library_built = False

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            errors.append({"name": file.filename, "error": "not a PDF"})
            continue
        try:
            dest = PDF_DIR / file.filename
            dest.write_bytes(await file.read())

            if file.filename.lower() == "description.pdf":
                # Build component library from this description document
                print("Building component library from description.pdf …")
                _component_library = build_component_library(str(dest))
                print(f"  → {len(_component_library.get('components', []))} components extracted.")
                library_built = True
            else:
                _cache[file.filename] = _process(dest)
                loaded.append(file.filename)
        except Exception as exc:
            errors.append({"name": file.filename, "error": str(exc)})

    return {"loaded": loaded, "errors": errors, "library_built": library_built}


@app.get("/drawing/{name}/image")
def drawing_image(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    return Response(_cache[name]["png"], media_type="image/png")


NO_CACHE = {"Cache-Control": "no-store"}

@app.get("/drawing/{name}/components")
def drawing_components(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    return JSONResponse(_cache[name]["components"], headers=NO_CACHE)

@app.get("/drawing/{name}/components-vector")
def drawing_components_vector(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    return JSONResponse(_cache[name]["components_vector"], headers=NO_CACHE)

@app.get("/drawing/{name}/estimate")
def drawing_estimate(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    return JSONResponse(_cache[name]["estimate"], headers=NO_CACHE)

@app.get("/drawing/{name}/estimate-vector")
def drawing_estimate_vector(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    return JSONResponse(_cache[name]["estimate_vector"], headers=NO_CACHE)

@app.post("/drawing/{name}/run-ai")
def drawing_run_ai(name: str, force: bool = False):
    """Explicitly trigger Claude AI extraction for one drawing. Costs tokens.
    Pass ?force=true to delete existing cache and recompute from scratch."""
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    if force:
        pdf_path = PDF_DIR / name
        if pdf_path.exists():
            cache_file = _ai_cache_path(str(pdf_path))
            if cache_file.exists():
                cache_file.unlink()
                print(f"  [AI] PDF cache cleared for {name}")
        if LEGEND_IMG.exists() and BODY_IMG.exists():
            from extract_ai import AI_CACHE_DIR
            img_key   = _images_cache_key(str(LEGEND_IMG), str(BODY_IMG))
            img_cache = AI_CACHE_DIR / f"images_{img_key}.json"
            if img_cache.exists():
                img_cache.unlink()
                print(f"  [AI] image cache cleared ({img_key})")
        _cache[name]["components_ai"] = None
        _cache[name]["estimate_ai"]   = None
    _run_ai(PDF_DIR / name if name != "project" else Path("project"))
    return {"status": "ok", "name": name}


@app.post("/drawing/{name}/run-ai-stream")
async def drawing_run_ai_stream(name: str, force: bool = False):
    """SSE endpoint — streams phase/symbol events as AI extraction progresses."""
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")

    if force:
        pdf_path = PDF_DIR / name
        if pdf_path.exists():
            cache_file = _ai_cache_path(str(pdf_path))
            if cache_file.exists():
                cache_file.unlink()
        if LEGEND_IMG.exists() and BODY_IMG.exists():
            from extract_ai import AI_CACHE_DIR
            img_key   = _images_cache_key(str(LEGEND_IMG), str(BODY_IMG))
            img_cache = AI_CACHE_DIR / f"images_{img_key}.json"
            if img_cache.exists():
                img_cache.unlink()
        _cache[name]["components_ai"] = None
        _cache[name]["estimate_ai"]   = None

    loop     = asyncio.get_running_loop()
    event_q: asyncio.Queue = asyncio.Queue()

    def _cb(event):
        loop.call_soon_threadsafe(event_q.put_nowait, event)

    def _run():
        _ai_running.add(name)
        try:
            if LEGEND_IMG.exists() and BODY_IMG.exists():
                ai_comp = extract_with_images(
                    str(LEGEND_IMG), str(BODY_IMG), _component_library,
                    drawing_pdf_path=str(PDF_DIR / name), progress_cb=_cb)
            else:
                ai_comp = extract_with_ai(str(PDF_DIR / name), _component_library)
            ai_est = _estimate_ai(ai_comp)
            _cache[name]["components_ai"] = ai_comp
            _cache[name]["estimate_ai"]   = ai_est
            loop.call_soon_threadsafe(event_q.put_nowait, {"type": "done"})
        except Exception as exc:
            loop.call_soon_threadsafe(event_q.put_nowait,
                                      {"type": "error", "message": str(exc)})
        finally:
            _ai_running.discard(name)
            loop.call_soon_threadsafe(event_q.put_nowait, None)  # sentinel

    threading.Thread(target=_run, daemon=True).start()

    async def _generate():
        while True:
            event = await event_q.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/drawing/{name}/components-ai")
def drawing_components_ai(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    if name in _ai_running:
        raise HTTPException(202, "AI extraction in progress")
    data = _cache[name].get("components_ai")
    if data is None:
        raise HTTPException(503, "AI extraction not available — click Run AI")
    return JSONResponse(data, headers=NO_CACHE)

@app.get("/drawing/{name}/estimate-ai")
def drawing_estimate_ai(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    data = _cache[name].get("estimate_ai")
    if data is None:
        raise HTTPException(503, "AI estimate not available (check ANTHROPIC_API_KEY)")
    return JSONResponse(data, headers=NO_CACHE)

@app.get("/drawing/{name}/search")
def search_drawing(name: str, q: str = ""):
    """
    Return bounding boxes of every occurrence of text `q` in the drawing PDF.
    Coordinates are in PDF-point space (same as page_width / page_height).
    Search is case-insensitive: 'w1', 'W1', and 'W1' all find the same spans.
    """
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    if not q or not q.strip():
        return JSONResponse({"query": q, "matches": [], "count": 0})

    pdf_path = PDF_DIR / name
    if not pdf_path.exists():
        # Image-based project — no PDF text layer to search
        return JSONResponse({"query": q, "matches": [], "count": 0})
    doc  = fitz.open(str(pdf_path))
    page = doc[0]

    # page.search_for() returns coordinates in the raw, unrotated MediaBox space —
    # the same space as page.get_text("dict") blocks.  This does NOT account for
    # the page's /Rotate metadata, so on rotated PDFs the raw y-axis can exceed
    # page.rect.height (e.g. y=2057 on a page whose rect.height=1684 with rot=270).
    #
    # _to_display_bbox() applies the same rotation transform used by extract.py for
    # component overlay bboxes, converting raw → display coordinate space.
    #
    # Dark-text filter: room labels and grid references are printed in light grey
    # (~0xd9d9d9).  Only spans whose colour ≤ 0x404040 are considered dark/regular
    # text — the same threshold used by extract_ai.py.  We build a set of dark-span
    # rects once and skip any search hit that does not intersect a dark span.
    _DARK = 0x404040
    dark_rects: list[fitz.Rect] = [
        fitz.Rect(sp["bbox"])
        for blk in page.get_text("dict")["blocks"]
        if blk.get("type") == 0
        for ln in blk["lines"]
        for sp in ln["spans"]
        if sp.get("color", 0) <= _DARK
    ]

    seen: set[tuple] = set()
    matches: list[dict] = []
    for term in dict.fromkeys([q, q.upper(), q.lower()]):
        for r in page.search_for(term):
            # Skip if this hit overlaps no dark span (it's light/grey text)
            if not any(r.intersects(dr) for dr in dark_rects):
                continue
            dx0, dy0, dx1, dy1 = _to_display_bbox(
                (r.x0, r.y0, r.x1, r.y1), page
            )
            key = (round(dx0), round(dy0), round(dx1), round(dy1))
            if key not in seen:
                seen.add(key)
                matches.append({"x0": dx0, "y0": dy0, "x1": dx1, "y1": dy1})
    doc.close()

    return JSONResponse({
        "query":   q,
        "matches": matches,
        "count":   len(matches),
    })


@app.get("/library")
def get_library():
    """Return the current component library (built from description.pdf)."""
    if _component_library is None:
        return JSONResponse({"components": [], "status": "not_built"}, headers=NO_CACHE)
    return JSONResponse({**_component_library, "status": "ready"}, headers=NO_CACHE)


@app.get("/", response_class=HTMLResponse)
def root():
    return open("index.html", encoding="utf-8").read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_excludes=["pdf/*", "ai_cache/*", "*.json", "__pycache__/*"],
    )
