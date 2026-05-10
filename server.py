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
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse
import fitz
from extract import extract
from extract_vector import extract_vectors
from estimate import estimate, COST_TABLE, COST_FALLBACK_PER_M
from extract_ai import build_component_library, load_component_library, extract_with_ai

app    = FastAPI()
PDF_DIR = Path("pdf")
PDF_DIR.mkdir(exist_ok=True)

# ── Per-drawing in-memory cache ────────────────────────────────────────────────
# { filename: {"png": bytes, "components": dict, "estimate": dict, ...} }
_cache: dict = {}

# ── Component library (built from description.pdf) ─────────────────────────────
_component_library: dict | None = None


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


def _process(pdf_path: Path) -> dict:
    """Render PNG + run extract + estimate for one PDF. Returns cache entry."""
    doc  = fitz.open(str(pdf_path))
    page = doc[0]
    pix  = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    png  = pix.tobytes("png")
    doc.close()

    comp     = extract(str(pdf_path))
    est      = estimate(str(pdf_path), comp)
    vec_comp = extract_vectors(str(pdf_path))
    vec_est  = _estimate_vector(vec_comp)

    # AI extraction — may be slow; runs only if ANTHROPIC_API_KEY is set
    ai_comp = None
    ai_est  = None
    try:
        ai_comp = extract_with_ai(str(pdf_path), _component_library)
        ai_est  = _estimate_ai(ai_comp)
    except Exception as exc:
        print(f"  [AI] extraction failed for {pdf_path.name}: {exc}")

    return {
        "png":               png,
        "components":        comp,
        "estimate":          est,
        "components_vector": vec_comp,
        "estimate_vector":   vec_est,
        "components_ai":     ai_comp,
        "estimate_ai":       ai_est,
    }


# ── Pre-load component library and all existing PDFs at startup ───────────────
_component_library = load_component_library()
if _component_library:
    print(f"Component library loaded ({len(_component_library.get('components', []))} components).")
else:
    print("No component library found — upload description.pdf to build one.")

for _p in sorted(PDF_DIR.glob("*.pdf")):
    if _p.name.lower() == "description.pdf":
        continue   # skip — not a drawing
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

@app.get("/drawing/{name}/components-ai")
def drawing_components_ai(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    data = _cache[name].get("components_ai")
    if data is None:
        raise HTTPException(503, "AI extraction not available (check ANTHROPIC_API_KEY)")
    return JSONResponse(data, headers=NO_CACHE)

@app.get("/drawing/{name}/estimate-ai")
def drawing_estimate_ai(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    data = _cache[name].get("estimate_ai")
    if data is None:
        raise HTTPException(503, "AI estimate not available (check ANTHROPIC_API_KEY)")
    return JSONResponse(data, headers=NO_CACHE)

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
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
