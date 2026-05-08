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
from estimate import estimate

app    = FastAPI()
PDF_DIR = Path("pdf")

# ── Per-drawing in-memory cache ────────────────────────────────────────────────
# { filename: {"png": bytes, "components": dict, "estimate": dict} }
_cache: dict = {}


def _process(pdf_path: Path) -> dict:
    """Render PNG + run extract + estimate for one PDF. Returns cache entry."""
    doc  = fitz.open(str(pdf_path))
    page = doc[0]
    pix  = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
    png  = pix.tobytes("png")
    doc.close()

    comp = extract(str(pdf_path))
    est  = estimate(str(pdf_path), comp)

    return {"png": png, "components": comp, "estimate": est}


# ── Pre-load all existing PDFs at startup ─────────────────────────────────────
for _p in sorted(PDF_DIR.glob("*.pdf")):
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
    Accept one or more PDF files, process each, return list of names.
    Files are saved to pdf/ and cached; re-uploading the same name
    replaces the cached result.
    """
    loaded = []
    errors = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            errors.append({"name": file.filename, "error": "not a PDF"})
            continue
        try:
            dest = PDF_DIR / file.filename
            dest.write_bytes(await file.read())
            _cache[file.filename] = _process(dest)
            loaded.append(file.filename)
        except Exception as exc:
            errors.append({"name": file.filename, "error": str(exc)})

    return {"loaded": loaded, "errors": errors}


@app.get("/drawing/{name}/image")
def drawing_image(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    return Response(_cache[name]["png"], media_type="image/png")


@app.get("/drawing/{name}/components")
def drawing_components(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    return JSONResponse(_cache[name]["components"])


@app.get("/drawing/{name}/estimate")
def drawing_estimate(name: str):
    if name not in _cache:
        raise HTTPException(404, f"Drawing '{name}' not found")
    return JSONResponse(_cache[name]["estimate"])


@app.get("/", response_class=HTMLResponse)
def root():
    return open("index.html", encoding="utf-8").read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
