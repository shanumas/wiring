"""
FastAPI server for the electrical component extractor.
  GET /            → index.html
  GET /image       → rendered PDF page as PNG
  GET /components  → extracted components as JSON
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, JSONResponse
import fitz
import json
from extract import extract

app = FastAPI()

# ── pre-render once at startup ────────────────────────────────────────────────
_doc   = fitz.open("pdf/2.pdf")
_page  = _doc[0]
_scale = 1.5                              # 150 DPI — sharp on retina, fast to serve
_mat   = fitz.Matrix(_scale, _scale)
_pix   = _page.get_pixmap(matrix=_mat)
_png   = _pix.tobytes("png")

_components = extract()                   # parse once, cache result
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/image")
def get_image():
    return Response(content=_png, media_type="image/png")


@app.get("/components")
def get_components():
    return JSONResponse(_components)


@app.get("/", response_class=HTMLResponse)
def root():
    return open("index.html", encoding="utf-8").read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
