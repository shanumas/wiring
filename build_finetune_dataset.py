"""
build_finetune_dataset.py
-------------------------
Convert training_data/samples/ → fine-tuning JSONL for Qwen2.5-VL.

Outputs
-------
training_data/
  images/              rendered PNG for each drawing (1× scale, ~2400px)
  finetune_llamafactory.jsonl   LLaMA Factory messages format
  finetune_swift.jsonl          ms-swift / modelscope format
  dataset_info.json             LLaMA Factory dataset registry

Usage
-----
    python3 build_finetune_dataset.py
    python3 build_finetune_dataset.py --pdf-dir /path/to/pdfs

Why two formats?
    LLaMA Factory  — most widely supported, good LoRA tooling
    ms-swift       — official Qwen recommendation, slightly simpler setup
Both formats are equivalent; pick whichever trainer you use.

Training target
---------------
The model is trained to produce the RAW Pass-1 JSON that _ask_full_drawing()
currently asks Claude for.  The post-processing steps (text counting,
variant discovery) remain in Python code — we only train the vision pass.

Key things the model learns from these samples:
  • measurement_type: "count" vs "length" (W1/W2/W3 = count, KS = length)
  • height attribution per legend row (V17 → ok_height, not uk_height)
  • legend exclusion (D3 = 0, not 1)
  • component identification (all codes present in this drawing type)
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import fitz   # PyMuPDF

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLES_DIR  = Path("training_data/samples")
IMAGES_DIR   = Path("training_data/images")
INDEX_FILE   = Path("training_data/index.jsonl")
PDF_DIR      = Path("pdf")

LF_OUT   = Path("training_data/finetune_llamafactory.jsonl")   # LLaMA Factory
SW_OUT   = Path("training_data/finetune_swift.jsonl")          # ms-swift
INFO_OUT = Path("training_data/dataset_info.json")             # LLaMA Factory registry

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

MAX_PX = 2400   # same resolution used during extraction

# ── Instruction prompt ────────────────────────────────────────────────────────
# This is the same prompt as _ask_full_drawing() in extract_ai.py.
# Kept in sync manually; if you update the prompt there, update it here too.
INSTRUCTION = """\
You are an expert quantity surveyor performing a drawing takeoff on a Swedish \
building services drawing.

Analyse this drawing image and return a full quantity takeoff.

Instructions:
1. Identify EVERY component code present — read codes EXACTLY as printed in the
   drawing (e.g. "V18" is not "V11", "D2" is not "D7"). Read each digit carefully.
   Use the legend / förklaringar box only to understand what each code means —
   do NOT count symbols shown inside the legend box itself.
   IMPORTANT: Only use codes that actually appear in the drawing labels.
2. For each code decide the measurement_type:
   • "count"  → point-installed items (fixtures, outlets, sensors, panels, switches …)
   • "length" → line-installed items (cable trays, cable ladders, conduits, pipes …)
   IMPORTANT: Prefabricated / plug-in wiring systems (e.g. WAGO Sladdställ,
   snabbkopplingssystem, pendelupphängning) are delivered from the factory in fixed
   lengths. They must ALWAYS be classified as "count" (antal), NOT "length".
3. Read the drawing scale from the title block.
4. Capture width/size annotations (e.g. "KS 400" → width_mm 400).
5. Capture mounting heights (ÖK = top of item, UK = bottom, in mm ÖFG).
   IMPORTANT: each height annotation belongs to the component in the SAME legend row.
   Never carry a height from one legend row to an adjacent row.
   ÖK means the TOP edge; UK means the BOTTOM edge. Do not swap them.
6. Note fire ratings (e.g. "EI 30-C") if shown alongside a component.

YOUR RESPONSE MUST BE A RAW JSON OBJECT WITH NO TEXT BEFORE OR AFTER IT.
No markdown fences, no explanation, no commentary.

{
  "scale": "1:50",
  "drawing_type": "one-line description",
  "components": [
    {"code": "P11", "name": "full name", "measurement_type": "count",
     "quantity": 8, "unit": "pcs", "width_mm": null,
     "ok_height_mm": null, "uk_height_mm": null, "fire_rating": null},
    {"code": "KS", "name": "Kabelstege", "measurement_type": "length",
     "quantity": 45.5, "unit": "m", "width_mm": 400,
     "ok_height_mm": 2700, "uk_height_mm": null, "fire_rating": null}
  ]
}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def render_pdf_png(pdf_path: Path, out_png: Path) -> Path:
    """Render first page of PDF to PNG at MAX_PX resolution."""
    doc  = fitz.open(str(pdf_path))
    page = doc[0]
    longest = max(page.rect.width, page.rect.height)
    scale   = min(MAX_PX / longest, 2.0)
    pix     = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    pix.save(str(out_png))
    doc.close()
    return out_png


def png_to_b64(png_path: Path) -> str:
    return base64.standard_b64encode(png_path.read_bytes()).decode()


def denormalize(verified: dict) -> dict:
    """
    Reconstruct the raw _ask_full_drawing() output format from the normalised
    cache.  This is the training TARGET — what the model should produce.

    Normalised cache fields → raw format:
      comp["type"]         → code
      comp["measurement_type"]  kept
      summary count/total  → quantity (use the VERIFIED count, not raw AI guess)
      comp["size"]         → width_mm
      comp["ok_height"]    → ok_height_mm
      comp["uk_height"]    → uk_height_mm
      comp["fire_rating"]  kept
    """
    # Build a lookup: code → summary row (for verified quantity)
    sum_by_code: dict = {}
    for s in verified.get("summary", []):
        code = s["system"]
        sum_by_code[code] = s

    raw_components = []
    seen = set()
    for comp in verified.get("components", []):
        code  = comp.get("type", "?")
        if code in seen:
            continue
        seen.add(code)

        mtype = comp.get("measurement_type", "count")
        s     = sum_by_code.get(code, {})

        if mtype == "length":
            qty  = s.get("total_length_m", comp.get("quantity", 0))
            unit = "m"
        else:
            qty  = s.get("count", comp.get("quantity", 0))
            unit = "pcs"

        width = comp.get("size")
        try:
            width = int(width) if width and str(width).isdigit() else None
        except (TypeError, ValueError):
            width = None

        ok_h = comp.get("ok_height")
        uk_h = comp.get("uk_height")
        try:
            ok_h = int(ok_h) if ok_h else None
        except (TypeError, ValueError):
            ok_h = None
        try:
            uk_h = int(uk_h) if uk_h else None
        except (TypeError, ValueError):
            uk_h = None

        raw_components.append({
            "code":             code,
            "name":             comp.get("name", code),
            "measurement_type": mtype,
            "quantity":         qty,
            "unit":             unit,
            "width_mm":         width,
            "ok_height_mm":     ok_h,
            "uk_height_mm":     uk_h,
            "fire_rating":      comp.get("fire_rating"),
        })

    return {
        "scale":        verified.get("scale", "unknown"),
        "drawing_type": verified.get("drawing_type", "unknown"),
        "components":   raw_components,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default=str(PDF_DIR),
                        help=f"Directory containing source PDFs (default: {PDF_DIR})")
    parser.add_argument("--embed-images", action="store_true",
                        help="Embed images as base64 in JSONL instead of file paths "
                             "(larger files, no external deps at training time)")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)

    # Collect verified samples
    samples = sorted(SAMPLES_DIR.glob("*_verified.json"))
    if not samples:
        sys.exit("No verified samples found in training_data/samples/. "
                 "Run save_training_sample.py first.")

    lf_rows = []   # LLaMA Factory
    sw_rows = []   # ms-swift

    for sample_path in samples:
        verified = json.loads(sample_path.read_text(encoding="utf-8"))
        meta     = verified.get("_training_meta", {})
        pdf_name = meta.get("source_pdf") or sample_path.stem.split("_verified")[0] + ".pdf"
        pdf_path = pdf_dir / pdf_name

        if not pdf_path.exists():
            print(f"  [skip] PDF not found: {pdf_path}", file=sys.stderr)
            continue

        # Render PNG
        png_name = sample_path.stem.replace("_verified", "") + ".png"
        png_path = IMAGES_DIR / png_name
        if not png_path.exists():
            print(f"  Rendering {pdf_name} → {png_path.name} …")
            render_pdf_png(pdf_path, png_path)
        else:
            print(f"  {png_path.name} already rendered, skipping.")

        # Build target output (raw Pass-1 format)
        target_json = json.dumps(denormalize(verified), ensure_ascii=False)

        # Image reference: path or base64
        if args.embed_images:
            img_ref = f"data:image/png;base64,{png_to_b64(png_path)}"
        else:
            img_ref = str(png_path.resolve())

        # ── LLaMA Factory (messages format) ───────────────────────────────────
        # https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md
        lf_rows.append({
            "messages": [
                {
                    "role":    "user",
                    "content": f"<image>\n{INSTRUCTION}",
                },
                {
                    "role":    "assistant",
                    "content": target_json,
                },
            ],
            "images": [img_ref],
        })

        # ── ms-swift ──────────────────────────────────────────────────────────
        # https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html
        sw_rows.append({
            "messages": [
                {
                    "role":    "user",
                    "content": f"<image>{INSTRUCTION}",
                },
                {
                    "role":    "assistant",
                    "content": target_json,
                },
            ],
            "images": [img_ref],
        })

        print(f"  Added: {pdf_name}  "
              f"({len(denormalize(verified)['components'])} components)")

    # Write outputs
    def write_jsonl(path: Path, rows: list):
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} samples → {path}")

    write_jsonl(LF_OUT, lf_rows)
    write_jsonl(SW_OUT, sw_rows)

    # LLaMA Factory dataset_info.json
    info = {
        "wiring_drawings": {
            "file_name":   "finetune_llamafactory.jsonl",
            "formatting":  "sharegpt",
            "columns": {
                "messages": "messages",
                "images":   "images",
            },
        }
    }
    INFO_OUT.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote dataset registry → {INFO_OUT}")

    print(f"\nDone. {len(lf_rows)} training sample(s) ready.")
    print("\nNext steps:")
    print("  LLaMA Factory:  copy dataset_info.json + finetune_llamafactory.jsonl")
    print("                  into LLaMA-Factory/data/ and add to dataset_info.json")
    print("  ms-swift:       swift sft --model Qwen/Qwen2.5-VL-7B-Instruct \\")
    print("                    --dataset finetune_swift.jsonl \\")
    print("                    --train_type lora --output_dir ./output")


if __name__ == "__main__":
    main()
