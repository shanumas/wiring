"""
save_training_sample.py
-----------------------
Mark a corrected ai_cache entry as a verified training sample.

Usage:
    python save_training_sample.py pdf/1.pdf
    python save_training_sample.py pdf/1.pdf --note "Fixed W1/W2/W3 to count, V17 uk_height"

What it saves to training_data/samples/:
  <stem>_<hash>_verified.json   — the corrected output (same schema as ai_cache)
  training_data/index.jsonl     — one-line-per-sample index for fine-tuning pipelines
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

AI_CACHE_DIR    = Path("ai_cache")
TRAINING_DIR    = Path("training_data/samples")
INDEX_FILE      = Path("training_data/index.jsonl")

TRAINING_DIR.mkdir(parents=True, exist_ok=True)


def pdf_hash(pdf_path: Path) -> str:
    return hashlib.sha1(pdf_path.read_bytes()).hexdigest()


def cache_path(pdf_path: Path) -> Path:
    h = pdf_hash(pdf_path)
    return AI_CACHE_DIR / f"{pdf_path.stem}_{h[:12]}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to the PDF (e.g. pdf/1.pdf)")
    parser.add_argument("--note", default="", help="Short note on what was corrected")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"PDF not found: {pdf_path}")

    cache = cache_path(pdf_path)
    if not cache.exists():
        sys.exit(f"No ai_cache entry for {pdf_path.name}. Run AI extraction first.")

    data = json.loads(cache.read_text(encoding="utf-8"))

    h     = pdf_hash(pdf_path)[:12]
    out   = TRAINING_DIR / f"{pdf_path.stem}_{h}_verified.json"

    # Embed metadata so we know when/why this was verified
    data["_training_meta"] = {
        "source_pdf":    pdf_path.name,
        "pdf_hash":      h,
        "verified_date": str(date.today()),
        "note":          args.note,
    }

    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out}")

    # Append to index
    entry = {
        "pdf":        pdf_path.name,
        "hash":       h,
        "sample":     str(out),
        "date":       str(date.today()),
        "note":       args.note,
        "n_components": len(data.get("components", [])),
    }
    with INDEX_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Index: {INDEX_FILE}  ({entry['n_components']} components)")


if __name__ == "__main__":
    main()
