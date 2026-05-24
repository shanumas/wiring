"""
Wiring takeoff test runner.
Loads test_expected.json, runs extract_with_images, compares results.

Usage:
    python run_tests.py
    python run_tests.py --no-cache    # wipe pass-C/D step caches and image cache
    python run_tests.py --text-only   # PDF text search only, no AI calls (fast)
"""

import json
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# Fix Windows SSL cert verification (same as server.py)
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    import certifi, ssl
    _real_create = ssl.create_default_context
    def _patched(*a, **kw):
        kw.setdefault("cafile", certifi.where())
        return _real_create(*a, **kw)
    ssl.create_default_context = _patched

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def _load_expected() -> dict:
    return json.loads((ROOT / "test_expected.json").read_text(encoding="utf-8"))

def _method_label(s: dict) -> str:
    m = (s.get("count_method") or "").lower()
    if "pass_c" in m or m == "text":
        return "text"
    if "pass_d" in m or m == "vision":
        return "vision"
    return m

def _wipe_caches(legend_path: str, body_path: str):
    from extract_ai import AI_CACHE_DIR, _images_cache_key
    wiped = 0
    # Wipe outer image-level cache so stale full results don't hide prompt changes
    img_key = _images_cache_key(legend_path, body_path)
    img_file = AI_CACHE_DIR / f"images_{img_key}.json"
    if img_file.exists():
        img_file.unlink()
        wiped += 1
    # Wipe pass-C and pass-D step caches
    for f in AI_CACHE_DIR.glob("step_pass_c*.json"):
        f.unlink()
        wiped += 1
    for f in AI_CACHE_DIR.glob("step_pass_d*.json"):
        f.unlink()
        wiped += 1
    print(f"  [cache] wiped {wiped} cache file(s)")

def run(wipe_cache: bool = False, text_only: bool = False) -> bool:
    cfg    = _load_expected()
    legend = str(ROOT / cfg["drawing"]["legend"])
    body   = str(ROOT / cfg["drawing"]["body"])

    if wipe_cache:
        _wipe_caches(legend, body)

    print(f"\nRunning extraction on:")
    print(f"  legend: {legend}")
    print(f"  body:   {body}\n")

    from extract_ai import extract_with_images, load_component_library, build_component_library
    from pathlib import Path as _Path
    lib = load_component_library()
    if lib is None:
        desc = ROOT / "images" / "description.pdf"
        if desc.exists():
            print("  Building component library from images/description.pdf …")
            lib = build_component_library(str(desc))
        else:
            print("  No component library — extra items will not be counted")
    pdf_path = str(ROOT / cfg["drawing"]["pdf"]) if cfg["drawing"].get("pdf") else None
    if pdf_path:
        print(f"  pdf:    {pdf_path}\n")

    pass_mode = "text" if text_only else "all"
    result = extract_with_images(legend, body, lib, drawing_pdf_path=pdf_path,
                                 pass_mode=pass_mode)

    # Build lookup: code -> first matching summary entry
    # Index by both original_code (e.g. "⊡m") and system/visual_id (e.g. "elcentral")
    # so test_expected.json can use either form.
    by_code: dict[str, dict] = {}
    for s in result.get("summary", []):
        for key in [s.get("original_code"), s.get("system")]:
            key = (key or "").strip()
            if key and key not in by_code:
                by_code[key] = s

    expected = cfg["expected"]
    if text_only:
        expected = [e for e in expected if e.get("method") == "text"]

    passes, fails = 0, 0

    print(f"\n{'CODE':<10} {'EXPECTED':>8} {'GOT':>8}  {'EXP METHOD':<12} {'GOT METHOD':<12}  RESULT")
    print("-" * 72)

    for item in expected:
        code     = item["code"]
        exp_cnt  = item["count"]
        exp_meth = item.get("method", "")

        s = by_code.get(code)
        if s is None:
            print(f"{code:<10} {exp_cnt:>8} {'?':>8}  {exp_meth:<12} {'not found':<12}  FAIL (missing)")
            fails += 1
            continue

        got_cnt  = int(s.get("count") or 0)
        got_meth = _method_label(s)

        cnt_ok  = (got_cnt == exp_cnt)
        meth_ok = (not exp_meth) or (got_meth == exp_meth)
        ok      = cnt_ok and meth_ok

        flags = ""
        if not cnt_ok:
            flags += f" <- count off by {got_cnt - exp_cnt:+d}"
        if not meth_ok:
            flags += f" <- wrong method"

        status = "PASS" if ok else "FAIL"
        print(f"{code:<10} {exp_cnt:>8} {got_cnt:>8}  {exp_meth:<12} {got_meth:<12}  {status}{flags}")

        if ok:
            passes += 1
        else:
            fails += 1

    print("-" * 72)
    total = passes + fails
    print(f"\n{passes}/{total} passed {'(all good!)' if fails == 0 else f'({fails} failing)'}\n")
    return fails == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true",
                        help="Wipe pass-C/D step caches and image cache before running")
    parser.add_argument("--text-only", action="store_true",
                        help="Run PDF text search only — no AI vision calls (fast, used in pre-commit)")
    args = parser.parse_args()
    ok = run(wipe_cache=args.no_cache, text_only=args.text_only)
    sys.exit(0 if ok else 1)
