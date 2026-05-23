"""
Wiring takeoff test runner.
Loads test_expected.json, runs extract_with_images, compares results.

Usage:
    python run_tests.py
    python run_tests.py --no-cache   # wipe step caches before running
"""

import json
import sys
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

def _load_expected() -> dict:
    return json.loads((ROOT / "test_expected.json").read_text(encoding="utf-8"))

def _method_label(s: dict) -> str:
    """Map count_method string to 'text' or 'vision'."""
    m = (s.get("count_method") or "").lower()
    if "pass_c" in m or m == "text":
        return "text"
    if "pass_d" in m or m == "vision":
        return "vision"
    return m

def _wipe_step_caches():
    from extract_ai import AI_CACHE_DIR
    wiped = 0
    for f in AI_CACHE_DIR.glob("step_pass_c*.json"):
        f.unlink()
        wiped += 1
    for f in AI_CACHE_DIR.glob("step_pass_d*.json"):
        f.unlink()
        wiped += 1
    print(f"  [cache] wiped {wiped} step cache files")

def run(wipe_cache: bool = False) -> bool:
    cfg = _load_expected()
    legend = ROOT / cfg["drawing"]["legend"]
    body   = ROOT / cfg["drawing"]["body"]

    if wipe_cache:
        _wipe_step_caches()

    print(f"\nRunning extraction on:")
    print(f"  legend: {legend}")
    print(f"  body:   {body}\n")

    # Import here so wipe_cache happens first
    from extract_ai import extract_with_images
    result = extract_with_images(str(legend), str(body))

    # Build lookup: code -> summary entry (use first match)
    by_code: dict[str, dict] = {}
    for s in result.get("summary", []):
        code = (s.get("original_code") or s.get("system") or "").strip()
        if code and code not in by_code:
            by_code[code] = s

    expected = cfg["expected"]
    passes, fails = 0, 0

    print(f"{'CODE':<10} {'EXPECTED':>10} {'GOT':>10}  {'METHOD EXP':<12} {'METHOD GOT':<12}  RESULT")
    print("-" * 75)

    for item in expected:
        code     = item["code"]
        exp_cnt  = item["count"]
        exp_meth = item.get("method", "")

        s = by_code.get(code)
        if s is None:
            print(f"{code:<10} {exp_cnt:>10} {'—':>10}  {exp_meth:<12} {'not found':<12}  FAIL (missing)")
            fails += 1
            continue

        got_cnt  = int(s.get("count") or 0)
        got_meth = _method_label(s)

        cnt_ok   = got_cnt == exp_cnt
        meth_ok  = (not exp_meth) or (got_meth == exp_meth)
        ok       = cnt_ok and meth_ok

        status = "✓ PASS" if ok else "✗ FAIL"
        cnt_flag  = "" if cnt_ok  else f" ← count off by {got_cnt - exp_cnt:+d}"
        meth_flag = "" if meth_ok else f" ← wrong method"
        print(f"{code:<10} {exp_cnt:>10} {got_cnt:>10}  {exp_meth:<12} {got_meth:<12}  {status}{cnt_flag}{meth_flag}")

        if ok:
            passes += 1
        else:
            fails += 1

    print("-" * 75)
    print(f"\n{passes}/{passes+fails} passed", "✓" if fails == 0 else "✗")
    return fails == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Wipe step caches before running")
    args = parser.parse_args()

    ok = run(wipe_cache=args.no_cache)
    sys.exit(0 if ok else 1)
