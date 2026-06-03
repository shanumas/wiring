"""Temp harness to iterate on DUBBELTRAPP / A_v1 / A_v2 counting until 3,3,0."""
import json, sys, os
from pathlib import Path
from dotenv import load_dotenv
ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
try:
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import extract_ai as E

legend_b64 = E._image_file_to_b64(str(ROOT / "images/legend.png"))
body_b64   = E._image_file_to_b64(str(ROOT / "images/body.png"))

syms = json.loads((ROOT / "ai_cache/step_legend5_98737a94a467.json").read_text(encoding="utf-8"))
by_vid = {s["visual_id"]: s for s in syms}
A1, A2, DT = by_vid["A_v1"], by_vid["A_v2"], by_vid["DUBBELTRAPP"]

mode = sys.argv[1] if len(sys.argv) > 1 else "perceive"

# Accurate shape descriptions derived from the model's own legend reading.
SHAPE = {
    "A_v1": "a circle with ONE straight vertical stem and a single curved branch on one side (asymmetric, one stem)",
    "A_v2": "a circle with ONE straight vertical stem and TWO curved branches on one side (asymmetric, one stem)",
    "DUBBELTRAPP": "a SYMMETRIC shape: two stems diverging in a 'V' from the base, each stem ending in a curved loop pointing OUTWARD in opposite directions (two stems, mirror-symmetric)",
}

def run_n(label, fn, n=3):
    print(f"\n===== {label} (x{n}) =====")
    for i in range(n):
        print(f"  run {i+1}: {fn()}")

if mode == "joint3":
    # Count all three together, two-step, with sharp shape contrast.
    def once():
        prompt = f"""IMAGE 1 = legend. IMAGE 2 = floor plan.
Count three switch-symbol variants in IMAGE 2. They look similar at small size, so judge by SHAPE
(ignore rotation — a rotated symbol is the same symbol):

  "A_v1": {SHAPE['A_v1']}
  "A_v2": {SHAPE['A_v2']}
  "DUBBELTRAPP": {SHAPE['DUBBELTRAPP']}

KEY DIFFERENCE: A_v1/A_v2 have ONE stem (asymmetric). DUBBELTRAPP has TWO stems forming a symmetric V.
STEP 1: count the TOTAL instances of any of these three in IMAGE 2 -> "total".
STEP 2: classify each into exactly one variant; sum MUST equal total. If a variant's exact shape
is absent, give it 0 (do not assign instances to a shape that isn't there).
Output ONLY JSON: {{"total": N, "A_v1": N, "A_v2": N, "DUBBELTRAPP": N}}"""
        return E._call_vision([legend_b64, body_b64], prompt, max_tokens=4096).strip()[:200]
    run_n("joint3", once)

elif mode == "dt_alone":
    def once():
        prompt = f"""IMAGE 1 = legend. IMAGE 2 = floor plan.
Count ONLY the DUBBELTRAPP symbol in IMAGE 2. Its shape: {SHAPE['DUBBELTRAPP']}.
It is SYMMETRIC with TWO diverging stems. Do NOT count the 'A' symbols, which have only ONE
stem with side-branches (the circle-on-a-stem 'A' symbols are a DIFFERENT symbol).
If the symmetric two-stem shape does not appear, the answer is 0.
Match by shape regardless of rotation. Do not count the legend itself.
Output ONLY JSON: {{"DUBBELTRAPP": N}}"""
        return E._call_vision([legend_b64, body_b64], prompt, max_tokens=4096).strip()[:200]
    run_n("dt_alone", once)

elif mode == "real":
    # Test the PRODUCTION _count_family_vision with the merged family + stored descriptions.
    def once():
        return E._count_family_vision(body_b64, [A1, A2, DT], legend_b64)
    run_n("real _count_family_vision([A_v1,A_v2,DUBBELTRAPP])", once, n=5)

elif mode == "voted":
    # Test the VOTED wrapper (n=3 internal calls) — should be stable at 3,3,0.
    def once():
        return E._count_family_vision_voted(body_b64, [A1, A2, DT], legend_b64, n=3)
    run_n("voted _count_family_vision_voted([A_v1,A_v2,DUBBELTRAPP])", once, n=3)

elif mode.startswith("tiles"):
    # Tile the body into N vertical strips (with overlap), zoom each, count the A
    # family per tile, sum. Tests whether lower per-image symbol density stabilises
    # the total at 6.  Usage: tiles2 / tiles3
    import fitz, base64
    ntiles = int(mode[5:] or "2")
    doc = fitz.open(str(ROOT / "images/body.png"))
    page = doc[0]
    W, H = page.rect.width, page.rect.height
    ov = 0.06  # 6% overlap each side
    def tile_b64(i):
        x0 = max(0, (i/ntiles - ov) * W)
        x1 = min(W, ((i+1)/ntiles + ov) * W)
        clip = fitz.Rect(x0, 0, x1, H)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip)
        return base64.standard_b64encode(pix.tobytes("png")).decode()
    tiles = [tile_b64(i) for i in range(ntiles)]
    print(f"  {ntiles} tiles, each ~{int(W/ntiles*(1+2*ov))}x{int(H)} px @3x")
    def once():
        tot_a, tot_dt = 0, 0
        for i, tb in enumerate(tiles):
            prompt = f"""This is a ZOOMED region of a Swedish electrical floor plan (IMAGE 2 = the region).
IMAGE 1 = legend. Count switch symbols of these shapes in the region:
  "A": {SHAPE['A_v1']} (one stem, circle + side branch(es))
  "DUBBELTRAPP": {SHAPE['DUBBELTRAPP']}
Judge by SHAPE, ignore rotation. Count ALL A-type (one-stem circle-on-stem) as "A".
Only count symbols whose center is INSIDE this region. Do not count the legend.
Output ONLY JSON: {{"A": N, "DUBBELTRAPP": N}}"""
            txt = E._call_vision([legend_b64, tb], prompt, max_tokens=4096)
            r = E._extract_json(txt)
            a, dt = int(r.get("A", 0) or 0), int(r.get("DUBBELTRAPP", 0) or 0)
            tot_a += a; tot_dt += dt
            print(f"      tile{i}: A={a} DT={dt}")
        return f"SUM A={tot_a} DUBBELTRAPP={tot_dt}"
    run_n(f"tiles{ntiles}", once, n=2)

elif mode.startswith("clean"):
    # Clean (no-overlap) vertical split, two-step total-then-classify per tile.
    import fitz, base64
    ntiles = int(mode[5:] or "2")
    doc = fitz.open(str(ROOT / "images/body.png"))
    page = doc[0]
    W, H = page.rect.width, page.rect.height
    def tile_b64(i):
        clip = fitz.Rect(i/ntiles * W, 0, (i+1)/ntiles * W, H)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=clip)
        return base64.standard_b64encode(pix.tobytes("png")).decode()
    tiles = [tile_b64(i) for i in range(ntiles)]
    print(f"  {ntiles} clean tiles, each ~{int(W/ntiles)}x{int(H)} px @3x")
    def once():
        tot_a, tot_dt = 0, 0
        for i, tb in enumerate(tiles):
            prompt = f"""IMAGE 1 = legend. IMAGE 2 = a ZOOMED region of an electrical floor plan.
Two switch symbol types may appear (judge by SHAPE, ignore rotation):
  TYPE A  = a circle with ONE stem and side-branch(es)  — {SHAPE['A_v1']}
  TYPE DT = {SHAPE['DUBBELTRAPP']}
STEP 1: count the TOTAL of BOTH types whose CENTER lies inside IMAGE 2 -> "total".
STEP 2: of that total, how many are TYPE DT (the symmetric two-stem double-loop)? -> "dt".
"a" = total - dt. A symbol cut off at the edge counts only if its CENTER is inside.
Do not count the legend. Output ONLY JSON: {{"total": N, "dt": N}}"""
            txt = E._call_vision([legend_b64, tb], prompt, max_tokens=4096)
            r = E._extract_json(txt)
            total, dt = int(r.get("total", 0) or 0), int(r.get("dt", 0) or 0)
            a = max(0, total - dt)
            tot_a += a; tot_dt += dt
            print(f"      tile{i}: total={total} dt={dt} -> A={a}")
        return f"SUM A={tot_a} DUBBELTRAPP={tot_dt}"
    run_n(f"clean{ntiles}", once, n=3)

elif mode == "perceive":
    # Can the model correctly describe each glyph from the legend alone?
    prompt = """IMAGE 1 is a legend from a Swedish electrical floor plan.
For EACH of these three legend rows, describe ONLY the small graphical glyph drawn
at the LEFT of the row (ignore the text). Be precise about: does it contain a circle?
a straight stem? curves/loops? Output JSON:
{"A_v1 (row: ÅTERFJÄDRANDE STRÖMST. 1-POL MED DALI)": "...",
 "A_v2 (row: ÅTERFJÄDRANDE STRÖMST. KRON MED DALI)": "...",
 "DUBBELTRAPP (row: DUBBELTRAPP. 1000 ÖFG.)": "..."}"""
    print(E._call_vision([legend_b64], prompt, max_tokens=4096))
