# Prompt Lessons — Electrical Drawing Extraction

## Setup — Qwen pass D (cross-model validator)

Pass D calls Qwen VL via OpenRouter as an independent second opinion alongside
the three Claude passes (A/B/C).  Enable it by setting:

```bash
export OPENROUTER_API_KEY=sk-or-...          # required
export QWEN_MODEL=qwen/qwen2.5-vl-72b-instruct  # optional, this is the default
```

Confidence with D enabled:
- **high**   = all 4 validators agree (A==B==C==D)
- **medium** = 3 of 4 agree (one outlier)
- **low**    = no clear majority — needs manual review

Without OPENROUTER_API_KEY: D shows as "—" in the UI and confidence falls back
to the 3-vote logic (high=3/3, medium=2/3, low=all differ).

---

Each entry records a failure, what caused it, and what rule was added to the prompt.
This file is the source of truth for **why** each clause in `extract_ai.py` exists.
When fine-tuning, pair each lesson with the corrected sample in `training_data/samples/`.

---

## L001 — Legend symbols must not be counted (2026-05-15)

**Drawing**: 1.pdf  
**Failure**: D1 counted as 5 instead of 3. D3 counted as 1 instead of 0.  
**Root cause**: Claude counted instances inside the FÖRKLARINGAR/legend box as real installed items.  
**Fix added to prompt**:
> "Use the legend / förklaringar box only to understand what each code means —
> do NOT count symbols shown inside the legend box itself."

**Fix in code**: `_count_from_pdf_text` scans for FÖRKLARINGAR label and cuts off text
at `y_label - 120` pts so legend-area spans are excluded from text search.

---

## L002 — N1 and N1-R are separate components (2026-05-15)

**Drawing**: 1.pdf  
**Failure**: N1 counted as 7. Should be N1=2, N1-R=5.  
**Root cause**: Regex `(?![\w-])` blocked matches in packed spans like "N1F2" (F = word char).  
**Fix in code**: Lookahead changed to two separate guards `(?!\d)(?!-[A-Z0-9])`.
`_inject_variant_codes` auto-discovers hyphen-suffix variants from PDF text.

---

## L003 — Light-font room labels must not be auto-discovered (2026-05-15)

**Drawing**: 1.pdf  
**Failure**: A1 appeared as a discovered component. A112–A118 also at risk.  
**Root cause**: General code scan found room labels printed in light grey (0xd9d9d9).  
**Fix in code**: `_inject_variant_codes` uses `dark_text` buffer (colour ≤ 0x404040) for
Pass 2 discovery. Room labels at 0xd9d9d9 are filtered out.

---

## L004 — WAGO / plug-in systems are count items, not length (2026-05-15)

**Drawing**: 1.pdf  
**Failure**: W1=85m, W2=35m, W3=20m (length). Should be counted as pieces.  
**Root cause**: Claude inferred "cable system" → measure by metre. But W1/W2/W3
are Sladdställ — factory-prefabricated plug-in leads ordered by piece.  
**Electrical engineer feedback**: "W1, W2 och W3 är snabbkopplingsystem med plug-in
funktion på kablarna. Dessa kommer i färdiga längder från fabrik. Räkna på antalen."  
**Fix added to prompt**:
> "Prefabricated / plug-in wiring systems (e.g. WAGO Sladdställ,
> snabbkopplingssystem, pendelupphängning) are delivered from the factory in fixed
> lengths. They must ALWAYS be classified as 'count' (antal), NOT 'length'."

---

## L005 — Height annotations belong to their own legend row only (2026-05-15)

**Drawing**: 1.pdf  
**Failure**: V17 got `uk_height=900` which belongs to adjacent legend item K.
Display showed "UK 900 mm ÖFG" on V17.  
**Root cause**: Legend table has K and V17 in adjacent rows; "UK 900 mm ÖFG" text
sits visually close to V17's row and Claude attributed it to V17.  
**Fix added to prompt**:
> "Each height annotation belongs to the component in the SAME legend row.
> Never carry a height from one legend row to an adjacent row.
> ÖK means the TOP edge; UK means the BOTTOM edge. Do not swap them."

**Cache patched manually**: V17 corrected to `ok_ofg_mm=900, uk_ofg_mm=null`.

---

## L006 — Code field must not include width or height annotations (2026-05-15)

**Drawing**: 2.pdf (cable tray routing plan)  
**Failure**: `type="KS 400"`, `type="KS 400 (OK 3000)"` instead of `code="KS"` + `width_mm=400`.  
**Root cause**: Claude treated the full label "KS 400" as the component code.  
**Fix added to prompt**:
> "The code field must ONLY contain the bare letter code (KS, FBK, KR).
> Extract width and height separately. KS 400 → code KS, width_mm 400."

---

## L007 — Architectural grid reference letters are not components (2026-05-15)

**Drawing**: 2.pdf  
**Failure**: AA=6, A=6, B=0 appeared as count items. These are zone/grid labels on
the drawing border, not installed components.  
**Root cause**: Single/double uppercase letters without digits are grid references on
Swedish architectural/engineering drawings (column A, B, C … row 1, 2, 3).  
**Fix in code**: After Pass 1, strip any code matching `^[A-ZÅÄÖ]{1,3}$` (pure letters,
no digits) from both count_items and length_items.

---

## L008 — L007 filter was too broad; replaced with structural code filter (2026-05-15)

**Drawing**: 3.pdf  
**Failure**: L007 regex `^[A-ZÅÄÖ]{1,3}$` would have removed KS, FBK, KR (2.pdf) and
FBA, NS, DA, KO, NB, HL (3.pdf) — all real component codes with no digits.  
**Fix in code**: Replaced with a structural filter on code shape:
- Contains whitespace → phrase (VIA NÖDSTOPP)
- Ends in ÖFG/ÖFK → height annotation (1000ÖFG)
- Pattern `\d+G\d` → cable spec (3G1,5 / 5G2,5)
- 3+ consecutive digits → room label (A112)
- Two dots: `\.\w+\.` → dot-separated tokens (TEXT.SLO.10)

**Fix added to prompt**:
> "Do NOT include cable spec labels (FRHF 3G1,5), height annotations (1000ÖFG),
> multi-word phrases (VIA NÖDSTOPP), or drawing grid references (A, B, C…)."

---

## L009 — Title block text bleeds into component counts (2026-05-15)

**Drawing**: 3.pdf  
**Failure**: FBA=+2, NS=+1, P=+1 overcounted because those codes appear in the
Swedish title block stamp (y=1615–2000), which is between the drawing body and
the FÖRKLARINGAR section. The FÖRKLARINGAR cut alone (y=2002) did not exclude it.  
**Root cause**: Swedish engineering drawings have a title block below the drawing
that contains FÖRFRÅGNINGSUNDERLAG / RELATIONSRITNING checkboxes, RITAD AV, DATUM
etc. Component codes printed there must not be counted.  
**Fix in code**: `_count_from_pdf_text` and `_inject_variant_codes` detect title
block markers (RITAD, FÖRFRÅGNINGSUNDERLAG, HANDLÄGGARE …) and record
`titleblk_y_cut`. Exclusion is x-AND-y: a span is dropped only when
`y ≥ titleblk_y_cut AND x < 300`. Spans in the drawing body (x > 600) at the
same y-band are kept. The FÖRKLARINGAR `legend_y_cut` remains a full-width cut.
Earlier v1 used `min(legend_y_cut, titleblk_y_cut)` as a single full-width cut
which dropped legitimate P11 labels in 1.pdf (count fell 27 → 21). The x-AND-y
approach fixes both the overcounting in 3.pdf and the regression in 1.pdf.

---

## L010 — Swedish description words (1-VÄGS, 2-VÄGS) must not be extracted as codes (2026-05-17)

**Drawing**: 1.pdf (and any drawing with DM, EK, DP type outlets)  
**Failure**: Code "1" appeared as a phantom component. Root entry: "DM 1-VÄGS UTTAG DISKMASKIN".  
**Root cause**: Two failure points acting together:
1. Pass 3 discovery used `(?<!\d)(\d{1,2})(?!\d)` which matched the "1" in "1-VÄGS" (not preceded/followed by digit). "1" appeared ≥3 times in legend text and was injected as a code.
2. Claude AI passes read "1-VÄGS" in the legend description and extracted "1" as a component code instead of "DM".
**Fix in code**: Extended `digit_pat` in Pass 3 to add `(?!-[A-Za-z\u00C0-\u024F])` so digits immediately followed by a hyphen+letter (Swedish description pattern) are not counted in `digit_freq`. "1-VÄGS", "2-VÄGS", "1-POL", "1-fas" no longer inflate the pure-digit frequency count.  
**Fix added to prompt**:
> "Swedish description words inside a legend entry text, such as '1-VÄGS', '2-VÄGS', '3-POL', '1-fas' describe the component type (1-way, 2-way, 3-pole) and are NOT codes. The code is the label BEFORE the dash-word, e.g. in 'DM 1-VÄGS UTTAG DISKMASKIN' the code is 'DM', not '1'."

---

## L011 — Single-letter component codes suppressed by grid-reference rule (2026-05-17)

**Drawing**: 1.pdf (lighting plan with återfjädrande dimmers/switches)  
**Failure**: Components with codes "A", "B", "T" etc. not counted. These are graphical symbols (spring-return dimmers, timers) with a single-letter code label in the legend. Count = 0 for all of them.  
**Root cause**: Two failure points:
1. Prompt rule "do not include architectural grid references (A, B, C…)" caused Claude to skip single-letter component codes — the rule did not distinguish between border grid labels and interior legend codes.
2. `_inject_variant_codes` Pass 2 only discovers `[letter]+[digit]` patterns, so pure-letter codes like "A", "T" are never auto-discovered from drawing body text.  
**Fix added to prompt**: Clarified that grid references appear ONLY in the outermost sheet border/margin, not inside the drawing body or legend. Single-letter codes next to graphical symbols in the legend ARE valid component codes.  
**Fix in code**: Added Pass 4 in `_inject_variant_codes`: scans the FÖRKLARINGAR section specifically for small isolated blocks (width < 80 pt, height < 40 pt) containing exactly one 1–3 letter dark token. These are the code-label cells in the legend table. Discovered codes go to vision counting (A/B/C/D) since they are never countable from text.

---

## L012 — Pure-digit legend codes in tall narrow blocks not discovered (2026-05-17)

**Drawing**: 3.pdf (4-vägguttag outlet)  
**Failure**: Code "4" (4-way outlet) never injected — legend scan found zero matches.  
**Root cause**: The FÖRKLARINGAR entry "4 4-VÄGSUTTAG, 300 ÖFG" is stored as one block
with width=11 pt (narrow) but height=129 pt (tall), because PyMuPDF keeps the code label
and its description in the same block when they wrap together. The height < 40 pt guard
blocked it.  
**Fix in code**: Split the legend scan into two cases:
- **Case A** (unchanged): block height ≤ 40 pt → exactly one token required.
- **Case B** (new): block width < 30 pt AND height > 40 pt → only the FIRST token is
  checked. This reliably picks up "4" from "4 4-VÄGSUTTAG, 300 ÖFG" while rejecting
  "2-VÄGSUTTAG" (first token fails `_SHORT_CODE` regex) and "3N/16A …" (contains "/").

---

## L013 — Legend row attribution: D1 gets Å/A's description, Å/A disappear (2026-05-17)

**Drawing**: 1.pdf (lighting plan — spring-return switches Å, A, RA)  
**Failure**: D1 and D2 showed descriptions "Återfjädrande strömst. 1-pol med DALI…" 
(which belongs to A/Å). Codes A and Å never appeared in Pass 1 output.  
**Root cause**: The legend contains D1–D5 (presence sensors) and below them Å, A, RA 
(spring-return switches). Claude read the D1 code label, then grabbed the nearby 
"Återfjädrande strömst. … med DALI…" description from the Å/A rows — confused by "DALI" 
starting with "D". Å and A were then never extracted as separate codes.  
**Fix added to prompt**:
> "IMPORTANT — reading legend rows: each row's description belongs ONLY to that row's 
> code. A 'D' inside a description word (e.g. 'DALI', 'MED DALI') is NOT a component 
> code — it is part of the description text. Codes that are a single letter or Å/Ä/Ö 
> (e.g. 'Å', 'A', 'RA') are valid component codes in their own rows — never merge them 
> into an adjacent numbered code (D1, D2…)."

---

## L014 — NT phantom hallucination: vision passes agree on a code that doesn't exist (2026-05-17)

**Drawing**: 1.pdf  
**Failure**: NT=2 (high confidence, A=2 B=2 C=2). "NT" appears ZERO times anywhere in the PDF.  
**Root cause**: Pass 1 hallucinated "NT" as a component code for "nödstopp" context it saw
visually. Vision passes A/B/C then consistently found 2 instances (likely the NÖDSTOPP-connected
switch symbols). All three agreeing on the same count gave false "high confidence."  
**Why it's hard to detect**: When all vision passes hallucinate the same value, vote-based
confidence cannot distinguish a correct unanimous count from a wrong unanimous hallucination.  
**Fix in code**: After vision counting, apply a phantom-hallucination guard: if a vision item's
code (≥ 2 chars) appears ZERO times anywhere in the full PDF text (all zones, word-boundary
search), force count to 0 and confidence to "low". Single-letter codes are excluded because
they appear constantly in Swedish prose and are always graphical-symbol codes from the legend.  
**Result**: NT is zeroed and flagged "PHANTOM?" in the server log. Real codes (FH, UT, D1…)
all pass because they appear at least once in the PDF text.

---

## L015 — Similar graphical variants (Å, A, RA) lumped together in vision count (2026-05-17)

**Drawing**: 1.pdf (spring-return switches: Å = 1-pol, A = 1-pol med DALI, RA = kron med DALI)  
**Failure**: A=3 counted (all three variants lumped), Å=0, RA missing entirely.  
**Root cause**:
1. Each `_count_one_symbol` call for "A" or "Å" had no context about sibling symbols, so
   Claude counted ALL spring-return switches as whichever variant it was asked for first.
2. "RA" in the legend has its "R" as a vector graphic (not PDF text) — only the "A" character
   appears in PDF text. So three isolated "A" blocks appear in the legend scan; "RA" is never
   auto-discovered. Pass 1 (with L013 fix) must correctly identify "RA".
**Fix in code**: Added `exclude_codes` parameter to `_count_one_symbol` and
`_count_one_symbol_legend`. When counting any vision item, all OTHER vision items are passed
as exclusions. The prompt explicitly tells Claude: "count ONLY 'A' (1-pol med DALI) — do NOT
count Å (1-pol) or RA (kron med DALI) here, they are counted in separate calls."

---

## Template for new lessons

```
## L00N — Short description (YYYY-MM-DD)

**Drawing**: <filename>
**Failure**: <what the model got wrong>
**Root cause**: <why it got it wrong>
**Electrical engineer feedback**: <if applicable>
**Fix added to prompt**: <exact clause added>
**Fix in code**: <if a code change was also made>
**Training sample**: training_data/samples/<hash>_verified.json
```
