# Prompt Lessons — Electrical Drawing Extraction

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
