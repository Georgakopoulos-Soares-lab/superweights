# E10 Phase 2 — encoder basis freeze (v2, corrected)

> **v1 of this file selected rows by `q1` and was wrong.** `q1` measures internal spectral
> concentration (rank-1-ness) and is scale-invariant — it carries no magnitude information and
> is not a high-gain score. See `PROTOCOL_CORRECTION_01.md` for the full audit. This v2 uses
> the correct criterion. **No causal measurement was run under v1.**

Per `e10_prompt.md` Phase 2: `MODEL_AND_BASIS_AUDIT.md` §6-§7 established that **only a single
canonical high-gain row exists pre-E10 for each encoder** — E8's 5 "control rows" are an
explicitly-random background sample, not additional candidates. Both encoders are **Case B**.

## Selection criterion (corrected)

**Exact ‖U_k‖_F relative to the row's own layer median** — the convention
`paper-salvage/src/uk_frobenius.py::layer_report` implements (`max_over_median`), using the
*exact* operator norm from `spectral_lib.py` (`sqrt(sum_j sigma_j^2)`, cross terms retained),
**not** `uk_frobenius.py`'s diagonal approximation and **not** `q1`.

Computed for **every** gated-FFN output row in each encoder, across all layers, weight-only,
no forward pass, no new metric (`rank_encoder_rows.py` produced the per-row values;
`audit_selection_criterion.py` re-ranked them by the corrected criterion). Rows were then
ranked and a fixed top-`K=10` selected **without reference to any intervention response** —
none has been measured.

`q1` appears below **as an annotation column only**, never as a selector.

---

## MosaicBERT — `n_E = 10`

9,216 rows scored (12 layers x 768). Canonical E8 detector row **L9/r287 is inside the basis at
rank 9** — the required check (`e10_prompt.md` Phase 2 step 5) passes; no STOP/audit triggered.
Within its own layer it is the **#0 row at 5.05x the layer median**, consistent with E8's
activation detector having found it there.

| Rank | Layer | Row | ‖U_k‖_F / layer median | exact ‖U_k‖_F | `q1` (annotation) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 287 | 12.29x | 39.198 | 0.6611 |
| 2 | 0 | 444 | 6.61x | 21.086 | 0.4298 |
| 3 | 1 | 287 | 6.45x | 23.109 | 0.0763 |
| 4 | 11 | 287 | 5.91x | 28.900 | 0.2642 |
| 5 | 0 | 302 | 5.84x | 18.623 | 0.3369 |
| 6 | 5 | 287 | 5.76x | 23.147 | 0.1275 |
| 7 | 3 | 287 | 5.67x | 22.427 | 0.1466 |
| 8 | 10 | 666 | 5.27x | 22.375 | 0.8886 |
| 9 | **9** | **287** | **5.05x** | 19.998 | 0.4766 | **E8 canonical** |
| 10 | 10 | 79 | 4.81x | 20.424 | 0.8121 |

Source artifacts: `results/e10_encoder_row_ranking_mosaicbert.json` (full 9,216-row per-row
values), `results/e10_selection_audit_mosaicbert.json` (corrected ranking).

**Note:** output row 287 dominates this basis, recurring at layers 0, 1, 3, 5, 9, and 11 — a
strong cross-layer structural echo, flagged for the Phase 5 retrospective interaction
inspection ("repeated structural families"), not acted on here.

## ModernBERT — `n_E = 10`

16,896 rows scored (22 layers x 768). Canonical E8 detector row **L15/r251 is rank 1** — the
single largest-magnitude `U_k` operator in the entire model (836.05 vs. 321.67 for the next
largest anywhere), at 23.55x its layer median. E8's forward-pass activation detector and this
weight-space magnitude criterion independently agree on the same row.

| Rank | Layer | Row | ‖U_k‖_F / layer median | exact ‖U_k‖_F | `q1` (annotation) |
|---:|---:|---:|---:|---:|---:|
| 1 | **15** | **251** | **23.55x** | 836.05 | 0.8970 | **E8 canonical** |
| 2 | 0 | 251 | 8.52x | 278.32 | 0.6342 |
| 3 | 11 | 251 | 8.03x | 321.67 | 0.7273 |
| 4 | 0 | 67 | 7.52x | 245.71 | 0.6548 |
| 5 | 9 | 251 | 7.52x | 304.09 | 0.6928 |
| 6 | 4 | 251 | 5.96x | 226.15 | 0.3420 |
| 7 | 1 | 251 | 5.08x | 184.83 | 0.2626 |
| 8 | 5 | 251 | 4.86x | 187.41 | 0.0725 |
| 9 | 10 | 251 | 4.80x | 192.40 | 0.0515 |
| 10 | 1 | 67 | 4.58x | 166.74 | 0.2478 |

Source artifacts: `results/e10_encoder_row_ranking_modernbert.json`,
`results/e10_selection_audit_modernbert.json`.

**Note:** output row 251 recurs at layers 0, 1, 4, 5, 9, 10, 11, 15 (8 of the 10 basis rows)
and row 67 at layers 0 and 1 — an even stronger echo than MosaicBERT's. Flagged for Phase 5,
not acted on.

---

## Robustness of the ranking form

Ranking by **global** exact ‖U_k‖_F (no layer normalization) yields the **identical 10-row set
for MosaicBERT** (10/10, reordered) and **9/10 for ModernBERT** (global would swap L1/r67 for
L6/r251). The layer-relative form is primary because it matches `layer_report`'s own
convention; the choice does not materially change either basis.

## Disclosed caveat — no weight-space eligibility threshold exists

The project's only numeric high-gain eligibility threshold is E7/E8's **activation-detector**
ratio (≥ 5.0x), an activation-space quantity. No weight-space ‖U_k‖_F threshold has ever been
defined here, and importing 5.0x to a weight-space ratio would invent a new structural
criterion, which E10's hard rules forbid. `K = 10` is retained for both encoders. Recorded but
**not acted on**: under such an import MosaicBERT would have 9 eligible rows and ModernBERT 7.

## What this freeze does NOT do

- Does not run any causal intervention or response measurement.
- Does not pad either basis with ordinary (non-ranked) rows.
- Does not impose layer diversity — the layer spread is whatever the exact ranking produced.
- Does not use `q1`, `PR_spec`, or `stable_rank` for any selection decision.
- Does not reorder or drop any row after this file is committed and the v2 prereg is locked.
