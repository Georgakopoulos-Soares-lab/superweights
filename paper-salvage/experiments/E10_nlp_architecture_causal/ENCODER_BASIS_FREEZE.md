# E10 Phase 2 — encoder basis freeze

Per `e10_prompt.md` Phase 2: for MosaicBERT and ModernBERT, `MODEL_AND_BASIS_AUDIT.md` §6-§7
already established that **only a single canonical high-gain row exists pre-E10 for each** —
the 5 "control rows" recorded in E8 are an explicitly-random background sample, not additional
high-gain candidates. This is **Case B** for both encoders.

Per Case B's procedure, the exact structural score already used in E7/E8
(`spectral_lib.row_spectral_metrics`, unmodified — `q1 = sigma_1^2 / sum_j sigma_j^2`) was
computed for **every** gated-FFN output row in each encoder, across all layers
(`rank_encoder_rows.py`, this directory; weight-only, no forward pass, no new metric). Rows
were ranked by `q1` descending and a fixed top-`K=10` set selected **without looking at any
intervention response** — no causal measurement has been run.

---

## MosaicBERT — `n_E = 10`

9,216 rows scored (12 layers x 768 rows). Canonical E8 row **L9/r287 (q1=0.4766) is naturally
inside the top-10, at rank 6** — the required check (`e10_prompt.md` Phase 2 step 5) passes;
no STOP/audit triggered.

| Rank | Layer | Row | `q1` | `PR_spec` | Note |
|---:|---:|---:|---:|---:|---|
| 0 | 10 | 666 | 0.8886 | 1.2656 | new (not previously known) |
| 1 | 10 | 79  | 0.8121 | 1.5151 | new |
| 2 | 10 | 333 | 0.7333 | 1.8576 | new |
| 3 | 0  | 287 | 0.6611 | 2.2682 | new |
| 4 | 0  | 416 | 0.5794 | 2.9399 | new |
| 5 | 1  | 79  | 0.5008 | 3.8244 | new |
| 6 | **9** | **287** | **0.4766** | **4.1159** | **E8 canonical row** |
| 7 | 5  | 79  | 0.4682 | 4.3305 | new |
| 8 | 9  | 666 | 0.4651 | 4.5459 | new |
| 9 | 0  | 444 | 0.4298 | 5.3251 | new |

Source artifact: `results/e10_encoder_row_ranking_mosaicbert.json` (full 9,216-row ranking
retained, not just the top-10, for reproducibility/audit).

Selection basis: **top-K exact-`U_k` construction** (Case B) — not a pre-existing set.

**Note (declared, not smoothed over):** every row in this top-10 except the canonical one is
newly identified by this ranking, not previously known to any E7/E8 artifact. Row 79 and row
287 each recur at multiple layers (79 at layers 1, 5, 10; 287 at layers 0 and 9) — a structural
echo worth noting for the retrospective interaction inspection later (Phase 5's "repeated
structural families" check), but not acted on here; the basis is frozen exactly as ranked.

## ModernBERT — `n_E = 10`

16,896 rows scored (22 layers x 768 rows). Canonical E8 row **L15/r251 (q1=0.8970) is
naturally inside the top-10, at rank 1** (second-highest of all 16,896 rows scored) — the
required check passes cleanly; no STOP/audit triggered.

| Rank | Layer | Row | `q1` | Note |
|---:|---:|---:|---:|---|
| 0 | 11 | 254 | 0.9127 | new (not previously known) |
| 1 | **15** | **251** | **0.8970** | **E8 canonical row** |
| 2 | 14 | 583 | 0.7526 | new |
| 3 | 11 | 251 | 0.7273 | new |
| 4 | 15 | 142 | 0.7037 | new |
| 5 | 9  | 251 | 0.6928 | new |
| 6 | 0  | 31  | 0.6897 | new |
| 7 | 15 | 67  | 0.6548 | new |
| 8 | 0  | 67  | 0.6548 | new |
| 9 | 11 | 67  | 0.6422 | new |

Source artifact: `results/e10_encoder_row_ranking_modernbert.json` (full 16,896-row ranking
retained).

Selection basis: **top-K exact-`U_k` construction** (Case B) — not a pre-existing set.

**Note (declared, not smoothed over):** row 251 recurs at layers 9, 11, and 15 (all three in
the top-10), and row 67 recurs at layers 0, 11, and 15 (also all three in the top-10) — a
stronger structural echo than MosaicBERT's. Flagged for the retrospective interaction
inspection (Phase 5), not acted on here.

---

## What this freeze does NOT do

- Does not run any causal intervention or response measurement.
- Does not pad either basis with ordinary (non-ranked) rows.
- Does not impose layer diversity — MosaicBERT's top-10 spans only 5 distinct layers (0, 1, 5,
  9, 10) and ModernBERT's spans only 6 (0, 9, 11, 14, 15) purely because that is what the
  exact ranking produced; no diversity constraint exists in the prior structural pipeline, so
  none is imposed here.
- Does not reorder or drop any row after this file is written.

Once this file (with both models complete) is committed, the basis for both encoders is
locked — no reselection, no reordering, per `e10_prompt.md` Phase 2's closing instruction.
