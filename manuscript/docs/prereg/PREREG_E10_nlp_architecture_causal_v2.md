# PREREG — E10: NLP architecture-associated causal phenotypes (v2)

> **This supersedes `PREREG_E10_nlp_architecture_causal.md` (v1, locked
> `b946f6901df0d3cac2fdc6e53f747bfab0c8131c891c5c453b436004cb646dea`,
> 2026-08-23T00:20:27Z).** v1's intervention bases were selected by `q1`, a scale-invariant
> measure of internal spectral concentration that carries no magnitude information and is not
> a high-gain score. v2 replaces those bases with ones selected by **exact ‖U_k‖_F relative to
> layer median**. Full audit: `experiments/E10_nlp_architecture_causal/PROTOCOL_CORRECTION_01.md`.
>
> **No causal response was measured under v1** — this is a correction to an unused design, not
> a post-hoc revision after seeing data. The v1 file is left byte-identical so its lock
> continues to verify; both entries remain in `LOCKS.jsonl`.
>
> Changed in v2: OLMo row order, Phi-3 row order, MosaicBERT basis (5/10 rows), ModernBERT
> basis (6/10 rows). Unchanged: all K=1 decoder rows, all top-1 rows, all control rows,
> endpoints, corpora, masking, alpha/epsilon grids, statistics, and the Step D4 thresholds.

**Lock before any causal intervention response is measured, on either arm:**
`python3 src/prereg_lock.py lock docs/prereg/PREREG_E10_nlp_architecture_causal.md`

**Verify the lock before running any measurement:**
`python3 src/prereg_lock.py verify docs/prereg/PREREG_E10_nlp_architecture_causal.md`

This prereg incorporates the 2026-08-22 correction to the original E10 design
(`e10_prompt_correction.md`, repo root) — the decoder arm below is the corrected
structural-top-K / singleton-spectrum design, **not** the original single-row dose-response
design. The encoder arm is unchanged from the original E10 design.

---

## Disclosure

E10 replicates, on already-characterized NLP models, the structural pattern found in E7/E8:
decoders tend to be near-rank-1 (high `q1`), encoders tend to be less collapsed. This
prereg asks whether a *causal* phenotype (not just the structural `q1` geometry) tracks the
same encoder/decoder axis, using GENERator's single-pathway assay as the decoder analog and
DNABERT-2's F0-F3 mechanistic tomography as the encoder analog — as **architecture-aligned
but non-identical protocols**, per `e10_prompt.md`'s explicit framing. No training-mechanism
claim, no universal encoder/decoder causal law. Association only, in this tested panel of 5
decoders and 2 encoders.

## Seven models, exact revisions

| Model | Checkpoint | Revision | Source |
|---|---|---|---|
| Llama | `huggyllama/llama-7b` | unpinned (`main` at fetch time) — disclosed gap, not retroactively fixable for E1/E7's historical numbers | `MODEL_AND_BASIS_AUDIT.md` §1 |
| Mistral | `mistralai/Mistral-7B-v0.1` | unpinned | `MODEL_AND_BASIS_AUDIT.md` §2 |
| OLMo | `allenai/OLMo-7B-0724-hf` | unpinned | `MODEL_AND_BASIS_AUDIT.md` §3 |
| Phi-3 | `microsoft/Phi-3-mini-4k-instruct` | `f39ac1d28e925b323eae81227eaba4464caced4e` | `MODEL_AND_BASIS_AUDIT.md` §4 |
| Qwen2.5 | `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` | `MODEL_AND_BASIS_AUDIT.md` §5 |
| MosaicBERT | `mosaicml/mosaic-bert-base` | `c89bbadc24278928f22bcdd7de6b61a5a2d08553` (confirm still `main` HEAD before treating as pin) | `MODEL_AND_BASIS_AUDIT.md` §6 |
| ModernBERT | `answerdotai/ModernBERT-base` | `8949b909ec900327062f0ebf497f51aef5e6f0c8` (same caveat) | `MODEL_AND_BASIS_AUDIT.md` §7 |

## Decoder arm — frozen rows (Step D1, see `DECODER_INTERVENTION_FREEZE.md` for full detail)

| Model | K | Rows (layer/row), rank order | Control layer / rows |
|---|---:|---|---|
| Llama | 1 | L2/r3968 | L2: [892, 2034, 2379, 3729, 3751] |
| Mistral | 1 | L1/r2070 | L1: [190, 305, 746, 1912, 2687] |
| OLMo | 4 | L24/r269, **L7/r269, L1/r269**, L2/r269 | L24: [292, 437, 2877, 2908, 3160] |
| Phi-3 | 6 | L2/r525, **L2/r1693, L2/r1113, L4/r525, L4/r1113, L4/r1693** | L2: [356, 2345, 2759, 2830, 2983] |
| Qwen2.5 | 1 | L26/r458 | L26: [62, 1257, 2926, 3419, 3553] |

Ranked by **exact ‖U_k‖_F relative to layer median** (`uk_frobenius.layer_report`'s
`max_over_median` convention, using `spectral_lib`'s exact norm — cross terms retained, not
`uk_frobenius.py`'s diagonal approximation). The three K=1 models have a single pre-existing
candidate each and were never ranked. `q1` is recorded as an annotation only and is **not** a
selection criterion. No row chosen, ordered, or dropped by looking at any causal response.

## Encoder arm — frozen basis (Phase 2, see `ENCODER_BASIS_FREEZE.md` for full detail)

Both encoders are Case B (only one canonical row pre-existed E10). Top-`K=10` by **exact
‖U_k‖_F relative to layer median**, computed across every layer's gated-FFN output row (9,216
rows for MosaicBERT, 16,896 for ModernBERT; weight-only, no forward pass, no new metric).
Both models' E8 canonical detector rows land inside the top-10 (MosaicBERT L9/r287 at rank 9;
ModernBERT L15/r251 at rank **1**) — no stop/audit condition triggered, `n_E = 10` for both.

**MosaicBERT basis** (ratio to layer median; exact ‖U_k‖_F in parens): L0/r287 (12.29x, 39.20),
L0/r444 (6.61x, 21.09), L1/r287 (6.45x, 23.11), L11/r287 (5.91x, 28.90), L0/r302 (5.84x, 18.62),
L5/r287 (5.76x, 23.15), L3/r287 (5.67x, 22.43), L10/r666 (5.27x, 22.38),
**L9/r287 (5.05x, 20.00, E8 canonical)**, L10/r79 (4.81x, 20.42).

**ModernBERT basis**: **L15/r251 (23.55x, 836.05, E8 canonical)**, L0/r251 (8.52x, 278.32),
L11/r251 (8.03x, 321.67), L0/r67 (7.52x, 245.71), L9/r251 (7.52x, 304.09), L4/r251 (5.96x, 226.15),
L1/r251 (5.08x, 184.83), L5/r251 (4.86x, 187.41), L10/r251 (4.80x, 192.40), L1/r67 (4.58x, 166.74).

Robustness: ranking by *global* exact ‖U_k‖_F instead gives the identical set for MosaicBERT
(10/10) and 9/10 for ModernBERT. `K=10` retained for both — no weight-space eligibility
threshold exists in this project, and importing E7/E8's 5.0x *activation*-detector threshold to
a weight-space ratio would invent a new structural criterion (forbidden by E10's hard rules).
Disclosed, not acted on: under such an import MosaicBERT would have 9 eligible rows, ModernBERT 7.

## Corpora / sequence IDs

Both arms use `wikitext`, `wikitext-2-raw-v1`, `test` split — see `ENDPOINTS.md` for full
window-construction, masking, and batching parameters (decoder: `N=100` windows / 512 tokens;
encoder: `N=256` windows / 512 tokens / `mask_prob=0.15` / mask seed 42, reusing E9's
`tomography_lib.build_fixed_batches` unmodified).

## Intervention semantics

Weight-space row scale on the down-projection-equivalent module's identified output row,
generalizing E9's frozen convention (`INTERVENTION_BASIS.md`) to all 7 models:

```
alpha = 1 - epsilon * a
mod.weight.data[row, :] = saved_row * alpha
```

- Decoder arm (Step D2): `alpha in {1.0, 0.0}` for every frozen row and every control row
  (baseline + full ablation). `alpha=0.5` is retained **only** for each model's rank-1
  (largest exact ‖U_k‖_F relative to layer median) row, since that is already part of the
  row-scale implementation at no extra engineering cost — not run for lower-ranked rows or
  controls.
- Encoder arm (Arm B): `alpha_i = 1 - epsilon * a_i`, `epsilon in {0.5, 1.0}`, `a_i in {0,1}`
  per mask, identical to E9.

## Alpha / epsilon values

Decoder: `{1.0, 0.0}` (+`0.5` for rank-1 rows only). Encoder: `epsilon in {0.5, 1.0}`.

## Mask generation (encoder arm only)

`n_E = 10` for both MosaicBERT and ModernBERT — identical basis size to E9's DNABERT-2 basis,
so E9's mask-generation logic and pool sizes (`generate_masks.py`) are reused **unmodified**,
per-model (each model gets its own independent mask draw over its own 10-row basis, same
seed/procedure, not a shared mask set): singletons (all 10, exhaustive), fit
(`rho_0.25:22, rho_0.5:34, rho_0.75:22`), calibration (`6,8,6`), held_out (`6,8,6`), densities
`rho in {0.25, 0.5, 0.75}` (`k in {2,3}, {5}, {7,8}` active components of 10). Held-out masks
are generated but never inspected until the primary F0-F3 decision is frozen.

## Train/calibration/test split

Held-out masks are never touched during lambda selection, model selection, pair selection, or
threshold tuning — identical discipline to E9.

## Primary endpoints

- Decoder: mean per-token causal-LM NLL over the frozen 100-window WikiText-2 set
  (`ENDPOINTS.md`).
- Encoder: mean MLM loss over masked positions on the frozen 256-window WikiText-2 set
  (`ENDPOINTS.md`).

## Control selection

Decoder: 5 same-layer random rows per model, seeded (`SeedSequence(42).spawn(5)`, panel order
Llama/Mistral/OLMo/Phi-3/Qwen2.5), frozen before any response (`e10_decoder_control_rows.json`).
Encoder: reuses E8's per-candidate 5-control-row convention if a same-layer control analysis is
run (optional per `e10_prompt.md`'s "Optional same-layer ordinary controls for encoders").

## Statistics

- Decoder Step D3 (causal-concentration statistic, preregistered before measurement):
  `C1 = |ΔL_top1| / sum_i |ΔL_i|` (top-1 share of total absolute singleton effect across the
  model's frozen row set); `C2 = (|ΔL_top1| + |ΔL_top2|) / sum_i |ΔL_i|` reported alongside.
  Absolute effects only for C1/C2; signed effects reported separately, never hidden behind the
  concentration statistic.
- Decoder effect normalization: raw `ΔNLL`, relative percent change from baseline, and
  `target_effect / median(|random_control_effect| + epsilon_small)` (control-normalized),
  `epsilon_small = 1e-6`, fixed before measurement.
- Encoder: R², MAE, RMSE, normalized MAE, F2→F3 relative MAE improvement, bootstrap CI over
  the natural independent unit (context window) — identical to E9's own statistics.
- Bootstrap: resample over sequence/context units, preserving mask clustering where
  applicable — identical to E9.
- Architecture-level synthesis: primary unit is the MODEL (n=5 decoders, n=2 encoders) — no
  p-values treating hundreds of masks/windows as model-level replicates.

## Architecture-level decision rules

**Decoder, per model (Step D4) — thresholds confirmed by the author before lock:**
- **SINGLE_COMPONENT_DOMINANT** — top-1 row's `|ΔL|` is overwhelmingly larger than every other
  frozen row's and every control's `|ΔL|`, operationalized as: `C1 > 0.5` (the single largest
  row already accounts for more than half of the total absolute singleton effect across the
  frozen set) AND the top row's control-normalized effect (`target_effect /
  median(|random_control_effect| + 1e-6)`) `> 3.0`. STOP — do not run decoder tomography for
  this model merely for symmetry.
- **MULTI_COMPONENT_CANDIDATE** — the top row fails the SINGLE_COMPONENT_DOMINANT test above
  (either `C1 <= 0.5` or control-normalized effect `<= 3.0`), but at least 2 rows in the frozen
  set individually clear the control-normalized threshold (`> 3.0`). Flag as
  tomography-eligible; do not automatically run F0-F3 — only if compute/time is acceptable and
  this prereg is amended (a fresh, separate lock) to explicitly authorize it before any such
  run.
- **STRUCTURAL_CAUSAL_DISSOCIATION** — the top structural row's control-normalized effect does
  not exceed 3.0 (i.e. fewer than 2 rows clear the threshold, including the top row itself).
  Report the dissociation; do not search other rows post hoc.

**K=1 edge case (Llama, Mistral, Qwen2.5):** `C1 = 1.0` trivially (only one row exists), so the
decision collapses to whether that single row's control-normalized effect exceeds 3.0:
**SINGLE_COMPONENT_DOMINANT** if yes, **STRUCTURAL_CAUSAL_DISSOCIATION** if no.
MULTI_COMPONENT_CANDIDATE cannot occur for a K=1 model — a null result at K=1 is a dissociation
finding, not evidence of "insufficient" causal spread, and is reported as such, not treated as
inconclusive.

**Encoder (unchanged from E9's own adequacy rule, Phase 6):** PAIR_TERMS_REQUIRED if F0/F1/F2
fail or remain materially worse, F3 improves held-out MAE by the preregistered practical
threshold, the bootstrap CI on that improvement excludes zero, and the lifted design is
identifiable. F3 need not reach perfect adequacy.

## Stop rules

- No new alpha/epsilon values, dose points, corpora, or endpoints added after any response is
  observed, on either arm.
- No row added, dropped, or reordered after a causal response is measured.
- Decoder tomography (F0-F3 on any decoder) is NOT run under this lock — it requires a
  separate, explicit amendment per Step D4's decision rule above.
- If Branch C/D/E (per `e10_prompt.md`'s predeclared outcome branches) occurs, report the
  negative/null result and stop; do not rescue with new rows, corpora, or metrics.

## What E10 does not establish (carried from `e10_prompt.md`, unchanged)

"All decoders are additive/single-component," "all encoders are interactional," "q1 causes
causal concentration/redundancy," "architecture determines mechanism universally" — none of
these may be written regardless of outcome.
