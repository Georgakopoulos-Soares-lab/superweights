# PREREG — E10b: Phi-3 mechanistic tomography

**Lock before any multi-row response is measured:**
`python3 src/prereg_lock.py lock docs/prereg/PREREG_E10b_phi3_tomography.md`

**Verify before and after measurement:**
`python3 src/prereg_lock.py verify --all`

## Why this experiment exists

E10 v2 (locked, commit `626cddd`) resolved 4 of 5 decoders to `SINGLE_COMPONENT_DOMINANT`
and one — Phi-3 — to `MULTI_COMPONENT_CANDIDATE` (C1=0.333; ranks 1 and 2 of its 6 frozen
rows show nearly tied singleton effects, +4.62% and +4.41%). Per the E10 v2 prereg's own
Step D4 rule, this is the only decoder for which multi-component tomography is licensed; no
other decoder tomography is authorized by this document or by E10. No E10 result is modified,
rerun, or reinterpreted here (`PHI3_BASIS_AUDIT.md`).

## Frozen inputs (unchanged from E10 v2 — cited, not recomputed)

- Checkpoint: `microsoft/Phi-3-mini-4k-instruct` @ `f39ac1d28e925b323eae81227eaba4464caced4e`
- Module path / hook: `model.layers.{i}.mlp.down_proj`, row-scale
  `alpha = 1 - epsilon*a`, via `_resolve_module`/`e10_lib.masked` (identical mechanism to E10)
- Basis, `K=6`, ranked by exact ‖U_k‖_F / layer median (**not** `q1`):
  `(2,525)` rank1, `(2,1693)` rank2, `(2,1113)` rank3, `(4,525)` rank4, `(4,1113)` rank5,
  `(4,1693)` rank6 — order and membership frozen, not reselected
- Intrinsic endpoint: mean per-token causal-LM NLL (secondary: perplexity, logit KL,
  next-token entropy) — identical to E10 Arm A, not changed
- Corpus / contexts: WikiText-2-raw-v1 test split, **the same 100 windows** used for Phi-3's
  E10 singleton measurement — reproduced deterministically by calling
  `e10_lib.build_windows(tokenizer, n_windows=100, max_tokens=512, seed=42)` again (pure
  function of tokenizer+seed+counts, so this is the identical evaluation distribution, not a
  new one)
- Tokenization: Phi-3's own tokenizer, 512-token truncation, batch_size=8 (same as E10)
- dtype: float32 (same as E10)

## What is new in E10b (not previously measured)

E10's Phi-3 run only measured each row's *individual* full ablation (plus α=0.5 for rank-1
alone) — it never measured **combinations** of rows, and never measured singleton effects at
`epsilon=0.5` for ranks 2-6. E10b measures the response under the mask design below, at both
epsilons, for every mask (including fresh singleton measurements at both epsilons, needed for
F0/F1 — this is "independently measured" in the sense of being measured under this
experiment's own controlled mask protocol, not reused from E10's differently-scoped run).

## Intervention semantics

`alpha_i = 1 - epsilon*a_i`, `epsilon in {0.5, 1.0}`, `a_i in {0,1}` per mask — identical to
E9/E10's frozen ontology. Multiple rows in a mask are scaled simultaneously
(`e10_lib.masked(model, pattern, coords, alphas)`, already generic to multi-row lists).

## Mask-generation scheme (Phase 2/3, frozen after a pre-measurement design dry run)

E9's `generate_masks.py` disjoint-sampling algorithm, reused by direct import
(`mask_lib_e10b.py`), re-parameterized for `N=6` because K=6 has far fewer available masks
than K=10 (`2^6=64` vs `2^10=1024`) — E9's exact counts do not transfer and were re-derived
here, **before any response was measured**, purely from mask combinatorics:

- `k=1` (all 6 one-hot masks) reserved exclusively for the singleton pool, excluded from
  every density bucket (matching E9's convention).
- Densities: `rho=0.25 -> k=2`, `rho=0.5 -> k=3` (exact), `rho=0.75 -> k in {4,5}`
  (mean 4.5/6 = 0.75 exactly).
- Pool counts (uses all 56 available non-singleton, non-trivial masks — `15+20+15+6` across
  `k=2..5` — no slack left unused): `fit = {2:9, 3:14, 4/5:13}` (36 total),
  `calibration = {2:3, 3:3, 4/5:4}` (10 total), `held_out = {2:3, 3:3, 4/5:4}` (10 total).
- Seed: `20260823` (`random.Random`, E9's `sample_disjoint`).

**Design validity, verified before any response was measured** (Phase 3 dry run,
`mask_lib_e10b.build_design()` + `generate_masks.py::check_design`, no model loaded):
additive design rank `6/6` (full), lifted design rank `21/21` (full — `K + K*(K-1)/2 = 6+15`),
`cond(XtX)_lifted = 588.8`, `max|corr| = 0.672`, no near-duplicate columns, `14/15` pairs
covered in the held-out pool (all 15 covered in fit). No regeneration was needed — the first
dry-run configuration already passed; this is disclosed as the actual sequence of events, not
retroactively cleaned up.

Held-out masks are not touched for lambda selection, pair selection, hyperparameter tuning,
threshold tuning, or basis changes.

## Observer ladder (Phase 5, identical fitting discipline to E9/E10 — imported, not re-derived)

- `F0`: `y_hat(a) = sum_i a_i x_i`, `x_i` = this run's own measured singleton effect (one-hot
  mask response) at the matching epsilon.
- `F1`: `y_hat(a) = g * F0(a)`, `g` fit on calibration only.
- `F2`: `y_hat(a) = sum_i beta_i a_i`, ridge, lambda selected on calibration
  (`LAMBDA_GRID` from `run_fit_observers.py`, unchanged: `[1e-3,...,1000.0]`).
- `F3`: `y_hat(a) = sum_i beta_i a_i + sum_{i<j} Gamma_ij a_i a_j`, ridge, lambda on
  calibration.
- Primary comparison: F2 vs. F3 on untouched held-out masks.

## Metrics (Phase 6)

R², MAE, RMSE, normalized MAE (MAE / held-out target range), F2->F3 absolute and relative MAE
improvement, bootstrap CI on the improvement (5000 resamples over **context batches**, paired
with the baseline draw — `bootstrap_mae_diff`, imported from `run_fit_observers.py`
unmodified). Masks are not treated as independent model-level replicates.

## Decision rule (Phase 7 — identical thresholds to E9/E10, not re-derived)

- `PAIR_TERMS_REQUIRED` iff F2->F3 relative MAE improvement `>= 10%` AND the bootstrap CI on
  the improvement excludes zero AND the lifted design is identifiable (already verified above).
- `ADDITIVE_SUFFICIENT` iff F2 already meets `R2 >= 0.90` and `normalized_MAE <= 0.10`
  (E9/E10's adequacy thresholds) and F3 does not clear the pair-required bar.
- `MIXED_OR_UNRESOLVED` otherwise (F3 improves somewhat but doesn't clear the bar, or neither
  family is adequate).

No threshold is changed after seeing results.

## Retrospective pair inspection (Phase 8 — after the decision only, descriptive, no refit)

Rank pairs by `|Gamma|`; check whether the strongest pair involves the two largest E10
singleton-effect rows (ranks 1 and 2: `(2,525)`,`(2,1693)`); check same-layer vs. cross-layer
enrichment. No critical pair is known in advance for Phi-3; none will be manufactured if the
evidence doesn't support one.

## Interpretation branches (fixed before measurement)

- **Branch A (additive Phi-3)**: F2 adequate, F3 adds little →
  "Phi-3 is a multi-component decoder in terms of singleton causal contribution, but its
  combined finite response remains largely additive."
- **Branch B (pair-interactional Phi-3)**: F3 materially improves →
  "interaction-dependent high-gain causal organization is not encoder-specific."
- **Branch C (unresolved)**: neither family adequate → report and stop; do not add triples,
  HVPs, SAEs, PCA/learned directions, or any new representation.

## Stop rule

Phi-3 only. No other decoder tomography is authorized by this lock. No rerun, reselection, or
threshold change after any response is observed. If Branch C occurs, report and stop — no
higher-order model is added.
