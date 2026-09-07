# PREREG — E9, mechanistic tomography of high-gain FFN causal response

**Lock before running:** `python3 src/prereg_lock.py lock docs/prereg/PREREG_mechanistic_tomography_E9.md`

Written after Phase 0-3 (provenance, basis freeze, mask design/conditioning) and before any
mask-response value is collected. Full provenance in
`experiments/E9_mechanistic_tomography/{PROVENANCE_AND_BASELINES,INTERVENTION_BASIS,BASIS_FREEZE,MASK_DESIGN}.md`.

## Question

Given a declared high-gain intervention basis, what is the simplest causal observer that
predicts unseen finite interventions, for GENERator EUK and DNABERT-2?

## Scope (frozen, `BASIS_FREEZE.md`)

- **DNABERT-2**: full F0-F3 observer-complexity ladder over `n_D=10` basis rows (layers
  3/5/6/7/9, see `BASIS_FREEZE.md` table), using the mask pools in `masks_dnabert2.json`
  (78 fit / 20 calibration / 20 held-out + 10 singletons, deterministic seed `20260822`).
- **GENERator EUK**: one-dimensional dose-response only, on row 2371 (layer 4, primary) and
  row 1522 (layer 4, secondary/consistency check) — no F0-F3 ladder, no pair lifting. Only
  two real pre-existing high-gain candidates exist; a padded 6-12 row basis was rejected as
  fishing-adjacent (`DECISIONS.md` D-025).
- **NTv3**: not run in this prereg. If run at all, only after both decisions above are
  frozen, per Phase 9, as a predeclared negative-response control — its own scope is not
  fixed here.

## Intervention scales (frozen, no others will be added after seeing results)

`alpha_i = 1 - epsilon * a_i`. `epsilon in {0.5, 1.0}` — partial suppression and full
ablation. `alpha=1.0` (untouched) is the reference point for both models.

## Endpoints (frozen)

- **DNABERT-2 primary**: pretrained masked-LM loss (`AutoModelForMaskedLM`, no fine-tuning,
  no task head), token-weighted mean over one fixed mask realization (seed 42,
  `mask_prob=0.15`) applied to 256 held-out hg38 windows (600bp, `random_262kb.bed`),
  identical to `run_pretrained_epistasis.py`'s existing protocol, generalized to arbitrary
  multi-row masks via `tomography_lib.dnabert2_response`. Effect direction: higher loss =
  worse (ablation-positive convention, inverted vs. accuracy-based scripts).
- **DNABERT-2 secondary**: layer-9 residual channel norm (explicit-layer hook convention,
  see `PROVENANCE_AND_BASELINES.md`) — used only for the Phase 7/H6 covariation check, not
  for fitting F0-F3.
- **DNABERT-2 splice/GUE endpoint: out of scope for E9** — required checkpoints and dataset
  are absent on this filesystem (see `PROVENANCE_AND_BASELINES.md`). Not pursued as a
  rescue if the primary endpoint's results are unwelcome.
- **GENERator primary**: mean generated GC fraction, `n_prompts=24`, `prompt_bp=120`,
  `max_new_tokens=64`, `do_sample=True, top_k=50, temperature=1.0`, hg38-window prompts
  (seed 42), identical generation config to `run_sw_steering.py`.
- **GENERator secondary (quality frontier)**: perplexity under the unmodified model,
  dinucleotide KL vs. hg38 background, longest homopolymer run — `run_steering_biological.py`'s
  metric set, at the same alpha values as the primary endpoint.
- **GENERator support endpoint**: none beyond the above (no next-token GC probability proxy
  is introduced).

Primary endpoints will not be changed after seeing results, per Phase 16.

## Observer families (DNABERT-2 only; Phase 5)

- **F0** — singleton additive: `y_hat(a) = sum_i a_i * x_i`, `x_i` = each basis row's
  measured singleton effect (from the `singletons` pool), no fitting.
- **F1** — scalar-calibrated: `y_hat(a) = g * sum_i a_i * x_i`, `g` fit by least squares on
  the `calibration` pool only.
- **F2** — jointly fit additive: `y_hat(a) = sum_i beta_i * a_i`, ridge regression fit on
  `fit`, penalty selected by minimizing RMSE on `calibration`, evaluated on `held_out`.
- **F3** — lifted main+pair: `y_hat(a) = sum_i beta_i a_i + sum_{i<j} Gamma_ij a_i a_j`,
  ridge on the 55-column lifted design (`fit`→train, `calibration`→penalty selection,
  `held_out`→evaluation only). Sparse OMP/lasso is not used — the design's purpose here is
  held-out prediction, not support identification (Phase 5's explicit preference).

Each family is fit and evaluated **separately per epsilon** (0.5 and 1.0) — no pooling
across scales into one regression.

## Practical adequacy thresholds (frozen, Phase 6 — not tuned after seeing E9 values)

- **ADDITIVE ADEQUATE**: held-out R² ≥ 0.90 **and** normalized MAE ≤ 10% of the held-out
  response span.
- **CALIBRATION SUFFICIENT**: F1 meets the above **and** F3 gives no practically meaningful
  improvement (see next bullet's threshold, applied in reverse — F3 does not clear it).
- **PAIR TERMS REQUIRED**: F0/F1/F2 fail adequacy or show structured residual (residual vs.
  mask density or vs. predicted magnitude is visibly non-flat — reported qualitatively, not
  given its own numeric gate) **and** F3 improves held-out MAE by **≥10% relative to F2**
  **and** that improvement's bootstrap CI (see below) excludes zero **and** the lifted
  design was confirmed identifiable in `MASK_DESIGN.md` (it was: 55/55 rank on the fit
  pool).
- If none of the above cleanly resolves (e.g., adequate-but-marginal, or improvement exists
  but CI straddles zero), the result is reported as an explicit intermediate case, not
  forced into the nearest bucket.

## Resampling unit (frozen, Phase 17)

- **DNABERT-2**: the fixed mask realization is shared across all conditions, so the natural
  independent unit for bootstrap CIs is the **held-out window/sequence batch** (16-sequence
  batches over the 256-window pool) — resampling batches with replacement, recomputing
  held-out MAE/R² for F2 vs. F3 each draw. Individual masks are not treated as independent
  replicates of each other (they share the same underlying 256 windows), but they ARE
  independent as *design points* for regression evaluation — the batch-level bootstrap
  captures uncertainty from the finite window sample, not from the (deterministic, one-shot)
  mask realizations.
- **GENERator**: paired-seed resampling over generated sequences within each condition
  (`torch.manual_seed(SEED+i)` per prompt, `i` shared across conditions) — bootstrap over
  the 24 prompts, since each prompt's generation is the independent unit and conditions
  share prompts/seeds by construction.

## Predictions (state before running)

- DNABERT-2 F0 (singleton additive): **predicted to fail** held-out adequacy — the known
  critical pair's individual ablations show little effect while joint ablation is large
  (C-036), so a purely additive singleton sum should systematically underestimate masks
  that include both members. Confidence: 4/5.
- DNABERT-2 F1 (scalar-calibrated): **predicted to still fail** — a uniform gain cannot fix
  a structured (pair-localized) underestimate. Confidence: 3/5.
- DNABERT-2 F2 (freely fit additive): **predicted to still show structured residual**,
  though possibly closer to the adequacy line than F0/F1 since ridge can partially absorb
  some of the pair signal into inflated main-effect coefficients for rows 264/294.
  Confidence: 3/5.
- DNABERT-2 F3 (lifted): **predicted to materially improve held-out prediction over F0-F2**,
  clearing the ≥10% relative-MAE-improvement bar. Confidence: 4/5.
- DNABERT-2 H5 (retrospective): **predicted** that L9/r264+r294 appears among the largest
  stable pair coefficients, but this is checked only after F3 is frozen. Confidence: 3/5.
- GENERator dose-response (row 2371): **predicted** GC moves monotonically with `alpha`
  (matches C-040's direction) and materially more than the random-row control at the same
  `epsilon` values. Whether this rises to "graded steering" per Phase 8's stricter
  definition (reproducible intermediate outcomes + acceptable-quality operating region) is
  **not** predicted in advance — genuinely open. Confidence in monotonic direction only: 4/5.
- GENERator row 1522 (secondary): **predicted** to move GC in the same direction as row
  2371 but with smaller magnitude (matches its lower `out_max` rank). Confidence: 3/5.
- NTv3 (if run): **predicted** functional response stays near the noise floor
  (STRUCTURALLY HIGH-GAIN / FUNCTIONALLY INERT), consistent with the existing C-029/C-038
  null. Confidence: 4/5.

## Pre-committed reporting

Every predicted outcome above is reported regardless of direction, including a clean miss.
The mechanical Branch A-E outcome (Phase 12) is determined solely by the adequacy/threshold
rules above, not by which branch would make the better narrative.

---

_Locked: (filled by prereg_lock.py)_
