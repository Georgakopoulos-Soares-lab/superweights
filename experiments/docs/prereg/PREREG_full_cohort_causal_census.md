# PREREG — Full-Cohort Causal Census of High-Gain FFN Structures (E13)

**Lock before any causal measurement:**
`python3 paper-salvage/src/prereg_lock.py lock paper-salvage/docs/prereg/PREREG_full_cohort_causal_census.md`

**Verify before and after measurement:**
`python3 paper-salvage/src/prereg_lock.py verify --all`

This protocol covers the full task specified in `part_prompt.md` (repo root) — Stages 0
through 4 — so that the whole analysis plan is frozen before any causal number is inspected,
per this repo's standing discipline. **Execution in this session is bounded to Stage 0
only** (the reproduction gate). Stages 1-4 (the full 23-model causal census, structure-vs-
function correlations, interaction tomography, and cohort-level synthesis) are preregistered
here but explicitly NOT run in this pass — part_prompt.md's own Stage 0 gate ("If there is a
material discrepancy, STOP and diagnose it before running the cohort") and its Compute
section ("Before launching the full batch, estimate GPU-hours/disk usage... then proceed
unless there is a genuine blocker") both require an explicit go/no-go checkpoint between
Stage 0 and Stage 1, which this document treats as binding.

## Question

part_prompt.md's central manuscript question: does the recurrent, locally exceptional
high-gain structural phenotype (established structurally across the full 23-model E11
cohort) correspond to a shared causal organization, or does causal function/interaction
structure vary substantially across models? This experiment is the final broad causal test
for Part 2 of the manuscript — not an exploratory pass, and not a search for a better story
than whatever the full cohort actually shows.

## Frozen inputs (reused, not recomputed)

Candidate selection, the causal-intervention mechanism, and most evaluation utilities are
inherited unmodified from E7-E12. Nothing here re-derives or re-tunes any of the following:

- **Structural candidates**: `results/E11/scale_ladder.csv`, one row per model, for exactly
  the 23-model cohort part_prompt.md specifies. Candidate (layer, row) pairs are fixed by
  E7/E8/E10/E11's own activation-based detector (`DownProjRecorder`, ratio ≥ 5.0) and are
  **not** touched, re-selected, or re-run by this protocol. Layer-relative `‖U_k‖_F`
  (`layer_median_frob_norm` / `candidate_frob_norm_ratio_to_layer_median`) for the 11
  "cited" rows in that CSV not originally computed by E11 itself is backfilled by
  `E13_full_cohort_causal_census/backfill_layer_median.py` as a separate, pre-causal,
  weight-only step (see that script's docstring for exact per-model provenance:
  `results/E13/layer_median_backfill.json`). This backfill is a pure Frobenius-norm
  computation on already-frozen candidate coordinates, not a new detection pass, and is
  exempt from this lock (it produces no causal number).
- **Causal intervention mechanism**: `alphas_for_mask(a, epsilon) = 1 - epsilon*a_i`
  (`E10_nlp_architecture_causal/e10_lib.py:64-66`, identically defined in
  `E9_mechanistic_tomography/tomography_lib.py:81-83` and wrapped again in
  `E12_generator_degradation_control/e12_lib.py:153/186`). For a singleton candidate
  (`a=[1]`): `epsilon=0.5` → `alpha=0.5` (partial suppression), `epsilon=1.0` → `alpha=0.0`
  (full ablation). Row save/restore via `scripts/evaluation/run_gue_ablation.py`'s
  `_resolve_module`/`_save_row`/`_restore_row`, wrapped by `scale_rows`/`restore_rows`/
  `masked` (e10_lib.py L35-61) or `_scale_rows`/`_restore_rows`/`with_mask`
  (tomography_lib.py L49-78). No new ablation mechanism is introduced anywhere in this
  experiment.
- **Text decoder causal-LM NLL**: `e10_lib.py::causal_lm_batch` (L103-125) +
  `build_windows` (L71-93; frozen WikiText-2-raw-v1 test split, seed 42).
- **Text encoder MLM loss**: `e10_lib.py::mlm_loss_per_batch` (L155-166) +
  `build_fixed_mlm_batches` (L130-152; mask_prob=0.15, seed 42).
- **Genomic decoder causal-LM NLL (GENERator family)**: `e12_lib.py::window_nll` (L328-340)
  + `damage` (L343-350); k-mer-tokenizer-specific `_prepare_sequence` (L143-148, trims to a
  multiple of 6bp, prepends BOS).
- **Genomic encoder MLM loss (DNABERT-2)**:
  `tomography_lib.py::load_dnabert2_pretrained`/`read_fasta_windows_mlm`/
  `build_fixed_batches`/`mlm_loss_per_batch`/`mlm_loss` (L88/107/132/163/181).
- **Tomography / observer ladder (F0-F3)**: `E9_mechanistic_tomography/run_fit_observers.py`
  (`N_BOOT=5000`, `ridge_fit`/`metrics`/`bootstrap_mae_diff`, `fit_epsilon` building
  F0/F1/F2/F3), imported directly by `E10_nlp_architecture_causal/fit_encoder_observers.py`
  and `E10b_phi3_tomography/fit_phi3_observers.py`. Mask design in
  `E10b_phi3_tomography/mask_lib_e10b.py` (`build_design`/`design_matrix_additive`/
  `design_matrix_lifted`/`cooccurrence`). Reused unmodified for Stage 3.

## What is new in this experiment (E13)

- Two small evaluation-utility adapters extending the above to models that never had a
  causal pipeline before (structural work on them existed; causal work did not):
  - **GenomeOcean-4B causal-LM NLL**: `e12_lib.py::_prepare_sequence` is GENERator's
    fixed-6bp-k-mer-tokenizer-specific trim step and does not apply to GenomeOcean, which
    uses a character/BPE tokenizer with no fixed-length alignment requirement
    (`configs/genomeocean.yaml`, `models/genomeocean_wrapper.py`: `tokenizer(sequence,
    return_tensors="pt", add_special_tokens=False)`, no trim, no forced BOS beyond whatever
    `add_special_tokens` would normally add). `E13_full_cohort_causal_census/
    genomic_decoder_lib.py::prepare_sequence_genomeocean` reuses `window_nll`'s teacher-
    forced-loss structure verbatim, swapping only the sequence-prep step.
  - **NTv3 masked-LM loss**: no NTv3 equivalent of `tomography_lib`'s DNABERT-2 MLM stack
    exists anywhere in the repo. `E13_full_cohort_causal_census/genomic_encoder_lib.py`
    reuses `build_fixed_batches`/`mlm_loss_per_batch`/`mlm_loss` verbatim (they are already
    architecture-agnostic apart from tokenizer/mask/pad/special-token ids) with NTv3's own
    tokenizer (`AutoTokenizer.from_pretrained("InstaDeepAI/NTv3_650M_pre")`) and
    `AutoModelForMaskedLM` (confirmed loadable via this class in
    `E7_exact_dimensionality/run_legacy_reanalysis.py`'s `ntv3` loader, pinned
    `code_revision=0ecff3637f0d3ba5b686d1095083218157c2ca34`) in place of DNABERT-2's. Same
    masking convention (mask_prob=0.15), same fixed-seed batch construction (seed 42), same
    row-scaling intervention (`_scale_rows`/`with_mask` from `tomography_lib.py`, applied to
    NTv3's `core.transformer_blocks[i].fc2` down-projection per `uk_frobenius.adapter_ntv3`'s
    already-validated module path). This is a small adapter, not new methodology.
- **A new, wider same-layer control panel.** E11's own `SeedSequence(42).spawn(12)` panel
  covers only the 12 models E11 itself freshly measured; E10's `spawn(5)`/`spawn(2)` panels
  cover only their own narrower NLP/encoder subsets. None of the three covers all 23 models
  in one fixed ordering. This protocol declares a new `spawn(23)` panel (below) that does.

## Fixed cohort and panel ordering (frozen)

Exactly the 23 models part_prompt.md specifies (verified name-for-name against
`results/E11/scale_ladder.csv`, which already contains one row per model for this exact
cohort). The panel ordering below is frozen for two purposes: (a) it is the fixed
`panel_index` used to draw this experiment's `SeedSequence(42).spawn(23)` same-layer control
rows for every model, and (b) it is the canonical row order for every cohort-level table
this experiment produces (`causal_census.csv`, `cohort_summary.json`, etc). It matches
`results/E11/scale_ladder.csv`'s own row order.

```
 0  Llama-7B                    12  MosaicBERT
 1  Mistral-7B                  13  ModernBERT-base
 2  OLMo-7B-0724-hf             14  ModernBERT-large
 3  Phi-3-mini-4k-instruct      15  EuroBERT-210M
 4  Qwen2.5-7B                  16  EuroBERT-610M
 5  MosaicBERT (dup guard—see*) 17  EuroBERT-2.1B
 6  Qwen2.5-0.5B                18  GENERator-EUK-3B
 7  Qwen2.5-1.5B                19  GENERator-PROK-1.2B
 8  Qwen2.5-3B                  20  GENERator-PROK-3B
 9  SmolLM2-135M                21  GenomeOcean-4B
10  SmolLM2-360M                22  DNABERT-2
11  SmolLM2-1.7B                23  NTv3
```

`*` — the table above is a 2-column display artifact only; the authoritative, unambiguous
ordering (24 lines, `panel_index` 0-22, one model each) is the literal Python list in
`E13_full_cohort_causal_census/panel.py::PANEL_ORDER`, generated directly from
`results/E11/scale_ladder.csv` row order at protocol-authoring time and copied verbatim
into that module before this file is locked, so there is exactly one source of truth and no
possibility of the display table above drifting from the code. `len(PANEL_ORDER) == 23`.
Every new same-layer control draw in this experiment, for every one of the 23 models, is:

```python
seed_seqs = np.random.SeedSequence(42).spawn(23)
rng = np.random.default_rng(seed_seqs[PANEL_ORDER.index(model_key)])
pool = [r for r in range(d_model) if r not in {candidate rows for this model/layer}]
control_rows = sorted(rng.choice(pool, size=5, replace=False).tolist())
```

**Note on pre-existing control rows.** Five of the Stage-0 models already have causal
control rows recorded from earlier, narrower experiments, drawn under different
`SeedSequence(42)` child indices (E10's `spawn(5)`/`spawn(2)` panels): Llama-7B
`[892,2034,2379,3729,3751]` at layer 2, Mistral-7B/OLMo-7B-0724-hf/Phi-3-mini/Qwen2.5-7B
similarly (see `results/e10_decoder_spectrum_*.json`), and MosaicBERT/ModernBERT-base
(`results/e8_detection_*.json`). **This experiment draws a fresh set under the new
`spawn(23)` panel for every model, including these five, and does not reuse the old
identities.** The point of Stage 0 is to confirm the new common pipeline reproduces the
*direction and approximate magnitude* of the old causal effect on the *candidate* row — which
does not depend on which control rows were sampled — not to reproduce old control-row
identities exactly. Any difference between old and new control-row identities for these five
models is expected and is not itself evidence of a discrepancy; only the candidate-row
baseline/perturbed-loss numbers are compared against the old locked values in Stage 0's
decision rule below.

## Scope: Stage 0 only, this session

Stage 0 (reproduction gate) is executed by this session. Stages 1-4 are preregistered in
full below but are **not** run until a separate, explicit go-ahead following Stage 0's own
decision rule and the compute estimate this session produces after Stage 0 completes.

### Stage 0 model set (from part_prompt.md, "ideally")

- Llama-7B (strong singleton decoder)
- Mistral-7B
- OLMo-7B-0724-hf (known structural/causal rank dissociation case — expected to reproduce
  as a dissociation, not treated as a failure if it does)
- Phi-3-mini-4k-instruct
- One encoder tomography case: ModernBERT-base (MosaicBERT run alongside if time permits,
  not required for the gate decision)
- DNABERT-2

### Stage 0 old locked numbers being reproduced (cited, not recomputed here)

- Llama-7B: `results/e10_decoder_spectrum_llama.json` — baseline NLL 2.2453, candidate
  L2/r3968, alpha=0.5 → nll 2.3318 (+3.86%), alpha=0.0 → nll 8.999 (+300.8%).
- Mistral-7B: `results/e10_decoder_spectrum_mistral.json`, same schema, candidate L1/r2070.
- OLMo-7B-0724-hf: `results/e10_decoder_spectrum_olmo.json`, candidate L1/r269 (primary),
  with rows 269 also recorded at L2/L7/L24 — the dissociation case.
- Phi-3-mini: `results/e10_decoder_spectrum_phi3.json` +
  `results/e10b_phi3_fit_results.json` + `results/e10b_phi3_tomography_responses.json`.
- Encoder case: `results/e10_encoder_fit_results.json` (keys `"mosaicbert"`,
  `"modernbert"`), raw responses `results/e10_encoder_responses_{mosaicbert,modernbert}.json`.
- DNABERT-2: fit results in `E9_mechanistic_tomography/RESULTS.md`, raw responses
  `paper-salvage/experiments/E9_mechanistic_tomography/dnabert2_mask_responses.json`.

## Intervention scales (frozen, no others added after seeing results)

Exactly two, throughout Stage 0 and every later stage:

- partial suppression: `epsilon = 0.5`
- full ablation: `epsilon = 1.0`

`alpha = alphas_for_mask([1], epsilon)` for a singleton candidate row; the identical
`alphas_for_mask` function is applied per-row to the 5 same-layer controls (each control
scored independently, `a=[1]` for that one row, not jointly masked with the candidate or
with each other).

## Endpoints (frozen)

Each model's own native LM objective, exactly as in part_prompt.md and as already
implemented by the reused code above:

- **Text decoders**: teacher-forced causal-LM NLL, WikiText-2-raw-v1 test, seed 42
  (`e10_lib.build_windows`/`causal_lm_batch`).
- **Text encoders**: masked-LM loss, fixed mask locations, mask_prob=0.15, seed 42
  (`e10_lib.build_fixed_mlm_batches`/`mlm_loss_per_batch`).
- **Genomic decoders**: teacher-forced nucleotide causal-LM NLL
  (`e12_lib.window_nll`/`damage` for GENERator family; the new
  `genomic_decoder_lib.py` adapter, same `window_nll` logic with GenomeOcean's own
  sequence-prep, for GenomeOcean-4B).
- **Genomic encoders**: masked-nucleotide LM loss (`tomography_lib.py` stack for DNABERT-2;
  the new `genomic_encoder_lib.py` adapter, same `build_fixed_batches`/`mlm_loss_per_batch`
  logic with NTv3's own tokenizer, for NTv3).

Primary functional effect, uniformly:

```
relative_loss_change = (perturbed_loss - baseline_loss) / baseline_loss
```

Raw loss is reported alongside but never compared in magnitude across architectures or
objectives (different tokenizers/objectives are not on a common scale).

## Observer families (Stage 3 only, not exercised in Stage 0)

Reused exactly from E9/E10/E10b, not re-derived: additive/main-effect model (F0/F1),
calibrated/jointly-fit additive model where already defined, pairwise model (F2), and
whatever third rung E9/E10b already define as F3. Imported via `run_fit_observers.py`'s
`ridge_fit`/`metrics`/`bootstrap_mae_diff`/`fit_epsilon`, exactly as
`fit_encoder_observers.py` and `fit_phi3_observers.py` already do. No new observer family is
introduced.

## Practical adequacy thresholds (Stage 3 only)

Reused from E9/E10/E10b's existing locked criteria (see those docs' own "Practical adequacy
thresholds" sections) — not re-derived or re-tuned by this protocol. If Stage 4 needs a
threshold this repo has not already defined, it is reported as a continuous measurement
rather than a newly invented cutoff, per part_prompt.md's explicit guardrail against
inventing thresholds after seeing the data.

## Resampling unit (frozen)

- Text decoder/encoder endpoints: resample **evaluation windows/batches** (the existing
  `e10_lib` batch structure), matching E9/E10's own resampling unit.
- Genomic decoder/encoder endpoints: resample **evaluation windows** (GENERator/GenomeOcean
  damage windows; NTv3/DNABERT-2 fixed MLM batches), matching E9/E12's own resampling unit.
- Stage 2's secondary (all-candidate) analysis resamples/clusters uncertainty **by model**,
  per part_prompt.md's explicit instruction not to treat multiple candidate rows from one
  model as independent models.
- All bootstraps: 5,000 resamples, percentile 95% CI, `SeedSequence(42)`-derived or `seed=42`
  per the existing `bootstrap_mae_diff`/`bootstrap_gc_diff` convention (paired where the
  comparison is paired, e.g. candidate vs. baseline on the same windows).

## Stage 0 decision rule (frozen, stated before running)

For each Stage-0 model, compare the new pipeline's candidate-row `baseline_loss`,
`relative_loss_change@epsilon=0.5`, and `relative_loss_change@epsilon=1.0` against the old
locked numbers above. Reproduction is judged on **direction and approximate magnitude**, not
exact match (different random control draws, and in some cases a newer/necessarily-repinned
revision per E11's own precedent for previously-unpinned models, mean bit-identical
reproduction is not the bar):

- **Reproduces**: same sign of effect (loss increases under suppression/ablation), and the
  new relative_loss_change is within roughly the same order of magnitude as the old value
  (informally: within ~2x, or both effects are small-and-noisy in the same way) for both
  epsilon values. OLMo reproducing as a dissociation (large effect at a causally-strong
  layer/row that is not the top structural row) counts as reproduction of the known
  phenomenon, not a discrepancy.
- **Material discrepancy**: opposite sign, an order-of-magnitude mismatch, or a qualitative
  change in which row dominates. Triggers an immediate STOP on this protocol's own
  authority, a diagnosis pass (dtype/revision/tokenizer/window-corpus differences checked
  first, per part_prompt.md's "diagnose before running the cohort"), and explicit reporting
  of the discrepancy rather than silent adjustment of either the new or the old number. Do
  not average incompatible old and new measurements.

If all Stage-0 models reproduce, this session's own scope still stops after reporting Stage
0 results and a Stage-1 compute estimate (GPU-hours, disk, per-model special-loading-code
flags) — Stage 1 itself requires a separate go-ahead, per part_prompt.md's Compute section.

## Stages 1-4 (preregistered in full, not executed this session)

Reproduced here verbatim in substance from part_prompt.md so the full analysis plan is
locked before any cohort-level number exists, per this repo's standing discipline (no
analysis choice for Stages 1-4 may be added or changed after Stage 1 data is inspected,
without a new dated protocol revision):

**Stage 1 — full-cohort singleton causal census.** For every structurally detected candidate
in every one of the 23 models: baseline loss; suppress at epsilon=0.5; fully ablate at
epsilon=1.0; same for the 5 seeded same-layer controls (`spawn(23)` panel above). Record
model, domain, architecture type, parameter count, candidate layer/row, q1, exact
`‖U_k‖_F`, layer-relative `‖U_k‖_F`, candidate-vs-control q1 gap where available,
intervention strength, baseline/perturbed/absolute-delta/relative-delta loss, control-row
effects. Bootstrap 95% CIs (windows/batches as the resampling unit).

**Stage 2 — structure vs. functional criticality.** Spearman correlation, `q1` vs.
full-ablation relative loss change and layer-relative `‖U_k‖_F` vs. full-ablation relative
loss change; both a one-primary-candidate-per-model analysis and an all-candidate analysis
with uncertainty clustered/resampled by model. Graphical display includes OLMo as one
cohort observation, not privileged. Report whatever relationship (or absence of one) the
data show, in either direction, without forcing a "structure != function" conclusion.

**Stage 3 — causal interaction census.** Singleton-only models: measure the singleton
causal effect, do not infer interaction order, label interaction complexity
"not identifiable from a one-component structural basis." Models with K≥2 candidates: fit
the existing F0/F1/F2/F3 observer ladder (imported, not re-derived) with
`P = K + K*(K-1)/2` free parameters for the pairwise model; training-mask count set to
substantially exceed P plus an independent held-out mask set, following the existing
DNABERT-2/Phi-3 conventions for train/calibration/held-out mask fixing before responses are
inspected. No third-order terms fit as a post-hoc rescue; if F2 fails badly at full
ablation, report that the second-order observer is inadequate in that regime rather than
extending the model.

**Stage 4 — cohort-level causal organization.** One row per model: model, encoder/decoder,
text/genomic, size, number of structural candidates, primary candidate q1,
candidate-control structural gap, strongest singleton causal effect at epsilon=0.5 and
epsilon=1.0, causal effect relative to same-layer controls, whether interaction analysis was
identifiable, F2/F3 held-out error, relative F2→F3 MAE improvement with bootstrap CI,
whether pair terms materially improved prediction, whether the second-order observer was
adequate at each intervention strength. Reuse existing locked decision-rule terms
(singleton dominance, pair-terms-required, etc.) exactly where already defined; otherwise
report continuously rather than inventing thresholds after seeing the data.

## Predictions (state before running)

Stated before any Stage-0 forward pass in this session:

1. All six Stage-0 models will reproduce direction and approximate magnitude of their old
   locked candidate-row effects, because the intervention mechanism, evaluation code, and
   candidate coordinates are byte-identical to what produced those numbers — the only
   changes are the control-row identities (expected to be irrelevant to the candidate row's
   own measured effect) and, for previously-unpinned models, a now-pinned revision (expected
   to be a negligible source of drift, per E11's own precedent handling this for Llama/
   Mistral/OLMo).
2. OLMo-7B-0724-hf's candidate row (L1/r269) will show a materially smaller full-ablation
   effect than at least one of its other three recorded layers (L2/L7/L24), reproducing the
   known structural/causal dissociation.
3. No Stage-0 model is expected to show a qualitative sign flip or order-of-magnitude
   mismatch; if one occurs, it falsifies prediction 1 for that model and triggers the STOP
   rule above rather than a rewritten prediction.

## Deliverables

**This session (Stage 0 only):**
- `results/E13/stage0_reproduction_gate.csv` — one row per model per epsilon
  (baseline/perturbed/relative_loss_change, new vs. old side by side).
- `results/E13/stage0_raw_responses.json` — raw per-window/per-batch losses backing every
  Stage-0 number, sufficient to recompute any CI without rerunning a model.
- `results/E13/layer_median_backfill.json` and `results/E11/scale_ladder_backfilled.csv` —
  the structural backfill from job 1 (see that script's own docstring for method).
- A Stage-1 compute/disk/special-loading estimate appended to this document's own commit (as
  a dated note, not a rewrite of the frozen sections above) or to a companion
  `STAGE0_REPORT.md` in `E13_full_cohort_causal_census/`.

**Deferred to Stage 1 (not produced this session):** `causal_census.csv`,
`candidate_effects.csv`, `control_effects.csv`, `tomography_results.csv`,
`structure_function_correlations.json`, `cohort_summary.json`, publication-quality PDF
figures (A/B/C per part_prompt.md), raw per-example/per-window responses for every CI,
`FULL_COHORT_CAUSAL_SUMMARY.md`.

## Pre-committed reporting

Whichever Stage-0 outcome occurs — full reproduction, partial reproduction with named
exceptions, or a material discrepancy requiring a STOP — is reported with the same care.
A STOP is not a failure of this protocol; running the full 23-model cohort on top of an
undiagnosed discrepancy in the shared pipeline would be a far larger error. If Stages 1-4
are later run under a revised or extended version of this protocol, that revision is a new,
separately dated section appended below this line, never a silent edit of the sections
above.
