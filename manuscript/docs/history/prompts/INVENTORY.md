# INVENTORY.md — orientation for E11 (scale ladder) and E12 (GENERator degradation control)

Written per `e_11_e_12_prompt.md` §0. Read-only investigation, no measurement run yet.

**Key structural fact**: the live, current implementation of everything this task needs lives
under `manuscript/experiments/` and `manuscript/src/`, not under the top-level
`detection/`, `analysis/` directories (those are legacy/pre-salvage code — e.g.
`detection/identify_spikes.py` is a z-score-threshold detector from an earlier design, **not**
the ratio≥5.0 detector this task must reuse). `manuscript/CLAUDE.md` is the binding
operating doc for that tree: thesis is frozen, several claims were retired/relitigated
(D-016, D-024), and there is a hard prereg-lock discipline — no detection/measurement script
runs until `python manuscript/src/prereg_lock.py verify <file>` prints `OK` for its
prereg doc. E11/E12 must follow this same lock convention.

---

## 1. U_k operator, q1, PR_spec, ‖U_k‖_F — import from `spectral_lib.py`

`manuscript/experiments/E7_exact_dimensionality/spectral_lib.py`

- `materialize_Uk(W_gate, W_up, w_down_row, dtype=torch.float64, device=None)` (L36-44) — exact
  `U_k = (w_down_row * W_gate)^T @ W_up`, float64, no approximation.
- `singular_values(Uk)` (L47-51) — `torch.linalg.svdvals`.
- `SpectralMetrics` dataclass (L54-61): `q1, pr_spec, stable_rank, sigma1, frob_norm, n_singular_values`.
  - `q1 = σ1² / Σσj²`; `PR_spec = (Σσj²)² / Σσj⁴`; `stable_rank = 1/q1` (secondary, never primary).
- `spectral_metrics(sigmas)` (L64-72) and the one-call wrapper **`row_spectral_metrics(W_gate, W_up, w_down_row, device=None) -> (SpectralMetrics, sigmas)`** (L75-85).

**Import**: `from spectral_lib import row_spectral_metrics` (already the pattern used by
`E8_encoder_decoder/run_detection_and_spectral.py:28`). `frob_norm` in `SpectralMetrics` is
asserted equal to ‖U_k‖_F exactly, so this one function covers q1/PR_spec/‖U_k‖_F together — no
need to separately reimplement the Frobenius-norm-only screen variant that exists in
`scripts/analysis/run_dnabert2_uk_audit.py::frob_norm_uk` (that one is a memory-saving,
no-SVD, all-rows-at-once screen; not needed here since E11 already knows the candidate row).

## 2. Activation-based candidate detector (ratio ≥ 5.0) — import pattern from `run_phase1_detection.py`

`manuscript/experiments/E7_exact_dimensionality/run_phase1_detection.py`
- `RATIO_THRESHOLD = 5.0` (L32).
- `DownProjRecorder` (L80-104): forward hook on the down-proj-equivalent module; records global
  max |output| per layer and the **median** of per-channel max as the denominator.
- `finalize(...)` (L158-183): `ratio = out_max[i] / median_channel_max[i]`; `best = argmax(out_max)`;
  accept iff `best.ratio >= RATIO_THRESHOLD`. Returns `candidate` / `null_outcome` / `null_reason` / `per_layer`.
- Same logic re-instantiated (not imported cross-file) in
  `E8_encoder_decoder/run_detection_and_spectral.py::finalize()` (L165-221) for the encoder panel.

**Plan**: reuse this exact `DownProjRecorder` + ratio/threshold pattern for every new E11 model
(copy the ~40-line hook+finalize idiom the way E8 did from E7, since it's not currently
exported as a standalone importable helper — E7→E8 precedent is copy-with-attribution, not
`import`).

## 3. Same-layer control sampling (`SeedSequence(42)` + child seed) — copy the inlined idiom

No standalone function exists; the exact 4-line idiom is duplicated verbatim at both call sites
(consistent with repo convention for detector code):

```python
seed_seqs = np.random.SeedSequence(42).spawn(len(PANEL))       # spawn once per full panel
rng = np.random.default_rng(seed_seqs[panel_idx])                # index = this model's position
pool = [r for r in range(d_model) if r not in excluded_rows]
control_rows = sorted(rng.choice(pool, size=5, replace=False).tolist())
```
Sources: `E8_encoder_decoder/run_detection_and_spectral.py` L204-207 (`PANEL_INDEX` dict, N=2);
`E10_nlp_architecture_causal/generate_decoder_controls.py` (full file, N=5, generalized from E8).
**E11 plan**: extend this same convention with `len(PANEL) = ` the full E11 ladder size, panel
index = the new model's position in a single fixed ordering declared in the E11 protocol
(matches "fixed model-specific child seed" in the prompt).

## 4. GENERator GC generation — import from `tomography_lib.py`, baseline is `run_baseline_regression.py`

`manuscript/experiments/E9_mechanistic_tomography/tomography_lib.py`
- `SEED = 42` (L41); `HG38_FASTA`/`HG38_BED` (L43-44).
- `read_windows(fasta, bed, n, win_bp, rng)` — n windows of `win_bp`; used as `read_windows(..., 24, 170, rng)`.
- **`generator_gc_response(model, tok, pattern, layer, row, alpha, prompts, max_new, seed=SEED) -> (gcs, seqs)`** (L249-276):
  scales `down_proj.weight[row,:] *= alpha`, generates with `torch.manual_seed(seed+i)` per prompt
  `i`, `do_sample=True, top_k=50, temperature=1.0, max_new_tokens=max_new`, filters to `[ACGT]`,
  requires `len(new) >= 10`, returns per-prompt GC fractions.
- Verified hyperparameters match the prompt exactly: 24 windows, 170bp (`120+50`), first 120bp as
  prompt, 64 new tokens, top_k=50, temp=1.0, seed 42+i.

`E9_mechanistic_tomography/run_baseline_regression.py::run_generator()` (L107-170) is the actual
E9-frozen GC baseline being extended for E12: `GENERATOR_ROW_PRIMARY=2371`,
`GENERATOR_ROW_SECONDARY=1522`, `GENERATOR_LAYER=4`, alpha ∈ {1.0, 0.5, 0.0}, 5 random control
rows via `random.Random(SEED).sample(...)`. Output:
`E9_mechanistic_tomography/baseline_regression_results.json`.

**Important**: do NOT use `scripts/mechanism/run_sw_steering.py` as the E12 baseline — it's an
earlier, superseded 5-point-alpha-grid script; repo docs (`MANUSCRIPT_SOURCE_OF_TRUTH.md`,
`figures/FIGURE_PROVENANCE.md`) explicitly distinguish its result (C-040, colleague-branch,
unreproduced) from E9's own canonical 3-point measurement (C-045). **Extend
`run_baseline_regression.py`/`tomography_lib.py`, not the old script.**

Documented E9 numbers to cross-check against (must not silently adjust if E12 disagrees, per
working rules): row 2371 GC 0.3961→0.3863→0.2944 (α=1.0/0.5/0.0), span 0.1017, monotonic; row
1522 GC 0.3961→0.4131→0.3466, span 0.0665, **non-monotonic**; 5-row random control span ≈0.00026–0.0003.

## 5. Preregistration / freeze convention (E7–E10b)

Lock mechanism: `manuscript/src/prereg_lock.py` — `lock <file>` hashes+appends to the
append-only ledger `manuscript/docs/prereg/LOCKS.jsonl`; `verify --all` / `verify <file>`
re-hashes and prints `OK`/`CHANGED`/`MISSING`. Every detection script calls a `verify_lock()`
gate that shells out to this and refuses to run if the file changed since locking.

Protocol files live in **`manuscript/docs/prereg/`**, named `PREREG_<name>.md`
(`PREREG_exact_operator_dimensionality.md` = E7, `PREREG_encoder_decoder_dimensionality.md` =
E8, `PREREG_E10_nlp_architecture_causal[_v2].md`, `PREREG_E10b_phi3_tomography.md`,
`PREREG_mechanistic_tomography_E9.md`, plus an `archive/` for superseded versions).

Recurring section convention (stable across E9→E10→E10b, later docs say "identical to
E9/E10, not re-derived" instead of restating):
`## Question` → `## Scope (frozen, ...FREEZE.md)` → `## Intervention scales (frozen, no
others added after seeing results)` → `## Endpoints (frozen)` → `## Observer families` →
`## Practical adequacy thresholds (frozen)` → `## Resampling unit (frozen)` →
`## Predictions (state before running)` → `## Pre-committed reporting`. Companion per-experiment
freeze docs sit next to the experiment code, e.g.
`E10_nlp_architecture_causal/DECODER_INTERVENTION_FREEZE.md`,
`E9_mechanistic_tomography/BASIS_FREEZE.md`.

**Plan**: write `manuscript/docs/prereg/PREREG_E11_scale_ladder.md` and
`PREREG_E12_generator_degradation_control.md`, following this exact section skeleton, then
`python manuscript/src/prereg_lock.py lock <file>` each before any measurement — matching
convention exactly (not the bare top-level `E11_protocol.md`/`E12_protocol.md` names the task
prompt suggests as a fallback, since the repo's actual convention is `PREREG_*.md` under
`docs/prereg/`, with the experiment's own dir holding the freeze companion + code + results).

## 6. Model panel (existing pins)

- Genomic model configs (no revision field): `configs/*.yaml` (dnabert2, evo1, evo2, evo2_7b,
  generator, generator_prokaryote[_1b], genomeocean, hybridna, megadna, ntv3).
- NLP decoder panel with rows/controls: `E10_nlp_architecture_causal/run_decoder_spectrum.py`
  `PANEL` dict (L41-70) — Llama-7B/Mistral-7B/OLMo-7B-0724/Phi-3-mini-4k-instruct/Qwen2.5-7B.
  **Llama, Mistral, OLMo have `revision=None`** — these are the "four models with no pinned
  revision" the task prompt references (the fourth is presumably one of the genomic configs;
  I have not yet confirmed which — will do before finalizing the E11 protocol).
- Encoder panel: MosaicBERT (`mosaicml/mosaic-bert-base`) / ModernBERT-base
  (`answerdotai/ModernBERT-base`) in `E8_encoder_decoder/run_detection_and_spectral.py` L108,139.
- Best cross-family summary table (repo+revision+row+q1):
  `E10_nlp_architecture_causal/MODEL_AND_BASIS_AUDIT.md` L314-324.
- No single unified model registry spans genomic+NLP; will need to consult both.

## 7. Bootstrap machinery

`E9_mechanistic_tomography/run_fit_observers.py`: `N_BOOT=5000` (L24),
`bootstrap_mae_diff(...)` (L64-97) — resamples **batches** with replacement, same draw applied
paired across conditions, percentile CI. Reused by direct import in
`E10_nlp_architecture_causal/fit_encoder_observers.py` and `E10b_phi3_tomography/fit_phi3_observers.py`.

**Gap**: no existing `bootstrap_gc(...)` for GC arrays — `bootstrap_mae_diff` is shaped for
per-batch MLM loss deltas, not per-prompt GC fractions. E12 will need a thin GC-specific
resampling function (same 5,000-resample, percentile-CI pattern, resampling over prompts
instead of batches) rather than calling `bootstrap_mae_diff` directly on GC arrays. This is a
small new helper, not a reimplementation of the existing machinery.

## 8. Row 2371/1522 dose-response — see item 4 above (same source)

---

## Open items before writing the protocols (will resolve next, not yet done)

1. Identify the fourth unpinned model referenced by the task prompt's "four models with no
   pinned revision" claim (three are Llama/Mistral/OLMo above).
2. HF search for: a second gated-FFN decoder family with ≥3 released sizes (beyond Qwen2.5),
   any gated bidirectional encoder ≥300M-1B+ besides ModernBERT-large, whether NTv3 has larger
   released variants, and whether any genomic family in the panel has multiple sizes. None of
   this is derivable from the repo — it requires live HF Hub queries.
3. Confirm `manuscript/docs/PAPER_OUTLINE.md` and `NEW_DIRECTION_EVIDENCE_AUDIT.md` (both
   flagged as required reading in `manuscript/CLAUDE.md` before writing anything
   headline-shaped) don't already constrain or contradict how E11/E12 should be framed.

I'll do these three before drafting the two PREREG files, unless you'd rather I fold them into
the protocol-writing pass directly.
