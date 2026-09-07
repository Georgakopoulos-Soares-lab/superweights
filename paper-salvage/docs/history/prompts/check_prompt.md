# Claude Code task: Part 2 evidence audit for "Structure Is Not Mechanism"

## What this task is

This is a **read-and-report audit**, not a fix-it task. I am revising the Results and Methods for
Part 2 of a manuscript and I need to know exactly what evidence exists in this repository, what
does not exist, and where every number came from. Do not modify analysis code, do not "improve"
anything, and do not re-run expensive experiments unless a section below explicitly authorizes it.

## Hard rules

1. **Never state a number you did not read from a file or compute in this session.** If a value is
   not in the repo, write `NOT FOUND` and say where you looked. An honest gap is far more useful to
   me than a plausible reconstruction.
2. **Every number gets provenance**: absolute file path, and for each file the git commit that last
   touched it (`git log -1 --format='%H %ci' -- <path>`) plus mtime. Note when a file is untracked
   or dirty.
3. **Label how you obtained each number** using exactly one of:
   - `STORED` — read directly from a results file (JSON/CSV/parquet/log)
   - `RECOMPUTED` — you computed it this session from stored raw artifacts (state the input files)
   - `CODE-ONLY` — you inferred it from reading source code without a corresponding results file
   - `NOT FOUND`
4. **If two files disagree, report both values and both paths.** Do not pick a winner. Conflicts are
   one of the main things I am looking for.
5. **Distinguish raw from derived.** If a summary table exists but the raw per-unit artifacts it was
   built from are missing, say so explicitly — I need to know which numbers I can still re-derive.
6. Prefer `rg` over `grep`; prefer reading result files over re-executing pipelines.

## Orientation phase (do this first, report before continuing)

- Print the repo root, `git log --oneline -20`, `git status --short`, and any branch structure.
- Produce a directory map to depth 3, excluding `.git`, caches, and model weight blobs.
- Locate and list: results directories, preregistration or protocol-lock documents, run manifests,
  config files, and any `CLAIMS`, `STATE`, `PROTOCOL`, or `PREREG` style documents.
- List every file matching the census, tomography, and GENERator experiments by name, with sizes
  and mtimes.

Give me this orientation summary as your first output. Then continue through Sections 1–6.

---

## Section 1 — DNABERT-2 tomography: what exactly is held out (HIGHEST PRIORITY)

This is the most important item in the audit. The manuscript reports observer families F0–F3 fit on
a fixed 10-row basis, with "78 fit masks, 20 calibration masks, 20 held-out masks" (seed 20260822)
and a 78×55 lifted design matrix of full column rank. Held-out R² rises from −0.023 (F0) to 0.888
(F3) at ε=0.5, and −0.040 to 0.790 at ε=1.0, with F3 reducing held-out MAE by 54.7% and 24.4%.

**The question I need answered unambiguously: is the held-out split over masking realizations, over
row-subset combinations (i.e. which rows are ablated together), or over both?**

The design matrix has 55 columns (10 main effects + 45 pair terms), so its rows must vary over
*interventions*, not over masks. But the split is described in units of *masks*. These cannot both
be the primary axis of variation. Resolve it.

Do the following:

1. Find the script(s) that construct the tomography design matrix and the fit/calibration/held-out
   splits. Quote the relevant code blocks with file paths and line numbers.
2. State plainly, in one sentence, what a single row of the 78×55 design matrix corresponds to.
3. Enumerate the intervention conditions (row subsets) in each of the three splits. Emit
   `tomography_splits.csv` with columns:
   `split, condition_id, rows_ablated (semicolon-separated), epsilon, mask_pool_id, n_masks`
4. **Compute and report the overlap explicitly**: how many distinct row-subsets appear in the
   held-out split, and how many of those also appear in the fit or calibration splits. Give the
   intersection size and the actual overlapping subsets. If the intersection is non-empty, say so in
   bold at the top of the section.
5. Report whether the masking realization differs between fit and held-out conditions, or whether a
   single seed-42 mask realization is reused everywhere (the DNABERT-2 MLM Methods text implies the
   latter for the pretrained-objective experiment — check whether tomography inherits that).
6. Report the rank and condition number of the actual stored design matrix, not the claimed rank.
   If the matrix is stored, compute `numpy.linalg.matrix_rank` and `cond` yourself and mark
   `RECOMPUTED`.
7. Report the ridge grid, the selected λ at each ε, and on which split λ was selected.
8. Verify the held-out R², MAE, RMSE, nMAE values and the 54.7% / 24.4% MAE improvements against
   stored outputs. Flag any mismatch with the manuscript numbers above.
9. Report how the 5,000 paired bootstrap resamples over "batch indices" were constructed — what a
   batch index refers to here, and whether the resampling unit is a mask, a batch of windows, or an
   intervention condition.

Also verify the reported epistasis chain for L9/r264 × L9/r294: singleton Δloss +0.0218 and +0.0064,
joint +2.0400, epistasis +2.0118 at ε=1.0, and +0.0528 at ε=0.5. And confirm that this pair ranks
first of 45 by |Γ| at both ε (Γ=0.077 and 0.739) in the stored coefficient tables — emit the full
sorted coefficient table for both ε as `tomography_pair_coeffs.csv`.

---

## Section 2 — Full per-model census table

Emit `census_master.csv`, one row per model in the 22-model functional-criticality census, with:

```
model_id, hf_repo, hf_revision, domain (text/genomic), architecture (encoder/decoder),
n_nonembedding_params, candidate_layer, candidate_row, selection_statistic_used,
activation_ratio, activation_max, q1, PR_spec, frobenius_norm, layer_relative_frobenius,
endpoint_type, n_eval_units, batch_size, baseline_loss,
R_cand_eps0.5, R_cand_eps1.0,
control_1_row ... control_5_row,
R_ctrl1_eps0.5 ... R_ctrl5_eps0.5,
R_ctrl1_eps1.0 ... R_ctrl5_eps1.0,
median_control_eps0.5, median_control_eps1.0, G_eps0.5, G_eps1.0,
control_seed_stream, source_file, git_commit, obtained_via
```

Add a second file `census_controls_structural.csv` giving q1 and ‖U_k‖_F for each of the five
control rows per model where those were computed (this exists for at least the 12-model common
control protocol; report coverage across all 22).

Then verify these specific manuscript claims and report PASS / FAIL / NOT FOUND with the recomputed
value for each:

- 18/22 models with G at ε=0.5 > 0; 20/22 at ε=1.0
- cohort median G = +0.60% (ε=0.5) and +0.88% (ε=1.0)
- model-bootstrap CIs +0.23% to +3.06% and +0.61% to +111.78% (seeds 47 and 52)
- Spearman ρ=0.074, p=0.744, bootstrap CI [−0.391, 0.513] (seed 43), n=22
- named extremes: EuroBERT-610M −0.84% and NTv3 −0.50% at ε=0.5; ModernBERT-base +10.93%,
  Mistral-7B +301.38% at ε=0.5; NTv3 −0.79%, OLMo-7B +111.78%, Llama-7B +300.79%,
  ModernBERT-base +357.04%, SmolLM2-1.7B +698.87% at ε=1.0
- subgroup medians at ε=1.0: text decoders +85.47%, text encoders +0.65%, genomic decoders +0.68%,
  genomic encoders +0.02%
- Fig 1 structural claims: 23 models with accepted candidates; q1 range 0.389 (NTv3) to 0.9996
  (EuroBERT-610M); 12-model mean candidate-minus-mean-control gap 0.884, range 0.544–0.977;
  regression R²=0.158 with β₂=0.1162±0.0860 (p=0.192) and β₁=0.0143±0.0637 (p=0.824)
- Evo2-7B max activation ratio 2.22

Flag the +111.78% coincidence explicitly: the OLMo-7B ε=1.0 candidate effect and the upper bootstrap
CI limit for the cohort median gap are reported as the same number. Confirm from artifacts whether
this is genuine or a copy error.

---

## Section 3 — Frobenius magnitude vs causal effect

The manuscript's Limitations concede that "a secondary association between layer-relative Frobenius
magnitude and causal effect was observed in the fixed panel," with no completed leave-one-out or
covariate-adjusted analysis. I need to know what actually exists.

1. Find every artifact, notebook, script, or log touching this association. Report the association
   statistic actually computed, its value, n, and which panel ("the fixed panel" — identify whether
   this means all 22, the 12-model control subset, or the 5-model E10 decoder audit).
2. Report whether any leave-one-model-out, leave-one-family-out, covariate-adjusted, or influence
   analysis was started, and its state (complete / partial / stub / absent).
3. Using `census_master.csv` from Section 2, **recompute** and report, marked `RECOMPUTED`:
   - Spearman ρ and p between layer-relative ‖U_k‖_F and signed R at ε=1.0 across all models with
     both values available; same for ε=0.5
   - the same with a 5,000-draw model bootstrap CI, seed 43, matching the q1 procedure exactly
   - leave-one-model-out ρ values (n values, one per omitted model) and the min/max
   - the same correlations restricted to text decoders only
   - for comparison, q1 vs effect restricted to text decoders only, and q1 vs effect with a binary
     split at q1 ≥ 0.95
4. Report the empirical distribution of q1 across the 22 census models: how many fall above 0.95,
   above 0.97, and below 0.8. I need to characterize range restriction.
5. Report whether `layer_relative_frobenius` exists for all 22 models or only a subset, since it
   requires the five control norms.

---

## Section 4 — Norm-matched control feasibility

I am considering adding norm-matched same-layer controls (controls chosen to match the candidate's
operator magnitude rather than sampled at random) as a prospectively-defined additive control.

1. Determine whether per-row exact ‖U_k‖_F can be computed for **all** rows in a given layer with
   existing code — identify the function implementing the Gram identity ‖U_k‖²_F = d_kᵀ K d_k and
   confirm it is vectorizable over k without materializing every U_k.
2. Report which model weights are already cached locally (paths, sizes) and which would need
   downloading. Report available GPU/CPU/RAM.
3. For any model where weights are already local, **compute the full within-layer distribution of
   ‖U_k‖_F for the candidate's layer** and report: candidate norm, layer median, layer max, the
   candidate's rank and percentile, and the count of rows within ±10% and ±20% of the candidate
   norm. Emit `norm_matched_feasibility.csv`. Start with the two or three smallest cached models to
   establish cost, report timing, then stop and tell me the projected cost for the full panel rather
   than running it.
4. State whether the five existing random controls happen to include any row within ±20% of the
   candidate norm in any model.
5. Estimate wall-clock and GPU-hours for a five-row norm-matched control sweep at both ε across all
   22 models, reusing the existing intervention and evaluation harness. Identify which models
   dominate the cost.

---

## Section 5 — GENERator damage-matching

The manuscript reports: baseline GC 0.4204, post-ablation 0.3065 (Δ = −0.1139); row-2371 ablation
NLL 8.754; largest control damage ≈6.386; random-direction condition NLL 8.7536 with GC 0.3067;
longest homopolymer accounts for 3.9% of the GC decrease.

1. Verify each of those against artifacts, with provenance.
2. Emit `generator_alpha_sweeps.csv` with the full damage curves: `row_id, alpha, nll, gc_fraction,
   gc_ci_low, gc_ci_high, n_prompts, n_seeds` for row 2371, all five control rows across
   α ∈ {0, 0.1, …, 8.0}, and the random-direction conditions.
3. For each control row, report the **maximum achievable NLL over the swept α range** and whether
   the curve is still rising at α=8.0 or has saturated. This determines whether extending the sweep
   could reach the 8.754 target.
4. Report whether the harness supports **joint multi-row perturbation** in the generation pipeline
   (as opposed to the single-row context manager described in Methods). If yes, identify the entry
   point. If no, report what would need to change.
5. Estimate the cost of a matched-damage-elsewhere control: perturb k random non-2371 rows jointly,
   binary-searching on scale until held-out NLL ≈ 8.75, then measure GC over 96 prompts × 3 seeds.
   Report GPU-hours and any blockers.
6. Confirm the two hg38 pools (96×170bp generation, 100×512bp damage) are genuinely
   non-overlapping — compute the interval intersection from the stored BED-derived coordinates,
   don't take the docstring's word for it.
7. Verify the BOS reproduction numbers: 37.9% mean incoming attention at position 0, 33.0× uniform,
   78.8% of layer-head observations with argmax at position 0, over 40 windows at seed 42.

---

## Section 6 — DNABERT-2 detector reconciliation

Methods define the accepted detector as: per-layer channel max divided by the layer median channel
max; accept the model-wide argmax of that ratio if ratio ≥ 5.0. But DNABERT-2's candidate L5/r603 is
described as the global **activation maximum** (out_max = 944.5556, ratio 152.7345) and explicitly
"not the global ratio maximum." A separate note in my working files says a recent detector execution
gives L5/r603 an activation ratio of approximately 0.999, which contradicts 152.7345.

Resolve this completely.

1. Find every code path that performs activation-based candidate detection. There is likely more
   than one. List them all with paths, and diff them — I want to know precisely how they differ
   (hook location, pre- vs post-down-projection, special-token handling, batch size, dtype,
   truncation length, absolute vs signed activation, which axis the median is taken over).
2. Re-run detection on DNABERT-2 with the canonical 504-bp ACTB probe and tokenizer-default special
   tokens, using **each** distinct code path. For each path report: the argmax-by-activation
   coordinate, the argmax-by-ratio coordinate, and for L5/r603 specifically its out_max, its ratio,
   its activation rank, and its ratio rank. Mark `RECOMPUTED`.
3. Report which path reproduces 944.5556 / 152.7345 and which produces ≈0.999, and identify the
   single line or config difference responsible.
4. Report the q1 and ‖U_k‖_F of the ratio-argmax coordinate under the canonical path, so I can state
   what the strict ratio rule would have selected and whether it would change any Part 1 or Part 2A
   conclusion.
5. Audit the other 21 census models for the same inconsistency: for each, state whether the frozen
   candidate is the ratio-argmax, the activation-argmax, both, or neither, and whether it was
   selected fresh under the current detector or grandfathered from an earlier protocol. Emit
   `detector_provenance.csv`.
6. Report the recorded HF revision for every model and confirm which are unrecoverable (Methods
   admits NTv3's is not recorded exactly — check whether any others share this).

---

## Deliverables

Write to a new `audit/` directory:

- `AUDIT_REPORT.md` — the narrative report, organized by section, opening with a
  **Findings-of-concern list**: every conflict, every failed verification, every `NOT FOUND` that
  blocks a manuscript claim, ranked by how load-bearing it is.
- The CSVs named above.
- `verification_table.csv` — every manuscript number I listed, with columns
  `section, claim, manuscript_value, artifact_value, status (PASS/FAIL/NOT FOUND), source_file,
  obtained_via`.
- `provenance.json` — file → {git commit, mtime, sha256} for every artifact you read.

Any code you write for recomputation goes in `audit/scripts/` and must be deterministic and rerunnable.

## Working style

Work through the sections in order and **report after each one** rather than saving everything for
the end. Section 1 is the priority — if it turns out the tomography held-out split does not isolate
unseen row-subsets, that changes what the rest of the audit is for, and I want to know immediately
rather than after you have finished Section 6.

Where a section asks for something expensive, do the cheap diagnostic first, report the projected
cost, and wait for me before committing compute.

If you find something I did not ask about that undermines a manuscript claim, tell me. That is more
valuable than completing the checklist.