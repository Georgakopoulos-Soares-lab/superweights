# Claude Code task: Part 2 revision work — hygiene, detector reconciliation, GENERator re-analysis, norm-rank controls

## Context

You (or a previous session) produced `audit/AUDIT_REPORT.md` for this repository. That audit
found no wrong numbers, but surfaced several open items. This task acts on them. Read
`audit/AUDIT_REPORT.md` first — it is the reference for paths, artifact names, and what has
already been verified. Do not redo work it already completed.

Same evidence discipline as before:

- Never state a number you did not read from a file or compute in this session. `NOT FOUND` beats
  a plausible reconstruction.
- Label every number `STORED` / `RECOMPUTED` / `CODE-ONLY` / `NOT FOUND`, with file path and
  provenance.
- If two sources disagree, report both.
- Sections are ordered by cost. **Report after each section and wait before starting any section
  marked GATED.**

Deliverables go in `audit/round2/`. Extend `verification_table.csv` and `provenance.json` rather
than starting new ones.

---

## Section 0 — Repo hygiene (do this first, it is a prerequisite for everything else)

The audit found that `results/E11/`, `results/E12/`, `results/E13/` and the experiment code in
`paper-salvage/experiments/{E11_scale_ladder,E12_generator_degradation_control,E13_full_cohort_causal_census}/`
are gitignored or untracked. None of the Part 2 evidence survives a fresh clone. Fix that, but
separate signal from scratch first.

### 0a. Inventory loose working files

The repo root has accumulated untracked scratch files — `check_prompt.md`, several `*_prompt*.md`
variants, `manuscript.txt`, `previous_main.tex`, `"source_of _truth.md"` (note the space in the
filename), `INVENTORY.md`, and others. Before changing anything:

1. List every untracked `*.md`, `*.txt`, and `*.tex` at the repo root and one level down, with
   size, mtime, and first heading or first non-empty line.
2. Classify each into exactly one of:
   - **SCRATCH** — prompts, one-off instructions, transient notes. Candidates for gitignore.
   - **EVIDENCE** — real project state that should be committed (e.g. `PART2_EVIDENCE_PACKET.md`,
     the audit outputs, anything a co-author would need).
   - **UNCLEAR** — ambiguous; do not act, just flag with your reasoning.
3. Present the classification as a table and **stop for my confirmation before modifying
   `.gitignore` or committing anything.** Do not guess on UNCLEAR items.

Note explicitly: `paper-salvage/docs/MANUSCRIPT_SOURCE_OF_TRUTH.md` (99KB, untracked) is almost
certainly EVIDENCE, and the root-level `"source_of _truth.md"` may be a stale duplicate of it —
diff them and report whether they are the same document.

### 0b. Gitignore patterns

Once I confirm the classification, add patterns for the SCRATCH class. Prefer explicit,
narrow patterns over broad globs — `*prompt*.md` would also catch a legitimately useful
`docs/PROMPTING_NOTES.md` if one ever appears. Suggest patterns, show me what each would match
via `git check-ignore -v`, and let me approve before writing.

Remember gitignore does not affect already-tracked files. If any SCRATCH file is already tracked,
flag it separately — removing it from tracking is a different decision and I want to make it
explicitly.

### 0c. Commit the evidence base

Following this project's own convention in `paper-salvage/CLAUDE.md` (results are gitignored;
keepers get `git add -f`):

1. Commit the experiment code: `paper-salvage/experiments/{E11_scale_ladder,E12_generator_degradation_control,E13_full_cohort_causal_census}/`
   in full, including `E11_scale_ladder/regression.py` which backs a Figure 1 statistic.
2. Force-add the results these depend on: `results/E11/`, `results/E12/`, `results/E13/`,
   `results/E13_dnabert2_reproducibility/`, `results/mechanism/attention_sink_implicit_bias.json`,
   and `results/e7_legacy_reanalysis.json`. Check total size first and report it before adding —
   if any single artifact is over ~50MB, flag it rather than committing blind.
3. Commit `audit/` in full.
4. Commit the currently-modified tracked files only if their diffs are intentional project state —
   show me `git diff` for `scripts/mechanism/run_attention_sink.py` and
   `scripts/evaluation/run_sw_pairwise_epistasis.py` before committing, since the audit noted the
   former has an uncommitted hardcoded-path fix.

Use separate, descriptive commits per group. Report the resulting `git status --short` and confirm
a fresh clone would now contain every artifact cited in `audit/verification_table.csv`.

### 0d. Fix or remove the no-op shuffle

`results/mechanism/attention_sink_implicit_bias.json` has `pos0_shuffled_mean` exactly equal to
`pos0_real_mean` to full float precision. Read the shuffle code path in
`scripts/mechanism/run_attention_sink.py`, determine whether the shuffle is a genuine no-op (e.g.
shuffling a copy that is never used, or shuffling downstream of the measurement), and report your
finding. Do not fix it yet — tell me what is wrong and propose the minimal fix. The BOS numbers the
manuscript actually cites (37.9%, 33.0×, 78.8%) do not depend on this block, but the file will be
requested at review.

---

## Section 1 — GENERator c-grid re-analysis (no new compute; highest priority)

The audit revealed that the random-direction control replaces the row with `c · v̂` for a fixed
random unit direction `v̂`, so **c=0 is ablation** and NLL *decreases* monotonically with c (8.754
at c=0 down to 8.375 at c=8.0). The damage matcher, told to hit 8.754, could only move away from it
and returned the smallest nonzero grid step, c=0.0125. A vector of norm 0.0125 replacing a
high-norm row is numerically indistinguishable from ablation, so the manuscript's current
"damage-matched wrong-direction perturbation" claim reduces to "ablation reproduces ablation."

The data to fix this may already exist. Find out.

1. From `results/E12/damage_matching.csv` and any related raw artifacts, emit
   `audit/round2/generator_random_direction_grid.csv` with, for every c on the grid:
   `c, nll, gc_fraction, gc_ci_low, gc_ci_high, n_prompts, n_seeds, obtained_via`.
2. **State clearly which c values have a measured GC fraction and which have NLL only.** If GC was
   measured only at the matcher's selected point (c=0.0125), say so — that is the answer to whether
   the re-analysis is free or needs a short run.
3. Report the baseline NLL (the level the inert control rows sit at, ≈6.386) and compute, for each
   c, the fraction of the full ablation damage achieved:
   `(nll(c) − baseline) / (nll(c=0) − baseline)`.
4. If GC exists across the grid, produce the two plots I need and describe what they show:
   GC vs c, and GC vs NLL with the row-2371 α-sweep and the five control rows on the same axes.
   Save to `audit/round2/figures/`.
5. If GC exists only at c=0.0125, estimate the cost of measuring GC at a handful of larger c values
   — specifically the largest c on the grid plus two or three intermediate points — using the same
   96 prompts × 3 seeds protocol. Report projected GPU-time and **stop for my go-ahead.**

The claim I am trying to establish is: a large, unambiguously non-zero random direction at the
row-2371 location produces most of the ablation damage and reproduces the low-GC phenotype,
therefore the learned direction is not required. Tell me whether the data supports that, does not
support it, or cannot yet answer it.

Separately, restate the control-row result numerically for Methods: each of the five control rows
moves NLL by ~0.001–0.0016 across the entire α ∈ [0, 8] sweep, against a gap to target of ~2.37.
Give me the exact per-row numbers formatted as a sentence I can drop into the manuscript, framing
these as causally insensitive rather than merely under-damaged.

---

## Section 2 — Tomography condition-level uncertainty (cheap)

The existing bootstrap resamples the 16 fixed evaluation batches, so the CI on the F2→F3 MAE
improvement reflects evaluation noise but not the choice of which 20 row-subsets landed in the
held-out pool. With 20 held-out conditions, split choice is plausibly the larger variance source.

1. Implement a **repeated-split analysis** reusing `generate_masks.py`'s sampling logic: draw at
   least 50 fresh fit/calibration/held-out partitions from the same combination space under
   different seeds, refit F0–F3 on each (reusing the already-measured per-condition responses in
   `dnabert2_mask_responses.json` — no model re-runs required if all needed conditions were
   measured), and report the distribution of held-out R² per family and of the F2→F3 MAE
   improvement, at both ε.
2. **First check whether this is possible without new measurements**: how many distinct row-subsets
   have stored responses, versus how many a fresh partition would need. If fresh partitions would
   require unmeasured conditions, report that and instead do a **leave-one-condition-out jackknife**
   over the existing 20 held-out conditions, which needs no new data.
3. Report whichever you ran as: median and 2.5/97.5 percentiles of held-out R² for F0–F3, and the
   fraction of splits/folds in which F3 beats F2 on held-out MAE.
4. Also report ridge λ stability — how often the calibration-selected λ changes across splits.

Emit `audit/round2/tomography_split_stability.csv` and a short interpretation. If F3's advantage
holds across resplits, say so; if it is split-dependent, say that instead.

---

## Section 3 — Detector reconciliation for the five remaining grandfathered models (GATED)

The audit found six models with no ratio field, all from `results/e7_legacy_reanalysis.json`:
Llama-7B, Mistral-7B, OLMo-7B, NTv3, DNABERT-2, GENERator-EUK-3B. DNABERT-2 has been checked; the
other five have not. For DNABERT-2, the strict ratio rule would have selected L8/r603 at q1=0.379 —
below the entire cohort minimum — instead of the frozen L5/r603 at q1=0.793.

**GENERator-EUK-3B is the priority: it is the entirety of Figure 4.** NTv3 is second: it is the
cohort q1 minimum and one of only two negative-effect models. Do those two first and report before
touching the three 7B text decoders.

For each model, following the pattern of `run_dnabert2_repro_diagnostic.py`:

1. Run the current ratio-based detector on the model's canonical discovery probe (the 504-bp
   ACTB-derived sequence for genomic models, WikiText-2 first-20-lines for text) under both
   `add_special_tokens` settings, since that flag was the entire explanation for DNABERT-2.
2. Report, for each model: the global ratio-argmax coordinate and its ratio; the global
   activation-argmax coordinate and its ratio; the frozen candidate's activation rank and ratio
   rank; and whether the frozen candidate would pass the ratio ≥ 5.0 threshold.
3. Compute exact q1, PR_spec, and ‖U_k‖_F for the ratio-argmax coordinate using the same
   `row_spectral_metrics` function the rest of the structural analysis uses, so I can state what the
   strict rule would have selected and how structurally exceptional it is.
4. Update `audit/detector_provenance.csv` in place, replacing `NOT FOUND` with measured values.

For Llama-7B, Mistral-7B, and OLMo-7B, note in the output that these coordinates are
literature-derived (Yu et al., ref [1]) rather than detector-selected, and report whether the
detector independently recovers them. That is a calibration result, not a protocol inconsistency,
and I will frame it separately in Methods.

Report per model as you go rather than batching all five.

---

## Section 4 — Top-K-by-norm controls (GATED, most expensive)

Norm-*matched* controls are infeasible: the audit found the candidate is rank 1 by ‖U_k‖_F in every
layer tested, 12–32× the layer median, with zero rows within ±20%. The workable alternative is
**top-K-by-norm controls** — the rows ranked 2 through 6 by exact ‖U_k‖_F within the candidate's
layer. These exist by construction and are the control that makes the enrichment claim
non-tautological.

### 4a. Selection pass (cheap — run this and report before any causal measurement)

Using `dimensionality_lib.exact_uk_all_rows` and the loaders in `E11_scale_ladder/run_model.py`,
for every census model with cached weights:

1. Compute ‖U_k‖_F for all rows in the candidate's layer.
2. Emit `audit/round2/topk_norm_controls.csv`: `model, layer, candidate_row, candidate_norm,
   layer_median_norm, rank2_row ... rank6_row, rank2_norm ... rank6_norm,
   ratio_rank2_to_candidate, existing_random_control_rows, existing_control_norm_percentiles`.
3. Also compute q1 for each of the five top-K rows. **This alone answers a real question with no
   causal compute at all**: is the candidate's spectral concentration exceptional *conditional on*
   being high-norm, or does q1 simply track norm? Report the candidate-vs-topK q1 comparison per
   model and as a cohort summary, and tell me whether the Figure 1C claim survives this stricter
   comparison.
4. Report which of the 22 models lack cached weights and what downloading them would cost.

### 4b. Causal sweep (do not start without explicit approval)

If 4a looks worth pursuing, estimate the cost of measuring the five top-K rows at ε ∈ {0.5, 1.0}
across the panel, reusing the existing intervention and evaluation harness unchanged. Break the
estimate down by model, identify which models dominate, and propose a staged order — smallest
models first — so I can stop after a subset if the result is already clear.

Do not run it. Report the estimate and wait.

---

## Section 5 — Persist the Section 3 re-analysis from the previous audit

The previous audit recomputed the Frobenius and within-class correlations in-session but they exist
only as report text. Make them reproducible artifacts:

1. Commit `audit/scripts/section3_frobenius.py` and its outputs.
2. Emit `audit/round2/structure_function_correlations.csv` with, for every combination of
   {predictor: q1, layer_relative_frobenius} × {ε: 0.5, 1.0} × {panel: all-22, text-decoders-only,
   text-encoders-only, genomic-only}: `rho, p, ci_low, ci_high, n, seed`.
3. Add leave-one-model-out ρ ranges for each row where n permits.
4. Flag any panel with n < 8 as underpowered in a dedicated column rather than omitting it.

This is the artifact backing the manuscript's revised claim that neither predictor works within
architecture class, and that the nominal cross-class Frobenius association is confounded with
objective and endpoint scale. I want it committed and rerunnable.

---

## Reporting

After each section: a short summary, the artifacts written, and any finding that changes what the
next section should do. Do not proceed past a GATED section without my go-ahead.

If you find something I did not ask about that affects a manuscript claim, tell me — the previous
audit's discovery of the random-direction parameterization was the most valuable thing in it, and
it was not on the checklist.
