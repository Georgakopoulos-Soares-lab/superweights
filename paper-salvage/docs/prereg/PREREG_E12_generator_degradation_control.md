# PREREG — E12: degradation-matched control for the GENERator GC result

**Lock before any measurement:**
`python3 src/prereg_lock.py lock docs/prereg/PREREG_E12_generator_degradation_control.md`

**Verify before and after measurement:**
`python3 src/prereg_lock.py verify --all`

## Why this experiment exists

C-045 (E9, locked, `experiments/E9_mechanistic_tomography/{RESULTS.md,BASELINE_REGRESSION.md,baseline_regression_results.json}`) measured GENERator EUK row 2371: scaling its down-proj row to `alpha=0` moves mean generated GC from 0.3961 to 0.2944 (span 0.1017), against a 5-random-row control span of 0.0003 (~390×). C-045's own note already flags the open question this experiment answers: "does not clear Phase 8's 'graded steering' bar... causal sensitivity, not controlled steering, pending a wider basis." Two readings remain open:

1. Row 2371 exerts specific control over generated nucleotide composition.
2. Ablating row 2371 breaks generation generally, and broken/degenerate genomic output happens to be AT-rich.

Random-row controls do not distinguish these, because a random row barely perturbs the model — it is not damage-matched. This experiment adds a control matched on language-model damage, not on identity, and treats the two readings as separable, pre-committed decision branches (see Decision rules).

**Reconciliation note (D-027, 2026-08-23):** a second, unreconciled measurement of this same row exists — C-040 (colleague-adopted per D-024, wider grid `{0,0.5,1,2,5}×`, GC saturates by 2× then *reverses* at 5×, span ratio 38.59×). C-045 was never measured above `alpha=1.0`, so it cannot confirm or refute C-040's reversal. This experiment's B2 grid is extended to include amplification for row 2371 itself specifically to test this (see "Two separate questions" below) — that is a secondary, independently-reported question, not part of the primary damage-matching claim.

## Frozen inputs (reused, not recomputed)

- Checkpoint: `GenerTeam/GENERator-v2-eukaryote-3b-base` (`configs/generator.yaml`), loaded via `models.WRAPPER_MAP["generator"]` / `tomography_lib.load_generator()`. No new revision pin is introduced here; if the currently-loaded checkpoint has no pinned revision, that is inherited from the existing config, not newly created by E12, and is noted as-is in `results/E12/E12_summary.md` rather than silently fixed.
- Intervention module/pattern: `model.layers.{i}.mlp.down_proj`, layer `GENERATOR_LAYER=4` (`run_baseline_regression.py`).
- Rows: `GENERATOR_ROW_PRIMARY=2371`, `GENERATOR_ROW_SECONDARY=1522` (unchanged).
- Row-scaling mechanism: `tomography_lib._save_row`/`_restore_row`/direct `m.weight.data[row,:] = saved * alpha` — reused verbatim from `tomography_lib.generator_gc_response`. Note this file's `alpha` is the **raw multiplicative scale** (not E9/E10's `1-epsilon*a` framing) — `alpha=1.0` is untouched, `alpha=0.0` is full ablation, `alpha>1.0` is amplification. E12 uses this same raw-scale convention throughout, including for control rows and the damage metric, to avoid a second parameterization.
- Generation loop: `tomography_lib.generator_gc_response` — `do_sample=True, top_k=50, temperature=1.0`, `torch.manual_seed(seed+i)` per prompt `i`. Reused unmodified for every GC measurement in this experiment.
- Windowing primitive: `run_ensemble_encoding.read_windows(fasta, bed, n, win_bp, rng)`, `gc_frac`, `dinuc_freqs` — reused unmodified.
- Random-control-row selection convention: **E9's own** (`random.Random(SEED).sample(...)`, `run_baseline_regression.py` lines 120-121) — not the structural-detection panel's `numpy.random.SeedSequence(42)` convention (E7/E8/E10). This experiment extends `run_baseline_regression.py`/`tomography_lib.py` directly, so it inherits that file's own control-row convention for continuity and exact reproducibility of C-045's existing 5 control rows, rather than switching conventions mid-experiment. **The exact same 5 rows already computed for C-045 are reused** (regenerate via `random.Random(42).sample([r for r in range(nrows) if r not in (2371,1522)], 5)` against GENERator EUK's actual `down_proj` row count — deterministic, same output as the existing run).
- Existing numbers this experiment must not silently adjust if it disagrees (per working rules): C-045's own figures (0.3961/0.3863/0.2944; span 0.1017; ~390×) and C-040's figures (0.2944/0.3961/0.3549; span ratio 38.59×; saturates-then-reverses). Any discrepancy is reported in `E12_summary.md`, not resolved by editing either prior number.

## What is new in E12

- Generation-quality readouts (entropy, distinct-n, homopolymers, k-mer dominance, non-ACGT filtering rate, self-NLL) — not previously computed for this dose-response.
- Finer alpha grid, more prompts, multiple seeds, a greedy-decoding arm, bootstrap CIs.
- A **damage metric** (held-out causal-LM NLL, disjoint corpus) and a **damage-matched control procedure** for both structurally-arbitrary control rows and a random-direction perturbation at row 2371's own location — neither existed before this experiment.

## Corpus and windowing (frozen)

Two **disjoint** hg38 window pools, both drawn from `data/reference/hg38/hg38.fa` / `data/regions/hg38/random_262kb.bed`:

1. **Generation-prompt pool**: 96 windows, 170bp each (first 120bp used as prompt, matching the existing convention exactly), drawn via `read_windows(HG38_FASTA, HG38_BED, 96, 170, rng)`.
2. **Damage/held-out pool**: ≥100 windows, 512bp each, drawn via `read_windows(HG38_FASTA, HG38_BED, 100, 512, rng2)`, from BED regions **excluded** from pool 1's regions before either draw (partition the BED file's region list into two non-overlapping halves by a fixed-seed shuffle-then-split *before* sampling either pool, then sample each pool only from its own half; assert zero chromosome+coordinate overlap between the two pools' sampled windows before proceeding — if the assertion fails, re-partition with a new fixed seed and record that this happened).
- Both pools are frozen once drawn (same seed policy as existing E9 windows: base seed 42 for pool partitioning, a documented offset for each pool's own `read_windows` call).
- The damage pool is used **only** for the NLL damage metric, never for generation or GC. The prompt pool is used only for generation; the damage metric is never computed on generated text.

## Metrics (frozen definitions)

- **GC**: `gc_frac`, unchanged.
- **Damage** `D(row, alpha)`: mean per-token causal-LM NLL over the 512bp damage-pool windows, teacher-forced, float32, with the row scaled to `alpha` exactly as in generation. Computed once per (row, alpha) needed; not resampled per generation seed (it is deterministic given the weights, unlike sampled generation).
- **Predictive entropy**: mean per-token Shannon entropy (nats) of the model's softmax over the full vocabulary at each of the `max_new` sampling steps actually taken during generation (computed from the same forward pass already run for sampling — no extra forward pass), averaged per continuation, then across prompts/seeds within a condition.
- **Distinct-2/3/4**: unique nucleotide n-grams / total (overlapping) nucleotide n-grams, computed on the ACGT-filtered continuation string (character-level, not token-level).
- **Homopolymer runs**: on the filtered continuation, the longest maximal run of one repeated character, and the mean run length across all maximal runs (of any of A/C/G/T) in that continuation.
- **Top-kmer share** (3-mer and 6-mer): `count(most frequent k-mer in the continuation) / total number of overlapping k-mers in the continuation` — an occurrence-share, not a positional/base-coverage fraction (overlapping k-mers make base coverage ambiguous; this is the same denominator convention as distinct-n).
- **Non-ACGT filtering rate**: `1 - len(filtered_string) / len(raw_decoded_string_stripped_of_special_tokens)`, i.e. the fraction of raw decoded characters removed by the existing silent `"".join(c for c in new.upper() if c in "ACGT")` filter in `tomography_lib.generator_gc_response` (this experiment adds instrumentation to record both strings; it does not change the filter itself).
- **Self-NLL**: mean per-token NLL the (row-scaled) model assigns to its own already-generated continuation, teacher-forced, computed with the same row-scaling in effect as during generation.
- **Prompt GC**: `gc_frac` of each 120bp prompt, fixed across all conditions (reported as a reference line, not a per-condition metric).

## Design

### B1/B2 — extended dose-response

- Alpha grid for row 2371 and row 1522: `{1.0, 0.75, 0.5, 0.25, 0.0}` (suppression, matches the task spec) **plus an amplification arm** `{1.5, 2.0, 3.0, 5.0}` for row 2371 only, run under identical seeding/prompt conventions, specifically to test whether C-040's saturate-then-reverse shape (measured under a different, less-controlled script) replicates under this experiment's protocol. Row 1522 is not extended to amplification — C-040 does not report a secondary-row amplification result to compare against, and extending it would not resolve any pre-existing discrepancy.
- Prompts: 96 (see Corpus section), same 96 across every condition/seed/alpha.
- Seeds: base seeds `{42, 43, 44}` (≥3), each offsetting the per-prompt `torch.manual_seed(base_seed + i)` exactly as the existing convention does for `seed=42`.
- Greedy arm: `do_sample=False` (equivalently `top_k=1`), one seed (42, deterministic anyway), at every alpha in the suppression grid, for row 2371 and the 5 control rows.
- Bootstrap: 5,000 resamples over prompts (paired across conditions — the same resampled prompt-index draw applied to every condition being compared in one CI), percentile 95% CI. This is a **new** function (`bootstrap_gc_diff`, to be written in `E12/` alongside the measurement scripts, not yet existing) modeled on `run_fit_observers.bootstrap_mae_diff`'s paired-resampling structure but resampling prompts instead of batches, since no existing function operates on per-prompt GC arrays.

### B3 — damage-matched controls (core of E12)

1. Measure `D(2371, alpha)` across the full row-2371 grid (suppression + amplification, from B2) on the damage pool. `D(2371, alpha=0)` is the target damage level for matching.
2. For each of the same 5 seeded control rows used in C-045:
   - Evaluate damage at a fixed coarse grid `alpha ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0}` (covers strong suppression through 8× amplification).
   - Identify the alpha(s) whose damage is closest to `D(2371, alpha=0)`. If the target lies between two grid points, refine with up to 3 additional evaluations by bisecting toward the target (not assuming global monotonicity — only used to refine within the bracketing interval already observed). Report the final `alpha_c` and its measured damage.
   - **If no point on the full grid reaches `D(2371, alpha=0)`** (i.e. the control row cannot be damaged that much anywhere in this range): record this explicitly as its own finding, and fall back to matching at the highest-damage grid point available for that control row, with the resulting damage mismatch reported alongside every downstream comparison for that row (not silently treated as matched).
3. **Random-direction matched control** (row 2371's own location, not a different row): draw one random Gaussian vector of length `intermediate_size` with a fixed seed (`np.random.default_rng(20260823 + 2371)`), normalize it to unit norm, and replace row 2371's weight vector with `c * unit_random_direction * ||original_row_2371||_2` for a swept scale `c`. Sweep `c` over the same coarse grid as step 2 (`{0.0,0.1,...,8.0}`, interpreted as a multiple of the original row's own norm) and match to `D(2371, alpha=0)` using the same nearest-match-with-refinement procedure. This isolates "this specific learned direction at this location" from "any perturbation of this magnitude at this location" — the control ROWS (step 2) isolate "this location" from "other locations." These are two different axes and both are reported.
4. At each matched configuration (5 control rows at their matched `alpha_c` or fallback, plus the random-direction control at its matched `c`), run the full B1/B2 generation + quality-metric protocol (96 prompts, 3 seeds, GC + all B1 metrics).
5. **Decisive comparison**: bootstrap 95% CI (5,000 resamples over prompts, paired) on `GC(row2371, alpha=0) - GC(matched control)`, for each matched control separately and pooled across the 5 damage-matched control rows.

### B4 — degeneracy attribution

- Compare 3-mer/6-mer composition (top-5 enriched k-mers, `alpha=0` vs. baseline `alpha=1.0`) for row 2371.
- Report the fraction of the GC drop attributable to the single most frequent homopolymer run (operationalized as: recompute GC with all characters belonging to the single longest homopolymer run removed from the continuation, and report `1 - (GC_drop_without_that_run / GC_drop_with_it)`).

## Two separate questions — do not conflate when writing up

1. **Is row 2371 special, at matched damage?** Answered by B3 (control rows + random-direction control vs. row 2371, all matched to `D(2371, alpha=0)`). This is the primary claim this experiment exists to test.
2. **Does row 2371's own dose-response saturate then reverse at amplification (C-040) under this experiment's tighter protocol?** Answered by the row-2371 amplification arm added to B2. This bears on which of C-040/C-045's numbers is more trustworthy going forward; it does **not** bear on whether row 2371 is "special" relative to damage-matched controls, since no control row's amplification behavior is being compared to it on this axis. Report both, but do not use a reversal finding (or its absence) as evidence for or against the B3 specificity claim, and do not use the B3 specificity result as evidence for or against C-040's shape.

## Decision rules (state in advance)

- **Specific-control supported**: row 2371's GC shift at `alpha=0` exceeds every damage-matched control's GC shift (both control-row and random-direction) by a clear margin, with the pooled bootstrap CI on the difference excluding zero, *and* generation-quality metrics at `alpha=0` are not catastrophically degraded relative to the matched controls. → manuscript's claim stands and strengthens.
- **Degradation-explained**: matched controls (either type) produce comparable GC shifts, or row 2371's `alpha=0` output is degenerate by the quality metrics (e.g. collapse into a dominant homopolymer/k-mer materially exceeding matched-control levels). → downgrade to "ablating this row degrades generation, and degraded genomic output is AT-biased"; Fig. 4C is relabeled, not kept as a steering figure.
- **Intermediate**: partial specificity — quantify explicitly what fraction of row 2371's raw GC shift survives damage-matching (e.g. `(shift_2371 - mean_matched_shift) / shift_2371`), and report it as such. Do not round up to "supported."

Each branch is evaluated independently for the random-row-location comparison and the random-direction comparison; if they disagree (e.g. row 2371 beats damage-matched other rows but not a damage-matched random direction at its own location), report both axes separately rather than collapsing to one verdict.

## Resampling unit

Prompts (not seeds, not tokens) — same 96 prompts appear in every condition, so bootstrap resampling draws a set of prompt indices once per resample and applies it identically across every condition in a given comparison (paired), analogous to E9's batch-paired resampling in `bootstrap_mae_diff`.

## Deliverables

- `results/E12/dose_response_extended.csv` — per condition (row, alpha, seed, decoding arm) per prompt: GC + all B1 quality metrics.
- `results/E12/damage_matching.csv` — control row (or random-direction), scale sweep grid, measured damage per grid point, matched `alpha_c`/`c`, matched-condition GC + quality metrics, and a flag for any fallback (target unreachable) case.
- `results/E12/kmer_attribution.csv` — B4 output.
- `figures/E12/gc_vs_damage.pdf` — GC shift (y) vs. LM damage (x), with row 2371 (including its amplification arm), row 1522, damage-matched control rows, and the random-direction control as distinguishable series.
- `figures/E12/generation_quality.pdf` — B1 quality metrics by condition.
- `results/E12/E12_summary.md` — 3-4 paragraphs, drop-in ready, stating which decision-rule branch occurred, reporting the two separate questions (specificity vs. C-040 shape) independently, and reconciling or explicitly leaving open the C-040/C-045 numeric discrepancy per what B2's amplification arm finds.

## Pre-committed reporting

Whatever the decision-rule outcome, both the damage-matched-control result and the row-2371-amplification-shape result are reported, including if either is a null or a reversal-of-expectation — per the repo's working rule that negative results are deliverables, not smoothed over.
