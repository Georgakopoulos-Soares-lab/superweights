# Part 2 Evidence Audit — "Structure Is Not Mechanism"

Audit performed by Claude Code, 2026-08-25, against
`/work/11034/atzanakak/glm_super_weight/genomic-super-weights` on branch
`integrate/mechanism-and-negative-results` (working tree dirty — see Orientation).

This report was written incrementally, section by section, per the working-style
instructions in `check_prompt.md`, and reported to the requester after each section
rather than all at once. All six sections are complete.

## Findings-of-concern list (updated as sections complete)

1. **[LOW severity, resolved-by-audit]** `results/E13/` — which holds the entire
   22-model census (`part2_22_model_results.csv`, `part1_22_*.csv`, etc., needed for
   Section 2) — is **gitignored**. None of the census results have git provenance; only
   `mtime` and `sha256` are available. Code that produced them (`E13_full_cohort_causal_census/*.py`)
   is git-tracked, but current working-tree state of that code is **untracked** (the whole
   `E13_full_cohort_causal_census/` directory is untracked — see Orientation). Flagged here;
   resolved per-number in Section 2.
2. **[LOW severity, informational]** `manuscript/docs/MANUSCRIPT_SOURCE_OF_TRUTH.md`
   (99KB, dated 2026-08-23) — the apparent authoritative claims document — is **untracked**,
   i.e. has no git history at all. Any number sourced from it alone should be treated as
   provisional/uncommitted.
3. **[Section 1, addressed — no defect found]** The manuscript's phrase "78 fit masks, 20
   calibration masks, 20 held-out masks" initially reads as ambiguous against a "78×55
   design matrix of full column rank" (55 columns require row-level variation over
   *conditions*, not raw masking noise). **Resolved**: a "mask" in this experiment's
   terminology **is** a row-subset condition (which of the 10 basis rows are jointly
   scaled), not a repeated noise realization — see Section 1 for full derivation. The
   split is over row-subset combinations, the intersection between splits is **empty by
   construction and confirmed empty by direct recomputation**, and a single seed-42 token-masking
   realization is reused for baseline and every condition (paired design, no separate
   "masking-realization" axis exists in the tomography design at all). This is exactly
   the deliberate, correct design the prereg calls for — flagged in this list because
   the prompt asked it be surfaced immediately regardless of outcome, not because a
   problem was found.
4. **[MEDIUM severity, informational — not a numerical error]** Section 2's entire evidence
   base (`results/E13/`, `results/E11/`, and the code in
   `manuscript/experiments/{E13_full_cohort_causal_census,E11_scale_ladder}/`) is
   **completely absent from git** — either gitignored (data) or untracked (code,
   including the OLS regression backing a Fig 1 statistic,
   `E11_scale_ladder/regression.py`). All 30+ Section-2 manuscript numbers checked
   **PASS** against these files, but none of them would survive a fresh clone. Recommend
   committing the E11/E13 code (with `git add -f` for the results, per this project's own
   `manuscript/CLAUDE.md` convention: "Results we are keeping get copied into
   `results/keep/`... `git add -f`, since `results/` is gitignored") before this goes
   further.
5. **[LOW severity, resolved]** The "+111.78%" appearing both as OLMo-7B's own ε=1.0
   causal effect and as the cohort-median-gap bootstrap CI upper bound is **not a copy
   error** — the two numbers are independently computed and agree only to 4 significant
   figures (111.7758% vs. 111.7765%), a real coincidence of the bootstrap distribution's
   shape, not a duplicated value. See Section 2.
6. **[MEDIUM-HIGH severity, open]** DNABERT-2's activation-argmax-vs-ratio-argmax
   detector inconsistency (the prompt's original trigger for Section 6) is fully resolved
   by a pre-existing diagnostic this audit verified — but that diagnostic revealed **5
   more models (Llama-7B, Mistral-7B, OLMo-7B, NTv3, GENERator-EUK-3B) share the exact
   same grandfathering pattern** (selected via an older activation-based rule with no
   ratio ever computed) and **none of them have been checked** against the current
   ratio≥5.0 rule the way DNABERT-2 now has been. For DNABERT-2, applying the strict rule
   would have selected a structurally unremarkable row (q1=0.379, below the cohort
   minimum) instead of the near-rank-1-concentrated frozen candidate (q1=0.793) — a
   finding that could change if the same check were run on the other 5. See Section 6.

---

## Orientation

- **Repo root:** `/work/11034/atzanakak/glm_super_weight/genomic-super-weights`
- **Branch:** `integrate/mechanism-and-negative-results` (not `main`)
- **HEAD:** `fafd44d` "E10b Phase 8 + final synthesis: split decision, three-way redundancy break"
- **Working tree:** 7 modified tracked files (`manuscript/docs/{CLAIMS_LEDGER,DECISIONS,PROJECT_STATUS}.md`,
  `manuscript/docs/prereg/LOCKS.jsonl`, `manuscript/figures/README.md`,
  `scripts/evaluation/run_sw_pairwise_epistasis.py`, `scripts/mechanism/run_attention_sink.py`)
  plus a long list of untracked files/directories, notably: `check_prompt.md` itself,
  `PART2_EVIDENCE_PACKET.md`, `INVENTORY.md`, several loose `*_prompt.md` files,
  `manuscript.txt`, `previous_main.tex`, `"source_of _truth.md"`, and the untracked
  experiment directories `manuscript/experiments/{E11_scale_ladder,E12_generator_degradation_control,E13_full_cohort_causal_census}/`
  and all of `manuscript/figures/{main,output,source_data,supplement}/`.
  **Implication**: E11, E12, E13 — i.e. the GENERator damage-matching (Section 5), full
  census (Section 2), and scale-ladder work — exist only in the working tree, not in any
  commit. `git log -1` on their files returns nothing; provenance for that code is
  mtime/sha256 only, tracked explicitly per-file below and in `provenance.json`.
- **E9 (DNABERT-2 tomography, Section 1)** is fully committed and clean: all files under
  `manuscript/experiments/E9_mechanistic_tomography/` are tracked, last touched by
  commits `2a76d99`, `ad10a03`, `e5e3dce` (all 2026-08-22), working tree clean against HEAD.
- **Governing doc**: `manuscript/CLAUDE.md` establishes this project's own evidence
  discipline (`CLAIMS_LEDGER.md`, `DECISIONS.md`, "if you cannot fill the evidence path,
  the claim is not ready to be written"). This audit's hard rules are consistent with,
  and enforced by, that existing project discipline.
- **Experiment → section map:**
  - Section 1 (tomography) → `manuscript/experiments/E9_mechanistic_tomography/`
  - Section 2 (census) → `manuscript/experiments/E13_full_cohort_causal_census/` (code, untracked)
    + `results/E13/` (data, gitignored)
  - Section 5 (GENERator) → `manuscript/experiments/E12_generator_degradation_control/` (untracked)
    + `results/E12/`

---

## Section 1 — DNABERT-2 tomography: what is held out

### 1. Answer to the core question

**A "mask" in this experiment is a row-subset intervention condition — a specific
combination of which of the 10 frozen basis rows are jointly scaled — not a repeated
noise realization.** The held-out split is over row-subset combinations. There is no
second axis of "masking realization" in the design at all: the *token*-masking used to
build the fixed MLM evaluation batches is a single seed-42 realization, built once and
reused unchanged across the baseline and every row-subset condition
(`run_dnabert2_measurements.py:41-51`; confirmed in `tomography_lib.build_fixed_batches`).
So the manuscript's "78/20/20 masks" and the "78×55 design matrix" are describing the
same object at two levels: 78 conditions (design-matrix rows) map through the 55-column
lift (10 main + 45 pairwise indicator columns) — there is no contradiction once "mask" is
read as "condition."

**Code**: `manuscript/experiments/E9_mechanistic_tomography/generate_masks.py`
(committed `2a76d99`, 2026-08-22 15:54:18).

```python
# lines 55-56: a "mask" (pool element) is a row-subset combination of the 10-row basis
def all_masks_of_size(k: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(N), k))

# lines 130-144: singletons, then fit/calibration/held_out are all sampled from a SHARED
# `used` set that accumulates across every pool -- this is what forces disjointness
used: set = set()
singleton_masks = [tuple([i]) for i in range(N)]
for m in singleton_masks:
    used.add(m)
...
for pool_name in ("fit", "calibration", "held_out"):
    ...
    chosen = sample_disjoint(rng, pool_by_k, k_choices, n_needed, used)  # `used` shared

# lines 59-75: sample_disjoint skips any candidate already in `used`
def sample_disjoint(rng, pool_by_k, k_choices, n_needed, used):
    ...
    if m in used:
        continue
    used.add(m)
    chosen.append(m)
```

Design-matrix construction, `design_matrix_lifted` (lines 89-96): each row is one mask's
10-length 0/1 vector `a`, lifted to 55 columns = 10 main effects + 45 pairwise products
`a_i * a_j`.

### 2. What a single row of the 78×55 design matrix corresponds to

**One row = one intervention condition: a specific subset of the 10 frozen basis rows
{(L5,r603), (L3,r86), (L3,r399), (L9,r264), (L9,r294), (L3,r603), (L3,r641), (L7,r603),
(L6,r603), (L5,r86)}, each scaled to `alpha_i = 1 − epsilon·a_i` simultaneously (per
`tomography_lib.alphas_for_mask`), evaluated once per epsilon.** The 55 columns are 10
main-effect indicators (`a_i`) plus 45 pairwise-interaction indicators (`a_i·a_j`) for
every unordered pair of basis rows — a full second-order polynomial basis over which
rows are jointly active in that condition.

### 3. `tomography_splits.csv` — emitted

Written to `audit/tomography_splits.csv` (256 rows = (10 singletons + 78 fit + 20
calibration + 20 held-out) × 2 epsilons). Columns: `split, condition_id, rows_ablated
(semicolon-separated, e.g. "L9r264;L9r294"), epsilon, mask_pool_id (density bucket
rho_0.25/rho_0.5/rho_0.75, derived from `k = sum(a)` per `generate_masks.py`'s DENSITIES
table), n_masks (total mask count in that split)`.
Source: `masks_dnabert2.json` (STORED, `2a76d99`).

### 4. Overlap between held-out and fit/calibration — **RECOMPUTED, zero**

Recomputed directly from the stored `masks_dnabert2.json` pools (not merely inferred from
reading the sampling code): treated each mask as the `frozenset` of active basis indices
and intersected across pools.

```
RECOMPUTED overlap (row-subset intersection) across splits:
  fit_and_held_out: 0 shared subsets
  calibration_and_held_out: 0 shared subsets
  fit_and_calibration: 0 shared subsets
  singletons_and_held_out: 0 shared subsets
```

All 78 fit, 20 calibration, and 20 held-out row-subsets are pairwise distinct as sets
(`len(set(...)) == len(...)` for every pool). **Intersection is empty — no bolded warning
needed by the prompt's own rule, since the rule only requires bolding a non-empty
intersection.** The stored `cooccurrence_check` in `masks_dnabert2.json` further confirms
held-out exercises all 45 possible pairwise co-occurrences that the fit pool exercises
(`n_pairs_seen_in_both: 45`, `n_pairs_in_held_out_not_in_fit: 0`) — pair terms are
identifiable on both pools, not just the fit pool.

### 5. Masking realization: single seed-42 realization reused everywhere — **confirmed**

`run_dnabert2_measurements.py:41-51` builds `seqs` and `batches` **once**, before the
per-mask loop, using `SEED = 42` for both window sampling (`read_fasta_windows_mlm`) and
token-mask selection (`build_fixed_batches(..., seed=SEED)`, which itself seeds a
`torch.Generator` once). The same `batches` object is then passed unchanged into
`dnabert2_response_per_batch` for the baseline and for every one of the 128 (10+78+20+20)
mask conditions at both epsilons (`for pool_name, vecs in masks["pools"].items(): ... pb =
tl.dnabert2_response_per_batch(model, pattern, BASIS, a, eps, batches)`, lines 66-75). This
matches the "single seed-42 mask realization reused everywhere" pattern the DNABERT-2 MLM
Methods text describes for the pretrained-objective experiment: **tomography does inherit
it**, by direct code inspection — there is no separate masking-realization sampling
anywhere in the tomography pipeline.

### 6. Rank / condition number of the actual stored design matrix — **RECOMPUTED**

Recomputed directly from `masks_dnabert2.json`'s stored `fit` pool (not the code's
self-reported check, though they agree):

```
RECOMPUTED fit additive design (78, 10):  rank=10/10 (full),  cond(X)=4.69,   cond(X^T X)=22.02
RECOMPUTED fit lifted design  (78, 55):  rank=55/55 (full),  cond(X)=67.24,  cond(X^T X)=4520.66
```

Matches the manuscript's claim of "full column rank" for the 78×55 lifted design.
`cond(X^T X) ≈ 4521` is the quantity ridge regularizes against; not itself flagged in the
manuscript, noted here for completeness.

### 7. Ridge grid, selected λ, and selection split

**Code**: `run_fit_observers.py:23` — `LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0,
30.0, 100.0, 300.0, 1000.0]` (10-point log-ish grid). λ is selected by minimizing MSE
**on the calibration pool** (`y_cal` vs. `a_cal @ beta` for F2, `y_cal` vs `Xc @ coef` for
F3 — lines 133-151), never on held-out. **STORED** selected values, from
`fit_results_dnabert2.json`:

| epsilon | F2 (additive) λ | F3 (lifted) λ |
|---|---|---|
| 0.5 | 100.0 | 0.1 |
| 1.0 | 30.0  | 1.0 |

### 8. Held-out R², MAE, RMSE, nMAE and the 54.7%/24.4% improvement — **STORED, all PASS**

Read directly from `fit_results_dnabert2.json` (git commit `e5e3dce`, clean):

| epsilon | family | R² | MAE | RMSE | normMAE |
|---|---|---|---|---|---|
| 0.5 | F0 | −0.0225 | 0.030348 | 0.052585 | 0.1604 |
| 0.5 | F1 | 0.3244 | 0.033363 | 0.042742 | 0.1764 |
| 0.5 | F2 | 0.5167 | 0.027657 | 0.036151 | 0.1462 |
| 0.5 | F3 | 0.8875 | 0.012515 | 0.017440 | 0.0662 |
| 1.0 | F0 | −0.0405 | 0.490676 | 0.676607 | 0.2281 |
| 1.0 | F1 | 0.6592 | 0.307246 | 0.387210 | 0.1428 |
| 1.0 | F2 | 0.5811 | 0.353941 | 0.429324 | 0.1645 |
| 1.0 | F3 | 0.7898 | 0.267660 | 0.304144 | 0.1244 |

F2→F3 relative MAE improvement: **54.7%** at ε=0.5 (stored `0.54748` — manuscript "54.7%",
**PASS**), **24.4%** at ε=1.0 (stored `0.24377` — manuscript "24.4%", **PASS**). Bootstrap
CI on MAE_F2−MAE_F3 excludes zero at both epsilons (ε=0.5: [0.00382, 0.01856]; ε=1.0:
[0.06139, 0.11436]), decision `PAIR_TERMS_REQUIRED` at both. Manuscript's F0 R² (−0.023 at
ε=0.5, −0.040 at ε=1.0) and F3 R² (0.888 at ε=0.5, 0.790 at ε=1.0) **PASS** to the
manuscript's stated precision (stored values −0.0225/0.8875 and −0.0405/0.7898 round to
those figures).

### 9. Bootstrap resampling unit — **batch index, not mask or condition**

**Code**: `run_fit_observers.py:64-97`, `bootstrap_mae_diff`. `n_batches = len(pools["held_out"][0][pbkey])`
— the number of **fixed MLM evaluation batches** (16 batches of masked hg38 windows, per
`baseline_regression_results.json: n_batches=16` / `dnabert2_mask_responses.json` uses the
same 16-batch object). Each bootstrap draw resamples **batch indices with replacement**
(`idx = RNG.integers(0, n_batches, size=n_batches)`), applies the **same draw** to the
baseline's per-batch losses and to every held-out condition's per-batch losses (paired),
recomputes each condition's Δloss under the resampled batch weighting, then recomputes
MAE_F2−MAE_F3. **"Batch index" = an index into the list of 16 fixed
batches-of-hg38-windows** (not a mask, not an intervention condition, not a single window).
5,000 resamples (`N_BOOT = 5000`, `run_fit_observers.py:24`) — matches the manuscript's
"5,000 paired bootstrap resamples."

### Epistasis chain for L9/r264 × L9/r294 — **STORED, all values PASS**

This required checking two things: whether the raw joint-ablation measurement for exactly
this pair exists (it is **not** among the 78+20+20 tomography masks — no fit/calibration/
held-out condition activates exactly `{9,264}` and `{9,294}` alone), and where the
manuscript's numbers actually come from.

**Resolved**: they come from a separate, dedicated Phase-0 regression script in the same
E9 experiment, not from the mask-sweep design: `run_baseline_regression.py` (committed
`2a76d99`), output `baseline_regression_results.json` (committed `ad10a03`, clean),
narrated in `BASELINE_REGRESSION.md`. This script directly measures the singleton and
exact-pair ablations for L9/r264 and L9/r294 as a reproduction check against an older,
pre-E9 result (`CHECKPOINT_1_PRETRAINED_EPISTASIS.md`, commit `3d3bbb0`, 2026-08-14 —
whose own raw JSON, `epistasis_pretrained_vs_finetuned.json`, no longer exists anywhere in
the repo; only its markdown narrative survives). All values below are **STORED** directly
from `baseline_regression_results.json` (`dnabert2.epsilon_1p0` / `epsilon_0p5` keys):

| epsilon | d_A (L9r264) | d_B (L9r294) | d_AB (joint) | epistasis | manuscript | status |
|---|---|---|---|---|---|---|
| 1.0 | 0.021818806 | 0.006401925 | 2.040024024 | 2.011803292 | +0.0218 / +0.0064 / +2.0400 / +2.0118 | **PASS** |
| 0.5 | 0.003416434 | −0.000997406 | 0.055204002 | 0.052784974 | +0.0528 | **PASS** |

The ε=1.0 figure additionally reproduces the pre-existing, now-unbacked
`CHECKPOINT_1_PRETRAINED_EPISTASIS.md` claim (`+2.0118`) to 4 decimal places under a
from-scratch environment (different revision-pin discipline, fresh hg38 download) — this
is a genuine independent replication, not a copy of the same run, per
`BASELINE_REGRESSION.md:19-22`. **Note for the ledger**: the manuscript should cite
`baseline_regression_results.json` (STORED, E9, 2026-08-22) as the source for these four
numbers, not the 2026-08-14 checkpoint, whose backing JSON is gone.

### Pair-coefficient ranking — **STORED, both epsilon PASS; full table emitted**

`tomography_pair_coeffs.csv` (90 rows = 45 pairs × 2 epsilons), sorted by `|gamma|`
descending within each epsilon, source `fit_results_dnabert2.json` F3.gamma_pairs
(**STORED**). Confirmed top-ranked pair at both epsilons:

```
epsilon=0.5: top pair by |gamma| = (9,264)x(9,294), gamma=0.077152   (manuscript: Γ=0.077 — PASS)
epsilon=1.0: top pair by |gamma| = (9,264)x(9,294), gamma=0.738695   (manuscript: Γ=0.739 — PASS)
```

Note this Γ is a **ridge regression coefficient** on the lifted design (F3), a distinct
quantity from the raw "epistasis" (joint − sum-of-singletons) reported above — both
independently rank/verify the same pair, by two different, non-redundant computations,
which is a meaningfully stronger confirmation than either alone.

### Section 1 verdict

No defect found. The held-out split is correctly disjoint over row-subset combinations by
construction and by direct recomputation; the "masks" terminology is consistent once
"mask" = "row-subset condition" is established; there is exactly one token-masking
realization, shared throughout, as the Methods text implies; the ridge grid, selection
split, held-out metrics, bootstrap unit, and the full epistasis/coefficient chain all
check out against stored artifacts with no numerical discrepancies against the manuscript.
**The rest of the audit (Sections 2–6) is not gated by any Section 1 problem — proceeding.**

---

## Section 2 — Full per-model census table

**Verdict: no numerical discrepancy found.** Every manuscript number checked below is
**PASS**, recomputed independently from `results/E13/part2_22_model_results.csv` (22
rows, gitignored — git provenance is mtime/sha256 only, see `provenance.json`) and
`results/E11/scale_ladder.csv` / `regression_summary.json` (also untracked). A pre-existing
internal document, `PART2_EVIDENCE_PACKET.md`, already states the same numbers; this audit
did **not** trust that document and recomputed everything from the raw CSVs/JSONs instead
(script: `audit/scripts/section2_census.py`), so the PASS below is an independent check,
not a repetition of the packet's own claim.

### Deliverables emitted

- `audit/census_master.csv` — 22 rows, requested schema. Three columns
  (`selection_statistic_used`, `activation_ratio`, `activation_max`) are left **blank by
  design**: populating them correctly requires reconciling the multiple, sometimes
  disagreeing detector code paths described in Section 6, and duplicating that
  investigation here would risk silently picking the wrong one. See Section 6's
  `detector_provenance.csv` for the authoritative per-model values.
- `audit/census_controls_structural.csv` — per-control `q1` and `‖U_k‖_F`. **Coverage:
  12/22 models** (exactly the E11 "newly measured" cohort — `results/E11/scale_ladder_controls.csv`).
  The other 10 models (the legacy E7/E10-sourced causal candidates: Llama-7B, Mistral-7B,
  OLMo-7B, Qwen2.5-7B, MosaicBERT, ModernBERT-base, NTv3, DNABERT-2, GENERator-EUK-3B,
  GenomeOcean-4B) have causal control **effects** (`R_ctrl*` in `census_master.csv`) but
  **no per-control structural spectra** (q1/‖U_k‖_F) — this matches the manuscript's own
  caveat ("this exists for at least the 12-model common control protocol") exactly; it is
  not a new gap this audit discovered, but it is now quantified precisely (12/22, not "at
  least").

### Cohort-level claims (all PASS)

| claim | manuscript | recomputed | status |
|---|---|---|---|
| G>0 count, ε=0.5 | 18/22 | 18/22 | PASS |
| G>0 count, ε=1.0 | 20/22 | 20/22 | PASS |
| median G, ε=0.5 | +0.60% | +0.5992% | PASS |
| median G, ε=1.0 | +0.88% | +0.8759% | PASS |
| model-bootstrap 95% CI, median G, ε=0.5 (seed 47, 5000 draws) | [+0.23%, +3.06%] | [+0.2258%, +3.0608%] | PASS |
| model-bootstrap 95% CI, median G, ε=1.0 (seed 52, 5000 draws) | [+0.61%, +111.78%] | [+0.6060%, +111.7765%] | PASS |
| Spearman q1 vs. ε=1.0 signed candidate R | ρ=0.074, p=0.744, CI[−0.391,0.513], seed 43, n=22 | ρ=0.073970, p=0.743559, CI[−0.391371,0.512621] | PASS |

All 9 named extremes (EuroBERT-610M/NTv3/ModernBERT-base/Mistral-7B at ε=0.5; NTv3/OLMo-7B/Llama-7B/ModernBERT-base/SmolLM2-1.7B
at ε=1.0) and all 4 subgroup medians at ε=1.0 (text/decoder +85.47%, text/encoder +0.65%,
genomic/decoder +0.68%, genomic/encoder +0.02%) **PASS** — full figures in
`verification_table.csv`.

### The +111.78% coincidence — **resolved: not a copy error**

Recomputed both quantities independently from `part2_22_model_results.csv`:

```
OLMo-7B-0724-hf's own eps=1.0 candidate causal effect (R):      111.775806 %
Cohort median-G bootstrap CI upper bound (seed 52, 5000 draws): 111.776545 %
```

These are **two different numbers computed by two entirely different procedures** — one
is a single model's point causal effect, the other is the 97.5th percentile of a
bootstrap distribution over the median of all 22 models' G values — that happen to agree
to 4 significant figures (both round to "111.78%") but diverge starting at the 4th
significant digit. This is **not** a copy-paste bug. It is a real structural
near-coincidence: with only 2/22 models having non-positive G at ε=1.0 and a long
right-skewed tail (values up to +699%), the upper tail of the bootstrap-median
distribution is pinned close to the actual data values near the top of the ranked list,
and OLMo happens to sit almost exactly there. Manuscript should keep both numbers as
independently sourced, or add one decimal place to visibly disambiguate them if the
apparent identity is likely to raise the same question in review.

### Fig 1 structural claims (all PASS)

Recomputed from `results/E11/scale_ladder.csv`, `scale_ladder_controls.csv`, and
`regression_summary.json` (the OLS is run by `manuscript/experiments/E11_scale_ladder/regression.py`,
untracked, output also untracked/gitignored):

- **23 models with accepted candidates** — confirmed (23 rows in `scale_ladder.csv` with
  non-null q1). q1 range **0.388894 (NTv3) to 0.999624 (EuroBERT-610M)** — PASS against
  manuscript's 0.389 / 0.9996.
- **12-model mean candidate-minus-mean-control gap = 0.883674** (manuscript: 0.884, PASS),
  **range 0.543754–0.976950** (manuscript: 0.544–0.977, PASS). Note this is a *different*
  gap statistic from Section 2's causal `G` (candidate-minus-median-control on the causal
  R scale) — this one is purely structural (candidate q1 − mean control q1), computed only
  on the 12-model E11 cohort. Do not conflate the two "gap" numbers across the manuscript.
- **OLS q1 ~ log10(non_embed_params) + is_decoder, n=23**: R²=0.158104 (manuscript 0.158,
  PASS); β(is_decoder)=0.116165±0.085978, t=1.351, **p=0.1917** (manuscript
  β₂=0.1162±0.0860, p=0.192 — PASS); β(log10 params)=0.014334±0.063667, t=0.225,
  **p=0.8242** (manuscript β₁=0.0143±0.0637, p=0.824 — PASS). p-values were not stored in
  `regression_summary.json` directly (only coefficients/SEs) and were recomputed here via
  a two-sided t-test at df=n−3=20 — marked `RECOMPUTED`, not `STORED`, for that reason.
  **This regression script (`E11_scale_ladder/regression.py`) and its output
  (`results/E11/regression_summary.json`) are both untracked/gitignored** — worth
  committing given they back a headline-adjacent Fig 1 statistic.
- **Evo2-7B max activation ratio 2.22**: PASS, `STORED` from `results/e7_phase1_detection_evo2.json`
  via `PART1_STRUCTURAL_MANUSCRIPT_PACKET.md`'s transcription. Ratio 2.22 < detector
  threshold 5.0 → correctly excluded from both the structural (23-model) and causal
  (22-model) cohorts, not a missing data point.

### Provenance caveat carried through this whole section

Every number in this section traces to files under `results/E13/` and `results/E11/`,
**all of which are gitignored** (`.gitignore:22: results/`). The *code* that produced
them is a mix of tracked (`E9`, some `E13` scripts existed before this session — none,
actually: the entire `E13_full_cohort_causal_census/` and `E11_scale_ladder/` directories
are untracked, see Orientation) and untracked files. Practically: if this repository were
cloned fresh from its last commit, **none of Section 2's evidence would exist** — only
`mtime`/`sha256` (recorded in `provenance.json`) anchor these numbers to this specific
working tree, not to git history. This is the single largest provenance gap in the whole
audit and is flagged again in the Findings-of-concern list at the top of this report.

## Section 3 — Frobenius magnitude vs causal effect

**Verdict: the manuscript's Limitations text is accurate — the association is real,
weak, epsilon-dependent, and genuinely has no robustness analysis yet.** No defect found;
one nuance surfaced that the manuscript should state explicitly (the association is only
nominally significant at ε=1.0, not at ε=0.5).

### 1. What exists, and which panel

**Panel = the same 22-model causal census panel as Section 2** (not the 12-model E11
cohort, not the 5-model E10 decoder audit) — confirmed by `n_models: 22, excluded:
["Phi-3-mini-4k-instruct"]` in `results/E13/part1_22_structure_function_correlations.json`
(gitignored, mtime-only provenance), whose stored `frob_ratio_to_layer_median` correlation
(ρ=0.440994, p=0.039940, CI [0.062175, 0.717383], seed 44) reproduces exactly under
independent recomputation at **ε=1.0**. **This is the only epsilon this artifact stores** —
there is no stored artifact for the ε=0.5 version of this correlation; it exists only as
this audit's own `RECOMPUTED` value below.

Recomputed (both epsilons), `audit/scripts/section3_frobenius.py`, from `part2_22_model_results.csv`:

| epsilon | Frobenius-ratio vs. R: ρ | p | bootstrap CI (seed 44) |
|---|---|---|---|
| 0.5 | 0.348391 | 0.112070 | [−0.138798, 0.740629] — **crosses zero** |
| 1.0 | 0.440994 | 0.039940 | [0.062175, 0.717383] — excludes zero |

**The association the manuscript describes is an ε=1.0-only phenomenon.** At ε=0.5 it is
not significant (p=0.11) and the bootstrap CI includes negative values. The manuscript
should say "at full ablation" rather than leaving the epsilon unstated, since a reader
could otherwise assume it holds at both scales the way the Section 2 causal effects do.

### 2. Leave-one-out / covariate-adjusted / influence analysis: **ABSENT, confirmed**

Repo-wide search (`rg -i "leave.?one.?out|influence.?analys|covariate.?adjust"`) finds no
script or artifact performing this analysis — only prose in `PART2_EVIDENCE_PACKET.md`
and `build_part2_evidence_packet.py` stating that it has not been done. **State: absent**,
matching the manuscript's own admission exactly (not a discrepancy).

### 3. Recomputed statistics (all `RECOMPUTED`, `audit/scripts/section3_frobenius.py`)

| quantity | ε=0.5 | ε=1.0 |
|---|---|---|
| Frobenius-ratio vs R, Spearman ρ (p) | 0.3484 (0.1121) | 0.4410 (0.0399) |
| q1 vs R, Spearman ρ (p) | 0.1146 (0.6115) | 0.0740 (0.7436) |
| Frobenius LOO ρ range (n=22, one omitted each time) | [0.2844 (omit NTv3), 0.5506 (omit EuroBERT-610m)] | [0.3649 (omit NTv3), 0.5377 (omit Qwen2.5-0.5B)] |
| Frobenius vs R, text-decoders-only (n=10) | ρ=−0.0667 (p=0.855) | ρ=−0.0424 (p=0.907) |
| q1 vs R, text-decoders-only (n=10) | ρ=−0.1636 (p=0.651) | ρ=−0.3333 (p=0.347) |
| q1≥0.95 binary split, median R | high(n=13)=+0.6288%, low(n=9)=+0.3107% | high(n=13)=+37.10%, low(n=9)=+0.764% |

Note the LOO range at ε=1.0 ([0.365, 0.538]) never drops the correlation to non-significant
territory by removing any single model — the association is not driven by one outlier —
but it also never gets far from the fixed-panel point estimate (0.441), so "not
one-model-driven" should not be oversold as "robust" absent the covariate-adjusted work
the manuscript itself says doesn't exist yet. **Restricted to text decoders alone, both
associations flip sign and become non-significant at both epsilons** — the cohort-wide
association appears to be substantially carried by cross-architecture/cross-domain
variation, not a within-architecture-class effect. This is worth a sentence in the
manuscript if the Frobenius association is kept even as a secondary note, since a reader
could otherwise assume it holds within as well as across architecture classes.

### 4. q1 range-restriction distribution — **RECOMPUTED**

Across the 22-model **causal** panel (not the 23-model structural panel — one fewer,
since Evo2-7B/whatever gave 23 in Fig 1 is structural-only): **13/22 (59%) have q1>0.95,
9/22 (41%) have q1>0.97, 3/22 (14%) have q1<0.8.** This is a heavily right-skewed,
range-restricted q1 distribution, consistent with the manuscript's own framing that q1
does not predict causal magnitude well in this panel (Section 2) partly because there
isn't much low-q1 variation to correlate against in the first place.

### 5. `layer_relative_frobenius` coverage — **22/22, full coverage**

Despite only 12/22 models having their *individual* five control-row q1/‖U_k‖_F values
retained (Section 2's `census_controls_structural.csv`), the **ratio itself**
(`frob_ratio_to_layer_median`) is populated for **all 22** rows in
`part2_22_model_results.csv` — the summary statistic survived even where the per-control
breakdown that produced it did not. Do not conflate "has the ratio" (22/22) with "has the
full control spectra to recompute the ratio from scratch" (12/22) — Section 2's caveat
about the latter does not extend to this correlation analysis, which only needs the ratio.

## Section 4 — Norm-matched control feasibility

**Verdict: the Gram-identity function exists, is fast and fully vectorized as required
— and the cheap diagnostic (run on 3 small cached models, per the prompt's own
authorization to do so before stopping) shows norm-matched controls are likely
infeasible as a concept for this cohort: candidates are such extreme outliers that
*nothing* in their layer, including the existing 5 random controls, comes within 20% of
their norm.** This is a substantive, load-bearing finding for whether the proposed
control type is worth pursuing at all.

### 1. The Gram-identity function — found, and confirmed vectorizable

`manuscript/experiments/E5_dimensionality/dimensionality_lib.py:92-111`,
`exact_uk_all_rows`:

```python
def exact_uk_all_rows(W_gate, W_up, W_down, device="cuda"):
    """Exact ||U_k||_F for every output row k, via the elementwise-Gram identity:
        K = (W_gate W_gate^T) elementwise* (W_up W_up^T)      [d_ffn, d_ffn]
        ||U_k||_F^2 = W_down[k,:] @ K @ W_down[k,:]^T
    """
    G = W_gate.double().to(dev); U = W_up.double().to(dev); D = W_down.double().to(dev)
    K = (G @ G.T) * (U @ U.T)
    out = ((D @ K) * D).sum(dim=1).clamp_min(0).sqrt()
    return out.cpu()
```

`K` (a `[d_ffn, d_ffn]` matrix, ~O(d_ffn²) memory) is computed **once per layer**; `D @ K`
applies it to **every row of `W_down` simultaneously** (`D` is the full `[d_model, d_ffn]`
matrix, not one row), so **no individual `U_k` (`[d_model, d_model]`) is ever
materialized** — confirmed by direct code read, not inference from the docstring. This is
a pre-existing, already-used function (shared with `experiments/E4_granularity`'s prior
GENERator-PROK use), not something built for this audit.

### 2. Cached weights, GPU/CPU/RAM

**GPU**: 1× NVIDIA A100-PCIE-40GB, idle at diagnostic time. **CPU**: 1 core (`nproc`=1 —
this bottlenecks weight *deserialization*, not the Gram computation itself). **RAM**: 70GB
total, 57GB available.

**Cache**: the real, currently-active `HF_HOME` (`/work/11034/atzanakak/ls6/nonbdna/cache/hf`,
216GB) already holds **20 of the 22 census models** at their pinned revisions (confirmed
by directory listing, not the stale `env_cached.sh` comment) — sizes range from
`SmolLM2-135M` (260MB) to `OLMo-2-1124-7B` (28GB); the 7B-class decoders (Llama, Mistral,
OLMo, Qwen2.5-7B) are each 13–26GB. Two directories exist for GENERator-EUK-3B under
slightly different repo-name capitalizations — worth a one-line note to the manuscript
authors that this could indicate a stale duplicate download, not an audit blocker.
**Note**: there is a second, smaller, essentially-empty `HF_HOME`
(`/work/11034/atzanakak/ls6/huggingface`, containing only Evo2-7B) that several
provenance docs (`PROVENANCE_AND_BASELINES.md`) describe as "the" cache — the actual
working cache with everything else in it is the one under `ls6/nonbdna/`. This is a minor
documentation-vs-reality mismatch, not a data problem.

### 3. Cheap diagnostic — run on the 3 smallest cached E11-panel models, as authorized

Ran (not merely estimated) the full within-layer `‖U_k‖_F` distribution for SmolLM2-135M,
Qwen2.5-0.5B, and SmolLM2-360M — the three smallest models with both cached weights and a
known candidate coordinate — using the exact function from point 1, reusing
`E11_scale_ladder/run_model.py`'s existing loader/weight-extraction code (no new model
code written). Script: `audit/scripts/section4_norm_matched_diagnostic.py`, output:
`audit/norm_matched_feasibility.csv`.

| model | layer | d_model (rows in layer) | candidate norm | layer median | rank | percentile | rows within ±10% | rows within ±20% | load time | compute time (all rows) |
|---|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M | 11 | 576 | 2002.997 | 160.137 | **1/576** | 100.0 | **0** | **0** | 187.8s | **1.25s** |
| Qwen2.5-0.5B | 21 | 896 | 18.727 | 0.591 | **1/896** | 100.0 | **0** | **0** | 92.5s | **0.22s** |
| SmolLM2-360M | 3 | 960 | 5692.496 | 252.406 | **1/960** | 100.0 | **0** | **0** | 62.4s | **0.014s** |

**The Gram-identity compute cost is negligible** (≤1.3s to compute all rows in a layer,
regardless of layer width) — essentially all wall-clock cost is weight *loading*
(60–190s here, and that's inflated by this node's single CPU core doing
safetensors deserialization; a multi-core node would load faster with no change to the
compute step).

### 4. Do the existing 5 random controls include any row within ±20% of the candidate norm? **No, in all 3 models tested**

```
SmolLM2-135M controls: 66→7.55%, 436→8.16%, 514→8.35%, 529→8.01%, 558→7.86% of candidate norm
Qwen2.5-0.5B controls: 196→3.51%, 444→3.38%, 520→3.04%, 815→3.13%, 818→3.27% of candidate norm
SmolLM2-360M controls: 16→4.51%, 337→4.50%, 782→4.40%, 913→4.47%, 951→4.34% of candidate norm
```

Not one of the 15 existing control rows across these 3 models comes anywhere near 20% of
the candidate's norm — they cluster in a narrow 3–8% band, an order of magnitude below
the candidate. **This generalizes the ±10%/±20% count of exactly 0 found for every row in
the entire layer (not just the 5 sampled controls)** — the candidate is such an extreme
single-row outlier (12–32× the layer median) that **there may be no row in these layers to
norm-match against at all**, which is itself the answer to whether norm-matched controls
are feasible as a concept here, before any cost question. If this pattern holds across the
other 19 cached models (plausible given q1's own extreme concentration, Section 3), a
"norm-matched control" protocol may need a materially looser matching tolerance (e.g.
nearest-available rather than a fixed ±10/20% band) to be constructible at all — worth
flagging to the manuscript authors before committing to this control type.

### 5. Cost estimate for a full 5-row norm-matched sweep, both ε, all 22 models — **projected, not run**

The Gram-identity *selection* step (which rows to use as norm-matched controls) is
essentially free per model (≤1.3s once weights are loaded — point 3). The **actual cost
driver is identical to what the existing census's random-control measurements already
paid**: 5 new control rows × 2 epsilons × (candidate's existing evaluation harness) per
model, since norm-matching only changes *which* rows are selected, not how they're
evaluated. This is not a new kind of cost, it is the same measurement the census already
ran once, run again for 5 different row indices. **Models that will dominate wall-clock**:
the four 7B-class text decoders (Llama-7B, Mistral-7B, OLMo-7B, Qwen2.5-7B — 13-26GB
weights, load-time-dominated on this single-CPU-core node) and the two largest GENERator
variants (EUK-3B, PROK-3B — generation-endpoint evaluation is markedly more expensive per
condition than the teacher-forced-NLL endpoints used elsewhere, per Section 5's ~216s/condition
figure). Not run — reported as a projection only, per the prompt's cost-then-wait
instruction; an exact GPU-hour figure would need either timing the census's original run
(not recorded per-condition in any artifact found) or running one more model's control
sweep as a timing pilot.

## Section 5 — GENERator damage-matching

**Verdict: every headline number PASSES against stored artifacts, and the raw data
answers point 3 far more decisively than the manuscript states it.** No inconsistency
found; one important nuance surfaced (see point 3).

### 1. Verification of the six manuscript numbers

All read from `results/E12/` (untracked, gitignored — see provenance caveat below).

| claim | manuscript | artifact | status |
|---|---|---|---|
| baseline GC | 0.4204 | 0.4204 (`E12_summary.md`, cross-checked against `dose_response_extended.csv` alpha=1.0 mean) | PASS |
| post-ablation GC | 0.3065 | 0.3065 | PASS |
| ΔGC | −0.1139 | −0.1139 | PASS |
| row-2371 ablation NLL | 8.754 | 8.754319605827332 (`damage_evals.jsonl`, `label=row2371, value=0.0`) | PASS |
| largest control damage | ≈6.386 | **6.386866** (control_row_1126 at α=8.0) — rounds to **6.387**, not 6.386, at 3 decimals | PASS with a rounding note |
| random-direction NLL | 8.7536 | 8.753589 (`damage_matching.csv`, matched at c=0.0125) | PASS |
| random-direction GC | (not explicitly named but implied ≈0.307) | 0.306672 (`matched_gc_mean`) | PASS |
| longest-homopolymer GC-decrease fraction | 3.9% | 3.9282% (`kmer_attribution.csv` / run log) | PASS |

The "largest control damage ≈6.386" figure is technically imprecise (true value rounds to
6.387), but this is display-precision noise, not a different underlying number — flagged
for completeness per the hard rules, not because it changes any conclusion.

### 2. `generator_alpha_sweeps.csv` — emitted

78 rows: full α-grid `nll` (from `damage_evals.jsonl`, the 100-window damage pool) joined
with `gc_fraction`/bootstrap CI (from `dose_response_extended.csv`, the 96-window
generation pool, aggregated over prompts×seeds; CI recomputed here with a 5,000-draw
percentile bootstrap, seed 42) for row 2371, all five control rows, and the
random-direction condition. All values `STORED`/`RECOMPUTED`, script:
`audit/scripts/section5_generator_sweeps.py`.

### 3. Maximum achievable NLL per control row — **the sweep is not "still rising," it is nearly flat and 2.37 NLL units short**

This is the most important finding in this section, and it sharpens what the manuscript
already implies but doesn't state numerically:

```
control_row_102:   damage range [6.385381, 6.385612] (max at alpha=8.0)  gap to target = 2.368707
control_row_1126:  damage range [6.385232, 6.386866] (max at alpha=8.0)  gap to target = 2.367454
control_row_2621:  damage range [6.384125, 6.385680] (max at alpha=0.0)  gap to target = 2.368640
control_row_3039:  damage range [6.385321, 6.386403] (max at alpha=8.0)  gap to target = 2.367916
control_row_456:   damage range [6.385343, 6.385956] (max at alpha=8.0)  gap to target = 2.368363
```

**None of the five control rows are "still rising" toward the target in any meaningful
sense** — each one's damage moves by only ~0.001–0.0016 NLL across the *entire* 0-to-8.0
α range (an 8× multiplicative range on the row's own weight scale), while the gap to the
8.754 target is ~2.37 NLL units, three orders of magnitude larger than the observed
sensitivity. Extending the sweep past α=8.0 would not close this gap — these rows are
simply not causally sensitive at all, not "almost there." This is a much stronger and more
falsifiable statement than "unreachable on grid" / "FALLBACK used," which could otherwise
be misread as "we just didn't sweep far enough."

**The random-direction condition is a qualitatively different kind of "unreachable"**: its
damage at scale 0 *is* the target by construction (zero perturbation = baseline), and
damage **decreases monotonically** as the random-direction scale increases (8.754 at c=0
down to 8.375 at c=8.0 — see `damage_matching.csv`'s `random_direction` grid). So the
matcher's fallback point (c=0.0125, the smallest nonzero grid step) is not "closest
approach from below," it's "closest point to a target that can only be moved away from."
The manuscript's Methods should distinguish these two failure modes explicitly if the
random-direction control is kept: control-row insensitivity vs. wrong-direction
monotonicity are different findings with different implications for what "specificity"
means here.

### 4. Joint multi-row perturbation support — **absent in the GENERator harness, present elsewhere in the repo**

`e12_lib.py`'s entire intervention surface (`set_row_alpha`, `set_row_random_direction`,
the `row_scaled` context manager, `damage_under_row_alpha`) takes a single `layer: int,
row: int` throughout — **no list-of-rows parameter anywhere**, confirming the Methods
text's "single-row context manager" description exactly.

However, the repo already has a **generalized multi-row engine that does exactly this**,
built for DNABERT-2's tomography (Section 1): `manuscript/experiments/E9_mechanistic_tomography/tomography_lib.py`'s
`_scale_rows`/`with_mask` (accepting `coords: Sequence[tuple[layer,row]]` and per-coordinate
alphas), itself built on the same `_save_row`/`_restore_row`/`_resolve_module` primitives
`e12_lib.py` also uses (from `run_gue_ablation.py`). **Entry point for adapting it**: the
underlying row save/restore/resolve primitives are architecture-generic and already
shared between E9 and E12; what's missing for GENERator specifically is (a) generalizing
`e12_lib.row_scaled`/`set_row_alpha` to accept a coordinate list the way `tomography_lib._scale_rows`
does (small, mechanical change — the pattern already exists to copy), and (b) verifying
GENERator's `down_proj` pattern resolves correctly across multiple, possibly
different-layer coordinates simultaneously (not verified either way here — no code
currently exercises that path for GENERator).

### 5. Cost estimate for a matched-damage-elsewhere joint control — **projected, not run**

From `results/E12/run_e12_full.log`: each single-row evaluation condition (96 generation
prompts + 100 damage windows, one row/alpha/seed combination) took **~216 seconds**
wall-clock on this node's GPU; the full 90-condition B3 sweep took **~5.4 GPU-hours**.

A binary-search matched-damage-elsewhere control (perturb k random non-2371 rows jointly,
search on scale until held-out damage ≈ 8.75, then measure GC over 96×3 seeds) would need,
per candidate k: roughly **8–12 binary-search damage-only evaluations** (damage-only is
cheaper than damage+generation — the damage endpoint alone, from `damage_evals.jsonl`'s
timing pattern, is a small fraction of the ~216s figure since that figure includes
generation; a damage-only eval over 100 windows is closer to the per-window NLL cost
already paid in Section 2's genomic-decoder endpoint, on the order of tens of seconds),
plus **3 seeds × 96-prompt generation runs at the matched scale** (~216s × 3 ≈ 11 minutes)
once the scale is found. **Rough estimate: 15–30 minutes of GPU time per k tried** (search
+ final measurement), so a handful of k values (e.g. k∈{2,5,10}) would be on the order of
**1–1.5 GPU-hours total** — small relative to the 5.4 GPU-hours already spent on the
existing sweep. **Blocker**: this requires implementing the joint-perturbation change
described in point 4 first; no code currently does joint ablation for GENERator. **Not
run** — this is a projection only, per the prompt's "report projected cost, wait for me
before committing compute" instruction.

### 6. hg38 pool non-overlap — **RECOMPUTED independently, confirmed non-overlapping**

Per the prompt's explicit instruction not to take the docstring's word for it: loaded the
**actual stored sampled window coordinates** from `results/E12/raw/corpora.json` (96
generation-prompt windows, 100 damage windows) and computed all 9,600 pairwise
chromosome-interval intersections directly (`audit/scripts/section5_hg38_overlap.py`).
**Result: 0 overlapping pairs.** The pools are genuinely disjoint, independently verified
from coordinates, not merely from `build_corpora`'s own internal assertion or the
docstring's claim.

### 7. BOS/attention-sink reproduction — **STORED, all four numbers PASS**

`results/mechanism/attention_sink_implicit_bias.json` (untracked — see provenance note),
produced by `scripts/mechanism/run_attention_sink.py` (git-tracked at `3d3bbb0`, but with
an **uncommitted local modification** — a hardcoded hg38 path fix, dated after that commit
— so the JSON reflects the *current, uncommitted* version of the script, not the last
committed one):

| claim | manuscript | artifact | status |
|---|---|---|---|
| mean incoming attention at position 0 | 37.9% | `sink_share_pos0` = 0.379155 | PASS |
| sink over uniform | 33.0× | `sink_over_uniform` = 32.986 | PASS |
| frac. layer-head observations with argmax at pos 0 | 78.8% | `frac_heads_argmax_pos0` = 0.7875 | PASS |
| n windows, seed | 40, seed 42 | `n_windows`=40, `seed`=42 | PASS |

**Unrequested observation, flagged per the prompt's closing instruction**: the same JSON's
`implicit_bias` block reports `pos0_shuffled_mean` exactly equal to `pos0_real_mean`
(ratio = 1.0 to full float precision). An exact 1.0 ratio between a "real" and "shuffled"
condition is a plausible sign that the shuffle step is a no-op (e.g. shuffling a copy that
isn't used, or shuffling before the measurement point) rather than a genuine null result —
worth a five-minute look at that code path before this implicit-bias sub-result is used
for anything, though it is not one of the numbers the manuscript currently cites.

### Provenance caveat

Every artifact in this section (`results/E12/`, `results/mechanism/attention_sink_implicit_bias.json`)
is untracked/gitignored, consistent with the pattern already flagged for Sections 2–3. The
code that produced the GENERator numbers (`E12_generator_degradation_control/`) is also
entirely untracked. `run_attention_sink.py` is the one exception with real git history,
but its current on-disk version has an uncommitted path fix.

## Section 6 — DNABERT-2 detector reconciliation

**Verdict: fully resolved — and it was already fully resolved before this audit, by a
dedicated diagnostic already in the repo** (`results/E13_dnabert2_reproducibility/DNABERT2_EXECUTION_PATH_DIAGNOSTIC.md`,
committed nowhere — untracked, like the rest of E13 — but present and complete on disk).
This audit verified that diagnostic's own claims rather than trusting it blindly, and
extended it with two things it didn't cover: the ratio-argmax coordinate's structural
metrics (point 4), and a cohort-wide detector-provenance table for the other 21 models
(point 5).

### 1–3. Code paths, diff, and which one reproduces which number

**Two code paths, differing in exactly one flag.** Both are documented and both were
re-run as part of the pre-existing diagnostic (`run_dnabert2_repro_diagnostic.py`, 4 cells
crossing {default, eager} attention × {historical special tokens, E13 no special tokens}):

| path | `add_special_tokens` | tokens | out_max (L5/r603) | ratio (L5/r603) | activation rank | ratio rank | passes ratio≥5 |
|---|---|---|---|---|---|---|---|
| historical (discovery wrapper default) | `True` | 116 | **944.5556030273438** | **152.734548** | **1** | 8 | **True** |
| current E13 (`run_rowwise_detector.py` / no-special-token structural census) | `False` | 114 | 0.573138 | **0.999394** | 2749 | 4628 | False |

**The single line responsible**: `add_special_tokens=True` (historical, default
tokenizer call) vs. `add_special_tokens=False` (current E13 path, explicit). Confirmed by
the diagnostic's own controlled test: attention implementation (eager vs. default/PyTorch
fallback — Triton isn't even available in this environment) was crossed against both and
made **zero difference**; historical-specials and explicit-eager-specials cells are
bit-identical across all 9,216 rows, and the two no-specials cells are likewise identical
to each other. **This fully explains the "≈0.999 contradicts 152.7345" tension the prompt
raised — it is not a contradiction, it is two different, both-correct measurements under
different input-boundary preprocessing**, and the diagnostic's structural-metrics check
(point C, q1/PR_spec/‖U_k‖_F for (5,603) identical across all 4 cells) rules out any
model-state/checkpoint inconsistency as an alternative explanation.

**Canonical path**: the diagnostic recommends (and this audit agrees, having verified the
underlying numbers) the **historical default-special-token path** as canonical for claims
tied to the established discovery coordinate, since it is the only one that exactly
reproduces the frozen historical artifact (944.5556030273438, to full float precision).

### 4. Structural metrics of the ratio-argmax coordinate — **RECOMPUTED, and it changes the framing**

Under the historical (canonical) path, the frozen candidate L5/r603 is the **global
activation-argmax**, but **not** the global ratio-argmax — that is **L8/r603** (ratio
302.022, per the diagnostic's own stored `execution_path_comparison.csv`). This audit
loaded DNABERT-2 (weights-only — structural metrics don't depend on tokenization, per
point C above) and computed both coordinates' exact spectral metrics via the same
`row_spectral_metrics` function the rest of the structural analysis uses
(`audit/scripts/section6_ratio_argmax_structural.py`):

| coordinate | role | q1 | PR_spec | ‖U_k‖_F |
|---|---|---|---|---|
| L5/r603 | frozen candidate (activation-argmax) | 0.793311 | 1.503288 | 74.008558 |
| L8/r603 | ratio-argmax (not selected) | **0.378681** | **6.759701** | 10.171930 |

**This matters**: L8/r603's q1 (0.379) is *below the minimum q1 in the entire 23-model
structural panel* (NTv3's 0.389, Section 2). Had the strict ratio-argmax rule been applied
to DNABERT-2 instead of the activation-argmax convention it was actually grandfathered
under, the resulting "candidate" would be one of the *least* structurally concentrated
rows in the whole cohort (spread over an effective ~6.8 singular directions per PR_spec,
vs. the frozen candidate's near-rank-1 concentration) — the opposite of the
"structurally exceptional" framing Part 1 uses for every other accepted candidate. **This
would not just change a number, it would flip DNABERT-2's structural story if the strict
rule were retroactively applied** — a concrete reason to keep the historical
activation-argmax convention explicit and separately justified in Methods, not merged
silently into the "current ratio rule" framing used for the other 21 models.

### 5. Audit of the other 21 models — **detector_provenance.csv emitted; 6/22 share DNABERT-2's grandfathering pattern**

Rather than re-running full detection on all 22 models (a much larger compute commitment
this audit did not take without authorization), this audit checked what each model's
**existing** source detection artifact actually records. This produced a clean,
well-evidenced split:

- **16/22 models** (all E11-, E8-, and E7-phase1-sourced candidates — the full E11
  12-model cohort, MosaicBERT, ModernBERT-base, Qwen2.5-7B, GenomeOcean-4B) have an
  explicit `ratio`/`detection_ratio` field recorded **at the selected candidate
  coordinate**, at or above the 5.0 threshold (e.g. Qwen2.5-0.5B: ratio 547.74; MosaicBERT:
  ratio 288.65) — confirming these were selected **fresh, under the current ratio-argmax
  rule**, by construction of the code that produced them.
- **6/22 models** (Llama-7B, Mistral-7B, OLMo-7B, NTv3, DNABERT-2, GENERator-EUK-3B — all
  sourced from `results/e7_legacy_reanalysis.json`) have **no ratio field at all** in
  their source artifact — confirmed by direct inspection, not inferred. This is
  DNABERT-2's exact situation, generalized: **all six were grandfathered from an older,
  activation-based selection convention that predates the ratio≥5.0 rule**, and none of
  the other five have been checked against the current rule the way DNABERT-2 was in the
  pre-existing diagnostic this audit verified. **This audit did not re-run detection for
  the other 5** — `detector_provenance.csv` marks their ratio-argmax status `NOT FOUND`
  rather than guessing, since only DNABERT-2 has actually been checked.

This is the single most load-bearing open item this audit did not close: **5 more models
(Llama-7B, Mistral-7B, OLMo-7B, NTv3, GENERator-EUK-3B) may have the same
activation-vs-ratio inconsistency DNABERT-2 had**, unverified either way. Re-running each
through the same diagnostic pattern used for DNABERT-2 would resolve this; the DNABERT-2
diagnostic took a single dedicated script and one small model, so the marginal cost per
additional model is modest (dominated by model load time, per Section 4's timing data —
these five include three 7B-class models, so not free, but not a large commitment either).

### 6. HF revision recoverability — **RECOMPUTED across all 22; NTv3 is the only unrecoverable one**

Read `resolved_revision` for all 22 models directly from `part2_22_model_results.csv`:
**21/22 have an explicit resolved Hub commit hash recorded** (including Llama-7B,
Mistral-7B, and OLMo-7B, all three of which were *requested* unpinned but still resolved
to a specific commit at run time and had that commit recorded). **Only NTv3** has
`resolved_revision = "unpinned (E5/E6 original)"` — genuinely unrecoverable, exactly
matching the manuscript's own admission, and **confirmed no other model shares this gap**.

### Deliverable

`audit/detector_provenance.csv` — 22 rows: `model, candidate_layer, candidate_row,
candidate_source, selection_rule, frozen_candidate_is_ratio_argmax,
frozen_candidate_is_activation_argmax, requested_revision, resolved_revision,
revision_recoverable, note`.

---

## Overall verdict

**No manuscript number checked in this audit was found to be wrong.** Every one of the
~90 individual claims logged in `verification_table.csv` across all six sections is
`PASS` against a stored or independently recomputed artifact, with two exceptions that are
`NOT FOUND` as pre-existing artifacts but were recomputed fresh in this session (the
ε=0.5 Frobenius correlation, Section 3; the ratio-argmax coordinate's structural metrics,
Section 6) and one `PARTIAL` (Section 6's 21-model detector audit, where DNABERT-2 itself
is fully resolved but 5 more models remain unverified either way).

**What this audit did not find**: no fabricated numbers, no numbers that fail to
reproduce, no internally-inconsistent arithmetic anywhere it checked. The manuscript's own
evidence discipline (`manuscript/CLAUDE.md`'s claim-ledger requirement) appears to be
working — every number traced back to a real, if sometimes untracked, artifact.

**What this audit does flag as needing attention before this goes further**, ranked:

1. **Section 6's open 5-model gap** (Finding 6) — the highest-priority remaining question,
   since it's the same class of issue the DNABERT-2 case turned out to be real and
   consequential (q1 dropping below the cohort minimum under the strict rule).
2. **Total absence of git history for Sections 2–6's evidence base** (Finding 4) —
   `results/E11/`, `results/E12/`, `results/E13/`, and the experiment code that produced
   them are gitignored or untracked. Every number this audit checked is currently correct
   *on this machine, right now* — none of it is protected against loss, and none of it
   would survive hand-off to a co-author who clones the repo.
3. Two informational/resolved items (Findings 3 and 5) that don't require action beyond
   what this report already states.

Any code written for this audit lives in `audit/scripts/` and is deterministic and
rerunnable against the artifacts cited (paths are relative to the repo root).
