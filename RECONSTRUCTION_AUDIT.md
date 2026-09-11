# Reconstruction audit — TMLR submission prep

Branch `tmlr-submission-prep`. Task 1 (inventory only, no edits).

## Baseline regression gate

```
50 CLI entry points: 50 ok, 0 skipped, 0 broken
117 non-CLI scripts were not executed (checked statically instead)
```

`python tests/test_entrypoints.py` — this is the gate to re-run after every path-touching change.

**Known limitation of the gate**, relevant below: it executes only the 50 scripts that use
`argparse` (where `--help` exits before work), and for the other 117 it verifies only that
`sys.path` inserts resolve. It does **not** validate data paths. Issue 8 was found by hand, not
by the gate.

---

## Issue 1 — the "empty stale tracked directories" are not tracked

**Task 2's premise does not hold.** Git does not track empty directories. Every directory on
that list has **zero tracked files**, so none of them appears in a fresh clone at all; there is
nothing to remove for the submission artifact.

| directory | tracked | files on disk | gitignored | referenced by tracked code |
|---|---|---|---|---|
| `results/mechanism/` | 0 | 96 | yes | **20 files** |
| `results/sae/` | 0 | 21 | yes | 1 |
| `results/prokaryote/` | 0 | 17 | yes | 0 |
| `results/compression/` | 0 | 15 | yes | **4** |
| `results/gue_checkpoints/` | 0 | 10 | yes | **21** |
| `results/gue_checkpoints_multiseed/` | 0 | 18 | yes | (same 21) |
| `results/paper_closing/` | 0 | 2 | yes | 0 |
| `results/svd_spectral_plots/` | 0 | 4 | yes | 0 |
| `results/svd_spectral_plots_smoketest/` | 0 | 1 | yes | 0 |
| `results/pipeline_logs/` | 0 | 1 | yes | **2** (the pipelines' `LOGDIR`) |
| `results/E13/` | 0 | 0 | yes | 0 |
| `audit/round2/` | 0 | 0 | **no** | 0 |

Two consequences:

1. **Deleting them would destroy the authors' local, gitignored output** — 96 files under
   `results/mechanism/`, 21 under `results/sae/`, 18 checkpoints, and so on. It would not change
   the clone by one byte.
2. **Four are live output targets** named in tracked code. Per Task 2's own rule ("if any of
   these names is still needed as an output target, do NOT delete"), these must stay:
   `results/mechanism/` (default `--out` for `run_granularity_multimodel.py`,
   `run_norm_matched_control.py`, `run_sw_pairwise_epistasis.py` and others),
   `results/gue_checkpoints{,_multiseed}/` (GUE fine-tuning checkpoint roots),
   `results/compression/`, `results/pipeline_logs/`.

**Only `audit/round2/` and `results/E13/` are pure leftovers** — empty on disk, untracked,
referenced by nothing. Removing them is a local-tidiness no-op with no effect on the submission.

## Issue 2 — `results/experiments/keep/E4_granularity/`

| file | tracked |
|---|---|
| `PROVENANCE.md` | yes |
| `e4_granularity.json` | yes |
| `e4_ntv3_shared_adapter.json` | yes |

Task 2 names only the `PROVENANCE.md`. Note the directory also holds **two tracked data
artifacts** from the same cut line. Removing the provenance note while keeping its data would
leave the data unexplained — worse than either keeping or removing both. Flagged for a decision.

## Issue 3 — preregistrations: 17 files, 12 locked

Locked and verifying (12): `full_cohort_causal_census`, `E11_scale_ladder`,
`E12_generator_degradation_control`, `E10_nlp_architecture_causal`, `..._v2`,
`E10b_phi3_tomography`, `mechanistic_tomography_E9`, `exact_operator_dimensionality`,
`encoder_decoder_dimensionality`, `cross_geometry_stageA`, `dimensionality_gate0`,
`evo1_broadcast`.

Present but **not** in the lock ledger (3): `PREREG_evo1_broadcast_v2_source.md`,
`PREREG_nlp_prospective.md`, `PREREG_steering.md`.

Plus `LOCKS.jsonl` and `archive/README.md` = 17 entries.

Preregs for lines **not** in the paper: `cross_geometry_stageA`, `evo1_broadcast` (+ its v2
source), `dimensionality_gate0`, `steering`, `nlp_prospective`. Task 3's disposition table needs
one row per file; the paper-status column requires author input for `nlp_prospective` and
`steering`, whose scope I cannot infer from the tree.

## Issue 4 — `experiments/E5_dimensionality/` holds one file

`dimensionality_lib.py`, imported by three surviving builders:
`audit/scripts/section4_norm_matched_diagnostic.py`,
`audit/rederivations/scripts/section4a_batch2.py`, `.../section4a_topk_norm_q1.py`.
Spec forbids moving it. No action.

## Issue 5 / Task 7 — the two figure trees

`experiments/figures/` (49 files, 12 py) is authoritative.
`audit/rederivations/scripts/` (18 files) is the re-derivation set.

**Task 7's removal option (b) is unsafe.** Nearly every script there builds a *tracked*
artifact:

| script | tracked output |
|---|---|
| `section2_split_stability.py` | `tomography_split_stability.csv` — **feeds Supplementary S6** |
| `row_index_recurrence.py` | `row_index_recurrence.csv` |
| `norm_vs_random_controls.py` | `norm_controls_vs_random_controls.csv` |
| `fig1c_full22.py`, `fig1c_vs_topk_gaps.py` | `fig1c_random_vs_topk_gaps.csv` |
| `fig1_full22_two_panel.py`, `fig2_panel_topnorm.py` | `fig1c_random_vs_topk_gaps_full22.csv` |
| `olmo_row269_structural.py` | `section3_text_decoder_calibration.json` |

Also: 9 files under `audit/rederivations/scripts/` are named `section*.py`. They do not match the
protected glob `audit/*/section*.py` (which is one level deep) but are load-bearing for the same
reason. **Proceeding with option (a), archival marking**, as the spec prefers when unsure.

## Issue 6 — `audit/rederivations/final_check.md`

The last surviving audit narrative; the other round reports were removed earlier. Task 7 says to
keep it and say so in the archival README. No conflict.

## Issue 7 — `scripts/` vs `experiments/`

`experiments/` = 67 py (preregistered harnesses, E1–E13 + figures).
`scripts/` = 50 py (derived analyses). Out of scope per the spec. No action.

## Issue 8 — NEW: Figure 4's render script loads a moved file

Not in the original seven. `experiments/figures/fig4_generator.py:79` executes

```python
sink = json.loads((RESULTS / "mechanism" / "attention_sink_implicit_bias.json").read_text())
```

but that artifact now lives at `results/analyses/mechanism_generator/attention_sink_implicit_bias.json`.
Figure 4 therefore cannot render. The gate missed it because `fig4_generator.py` has no
`argparse`, so it is never executed, and the static check covers `sys.path` inserts only.

This is a one-line data-path fix in an authoritative figure script, and it should be made before
Task 4 claims Figure 4 is reproducible. Flagged for Checkpoint 1.

---

# Task 2 (revised per ruling) + Issue 8 — executed

## Issue 8 — HIGH PRIORITY, and it was not one bug but six

`experiments/figures/fig4_generator.py` was the symptom. Generalising it into a gate
(`tests/test_figure_inputs.py`) found that **5 of 10 figure scripts could not render** from the
committed tree:

| script | input it could not find | now |
|---|---|---|
| `fig2_causal.py` | `results/e10_exact_uknorm_olmo.json`, `results/e10b_phi3_fit_results.json` | `results/experiments/E10/…` |
| `fig4_generator.py` | `results/mechanism/attention_sink_implicit_bias.json`, `results/e7_legacy_reanalysis.json` | `results/analyses/mechanism_generator/…`, `results/experiments/E7/…` |
| `fig4_generator_specificity.py` | both of the above | same |
| `supplement/fig_s2_random_direction.py` | `audit/round2/generator_random_direction_full.csv` | `audit/rederivations/…` |

All 10 now resolve every declared input. The gate was itself validated by injecting a wrong
path, confirming it fails, and reverting.

Two rounds of false positives in the gate were fixed rather than worked around: it originally
guessed the base directory, and then could not follow a chained base
(`E12 = RESULTS/"experiments"/"E12"`). It now parses base assignments and resolves them to a
fixed point, so `fig4_generator_specificity.py`'s inputs resolve correctly.

**This is why Task 4 could not have honestly claimed reproducibility beforehand.** Figures 2
and 4 and Supplementary Fig. S2 would all have been listed as reproducible while being
unrenderable.

## Task 2 — corrected scope

Removed, as the only two pure leftovers (0 tracked files, 0 files on disk, referenced by
nothing — each contained one empty subdirectory, which is why `rmdir` initially refused):

* `audit/round2/` (held an empty `figures/`)
* `results/E13/` (held an empty `priority1_robustness/`)

**Not removed**, per the ruling and Task 2's own live-output-target rule: `results/mechanism/`,
`results/sae/`, `results/prokaryote/`, `results/compression/`, `results/gue_checkpoints{,_multiseed}/`,
`results/paper_closing/`, `results/svd_spectral_plots{,_smoketest}/`, `results/pipeline_logs/`.
None is tracked, so none reaches a clone; four are live `--out`/checkpoint/log targets named in
tracked code; together they hold 185 gitignored local files belonging to the authors.

### Reviewer-facing scope check (the replacement for directory deletion)

Grepped all tracked markdown for the dead lines. Result: **no dangling reference a reviewer
would read.**

* `README.md` — mentions E2/E4/E6, the SAE withdrawal and shadow redundancy **deliberately**, in
  the retractions and known-limitations sections. That is the transparency the ruling asks for.
* `data/README.md` "prokaryote" — the committed *E. coli* region files, which are live.
* `README.md` "hybridna" — the `stubs/` description, live.
* `experiments/E7_exact_dimensionality/{RESULTS,MODEL_PANEL}.md` — refer to "E5/E6" as
  historical context for the panel's own construction. Left as provenance prose; README item 4
  now explains what happened to those lines, and E5's surviving library is called out there.
* `experiments/E9_*/RESULTS.md`, `experiments/E10b_*/E10B_SYNTHESIS.md` "SAE" — false positives:
  both discuss an SAE-derived *basis* as a methodological alternative considered, not the
  removed package.

## E4 — removed all three, per ruling

`results/experiments/keep/E4_granularity/{PROVENANCE.md, e4_granularity.json,
e4_ntv3_shared_adapter.json}`. Checked first, as instructed: the only tracked references to
either JSON were in this audit document, and the string `E4` appears **0 times** in both
`docs/EXPERIMENT_MAP.tsv` and `experiments/figures/FIGURE_PROVENANCE.md`. No live reader.

## Gate status after these changes

```
50 CLI entry points: 50 ok, 0 skipped, 0 broken     (unchanged from baseline)
10 figure scripts checked, 0 missing input path(s)  (was 6 missing)
```
