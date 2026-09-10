# Supplementary tables

Machine-readable supplementary tables for *"Structure is not mechanism."* Each file is
tab-separated with a single header row.

| table | file | rows | source artifacts |
|---|---|---|---|
| **S7** | `table_S6_within_layer_activation_ratio_sweep.tsv` | 72 (36 per model) | `../within_model_slope.json`, `../within_model_slope_smollm2_1.7b.json` |
| **S7** | `table_S7_smollm2_l7_interaction_and_geometry.tsv` | 12 (6 ablation + 6 geometry) | `../smollm2_second_row_epistasis.json`, `../smollm2_161_749_geometry.json` |

Regenerate both from the JSON artifacts with the snippet recorded in the commit that added
them; no value here is transcribed by hand.

**S5 columns.** `model, layer, row, ratio_rank, activation_ratio, nll, delta_nll, rel_delta,
above_detector_threshold, is_frozen_candidate`. `rel_delta` is the relative native-loss
increase under α=0 ablation of that single row; `above_detector_threshold` flags
`activation_ratio ≥ 5.0`, the detector's own acceptance rule, and is the partition used in the
main text.

**S6 columns.** `panel` separates the `ablation` block (three single-row and three pairwise
conditions) from the `geometry` block (three row norms and three pairwise cosines). In the
ablation block `interaction = effect_rel − sum_of_singles`; because the endpoint is a **loss
increase**, a positive interaction denotes super-additivity — the opposite sign convention to
this project's accuracy-endpoint epistasis numbers (e.g. DNABERT-2's −33.63 pp), which must be
converted before comparison. In the geometry block `joint_over_sum` carries the row's L2 norm
or the cosine's z-score against the 19,900-pair reference distribution, as stated per row.

## Status of S1–S4

**S1–S4 are described in the manuscript's supplementary list but do not exist as table files
in this repository.** They were not produced by this round and are not blocked by anything
here; the underlying values live in `audit/census_master.csv` (S1 provenance, S2 structural
metrics, S3 causal census) and `../activation_vs_causality.tsv` (S4 structure–function
correlations). Building them is a separate, straightforward task.
