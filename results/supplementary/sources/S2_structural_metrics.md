# S2 -- Structural metrics

Sources: candidate q1 / PR_spec / ||U_k||_F / layer-relative Frobenius -- `audit/census_master.csv` (STORED). Layer median norm -- `audit/rederivations/topk_norm_controls.csv` col `layer_median_norm` (STORED, all 22 models). Mean random-control q1 and both gap columns -- `audit/rederivations/fig1c_random_vs_topk_gaps_full22.csv` (STORED, reused verbatim per task spec). Mean top-norm-control q1 -- RECOMPUTED as the average of the 5 `rank2_by_norm`..`rank6_by_norm` q1 values per model from `audit/rederivations/topk_norm_q1_summary.csv`.

**Double-check performed:** candidate q1 (census_master.csv) vs candidate_q1 (`full22_topk_gap_confounds.csv`), and frobenius_norm (census_master.csv) vs candidate_norm (`topk_norm_controls.csv`), for all 10 batch-2 models (the models for which census_master.csv predates round 2). **Result: no discrepancy** -- all values agree to within ~1e-9 to 1e-13 relative difference (float-precision-level noise from independent recomputation runs, not a substantive disagreement). Full per-model reldiffs were computed; none exceeded 1e-6.

**Gap definitions (important, not a data error):** `topk_by_norm_gap` (reused verbatim from `fig1c_random_vs_topk_gaps_full22.csv`) is `candidate_q1 - MAX(top-5-by-norm control q1s)` -- the gap to the single toughest (highest-q1) top-norm competitor, per `audit/rederivations/scripts/full22_cohort_summary.py` line 52. This is a **different quantity** from `candidate_q1 - mean_topk_norm_control_q1` (the mean-based gap implied by this table's own `mean_topk_norm_control_q1` column) -- the two are not expected to be arithmetically consistent with each other, by design. `random_control_gap` is `candidate_q1 - mean(5 random control q1s)`.

| Model | Candidate q1 | PR_spec | ‖U_k‖_F | Layer median norm | Layer-rel. Frobenius | Mean random-ctrl q1 | Mean top-norm-ctrl q1 | Random-ctrl gap | Top-norm gap (vs max) |
|---|---|---|---|---|---|---|---|---|---|
| DNABERT-2 | 0.7933 | 1.503 | 74.01 | 10.46 | 5.018 | 0.07057 | 0.3668 | 0.7227 | -0.07311 |
| EuroBERT/EuroBERT-2.1B | 0.9977 | 1.005 | 2.751 | 0.1317 | 20.89 | 0.04853 | 0.9962 | 0.9492 | -5.801e-05 |
| EuroBERT/EuroBERT-210m | 0.9838 | 1.033 | 2.042 | 0.1374 | 14.95 | 0.1341 | 0.9828 | 0.8497 | -0.001297 |
| EuroBERT/EuroBERT-610m | 0.9996 | 1.001 | 8.107 | 0.1404 | 57.22 | 0.03247 | 0.9784 | 0.9672 | 0.0005923 |
| GENERator-EUK-3B | 0.9689 | 1.065 | 522.1 | 18.03 | 12.24 | 0.01713 | 0.6822 | 0.9517 | -0.02206 |
| GenerTeam/GENERator-v2-prokaryote-1.2b-base | 0.8469 | 1.394 | 6.22 | 1.931 | 3.222 | 0.01278 | 0.325 | 0.8341 | 0.08173 |
| GenerTeam/GENERator-v2-prokaryote-3b-base | 0.935 | 1.144 | 52.16 | 6.788 | 7.657 | 0.009023 | 0.588 | 0.926 | 0.01562 |
| GenomeOcean-4B | 0.8989 | 1.224 | 2.883 | 0.1088 | 19.5 | 0.02665 | 0.9084 | 0.8723 | -0.07464 |
| HuggingFaceTB/SmolLM2-1.7B | 0.9664 | 1.071 | 3923 | 253.9 | 15.54 | 0.01313 | 0.4169 | 0.9533 | 0.04432 |
| HuggingFaceTB/SmolLM2-135M | 0.9244 | 1.169 | 2003 | 160.1 | 12.48 | 0.03673 | 0.1477 | 0.8877 | 0.5788 |
| HuggingFaceTB/SmolLM2-360M | 0.9743 | 1.054 | 5692 | 252.4 | 22.38 | 0.01661 | 0.506 | 0.9576 | -0.01562 |
| Llama-7B | 0.9888 | 1.023 | 137.6 | 4.221 | 29.4 | 0.009568 | 0.522 | 0.9792 | 0.1678 |
| Mistral-7B | 0.9922 | 1.016 | 0.3883 | 0.0141 | 26.83 | 0.01416 | 0.8472 | 0.978 | 0.02628 |
| ModernBERT-base | 0.897 | 1.234 | 836 | 35.5 | 23.55 | 0.03319 | 0.6062 | 0.8638 | 0.1933 |
| MosaicBERT | 0.4766 | 4.116 | 20 | 3.961 | 5.049 | 0.03669 | 0.2766 | 0.4399 | 0.01145 |
| NTv3 | 0.3889 | 6.48 | 441.5 | 108.9 | 3.613 | 0.01788 | 0.01851 | 0.371 | 0.3694 |
| OLMo-7B-0724-hf | 0.9646 | 1.075 | 0.9111 | 0.02324 | 39.2 | 0.01831 | 0.4387 | 0.9463 | 0.3512 |
| Qwen/Qwen2.5-0.5B | 0.9982 | 1.004 | 18.73 | 0.5905 | 30.62 | 0.02966 | 0.7602 | 0.9685 | 0.138 |
| Qwen/Qwen2.5-1.5B | 0.9951 | 1.01 | 86.69 | 3.798 | 22.29 | 0.01815 | 0.9802 | 0.977 | 0.0007197 |
| Qwen/Qwen2.5-3B | 0.8174 | 1.439 | 37.91 | 4.589 | 8.318 | 0.02716 | 0.6792 | 0.7902 | 0.06043 |
| Qwen2.5-7B | 0.9529 | 1.1 | 29.69 | 2.629 | 9.283 | 0.009511 | 0.8041 | 0.9434 | 0.003815 |
| answerdotai/ModernBERT-large | 0.9701 | 1.062 | 708.1 | 24.39 | 26.58 | 0.4264 | 0.9402 | 0.5438 | -0.002569 |

`control_source` and `cross_check_batch` (original E11 12-panel vs round-2 batch-2 provenance flag) are retained in the companion CSV.
