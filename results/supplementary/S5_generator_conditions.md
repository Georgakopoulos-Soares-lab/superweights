# S5 -- GENERator conditions (alpha grid and c grid)

106 rows: 78 alpha-grid rows (from `audit/generator_alpha_sweeps.csv`, STORED) + 28 c-grid rows (from `audit/rederivations/generator_random_direction_full.csv`, STORED). The two perturbation types are kept distinguished by the `perturbation_type` column (`alpha_sweep` vs `c_grid`) and `condition_class` (`row2371_own_alpha_sweep`, `random_direction_own_alpha_sweep` / `random_direction_c_grid`, `inert_control_row`) rather than being conflated into one x-axis, per task instructions -- `x_value_type` also names which of alpha or c the `x_value` column is.

**Alpha-grid condition labels found:** control_row_102, control_row_456, control_row_1126, control_row_2621, control_row_3039 (5 inert control rows, 11 alphas each), random_direction (14 alphas), row2371 (9 alphas, row2371's own ablation-strength sweep). Total 78 rows -- matches the reconnaissance note ("row2371, row1522" was speculative; the actual second candidate-style series in this file is `random_direction`, not a `row1522` label -- no `row1522` condition_label exists in `audit/generator_alpha_sweeps.csv`).

**c-grid series found:** random_direction (14 c values), row2371_own_alpha_sweep (9 values), and 5 control_row_* series (1 "matched" point each). Total 28 rows.

**Fraction of ablation damage:** computed as `(nll - BASELINE_NLL) / (TARGET_NLL - BASELINE_NLL)` with `BASELINE_NLL = 6.38538052380085` and `TARGET_NLL = 8.754319605827332`, RECOMPUTED here using the exact constants from `audit/rederivations/scripts/section1_final_plots.py` (lines 22-23). **Verified, not just reused:** `row2371`/`row2371_own_alpha_sweep` at alpha/c=1.0 reproduces BASELINE_NLL exactly in both source files, and `random_direction` at alpha/c=0.0 reproduces TARGET_NLL exactly in both source files -- the two source files agree with each other and with the script's constants, so no disagreement to flag.

**Missing GC cells (NOT FOUND, not blank):** GC (composition) was measured only at a sparse subset of alpha/c grid points -- 86/106 rows have `gc_fraction = NOT FOUND` because the source file records `n_prompts=0` there (GC simply was not evaluated at that grid point; all 78 alpha-grid rows lack GC entirely -- `audit/generator_alpha_sweeps.csv` never populates gc_fraction for any row). Of the 28 c-grid rows, 20 have a GC value; the other 8 (all in the `random_direction` c-grid series, at c in {0.025, 0.05, 0.1, 0.25, 0.75, 1.5, 2.0, 5.0}) are `NOT FOUND` for the same n_prompts=0 reason. Additionally, 14/20 GC-bearing rows (the `row2371_own_alpha_sweep` and `control_row_*` series) have a `gc_fraction` mean but `NOT FOUND` for `gc_ci_low`/`gc_ci_high` -- the source file records no bootstrap CI for those single-condition series, only for the `random_direction` series' 6 populated points.

**Reference points named in the manuscript text:** intact baseline GC = 0.4204 (`row2371_own_alpha_sweep` alpha=1.0, gc=0.4204011140046296, = BASELINE_NLL) and full-ablation point GC = 0.3065, NLL = 8.754 (`random_direction`/`row2371_own_alpha_sweep` alpha/c=0.0, gc=0.3064959490740741, nll=8.754319605827332 = TARGET_NLL) -- both reproduced exactly in this table (rows below).

| Perturbation | Condition | Class | x-type | x-value | NLL | GC | GC CI low | GC CI high | n_prompts | Frac. ablation damage |
|---|---|---|---|---|---|---|---|---|---|---|
| alpha_sweep | control_row_102 | inert_control_row | alpha | 0 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 6.065e-06 |
| alpha_sweep | control_row_102 | inert_control_row | alpha | 0.1 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 5.284e-06 |
| alpha_sweep | control_row_102 | inert_control_row | alpha | 0.25 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 4.001e-06 |
| alpha_sweep | control_row_102 | inert_control_row | alpha | 0.5 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 2.344e-06 |
| alpha_sweep | control_row_102 | inert_control_row | alpha | 0.75 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 8.482e-07 |
| alpha_sweep | control_row_102 | inert_control_row | alpha | 1 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0 |
| alpha_sweep | control_row_102 | inert_control_row | alpha | 1.5 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 7.754e-06 |
| alpha_sweep | control_row_102 | inert_control_row | alpha | 2 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 4.756e-05 |
| alpha_sweep | control_row_102 | inert_control_row | alpha | 3 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 5.899e-05 |
| alpha_sweep | control_row_102 | inert_control_row | alpha | 5 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 8.951e-05 |
| alpha_sweep | control_row_102 | inert_control_row | alpha | 8 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 9.785e-05 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 0 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -6.273e-05 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 0.1 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -5.817e-05 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 0.25 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -5.083e-05 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 0.5 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -3.666e-05 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 0.75 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -1.914e-05 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 1 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 1.5 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 8.779e-05 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 2 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.0001289 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 3 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.0002152 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 5 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.0003807 |
| alpha_sweep | control_row_1126 | inert_control_row | alpha | 8 | 6.387 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.0006269 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 0 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.0001264 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 0.1 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.0001077 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 0.25 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 8.038e-05 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 0.5 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 4.466e-05 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 0.75 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 2.076e-05 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 1 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 1.5 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -4.404e-05 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 2 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -9.194e-05 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 3 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -0.0001854 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 5 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -0.0003236 |
| alpha_sweep | control_row_2621 | inert_control_row | alpha | 8 | 6.384 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -0.00053 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 0 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -2.494e-05 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 0.1 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -2.301e-05 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 0.25 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -1.938e-05 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 0.5 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -1.329e-05 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 0.75 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -6.682e-06 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 1 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 1.5 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 1.442e-05 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 2 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 2.831e-05 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 3 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 8.227e-05 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 5 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.000215 |
| alpha_sweep | control_row_3039 | inert_control_row | alpha | 8 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.0004317 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 0 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -1.577e-05 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 0.1 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -1.53e-05 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 0.25 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -1.333e-05 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 0.5 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -9.569e-06 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 0.75 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | -4.724e-06 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 1 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 1.5 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 1.051e-05 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 2 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 2.275e-05 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 3 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 5.021e-05 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 5 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.0001116 |
| alpha_sweep | control_row_456 | inert_control_row | alpha | 8 | 6.386 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.0002431 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 0 | 8.754 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 1 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 0.0125 | 8.754 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9997 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 0.025 | 8.753 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9994 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 0.05 | 8.752 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9988 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 0.1 | 8.748 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9975 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 0.25 | 8.743 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9952 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 0.5 | 8.733 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9911 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 0.75 | 8.723 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9866 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 1 | 8.712 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.982 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 1.5 | 8.685 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9706 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 2 | 8.663 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9616 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 3 | 8.608 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9383 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 5 | 8.495 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.8907 |
| alpha_sweep | random_direction | random_direction_own_alpha_sweep | alpha | 8 | 8.375 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.8399 |
| alpha_sweep | row2371 | row2371_own_alpha_sweep | alpha | 0 | 8.754 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 1 |
| alpha_sweep | row2371 | row2371_own_alpha_sweep | alpha | 0.25 | 7.791 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.5933 |
| alpha_sweep | row2371 | row2371_own_alpha_sweep | alpha | 0.5 | 6.659 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.1156 |
| alpha_sweep | row2371 | row2371_own_alpha_sweep | alpha | 0.75 | 6.413 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.01162 |
| alpha_sweep | row2371 | row2371_own_alpha_sweep | alpha | 1 | 6.385 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0 |
| alpha_sweep | row2371 | row2371_own_alpha_sweep | alpha | 1.5 | 6.491 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.04461 |
| alpha_sweep | row2371 | row2371_own_alpha_sweep | alpha | 2 | 6.638 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.1065 |
| alpha_sweep | row2371 | row2371_own_alpha_sweep | alpha | 3 | 6.982 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.2517 |
| alpha_sweep | row2371 | row2371_own_alpha_sweep | alpha | 5 | 7.138 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.3176 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 0 | 8.754 | 0.3065 | 0.301 | 0.3121 | 288 | 1 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 0.0125 | 8.754 | 0.3068 | 0.3013 | 0.3127 | 288 | 0.9997 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 0.025 | 8.753 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9994 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 0.05 | 8.752 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9988 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 0.1 | 8.748 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9975 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 0.25 | 8.743 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9952 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 0.5 | 8.733 | 0.3071 | 0.3014 | 0.3126 | 288 | 0.9911 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 0.75 | 8.723 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9866 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 1 | 8.712 | 0.3078 | 0.3018 | 0.3138 | 288 | 0.982 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 1.5 | 8.685 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9706 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 2 | 8.663 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.9616 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 3 | 8.608 | 0.313 | 0.3067 | 0.3198 | 288 | 0.9383 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 5 | 8.495 | NOT FOUND* | NOT FOUND* | NOT FOUND* | 0 | 0.8907 |
| c_grid | random_direction | random_direction_c_grid | c_or_alpha | 8 | 8.375 | 0.3402 | 0.3317 | 0.3491 | 288 | 0.8399 |
| c_grid | row2371_own_alpha_sweep | row2371_own_alpha_sweep | c_or_alpha | 0 | 8.754 | 0.3065 | NOT FOUND* | NOT FOUND* | 288 | 1 |
| c_grid | row2371_own_alpha_sweep | row2371_own_alpha_sweep | c_or_alpha | 0.25 | 7.791 | 0.3564 | NOT FOUND* | NOT FOUND* | 288 | 0.5933 |
| c_grid | row2371_own_alpha_sweep | row2371_own_alpha_sweep | c_or_alpha | 0.5 | 6.659 | 0.3982 | NOT FOUND* | NOT FOUND* | 288 | 0.1156 |
| c_grid | row2371_own_alpha_sweep | row2371_own_alpha_sweep | c_or_alpha | 0.75 | 6.413 | 0.4178 | NOT FOUND* | NOT FOUND* | 288 | 0.01162 |
| c_grid | row2371_own_alpha_sweep | row2371_own_alpha_sweep | c_or_alpha | 1 | 6.385 | 0.4204 | NOT FOUND* | NOT FOUND* | 288 | 0 |
| c_grid | row2371_own_alpha_sweep | row2371_own_alpha_sweep | c_or_alpha | 1.5 | 6.491 | 0.4252 | NOT FOUND* | NOT FOUND* | 288 | 0.04461 |
| c_grid | row2371_own_alpha_sweep | row2371_own_alpha_sweep | c_or_alpha | 2 | 6.638 | 0.412 | NOT FOUND* | NOT FOUND* | 288 | 0.1065 |
| c_grid | row2371_own_alpha_sweep | row2371_own_alpha_sweep | c_or_alpha | 3 | 6.982 | 0.3649 | NOT FOUND* | NOT FOUND* | 288 | 0.2517 |
| c_grid | row2371_own_alpha_sweep | row2371_own_alpha_sweep | c_or_alpha | 5 | 7.138 | 0.3764 | NOT FOUND* | NOT FOUND* | 288 | 0.3176 |
| c_grid | control_row_2621 | inert_control_row | c_or_alpha | matched | 6.386 | 0.4202 | NOT FOUND* | NOT FOUND* | 288 | 0.0001264 |
| c_grid | control_row_456 | inert_control_row | c_or_alpha | matched | 6.386 | 0.4212 | NOT FOUND* | NOT FOUND* | 288 | 0.0002431 |
| c_grid | control_row_102 | inert_control_row | c_or_alpha | matched | 6.386 | 0.4208 | NOT FOUND* | NOT FOUND* | 288 | 9.785e-05 |
| c_grid | control_row_3039 | inert_control_row | c_or_alpha | matched | 6.386 | 0.4202 | NOT FOUND* | NOT FOUND* | 288 | 0.0004317 |
| c_grid | control_row_1126 | inert_control_row | c_or_alpha | matched | 6.387 | 0.4198 | NOT FOUND* | NOT FOUND* | 288 | 0.0006269 |

`*` = `NOT FOUND` (see the two paragraphs above for the reason in each case; the full reason string is preserved verbatim in the companion CSV's `gc_fraction`/`gc_ci_low`/`gc_ci_high` cells, not truncated there).
