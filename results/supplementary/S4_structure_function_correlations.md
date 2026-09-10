# S4 -- Structure-function correlations

Source: `audit/rederivations/structure_function_correlations.csv` (STORED, reformatted verbatim -- no values recomputed). 16 rows: 2 predictors (`q1`, `layer_relative_frobenius`) x 2 epsilons (0.5, 1.0) x 4 panels (all-22, text-decoders-only, text-encoders-only, genomic-only). The `underpowered_n_lt_8` flag is retained as-is (not dropped or renamed) -- panels with n<8 (text-encoders-only, n=6; genomic-only, n=6) are flagged `True`.

| Predictor | epsilon | Panel | n | rho | p | CI low | CI high | seed | LOO rho min | LOO rho max | Underpowered (n<8) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q1 | 0.5 | all-22 | 22 | 0.1146 | 0.6115 | -0.4095 | 0.6179 | 43 | 0.009091 | 0.2818 | False |
| layer_relative_frobenius | 0.5 | all-22 | 22 | 0.3484 | 0.1121 | -0.1388 | 0.7406 | 44 | 0.2844 | 0.5506 | False |
| q1 | 1.0 | all-22 | 22 | 0.07397 | 0.7436 | -0.3914 | 0.5126 | 43 | -0.06494 | 0.1896 | False |
| layer_relative_frobenius | 1.0 | all-22 | 22 | 0.441 | 0.03994 | 0.06217 | 0.7174 | 44 | 0.3649 | 0.5377 | False |
| q1 | 0.5 | text-decoders-only | 10 | -0.1636 | 0.6515 | -0.8077 | 0.775 | 43 | -0.3167 | 0.15 | False |
| layer_relative_frobenius | 0.5 | text-decoders-only | 10 | -0.06667 | 0.8548 | -0.64 | 0.6981 | 44 | -0.2333 | 0.2333 | False |
| q1 | 1.0 | text-decoders-only | 10 | -0.3333 | 0.3466 | -0.8875 | 0.5347 | 43 | -0.5167 | -0.08333 | False |
| layer_relative_frobenius | 1.0 | text-decoders-only | 10 | -0.04242 | 0.9074 | -0.6994 | 0.6842 | 44 | -0.2167 | 0.2333 | False |
| q1 | 0.5 | text-encoders-only | 6 | -0.4286 | 0.3965 | -1 | 0.7419 | 43 | -1 | 0 | True |
| layer_relative_frobenius | 0.5 | text-encoders-only | 6 | 0.02857 | 0.9572 | -1 | 1 | 44 | -0.2 | 0.8 | True |
| q1 | 1.0 | text-encoders-only | 6 | -0.3714 | 0.4685 | -1 | 0.8 | 43 | -0.7 | -0.2 | True |
| layer_relative_frobenius | 1.0 | text-encoders-only | 6 | 0.5429 | 0.2657 | -1 | 0.8 | 44 | 0.5 | 0.5 | True |
| q1 | 0.5 | genomic-only | 6 | 0.6 | 0.208 | -0.8 | 1 | 43 | 0.3 | 0.9 | True |
| layer_relative_frobenius | 0.5 | genomic-only | 6 | 0.6571 | 0.1562 | -0.3333 | 1 | 44 | 0.5 | 0.8 | True |
| q1 | 1.0 | genomic-only | 6 | 0.4857 | 0.3287 | -1 | 1 | 43 | 0.1 | 0.7 | True |
| layer_relative_frobenius | 1.0 | genomic-only | 6 | 0.4857 | 0.3287 | -0.6 | 1 | 44 | 0.3 | 0.6 | True |
