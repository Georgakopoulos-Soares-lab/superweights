# S8 — Two critical rows in SmolLM2-1.7B layer 7: ablations and geometry

**Sign convention:** the endpoint is a loss increase, so `interaction = effect_rel - sum_of_singles` is positive for super-additivity. This is the opposite sign to this project's accuracy-endpoint epistasis figures (e.g. DNABERT-2's -33.63 pp), which must be converted before comparison. All six ablation conditions were re-measured with an independent implementation that zeroes rows by explicit indexing and re-checks the intact loss after every restoration; every value reproduced exactly and the result was invariant to ablation order. Geometry is data-free. Sources: `results/analyses/within_layer_sweep/smollm2_second_row_epistasis.json`, `results/analyses/within_layer_sweep/smollm2_161_749_geometry.json`.

## A — Single and pairwise ablations

| condition | rows | activation_ratio | ratio_rank | nll | effect_rel | sum_of_singles | interaction | joint_over_sum | verdict |
|---|---|---|---|---|---|---|---|---|---|
| single r227 | 227 | 3181.7 | 1 | 20.88682623 | 6.988737 |  |  |  |  |
| single r161 | 161 | 63.54 | 4 | 10.12298247 | 2.871811 |  |  |  |  |
| single r749 | 749 | 358.56 | 2 | 2.67224864 | 0.022075 |  |  |  |  |
| joint r227+r161 | 227;161 |  |  | 25.27078974 | 8.665504 | 9.860549 | -1.195045 | 0.8788 | SUB-ADDITIVE (partly shared/overlapping damage) |
| joint r227+r749 | 227;749 |  |  | 22.82208231 | 7.728929 | 7.010812 | 0.718117 | 1.1024 | SUPER-ADDITIVE (synergy: joint damage exceeds the sum) |
| joint r161+r749 | 161;749 |  |  | 2.66870927 | 0.020721 | 2.893886 | -2.873165 | 0.0072 | MASKING (the joint ablation largely CANCELS the single-row damage) |

## B — Row geometry against a same-layer null

| quantity | rows | value | z_vs_reference | percentile_vs_reference |
|---|---|---|---|---|
| L2 norm of row 161 | 161 | 7.9069 |  |  |
| L2 norm of row 749 | 749 | 24.609 |  |  |
| L2 norm of row 227 | 227 | 26.8393 |  |  |
| cosine(r161, r749) | 161;749 | 0.3952 | 17.64 | 100.0 |
| cosine(r161, r227) | 161;227 | -0.0323 | -1.44 | 6.91 |
| cosine(r749, r227) | 749;227 | -0.398 | -17.76 | 0.0 |
| reference distribution (19900 pairs, 200 highest-norm rows of layer 7 down_proj) |  | mean -0.0001 | sd 0.0224 | min -0.3980 / max +0.1160 |
