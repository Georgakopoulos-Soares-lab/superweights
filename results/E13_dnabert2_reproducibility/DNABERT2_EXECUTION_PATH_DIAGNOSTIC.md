# DNABERT-2 execution-path reproducibility diagnostic

Scope: `(layer 5, row 603)` activation discrepancy only. No causal intervention, tomography, candidate redesign, or threshold change was performed.

## A. Reproduction status

- **Historical endpoint: PASS.** Default-special-token ACTB preprocessing reproduces historical `out_max=944.5556030273438` exactly; `(5,603)` is activation rank 1, detector ratio 152.734548, ratio rank 8, and passes ratio 5. Retrospective frozen-threshold set size is K=89.
- **Current E13 endpoint: PASS.** No-special-token preprocessing reproduces the census activation and ratio exactly: `activation_max=0.573138356`, ratio 0.999393546, ratio rank 4628, activation rank 2749, and K=12.

Historical provenance caveat: the original tracked wrapper at commit `e790606` requested Hugging Face revision `main`; the discovery artifact did not store the resolved commit, package versions, or GPU. The later project pin `7bce263b...` was tested here. Exact reproduction of the stored activation on that pin is strong empirical checkpoint agreement, but does not retroactively create missing historical metadata.

## B. Root cause

**Demonstrated cause: tokenizer special-token handling.** The historical wrapper called the tokenizer with default `add_special_tokens=True`, producing 116 tokens: token ID 1, the exact 114-token ACTB interior used by E13, then token ID 2. E13 called the tokenizer with `add_special_tokens=False`, producing only the 114-token interior. The interior token IDs are exactly equal.

Adding the two historical boundary tokens switches `(5,603)` from activation 0.573138, ratio 0.999394, to activation 944.555603, ratio 152.734548. All other tested settings are held fixed.

Attention is **not the cause of the reproduced discrepancy**. Triton is unavailable in the diagnostic environment, so both `default` and explicitly `eager` requests execute the PyTorch fallback. Historical-special and explicit-eager-special cells are identical for all 9,216 rows; current-no-special and default-no-special cells are also identical for all rows. Because the historical stored activation reproduces exactly under eager attention, a historical flash kernel is not required to produce the spike. This study does not claim numerical equivalence between an unavailable flash kernel and eager attention.

A separate protocol mismatch was also confirmed: the historical discovery selected the global maximum activation iteratively; it did not originally define the full ratio≥5 set. K=89 above is therefore a retrospective application of the frozen E13 set rule to the historical activation path, not a claim about the original iterative basis size.

## C. Structural-metric stability

All four cells give exactly the same weight-derived values for `(5,603)`: q1=0.793311398288791, PR_spec=1.503287698117238, exact `||U_k||_F`=74.008557746091739, stable rank=1.260539054597029. This rules out a checkpoint/model-state inconsistency in the tested state.

## D. Candidate identity impact

With historical special tokens, `(5,603)` is the global activation candidate; the global ratio candidate is `(8,603)` at ratio 302.022. Without specials, the global activation candidate becomes `(1,603)`, the global ratio candidate becomes `(2,641)`, and `(5,603)` does not pass. The historical-coordinate comparison CSV shows that all ten established coordinates change substantially, not only the primary.

## E. Manuscript impact and protocol decision

Classification: **1. Benign execution-path dependence**, specifically input-boundary preprocessing dependence. Both endpoints reproduce exactly, the switching factor is isolated, and structural metrics are stable.

For claims tied to the established DNABERT-2 discovery coordinate, the canonical path should be the historical wrapper's default-special-token preprocessing because it exactly reproduces the frozen artifact. The current E13 no-special-token DNABERT-2 result is valid for the code that was run but is not selection-compatible with that historical candidate and should not be used to validate or replace it silently. Any recomputation of the DNABERT-2 census with historical preprocessing requires an explicit protocol amendment and a separate artifact; this diagnostic does not make that change.

Universal Part 2B tomography remains stopped.

## Artifacts

- `results/E13_dnabert2_reproducibility/execution_path_comparison.csv`
- `results/E13_dnabert2_reproducibility/historical_coordinate_impact.csv`
- `results/E13_dnabert2_reproducibility/diagnostic_provenance.json`
- `results/E13_dnabert2_reproducibility/{historical_default_specials,current_eager_no_specials,eager_specials,default_no_specials}.json`
- `results/E13_dnabert2_reproducibility/logs/*.log`
- `paper-salvage/experiments/E13_full_cohort_causal_census/run_dnabert2_repro_diagnostic.py`
- `paper-salvage/experiments/E13_full_cohort_causal_census/summarize_dnabert2_repro_diagnostic.py`
