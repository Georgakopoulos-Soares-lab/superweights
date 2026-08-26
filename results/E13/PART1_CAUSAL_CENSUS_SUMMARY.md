# Part 1 Causal Census — 22-Model Report

## Scope and completion

This report summarizes the completed Part 1 cohort of 22 models. Phi-3 is excluded by explicit scope clarification and is not treated as a missing model. All 22 included models have complete atomic raw responses: one frozen structural candidate, five seeded same-layer ordinary controls, two intervention strengths (epsilon 0.5 and 1.0), and paired per-unit evaluation data.

The resulting census contains 44 candidate effects and 220 control effects. Text decoders use WikiText-2 causal-LM NLL, text encoders fixed-mask WikiText-2 MLM loss, genomic decoders fixed hg38 teacher-forced NLL, and genomic encoders fixed-mask hg38 MLM loss. Confidence intervals use 5,000 paired bootstrap resamples.

## Main results

At epsilon 0.5, 18/22 candidates exceeded their same-layer median control; the median candidate-minus-control effect was +0.60% (95% model-bootstrap CI +0.23% to +3.06%).

At full ablation (epsilon 1.0), 20/22 candidates exceeded their same-layer median control; the median candidate-minus-control effect was +0.88% (95% model-bootstrap CI +0.61% to +111.78%).

Across the 22 frozen candidates, q1 versus full-ablation causal effect had Spearman rho=0.074 (95% bootstrap CI -0.391 to 0.513; p=0.744).

Across the 22 frozen candidates, layer-relative Frobenius magnitude versus full-ablation causal effect had Spearman rho=0.441 (95% bootstrap CI 0.062 to 0.717; p=0.0399).

## Full cohort

| Model | Type | Candidate effect eps=.5 | Candidate effect eps=1 | Candidate - median control eps=1 |
|---|---|---:|---:|---:|
| Llama-7B | text/decoder | +3.86% | +300.79% | +300.79% |
| Mistral-7B | text/decoder | +301.38% | +287.15% | +287.15% |
| OLMo-7B-0724-hf | text/decoder | +0.63% | +111.78% | +111.78% |
| Qwen2.5-7B | text/decoder | +0.60% | +0.74% | +0.74% |
| MosaicBERT | text/encoder | -0.17% | +0.37% | +0.38% |
| ModernBERT-base | text/encoder | +10.93% | +357.04% | +357.04% |
| NTv3 | genomic/encoder | -0.50% | -0.79% | -0.79% |
| DNABERT-2 | genomic/encoder | +0.31% | +0.83% | +0.83% |
| GENERator-EUK-3B | genomic/decoder | +4.29% | +37.10% | +37.10% |
| GenomeOcean-4B | genomic/decoder | +0.43% | +0.76% | +0.77% |
| Qwen/Qwen2.5-0.5B | text/decoder | +0.52% | +0.56% | +0.54% |
| Qwen/Qwen2.5-1.5B | text/decoder | +0.60% | +0.65% | +0.65% |
| Qwen/Qwen2.5-3B | text/decoder | +2.04% | +3.18% | +3.17% |
| HuggingFaceTB/SmolLM2-135M | text/decoder | +3.78% | +313.42% | +313.41% |
| HuggingFaceTB/SmolLM2-360M | text/decoder | +2.27% | +59.15% | +59.15% |
| HuggingFaceTB/SmolLM2-1.7B | text/decoder | +3.91% | +698.87% | +698.87% |
| GenerTeam/GENERator-v2-prokaryote-1.2b-base | genomic/decoder | +0.18% | +0.60% | +0.61% |
| GenerTeam/GENERator-v2-prokaryote-3b-base | genomic/decoder | +0.14% | +0.25% | +0.25% |
| EuroBERT/EuroBERT-210m | text/encoder | +0.10% | +0.16% | +0.16% |
| EuroBERT/EuroBERT-610m | text/encoder | -0.84% | +0.92% | +0.92% |
| EuroBERT/EuroBERT-2.1B | text/encoder | -0.00% | -0.01% | -0.01% |
| answerdotai/ModernBERT-large | text/encoder | +8.09% | +220.66% | +220.66% |

## Largest full-ablation candidate effects

1. HuggingFaceTB/SmolLM2-1.7B: +698.87% (candidate-minus-control +698.87%).
2. ModernBERT-base: +357.04% (candidate-minus-control +357.04%).
3. HuggingFaceTB/SmolLM2-135M: +313.42% (candidate-minus-control +313.41%).
4. Llama-7B: +300.79% (candidate-minus-control +300.79%).
5. Mistral-7B: +287.15% (candidate-minus-control +287.15%).

## Descriptive architecture/domain medians

- text/decoder: n=10, median full-ablation candidate effect +85.47%.
- text/encoder: n=6, median full-ablation candidate effect +0.65%.
- genomic/encoder: n=2, median full-ablation candidate effect +0.02%.
- genomic/decoder: n=4, median full-ablation candidate effect +0.68%.

## Interpretation

The Part 1 census supports the claim that frozen structurally prominent rows are usually more causally consequential than ordinary rows in the same layer: this holds for 18/22 models at partial suppression and 20/22 at full ablation, with positive cohort-median candidate-minus-control gaps at both strengths. The result is not universal: NTv3 and EuroBERT-2.1B do not show a positive full-ablation gap.

The data do not support q1 as a cross-model predictor of native-objective damage (rho=0.074 and a bootstrap interval spanning substantial negative and positive values). They do support a modest positive association for layer-relative Frobenius magnitude (rho=0.441; bootstrap CI 0.062 to 0.717), although this should be interpreted as one fixed-panel association rather than a universal law.

Causal magnitude is highly heterogeneous: the median full-ablation effect is +85.47% for text decoders, versus +0.65% for text encoders, +0.68% for genomic decoders, and +0.02% for genomic encoders. A few rows produce catastrophic loss increases while many produce sub-percent changes, so the evidence falsifies a single common effect-size regime across architectures. These native-objective measurements do not establish downstream-task mechanisms or interaction order.

## Artifacts

- `results/E13/part1_22_candidate_effects.csv`

- `results/E13/part1_22_control_effects.csv`

- `results/E13/part1_22_cohort_summary.csv`

- `results/E13/part1_22_structure_function_correlations.json`

- Raw paired responses: `results/E13/raw/`
