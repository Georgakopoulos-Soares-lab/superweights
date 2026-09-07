# Part 2 functional-criticality evidence packet

Scope: evidence assembly for Results, Methods, and Figure 2; not polished manuscript prose. No new model response, causal intervention, or tomography was run.

## 1. Experiment and frozen cohort

The completed experiment contains 22 models, one frozen primary candidate per model, five independently scored same-layer controls, and ε∈{0.5,1.0}. Phi-3 is outside this 22-model analysis by explicit scope clarification; it is not missing. It is also the only manifest entry with a six-row pre-existing basis, whereas every included model has one primary row. Its prior tomography remains case-study material.

All 22 raw candidate coordinates match the corresponding primary coordinate in `candidate_manifest.json` exactly; all five raw control coordinates match the frozen control set exactly.

| Model | Checkpoint / requested revision | Resolved revision | Type | Candidate | Candidate provenance | Five controls |
|---|---|---|---|---:|---|---|
| Llama-7B | `huggyllama/llama-7b` @ `unpinned (E5/E6 original)` | `4782ad278652c7c71b72204d462d6d01eaaf7549` | text/decoder | L2/r3968 | `results/e7_legacy_reanalysis.json` | L2: 892;2034;2379;3729;3751 |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` @ `unpinned (E5/E6 original)` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | text/decoder | L1/r2070 | `results/e7_legacy_reanalysis.json` | L1: 190;305;746;1912;2687 |
| OLMo-7B-0724-hf | `allenai/OLMo-7B-0724-hf` @ `unpinned (E5/E6 original)` | `1ee306df318ee15bfe4a76ebd5c002b0105b1ab6` | text/decoder | L1/r269 | `results/e7_legacy_reanalysis.json` | L1: 292;437;2877;2908;3160 |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` @ `d149729398750b98c0af14eb82c78cfe92750796` | `d149729398750b98c0af14eb82c78cfe92750796` | text/decoder | L26/r458 | `results/e7_phase1_detection_qwen25.json` | L26: 62;1257;2926;3419;3553 |
| MosaicBERT | `mosaicml/mosaic-bert-base` @ `c89bbadc24278928f22bcdd7de6b61a5a2d08553` | `c89bbadc24278928f22bcdd7de6b61a5a2d08553` | text/encoder | L9/r287 | `results/e8_detection_mosaicbert.json` | L9: 56;173;193;517;545 |
| ModernBERT-base | `answerdotai/ModernBERT-base` @ `8949b909ec900327062f0ebf497f51aef5e6f0c8` | `8949b909ec900327062f0ebf497f51aef5e6f0c8` | text/encoder | L15/r251 | `results/e8_detection_modernbert.json` | L15: 23;325;421;487;687 |
| NTv3 | `InstaDeepAI/NTv3_650M_pre` @ `unpinned (E5/E6 original)` | `unpinned (E5/E6 original)` | genomic/encoder | L11/r1472 | `results/e7_legacy_reanalysis.json` | L11: 2;142;527;924;1152 |
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` @ `7bce263b15377fc15361f52cfab88f8b586abda0` | `7bce263b15377fc15361f52cfab88f8b586abda0` | genomic/encoder | L5/r603 | `results/e7_legacy_reanalysis.json` | L5: 72;216;513;542;757 |
| GENERator-EUK-3B | `GenerTeam/GENERator-v2-eukaryote-3b-base` @ `7dc01bccce5b65e15141170538afdc2ff09d8dde` | `7dc01bccce5b65e15141170538afdc2ff09d8dde` | genomic/decoder | L4/r2371 | `results/e7_legacy_reanalysis.json` | L4: 435;915;1058;1490;2073 |
| GenomeOcean-4B | `DOEJGI/GenomeOcean-4B` @ `2bed2fc3ed47c5f6955ba3e64563512c9b338dfb` | `2bed2fc3ed47c5f6955ba3e64563512c9b338dfb` | genomic/decoder | L1/r2604 | `results/e7_phase1_detection_genomeocean.json` | L1: 569;739;1012;1471;2581 |
| Qwen/Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` @ `060db6499f32faf8b98477b0a26969ef7d8b9987` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | text/decoder | L21/r62 | `results/E11/raw/qwen25-0.5b.json` | L21: 35;488;620;714;763 |
| Qwen/Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` @ `8faed761d45a263340a0528343f099c05c9a4323` | `8faed761d45a263340a0528343f099c05c9a4323` | text/decoder | L26/r408 | `results/E11/raw/qwen25-1.5b.json` | L26: 443;571;930;1118;1343 |
| Qwen/Qwen2.5-3B | `Qwen/Qwen2.5-3B` @ `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | text/decoder | L30/r318 | `results/E11/raw/qwen25-3b.json` | L30: 94;128;147;668;1146 |
| HuggingFaceTB/SmolLM2-135M | `HuggingFaceTB/SmolLM2-135M` @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | text/decoder | L11/r507 | `results/E11/raw/smollm2-135m.json` | L11: 294;320;402;441;448 |
| HuggingFaceTB/SmolLM2-360M | `HuggingFaceTB/SmolLM2-360M` @ `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | text/decoder | L3/r87 | `results/E11/raw/smollm2-360m.json` | L3: 238;258;338;577;934 |
| HuggingFaceTB/SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B` @ `effd688a12921b4cc83e3312b6feb579f70f9c71` | `effd688a12921b4cc83e3312b6feb579f70f9c71` | text/decoder | L7/r227 | `results/E11/raw/smollm2-1.7b.json` | L7: 317;582;1422;1547;2043 |
| GenerTeam/GENERator-v2-prokaryote-1.2b-base | `GenerTeam/GENERator-v2-prokaryote-1.2b-base` @ `8b2f768b0d293953518ff91d34600f9322ef1f94` | `8b2f768b0d293953518ff91d34600f9322ef1f94` | genomic/decoder | L4/r798 | `results/E11/raw/generator-prok-1.2b.json` | L4: 411;1094;1207;1619;1799 |
| GenerTeam/GENERator-v2-prokaryote-3b-base | `GenerTeam/GENERator-v2-prokaryote-3b-base` @ `b18ac86df77359d894d7bc050cea78e2d0713021` | `b18ac86df77359d894d7bc050cea78e2d0713021` | genomic/decoder | L8/r260 | `results/E11/raw/generator-prok-3b.json` | L8: 960;1268;1327;1440;1680 |
| EuroBERT/EuroBERT-210m | `EuroBERT/EuroBERT-210m` @ `39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e` | `39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e` | text/encoder | L3/r300 | `results/E11/raw/eurobert-210m.json` | L3: 97;316;361;486;620 |
| EuroBERT/EuroBERT-610m | `EuroBERT/EuroBERT-610m` @ `d9af784ed20db6c2096e335ec6a67dd4a219924c` | `d9af784ed20db6c2096e335ec6a67dd4a219924c` | text/encoder | L13/r762 | `results/E11/raw/eurobert-610m.json` | L13: 47;405;830;979;1088 |
| EuroBERT/EuroBERT-2.1B | `EuroBERT/EuroBERT-2.1B` @ `81245a4d71f43452badf5e04458e4ddb831ff109` | `81245a4d71f43452badf5e04458e4ddb831ff109` | text/encoder | L14/r2198 | `results/E11/raw/eurobert-2.1b.json` | L14: 560;1905;1946;2212;2286 |
| answerdotai/ModernBERT-large | `answerdotai/ModernBERT-large` @ `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | text/encoder | L19/r379 | `results/E11/raw/modernbert-large.json` | L19: 144;154;374;841;940 |

Control selection: `SeedSequence(42).spawn(23)[panel_index]`; sample five distinct rows without replacement from `0..d_model-1`, excluding every candidate in that model/layer, then sort. One deterministic stream per model; Phi-3 consumes its stream across layers 2 then 4. Source: `build_candidate_manifest.py` and `candidate_manifest.json`.

### DNABERT-2 selection versus evaluation provenance

The canonical discovery path for L5/r603 used the 504-bp ACTB probe with tokenizer default special tokens and exactly reproduces historical `out_max=944.5556030273438` (ratio 152.734548, activation rank 1). The later no-special-token structural census is not used to invalidate it. The Part 2 causal endpoint is separate: 256 seed-42 hg38 windows, fixed masked-token batches. Candidate-selection ACTB preprocessing and causal-evaluation hg38 preprocessing are not the same assay.

## 2. Exact intervention and endpoints

For one candidate or one control at a time, clone down-projection row `w`, set `w_ε=(1-ε)w`, evaluate, and restore the cloned row in a context-manager/finally path. Thus ε=0.5 means 50% row scaling and ε=1 means a zero row. Models are in eval mode and loaded/evaluated in float32; no autocast is declared. Candidate and controls are never masked jointly.

| Group | Evaluation units and preprocessing | Endpoint | Exact implemented batch/unit structure |
|---|---|---|---|
| Text decoders (n=10) | WikiText-2-raw-v1 test; nonempty lines shuffled with `random.Random(42)`, concatenated and each model tokenized into 100 exactly-512-token windows. | Teacher-forced shifted-label mean token NLL: logits `[:-1]`, labels `[1:]`, summed NLL divided by predicted-token count. | Batch 4 for loaded models >3B parameters (25 bootstrap units); otherwise batch 8 (13 units, final partial batch). |
| Text encoders (n=6) | Same WikiText construction, 256 × 512-token windows; batch 16. One fixed mask realization, seed 42, probability .15; CLS/SEP/PAD excluded; selected tokens replaced by MASK and all other labels set to -100. | Summed cross-entropy over masked tokens divided by masked-token count. | 16 fixed MLM batches; identical inputs/masks reused for baseline and every condition. MosaicBERT uses `bert-base-uncased` tokenizer; others use checkpoint tokenizers. |
| Genomic decoders (n=4) | hg38 `random_262kb.bed`; seed-42 disjoint partition; 100 damage windows of 512 bp, <1% N. GENERator trims the left `len%6` bases, prepends BOS, then tokenizes with specials disabled. GenomeOcean does no 6-bp trim or forced BOS and tokenizes with specials disabled. | Teacher-forced shifted causal-LM mean token NLL (`labels=input_ids`; internal label shift), weighted over tokens. | 100 individual-window bootstrap units. The separately constructed 96-window prompt pool is not used in this endpoint. |
| Genomic encoders (n=2) | hg38 FASTA + `random_262kb.bed`; regions shuffled and starts sampled with `random.Random(42)`; 256 windows of 600 bp, <1% N; tokenizer padding/truncation to 256 tokens; batch 16. Fixed .15 masks with seed 42; special/PAD excluded. | Masked-nucleotide summed loss divided by masked-token count. | 16 fixed MLM batches. DNABERT-2 uses the pinned pretrained MLM/eager compatibility loader; NTv3 uses its pinned remote code revision. |

Raw `endpoint` metadata and actual unit counts are preserved model-by-model in `part2_22_model_results.csv` and `results/E13/raw/*.json`.

Checkpoint provenance note: NTv3's weight revision remains recorded as unpinned (`E5/E6 original`); only its required remote-code loader is pinned to commit `0ecff3637f0d3ba5b686d1095083218157c2ca34` (matches `NTV3_CODE_REVISION` in `genomic_encoder_lib.py`). That is a provenance limitation, not an exact weight revision. For models requested without a revision, the resolved Hub commit recorded by the completed run is reported in the table above.

## 3. Effect definitions

For evaluation unit j, raw records store `(S_j,N_j)`: summed loss and contributing-token count. `L0=ΣS0j/ΣN0j`; `Lε=ΣSεj/ΣNεj`; absolute change `ΔL=Lε-L0`; signed relative change `R=(Lε-L0)/L0`; percent change is `100R`. Candidate effect is R for the frozen candidate. Each control effect is its independently measured R. Same-layer median control is the median of five signed control R values. Candidate-minus-control is `G=Rcandidate-median(Rcontrol,1..5)`.

The 18/22 and 20/22 counts use `G>0`. Cohort +0.60%/+0.88% summaries are medians of G across 22 models. The q1 correlation and Figure 2C y-axis use signed full-ablation candidate R—not G and not an absolute value. Figure 2A/B show signed candidate and individual-control R.

## 4. Recomputed cohort results

### ε=0.5

- Candidate > median control: **18/22**.
- Median candidate effect: **+0.60%**.
- Median median-control effect: **+0.00%**.
- Median candidate-minus-control: **+0.60%**.
- 95% model-bootstrap percentile CI for median gap: **+0.23% to +3.06%** (5,000 draws; seed 47).

Sorted candidate effects (weakest to strongest):

| Model | Signed effect |
|---|---:|
| EuroBERT/EuroBERT-610m | -0.84% |
| NTv3 | -0.50% |
| MosaicBERT | -0.17% |
| EuroBERT/EuroBERT-2.1B | -0.00% |
| EuroBERT/EuroBERT-210m | +0.10% |
| GenerTeam/GENERator-v2-prokaryote-3b-base | +0.14% |
| GenerTeam/GENERator-v2-prokaryote-1.2b-base | +0.18% |
| DNABERT-2 | +0.31% |
| GenomeOcean-4B | +0.43% |
| Qwen/Qwen2.5-0.5B | +0.52% |
| Qwen/Qwen2.5-1.5B | +0.60% |
| Qwen2.5-7B | +0.60% |
| OLMo-7B-0724-hf | +0.63% |
| Qwen/Qwen2.5-3B | +2.04% |
| HuggingFaceTB/SmolLM2-360M | +2.27% |
| HuggingFaceTB/SmolLM2-135M | +3.78% |
| Llama-7B | +3.86% |
| HuggingFaceTB/SmolLM2-1.7B | +3.91% |
| GENERator-EUK-3B | +4.29% |
| answerdotai/ModernBERT-large | +8.09% |
| ModernBERT-base | +10.93% |
| Mistral-7B | +301.38% |

### ε=1.0

- Candidate > median control: **20/22**.
- Median candidate effect: **+0.88%**.
- Median median-control effect: **+0.00%**.
- Median candidate-minus-control: **+0.88%**.
- 95% model-bootstrap percentile CI for median gap: **+0.61% to +111.78%** (5,000 draws; seed 52).

Sorted candidate effects (weakest to strongest):

| Model | Signed effect |
|---|---:|
| NTv3 | -0.79% |
| EuroBERT/EuroBERT-2.1B | -0.01% |
| EuroBERT/EuroBERT-210m | +0.16% |
| GenerTeam/GENERator-v2-prokaryote-3b-base | +0.25% |
| MosaicBERT | +0.37% |
| Qwen/Qwen2.5-0.5B | +0.56% |
| GenerTeam/GENERator-v2-prokaryote-1.2b-base | +0.60% |
| Qwen/Qwen2.5-1.5B | +0.65% |
| Qwen2.5-7B | +0.74% |
| GenomeOcean-4B | +0.76% |
| DNABERT-2 | +0.83% |
| EuroBERT/EuroBERT-610m | +0.92% |
| Qwen/Qwen2.5-3B | +3.18% |
| GENERator-EUK-3B | +37.10% |
| HuggingFaceTB/SmolLM2-360M | +59.15% |
| OLMo-7B-0724-hf | +111.78% |
| answerdotai/ModernBERT-large | +220.66% |
| Mistral-7B | +287.15% |
| Llama-7B | +300.79% |
| HuggingFaceTB/SmolLM2-135M | +313.42% |
| ModernBERT-base | +357.04% |
| HuggingFaceTB/SmolLM2-1.7B | +698.87% |

## 5. Structure-function associations

q1 versus signed full-ablation candidate R: Spearman ρ=0.073969509, asymptotic two-sided p=0.743558799, 5,000-model-bootstrap percentile CI [-0.391371102,0.512621074], seed 43, n=22.

Layer-relative Frobenius magnitude: ρ=0.440993789, p=0.039939999, CI [0.062174586,0.717382618], seed 44, n=22.

Repository search found no completed leave-one-model-out, leave-one-family-out, within-group, covariate-adjusted, or influence diagnostic for the Frobenius association. Therefore it is not promoted to Figure 2; it remains a fixed-panel secondary association requiring robustness work before headline use.

## 6. Exact uncertainty procedures

Per-condition CI: 5,000 percentile draws. A single resampled index vector selects paired baseline and perturbed units jointly, then each loss is recomputed as total summed loss / total contributing tokens and converted to R. Units are actual stored batches for text decoders and both encoder groups, and individual windows for genomic decoders. Condition RNG streams come sequentially from `SeedSequence(42).spawn(1000)` in frozen panel/condition order.

Cohort gap CI: resample the 22 model-level G values with replacement, take the median per draw; 5,000 percentile draws, seeds 47 (.5) and 52 (1.0). Correlation: SciPy `spearmanr` on 22 model rows; default asymptotic two-sided p. CI resamples 22 model indices with replacement, recomputes rho, drops only nonfinite bootstrap replicates, and takes percentiles; seeds 43 (q1) and 44 (Frobenius). Negative and zero effects are retained unchanged.

## 7. Descriptive subgroup medians

- text/decoder: n=10, median signed full-ablation candidate effect **+85.47%**.
- text/encoder: n=6, median signed full-ablation candidate effect **+0.65%**.
- genomic/encoder: n=2, median signed full-ablation candidate effect **+0.02%**.
- genomic/decoder: n=4, median signed full-ablation candidate effect **+0.68%**.

These are descriptive only. Objectives, architectures, domains, tokenizers, and panel composition are confounded; no architecture/domain determination claim is licensed.

## 8. Figure 2 deliverables

Panels A/B use a signed symmetric-log axis (`linthresh=0.1 percentage points`), retaining negative, sub-percent, and catastrophic effects. Diamonds are candidates; gray points are all five controls; short bars are within-model control medians. Models share one order based on full-ablation candidate effect. Panel C uses q1 and the exact signed full-ablation candidate R used in the correlation. No Frobenius panel is included because robustness analyses do not exist.

- `results/E13/figure2_candidate_control_data.csv`
- `results/E13/figure2_structure_function_data.csv`
- `paper-salvage/figures/main/fig2_part2_functional_criticality.png`
- `paper-salvage/figures/main/fig2_part2_functional_criticality.pdf`

## Claims directly supported by the completed experiment

- Frozen candidates exceed same-layer median controls in 18/22 models at ε=.5 and 20/22 at ε=1.
- The cohort-median signed candidate-minus-control gap is positive at both strengths with model-bootstrap CIs above zero.
- Full-ablation candidate effects span negative/sub-percent values through approximately +699%.
- q1 does not predict full-ablation causal magnitude in this 22-model panel; its CI spans substantial negative and positive correlations.
- The measured candidate effect is heterogeneous across models and native objectives.

## Claims not supported / caveats

- No universal effect: NTv3 and EuroBERT-2.1B have nonpositive full-ablation candidate-minus-control gaps.
- No claim that architecture or domain causes the descriptive subgroup differences.
- Native-objective relative losses are useful within each model but objectives/tokenizers differ across groups.
- No downstream-task mechanism, interaction order, or cohort-wide tomography conclusion follows from singleton effects.
- The Frobenius association lacks existing influence/covariate/leave-group robustness analyses and is not a headline result.
- DNABERT-2 discovery is input-boundary dependent; its ACTB discovery preprocessing must not be conflated with its hg38 causal endpoint.

## Remaining manuscript issues

- Decide how briefly to disclose that the original DNABERT-2 wrapper recorded revision `main`; the pinned state reproduces its stored activation exactly, but the original resolved commit/environment were not recorded.
- Replace the old five-decoder/tomography Figure 2 references and captions with this 22-model singleton census; retain tomography only as mechanistic case-study material.
- Keep the Frobenius result secondary unless a separately authorized robustness analysis is completed.
- Universal Part 2B tomography remains stopped.

## Primary provenance paths

- `results/E13/candidate_manifest.json`
- `results/E11/scale_ladder_backfilled.csv`
- `results/E13/raw/*.json`
- `results/E13/part1_22_candidate_effects.csv`
- `results/E13/part1_22_control_effects.csv`
- `results/E13/part1_22_cohort_summary.csv`
- `results/E13/part1_22_structure_function_correlations.json`
- `paper-salvage/experiments/E13_full_cohort_causal_census/run_singleton_census.py`
- `paper-salvage/experiments/E13_full_cohort_causal_census/write_part1_22_report.py`
- `paper-salvage/experiments/E10_nlp_architecture_causal/e10_lib.py`
- `paper-salvage/experiments/E9_mechanistic_tomography/tomography_lib.py`
- `paper-salvage/experiments/E12_generator_degradation_control/e12_lib.py`
- `results/E13_dnabert2_reproducibility/DNABERT2_EXECUTION_PATH_DIAGNOSTIC.md`
