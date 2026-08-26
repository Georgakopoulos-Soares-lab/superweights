# Full-Cohort Causal Census of High-Gain FFN Structures

## Goal

Run a systematic causal analysis on the **same 23-model cohort used in the expanded structural analysis (E11)**.

The purpose is to answer one central manuscript question:

> A recurrent, locally exceptional high-gain structural phenotype exists across diverse foundation models. Does that shared structural phenotype correspond to a shared causal organization, or do function and interaction structure vary substantially across models?

This experiment should remove any reasonable concern that the manuscript's causal examples were cherry-picked from the larger structural cohort.

This is intended to be the **final broad experiment for Part 2 of the paper**. Do not add exploratory side experiments or search for a prettier story after seeing the results.

---

# Fixed cohort

Use exactly these 23 models:

## Text decoders
1. Llama-7B
2. Mistral-7B
3. OLMo-7B-0724-hf
4. Phi-3-mini
5. Qwen2.5-7B
6. Qwen2.5-0.5B
7. Qwen2.5-1.5B
8. Qwen2.5-3B
9. SmolLM2-135M
10. SmolLM2-360M
11. SmolLM2-1.7B

## Text encoders
12. MosaicBERT
13. ModernBERT-base
14. ModernBERT-large
15. EuroBERT-210M
16. EuroBERT-610M
17. EuroBERT-2.1B

## Genomic decoders
18. GENERator-EUK-3B
19. GENERator-PROK-1.2B
20. GENERator-PROK-3B
21. GenomeOcean-4B

## Genomic encoders
22. DNABERT-2
23. NTv3

Do not silently remove a model. If one is technically impossible to evaluate, document exactly why and leave it explicitly marked as missing.

---

# General principles

Reuse the project's existing structural candidate coordinates, causal intervention code, evaluation utilities, and tomography code wherever possible.

Do **not** re-select candidates based on causal results.

The structural candidate set must be fixed from the existing structural analysis before any causal measurement is examined.

Reuse the same row-scaling intervention implementation already used in the manuscript rather than creating a new ablation mechanism.

Use two intervention strengths throughout:

- partial suppression: `epsilon = 0.5`
- full ablation: `epsilon = 1.0`

Use the project's existing convention for translating epsilon into row scaling.

For each candidate layer, also use **5 seeded ordinary same-layer control rows**, using the same fixed control-selection convention already used in E11 (`SeedSequence(42)`), excluding detected candidate rows.

All evaluation examples, masking patterns, prompts/windows, random seeds and controls must be frozen before causal measurements are inspected.

---

# Evaluation endpoints

Use each model's native language-model objective.

### Text decoders
Teacher-forced causal-LM NLL on one fixed text evaluation pool shared across all compatible text decoders.

### Text encoders
Masked-LM loss on one fixed text evaluation pool with fixed mask locations.

### Genomic decoders
Teacher-forced nucleotide causal-LM NLL on one fixed genomic evaluation pool shared across compatible genomic decoders.

### Genomic encoders
Masked-nucleotide LM loss on one fixed genomic evaluation pool with fixed mask locations.

Use identical underlying examples within each objective/domain group wherever technically possible.

The primary functional effect is:

`relative_loss_change = (perturbed_loss - baseline_loss) / baseline_loss`

Report raw loss as well, but do not compare raw loss magnitudes between architectures/objectives.

---

# Stage 0 — reproduction gate

Before running the full cohort, choose several models with existing locked causal results, ideally:

- one strong singleton decoder such as Llama or Mistral
- OLMo
- Phi-3
- one encoder tomography case such as ModernBERT or MosaicBERT
- DNABERT-2

Run the new common pipeline and confirm that it reproduces the direction and approximate magnitude of the existing causal measurements.

If there is a material discrepancy, STOP and diagnose it before running the cohort.

Do not average incompatible old and new measurements.

---

# Stage 1 — full-cohort singleton causal census

For every structurally detected candidate in every model:

1. measure baseline loss;
2. suppress the candidate at epsilon=0.5;
3. fully ablate it at epsilon=1.0;
4. measure the same interventions for the 5 seeded same-layer ordinary controls.

Save, at minimum:

- model
- domain
- architecture type
- parameter count
- candidate layer / row
- q1
- exact `||U_k||_F`
- layer-relative `||U_k||_F`
- candidate-vs-control q1 gap where available
- intervention strength
- baseline loss
- perturbed loss
- absolute delta loss
- relative delta loss
- control-row effects

Bootstrap evaluation examples/windows to obtain 95% CIs for causal effects.

Do not call a candidate "causally important" merely because it is structurally extreme.

---

# Stage 2 — test structure versus functional criticality

Across the full cohort, explicitly test whether structural prominence predicts causal importance.

Primary analyses:

1. `q1` versus full-ablation relative loss change.
2. layer-relative `||U_k||_F` versus full-ablation relative loss change.

Use Spearman correlation as the main descriptive association.

Because some models contain multiple candidates, do both:

- a one-primary-candidate-per-model analysis;
- an all-candidate secondary analysis with uncertainty clustered/resampled by model.

Do not treat candidate rows from the same model as independent models.

Also show the result graphically. The OLMo structural/causal dissociation should simply appear as one observation in the cohort rather than being specially privileged.

If there is little or no relationship, report that directly.

If there is a strong relationship, report that directly.

Do not force the expected "structure != function" result.

---

# Stage 3 — causal interaction census

For models with only one structurally detected candidate:

- measure its singleton causal effect;
- do **not** infer anything about interaction order;
- label interaction complexity as not identifiable from a one-component structural basis.

For models with >=2 candidate components, perform finite-intervention tomography using the same logic already established in the repo.

Use the existing observer ladder:

- additive/main-effect model
- calibrated/jointly fit additive model where already defined
- pairwise model

Reuse the existing F0/F1/F2/F3 definitions exactly if possible rather than inventing new versions.

The important comparison is:

> Does adding pairwise terms materially improve prediction on genuinely held-out intervention masks?

The train/calibration/held-out masks must be fixed before looking at responses.

Scale the number of training masks with the number of coefficients so the pairwise model is identifiable and comfortably overdetermined.

Let:

`P = K + K*(K-1)/2`

where K is the number of fixed candidate components.

Use at least enough training interventions to exceed P substantially, plus an independent held-out mask set. Follow existing DNABERT/Phi-3 conventions where possible.

Do not fit third-order terms as a post-hoc rescue.

If the second-order model fails badly at full ablation, report that the second-order observer is inadequate in that intervention regime.

---

# Stage 4 — cohort-level causal organization

Produce a single cohort table containing, for every model:

- model
- encoder/decoder
- text/genomic
- size
- number of structural candidates
- primary candidate q1
- candidate-control structural gap
- strongest singleton causal effect at epsilon=0.5
- strongest singleton causal effect at epsilon=1.0
- causal effect relative to same-layer controls
- whether interaction analysis was identifiable
- F2 held-out error
- F3 held-out error
- relative F2 -> F3 MAE improvement
- bootstrap CI for that improvement
- whether pair terms materially improved prediction
- whether the second-order observer was adequate at each intervention strength

Do **not** create categories just because the manuscript would benefit from neat categories.

If existing manuscript decision rules already define terms such as singleton dominance, pair-terms-required, etc., reuse the existing locked criteria exactly.

Otherwise report the measurements continuously rather than inventing thresholds after seeing the data.

---

# Main questions to answer

At the end, answer these explicitly:

### Q1
Are structurally detected high-gain rows generally more causally consequential than ordinary same-layer rows?

### Q2
Does structural concentration (`q1`) predict causal effect magnitude across models?

### Q3
Does layer-relative structural magnitude (`||U_k||_F`) predict causal effect magnitude?

### Q4
Among models where interaction structure is identifiable, how often is an additive observer adequate and how often do pair terms materially improve held-out prediction?

### Q5
Is there evidence for one universal causal organization across the cohort, or is causal realization genuinely heterogeneous?

### Q6
Do encoder/decoder architecture, text/genomic domain, or model scale obviously organize the causal results?

For Q6, keep this descriptive unless the sample supports a sensible statistical analysis. Do not manufacture a new architecture law.

---

# Figures

At minimum produce:

## Figure A — structure versus causal importance
Scatter of candidate structural measurement versus full-ablation relative loss effect.

Prefer two versions:
- q1 versus causal effect
- layer-relative `||U_k||_F` versus causal effect

Mark architecture/domain visually but do not overinterpret clustering.

## Figure B — full cohort causal census
One row per model showing candidate causal effect relative to same-layer controls.

## Figure C — interaction complexity
For every model where K>=2, show held-out F2 -> F3 improvement at epsilon=0.5 and 1.0 with bootstrap CI.

The figure should make missing/not-identifiable cases explicit rather than silently omitting them.

---

# Compute / orchestration

The node has 4 H100 GPUs available.

Parallelize up to 4 models at once when memory permits.

Do not put two large models on one GPU merely to increase throughput.

Checkpoint every completed model so a failed job does not require rerunning the cohort.

Before launching the full batch, estimate:
- expected GPU-hours;
- expected disk usage;
- which models require special loading code.

Then proceed unless there is a genuine blocker.

---

# Required outputs

Create a new results directory for this experiment containing:

- `causal_census.csv`
- `candidate_effects.csv`
- `control_effects.csv`
- `tomography_results.csv`
- `structure_function_correlations.json`
- `cohort_summary.json`
- publication-quality PDF figures
- raw per-example/per-window responses needed to reproduce every CI
- `FULL_COHORT_CAUSAL_SUMMARY.md`

The summary must contain:

1. exact methods;
2. technical failures/missing models;
3. full cohort table;
4. structure-function association results;
5. tomography results;
6. answers to Q1-Q6;
7. what the results support;
8. what they falsify;
9. what they leave unresolved.

---

# Scientific guardrails

This experiment is meant to settle the manuscript story, not create another cycle of rescue experiments.

Therefore:

- no candidate reselection after seeing causal results;
- no dropping inconvenient models;
- no higher-order interaction fitting merely because F3 fails;
- no new structural metric invented after looking at outcomes;
- no downstream-task fishing;
- no architecture subgroup created post hoc because it looks good;
- no manuscript rewrite until the entire cohort analysis is complete.

Unexpected results are results.

If most models are singleton-dominant, say so.

If interaction-dependent models are rare exceptions, say so.

If causal organization is broadly heterogeneous, quantify it.

If q1 unexpectedly predicts causal importance, report it rather than defending the previous thesis.

The final output should tell us what story the **full fixed cohort** supports.

Once this experiment and the separate final GENERator degradation-matching control are complete, the intended default is to freeze the experimental scope and write the manuscript around the resulting evidence.