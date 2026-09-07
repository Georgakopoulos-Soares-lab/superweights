# Part 2B — Full-Cohort Causal Interaction Census

## Goal

Part 2A is complete.

Across the fixed 22-model cohort, we have already measured the causal effect of one frozen primary high-gain candidate per model versus five same-layer controls.

Part 2B asks a different question:

> When a model contains several structurally prominent high-gain components, how do those components work together causally?

We want to determine whether their combined effects are:

- approximately additive;
- dependent on pairwise interactions;
- or not adequately described by a second-order model.

The eventual manuscript question is:

> Does the recurring structural phenotype correspond to a common causal organization, or do different models realize it differently?

---

# IMPORTANT: do not run tomography yet

The current E13 `candidate_manifest.json` contains only **one primary candidate per model**.

That basis was appropriate for Part 2A, but it makes interaction analysis mathematically impossible.

Do NOT conclude that the other models have no interactions.

Do NOT arbitrarily add rows.

Do NOT use causal results to choose additional components.

Before any new causal measurements, perform the basis audit below.

---

# Stage 0 — recover and audit previous multi-row bases

Recover exactly how the existing multi-component bases were constructed for:

- DNABERT-2
- MosaicBERT
- ModernBERT
- Phi-3

For each, report:

- exact row coordinates;
- K;
- structural quantities used to select them;
- whether selection used activation magnitude, exact `||U_k||_F`, q1, layer-relative magnitude, prior super-weight coordinates, or another criterion;
- whether any causal result influenced selection;
- source artifact / code / commit.

Also compare these rules with the E13 one-primary-candidate selection rule.

The purpose is to understand whether there is already a principled structural definition that can be generalized.

Phi-3 is **not part of the 22-model cohort**. Use it only as methodological provenance for basis construction and previous tomography, not as a cohort observation.

---

# Stage 1 — propose one universal causal-blind basis rule

Using only information available **before causal intervention**, propose a single basis-selection rule that can be applied uniformly to all 22 models.

The rule must:

1. use only structural/activation information;
2. never use causal loss effects;
3. never use downstream results;
4. be deterministic;
5. apply equally to encoders and decoders;
6. avoid manually selecting different K values because particular models look interesting;
7. produce a scientifically meaningful set of high-gain components rather than padding with ordinary rows merely to enable tomography.

Possible ingredients may include existing detector scores, exact operator magnitude, layer-relative magnitude, or another already-established project quantity.

Do NOT invent a new structural score simply because it produces convenient bases.

### Critical requirement

Before running tomography, write:

`PART2B_BASIS_RULE_PROPOSAL.md`

containing:

- the proposed rule;
- why it is scientifically justified;
- how it relates to previous DNABERT/MosaicBERT/ModernBERT/Phi-3 bases;
- K produced for every one of the 22 models;
- exact coordinates selected;
- any models for which the rule produces K=1;
- any limitations.

Then freeze the basis.

### Stop condition

If there is **no defensible universal causal-blind rule**, STOP.

Do not force a full-cohort interaction experiment.

Instead explain why the existing data support mechanistic case studies but not a cohort-wide interaction census.

That is preferable to manufacturing a basis.

---

# Stage 2 — freeze the 22-model basis

If the rule is defensible, generate:

`part2b_candidate_basis.csv`

with:

- model
- domain
- encoder/decoder
- size
- K
- layer
- row
- structural detector score
- q1
- exact `||U_k||_F`
- layer-relative `||U_k||_F`
- basis-selection reason
- provenance

Freeze this before inspecting any new combination-intervention responses.

Models with K=1 remain in the cohort but have:

`interaction_identifiable = false`

Do not pad them.

---

# Stage 3 — reproduce existing tomography cases

Before scaling to the cohort, verify that the common implementation reproduces the qualitative previous findings for bases that overlap sufficiently with prior experiments.

In particular inspect:

- DNABERT-2
- MosaicBERT
- ModernBERT

Phi-3 may be used as an external implementation check but remains outside the 22-model cohort.

If the newly frozen universal basis differs from an old mechanistic basis, do not expect identical numbers automatically. Explain the difference.

If there is an unexplained methodological contradiction, STOP and diagnose before launching the cohort.

---

# Stage 4 — finite-intervention tomography

For every model with `K >= 2`, run tomography separately at:

- `epsilon = 0.5`
- `epsilon = 1.0`

Reuse exactly the same:

- row-scaling implementation;
- native-objective evaluation datasets;
- masking patterns;
- prompts/windows;
- random seeds;
- endpoint definitions

as Part 2A wherever applicable.

## Observer models

Use the established project observer ladder.

The main comparison is:

### F2 — additive/main-effects observer

Combination effects are explained from individual component contributions.

### F3 — main + pairwise interaction observer

Includes:

- K main effects
- K(K-1)/2 pair terms

Do not add triples.

If F3 fails, report failure of the second-order observer.

---

# Experimental design

For each eligible model:

1. construct deterministic intervention masks;
2. freeze them before evaluating responses;
3. split into fitting / calibration / held-out sets using the established tomography framework;
4. ensure the F3 design matrix is full rank;
5. ensure the model is adequately overdetermined;
6. evaluate predictions only on genuinely held-out masks.

For small K, exhaustive or near-exhaustive designs are fine if a held-out set is preserved.

For larger K, use a seeded balanced design.

Do not tune the design after seeing causal responses.

---

# Primary measurements

For every model and epsilon report:

- F2 held-out MAE
- F2 normalized MAE
- F2 held-out R²
- F3 held-out MAE
- F3 normalized MAE
- F3 held-out R²
- relative F2→F3 MAE improvement
- paired-bootstrap 95% CI
- whether the CI excludes zero
- whether pair terms materially improve prediction
- whether F2 is adequate
- whether F3 is adequate

Use existing locked adequacy/material-improvement criteria where they already exist.

Do not invent thresholds after seeing the results.

---

# Interpretation

Keep these three axes strictly separate:

## 1. Structural geometry
Already measured in Part 1:
- q1
- operator magnitude
- candidate-vs-control structural exceptionalness

## 2. Functional criticality
Already measured in Part 2A:
- singleton causal loss effect

## 3. Causal response complexity
Measured here:
- additive adequacy
- pairwise improvement
- second-order adequacy

The key question is whether these three quantities align across models.

---

# Cohort-level questions

Answer explicitly:

### Q1
How many of the 22 models have K=1 versus K>=2 under the universal structural rule?

### Q2
Among K>=2 models, how often is F2/additive prediction adequate?

### Q3
How often do pair terms materially improve genuinely held-out prediction?

### Q4
How often is F3 still inadequate?

### Q5
Does causal organization change between epsilon=.5 and epsilon=1?

### Q6
Does q1 predict interaction complexity?

### Q7
Does layer-relative Frobenius magnitude predict interaction complexity?

### Q8
Does Part 2A singleton causal magnitude predict interaction complexity?

### Q9
Is there evidence for one common causal organization across the cohort?

### Q10
Do architecture, domain, or scale show any obvious descriptive relationship with causal response complexity?

Do not search aggressively for a new predictor or architecture law.

---

# Main integrated output

Create one table with all 22 models:

`model | K | q1 | layer-relative Frobenius | singleton causal effect | interaction identifiable | F2 MAE eps=.5 | F3 MAE eps=.5 | F2→F3 improvement | F3 adequate | F2 MAE eps=1 | F3 MAE eps=1 | F2→F3 improvement | F3 adequate`

Models with K=1 must remain visible as:

`interaction not identifiable under frozen structural basis`

They must not disappear from the figure/table.

---

# Figures

## Figure A — full-cohort interaction census

For every K>=2 model show F2→F3 relative held-out MAE improvement with 95% CI at:

- epsilon=.5
- epsilon=1

Clearly state how many of the 22 models were not interaction-identifiable.

## Figure B — three-axis relationship

Relate:

- structural geometry
- singleton functional criticality
- interaction complexity

without assigning fake interaction values to K=1 models.

---

# Scientific guardrails

Absolutely no:

- causal-based basis selection;
- hand-picking extra rows for interesting models;
- padding K with controls;
- dropping models with inconvenient outcomes;
- third-order rescue;
- new downstream tasks;
- post-hoc subgroup construction;
- new structural metric invented after seeing tomography results.

The purpose is to learn what the fixed cohort supports, not to preserve the current manuscript story.

---

# Final deliverables

Produce:

- `PART2B_BASIS_RULE_PROPOSAL.md`
- `part2b_candidate_basis.csv`
- frozen intervention designs
- raw causal responses
- `part2b_tomography_results.csv`
- bootstrap results
- integrated 22-model table
- publication-quality figures
- `PART2B_FULL_COHORT_SUMMARY.md`

The final summary should tell me, in plain scientific language:

1. what basis rule was used and why;
2. how many models were interaction-identifiable;
3. what causal organizations were actually observed;
4. whether the current "common structure, heterogeneous causal realization" thesis survives;
5. what claims should be removed;
6. what claims are now stronger;
7. whether any further experiment is genuinely necessary.

Do not rewrite the manuscript yet.