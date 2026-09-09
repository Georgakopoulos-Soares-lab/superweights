# Claude Code Prompt — E10 NLP Architecture Replication of High-Gain Causal Phenotypes

Continue from the current integrated superweights repository state.

We are opening **one bounded follow-up experiment**:

# E10 — NLP replication of architecture-associated causal phenotypes

This experiment is motivated by the current structural result:

- exact high-gain operator geometry does **not** cleanly separate text from genomics;
- in the tested panel, **causal decoders are generally more near-rank-1**, while **bidirectional encoders are less completely collapsed / more distributed**;
- this is an association, not a universal law;
- ModernBERT is a known borderline case.

The goal now is to ask whether the **causal phenotype** observed in the genomic exemplars also appears in the already-characterized NLP models.

We are NOT adding new model families.

Use the exact NLP models already used in E7/E8:

## NLP decoders
- Llama
- Mistral
- OLMo
- Phi-3
- Qwen2.5

## NLP encoders
- MosaicBERT
- ModernBERT

Genomic models are anchors / prior evidence, not new primary measurements here:

- GENERator EUK = genomic decoder exemplar with strong single-row causal composition sensitivity
- DNABERT-2 = genomic encoder exemplar with pair-interactional causal response
- NTv3 = genomic encoder negative case, structurally high-gain but functionally inert

Do NOT add more models unless one of the above is technically impossible to run and the replacement is approved explicitly.

---

# Scientific question

We already have a structural architecture pattern:

> decoder high-gain operators tend to be more concentrated / near-rank-1, while encoder high-gain operators tend to be less completely collapsed.

Now test whether the **characteristic causal phenotype** also tends to differ by architecture.

For decoders, replicate the **GENERator-style single-pathway intervention experiment** on the already-established dominant super-weight/high-gain row.

For encoders, replicate the **DNABERT-style multi-component mechanistic-tomography experiment**.

Important:

These are intentionally architecture-aligned protocols, NOT identical assays.

Therefore E10 may support:

> the tested decoder and encoder models show different characteristic causal phenotypes consistent with their structural organization.

E10 by itself does NOT prove:

> decoders are additive and encoders are interactional.

The decoder arm does not test pairwise observer complexity unless a separate experiment is later justified.

---

# Current structural values / orientation

Use repository artifacts as the source of truth, but the expected existing exact-q1 landscape is approximately:

## Text decoders
- Mistral ~0.992
- Llama ~0.989
- OLMo ~0.965
- Qwen2.5 ~0.953
- Phi-3 ~0.903

## Text encoders
- MosaicBERT ~0.477
- ModernBERT ~0.897

Do not rewrite or recompute the structural story unless required to locate the already-frozen intervention objects.

---

# E10 hard rules

- No new model discovery.
- No new structural metric.
- No new q1 hypothesis.
- No post-hoc row selection from causal effects.
- No changing endpoints after seeing results.
- No adding extra dose points after seeing results.
- No changing encoder basis after intervention responses are measured.
- No rescue with SAEs, PCA, learned directions, triples, HVPs, or new tasks.
- No claim that encoder/decoder is universal.
- No manuscript rewrite during measurement.
- All new results must have machine-readable raw artifacts.
- Preregister before any new causal measurement.

---

# Phase 0 — audit prior structural objects

Create:

`manuscript/experiments/E10_nlp_architecture_causal/`

Write:

`MODEL_AND_BASIS_AUDIT.md`

For each of the seven NLP models, identify from PRE-E10 artifacts:

- model checkpoint / exact revision;
- architecture / module path;
- already-established high-gain row(s);
- exact `U_k` / `Ux` structural artifacts;
- q1 / PRspec artifact;
- prior activation-detection artifact if available;
- corpus / probe used in E7/E8;
- whether there is already a pre-existing SET of high-gain rows or only one canonical row.

Do not run causal interventions yet.

---

# Phase 1 — freeze decoder intervention objects

For each decoder:

- Llama
- Mistral
- OLMo
- Phi-3
- Qwen2.5

Use the exact already-established dominant super-weight / high-gain row from pre-E10 work.

Do NOT choose a different row based on causal response.

Record for each:

- layer;
- row;
- tensor/module;
- exact intervention hook location;
- whether it is an output row / residual write;
- relation to the exact `U_k`;
- previous published or project-specific identity if applicable.

Create:

`DECODER_INTERVENTION_FREEZE.md`

The decoder experiment is a **single-row dose response**, conceptually matching GENERator.

---

# Phase 2 — freeze encoder intervention bases

For:

- MosaicBERT
- ModernBERT

First answer explicitly:

> Do we already have a pre-E10 set of structurally selected high-gain rows, or only one canonical row?

## Case A — a pre-existing set exists

If a multi-row set was already frozen / stored before E10:

- use it exactly;
- do not reorder by causal response;
- do not drop weak rows;
- record provenance.

Target basis size should preferably be in the same rough range as DNABERT (`~8–12`) if the pre-existing set naturally supports that.

## Case B — only one canonical high-gain row exists

If E8 only identified one canonical row, construct the encoder intervention set **before any causal measurements** using the existing exact structural operator.

Use the repository's already-defined exact `U_k` / `Ux` machinery.

Do NOT invent a new causal score.

Procedure:

1. verify the exact `U_k` / `Ux` computation already used in E7/E8;
2. compute or load the exact structural score for all eligible gated-FFN output rows in that encoder;
3. rank rows using the pre-existing structural score / exact operator norm already used by the project;
4. select a fixed top-K set **without looking at any intervention response**;
5. include the known E8 canonical high-gain row if it is naturally inside the top-K; if it is not, STOP and audit why before proceeding.

Default `K = 10` if the architecture and structural artifacts make this sensible.

If fewer than 10 rows meet the project's existing high-gain eligibility criterion, use all eligible rows and report the smaller K.

Do NOT pad with arbitrary ordinary rows merely to reach K=10.

Do NOT impose post-hoc layer diversity unless the prior structural pipeline already had such a rule.

The purpose is to define a **structurally selected intervention basis**, not a causally optimized basis.

Create:

`ENCODER_BASIS_FREEZE.md`

For every selected row record:

- model;
- layer;
- row;
- exact structural score;
- q1 if available for that row;
- source artifact;
- whether selected pre-existing vs top-K exact-U construction.

Once this file is committed / prereg locked, the basis cannot change.

---

# Phase 3 — define intrinsic endpoints

We want model-intrinsic endpoints so the NLP replication is not dependent on arbitrary downstream tasks.

## Decoder primary endpoint

Use causal-LM loss / NLL on the same frozen evaluation distribution already used in the prior E7 structural/detection work if available.

Prefer reuse of the existing WikiText-2 evaluation windows / exact corpus slice.

Freeze:

- sequence set;
- tokenization;
- context length;
- batch size;
- dtype;
- deterministic settings;
- model revision.

Primary scalar:
- mean NLL / loss

Secondary:
- perplexity
- logit KL vs baseline
- next-token entropy

Do NOT create a hand-selected semantic steering task.

This arm is about **single-pathway causal dose response**, not necessarily semantic steering.

## Encoder primary endpoint

Use pretrained MLM loss / NLL on a frozen held-out text set.

Prefer the exact corpus / masking protocol already used in E8 if available.

Freeze:
- sequences;
- masking positions;
- masking seed;
- tokenization;
- max length;
- batch size;
- dtype;
- model revision.

Primary scalar:
- mean MLM loss

Secondary:
- logit KL at masked positions
- residual norm at the intervention layer, if easy and predeclared

Do not add downstream classification tasks.

---

# Phase 4 — preregistration

Before any new intervention response is measured, create and lock:

`manuscript/docs/prereg/PREREG_E10_nlp_architecture_causal.md`

The prereg must freeze:

- seven models;
- exact revisions;
- decoder rows;
- encoder bases;
- corpora / sequence IDs;
- intervention semantics;
- alpha / epsilon values;
- mask generation;
- train/calibration/test split;
- primary endpoints;
- control selection;
- statistics;
- architecture-level decision rules;
- stop rules.

Commit the prereg and verify all previous locks before proceeding.

---

# ARM A — NLP DECODERS
# Replicate GENERator-style single-pathway intervention

For each decoder, intervene on the frozen dominant row.

Use the same conceptual scaling:

`alpha = 1.0` = untouched  
`alpha = 0.5` = half-strength  
`alpha = 0.0` = full suppression / ablation

Do NOT add a denser alpha grid after seeing results.

If the exact existing hook naturally parameterizes as epsilon, map it explicitly to alpha and document.

For each model measure the primary endpoint at all three alpha values.

Use paired contexts and deterministic evaluation.

---

# Decoder controls

For each decoder select controls BEFORE responses:

- 5 same-layer random output rows, OR
- the exact control convention already used in the model's prior super-weight work.

Freeze control RNG seed.

Apply the same alpha grid.

Do not choose "nice flat" controls after seeing data.

---

# Decoder primary questions

For each model:

1. Does suppressing the dominant high-gain row cause a substantially larger intrinsic loss change than random controls?
2. Is the partial intervention (`alpha=0.5`) directionally intermediate between baseline and ablation?
3. Is the effect monotonic across the 3 frozen points?
4. Does logit KL / entropy move consistently with loss?

Important wording:

With only 3 points, do NOT call this a validated continuous steering curve.

Allowed phenotype labels:

- strong single-row causal sensitivity, directionally monotonic over the tested doses;
- strong but non-monotonic causal sensitivity;
- weak / null single-row effect.

Do NOT call a decoder "additive" from this experiment.

---

# Decoder effect normalization

Because model losses live on different scales, report BOTH raw and normalized effects.

For each model:

- raw delta NLL / loss;
- relative percent change from baseline;
- standardized effect relative to same-layer random controls.

A reasonable control-normalized statistic is:

`target_effect / median(|random_control_effect| + small_epsilon)`

but define the exact form BEFORE measuring.

Do not use one normalization to hide absolute model differences.

---

# ARM B — NLP ENCODERS
# Replicate DNABERT-style mechanistic tomography

For each encoder basis of size `n_E`:

Generate subset interventions over the frozen structurally selected rows.

Use the same E9 conceptual intervention:

`alpha_i = 1 - epsilon * a_i`

with:

- `epsilon = 0.5`
- `epsilon = 1.0`

Reuse the E9 mask-generation / conditioning logic where architecture permits.

Target mask densities:

`rho in {0.25, 0.5, 0.75}`

Create independent:

- singleton measurements;
- fit masks;
- calibration masks;
- final held-out masks.

The final held-out masks must not be touched during:
- lambda selection;
- model selection;
- pair selection;
- threshold tuning.

---

# Encoder design-size requirement

For an encoder basis of size `n_E`, the lifted model has:

`n_E + n_E*(n_E-1)/2`

features.

Before any final measurement:

- ensure the training lifted matrix has full rank if feasible;
- inspect condition number;
- inspect max column correlation;
- reject duplicated / aliased columns.

If a K=10 basis is used, the lifted model has 55 features.

Use enough fit masks to make the lifted design identifiable and reasonably conditioned.

Do NOT repeat the earlier initial rank-deficiency mistake.

Generate / freeze masks first, test their design mathematically, then measure responses.

---

# Encoder observer ladder

Run the exact E9 family logic.

## F0 — singleton additive

`y_hat(a) = sum_i a_i x_i`

where `x_i` is the measured singleton finite effect.

Evaluate on final held-out masks.

## F1 — scalar-calibrated additive

`y_hat(a) = g * sum_i a_i x_i`

Fit `g` only on calibration data.

## F2 — jointly fitted additive

`y_hat(a) = sum_i beta_i a_i`

Use ridge or the exact E9 estimator.

Tune hyperparameters only on training/calibration.

## F3 — main + pair interaction

`y_hat(a) = sum_i beta_i a_i + sum_{i<j} Gamma_ij a_i a_j`

Use the same E9 ridge / estimator unless an architecture-specific technical issue forces a change.

If changed, document before final held-out evaluation.

Primary comparison:

**F2 vs F3 held-out prediction**

---

# Encoder adequacy / pair-required rule

Use the SAME preregistered decision logic as E9 unless there is a compelling reason to change it before measurement.

Report:

- R²
- MAE
- RMSE
- normalized MAE
- F2→F3 relative MAE improvement
- bootstrap CI over the natural independent unit

PAIR TERMS REQUIRED if:

- F0/F1/F2 fail or remain materially worse;
- F3 improves held-out MAE by the preregistered practical threshold;
- bootstrap CI excludes zero;
- lifted design is identifiable.

Do not require F3 itself to reach perfect adequacy.

As in DNABERT, pair terms may be **necessary but not sufficient**.

---

# Encoder retrospective interaction inspection

Only AFTER the primary held-out decision is frozen:

- rank all pair coefficients by `|Gamma|`;
- inspect whether the strongest terms cluster among the highest structural rows / shared layers / repeated structural families;
- compare to the canonical E8 high-gain row.

Do NOT use this retrospective inspection to change the basis or refit the model.

For MosaicBERT / ModernBERT there may be no independently known critical pair.

Do not manufacture one.

---

# Optional same-layer ordinary controls for encoders

If existing E8 same-layer ordinary-row controls can be intervened with minimal extra compute, include a small frozen control analysis.

Question:

> does a tomography basis built from ordinary same-layer rows show the same pair requirement?

This is OPTIONAL.

Do not let this expand E10.

If it requires substantial new engineering, skip and report why.

---

# Phase 5 — architecture-level synthesis

After ALL seven NLP model results are frozen, create:

`ARCHITECTURE_SYNTHESIS.md`

Do not inspect or write this section model-by-model before all primary results are frozen.

Create a table:

| Model | Domain | Architecture | q1 | high-gain intervention type | primary causal result | pair tomography run? | simplest supported causal description |
|---|---|---|---:|---|---|---|---|

Include genomic anchors descriptively:

- GENERator EUK
- DNABERT-2
- NTv3

But keep E10's new statistical evidence separate from prior genomic evidence.

---

# What E10 CAN establish

Potential strong result:

### Decoder side
Across the already-characterized NLP decoders, the canonical dominant high-gain row repeatedly shows a large, directionally consistent finite causal response relative to random controls.

### Encoder side
Across MosaicBERT and ModernBERT, additive observers fail and pair-aware observers materially improve held-out finite-intervention prediction.

If both happen, allowed conclusion:

> The structural encoder/decoder pattern is accompanied by a corresponding difference in characteristic causal phenotype in the tested models: dominant decoder high-gain rows often act as strong single-pathway intervention points, whereas encoder high-gain bases can require explicit component interactions to predict finite perturbations.

This is a **tested-panel architectural association**, not a universal law.

---

# What E10 CANNOT establish

Even if results are clean, do NOT write:

- "all decoders are additive";
- "all encoders are interactional";
- "q1 causes pairwise redundancy";
- "rank-1 structure causes single-row causality";
- "distributed structure causes interactions";
- "the architecture determines the causal mechanism";
- "the encoder/decoder causal distinction is universal".

Why:

The decoder arm is a single-row dose-response assay, not full multi-row tomography.

Therefore the cleanest causal-complexity claim remains on the encoder side.

If we later want a symmetric statement about **observer complexity**, decoders would also need a defensible multi-component basis and the F0–F3 ladder. That is NOT part of E10 unless explicitly approved later.

---

# Predeclared outcome branches

## Branch A — architecture-consistent causal phenotypes

- most / all tested NLP decoders show strong dominant-row causal sensitivity substantially above controls;
- both text encoders show PAIR_TERMS_REQUIRED under held-out tomography.

Interpretation:

> The previously observed structural architecture association is mirrored by distinct characteristic causal phenotypes in this tested panel.

This materially supports reframing the manuscript as a broader CS / mechanistic-interpretability paper.

## Branch B — encoders interactional, decoders mixed

- both encoders require pair terms;
- decoder single-row effects are heterogeneous / some null.

Interpretation:

> Encoder interactional organization replicates beyond genomics, but a generic decoder single-pathway phenotype is not established.

Still valuable; keep stronger mechanistic emphasis on DNABERT and selected decoders.

## Branch C — encoder replication fails

- MosaicBERT and/or ModernBERT are adequately additive / pair terms do not materially help.

Interpretation:

> DNABERT's interactional causal response is model-specific despite the broader structural encoder/decoder association.

This is scientifically important.

Do NOT search for other text encoders.

## Branch D — decoder phenotype fails broadly

- canonical decoder super-weight rows are weak/null relative to controls.

Interpretation:

> near-rank-1 structural concentration does not imply a strong single-row causal intervention phenotype.

Do not rescue with other decoder rows.

## Branch E — mixed architecture result

No consistent causal organization by architecture.

Interpretation:

> encoder/decoder organization predicts structural geometry better than causal response phenotype.

This is a clean negative result and should stop the causal-architecture expansion.

---

# Statistics and independence

Do not treat rows or masks as independent model-level replicates.

For within-model effects:
- use paired contexts;
- bootstrap over the natural independent sequence/context unit;
- preserve mask clustering when appropriate.

For architecture-level synthesis:
- primary unit is the MODEL.

With only 5 decoders and 2 text encoders, avoid grand inferential claims.

Use:
- model-level descriptive consistency;
- exact binomial/permutation-style summaries only if justified and preregistered;
- confidence intervals on within-model causal effects.

Do not report p-values across hundreds of masks as if n=hundreds models.

---

# Figure outputs if Branch A/B is interesting

Do not build final manuscript figures until results are frozen.

Prepare source data for:

## Figure E10-A — decoder dose responses

One compact panel showing normalized target-row vs random-control effects across:
- Llama
- Mistral
- OLMo
- Phi-3
- Qwen2.5

Show the 3 alpha points honestly.
Guide-the-eye connectors may be dotted.
Do not imply fitted continuous curves.

## Figure E10-B — encoder tomography

For MosaicBERT and ModernBERT:
- F0/F1/F2/F3 held-out prediction metrics;
- F2→F3 MAE improvement with CI.

## Figure E10-C — structural + causal synthesis

Model-level summary:
- q1 / structural class;
- dominant-row causal phenotype or pair requirement.

This is descriptive.
Do NOT fit a q1→causal-complexity regression with this n.

---

# Relationship to current paper

If Branch A occurs, the paper may be reframed from primarily a genomic-model superweights study into a broader mechanistic-interpretability / CS paper that was motivated by genomics.

Possible high-level story:

1. High-gain FFN structures recur across text and genomic foundation models.
2. Exact structural geometry associates more with encoder/decoder organization than domain.
3. The NLP replication asks whether causal phenotype follows the same architectural axis.
4. DNABERT-2 remains the deepest interactional mechanistic case:
   redundant pair, pretraining origin, codominance, held-out tomography.
5. GENERator remains the deepest genomic decoder case:
   near-rank-1 structure, BOS-associated phenotype, strong causal composition sensitivity.
6. NTv3 remains the falsification case:
   strong structural high gain but causal null.

Do not rewrite the manuscript until E10's mechanical branch is frozen.

---

# Raw artifact requirement

Every E10 number must have machine-readable provenance.

Save:

- exact model revisions;
- frozen decoder rows;
- frozen encoder bases;
- exact U_k/Ux ranking used for basis construction;
- corpora / sequence IDs;
- masking patterns;
- mask matrices;
- split assignments;
- raw per-context responses;
- fitted observer coefficients;
- pair coefficients;
- held-out predictions;
- bootstrap outputs;
- control rows;
- plotting source data;
- exact commands;
- hashes/checksums where appropriate.

Do not allow Markdown summaries to become the sole surviving source of headline numbers.

---

# Git discipline

Use separate commits for:

1. model/basis audit
2. basis freeze
3. implementation/tests
4. prereg lock
5. decoder measurements
6. encoder measurements
7. observer fitting/statistics
8. synthesis
9. figures, only after results freeze

Never `git add -A`.

Verify all prereg locks before and after.

---

# Final report

Return exactly:

1. Model and structural-artifact audit
2. Decoder intervention freeze
3. Encoder basis audit: pre-existing set vs single row
4. If needed, exact U_k/Ux-derived encoder top-K construction
5. Prereg hash and timestamp
6. Decoder baseline regression
7. Decoder dose-response results model-by-model
8. Decoder random-control comparison
9. MosaicBERT mask/design conditioning
10. MosaicBERT F0–F3 held-out tomography
11. ModernBERT mask/design conditioning
12. ModernBERT F0–F3 held-out tomography
13. Encoder pair-required decisions
14. Retrospective pair structure
15. Architecture synthesis table
16. Mechanical Branch A/B/C/D/E
17. What is established
18. What is NOT established
19. Whether the CS / mechanistic-interpretability reframing is now justified
20. Whether any further experiment is actually necessary
21. Raw artifact and commit inventory
22. ONE exact next action

Then stop.

---

# Core E10 question

Answer:

> Do the already-characterized NLP models show causal phenotypes consistent with the structural encoder/decoder organization we observed — strong dominant-pathway sensitivity in decoders and interaction-dependent finite response in encoders?

Do not optimize the experiment to make the answer yes.

If it fails, report the failure and stop.
