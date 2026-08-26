Open one final, tightly bounded follow-up experiment:

# E10b — Phi-3 mechanistic tomography

E10 v2 is complete and frozen at commit `626cddd`.

Do NOT modify, rerun, reinterpret, or “improve” any completed E10 result.

This experiment exists for exactly one reason:

> Under the preregistered E10 decoder decision rule, Phi-3 was the only decoder that resolved to `MULTI_COMPONENT_CANDIDATE`, with C1=0.333 and multiple structurally selected rows showing substantial singleton causal effects.

Therefore, unlike Llama/Mistral/OLMo/Qwen2.5, Phi-3 is the only decoder for which multi-component tomography is scientifically justified.

This is NOT a rescue experiment.
This is the exact second-stage experiment licensed by the original adaptive E10 logic.

No other decoder tomography is authorized.

---

# Scientific question

For Phi-3's frozen structurally selected high-gain basis:

> When multiple high-gain rows have substantial singleton causal effects, is the finite intervention response adequately described by additive contributions, or are explicit pair interactions required?

This directly resolves the only currently unresolved decoder causal regime.

---

# Hard constraints

- Phi-3 only.
- Use the exact Phi-3 checkpoint already used in E10.
- Use the exact frozen structurally selected row basis from E10 v2.
- Do NOT change K.
- Do NOT add/remove/reorder rows based on causal effects.
- Do NOT recompute the structural basis.
- Do NOT use q1 for selection.
- Do NOT add new tasks.
- Do NOT change the intrinsic endpoint.
- Do NOT add triples, HVPs, SAEs, PCA directions, learned directions, or any new representation.
- Do NOT run decoder tomography on any other model.
- Do NOT rewrite the manuscript while this is running.
- Preserve all raw outputs.

---

# Phase 0 — audit the frozen Phi-3 basis

Before any new measurement, write:

`paper-salvage/experiments/E10b_phi3_tomography/PHI3_BASIS_AUDIT.md`

Record:

- exact checkpoint / revision;
- architecture and module path;
- exact E10 v2 structurally selected rows;
- their structural ranks;
- exact ||U_k||_F / layer-relative magnitude values;
- q1 values as annotation only;
- E10 singleton causal effects;
- E10 C1;
- random-control effects;
- why Phi-3 alone qualifies for tomography under the prior locked rule.

Do not change anything.

---

# Phase 1 — preregister E10b

Create:

`paper-salvage/docs/prereg/PREREG_E10b_phi3_tomography.md`

Lock before measuring any new multi-row response.

Freeze:

- exact Phi-3 revision;
- frozen E10 v2 row basis;
- exact intrinsic endpoint;
- corpus / contexts;
- tokenization;
- intervention hook;
- epsilon values;
- mask-generation scheme;
- fit/calibration/held-out split;
- ridge grid;
- observer definitions;
- metrics;
- bootstrap procedure;
- pair-required decision rule;
- interpretation branches;
- stop rule.

Use the same tomography logic as E9/E10 encoders unless Phi-3's architecture forces a purely technical implementation change.

If any implementation change is needed, document it before final held-out evaluation.

---

# Phase 2 — intervention design

Use the same conceptual intervention:

`alpha_i = 1 - epsilon * a_i`

with:

- `epsilon = 0.5`
- `epsilon = 1.0`

Use the exact frozen Phi-3 basis size K from E10 v2.

Generate:

- singleton interventions;
- fit masks;
- calibration masks;
- final held-out masks.

Target mask densities:

`rho in {0.25, 0.5, 0.75}`

The final held-out masks must not be used for:
- lambda selection;
- pair selection;
- hyperparameter tuning;
- threshold tuning;
- basis changes.

---

# Phase 3 — design-matrix validity

Before measuring final responses, verify the design mathematically.

For K rows, lifted feature count is:

`K + K*(K-1)/2`

Check:

- additive design rank;
- lifted design rank;
- condition number;
- maximum column correlation;
- duplicate / aliased columns.

Use enough fit masks for identifiability.

Do NOT proceed with a rank-deficient lifted design.

If the initial mask pool is insufficient, regenerate masks BEFORE measuring responses and document the pre-measurement correction.

---

# Phase 4 — primary endpoint

Use the exact E10 decoder intrinsic endpoint:

- causal-LM loss / NLL

on the exact frozen evaluation distribution already used for Phi-3 E10 singleton measurements.

Secondary endpoints may remain:

- perplexity;
- logit KL;
- next-token entropy.

Do not add behavioral or linguistic tasks.

---

# Phase 5 — observer ladder

Run the same four observer families.

## F0 — singleton additive

Use independently measured singleton effects:

`y_hat(a) = sum_i a_i x_i`

where `x_i` is the finite singleton effect from the frozen intervention semantics.

## F1 — scalar-calibrated additive

`y_hat(a) = g * F0(a)`

Fit `g` only on calibration data.

## F2 — jointly fitted additive

`y_hat(a) = sum_i beta_i a_i`

Use ridge with lambda selected only on fit/calibration data.

## F3 — main + pair interactions

`y_hat(a) = sum_i beta_i a_i + sum_{i<j} Gamma_ij a_i a_j`

Use ridge and the same fitting discipline as E9/E10.

Primary comparison:

**F2 vs F3 on untouched held-out masks**

---

# Phase 6 — metrics

Report for each epsilon:

- R²
- MAE
- RMSE
- normalized MAE
- F2→F3 absolute MAE improvement
- F2→F3 relative MAE improvement
- bootstrap CI on the MAE improvement

Use the same bootstrap unit / context clustering discipline as E10.

Do not treat masks as independent model replicates.

---

# Phase 7 — preregistered decision

Use the same practical logic as E9/E10:

## `PAIR_TERMS_REQUIRED`

if:
- F3 materially improves held-out prediction over F2;
- the preregistered practical improvement threshold is met;
- bootstrap CI on improvement excludes zero;
- lifted design is identifiable.

## `ADDITIVE_SUFFICIENT`

if:
- F2 is already adequate under the preregistered adequacy criteria;
- F3 does not materially improve held-out prediction.

## `MIXED_OR_UNRESOLVED`

if:
- F3 improves somewhat but does not clearly meet the pair-required rule;
- or neither additive nor pairwise model adequately describes the response.

Do NOT change thresholds after seeing results.

---

# Phase 8 — retrospective pair inspection

Only after the primary decision is frozen:

- rank all pair coefficients by |Gamma|;
- inspect whether the strongest pair involves the two largest E10 singleton-effect rows;
- inspect whether pair structure tracks structural rank;
- inspect same-layer vs cross-layer enrichment descriptively.

Do NOT use this to refit or change the basis.

No pair was independently known before E10b, so do not manufacture a “critical pair” narrative unless the held-out evidence and coefficient structure genuinely support one.

---

# Interpretation branches

## Branch A — additive Phi-3

If F2 is adequate and F3 adds little:

Interpretation:

> Phi-3 is a multi-component decoder in terms of singleton causal contribution, but its combined finite response remains largely additive.

This would strengthen the architecture-level contrast:

- encoders: pair interactions required;
- decoder exception Phi-3: several components matter, but they combine simply.

This is the strongest outcome for a causal-architecture paper.

## Branch B — pair-interactional Phi-3

If F3 materially improves over F2:

Interpretation:

> interaction-dependent high-gain causal organization is not encoder-specific.

This would weaken any encoder-vs-decoder causal-complexity claim and support a more general conclusion:

> architecture organizes structural geometry better than it determines causal response complexity.

This is still a useful and publishable result.

## Branch C — unresolved / strongly nonlinear

If neither F2 nor F3 adequately predicts held-out response:

Interpretation:

> Phi-3 exhibits multi-component causal dependence that is not captured by first- or second-order finite-response models under this basis.

Do NOT add triples or higher-order models.

Stop.

---

# Relationship to the paper

Do not edit manuscript prose yet.

After E10b, the architecture section should be decided as follows:

### If Branch A:
Possible stronger statement:

> In the tested panel, decoder high-gain systems are often either single-component dominated or multi-component but approximately additive, whereas all tested encoder high-gain systems require explicit pair interactions.

Still not universal, and architecture is an association, not a causal determinant.

### If Branch B:
Use:

> Encoder/decoder organization predicts structural geometry more consistently than causal response complexity; interactional causal organization can also occur in decoders.

### If Branch C:
Use:

> Causal response complexity is model-specific beyond the structural architecture trend.

---

# Raw artifact requirement

Save machine-readable:

- exact frozen basis;
- mask matrices;
- split assignments;
- raw per-context responses;
- singleton effects;
- fit/calibration/held-out predictions;
- lambda choices;
- F0/F1/F2/F3 coefficients;
- Gamma matrix;
- bootstrap samples / summaries;
- all metrics;
- exact commands;
- model revision;
- environment details;
- checksums where appropriate.

No headline number may exist only in Markdown.

---

# Git discipline

Suggested commits:

1. E10b audit
2. E10b prereg + lock
3. implementation/tests
4. measurements
5. observer fitting/statistics
6. frozen synthesis

Never `git add -A`.

Verify all prior locks plus the new E10b lock before and after measurement.

---

# Final report

Return exactly:

1. Phi-3 frozen basis audit
2. E10 evidence that licensed tomography
3. E10b prereg hash/timestamp
4. design-matrix conditioning
5. baseline/singleton regression check
6. epsilon=0.5 F0–F3 results
7. epsilon=1.0 F0–F3 results
8. F2→F3 bootstrap comparison
9. mechanical decision: ADDITIVE_SUFFICIENT / PAIR_TERMS_REQUIRED / MIXED_OR_UNRESOLVED
10. retrospective pair structure
11. what this changes about the architecture-level paper claim
12. what this does NOT establish
13. raw artifact inventory
14. commit inventory
15. ONE exact next action

Then stop.

---

# Core question

Answer only this:

> In the one decoder that E10 identified as genuinely multi-component, does finite causal response remain approximately additive, or are explicit pair interactions required?

Do not optimize the experiment to favor either answer.