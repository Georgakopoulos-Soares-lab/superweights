Small but important correction to E10 before any decoder causal measurements are run.

Keep the **encoder arm exactly as currently specified**:
- MosaicBERT + ModernBERT
- audit whether a pre-existing multi-row high-gain set exists
- if only one row exists, define a frozen structurally selected top-K basis using the existing exact U_k/Ux machinery before any intervention data
- run the full DNABERT-style F0→F3 mechanistic tomography
- held-out prediction, pair-required decision, etc.

Change only the **decoder arm**.

# Revised decoder question

Do NOT only test the already-known #1 high-gain row at alpha = 1, 0.5, 0 and stop there.

That mostly re-demonstrates a known fact: canonical decoder superweights/high-gain rows can be causally important.

The missing question we actually care about is:

> How concentrated is causal importance across the structurally ranked high-gain rows in each decoder?

This determines whether a multi-component causal model is even meaningful.

Use the same already-characterized decoder models:
- Llama
- Mistral
- OLMo
- Phi-3
- Qwen2.5

## Step D1 — freeze structurally ranked decoder rows

For each decoder, BEFORE causal measurement:

1. locate the existing pre-E10 structural ranking / exact U_k or equivalent high-gain score;
2. freeze the top-K structurally ranked high-gain rows;
3. use K=5 by default if all five are legitimate pre-existing high-gain candidates;
4. use K up to 10 only if the existing structural pipeline naturally supports it and the extra rows are genuinely high-gain;
5. do NOT pad with ordinary rows;
6. do NOT choose rows based on causal effects.

Record:
- model
- layer
- row
- structural score
- rank
- source artifact
- exact intervention hook

Also freeze 5 same-layer random-control rows per model, selected before responses.

## Step D2 — measure singleton causal spectrum

For every frozen decoder high-gain row, measure its singleton causal effect under the same intrinsic endpoint.

Primary endpoint:
- causal-LM loss / NLL on a frozen evaluation set

Secondary:
- perplexity
- logit KL
- entropy

Use paired contexts and deterministic evaluation.

For the primary singleton spectrum, full ablation is sufficient if technically stable:

alpha = 1.0 baseline
alpha = 0.0 row suppressed

Optionally retain alpha = 0.5 for the top row if already part of the implementation, but the main new question is the ranked singleton spectrum, not the dose-response curve.

For each row compute:
- raw delta loss
- relative percent change
- effect relative to random controls

## Step D3 — quantify causal concentration

Do not invent many metrics.

At minimum report the ranked singleton causal-effect profile.

Also preregister one simple concentration statistic, for example:

C1 = |ΔL_top1| / sum_i |ΔL_i|

and optionally:

C2 = sum of top-2 absolute effects / sum_i |ΔL_i|

Use absolute effects only for the concentration statistic; also report signed effects separately.

Do not hide the raw values behind the concentration metric.

## Step D4 — decision rule for tomography

This is the key adaptive-but-preregistered logic.

### Case A — top-1 causally dominates

If the #1 structurally ranked row is overwhelmingly dominant relative to the rest and random controls, then STOP for that decoder.

Interpretation:

> causal importance is strongly concentrated in a single dominant high-gain component.

Do NOT run F0–F3 tomography merely for symmetry.

Reason:
if one row already dominates the causal response, a multi-component additive/interactions model is not the natural object, and combination interventions may mainly measure saturation after the model is already badly damaged.

### Case B — several top rows have comparable nontrivial causal effects

If multiple structurally selected rows have substantial singleton effects of comparable order, then tomography becomes scientifically meaningful.

ONLY in that case, flag the decoder as eligible for a second-stage multi-component F0→F3 experiment.

Do not automatically run it unless the prereg says to do so.

At minimum report:

> multi-component decoder causal structure remains unresolved and tomography is justified.

If compute/time is acceptable and the prereg explicitly permits it, then run the same held-out F0→F3 ladder on that decoder.

### Case C — top structural row is weak/null

If the structurally dominant row is not causally special relative to controls:

Interpretation:

> near-rank-1 structural concentration does not imply single-row causal concentration.

Do NOT rescue by searching other rows post hoc.

## Step D5 — architecture-level synthesis

The intended comparison is now:

### Decoder side
Ask:
> Is causal importance concentrated in one dominant structurally high-gain component?

### Encoder side
Ask:
> When single components are insufficient, do pair interactions materially improve held-out finite-intervention prediction?

This asymmetry is deliberate and scientifically motivated.

Do NOT state that decoder and encoder assays are identical.

The logic is hierarchical:

1. first ask whether one component dominates;
2. if yes, stop — no interaction model is needed;
3. if no, multi-component tomography becomes justified.

## Updated possible strong outcome

If:
- most/all decoders show strongly top-1-concentrated causal singleton spectra;
- MosaicBERT + ModernBERT replicate DNABERT with PAIR_TERMS_REQUIRED;

then the supported paper-level statement is:

> In the tested panel, decoder high-gain systems tend to exhibit single-component causal concentration, whereas encoder high-gain systems can require interactions among multiple structurally selected components.

Do NOT write:
- all decoders are single-component;
- all encoders are interactional;
- q1 causes causal concentration;
- architecture determines mechanism universally.

## Updated final report additions

Replace the old decoder dose-response section with:

1. decoder structural top-K freeze
2. decoder singleton causal spectrum for each model
3. random-control comparison
4. C1/C2 causal-concentration summary
5. per-decoder decision:
   - SINGLE_COMPONENT_DOMINANT
   - MULTI_COMPONENT_CANDIDATE
   - STRUCTURAL_CAUSAL_DISSOCIATION
6. whether tomography is justified for any decoder
7. only then, if prereg allows and justified, run decoder tomography for those specific cases

Please update the prereg accordingly BEFORE measuring any decoder responses.

Do not touch the encoder protocol.

The core revised E10 question is now:

> Does the existing structural encoder/decoder distinction correspond to a difference in the natural causal unit — single-component causal concentration in decoders versus interaction-dependent multi-component response in encoders?

Do not optimize the experiment to make this answer yes.