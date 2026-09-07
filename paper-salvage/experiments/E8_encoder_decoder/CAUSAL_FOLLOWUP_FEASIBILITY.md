# E8 causal-follow-up feasibility — design-only, no run

**Written because Branch A occurred** (`RESULTS.md`). This is a feasibility assessment, not a
plan to execute and not itself a preregistration. No ablation, fine-tune, pair perturbation,
or LayerNorm test is run here or authorized by this document.

## The question worth asking prospectively

> Do low-`q1` (encoder-side) operators require a coordinated, multi-parameter perturbation to
> disrupt the same functional phenotype that a single scalar disrupts in a high-`q1`
> (decoder-side) operator?

## Why this is not straightforward to test, even in principle

1. **A singular vector is not a parameter, a neuron, or a literal scalar.** `U_k`'s singular
   vectors live in `R^d_model` (input/output basis directions of the bilinear operator), not
   in the `d_ffn`-indexed space the existing scalar-ablation infrastructure operates on
   (`W_down[k,i]` for a specific intermediate coordinate `i`). There is no existing script in
   this repository, and no obvious one to write, that "ablates a singular vector" as a weight
   edit — you would need to project the *intermediate activation* onto the singular basis, an
   activation-space intervention, not the weight-space scalar/row ablations this project has
   used throughout (`analysis/ablation.py`, `run_gue_per_row_ablation.py`,
   `run_neuron_causal_intervention_dnabert2.py`). That is a materially different kind of
   experiment, not a drop-in reuse of existing generic infrastructure.
2. **The diagonal decomposition (`c_{k,i}`) — which the existing scalar/row ablation tooling
   is built around — is exactly the object this whole E5→E8 arc has been showing is
   unreliable for the genomic and (per E8) some encoder rows.** Ranking intermediate
   coordinates by `c_{k,i}` to choose "the top-K scalars to ablate" would import the same
   diagonal-approximation problem this arc exists to move past, unless the ranking is
   re-derived from something exact — which does not currently exist as a per-coordinate
   quantity (E6's exact form is a full quadratic form, not a per-`i` ranking; see E6
   `RESULTS.md`'s own discussion of why it resisted a clean per-coordinate reallocation).
3. **A native causal endpoint exists for every model in the confirmatory panel** — masked-
   token loss for MosaicBERT/ModernBERT (both `*ForMaskedLM`), matching this project's
   existing `masked_token_entropy` dispatcher (`analysis/ablation.py`) — so the *measurement*
   side is not a blocker. The blocker is entirely on the intervention side (point 1).

## What native endpoint each model already has (verified, not assumed)

| Model | Endpoint already supported | Where |
|---|---|---|
| MosaicBERT | masked-token entropy/loss (standard `*ForMaskedLM`) | `analysis/ablation.py::masked_token_entropy` (dispatcher already model-agnostic) |
| ModernBERT | masked-token entropy/loss (standard `*ForMaskedLM`) | same |
| DNABERT-2 | masked-token entropy/loss, **plus** an existing fine-tuned splice classifier and the full single-neuron causal-intervention pipeline (`scripts/interpretability/run_neuron_causal_intervention_dnabert2.py`) | already built, most mature of any model in this project |
| NTv3 | masked-token entropy/loss, plus a fine-tuned splice classifier (contested, N-014) | `analysis/ablation.py`, `run_gue_multiseed.py` |

## Verdict

**Feasible only as a new, non-trivial engineering effort — not a reuse of existing
infrastructure.** The measurement side (native loss) is ready for every candidate model. The
intervention side would need a genuinely new activation-space (not weight-space) perturbation
tool operating in the singular-vector basis, which this repository does not have and E8 does
not build. A future session pursuing this should scope it as new infrastructure work, not a
quick follow-up ablation run.

## What is NOT proposed here

No pair-perturbation test, no LayerNorm mechanism test, no connection to the collaborator-
reported DNABERT-2 "redundant pair" claim (still without any artifact in this repository),
and no fine-tuning of any kind. None of this is run by E8, and this document does not
authorize a future session to run it without its own separate scoping and preregistration.
