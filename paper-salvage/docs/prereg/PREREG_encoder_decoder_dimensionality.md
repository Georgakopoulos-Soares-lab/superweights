# PREREG — E8: encoder-vs-decoder exact operator dimensionality

**Lock before any forward pass on MosaicBERT or ModernBERT, and before any spectral value is
computed:**
`python3 src/prereg_lock.py lock docs/prereg/PREREG_encoder_decoder_dimensionality.md`

**Verify the lock before running any confirmatory step:**
`python3 src/prereg_lock.py verify docs/prereg/PREREG_encoder_decoder_dimensionality.md`

`test_spectral_lib.py` (E7's, unmodified, 6/6 green) already validates every formula this
prereg uses. No new synthetic tests are needed — E8 reuses E7's `spectral_lib.py` verbatim.

---

## Disclosure

The encoder-vs-decoder pattern this prereg tests was noticed only after inspecting E7's
results (`experiments/E7_exact_dimensionality/RESULTS.md`): DNABERT-2 and NTv3 (both
bidirectional encoders) were the two models with substantially reduced exact-operator
concentration; every decoder measured, NLP or genomic, was comparatively near-rank-1.
**DNABERT-2 and NTv3 are discovery evidence for this hypothesis, not confirmation of it.**
The genuinely new test is whether two independently selected, previously-unmeasured text
encoders pattern with DNABERT-2/NTv3 or with the decoder cluster.

## Hypothesis (association only)

> Among models containing a mathematically comparable gated FFN and exhibiting a
> prospectively detected high-gain row, exact bilinear-operator dimensionality is associated
> more strongly with encoder-vs-decoder organization than with text-vs-genomic domain.

No training-mechanism claim, no attention-masking-causal claim. Association only.

## Model panel (frozen; see `MODEL_AUDIT.md` for the full compatibility trace)

| Model | Checkpoint | Architecture | Attention |
|---|---|---|---|
| MosaicBERT | `mosaicml/mosaic-bert-base` | `BertGatedLinearUnitMLP` (identical class to DNABERT-2) | bidirectional |
| ModernBERT | `answerdotai/ModernBERT-base` | `ModernBertMLP` (independent codebase) | bidirectional, local/global |

Both ELIGIBLE per `MODEL_AUDIT.md`'s hard gate (≥2 independent bidirectional text-encoder
families) — no replacement model is needed or used.

## Predeclared comparison set (existing values — nothing here is computed by E8)

**Decoder reference cluster** (`q1`, `PR_spec`; NLP and genomic decoders, all already
measured in E5/E6/E7):

| Model | Domain | q1 | PR_spec |
|---|---|---:|---:|
| Mistral-7B | text | 0.9922 | 1.0158 |
| Llama-7B | text | 0.9888 | 1.0227 |
| GENERator EUK | genomic | 0.9689 | 1.0653 |
| OLMo-7B | text | 0.9646 | 1.0747 |
| Qwen2.5-7B | text | 0.9529 | 1.0995 |
| Phi-3-mini-4k-instruct | text | 0.9028 | 1.2249 |
| GenomeOcean-4B | genomic | 0.8989 | 1.2243 |

Decoder-cluster floor (min `q1`): **0.8989** (GenomeOcean-4B).
Decoder-cluster ceiling (max `PR_spec`): **1.2249** (Phi-3).

**Encoder discovery cluster** (genomic; motivated this hypothesis, reported for reference
only, never re-entered as new evidence):

| Model | Domain | q1 | PR_spec |
|---|---|---:|---:|
| DNABERT-2 | genomic | 0.7933 | 1.5033 |
| NTv3 | genomic | 0.3889 | 6.4803 |

Evo 2 7B is a Phase-1 **detector null** in E7 and has no spectral value. It is not placed in
either cluster and is not retried here (out of scope for E8, per the governing instruction).

## Phase 2 — frozen prospective detection protocol

Reuses E7's WikiText-2 NLP protocol verbatim (`experiments/E7_exact_dimensionality/
run_phase1_detection.py`'s `qwen25` branch) — the same corpus, same sampling rule, same
statistic, same threshold, chosen because both new candidates are natural-language models, so
E7's already-frozen NLP detector is directly applicable without modification, per the
governing instruction to prefer reuse over inventing a new detector.

- **Input:** the first 20 non-empty lines of WikiText-2 (`wikitext-2-raw-v1`, `test` split),
  concatenated and truncated to the tokenizer's first 512 tokens. Identical text to E7's
  Qwen2.5-7B run (same deterministic extraction rule, re-applied per model's own tokenizer).
- **Preprocessing:** each model's own tokenizer, default special-token convention.
- **Precision:** float32 forward pass.
- **Eligible modules:** every layer's down-projection-equivalent module —
  MosaicBERT: `model.bert.encoder.layer[i].mlp.wo`, `i` in `0..11`; ModernBERT:
  `model.model.layers[i].mlp.Wo`, `i` in `0..21`.
- **Statistic:** `out_max` (max abs value of the module's output over all positions) and
  `out_channel` (its argmax), via a `forward_hook`, identical to E7/E5/E6's convention
  (`hooks/activation_hooks.py`-style).
- **Candidate rule:** the single `(layer, channel)` pair with the **global maximum** `out_max`
  across all eligible layers.
- **Acceptance criterion:** ratio (candidate layer's `out_max` / that layer's own median
  per-position channel-max) **>= 5.0x** — **the identical threshold E7 already fixed**, not
  re-derived here, chosen there from non-E7 published ratios (12.2x–37.8x range) and reused
  verbatim for consistency across this project's detection protocols.
- **Tie-breaking:** lower layer index wins.
- **Maximum candidates per model:** 1.
- **Null outcome:** if the global-max candidate's ratio is `< 5.0x`, record "no high-gain/
  SW-like row detected under the protocol" and stop for that model — no alternate corpus,
  layer search, or threshold change.
- **Seed:** n/a for candidate selection (the WikiText-2 extraction and the top-1 rule are
  both deterministic, no sampling involved).

## Control rows (secondary, descriptive only — never a group-comparison replicate)

For each model with a valid candidate: sample **5 control rows** from the same layer, seeded
random choice without replacement excluding the candidate row itself
(`numpy.random.SeedSequence(42).spawn(2)`, index 0 = MosaicBERT, index 1 = ModernBERT, fixed
by this panel's table order above). Compute `q1`/`PR_spec` for each control row with the
identical `spectral_lib.py` code. Purpose: distinguish "candidate-specific" (candidate
`q1`/`PR_spec` clearly differs from its layer's ordinary rows) from "layer-wide" (candidate
resembles its neighbors) — reported descriptively, never used to adjust the primary decision
or to select a different candidate.

## Phase 3 — primary metrics (identical to E7; no new metric)

`q1 = sigma_1^2 / sum_j sigma_j^2`, `PR_spec = (sum_j sigma_j^2)^2 / sum_j sigma_j^4`, both
from `spectral_lib.row_spectral_metrics`, unmodified.

## Group-level decision rule (mechanical; fixed before any MosaicBERT/ModernBERT value is
computed, anchored entirely to the predeclared reference clusters above)

**Step 1 — completeness (Q1, prevalence).** If **either** model returns a Phase-1 detection
null: **overall result is Branch D.** Stop. Do not loosen the threshold, try a second corpus,
or search another layer.

**Step 2 — dimensionality shift (Q2, conditional; only if both models produced a candidate).**
A model counts as **shifted toward the encoder side** iff its `q1` is **strictly below
0.8989** — the decoder reference cluster's own observed floor (GenomeOcean-4B), used
unadjusted, with **no interpolation and no buffer subtracted toward the encoder side** (a
model does not need to approach DNABERT-2's 0.79 or NTv3's 0.39; it only needs to fall
**outside the entire range every decoder in this project has ever shown**, genomic or NLP).
This is deliberately the more conservative of the two available anchors (decoder floor vs. an
interpolated midpoint) — using the actual worst-case decoder value, not a value chosen to
maximize apparent separation.

- **Both MosaicBERT and ModernBERT shift → Branch A.**
- **Exactly one shifts → Branch B.**
- **Neither shifts → Branch C.**

**Secondary consistency check, reported not gating:** the identical rule applied to
`PR_spec`, oriented so "shifted toward the encoder side" means `PR_spec > 1.2249` (the decoder
cluster's own ceiling, Phi-3). If this disagrees with the `q1`-based branch, the disagreement
is reported explicitly; the `q1`-based branch remains binding, matching this project's
standing discipline of a single primary decision metric.

## Branch interpretations (preregistered)

- **A — encoder pattern confirms.** *"Across the tested gated-FFN models, exact high-gain
  operator dimensionality tracks encoder/decoder organization more closely than sequence
  domain."* Only if Branch A: present the descriptive 2x2 table (no row-level statistics, no
  formal factorial claim) and write a design-only causal-feasibility note (no run).
- **B — mixed.** *"The encoder/decoder dimensionality hypothesis does not cleanly
  replicate."* Stop before any causal follow-up.
- **C — falsified.** *"The post-hoc E7 encoder/decoder pattern was incidental to those two
  genomic models."* Stop. Do not search for a replacement explanation in the same pass.
- **D — prevalence result.** *"E8 cannot confirm conditional spectral dimensionality; absence
  of a detectable high-gain phenotype in text encoders is reported as a separate prevalence
  observation."* Stop. Do not loosen detection.

## What even a Branch-A pass does NOT establish

Why attention directionality would produce this geometry; that causal masking mechanically
causes concentrated operators; any causal-dimensionality claim; any connection to the
collaborator-reported DNABERT-2 "redundant pair" claim (no artifact for that exists in this
repository); any claim beyond association across the specific models tested.

## No post-hoc changes

No metric, threshold, model, or control-row rule is added, dropped, or altered after this
file is locked or after any MosaicBERT/ModernBERT weight or forward-pass result is inspected.
