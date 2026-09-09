# PREREG — E7: exact bilinear-operator dimensionality, model-level confirmatory panel

**Lock before generating any confirmatory number from a real target-layer weight (Phi-3) or
any forward pass on a confirmatory-panel model (Evo 2, GenomeOcean-4B, Qwen2.5-7B):**
`python3 src/prereg_lock.py lock docs/prereg/PREREG_exact_operator_dimensionality.md`

**Verify the lock before running any confirmatory step:**
`python3 src/prereg_lock.py verify docs/prereg/PREREG_exact_operator_dimensionality.md`

Synthetic-tensor tests (`experiments/E7_exact_dimensionality/test_spectral_lib.py`, 6/6 green)
may run, and did run, before this lock — no confirmatory-panel weight was used.

---

## Disclosure

E5 (diagonal `f_cross`) and E6 (independent row-level confirmation, Branch C near-miss)
motivated a reformulated hypothesis about the **exact bilinear operator** `U_k` itself, not
its diagonal approximation. This hypothesis was formulated after seeing E5's and E6's
geometry. Per the governing instruction, **all models measured in E5/E6 — Llama-7B,
Mistral-7B, OLMo-7B, GENERator EUK, DNABERT-2, NTv3, and E6's additional OLMo/DNABERT-2/EUK
rows — are discovery/supporting data for this hypothesis, not independent confirmation.**
They are reanalyzed under the new metric only *after* the new-model confirmatory decision is
frozen (Phase 6), and never enter the primary group comparison.

## Core question

For the exact gated-FFN bilinear operator `U_k = sum_i W_down[k,i] * outer(W_gate[i,:],
W_up[i,:])`, are canonical NLP super-weight operators intrinsically close to rank 1, while
genomic high-gain operators are intrinsically higher-dimensional? **Only the structural
question is tested** — NLP SW → low-rank exact `U_k`, vs. genomic high-gain row → higher-rank
exact `U_k`. The causal arrow (structure → causal unit) is explicitly out of scope.

## Exact formula (validated pre-lock, `test_spectral_lib.py`, 6/6 green)

```
U_k = (d[:, None] * W_gate)^T @ W_up             d_i = W_down[k, i]        [d_model, d_model]
sigma_1 >= sigma_2 >= ... = singular values of U_k (torch.linalg.svdvals, float64)

q1      = sigma_1^2 / sum_j sigma_j^2                          -- top singular-energy share
PR_spec = (sum_j sigma_j^2)^2 / sum_j sigma_j^4                -- spectral participation ratio
stable_rank = 1 / q1     -- secondary, algebraically redundant with q1, never primary
```

Numerical tolerance (fixed pre-lock, matches `test_spectral_lib.py`): `RTOL = 1e-8` for
float64 exact-vs-exact agreement (matrix-free construction vs. brute-force summation vs.
direct SVD on identical tensors). No approximate/randomized SVD is used anywhere in E7 — every
target `d_model` here is at most 4,096, and full `svdvals` on a `d_model x d_model` matrix is
confirmed practical (E5/E6 already computed comparable-cost operations in single-digit
seconds to about a minute on this node).

**Why this supersedes the old diagonal PR, and its own limitation** (stated for the record,
matches `test_spectral_lib.py::test_cancellation_case_diagonal_pr_would_disagree`, which
demonstrates a synthetic case where diagonal PR reports near-maximal apparent dimensionality
while the exact operator is the zero matrix): the exact spectrum retains all cross terms and
measures the actual bilinear surrogate operator, with no arbitrary reallocation of pairwise
mass onto individual coordinates. It is still a **bilinear structural surrogate that ignores
the real SiLU/GELU nonlinearity and the data distribution** (except, notably, for Evo 2 layers
> 0, where the real block computes `l3(z1 * z2)` with an `Identity` activation — there, `U_k`
is not a surrogate, it is the literal operator; see `MODEL_PANEL.md`). This module's outputs
are called **exact bilinear-amplifier operator dimensionality**, never "the dimensionality of
the full nonlinear FFN function."

## Confirmatory model panel (frozen; see `MODEL_PANEL.md` for the full audit trail)

| Group | Model | Class | Coordinate source |
|---|---|---|---|
| NLP | Phi-3-mini-4k-instruct | ELIGIBLE-PUBLISHED | Yu et al. Table 2, 6 coordinates (L2 x3, L4 x3) |
| NLP | Qwen2.5-7B | ELIGIBLE-PROSPECTIVE | Frozen detection protocol below |
| Genomic | Evo 2 7B (`arcinstitute/evo2_7b`) | ELIGIBLE-PROSPECTIVE | Frozen detection protocol below |
| Genomic | GenomeOcean-4B (`DOEJGI/GenomeOcean-4B`) | ELIGIBLE-PROSPECTIVE | Frozen detection protocol below |

**Excluded:** Gene42 (UNAVAILABLE — no public checkpoint), Gemma 2 (UNAVAILABLE — gated, no
valid HF token in this environment). Neither is replaced with a second substitute beyond
GenomeOcean-4B, per the model-expansion stop rule (Gene42 + Evo 2 + at most one replacement).

**Independence rule:** the primary unit is **model/checkpoint**, never row. Phi-3's six
published rows aggregate to **one** model-level value (median, fixed before any spectral
value is inspected — see Model-level aggregation below) before entering the group comparison.

## Phase 1 — frozen prospective candidate-selection protocol (Evo 2, GenomeOcean-4B, Qwen2.5)

Independent of `U_k`, `f_cross`, singular values, or any spectral quantity. One forward pass
per model, hooking every eligible gated-FFN module simultaneously (conceptually identical to
this repository's own existing detector, `hooks/activation_hooks.py` +
`scripts/detection/run_detection.py`, reused in spirit, not modified).

**Genomic models (Evo 2, GenomeOcean-4B):**
- **Input:** the single human ACTB 504bp probe already used throughout this repository for
  cross-model SW detection (`_ACTB_504` / `actb_500`, e.g. `SW_TARGETS` in
  `scripts/interpretability/run_sw_broadcast_impulse.py`). Reusing an already-established,
  pre-existing probe — not a sequence chosen for E7 — is itself an anti-cherry-picking
  safeguard; it predates this experiment by months.
- **Preprocessing:** each model's own native tokenizer/preprocessing, BOS prepended if the
  tokenizer defines one, no truncation (504bp fits well within both models' context).
- **Precision:** float32 forward pass (matches this repository's standing D-013 precision
  mandate for detection-class work).
- **Eligible modules:** every layer's down-projection-equivalent module — Evo 2:
  `blocks.{i}.mlp.l3` for `i` in `0..31`; GenomeOcean-4B: `model.layers.{i}.mlp.down_proj` for
  `i` in `0..23`.
- **Statistic:** `out_max` (max absolute value of the module's output, over all sequence
  positions) and `out_channel` (its argmax channel), recorded per layer via a
  `forward_hook` — exactly `hooks/activation_hooks.py`'s existing convention.
- **Candidate rule:** the single `(layer, channel)` pair with the **global maximum** `out_max`
  across all eligible layers.
- **Acceptance criterion:** the candidate's layer `out_max` must exceed **5.0x** that layer's
  own median per-position channel-max (a "max/median" ratio, the same descriptive statistic
  already reported for every existing detection in this repository — Llama-7B 29.4x,
  Mistral-7B 26.8x, OLMo-7B 37.8x, GENERator EUK 12.2x, DNABERT-2 5.0x, NTv3 3.6x — 5.0x is
  chosen as a bar clearly below every one of these already-published, non-E7 ratios, so it is
  a real but not needlessly strict threshold, fixed from **prior, non-E7 data**, not from
  anything computed in this experiment).
- **Tie-breaking:** if two layers tie exactly on `out_max`, the lower layer index wins.
- **Null outcome:** if the global-max candidate's ratio is `< 5.0x`, record **"no high-gain/
  SW-like row detected under the protocol"** for that model and stop — do not search another
  layer, lower the threshold, or select by any spectral quantity.

**NLP model (Qwen2.5-7B):**
- **Input:** the first 20 non-empty lines of WikiText-2 (`wikitext-2-raw-v1`, `test` split, the
  standard corpus this project's own earlier feasibility audit already recommended for exactly
  this purpose), concatenated and truncated to the tokenizer's first 512 tokens — a fixed,
  literally-standard, non-cherry-picked natural-language input.
- **Preprocessing:** Qwen2.5's own tokenizer, BOS/EOS per its default convention.
- **Precision, modules, statistic, candidate rule, threshold, tie-break, null outcome:**
  identical in every particular to the genomic protocol above, applied to
  `model.layers.{i}.mlp.down_proj` for `i` in `0..27`.

**Checkpoint pinning:** none of the three prospective models is currently cached. Each is
fetched at whatever revision `main` resolves to at fetch time; the exact resolved commit hash
is recorded in the confirmatory results (same disclosed gap as E5/E6's unpinned NLP
checkpoints — Llama-7B, Mistral-7B, OLMo-7B).

## Model-level aggregation (fixed before any spectral value is inspected)

For Phi-3 (the only model with multiple predetermined rows): compute `q1` and `PR_spec` for
each of its six published rows individually (reported in full), then aggregate to **one**
model-level value per metric using the **median** across the six rows. No other aggregation
rule was considered or is used.

For every other model (one candidate row each): the model-level value **is** that row's value.

## Primary endpoints (exactly two, matching Phase 2's fixed set; no others)

1. **`q1`** (model-level, aggregated where applicable).
2. **`PR_spec`** (model-level, aggregated where applicable) — reported for consistency;
   algebraically related to `q1` only through the full spectrum shape, not a redundant
   restatement of it (unlike `stable_rank`, which is exactly `1/q1` and is never primary).

## Group-level decision rule (mechanical; stated in full before any confirmatory value is
computed)

With `n=2` new models per group, no p-value is treated as decisive (the best-case exact,
non-asymptotic rank-sum p-value at `n=2` vs. `n=2` is `1/6 ~= 0.167` even under complete
separation — reported for transparency only, via the same enumerated exact test E5/E6 used,
explicitly caveated as uninformative at this sample size).

**Step 1 — completeness check.** If either group has zero surviving candidates (all its
models returned a Phase-1 null outcome), the panel cannot support a group comparison at all:
**overall result is Branch D**, stop. If a group has exactly one surviving candidate (its
other model returned null), the panel **degrades to an exploratory extension** — it is
reported as such explicitly, per the governing instruction, and is not eligible for Branch A.

**Step 2 — Branch A criterion (only evaluated with 2 surviving models per group).**
**PASS** iff both:
- **Complete separation:** `min(NLP q1 values) > max(genomic q1 values)` (both NLP models
  individually exceed both genomic models — this structurally prevents one extreme model
  from carrying the whole result, per the governing instruction).
- **Between-group gap exceeds both within-group ranges:**
  `gap = min(NLP q1) - max(genomic q1)`; `within_nlp = max(NLP q1) - min(NLP q1)`;
  `within_genomic = max(genomic q1) - min(genomic q1)`.
  Require `gap > within_nlp` **and** `gap > within_genomic` — the groups must differ from each
  other by more than either group differs internally, an explicit, pre-specified operational
  form of "broad model-level consistency, not one extreme GLM."

**On PASS → Branch A.** Secondary/consistency, reported not gating: the identical two-part
check applied to `PR_spec` (oriented so genomic > NLP).

**Step 3 — Branch B criterion (only if Step 2 fails).** Branch B is assigned specifically if
the separation fails because **GenomeOcean-4B** (the architecture-bridge model, LLaMA/
Mistral-family on genomic data) lands within or above the NLP group's `q1` range, while Evo 2
remains clearly separated from both NLP models (`Evo2 q1 < min(NLP q1)`, with a gap exceeding
the NLP group's own within-group range). This is the one architecture-linked crossover pattern
this panel is positioned to detect cleanly, given GenomeOcean-4B's specific role — if the
crossover pattern is anything else (e.g., Evo 2 is the one that lands NLP-like, or both
genomic models scatter without a clean single-model explanation), assign **Branch C** instead;
do not force an architecture story onto a pattern this panel was not designed to diagnose.

**Step 4 — otherwise → Branch C** (no robust model-level difference; the direction fails, or
"passes" only through a within-group spread larger than the between-group gap).

## Branch interpretations (preregistered)

- **A:** *"In the tested models, canonical NLP super-weights occupy a near-low-rank/near-
  rank-1 gated-FFN amplifier regime, whereas genomic high-gain rows can occupy intrinsically
  higher-dimensional exact amplifier regimes."* No causal claim; no domain-causation claim;
  no universality claim. Triggers a **design-only** causal-follow-up feasibility write-up
  (no run) and a claim-ledger recommendation for C-032 (see below).
- **B:** *"Operator dimensionality may track architectural implementation more than
  biological domain"* — specifically evidenced by GenomeOcean-4B patterning with NLP while
  Evo 2 does not. Changes paper framing; does not rescue a pure domain contrast. Stop before
  any causal follow-up.
- **C:** E5/E6's cross-term split does not generalize to a genuinely new, model-level panel.
  Hold/retire the dimensionality direction. Do not add more models. Stop.
- **D:** Detection does not transfer to enough of the new panel to support a comparison.
  Report prevalence/nulls. Do not loosen the detector. Stop.

## Ordering (binding)

Per the governing instruction: **new confirmatory models are run and adjudicated first.**
Only after the Phase-5 branch is frozen does E7 apply the identical, unmodified spectral code
to the E5/E6 legacy panel (Phase 6) for supporting/generalization reporting. The legacy
panel's spectra are not inspected, computed, or referenced before the new-model decision is
recorded.

## What even a Branch-A pass does NOT establish

Why training produced this geometry; that genomic sequence data causes it; that operator rank
determines causal dimensionality (the causal arrow is not tested by E7 at all); that any
singular vector corresponds to an implementable parameter group or literal scalar/neuron;
universality across genomic or NLP models generally; anything about the collaborator-reported
DNABERT-2 "redundant pair" mechanism (still without an artifact in this repository).

## No post-hoc changes

No metric, threshold, model, or aggregation rule is added, dropped, or altered after this file
is locked or after any confirmatory-panel weight/forward-pass result is inspected. A
correction is a new, separately-locked file that discloses both.
