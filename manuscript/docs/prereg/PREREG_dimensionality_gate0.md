# PREREG — E5 Gate 0: weight-only mechanism of NLP-vs-genomic FFN scalarization

**Lock before generating any confirmatory number from real target-row weights:**
`python3 src/prereg_lock.py lock docs/prereg/PREREG_dimensionality_gate0.md`

**Verify the lock before running the confirmatory script:**
`python3 src/prereg_lock.py verify docs/prereg/PREREG_dimensionality_gate0.md`

Synthetic-tensor tests (`experiments/E5_dimensionality/test_dimensionality_lib.py`) may be
run before this file is locked — they validate the code, not the hypothesis. No tensor from
any of the six primary-panel checkpoints may be passed through the D/A/C decomposition, the
exactness check, or the permutation test before this file is locked.

---

## Premise (do not re-litigate here — see `experiments/E5_dimensionality/README.md`)

This is **not** a test of "Frobenius fails in genomic models." Row-level recovery
(`‖U_k‖_F` rank of the target row) is already rank 1/N in five of the six primary-panel
models and is not re-measured here. Gate 0 asks a narrower, within-row question: is the
observed granularity gap (NLP top-1 share 0.89–0.99, PR 1.02–1.24 vs. genomic top-1 share
0.18–0.39, PR 3.6–22.7) produced by an unusually strong *pairing* between the down-projection
term and the gated-amplifier term in NLP, or does it just reflect one of the two marginal
distributions being more extreme in NLP regardless of pairing?

## Primary panel (fixed; identical to `README.md`, restated here for self-containedness)

| # | Group | Model | Checkpoint | Layer | Row *k* | d_model | d_ffn |
|---|---|---|---|---|---|---|---|
| 1 | NLP | Llama-7B | `huggyllama/llama-7b` | 2 | 3968 | 4096 | 11008 |
| 2 | NLP | Mistral-7B | `mistralai/Mistral-7B-v0.1` | 1 | 2070 | 4096 | 14336 |
| 3 | NLP | OLMo-7B | `allenai/OLMo-7B-0724-hf` | 1 | 269 | 4096 | 11008 |
| 4 | Genomic | GENERator EUK 3B | `GenerTeam/GENERator-v2-eukaryote-3b-base` | 4 | 2371 | 3072 | 8448 |
| 5 | Genomic | DNABERT-2 117M | `zhihan1996/DNABERT-2-117M` @ `7bce263b15377fc15361f52cfab88f8b586abda0` | 5 | 603 | 768 | 3072 |
| 6 | Genomic | NTv3 650M | `InstaDeepAI/NTv3_650M_pre` @ code rev. `0ecff3637f0d3ba5b686d1095083218157c2ca34` | 11 | 1472 | 1536 | 6144 |

**Panel order above is fixed** and used to derive per-model permutation seeds (§0D). It is not
re-orderable after locking.

## Exclusions (fixed; no post-hoc exclusion beyond these two)

- **GENERator PROK** — excluded. C-001 on hold (N-009/N-015); stored rank-1 artifact does not
  reproduce at its own layer; cause undetermined; no alternative layer is searched.
- **Evo1** — excluded. SW interpretation itself contested (C-009/C-011: structurally
  concentrated, ΔPPL = 0.0%). Reserved as a possible negative control under a future explicit
  instruction only.

No model may be added to or dropped from the primary six after this file is locked.

## Weight extraction

- NLP models: fetch only the safetensors shard(s) holding `mlp.{gate,up,down}_proj.weight` at
  the target layer, via the existing tested loader
  `experiments/E1_nlp_validation/run_e1_retrospective.py::fetch_layer_tensors`. No full
  checkpoint download, no forward pass, no tokenizer.
- Genomic models: full weight load via `transformers.AutoModel.from_pretrained(...,
  trust_remote_code=True, torch_dtype=torch.float32)` on CPU, tensors extracted through the
  canonical adapters already in `paper-salvage/src/uk_frobenius.py`
  (`adapter_llama_swiglu` for GENERator EUK, `adapter_dnabert2`, `adapter_ntv3`), then the
  model object is discarded. No forward pass. This is the same loading pattern E1/E4 already
  used.
- All tensors promoted to **float64** immediately on load for every computation in this
  preregistration. No fp16/bf16 path is used anywhere in Gate 0 (weights only; the fp16-clamp
  and Triton-nondeterminism failure modes documented at N-013/N-007 are activation-side and do
  not apply to a weight-only assay, but float64 is used throughout regardless, matching
  `uk_frobenius.py`'s own stated rationale).

## Exact formulas

For target row *k* of a given model's target layer, with `W_gate, W_up : [d_ffn, d_model]`
and `W_down : [d_model, d_ffn]` in the canonical convention:

```
D_i = W_down[k, i]^2                                                  i = 0 .. d_ffn-1
A_i = ||W_gate[i, :]||^2 * ||W_up[i, :]||^2            (independent of k -- one A per layer)
C_i = D_i * A_i               (identical to c_{k,i} in uk_frobenius.uk_contributions)
```

Diagonal (already-implemented) row score:
```
S_diag^2 = sum_i C_i     =    uk_frobenius.uk_frobenius(...)[k]^2
```

Exact (cross-term) row score, memory-safe via the elementwise-Gram identity already used for
GENERator PROK in `experiments/E4_granularity/resolve_n009_prok_layer.py`:
```
K = (W_gate @ W_gate^T) ⊙ (W_up @ W_up^T)        # [d_ffn, d_ffn], K[i,i] = A_i exactly
S_exact^2 = W_down[k,:] @ K @ W_down[k,:]^T
```
computed for **every** row of the target layer (`S_exact` = a `[d_model]` vector,
`((D_mat @ K) * D_mat).sum(1).sqrt()` with `D_mat = W_down`), so the target row's rank under
the exact form is available alongside its value, not just its own score in isolation.

Cross-term fraction (signed; reported per model, not per-permutation):
```
f_cross = (S_exact[k]^2 - S_diag[k]^2) / S_exact[k]^2
```

## 0B — Exactness robustness check

**Device:** CUDA (A100-40GB, confirmed idle and available on this node) if
`torch.cuda.is_available()`, else CPU with the fact recorded. Either way, float64
throughout, matching `resolve_n009_prok_layer.py`'s own precedent for this exact computation.

**Memory-safety plan (checked against the largest primary-panel model, Mistral-7B,
d_ffn=14336):** `K` is `14336 × 14336` float64 ≈ 1.64 GB; the two `d_ffn × d_ffn` Gram
matmuls that build it are the dominant cost (~1.7 × 10¹² FLOPs combined) and fit comfortably
in A100-40GB. If any single primary model's `K` construction fails on the GPU (OOM) or is
impractical on CPU within a 30-minute budget, that model's exactness arm is **stopped and
reported as such** — the definition of `f_cross` is not changed, weakened, or approximated
further to force a number.

**Stop-the-study trigger (mechanical, checked immediately after 0B, before 0C/0D run on real
weights):**

Gate 0 (the entire E5 sequence) is aborted, and the finding written up as "foundation
invalid," if for **any** of the six primary-panel models:

- `|f_cross| > 0.20` for the target row (cross terms explain more than a fifth of the exact
  squared row norm — the diagonal approximation underlying 0C/0D would then be missing a
  material share of the true bilinear form), **or**
- the target row's rank under `S_exact` falls outside the top 1% of that layer's `d_model`
  rows (i.e., the row that is high-gain under the diagonal form is not still clearly
  high-gain under the exact form).

This threshold is chosen, and disclosed, **before** any real weight is passed through the
exact computation. It is not tuned after seeing results.

## 0C — Factor decomposition

For each of the three vectors `D`, `A`, `C` (each length `d_ffn`), report:

- `top1_index`, `top1_share` (= max / sum), `top5_share`, `top10_share` (cumulative,
  matching `uk_frobenius.row_granularity`'s existing definitions exactly — no new metric
  invented),
- `participation_ratio` = `(sum)^2 / sum(squares)`,
- the factor's own winner's rank/percentile within itself (trivially rank 1 / 100th
  percentile — reported for schema parity with `LayerReport`'s existing `query_ranks`
  format, not as a new finding).

Additionally, for `C`'s winning coordinate `i*` specifically (the non-trivial cross-factor
question):
- its rank under `D` alone (rank of `D[i*]` among all `D_i`),
- its rank under `A` alone (rank of `A[i*]` among all `A_i`).

No additional concentration metric is introduced after real values are inspected.

## 0D — Multiplicative-alignment permutation test

For each model, holding the observed `D` and `A` vectors fixed:

```
for π in permutations:
    C_π[i] = D[i] * A[π(i)]
    record  max(C_π),  top1_share(C_π) = max(C_π)/sum(C_π),  PR(C_π) = sum(C_π)^2 / sum(C_π^2)
```

**Permutations:** `n_perm = 20,000` per model. Chosen for percentile resolution to ≈0.005 with
headroom, and because the per-permutation cost is `O(d_ffn)` (no `d_ffn × d_ffn` matrix), so
20,000 permutations complete in well under a minute per model even at Mistral's `d_ffn`.

**Seeds:** a single master seed `42` (matching this repository's existing convention, e.g.
E1's model-sampling seed and GUE's default seed), expanded deterministically and
reproducibly via NumPy's `SeedSequence.spawn`:

```python
children = np.random.SeedSequence(42).spawn(6)
rng = np.random.default_rng(children[panel_index])   # panel_index = row # above, 0-based
```

`panel_index` is assigned by the fixed table order above (Llama=0, Mistral=1, OLMo=2,
GENERator EUK=3, DNABERT-2=4, NTv3=5) — fixed now, not chosen after seeing any result.

**Orientation** (so "higher = more concentrated / more surprising than null" is consistent
across all three metrics):

```
percentile_max        = 100 * P_π[ max(C_π)        ≤ max(C_obs) ]
percentile_top1share   = 100 * P_π[ top1_share(C_π) ≤ top1_share(C_obs) ]
percentile_PR          = 100 * P_π[ PR(C_π)         ≥ PR(C_obs) ]     # note ≥: low PR = concentrated
```

**Alignment-excess effect size** (robust z-score against the permutation null; MAD scaled by
1.4826 for approximate z-comparability, no normality assumed — only used to *rank* models,
not as a parametric threshold):

```
AE_max        = ( max_obs        − median_π(max)        ) / (1.4826 * MAD_π(max))
AE_top1share  = ( top1share_obs  − median_π(top1share)   ) / (1.4826 * MAD_π(top1share))
AE_PR         = ( median_π(PR)   − PR_obs                ) / (1.4826 * MAD_π(PR))
```

Positive `AE_*` in all three = the observed `C` is more concentrated than random `D`–`A`
pairing produces, given each model's own marginal `D` and `A` distributions. If any
`MAD_π(metric) = 0`, the corresponding `AE` is reported as `undefined (MAD=0)`, not silently
coerced to `inf`/`NaN`, and is excluded from that metric's group comparison with the reason
stated.

## GATE-0 DECISION RULE (mechanical; stated in full before any real weight is loaded)

**Primary decision metric:** `AE_top1share`, one value per model (six total). Chosen as
primary because top-1 share is the concentration statistic already used as the headline
granularity number throughout this repository's existing ledger (C-005, `CANONICAL_TABLE.md`)
and is the most directly interpretable measure of "does one scalar pathway dominate."

**Primary pass criterion — complete rank separation, exact (not asymptotic) permutation
test:** Gate 0 **passes** if and only if every one of the three NLP models' `AE_top1share`
strictly exceeds every one of the three genomic models' `AE_top1share`. For `n1 = n2 = 3`,
this is the most extreme of the `C(6,3) = 20` equally-likely rank arrangements under
exchangeability (Mann–Whitney `U = 9`/9 concordant pairs), giving an **exact** one-sided
`p = 1/20 = 0.05` — not an asymptotic normal approximation, and appropriate to the sample
size per the governing instruction to prioritize a large, interpretable separation over a
conventional asymptotic p-value.

**There is no partial-credit path.** Any overlap (≥1 NLP `AE_top1share` ≤ ≥1 genomic
`AE_top1share`) is a **FAIL**, even an 8-of-9-concordant near-miss. A near-miss is reported in
full, honestly, as a near-miss — it is not treated as a pass and not used to justify widening
the criterion after the fact.

**Secondary, reported but non-gating** (predefined now, computed and reported regardless of
the primary outcome, for interpretive richness and consistency-checking only — never
substituted for the primary criterion after seeing results):
- The same complete-separation check applied to `AE_max` and `AE_PR` — concordance or
  discordance with the primary decision is reported.
- Whether `C`'s winning coordinate is also jointly more extreme in rank under `D`-alone and
  `A`-alone in NLP than in genomic models (descriptive table from §0C).
- `f_cross` per model from §0B (also the mechanical stop trigger described there).

## FAIL condition and required response

If the primary criterion is not met: **stop the entire E5 sequence.** Write
`GATE0_RESULTS.md` with the full measurement (all metrics, all models, the exact decision
computation), state the negative result plainly, update `docs/PROJECT_STATUS.md`, commit with
explicit paths, and stop. Do not run Gate 1. Do not invent or substitute a replacement
hypothesis in the same pass.

## PASS condition

If the primary criterion is met: `GATE0_RESULTS.md` documents the pass, and Gate 1's own
preregistration and infrastructure scoping may begin in a subsequent step of this same
session, per the autonomous-progression rule in the governing instruction (no unresolved
instrumentation/provenance blocker).

## No post-hoc changes

No metric is added, and no threshold in this document is altered, after real weights from any
primary-panel model are inspected. If a real deviation from this plan becomes necessary
(e.g., an OOM on a specific model), it is handled exactly as §0B specifies (stop that model's
arm, report why) — not by editing this locked file. This file is not edited after locking; a
correction is a new, separately-locked file that discloses both.
