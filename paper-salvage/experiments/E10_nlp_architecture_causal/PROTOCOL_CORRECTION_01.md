# E10 Protocol Correction 01 — intervention bases were selected by `q1`, not by high-gain magnitude

**Status: pre-measurement correction. No causal measurement had been run when this was found;
none was run before it was fixed.** Raised by the author on 2026-08-23, before ARM A/B launch.

## 1. The defect

E10's Phase 1 and Phase 2 basis construction ranked and selected intervention rows by **`q1`**:

```python
# rank_encoder_rows.py:79  (as committed in a81edd5)
all_rows.sort(key=lambda r: r["q1"], reverse=True)
```

`q1 = sigma_1^2 / sum_j sigma_j^2` measures the **internal spectral concentration
(rank-1-ness)** of an operator `U_k`. It is scale-invariant: multiplying `U_k` by any constant
leaves `q1` unchanged. It therefore carries **no information about operator magnitude** and is
not a high-gain / super-weight importance score. It is an *annotation* of geometry, and E7/E8
only ever used it as one — the rows those experiments selected were found by an **activation
spike detector** (`out_max` / layer-median ratio ≥ 5.0x), with `q1` computed *afterwards* on
the already-selected row.

E10 inverted that order, using the annotation as the selector. This was a genuine protocol
error, not sloppy wording in a summary.

## 2. The correct criterion

**Exact ‖U_k‖_F** — the operator's Frobenius magnitude — ranked **relative to its own layer's
median**, which is the convention `paper-salvage/src/uk_frobenius.py::layer_report` already
implements (`max_over_median`, per-layer `query_ranks`) and the form E7/E8's own detector
threshold takes (ratio to layer median).

Two definitions of ‖U_k‖_F exist in this repository and they are **not** the same object:

| | Formula | Cross terms | Where |
|---|---|---|---|
| Diagonal approximation | `sqrt(sum_i W_down[k,i]^2 ||W_gate[i,:]||^2 ||W_up[i,:]||^2)` | **dropped** | `uk_frobenius.py::uk_frobenius` (E1/E5) |
| **Exact** | `sqrt(sum_j sigma_j^2)` of `U_k = (d .* W_gate)^T @ W_up` | retained | `spectral_lib.py::SpectralMetrics.frob_norm` (E7/E8) |

The diagonal form is precisely the object the E5→E8 arc moved away from. This correction uses
the **exact** one throughout, per the author's instruction ("exact ‖U_k‖_F / operator
magnitude"). For OLMo L1/r269 the two differ (diagonal 0.8679 in `results/e1_nlp_retrospective.json`
vs. exact 0.9111) — a small gap here, but they are different quantities and are not
interchangeable.

### Computing exact ‖U_k‖_F for every row without SVD

Ranking all rows of a 7B decoder layer by exact ‖U_k‖_F naively needs one full SVD of a
`[d_model, d_model]` matrix per row (4096 SVDs of 4096x4096 per layer — infeasible). The exact
norm has a closed form that avoids SVD entirely:

```
||U_k||_F^2 = sum_{i,j} d_i d_j (G G^T)[i,j] (U U^T)[i,j] = d^T [ (G G^T) .* (U U^T) ] d
```

so with `K = (G G^T) .* (U U^T)` computed **once per layer**,
`||U_k||_F^2` for all `k` is `rowwise_sum((W_down @ K) * W_down)` — two matmuls per layer.
Implemented in `exact_uk_norm_all_rows.py`, and **validated against
`spectral_lib.row_spectral_metrics`'s SVD-based value to ~6e-13 relative error** on real rows
of both OLMo and Phi-3 (assertion in the script, printed in the run log). This is an exact
identity, not an approximation.

## 3. Per-model audit — what was used, and what the correct criterion gives

### Selection scalar actually used, per model

| Model | K | Scalar used for **selection** | `q1` role | Selection error? |
|---|---:|---|---|---|
| Llama | 1 | none — single published Yu et al. coordinate; no ranking performed | annotation only | **No** |
| Mistral | 1 | none — single published coordinate | annotation only | **No** |
| Qwen2.5 | 1 | none — single row from E7's own **activation detector** (ratio 639.2x) | annotation only | **No** |
| OLMo | 4 | **`q1`** (invalid) | used as selector | **Yes** — reordered |
| Phi-3 | 6 | **`q1`** (invalid) | used as selector | **Yes** — reordered |
| MosaicBERT | 10 | **`q1`** (invalid) | used as selector | **Yes** — set changed |
| ModernBERT | 10 | **`q1`** (invalid) | used as selector | **Yes** — set changed |

The three `K=1` models are unaffected: with one pre-existing candidate there is nothing to
rank, and that candidate came from Yu et al.'s published table or from E7's activation
detector — never from `q1`. Their frozen rows stand unchanged.

### OLMo — why L24 was promoted over canonical L1, and whether it survives

**The reason given in the original freeze was `q1` = 0.9990 (L24) vs 0.9646 (L1). That reason
is invalid** and is withdrawn. Under the correct exact-‖U_k‖_F criterion
(`results/e10_exact_uknorm_olmo.json`):

| Rank | Layer/Row | exact ‖U_k‖_F | x layer median | within-layer rank | `q1` (annotation) |
|---:|---|---:|---:|---:|---:|
| 1 | L24/r269 | 1.6369 | **53.4x** | #0 of 4096 | 0.9990 |
| 2 | L7/r269 | 1.1681 | 39.7x | #0 of 4096 | 0.9521 |
| 3 | L1/r269 **(canonical)** | 0.9111 | 39.2x | #0 of 4096 | 0.9646 |
| 4 | L2/r269 | 0.5979 | 20.8x | #0 of 4096 | 0.9611 |

**L24 does outrank L1 under the correct magnitude criterion** (53.4x vs 39.2x layer median;
1.637 vs 0.911 raw) — so the ordering survives, but **on entirely different and now valid
grounds**. The original justification was wrong even though the resulting order happens to
match. Note also that **L7 moves above L1**, which the `q1` ordering had placed last — so
OLMo's ordering does change (L24, L7, L1, L2 instead of L24, L1, L2, L7).

All four rows are the **#0 (top) row of their own layer** by exact ‖U_k‖_F, so all four are
genuinely high-gain and the set itself was never in question. L1 retains its status as this
project's designated canonical row (E1/E5/E7 primary) — recorded, but no longer used to
determine rank order.

### Phi-3 — top-1 unchanged, ranks 2-6 materially reordered

`results/e10_exact_uknorm_phi3.json`:

| Rank (corrected) | Layer/Row | exact ‖U_k‖_F | x layer median | within-layer rank | `q1` | rank under `q1` (old) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | L2/r525 | 120.77 | 7.8x | #0 | 0.9529 | 1 |
| 2 | L2/r1693 | 105.86 | 6.8x | #1 | 0.9137 | 3 |
| 3 | L2/r1113 | 97.52 | 6.3x | #2 | 0.7505 | 5 |
| 4 | L4/r525 | 72.12 | 4.4x | #0 | 0.9278 | 2 |
| 5 | L4/r1113 | 61.70 | 3.8x | #1 | 0.5584 | **6 (last)** |
| 6 | L4/r1693 | 46.49 | 2.8x | #2 | 0.8919 | 4 |

L4/r1113 was ranked *last* by `q1` (0.5584) but is the 5th-largest operator; L4/r1693 was 4th
by `q1` but is the *smallest* of the six. Top-1 is unchanged, so Phi-3's control layer
(layer 2) is unchanged.

### MosaicBERT — 5 of 10 basis rows were wrong

`results/e10_selection_audit_mosaicbert.json`. Corrected basis (exact ‖U_k‖_F / layer median):

| Rank | Layer/Row | x layer median | exact ‖U_k‖_F | `q1` (annotation) | in old `q1` basis? |
|---:|---|---:|---:|---:|---|
| 1 | L0/r287 | 12.29x | 39.198 | 0.6611 | yes |
| 2 | L0/r444 | 6.61x | 21.086 | 0.4298 | yes |
| 3 | L1/r287 | 6.45x | 23.109 | 0.0763 | **no** |
| 4 | L11/r287 | 5.91x | 28.900 | 0.2642 | **no** |
| 5 | L0/r302 | 5.84x | 18.623 | 0.3369 | **no** |
| 6 | L5/r287 | 5.76x | 23.147 | 0.1275 | **no** |
| 7 | L3/r287 | 5.67x | 22.427 | 0.1466 | **no** |
| 8 | L10/r666 | 5.27x | 22.375 | 0.8886 | yes |
| 9 | **L9/r287 (E8 canonical)** | **5.05x** | 19.998 | 0.4766 | yes |
| 10 | L10/r79 | 4.81x | 20.424 | 0.8121 | yes |

Only **5/10 overlap** with the `q1` basis. The `q1` basis admitted rows that are near-rank-1
but weak — L5/r79 (‖U_k‖_F = **3.45**), L1/r79 (**4.61**), L0/r416 (**7.67**), L10/r333
(**9.90**) — while excluding genuinely high-gain rows with unremarkable geometry: L11/r287
(28.90), L5/r287 (23.15), L1/r287 (23.11), L3/r287 (22.43), L0/r302 (18.62). L5/r79 at
‖U_k‖_F = 3.45 is **more than 11x weaker** than the top row it was seated alongside. Those are
ordinary rows, and intervening on them would have measured mostly nothing.

**Canonical row check:** L9/r287 is rank **9/10** under the corrected criterion (it was 7/10
under `q1`) — still inside the basis, so no STOP/audit condition is triggered. Within its own
layer it is the **#0 row at 5.05x the layer median**, i.e. genuinely the top high-gain row of
layer 9, consistent with why E8's activation detector found it there.

### ModernBERT — 4 of 10 basis rows were wrong, and the canonical row is *better* validated

`results/e10_selection_audit_modernbert.json`. Corrected basis:

| Rank | Layer/Row | x layer median | exact ‖U_k‖_F | `q1` (annotation) | in old `q1` basis? |
|---:|---|---:|---:|---:|---|
| 1 | **L15/r251 (E8 canonical)** | **23.55x** | **836.05** | 0.8970 | yes (rank 2) |
| 2 | L0/r251 | 8.52x | 278.32 | 0.6342 | **no** |
| 3 | L11/r251 | 8.03x | 321.67 | 0.7273 | yes |
| 4 | L0/r67 | 7.52x | 245.71 | 0.6548 | yes |
| 5 | L9/r251 | 7.52x | 304.09 | 0.6928 | yes |
| 6 | L4/r251 | 5.96x | 226.15 | 0.3420 | **no** |
| 7 | L1/r251 | 5.08x | 184.83 | 0.2626 | **no** |
| 8 | L5/r251 | 4.86x | 187.41 | 0.0725 | **no** |
| 9 | L10/r251 | 4.80x | 192.40 | 0.0515 | **no** |
| 10 | L1/r67 | 4.58x | 166.74 | 0.2478 | **no** |

Only **4/10 overlap**. Critically, the `q1` basis ranked **L11/r254 first** (`q1` = 0.9127)
— a row whose exact ‖U_k‖_F is **124.0**, nearly **7x smaller** than the canonical row's
836.05. It displaced the true top row on a scale-invariant statistic.

**Canonical row check:** under the corrected criterion L15/r251 is rank **1 of 16,896** — the
single largest-magnitude `U_k` operator in the entire model, at 23.5x its layer median (the
next-largest row anywhere is 321.67, a 2.6x gap). This is a **strong independent
cross-validation**: E8's forward-pass activation detector and the weight-space exact operator
magnitude, two entirely independent measurements, agree that L15/r251 is ModernBERT's top
high-gain row. Under `q1` they *disagreed*. The corrected criterion is the one that reproduces
the detector.

## 4. Robustness of the corrected criterion

Ranking by **global** exact ‖U_k‖_F (raw magnitude, no layer normalization) instead of by
layer-relative ratio gives:

- **MosaicBERT: the identical 10-row set** (10/10), only reordered.
- **ModernBERT: 9/10** — global includes L6/r251, layer-relative includes L1/r67 instead.

The layer-relative form is used as primary because it is the convention
`uk_frobenius.layer_report` already implements and the form E7/E8's detector threshold takes;
the global ordering is recorded in the same artifacts as a robustness check. The choice does
not materially change either basis.

## 5. Disclosed caveat — no weight-space eligibility threshold exists

`e10_prompt.md` Phase 2 says to use "all eligible rows" if fewer than 10 meet "the project's
existing high-gain eligibility criterion." The project's only numeric eligibility threshold is
E7/E8's **activation-detector** ratio (`out_max` / layer median ≥ 5.0x). That is an
*activation-space* threshold; no weight-space ‖U_k‖_F eligibility threshold has ever been
defined here, and importing the 5.0x number to a weight-space ratio would be **inventing a new
structural criterion**, which E10's hard rules forbid.

`K = 10` is therefore retained for both encoders. Disclosed for transparency, and **not acted
on**: were the 5.0x figure imported anyway, MosaicBERT would have **9** eligible rows (rank 10
sits at 4.81x) and ModernBERT **7** (ranks 8-10 sit at 4.86x/4.80x/4.58x). This is recorded so
the boundary rows are visible, not hidden behind a round number.

## 6. What changed and what did not

| Item | Changed? |
|---|---|
| Llama / Mistral / Qwen2.5 frozen rows | **No** — K=1, never ranked by `q1` |
| OLMo row **set** | No (same 4 published rows) |
| OLMo row **order** | **Yes** — L24, L7, L1, L2 (was L24, L1, L2, L7) |
| Phi-3 row set | No (same 6 published rows) |
| Phi-3 row order | **Yes** — ranks 2-6 reordered |
| MosaicBERT basis | **Yes** — 5 of 10 rows replaced |
| ModernBERT basis | **Yes** — 6 of 10 rows replaced |
| All 5 decoders' top-1 row | **No** — unchanged in every case |
| All 5 decoders' control rows | **No** — control layer is set by the top-1 row's layer, and no top-1 changed |
| Endpoints, corpora, masking, alpha/epsilon, statistics, D4 thresholds | **No** |

Because no top-1 row moved, `results/e10_decoder_control_rows.json` stands unchanged and did
not need reseeding.

## 7. Prereg handling

`PREREG_E10_nlp_architecture_causal.md` (locked `b946f690…`, 2026-08-23T00:20:27Z) named the
`q1`-selected bases and is therefore **superseded**. Per the author's instruction and this
project's own convention (`prereg_lock.py` docstring: "If you must revise, lock again and
disclose both entries"):

- The v1 file is left **byte-identical** so its existing lock continues to verify `OK` — the
  integrity chain is preserved rather than rewritten.
- `PREREG_E10_nlp_architecture_causal_v2.md` supersedes it, carries the corrected bases, and
  is locked separately. Both entries remain in `LOCKS.jsonl`.
- No causal response was measured under v1. This is a correction to an unused design, not a
  post-hoc revision after seeing data.

## 8. Verdict

The defect was real, material for 4 of 7 models, and is now corrected. Both encoder canonical
rows survive inside their corrected bases (MosaicBERT rank 9/10, ModernBERT rank 1/10), so no
STOP/audit condition fires and E10 remains launchable — under the corrected v2 prereg only.
