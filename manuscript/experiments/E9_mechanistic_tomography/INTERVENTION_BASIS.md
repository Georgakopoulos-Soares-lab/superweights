# INTERVENTION_BASIS.md — E9 mechanistic tomography

Locks the intervention ontology before any mask is designed. Written from
`PROVENANCE_AND_BASELINES.md`; nothing here has looked at an E9 response value.

---

## Shared intervention semantics (both models)

| Question | Answer |
|---|---|
| What is scaled? | A single down_proj output **row** (`mod.weight.data[row, :]`) — the vector the MLP write deposits into the residual stream at one coordinate of the pre-nonlinearity... no: the row *is* the linear map from the (post-activation) intermediate to one downstream coordinate. Concretely: `mod` is the down-projection `Linear`; `weight.data[row, :]` is the row of weights producing residual-stream write component `row`. |
| When in the forward pass? | At the down-projection matmul itself. |
| Before/after nonlinear activation? | **After** — down_proj consumes the already-activated (SwiGLU for GENERator, GLU for DNABERT-2) intermediate. |
| Before/after down projection? | This *is* the down projection — scaling the row scales that one output coordinate's contribution before summation with the other coordinates' contributions (row-scaling changes one output element of the linear map, not the whole matmul). |
| Before/after residual addition? | **Before** — the scaled/zeroed write is what gets added to the residual stream. |
| Before/after LayerNorm/RMSNorm? | **Before** — GENERator's next RMSNorm and DNABERT-2's encoder LayerNorm both see the modified residual stream downstream of this point. |
| Does scaling affect one output coordinate or a broader pathway? | One row = one output coordinate of the down_proj map = one residual-stream channel's contribution from this layer's MLP. It is *not* the whole MLP output; it is exactly the coordinate the super-weight/high-gain detection identified. |

**Parameterization (frozen, per `next_prompt.md` Phase 2):**

```
alpha_i = 1 - epsilon * a_i
mod.weight.data[row_i, :] = saved_row_i * alpha_i
```

`a_i in {0,1}` per the mask convention (Phase 3); `epsilon in {0.5, 1.0}` (frozen scales).
`alpha_i = 1.0` (epsilon=0 or a_i=0) is the untouched natural state. `alpha_i = 0.0`
(epsilon=1.0, a_i=1) is full ablation/removal — identical to the existing `_zero_row`
convention already used throughout the repo. `alpha_i = 0.5` (epsilon=0.5, a_i=1) is partial
suppression — **new**, not previously implemented by any existing script; implemented as a
single generalized helper, `_scale_row(model, pattern, layer, row, alpha)`, that reuses the
existing `_save_row`/`_restore_row` (`scripts/evaluation/run_gue_ablation.py:470-484`) rather
than introducing a new intervention mechanism. This is the same wire every prior mechanism
script already perturbs — only the coefficient granularity is new.

No gradient/HVP method is used as the primary instrument (Phase 2 rule). All measurements
below are finite forward interventions.

---

## GENERator EUK

- Tensor: `model.layers.4.mlp.down_proj.weight.data[row, :]`, `row ∈ {2371, 1522}`.
- Semantics: as above.
- Endpoint hook point: none needed beyond the row scale itself — GC fraction is read off the
  final generated string, not an intermediate activation.
- Scale ↔ epsilon mapping: `scale = alpha = 1 - epsilon*a_i`. Both `epsilon=0.5` (`scale=0.5`)
  and `epsilon=1.0` (`scale=0.0`) are already inside the previously-tested scale grids
  `{0.0,0.5,1.0,2.0,5.0}` and `{0.0,0.25,0.5,1.0,2.0}` — no new scale value is introduced,
  only the framing (`alpha` vs. raw `scale`) changes, and that framing is identical since
  `run_sw_steering.py` already treats `scale` as a raw multiplicative row factor.

## DNABERT-2

- Tensor: `bert.encoder.layer.9.mlp.wo.weight.data[row, :]`, `row ∈` the 10-row ensemble
  (see `PROVENANCE_AND_BASELINES.md`), only 2 of which (264, 294) are at layer 9 — the other
  8 basis rows live at layers 3/5/6/7 per the table above. **Each basis component's `(layer,
  row)` pair is intervened at its own layer**, not forced to layer 9; "layer 9" only
  describes the critical pair and the residual-norm secondary endpoint's hook location.
- Endpoint hook point (secondary, residual norm): explicit forward hook on
  `bert.encoder.layer[9]` output (standardized convention, see
  `PROVENANCE_AND_BASELINES.md`), reading `h[:, channel].abs().mean()` — used only for the
  secondary residual-norm-vs-functional-damage analysis (Phase 7, H6), not for fitting the
  primary observer families.
- Primary endpoint: pretrained MLM loss, computed on the same fixed masked-token batches
  regardless of which rows are ablated (`_build_fixed_batches`, seed 42, `mask_prob=0.15`,
  `n_windows=256`, `win_bp=600`).

---

## Basis-dependence acknowledgement (required by Phase 11, stated here early)

Both models' intervention coordinates are **down_proj output rows in the residual-channel
basis already used by prior detection/ablation work** — not a rotated, learned, or
PCA/SAE-derived basis. Any interaction-order finding from E9 is a statement about this
declared basis, not an intrinsic property of the underlying computation. This is repeated in
full in `RESULTS.md`'s "Basis dependence" subsection once results exist.

---

## Open ontological ambiguity check (Phase 0 gate)

Per instruction: "Do not proceed while the intervention ontology is ambiguous." The above
resolves every item on the required checklist for both models identically (same wire, same
point in forward pass, same before/after-activation/residual/LayerNorm answers) except the
per-component layer index, which is explicit per basis row above. **No ambiguity remains.**
Proceeding to baseline regression.
