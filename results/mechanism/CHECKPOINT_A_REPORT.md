# ⏸ CHECKPOINT A — Phase A results (Task Spec v2)

Session date: 2026-08-07. Phase A only. Phase B not started (gated on this review).
All outputs in `results/mechanism/`; no canonical file overwritten.

---

## 🚩 HEADLINE: A1 KILL — the mechanism does not survive generalization to k dimensions

**Task Spec A1 kill condition fired.** Generalizing retention from a single coordinate to
the full super-weight *subspace* does **not** rescue the structural/systemic decomposition.

| Model | \|S\| | `subspace_fraction_final` | random \|S\|-dim control | z | ablation effect | load-bearing? |
|---|---|---|---|---|---|---|
| GENERator EUK | 2 | **0.19039** ± 0.072 | 0.00150 | **+71.3** | +63.8% ΔPPL | yes |
| GENERator PROK (corrected L8/r260) | 2 | **0.26659** ± 0.074 | 0.01849 | **+12.1** | +58.1% ΔPPL | yes |
| **DNABERT-2** | **6** | **0.00803** ± 0.001 | 0.01478 | **−1.58** | **−25.5% Δacc** | **yes** |
| Evo1 | 10 | 0.00510 ± 0.001 | 0.00244 | +2.99 | +0.01% ΔPPL | **no** |
| NTv3 | — | not computable | — | — | −11.9 ΔMCC | yes |

Three independent ways the rule fails, all pointing the same direction:

1. **DNABERT-2's SW subspace does not beat its own random control** (0.00803 vs 0.01478 ± 0.00427,
   z = **−1.58**). It retains *no more* of the ablation perturbation than an arbitrary
   6-dimensional coordinate subspace — in fact slightly less. This is the load-bearing model.
2. **DNABERT-2 groups with the null model, not the bottleneck models.** It sits **24× below**
   GENERator and only **1.6× above** causally-null Evo1, despite a −25.5% splice accuracy collapse.
3. **The ordering inverts.** Null Evo1 separates from its random floor (z = +2.99) *better* than
   load-bearing DNABERT-2 does (z = −1.58). A metric that ranks the null model above the
   load-bearing one is not measuring load-bearing-ness.

**Robustness — the verdict does not depend on how S is defined.** Both the spec's named
coordinate sets and the full frozen-index sets were run:

| Model | spec S | frac | z | index S | frac | z |
|---|---|---|---|---|---|---|
| GENERator EUK | {2371} | 0.14772 | +227.4 | {1522, 2371} | 0.19039 | +71.3 |
| GENERator PROK | {260} | 0.36779 | +11.5 | {260, 1325} | 0.26659 | +12.1 |
| DNABERT-2 | 6 coords | 0.00803 | −1.58 | 6 coords | 0.00803 | −1.58 |
| Evo1 | {3776} | 0.00145 | +2.88 | 10 coords | 0.00510 | +2.99 |

**Correction to a premise in the task spec:** it describes Evo1's S as the single row `{3776}`,
but `results/super_weight_index.json` actually holds a **ten-row ensemble** at layer 11
(rows 156, 411, 616, 682, 843, 1636, 3292, 3582, 3776, 4033, all at col 9885). So
"DNABERT-2 is a redundant ensemble, Evo1 is a single row" is **not** the difference between
them — both are ensembles, and both disperse. That removes the main hypothesis A1 was built to
test (that DNABERT-2 only looked bad because a 1-D metric was applied to a distributed
representation). It was applied at the right dimensionality here, and still fails.

### Implication
Retention does not explain load-bearing-ness **at any dimensionality**. The structural/systemic
decomposition stays unsupported. **Paper A (rescued mechanism) is off the table.**

Deliverables: `subspace_retention_all_models.{json,csv}`, per-model
`subspace_retention_{model}.json`, scatter `paper/media/image_subspace_retention_decision.png`.

#### NTv3 — not computable (architectural, not evidence either way)
Its registered SW (L11/r1472) is at the last block of the 12-block transformer tower, but the
network's output passes through a further U-Net deconv tower whose stages change channel count
(`dim_in != dim_out`). A coordinate index has no stable identity downstream, so `span{e_k}` is
undefined and L == F makes any retention ratio tautological. Recorded as such, not as a data point.

---

## A2 — Evo1 fp64 adjudication: **REAL_BUT_DISPERSED**, not an artifact

The expected answer was "numerical artifact." The data says otherwise, and I'm reporting the
result rather than the expectation.

| Test | Result |
|---|---|
| fp64 recompute of L11 down-proj at row 3776 | fp64 = **1,247,245.4** vs bf16 = **1,245,184.0**, rel err **0.165%** |
| Is that gap saturation? | No — it is exactly bf16's grid spacing (ULP = 8192) at that magnitude. bf16 is *faithfully rounding* a genuine value. |
| Frozen at exactly 2²⁴? | **No.** Plateau = 29,360,128 = **1.75 × 2²⁴**; no block hits 2²⁴ exactly. |
| Ablation ΔPPL (bf16) | **+0.026%** — still null |

**Verdict per the spec's own rule:** fp64 gives a genuine large finite activation, ablation is
still null, and the prior rescue still failed → *"structurally generated, genuinely large, not
causally recoverable."* **Hypothesis level only** — the failed rescue means **no causal-gating
claim is licensed.**

**Where the magnitude actually comes from (new):** decomposing each block's contribution at row
3776 shows the MLP writes 1.25e6 at block 11, and then the **mixer** — not the MLP — amplifies it
~24× over the next two blocks (block 12 mixer: +14.68e6; block 13 mixer: +13.43e6; l3 contributes
≈0 in both). After block 13, nothing writes to the coordinate again.

**Two caveats I have to state plainly:**
- The "frozen at exactly 2²⁴" premise in the task spec is **not reproduced**. The earlier trace
  took a fresh argmax position per layer; this one fixes the position found at the SW layer.
- The blocks-13→31 plateau is exactly flat *as observed in bf16*, but bf16 cannot resolve
  additions below 65,536 (0.22% of the plateau). "Exactly frozen" and "drifting slightly" are
  **indistinguishable** at this precision, and Evo1 cannot run end-to-end above bf16
  (flash-attn asserts fp16/bf16). This does not change the verdict, which rests on the fp64
  *source* recomputation, not on the plateau.

**Two errors found and fixed during A2** (recorded for audit):
1. My first verdict function concluded "ARTIFACT" from `contributions < ULP/2`. That was circular
   — the tail contributions are ≈0 anyway, so being below ULP is trivially true and implies
   nothing. Rewritten to key off the fp64 fidelity test.
2. My first TEST 3 re-accumulated only `l3` outputs. The block-12→13 jump comes from the **mixer**
   path, so `l3`-only summation cannot reconstruct the residual and adjudicated nothing. Replaced
   with the contribution decomposition above.

Also noted: the pre-existing `results/sw_residual_attribution_evo1_fp32.json` is labelled
`"dtype": "float32"` but its generating script casts all non-pole/residue parameters to
**bfloat16**. It is a bf16 trace. Flagged, not modified.

Deliverable: `evo1_fp64_adjudication.json`.

---

## Where this leaves the paper (§7 decision tree)

```
A1 subspace retention ────────────► KILL  ✗ mechanism dead at every dimensionality
A2 Evo1 fp64 ─────────────────────► real-but-dispersed (hypothesis level, no causal claim)
                                     ↳ NOT the "protocol false-positive" branch
B splice/GC confound ─────────────► NOT YET RUN (gated on this review)
C contamination audit ────────────► mandatory, not yet run
```

**Paper A is eliminated.** The branch now runs through Phase B:

- **Paper B (two findings)** — available iff B2 is uniform *and* B3 persists: "SWs encode
  composition AND the ensemble is required for motif-dependent splice recognition."
- **Paper C (one finding + methodology cautions)** — the floor, fully supported today.

One thing worth flagging for the Paper C framing: the spec lists "retention ≠ load-bearing" as a
characterized failure mode, and A1 has now **quantified** it — including the inversion where the
null model outranks the load-bearing one. That is a stronger, more concrete methodological result
than the prior session had.

A2's outcome also **changes one line of the planned Paper C**: Evo1 is *not* a numerical
false-positive, so it cannot be listed as one. The three failure modes become: probe
contamination (real, PROK), **retention ≠ load-bearing** (real, quantified in A1), and
detection-magnitude ≠ causal-importance (Evo1 — a real activation that does nothing).

---

## Next steps (awaiting your go-ahead)

Per the operating rules I stopped here rather than starting Phase B. Ready to proceed with:
- **Phase B** (B1–B4 splice/GC confound on the DNABERT-2 fine-tuned splice checkpoint) — decides
  one finding vs two, i.e. Paper B vs Paper C.
- **Phase C** (contamination audit + index merge) — mandatory regardless of branch; can run in
  parallel with B since it touches different files.
