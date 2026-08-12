# E2 — STEP 2 gate: Evo1 input-sensitivity precondition

**Date:** 2026-08-12
**Prereg:** `docs/prereg/PREREG_evo1_broadcast.md` v2, §Precondition
**Verdict: PASSED.** An AC component exists. The amended protocol may proceed.

## Why the gate existed

The Evo1 residual had been reported "bit-for-bit identical from layer 13 onward". If that
held of the **hidden state** rather than of a summary statistic, Evo1's output would be
input-independent from L13 — incompatible with its reported PPL of 3.11, and fatal to every
measurement taken through this trace, including the residual-attribution numbers already in
the manuscript.

## Method

Three 504-nt probes, equal length so the diff is coordinate-aligned end to end:

| Probe | Source | Role |
|---|---|---|
| `actb_500` | human ACTB CDS | the probe the assay itself uses |
| `poly_a` | homopolymer | maximally different composition |
| `pseudomonadota_504` | `pseudomonadota[:504]` | unrelated *real* sequence |

Loaded through the **identical** path the assay uses — bf16 for every parameter except
`poles`/`residues`, forward under `torch.autocast("cuda", bfloat16)` — so the gate tests
the same numerical trace the assay runs on, not a cleaner one.

Script: `scripts/interpretability/evo1_input_sensitivity_gate.py`
Environment: `evo2.sif`, sha256 `ecb0011…61706` (see `docs/ENVIRONMENT.md`)

## Primary result — layer-20 hidden state, input vs input

| Pair | max \|Δ\| | mean \|Δ\| | median \|Δ\| | coords differing | bitwise identical |
|---|---|---|---|---|---|
| `actb_500` vs `poly_a` | **4.446e8** | **1.086e7** | 5.145e6 | **2,057,017 / 2,064,384 = 99.643%** | **False** |
| `actb_500` vs `pseudomonadota_504` | 4.614e8 | 3.840e6 | 1.311e6 | 2,044,160 / 2,064,384 = 99.020% | False |
| `poly_a` vs `pseudomonadota_504` | 5.243e8 | 1.067e7 | 5.235e6 | 2,060,688 / 2,064,384 = 99.821% | False |

At the SW coordinate specifically (row 3776): max |Δ| = 2.22e7 / 3.52e7 / 3.66e7 across the
three pairs, differing at 98.6–99.8% of token positions.

Scale check: max |Δ| is 73–109× the median |h| at that layer, so the difference is large
relative to the values themselves, not a rounding artifact.

**The layer-20 hidden state is strongly input-dependent. The gate passes.**

Input-dependence is present at every layer, not only L20 (`actb_500` vs `poly_a`):
99.7% of coordinates differ at L11, 99.6% at every layer from L13 to L31.

## Corrected finding — what "bit-for-bit identical from L13" actually was

**It was a property of the layer-to-layer max |Δ| statistic, not of the hidden state.**

The statistic that produced the original claim saturates: from L13 onward it is dominated by
a handful of very large coordinates that stop moving, so it reports the same value at every
layer while the rest of the state continues to change. The gate's own secondary table
reproduced the illusion — L13-vs-L20, L13-vs-L25 and L13-vs-L31 return byte-identical diff
statistics — which is what prompted the direct check.

Direct check (`scripts/interpretability/evo1_layer_freeze_check.py`, consecutive-layer and
every-layer-vs-final): **no layer is bitwise identical to any other.** `first layer
bit-identical to final = 31` (i.e. only L31 itself); `blocks whose output does not register
in the residual = 0`.

There is no freeze. Nothing in the manuscript that depends on the trace is invalidated by it.

## Ledger note — relative-change collapse (Phase 1b candidate, NOT to be chased now)

What is real, and is a different claim from "frozen", is a collapse in **relative** change
after the L12→L13 magnitude explosion (max |h| → 1.29e9):

| Transition | max \|Δ\| | mean \|Δ\| | frac coords ≠ 0 |
|---|---|---|---|
| L9→L10 | 9.366e2 | 6.380e1 | 1.0000 |
| L10→L11 | 1.253e6 | 2.870e5 | 1.0000 |
| L11→L12 | 2.232e7 | 1.800e6 | 0.9995 |
| L12→L13 | **1.289e9** | 1.726e7 | 0.9996 |
| L13→L14 | 6.554e4 | 5.058e2 | 0.1522 |
| L17→L18 | 2.621e5 | 4.583e3 | 0.3844 |
| L21→L22 | 2.000e0 | 1.729e-5 | 0.0000 |
| L25→L26 | 1.000e0 | 1.476e-6 | 0.0000 |
| L30→L31 | 3.125e-2 | 2.081e-8 | 0.0000 |

Against a residual of median |h| ≈ 4.23e6, consecutive-layer change falls to **~1e-6 of
residual magnitude**, and **blocks 21–31 alter < 0.2% of coordinates**.

This is logged as a **Phase 1b candidate observation**. It is not chased here, it is not
interpreted here, and it is not the AC/DC decomposition Branch C would require. The Phase 1
stop rule is binding.

## Secondary — AC/DC magnitudes (input to the STEP 3 ε formula)

Not a decomposition; just the quantity `ε_m = α · std(h − mean(h))` consumes.

| Layer | median \|h\| | max \|h\| | mean h (DC) | std AC | ε at α=0.01 |
|---|---|---|---|---|---|
| 11 (injection) | 2.826e5 | 1.253e6 | 6.715e1 | 3.326e5 | **3.326e3** |
| 12 | 1.384e6 | 2.241e7 | 9.138e3 | 2.859e6 | 2.859e4 |
| 13 | 4.227e6 | 1.292e9 | −1.904e6 | 5.531e7 | 5.531e5 |
| 20 | 4.227e6 | 1.292e9 | −1.904e6 | 5.531e7 | 5.531e5 |
| 31 | 4.227e6 | 1.292e9 | −1.904e6 | 5.531e7 | 5.531e5 |

These numbers are what triggered **D-013**: under bf16 the injected ε ≈ 3.3e3 sits ~10×
*below* the representable increment (≈3.3e4) downstream of L13, so the run would have been
unable to distinguish a flat result from T = 0. The re-run is therefore fp32, with the
headroom column as the explicit check.

## Artifacts

| File | Contents |
|---|---|
| `results/evo1_input_sensitivity_gate.json` | full per-pair, per-layer and DC/AC tables |
| `results/evo1_layer_freeze_check.json` | consecutive-layer and vs-final diffs |
| `logs/evo1_gate.log` | gate stdout |
| `logs/evo1_layer_freeze.log` | freeze-check stdout |

(Paths are repo-level, i.e. the old tree. Migration into `results/keep/` with `PROVENANCE.md`
follows the E2 re-run — see `docs/PROJECT_STATUS.md` §Open gaps.)
