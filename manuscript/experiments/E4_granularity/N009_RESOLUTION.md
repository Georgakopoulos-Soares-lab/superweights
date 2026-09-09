# N-009 / C-001 — GENERator PROK layer provenance

**Date:** 2026-08-12
**Branch taken:** *original layer identifiable, does NOT reproduce rank 1.*
**C-001 stays on hold.** No layer search was performed. Nothing was tuned.

## Provenance chain, traced backward from the claim

| Step | Artifact | Content |
|---|---|---|
| claim | `CLAIMS_LEDGER` C-001 | "ranks the empirical SW row 1/3,072 at the step-up layer in GENERator EUK and PROK", controls "layer median, 12× / 18× ratio" |
| result | `results/sw_mechanistic_generator_prokaryote.json` | `sw_rows_by_layer = {"2": [1927]}`; `frob_norm_uk_by_layer["2"][1927] = 2648.4773`; median 147.9776; **rank 1/3072**; **max/median 17.90** |
| log | `logs/generator-mechanistic-prok.3193910.out` | `[uk_audit] loading GenerTeam/GENERator-v2-prokaryote-3b-base on CPU`, run 29 May 2026 |
| script | `scripts/analysis/run_sw_mechanistic.py :: experiment2_frob_uk` | exact ‖U_k‖_F = ‖W_gateᵀ diag(W_down[k,:]) W_up‖_F |
| config | `configs/generator_prokaryote.yaml` | `GenerTeam/GENERator-v2-prokaryote-3b-base` |

**The claim's layer is 2**, row 1927 — the same layer E4 used. The layer-mismatch hypothesis
recorded in N-009 is **wrong**. The checkpoint *name* in the original log is identical to the
one in the current config.

The "18×" in C-001's control column is this file's 17.90; the "12×" is EUK's 12.24. Both
trace cleanly, so the claim's provenance is not in doubt — only its reproducibility.

## Recomputation at that exact layer (layer 2, row 1927)

Shapes verified before computing: gate (8448, 3072), up (8448, 3072), down (3072, 8448),
d_model 3072, d_ffn 8448. No transpose applied.

| Quantity | value | rank | percentile | median | max/median |
|---|---|---|---|---|---|
| **stored** (2026-05-29 artifact) | **2648.4773** | **1** / 3072 | 100.000 | 147.9776 | **17.90** |
| **A. exact** ‖U_k‖_F, current weights | 5.5106 | **1277** / 3072 | 58.46 | 5.4915 | 7.18 |
| **B. decomposed** √(Σᵢ c_{k,i}), current weights | 5.4710 | **1289** / 3072 | 58.07 | 5.4529 | 6.13 |

A is the same formula the original script used; B is `src/uk_frobenius.py`. **A and B agree
with each other** (rank 1277 vs 1289) and **both disagree with the stored artifact**. The
discrepancy is therefore not a formula difference between the two implementations.

Scale: stored value is 480.6× the exact recomputation; stored median is 26.9× the
recomputed median. The ratio is not uniform, which is why the rank moves.

### EUK reproduces; PROK does not

| Model | layer | row | stored max/median | recomputed max/median | stored rank | recomputed rank |
|---|---|---|---|---|---|---|
| GENERator EUK | 4 | 2371 | 12.24 | **12.24** | 1 | **1** |
| GENERator PROK | 2 | 1927 | 17.90 | 6.13 | 1 | **1289** |

EUK reproduces to four significant figures on the ratio and reproduces rank 1. PROK does
not. Whatever the cause, it is not global to the mechanistic pipeline.

## c_{k,i} decomposition at layer 2, row 1927

| top-1 index | top-1 share | PR | of d_ffn | regime |
|---|---|---|---|---|
| 7028 | 0.004741 | 2191.62 | 8,448 | distributed |

Cumulative contribution share:

| top-1 | top-5 | top-10 | top-50 | top-100 |
|---|---|---|---|---|
| 0.0047 | 0.0153 | 0.0251 | 0.0799 | 0.1286 |

## Comparison with the overnight layer-2 E4 result

Identical: E4 reported rank 1289/3072, max/median 6.13, top-1 share 0.0047, PR 2191.62.
The overnight number is confirmed, on the provenance-resolved layer. **Why it differs from
the stored artifact is not interpreted here.**

## Status and what was deliberately not done

- **C-001 remains on hold.** Not restored, not retired.
- The overnight layer-2 E4 PROK result is **not** replaced — it is now known to be at the
  claim's own layer.
- The stored 2026-05-29 artifact is **not** deleted or altered.
- No search over other layers for one that yields rank 1.
- No adjustment to any implementation to close the gap.

## Open, requires a decision

The stored PROK artifact is not reproducible from the current checkpoint of the same name at
the same layer, while EUK is. Distinguishing the possibilities — an upstream change to the
HF repo since 29 May 2026, a difference in how weights were loaded/dtyped in the original
run, or something else — needs a call that is outside a mechanical provenance trace.

Until that is settled, C-001's PROK half has no reproducible support.
