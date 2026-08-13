# E4 — c_{k,i} granularity decomposition

**Date:** 2026-08-12. Weights only, no forward pass.
Shape verification was mandatory per model before any number was computed.

## Results

At each model's SW row and SW layer (the coordinates in `SW_TARGETS`).

| Model | layer | row k | row rank | max/median | top-1 share | top-5 share | **PR** | of d_ffn | regime |
|---|---|---|---|---|---|---|---|---|---|
| GENERator EUK | 4 | 2371 | **1** / 3,072 | 12.2× | 0.3908 | 0.8207 | **4.56** | 8,448 | sparse |
| GENERator PROK | 2 | 1927 | **1289** / 3,072 | 6.1× | 0.0047 | 0.0153 | **2191.62** | 8,448 | distributed |
| DNABERT-2 | 5 | 603 | **1** / 768 | 5.0× | 0.3684 | 0.9613 | **3.64** | 3,072 | sparse |
| NTv3 | 11 | 1472 | **1** / 1,536 | 3.6× | 0.1754 | 0.3193 | **22.72** | 6,144 | sparse |
| Evo1 | 11 | 3776 | **48** / 4,096 | 2.8× | 0.0553 | 0.1536 | **122.27** | 10,928 | distributed |

PR = participation ratio, (Σc)² / Σc², the effective number of contributing hidden units.
"regime" is the descriptive label from `uk_frobenius.RowGranularity` — descriptive only, not
a causal claim.

For reference, the NLP models from E1 at their published coordinates:

| Model | top-1 share | PR | regime |
|---|---|---|---|
| Llama-7B | 0.8903 | 1.24 | scalar-dominated |
| Mistral-7B | 0.9884 | 1.02 | scalar-dominated |
| OLMo-7B | 0.9558 | 1.09 | scalar-dominated |

Association reported, no causal arrow: per PHASE_1_BLOCKING §E4, a low participation ratio
is **not** claimed to cause ensemble behaviour.

## Two things that need a decision, flagged not resolved

### 1. GENERator PROK ranks 1289 / 3,072, not 1

C-001 states the predictor *"ranks the empirical SW row 1/3,072 at the step-up layer in
GENERator EUK and PROK."* At the coordinates used here — PROK layer **2**, row 1927 — the
rank is **1289 / 3,072**, with top-1 share 0.0047 and PR 2191.6 out of 8,448. EUK at layer 4
does rank 1, as claimed.

The likely explanation is a **layer mismatch, not a contradiction**: `SW_TARGETS` layer 2 is
the *impulse source layer*, while C-001 refers to the *step-up layer*, which for PROK may be
a different layer. This was not investigated — resolving it means choosing which layer C-001
refers to, which is a call about an existing claim, not an E4 measurement.

**C-001 is left untouched.** It should not be migrated to the manuscript until the layer it
refers to is pinned down.

### 2. The `uk_frobenius` adapter registry is wrong for NTv3

`ADAPTERS["ntv3"] = adapter_llama_swiglu`, carrying the comment *"verify — Mistral-style,
confirm module path"*. It is not Mistral-style and the entry would raise: NTv3 has no
`model.model.layers` and no `mlp` module at all (`blk.mlp` resolves to an unrelated bound
method).

The real layout, verified by shape at L11: `SelfAttentionBlock` carries the FFN inline as
`fc1` **(12288, 1536)** and `fc2` **(1536, 6144)** with a SiLU — so `fc1` is *packed* gate+up
on adjacent row blocks (2 × 6144 = 12288), the **DNABERT-2 layout**. Corrected in
`run_e4_granularity.py`; the registry in `src/uk_frobenius.py` is **not** edited here, since
that file is shared with E1 and the fix belongs with a test.

Which half of `fc1` is gate and which is up cannot be read off the shapes — and does not
matter: c_{k,i} = W_down[k,i]²·‖W_gate[i,:]‖²·‖W_up[i,:]‖² is symmetric under swapping them,
so every number in this table is invariant to that choice.

## Shape verification, as required

| Model | gate | up | down | d_model | d_ffn | check |
|---|---|---|---|---|---|---|
| GENERator EUK | (8448, 3072) | (8448, 3072) | (3072, 8448) | 3,072 | 8,448 | OK |
| GENERator PROK | (8448, 3072) | (8448, 3072) | (3072, 8448) | 3,072 | 8,448 | OK |
| DNABERT-2 | (3072, 768) | (3072, 768) | (768, 3072) | 768 | 3,072 | OK |
| NTv3 | (6144, 1536) | (6144, 1536) | (1536, 6144) | 1,536 | 6,144 | OK (after adapter fix) |
| Evo1 | (10928, 4096) | (10928, 4096) | (4096, 10928) | 4,096 | 10,928 | OK |

Evo1's row rank of 48 / 4,096 is consistent with the 48–168 range recorded in
PHASE_1_BLOCKING §E2. NTv3's rank of 1 / 1,536 at L11 confirms C-003.

## Artifacts

`results/e4_granularity.json`, `logs/e4_evo1.log`
