# E4 — canonical granularity table

**Date:** 2026-08-13
**Measurement and provenance only.** No conclusions are drawn here about how the two groups
compare; the table exists so those numbers are available to be interpreted elsewhere.

Built by `build_canonical_table.py` **from already-generated artifacts only** — no model was
loaded and nothing was re-run. Cells the stored artifacts do not contain are printed `n/s`
(not stored) rather than recomputed, so no number in this table can have come from a fresh
forward pass.

## Sources

| Rows | Artifact |
|---|---|
| Llama-7B, Mistral-7B, OLMo-7B | `results/e1_nlp_retrospective.json` |
| GENERator PROK @ L2 | `results/n009_prok_layer_resolution.json` |
| NTv3 (corrected shared adapter) | `results/e4_ntv3_shared_adapter.json` |
| GENERator EUK, DNABERT-2, Evo1 | `results/e4_granularity.json` |


### Published NLP super-weights

| Model | checkpoint | layer | published (k,i) | row rank | row percentile | max/median | scalar rank of i | top1 index | top1 share | PR | cum top1 | top5 | top10 | top50 | top100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-7B | `huggyllama/llama-7b` | 2 | (3968, 7003) | **1** / 4,096 | 100.000 | 29.40× | **1** / 11,008 | 7003 | 0.8903 | 1.24 | 0.8903 | 0.9987 | n/s | n/s | n/s |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | 1 | (2070, 7310) | **1** / 4,096 | 100.000 | 26.83× | **1** / 14,336 | 7310 | 0.9884 | 1.02 | 0.9884 | 0.9941 | n/s | n/s | n/s |
| OLMo-7B | `allenai/OLMo-7B-0724-hf` | 1 | (269, 7467) | **1** / 4,096 | 100.000 | 37.79× | **1** / 11,008 | 7467 | 0.9558 | 1.09 | 0.9558 | 0.9688 | n/s | n/s | n/s |

### Genomic models

| Model | checkpoint | layer | published (k,i) | row rank | row percentile | max/median | scalar rank of i | top1 index | top1 share | PR | cum top1 | top5 | top10 | top50 | top100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GENERator EUK | `GenerTeam/GENERator-v2-eukaryote-3b-base` | 4 | (row 2371, i n/a) | **1** / 3,072 | 100.000 | 12.24× | n/s | 2536 | 0.3908 | 4.56 | 0.3908 | 0.8207 | 0.8683 | n/s | n/s |
| GENERator PROK **[CONTESTED]** | `GenerTeam/GENERator-v2-prokaryote-3b-base` | 2 | (row 1927, i n/a) | **1289** / 3,072 | 58.073 | 6.13× | n/s | 7028 | 0.0047 | 2191.62 | 0.0047 | 0.0153 | 0.0251 | 0.0799 | 0.1286 |
| DNABERT-2 | `zhihan1996/DNABERT-2-117M @ 7bce263` | 5 | (row 603, i n/a) | **1** / 768 | 100.000 | 5.02× | n/s | 1062 | 0.3684 | 3.64 | 0.3684 | 0.9613 | 0.9911 | n/s | n/s |
| NTv3 | `InstaDeepAI/NTv3_650M_pre @ 0ecff36` | 11 | (row 1472, i n/a) | **1** / 1,536 | 100.000 | 3.61× | n/s | 1713 | 0.1754 | 22.72 | 0.1754 | 0.3193 | 0.3607 | 0.4714 | 0.5158 |
| Evo1 | `togethercomputer/evo-1-8k-base @ 1.1_fix` | 11 | (row 3776, i n/a) | **48** / 4,096 | 98.853 | 2.78× | n/s | 9885 | 0.0553 | 122.27 | 0.0553 | 0.1536 | 0.1888 | n/s | n/s |

Notes:
- **Llama-7B**: published (k,i) recovered at rank 1 on both levels
- **Mistral-7B**: published (k,i) recovered at rank 1 on both levels
- **OLMo-7B**: published (k,i) recovered at rank 1 on both levels
- **GENERator PROK **[CONTESTED]****: **C-001 ON HOLD — N-009 unresolved.** Layer 2 is the claim's own layer, verified by provenance. Does NOT support the manuscript's rank-1 claim.
- **NTv3**: corrected shared adapter (N-010); reproduces the pre-fix value exactly

## Column availability

- **`scalar rank of i`** is defined only for the NLP models: they have a *published* scalar
  index to rank. The genomic models have an empirically identified SW **row**, with no
  published `i`, so the column is `n/s` for them by definition rather than by omission.
- **Cumulative top-50 / top-100** were stored only for GENERator PROK @ L2 and NTv3, whose
  artifacts were written with the full cumulative profile. For the other six rows the stored
  artifacts keep only the top-10 contributor list, so top-1/5/10 are recoverable and
  top-50/100 are not.
- **Cumulative top-10** is `n/s` for the three NLP rows: `e1_nlp_retrospective.json` stores
  `top1_share` and `top5_share` but not the contributor list.

Filling the `n/s` cells requires re-deriving `c_{k,i}` from weights — a model load. That is
outside "use only already-generated results" and was **not** done.

## GENERator PROK is contested, and is presented as contested

The PROK row is the **layer-2** measurement, which is the layer C-001 itself refers to,
established by tracing claim → result → log → script → config (N-009,
`N009_RESOLUTION.md`). It is included because it is verified, and marked because it conflicts
with the stored artifact:

| | value | rank | max/median |
|---|---|---|---|
| stored 2026-05-29 artifact | 2648.4773 | 1 / 3,072 | 17.90 |
| exact ‖U_k‖_F, current weights | 5.5106 | 1277 / 3,072 | 7.18 |
| decomposition (`uk_frobenius`), current weights | 5.4710 | **1289 / 3,072** | 6.13 |

**No other layer was examined, and none will be.** Searching for a layer that returns rank 1
is exactly the result-fishing the protocol forbids. C-001 stays on hold; this row does not
support it.

## What this table does not say

No claim is made or implied here that genomic models are "more distributed", that NLP
super-weights are scalar while genomic ones are rows, or that the participation ratio
explains functional criticality. Those are interpretations and none is drawn.
