# E1 — NLP cold-weight validation, retrospective arm

**Date:** 2026-08-12
**Arm:** retrospective only. The prospective arm was **not** run — it needs a model choice
and a lock.
**Ground truth:** Yu et al. (2024), arXiv:2411.07191, Table 2 "Super Weight Directory",
read from the local copy at `docs/superweight_paper.txt` rather than from memory.

Weights only. No forward pass, no GPU. Only the safetensors shards holding each target
layer's three MLP tensors were downloaded.

## Self-test, run first as instructed

```
$ python src/test_uk_frobenius.py
all checks passed
  planted row      rank 1, max/median 295.1
  planted scalar   rank 1, top1_share 1.0000, PR 1.00
  control row      PR 76.5 of 256, regime distributed
```

Passes. (It was failing until `2c83692`, on a float32 reference in the test's own assertion
rather than anything in the predictor — the predictor agrees with a float64 reference to
4.5e-16.)

## Level 1 — row recovery

Rank of the published output row *k* among all d_model rows of ‖U_k‖_F at the published
layer. The layer median is the implicit control: max/median states how far the recovered
row stands above a typical row.

| Model | layer | published k | **rank** | of | percentile | max/median |
|---|---|---|---|---|---|---|
| Llama-7B | 2 | 3968 | **1** | 4,096 | 100.000 | 29.4× |
| Mistral-7B v0.1 | 1 | 2070 | **1** | 4,096 | 100.000 | 26.8× |
| OLMo-7B 0724-hf | 1 | 269 | **1** | 4,096 | 100.000 | 37.8× |

**3/3 at rank 1.**

## Level 2 — scalar recovery

Within the recovered row, the per-*i* decomposition c_{k,i}.

| Model | published i | top1_index | match | **rank of i** | of | top1_share | top5_share | PR | regime |
|---|---|---|---|---|---|---|---|---|---|
| Llama-7B | 7003 | 7003 | ✓ | **1** | 11,008 | 0.8903 | — | 1.24 | scalar-dominated |
| Mistral-7B v0.1 | 7310 | 7310 | ✓ | **1** | 14,336 | 0.9884 | — | 1.02 | scalar-dominated |
| OLMo-7B 0724-hf | 7467 | 7467 | ✓ | **1** | 11,008 | 0.9558 | — | 1.09 | scalar-dominated |

**3/3 at rank 1, and in every case the top-1 contributor *is* the published index.**
Participation ratios of 1.02–1.24 mean the row's contribution is carried by
approximately one hidden unit.

## Reporting rule

PHASE_1_BLOCKING §E1: *"recovers the published super-weight output rows" if only Level 1
holds. Scalar recovery only if Level 2 also holds.*

**Level 2 holds for all three models, so scalar recovery may be claimed.** The predictor
recovers both the published output row and the published scalar index, from cold weights,
with no forward pass.

## Shape guards

The canonical convention (gate/up `[d_ffn, d_model]`, down `[d_model, d_ffn]`) was asserted
per model before any number was computed; a silent transpose produces a plausible-looking
but wrong ranking. All three matched: Llama and OLMo `(11008, 4096)` / `(4096, 11008)`,
Mistral `(14336, 4096)` / `(4096, 14336)`.

## Not done

The prospective arm (D-006) — pick an NLP model Yu et al. did not cover, predict cold, lock,
then run the forward-pass sweep. Requires a model choice and `prereg_lock.py`. C-006 stays
`pending`.

## Artifacts

`results/e1_nlp_retrospective.json`, `logs/e1_retrospective.log`
