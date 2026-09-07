# E10b Phase 0 — audit of the frozen Phi-3 basis from E10 v2

Read-only. No value below was recomputed; every number is a citation to an already-committed
E10 v2 artifact (commit `626cddd` and its ancestors). Nothing is changed by this file.

## Checkpoint / architecture

- Checkpoint: `microsoft/Phi-3-mini-4k-instruct`
- Revision: `f39ac1d28e925b323eae81227eaba4464caced4e` (pinned; matches
  `results/e10_decoder_spectrum_phi3.json`'s `resolved_revision` exactly)
- Module path: `model.layers.{i}.mlp.down_proj` (standalone `Linear`; only `gate_up_proj` is
  packed for Phi-3 — `down_proj` is architecturally identical in kind to the other 4 E10
  decoders) — `DECODER_INTERVENTION_FREEZE.md` §4
- Intervention hook: `_resolve_module(model, "model.layers.{i}.mlp.down_proj", layer)
  .weight.data[row, :] *= alpha` (`scripts/evaluation/run_gue_ablation.py`, reused via
  `e10_lib.py`)

## Frozen E10 v2 structurally selected basis — K=6 (unchanged here)

Ranked by exact ‖U_k‖_F relative to layer median (`e10_lib` / `exact_uk_norm_all_rows.py`,
protocol-corrected criterion — **not** `q1`), per `DECODER_INTERVENTION_FREEZE.md` §4 and
`results/e10_exact_uknorm_phi3.json`:

| Structural rank | Layer | Row | exact ‖U_k‖_F | ‖U_k‖_F / layer median | within-layer rank | `q1` (annotation only) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 525  | 120.77 | 7.75x | #0 | 0.9529 |
| 2 | 2 | 1693 | 105.86 | 6.79x | #1 | 0.9137 |
| 3 | 2 | 1113 | 97.52  | 6.26x | #2 | 0.7505 |
| 4 | 4 | 525  | 72.12  | 4.39x | #0 | 0.9278 |
| 5 | 4 | 1113 | 61.70  | 3.75x | #1 | 0.5584 |
| 6 | 4 | 1693 | 46.49  | 2.83x | #2 | 0.8919 |

`q1` is reproduced here strictly as an annotation, per the protocol correction
(`PROTOCOL_CORRECTION_01.md`) — it plays no role in this basis's construction or in E10b.

## E10 singleton causal effects (already measured, not remeasured here)

From `results/e10_decoder_spectrum_phi3.json` (baseline NLL = 2.384904, 100 WikiText-2 windows
x 512 tokens, seed 42, batch_size 8, float32):

| Structural rank | Layer | Row | `dNLL` (α=0.0) | relative % change | control-normalized | mean logit KL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 525  | +0.110243 | +4.62% | 1044.5x | 0.13249 |
| 2 | 2 | 1693 | +0.105221 | +4.41% | 996.9x  | 0.13214 |
| 3 | 2 | 1113 | +0.021774 | +0.91% | 206.3x  | 0.03986 |
| 4 | 4 | 525  | +0.058174 | +2.44% | 551.2x  | 0.06882 |
| 5 | 4 | 1113 | +0.007277 | +0.31% | 68.9x   | 0.01915 |
| 6 | 4 | 1693 | +0.028221 | +1.18% | 267.4x  | 0.03321 |

Rank-1 row also measured at α=0.5: `dNLL=+0.039422` (+1.65%), directionally intermediate
between baseline and full ablation.

## Random-control effects (already measured)

Layer 2, seed `SeedSequence(42).spawn(5)[3]` (Phi-3's panel index), rows
`[356, 2345, 2759, 2830, 2983]`, none overlapping the 3 layer-2 target rows:

| Row | `dNLL` |
|---:|---:|
| 356 | −0.000239 |
| 2345 | −0.000097 |
| 2759 | −0.000039 |
| 2830 | +0.000105 |
| 2983 | +0.000110 |

Control median `|dNLL|` = 0.00010455 — every one of the 6 target rows exceeds this by
69x-1045x, i.e. the entire causal-concentration question is about *which* rows carry the
effect, not *whether* the effect is real.

## E10 C1 / decision (already computed)

`results/e10_decoder_concentration.json`: **C1 = 0.3332**, **C2 = 0.6511**, decision
**MULTI_COMPONENT_CANDIDATE** — the largest single effect (rank 1) accounts for only 33.3% of
the total absolute singleton effect across the 6-row set; ranks 1 and 2 are nearly tied
(+0.1102 vs +0.1052).

## Why Phi-3 alone qualifies for tomography

Per the locked `PREREG_E10_nlp_architecture_causal_v2.md` Step D4 rule and
`e10_prompt_correction.md`'s hierarchical decision logic: tomography is licensed only when a
decoder's structurally selected rows show *comparable-order* singleton effects rather than
one row dominating. Of the 5 decoders in E10:

| Model | K | Decision |
|---|---:|---|
| Llama | 1 | SINGLE_COMPONENT_DOMINANT |
| Mistral | 1 | SINGLE_COMPONENT_DOMINANT |
| OLMo | 4 | SINGLE_COMPONENT_DOMINANT (C1=0.608) |
| **Phi-3** | **6** | **MULTI_COMPONENT_CANDIDATE (C1=0.333)** |
| Qwen2.5 | 1 | SINGLE_COMPONENT_DOMINANT |

Phi-3 is the only decoder whose primary decision is `MULTI_COMPONENT_CANDIDATE` — the only
one for which the prereg's own logic flags tomography as scientifically meaningful rather than
a symmetry exercise. This is not a rescue: it is the flagged-but-not-run second stage the E10
v2 prereg explicitly anticipated ("Flag as tomography-eligible; do not automatically run
F0-F3 — only if compute/time is acceptable and this prereg is amended (a fresh, separate lock)
to explicitly authorize it before any such run").

## What this file changes

Nothing. No E10 result is modified, rerun, or reinterpreted here. All values above are
citations to files already committed at or before `626cddd`.
