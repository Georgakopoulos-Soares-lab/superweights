# E10 Phase 1 / Step D1 — decoder structural top-K freeze (v2, corrected)

**Supersedes the single-row-only version of this file.** Per the 2026-08-22 correction to the
E10 prompt (`e10_prompt_correction.md`), the decoder arm is not a single-row dose-response
replication of GENERator — it is a **ranked singleton causal spectrum** across each decoder's
pre-existing structurally-ranked high-gain row set, to test whether causal importance is
concentrated in one component or spread across several, before deciding whether decoder
tomography is even a meaningful next step (Step D4).

> **v1 ranked OLMo's and Phi-3's multi-row sets by `q1`, which was wrong** — `q1` is
> scale-invariant and measures rank-1-ness, not gain. See `PROTOCOL_CORRECTION_01.md`. This v2
> ranks by **exact ‖U_k‖_F relative to layer median** (`uk_frobenius.layer_report`'s
> `max_over_median` convention, using `spectral_lib`'s exact norm). The three `K=1` models were
> never ranked and are unaffected. **No causal measurement was run under v1.** No top-1 row
> changed, so all control rows stand unchanged.

This file freezes only the **structural** side (Step D1): which rows, at what rank, by what
pre-existing score, plus same-layer random controls. No causal response has been measured.
`q1` appears below **as an annotation column only**, never as a selector.

## Intervention mechanism (unchanged from the original freeze)

All 5 decoders use an unpacked `down_proj` (`Linear(intermediate_size, hidden_size)`), so the
row-scale hook is architecturally identical across all 5 models, reusing the existing generic
resolver (`scripts/evaluation/run_gue_ablation.py::_resolve_module` / `_save_row` /
`_restore_row`):

```
alpha = 1 - epsilon * a
_resolve_module(model, "model.layers.{i}.mlp.down_proj", layer).weight.data[row, :] = saved_row * alpha
```

This is a weight-space row scale on the exact coordinate the structural detection already
identified — not the activation-space singular-vector projection that
`E8_encoder_decoder/CAUSAL_FOLLOWUP_FEASIBILITY.md` found infeasible without new tooling (that
verdict is about a different object, ablating a `U_k` singular *vector*; see the original
freeze rationale, preserved in git history).

## Step D1 procedure actually applied, per model

Per the correction: "locate the existing pre-E10 structural ranking / exact `U_k` or
equivalent high-gain score," "use K=5 by default if all five are legitimate pre-existing
high-gain candidates," "K up to 10 only if the existing structural pipeline naturally supports
it," "do NOT pad with ordinary rows," "do NOT choose rows based on causal effects."

- **Llama, Mistral, Qwen2.5**: pre-E10 artifacts record exactly **one** legitimate high-gain
  coordinate for each of these three checkpoints (`MODEL_AND_BASIS_AUDIT.md` §1, §2, §5) — no
  second candidate exists to rank. **K=1 for each**, not padded to 5.
- **OLMo**: Yu et al. Table 2 publishes the *same* output row (269) recurring at 4 layers with
  4 different scalar indices `i` — a genuine pre-existing SET. The 4 coordinates were already
  published pre-E10; only their structural scores were missing. Scoring them with the existing
  exact machinery is not a new metric or a new row search. **K=4 for OLMo** (all 4 published
  rows; no 5th exists, so no padding).
- **Phi-3**: pre-existing SET of 6 published rows. All 6 are genuinely high-gain published
  coordinates — dropping any to force K=5 would be an arbitrary cut with no structural
  justification, and the correction explicitly permits K up to 10 "if the extra rows are
  genuinely high-gain." **K=6 for Phi-3** (the full pre-existing set, no padding, no drop).

Ranking scalar for both multi-row sets: **exact ‖U_k‖_F / layer median**, computed for every
row of each target layer via the SVD-free Gram identity in `exact_uk_norm_all_rows.py`
(validated against `spectral_lib.row_spectral_metrics` to ~6e-13 relative error). No row above
was chosen or reordered by looking at any causal response — none has been measured.

---

## 1. Llama (`huggyllama/llama-7b`) — K=1

| Rank | Layer | Row | `q1` (annotation) | Source / selection basis |
|---:|---:|---:|---:|---|
| 1 | 2 | 3968 | 0.9888 | Yu et al. Table 2 published coordinate — sole candidate, no ranking performed (`MODEL_AND_BASIS_AUDIT.md` §1) |

Intervention hook: `_resolve_module(model, "model.layers.{i}.mlp.down_proj", 2).weight.data[3968, :] *= alpha`.

**Same-layer controls** (layer 2, seed `SeedSequence(42).spawn(5)[0]`, target row 3968
excluded from the pool): rows `[892, 2034, 2379, 3729, 3751]` (`results/E10/e10_decoder_control_rows.json`).

## 2. Mistral (`mistralai/Mistral-7B-v0.1`) — K=1

| Rank | Layer | Row | `q1` (annotation) | Source / selection basis |
|---:|---:|---:|---:|---|
| 1 | 1 | 2070 | 0.9922 | Yu et al. Table 2 published coordinate — sole candidate, no ranking performed (`MODEL_AND_BASIS_AUDIT.md` §2) |

Intervention hook: `_resolve_module(model, "model.layers.{i}.mlp.down_proj", 1).weight.data[2070, :] *= alpha`.

**Same-layer controls** (layer 1, seed index 1, target row 2070 excluded): rows
`[190, 305, 746, 1912, 2687]`.

## 3. OLMo (`allenai/OLMo-7B-0724-hf`) — K=4

Output row 269 recurring at 4 layers, each with a distinct scalar `i` (Yu et al. Table 2).
Ranked by exact ‖U_k‖_F relative to layer median (`results/E10/e10_exact_uknorm_olmo.json`).

| Rank | Layer | Row | ‖U_k‖_F / layer median | exact ‖U_k‖_F | within-layer rank | `q1` (annotation) | Published `i` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 269 | 53.4x | 1.6369 | #0 of 4096 | 0.9990 | 2300 |
| 2 | 7  | 269 | 39.7x | 1.1681 | #0 of 4096 | 0.9521 | 453  |
| 3 | 1  | 269 | 39.2x | 0.9111 | #0 of 4096 | 0.9646 | 7467 |
| 4 | 2  | 269 | 20.8x | 0.5979 | #0 of 4096 | 0.9611 | 8275 |

**All four rows are the #0 (top) row of their own layer** by exact ‖U_k‖_F — every member of
the set is genuinely high-gain, which is why the set itself was never in question.

**Note (declared, not smoothed over):** v1 of this file ranked L24 first *because its `q1` was
highest* — an invalid reason, now withdrawn (`PROTOCOL_CORRECTION_01.md` §3). Under the
corrected magnitude criterion L24 still outranks canonical L1 (53.4x vs 39.2x layer median),
so the top-1 choice survives, but **on different and now valid grounds**. The ordering does
change: **L7 moves above L1**, where `q1` had placed it last. L1 retains its status as this
project's designated canonical row (E1/E5/E7 primary) — recorded, but not used to set rank
order.

Intervention hooks: `_resolve_module(model, "model.layers.{i}.mlp.down_proj", L).weight.data[269, :] *= alpha`
for `L in {24, 7, 1, 2}`.

**Same-layer controls** (layer 24 — the top-ranked row's layer, seed index 2, target row 269
excluded): rows `[292, 437, 2877, 2908, 3160]`.

## 4. Phi-3 (`microsoft/Phi-3-mini-4k-instruct` @ `f39ac1d28e925b323eae81227eaba4464caced4e`) — K=6

Full pre-existing published set (Yu et al. Table 2), ranked by exact ‖U_k‖_F relative to layer
median (`results/E10/e10_exact_uknorm_phi3.json`).

| Rank | Layer | Row | ‖U_k‖_F / layer median | exact ‖U_k‖_F | within-layer rank | `q1` (annotation) | Published `i` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 525  | 7.8x | 120.77 | #0 | 0.9529 | 808  |
| 2 | 2 | 1693 | 6.8x | 105.86 | #1 | 0.9137 | 808  |
| 3 | 2 | 1113 | 6.3x | 97.52  | #2 | 0.7505 | 808  |
| 4 | 4 | 525  | 4.4x | 72.12  | #0 | 0.9278 | 2723 |
| 5 | 4 | 1113 | 3.8x | 61.70  | #1 | 0.5584 | 2723 |
| 6 | 4 | 1693 | 2.8x | 46.49  | #2 | 0.8919 | 2723 |

**Note:** v1 ranked these by `q1`, which placed L4/r1113 *last* (`q1` = 0.5584) though it is
the 5th-largest of the six operators, and L4/r1693 4th though it is the *smallest*. Top-1
(L2/r525) is unchanged, so the control layer is unchanged.

Intervention hooks: `_resolve_module(model, "model.layers.{i}.mlp.down_proj", L).weight.data[R, :] *= alpha`
for each `(L, R)` pair above (packed `gate_up_proj` is irrelevant here — `down_proj` is a
standalone `Linear`, identical in kind to the other 4 decoders).

**Same-layer controls** (layer 2 — the top-ranked row's layer, seed index 3, rows 525/1693/1113
excluded since all three appear at layer 2): rows `[356, 2345, 2759, 2830, 2983]`.

## 5. Qwen2.5 (`Qwen/Qwen2.5-7B` @ `d149729398750b98c0af14eb82c78cfe92750796`) — K=1

| Rank | Layer | Row | `q1` (annotation) | Source / selection basis |
|---:|---:|---:|---:|---|
| 1 | 26 | 458 | 0.9529 | This project's own frozen prospective **activation detector** (E7 Phase 1, ratio 639.2x) — sole candidate, no ranking performed (`MODEL_AND_BASIS_AUDIT.md` §5) |

Intervention hook: `_resolve_module(model, "model.layers.{i}.mlp.down_proj", 26).weight.data[458, :] *= alpha`.

**Same-layer controls** (layer 26, seed index 4, target row 458 excluded): rows
`[62, 1257, 2926, 3419, 3553]`.

---

## What this file does NOT do

- Does not run any forward pass or intervention — Step D2 (singleton causal measurement) is a
  separate, later step, gated by the Phase 4 prereg.
- Does not choose or drop a row based on any causal signal (none has been measured).
- Does not decide the primary endpoint (causal-LM loss/NLL) — that is Phase 3.
- Does not decide whether tomography (F0–F3) will be run on any decoder — that is Step D4,
  gated on the Step D2/D3 singleton results, per the correction's hierarchical decision rule.
- Does not pin the 3 unpinned decoder revisions (Llama/Mistral/OLMo) — open item, tracked in
  `MODEL_AND_BASIS_AUDIT.md`'s gaps section.
