# E10 Phase 1 / Step D1 — decoder structural top-K freeze

**Supersedes the single-row-only version of this file.** Per the 2026-08-22 correction to the
E10 prompt (`e10_prompt_correction.md`), the decoder arm is not a single-row dose-response
replication of GENERator — it is a **ranked singleton causal spectrum** across each decoder's
pre-existing structurally-ranked high-gain row set, to test whether causal importance is
concentrated in one component or spread across several, before deciding whether decoder
tomography is even a meaningful next step (Step D4).

This file freezes only the **structural** side (Step D1): which rows, at what rank, by what
pre-existing score, plus same-layer random controls. No causal response has been measured.

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
  4 different scalar indices `i` — a genuine pre-existing SET, but only L1's exact `q1` had
  been computed before E10 (`MODEL_AND_BASIS_AUDIT.md` §3, gap #3). Applying the *identical*
  already-existing exact metric (`spectral_lib.row_spectral_metrics`, via
  `compute_olmo_set_q1.py`, this directory) to the other 3 published layers is not a new
  metric or a new row search — the 4 coordinates were already published pre-E10, only their
  `q1` scores were missing. **K=4 for OLMo** (all 4 published rows; no 5th exists to add, so
  no padding).
- **Phi-3**: pre-existing SET of 6 published rows, `q1` already computed for all 6 in E7
  (`results/e7_phi3_spectral.json`). All 6 are genuinely high-gain published coordinates (range
  `q1=0.558`–`0.953`) — dropping any of them to force K=5 would be an arbitrary cut with no
  structural justification, and the correction explicitly permits K up to 10 "if the extra
  rows are genuinely high-gain." **K=6 for Phi-3** (the full pre-existing set, no padding, no
  drop).

No row above was chosen or reordered by looking at any causal response — every ranking is by
the already-existing exact `q1` score, computed or located before this file was written.

---

## 1. Llama (`huggyllama/llama-7b`) — K=1

| Rank | Layer | Row | `q1` | Source |
|---:|---:|---:|---:|---|
| 1 | 2 | 3968 | 0.9888 | Yu et al. Table 2; `MODEL_AND_BASIS_AUDIT.md` §1 |

Intervention hook: `_resolve_module(model, "model.layers.{i}.mlp.down_proj", 2).weight.data[3968, :] *= alpha`.

**Same-layer controls** (layer 2, seed `SeedSequence(42).spawn(5)[0]`, target row 3968
excluded from the pool): rows `[892, 2034, 2379, 3729, 3751]` (`results/e10_decoder_control_rows.json`).

## 2. Mistral (`mistralai/Mistral-7B-v0.1`) — K=1

| Rank | Layer | Row | `q1` | Source |
|---:|---:|---:|---:|---|
| 1 | 1 | 2070 | 0.9922 | Yu et al. Table 2; `MODEL_AND_BASIS_AUDIT.md` §2 |

Intervention hook: `_resolve_module(model, "model.layers.{i}.mlp.down_proj", 1).weight.data[2070, :] *= alpha`.

**Same-layer controls** (layer 1, seed index 1, target row 2070 excluded): rows
`[190, 305, 746, 1912, 2687]`.

## 3. OLMo (`allenai/OLMo-7B-0724-hf`) — K=4

Output row 269 recurring at 4 layers, each with a distinct scalar `i` (Yu et al. Table 2).
`q1` for L1 was already on record pre-E10; L2/L7/L24 computed here for the first time using
the unmodified E7 metric (`results/e10_olmo_set_q1.json`).

| Rank | Layer | Row | `q1` | Published `i` | Source |
|---:|---:|---:|---:|---:|---|
| 1 | 24 | 269 | 0.9990 | 2300 | `results/e10_olmo_set_q1.json` |
| 2 | 1  | 269 | 0.9646 | 7467 | `MODEL_AND_BASIS_AUDIT.md` §3 (pre-E10) |
| 3 | 2  | 269 | 0.9611 | 8275 | `results/e10_olmo_set_q1.json` |
| 4 | 7  | 269 | 0.9521 | 453  | `results/e10_olmo_set_q1.json` |

**Note (declared, not smoothed over):** under the exact metric, L24 is actually the
highest-`q1` row in OLMo's published set, not L1 — L1 was previously treated as "the"
canonical row in E1/E5/E7 only because it is the paper's first-listed (lowest-layer) entry, not
because it was the highest-scoring one. This freeze uses **all 4** and ranks by `q1`, so L1's
prior "canonical" status does not privilege it in the singleton spectrum.

Intervention hooks: `_resolve_module(model, "model.layers.{i}.mlp.down_proj", L).weight.data[269, :] *= alpha`
for `L in {24, 1, 2, 7}`.

**Same-layer controls** (layer 24 — the top-ranked row's layer, seed index 2, target row 269
excluded): rows `[292, 437, 2877, 2908, 3160]`.

## 4. Phi-3 (`microsoft/Phi-3-mini-4k-instruct` @ `f39ac1d28e925b323eae81227eaba4464caced4e`) — K=6

Full pre-existing published set (Yu et al. Table 2), `q1` from `results/e7_phi3_spectral.json`.

| Rank | Layer | Row | `q1` | Published `i` |
|---:|---:|---:|---:|---:|
| 1 | 2 | 525  | 0.9529 | 808  |
| 2 | 4 | 525  | 0.9278 | 2723 |
| 3 | 2 | 1693 | 0.9137 | 808  |
| 4 | 4 | 1693 | 0.8919 | 2723 |
| 5 | 2 | 1113 | 0.7505 | 808  |
| 6 | 4 | 1113 | 0.5584 | 2723 |

Intervention hooks: `_resolve_module(model, "model.layers.{i}.mlp.down_proj", L).weight.data[R, :] *= alpha`
for each `(L, R)` pair above (packed `gate_up_proj` is irrelevant here — `down_proj` is a
standalone `Linear`, identical in kind to the other 4 decoders).

**Same-layer controls** (layer 2 — the top-ranked row's layer, seed index 3, rows 525/1693/1113
excluded since all three appear at layer 2): rows `[356, 2345, 2759, 2830, 2983]`.

## 5. Qwen2.5 (`Qwen/Qwen2.5-7B` @ `d149729398750b98c0af14eb82c78cfe92750796`) — K=1

| Rank | Layer | Row | `q1` | Source |
|---:|---:|---:|---:|---|
| 1 | 26 | 458 | 0.9529 | This project's own frozen prospective detection (E7 Phase 1); `MODEL_AND_BASIS_AUDIT.md` §5 |

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
