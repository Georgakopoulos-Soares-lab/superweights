# E2 — instrument validation before the re-run (STEP 3)

**Date:** 2026-08-12
**Status: BLOCKER. The re-run was not started. The prereg was not locked.**

D-013 required a per-layer headroom column so that a null could be distinguished from an
unrepresentable perturbation. Building it surfaced two prior questions that headroom does
not answer, and one of them invalidates a published number.

---

## 1. Is the forward pass deterministic?

Δh is only a measurement of the perturbation if two identical passes agree. Checked by
running the same input twice with **no injection** and comparing that noise floor against
the actual SW signal (`scripts/interpretability/impulse_determinism_check.py`).

| Model | noise floor ‖Δh‖ | signal ‖Δh‖ | KL noise | KL signal | KL SNR | verdict |
|---|---|---|---|---|---|---|
| GENERator EUK | **0.0** (exact) | 0.112 → 0.355 | **0.0** | 0.0 | — | deterministic |
| GENERator PROK | **0.0** (exact) | 2.48e-4 → 0.104 | **0.0** | 0.0 | — | deterministic |
| NTv3 | **0.0** (exact) | — (no downstream layers) | **0.0** | 7.72e-8 | — | deterministic |
| **DNABERT-2** | **4.50 → 2.54** | 4.50 → 2.54 | **0.305** | 0.277 | **0.909** | **NOISE-DOMINATED** |

### DNABERT-2 is measuring its own nondeterminism

Per-layer SNR is 0.930–1.000. The injection changes the downstream residual by *the same
amount* as changing nothing at all.

The published claim (C-014) is **"DNABERT-2: C ≈ 0 immediately, largest impulse KL ≈ 0.31."**
The measured KL between two *identical* DNABERT-2 forward passes is **0.305**. The headline
number and the noise floor agree to two significant figures. The most likely reading is
that the largest impulse KL in the manuscript *is* the noise floor.

Not yet diagnosed (deliberately — no fix attempted without checking in). The likely source
is DNABERT-2's Triton flash-attention kernels; the model is in `eval()` and holds no dropout,
and the other three models on the same harness, same device and same fp32 path have an
exactly-zero floor, so this is model-specific rather than a harness bug.

### What this touches

| Item | Bearing |
|---|---|
| C-014 | Measured through the noise floor. Cannot stand as written. |
| C-015 "C describes routing geometry, not criticality" — strength `established` | Its evidence is C-012–014. The DNABERT-2 leg is the one supplying C ≈ 0. |
| D-004 | Rejected the joint (‖U_k‖_F, C) framing **because** DNABERT-2 has C ≈ 0 with the largest KL. That rationale rests on this measurement. |
| X-003 (retired claim) | Retired on the same evidence. |
| `CLAUDE.md` hard constraint: "DNABERT-2 has C ≈ 0 and the strongest encoder phenotype" | First clause is from the impulse assay and is affected. |
| C-027 (splice −25.5 pp), C-028 (per-row ablation) | **Unaffected.** Different experiment, not the impulse assay. The *phenotype* stands; only the *routing* number is in question. |

The conclusion "C is not necessary for criticality" may well survive — DNABERT-2's ablation
phenotype is independent evidence — but it can no longer rest on the impulse C value.

---

## 2. Is T comparable across models?

T = ‖Δh‖/ε is only a model property if the response is linear in ε, so the ratio is
ε-invariant (`scripts/interpretability/impulse_linearity_probe.py`).

**GENERator PROK (deterministic) — linear where powered:**

| ε | ‖Δh‖@L3 | T | KL |
|---|---|---|---|
| 1e-4 | 9.89e-6 | 9.89e-2 | 0.0 |
| 1e-3 | 2.23e-5 | 2.23e-2 | 0.0 |
| 1e-2 | 1.50e-4 | 1.497e-2 | 0.0 |
| **1.68e-2 (AC)** | 2.48e-4 | **1.473e-2** | 0.0 |
| 1e-1 | 1.45e-3 | 1.451e-2 | 0.0 |
| 1.0 | 1.45e-2 | 1.451e-2 | 2.35e-7 |
| 10 | 1.46e-1 | 1.457e-2 | 3.96e-5 |

T is flat to three significant figures across ε = 1e-2 … 10. Below 1e-3 it inflates — that
is the fp32 accumulation floor, exactly the regime D-013's headroom column is meant to flag.
The AC-relative ε lands inside the linear regime, and **T at AC-ε reproduces T at ε = 1.0 to
1.5%.** For this model the amended protocol is sound and the sanity expectation holds.

**DNABERT-2 — ε-independent, i.e. not a response at all:** over ε = 1e-4 … 10 (5 decades)
‖Δh‖ moved 1.14× (4.32 → 4.94). Fully explained by item 1: the measurement is noise at
every ε.

---

## 3. The prereg's sanity expectation is wrong on its own terms

> "α = 0.01 should land near ε = 1.0 for GENERator, DNABERT-2 and NTv3, whose activations
> are O(1–100) under normalisation."

Measured std of the AC component at each model's source layer:

| Model | std AC | ε at α=0.01 | vs ε=1.0 |
|---|---|---|---|
| GENERator EUK | 9.367e2 | **9.367** | 9× larger |
| GENERator PROK | 1.685 | **1.685e-2** | 59× smaller |
| DNABERT-2 | 3.714e-1 | **3.714e-3** | 269× smaller |
| NTv3 | 5.965e1 | **5.965e-1** | ≈ 1.0 ✓ |
| Evo1 | 3.326e5 (L11) | **3.326e3** | — |

ε spans **2,500×** across the four non-Evo1 models. Only NTv3 lands near 1.0. This does not
invalidate the protocol — PROK's linearity shows T is ε-invariant where powered, so the
values remain comparable — but the stated expectation should be corrected in the prereg
before locking, since it is currently a prediction that is already known to be false.

---

## 4. Other findings from the same work

- **Evo1 *can* run in pure fp32.** D-013 assumed it; the reference script
  `scripts/analysis/run_evo1_residual_attribution_fp32.py` asserts it cannot ("StripedHyena's
  flash-attn inner attn asserts fp16/bf16"). That is true only of the default path:
  `use_flash_attn` is baked into MHA at construction, so setting it to `False` **before**
  `StripedHyena(config)` is built gives a finite fp32 forward. Only 3 of 32 blocks change
  kernel (`attn_layer_idxs = [8, 16, 24]`). Residual magnitudes agree closely with bf16
  (L13 median |h| 4.09e6 vs 4.23e6; max 1.2953e9 vs 1.2918e9), so the scale structure is a
  property of the model. D-013's headroom projection is confirmed: ε = 3.4e3 against fp32
  ULP 128 at the max coordinate gives **27×**.
  - Note: that reference script writes `"dtype": "float32"` into its own output record while
    running bf16. Its stored results are mislabelled.
- **The harness only ever had three control arms, not four.** The prereg names
  random-coordinate, neighbouring-row, random-dense-direction and matched-norm; the code has
  the first three. Every "four control arms" statement about the existing models is
  inaccurate. A matched-norm arm is now implemented, with its definition stated in the code
  because the prereg does not define one — that definition needs confirmation before locking.
- **Original-run precision (D-013 asked):** GENERator EUK/PROK and NTv3 loaded with
  `torch_dtype=torch.float32`, DNABERT-2 defaulted to fp32, and no autocast wrapped any
  non-Evo1 path. **The three normalised models were already fp32**, so D-013's concern that
  their ε = 1.0 sat at ~1.3 bf16 ULP does not apply — no precision defect to record in the
  superseded numbers on that axis. Only Evo1 was bf16.
- **GENERator KL_sw is 0.0 at AC-ε for both models** (and 2.35e-7 at ε = 1.0 for PROK).
  Not obviously a defect — the perturbation is genuinely small in output terms — but it means
  KL carries no signal for the GENERator arm at α = 0.01, and any KL-based comparison in R3
  would rest on DNABERT-2, the one model whose KL is noise.

---

## Why the run was not started

Locking the prereg and running all five would have produced a five-model T/C/KL table in
which one column is run-to-run noise, presented next to four that are not, with a headroom
column certifying all five as adequately powered — because headroom measures representability,
not determinism. That is the same class of error D-011 was raised to correct.

Nothing here has been interpreted, no thesis sentence drafted, no AC/DC decomposition begun,
and the Phase 1b relative-change observation has not been pursued.

---

# Addendum — STEP 3a/3b/3c (2026-08-12)

## STEP 3a — determinism ladder

**Attempt 1 succeeded; attempts 2 and 3 were not needed.**

Disable Triton flash-attn, force the eager PyTorch path. Implemented as
`_dnabert2_force_eager_attention()`: the remote code guards the kernel behind
`if self.p_dropout or flash_attn_qkvpacked_func is None:`, so setting that module-global to
`None` selects the eager branch. `self.dropout` is identity in `eval()`.

| | before (Triton) | after (eager) |
|---|---|---|
| noise floor ‖Δh‖, L6–L11 | 4.50 → 2.54 | **0.0 exact** |
| KL noise floor | 0.305 | **0.0 exact** |
| SW signal ‖Δh‖ | 4.50 (= noise) | 2.42e-3 |
| SW signal KL | 0.277 (= noise) | 1.82e-8 |

The real signal is ~1,800× smaller than what the Triton path reported. **C-014's "largest
impulse KL ≈ 0.31" does not survive in any form** — the true value at the primary dose is
1.8e-8.

## STEP 3a guard — FAILED

Same weights, same inputs, same process; attention kernel the only difference. Three
repeats per arm, because the Triton path is not deterministic and a single value from it is
not a fixed quantity.

| Attention path | mean PPL | repeats | spread |
|---|---|---|---|
| Triton kernel, fp16 attention | 687.9 | 720.6 / 681.1 / 662.2 | **8.48%** |
| eager PyTorch, fp32 attention | **176.9** | 176.9 / 176.9 / 176.9 | **0.000%** |

**Δ = −74.29%, tolerance ±1%. The fix is not adopted.**

Interpretation is left open deliberately, but two facts constrain it:

- The Triton arm's perplexity varies by 8.5% run to run, so there is no stable "original"
  value for the fixed arm to match.
- The Triton path casts qkv and the attention bias to **fp16** (`convert_dtype` in
  `BertUnpadSelfAttention.forward`), so DNABERT-2's attention was never running the fp32
  D-013 mandates — the fp32 parameter dtype was cosmetic for attention.

The eager path is 3.9× better on perplexity and bit-reproducible. The natural reading is
that Triton is the defective path rather than that the fix damaged the model. That call is
substantive and is not made here.

### Unaudited scope

The loaded config has `attention_probs_dropout_prob = 0.0`, so `p_dropout` is falsy and the
guard selects **Triton by default**. (A second cached snapshot under the `transformers/`
cache carries 0.1, but it is not the one that loads.) Consequently every *inference-time*
DNABERT-2 analysis in the manuscript ran through Triton/fp16; fine-tuning runs, which set
dropout > 0, would have taken the eager path. Whether C-002, C-010, C-027 or C-028 are
affected **has not been audited** — flagged, not investigated, per the stop rule.

## STEP 3b — protocol corrections, all applied

1. **Matched-norm arm** redefined as the equal-norm random row: ε at a random k′ with
   ‖W_down[k′,:]‖ within ±10% of the SW row's, matching the v15 Fig 1B ablation control.
   Written into the prereg, not only the code. All "four control arms" statements corrected
   — the harness had three.
2. **Dual dose.** PRIMARY α = 0.01 (small-signal, T and C); SECONDARY α = 1.0 (AC-matched,
   KL). Necessary because at α = 0.01 the output KL is ~0 everywhere (EUK and PROK exactly
   0.0, DNABERT-2 1.8e-8). Both doses run for all five models and every arm.
3. **Sanity expectation corrected.** The "α = 0.01 lands near ε = 1.0" text is struck and
   replaced with the measured AC table; comparability now rests on ε-invariance of T,
   evidenced by the PROK linearity probe, not on ε being similar across models.

## STEP 3c — provenance

The prereg now carries the plain statement: *"The impulse protocol was revised twice during
instrument validation. Only the final protocol (v2, locked `<hash>`, `<date>`) was
preregistered; earlier versions were not."* The v1 transcript reconstruction has been
**deleted** and `docs/prereg/archive/README.md` records why. N-008 logs the
`run_evo1_residual_attribution_fp32.py` dtype mislabel.

## State

Everything in STEP 3b and 3c is done. The prereg is ready to lock the moment the DNABERT-2
kernel question is decided. Nothing has been interpreted, no thesis sentence drafted, no
AC/DC decomposition begun, and the Phase 1b observation is untouched.
