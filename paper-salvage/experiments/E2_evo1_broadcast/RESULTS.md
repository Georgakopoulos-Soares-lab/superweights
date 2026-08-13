# E2 — standardized broadcast, locked protocol

**Date:** 2026-08-13
**Prereg:** `docs/prereg/PREREG_evo1_broadcast.md` v2
**Lock:** sha256 `3d7515d0b7889f65038acd8479e85976248e7dbd0e5a701303de3a1b37dbfdc3`,
UTC `2026-08-13T02:59:20+00:00`, git `695eb92`, ledger `docs/prereg/LOCKS.jsonl`, `verify` OK.
All results below were generated **after** that lock.

Protocol: D-011 (AC-relative ε) + D-013 (fp32, headroom) + D-014 (DNABERT-2 eager) +
D-015 (T/C primary, KL secondary). α = 0.01 primary, α = 1.0 secondary. seed 42,
n_controls 5. **Tables only — no interpretation.**

**All five models complete**, with one dose failure recorded in §E.

---

## A. Instrumentation and provenance

| Model | source layer | SW row | token pos | d_model | dtype | attention / kernel | ε (α=0.01) | ε (α=1.0) | std AC | noise floor (KL) | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GENERator EUK | 4 | 2371 | 0 | 3,072 | float32 | default HF Llama | 9.3671e+00 | 9.3671e+02 | 9.3671e+02 | **0.0 exact** | complete |
| GENERator PROK | 2 | 1927 | 0 | 3,072 | float32 | default HF Llama | 1.6849e-02 | 1.6849e+00 | 1.6849e+00 | **0.0 exact** | complete |
| DNABERT-2 | 5 | 603 | 0 | 768 | float32 | **eager PyTorch** (Triton disabled, D-014) | 3.7666e-03 | 3.7666e-01 | 3.7666e-01 | **0.0 exact** | complete |
| NTv3 | 11 | 1472 | 2 | 1,536 | float32 | default | 5.9647e-01 | 5.9647e+01 | 5.9647e+01 | **0.0 exact** | complete, **no downstream layers** |
| Evo1 | 11 | 3776 | 324 | 4,096 | float32 | `use_flash_attn=False` before `StripedHyena(config)` | 3.3271e+03 | — | 3.3271e+05 | **0.0 exact** | **primary complete; secondary FAILED (OOM)** |

Headroom: **0 under-powered layers for every model** (minimum observed 4.31× at PROK L3;
Evo1 26× from L13 on and 1.66e+03× at L12; all others ≥ 300×). The ≥ 4× requirement of the null
rule is therefore satisfied everywhere a value is reported below.

Checkpoints: `GenerTeam/GENERator-v2-{eukaryote,prokaryote}-3b-base`;
`zhihan1996/DNABERT-2-117M @ 7bce263`; `InstaDeepAI/NTv3_650M_pre @ code_revision 0ecff36`.

## B. Primary endpoint — T and C (α = 0.01)

| Model | layers | T first → last | peak T | C first → last | peak C |
|---|---|---|---|---|---|
| GENERator EUK | 5–29 | 1.195e-02 → 3.791e-02 | 3.791e-02 | **+1.0000 → +0.9911** | +1.0000 |
| GENERator PROK | 3–29 | 1.473e-02 → **6.182e+00** | 6.182e+00 | **+0.9177 → +0.0056** | +0.9177 |
| DNABERT-2 (eager) | 6–11 | 6.437e-01 → 1.969e-01 | 8.009e-01 | **+0.0331 → −0.0221** | +0.0625 |
| NTv3 | — | **not defined** | — | **not defined** | — |
| Evo1 | 12–31 | 6.759e-03 → 1.1997e-01 | 1.1997e-01 | **+0.9516 → +0.7044** | +0.9516 |

**NTv3 has no downstream layers.** Its SW layer is 11 of 12, so the assay has nowhere to
measure (the H ≈ 0.08 observability covariate, C-018). T and C are undefined for NTv3 — this
is a structural property of where its SW sits, not a null.

Dose agreement (α = 1.0 secondary, T/C shown for the ε-invariance check only):
EUK peak T 3.709e-02 (vs 3.791e-02, 2.2%), PROK peak T 6.196e+00 (vs 6.182e+00, 0.2%),
DNABERT-2 peak T 7.960e-01 (vs 8.009e-01, 0.6%). C likewise stable
(EUK +0.9912 vs +0.9911; PROK +0.0055 vs +0.0056). **T and C are ε-invariant across the two
doses**, as the prereg's comparability argument requires.

No preregistered numerical threshold on C exists in the protocol, so no "layers above
threshold" column is reported. None is invented here.

## C. Secondary endpoint — KL (α = 1.0), against each model's noise floor

| Model | KL, SW arm | noise floor | random-coord (mean) | random-dense (mean) | matched-norm (mean) |
|---|---|---|---|---|---|
| GENERator EUK | 3.4351e-06 | 0.0 | 1.2794e-05 | 1.9580e-05 | **arm empty — see below** |
| GENERator PROK | 9.9404e-07 | 0.0 | 1.8974e-06 | 1.8077e-06 | 1.7510e-06 |
| DNABERT-2 (eager) | **4.6270e-05** | 0.0 | 1.2063e-05 | 1.8890e-05 | **arm empty** |
| NTv3 | 1.0919e-05 | 0.0 | 9.3069e-05 | 5.9372e-05 | **arm failed — see below** |
| Evo1 | **not measured — dose failed** | 0.0 | — | — | — |

At the primary dose KL is: EUK 0.0 (numerically zero), PROK 0.0 (numerically zero),
DNABERT-2 1.8243e-08, NTv3 7.7175e-08. **Tiny values are reported as tiny, not relabelled
zero**; the two exact zeros are exact.

Every noise floor is exactly 0.0, so the second half of the null rule (noise floor below
signal) is satisfied wherever a non-zero signal is reported.

### Control-arm execution facts

- **matched-norm is empty for GENERator EUK and DNABERT-2.** The prereg defines the arm as a
  random row whose ‖W_down[k′,:]‖ is within ±10% of the SW row's. For EUK the SW row's norm
  is 1.4260e+01 and **0 of 3,072 rows** qualify; for DNABERT-2 it is 5.1931e+00 with **0 of
  768**. PROK has 3,063 qualifying rows. The band was **not widened** — that would be tuning
  a locked protocol. Reported as an execution fact.
- **matched-norm failed for NTv3**: the impulse harness's own `_sw_output_matrix` could not
  locate NTv3's down-projection. This is the same defect class as N-010 and is now fixable
  via the shared `ADAPTERS["ntv3"]` fixed this session, but the fix was **not** applied and
  NTv3 was **not** re-run, because NTv3 has no downstream layers and the arm could not yield
  T or C regardless. Logged, not repaired.
- random-coordinate, neighbouring-row (±1) and random-dense arms ran for all four models.

## D. DNABERT-2 correction — required disclosure

- The **old Triton result was noise-dominated.** Two identical forward passes differed by
  KL = 0.305 and ‖Δh‖ = 4.50, against an "injected signal" of KL = 0.277 and ‖Δh‖ = 4.50;
  per-layer SNR 0.930–1.000.
- **C-014 is RETIRED** (X-006). Its headline "largest impulse KL ≈ 0.31" matched that noise
  floor to two significant figures. It is not revised and must never be resurrected.
- **Clean eager values** (this run): KL = **1.824e-08** at α = 0.01 and **4.627e-05** at
  α = 1.0; C = +0.0331 → −0.0221, peak +0.0625; T peak 8.009e-01. Noise floor exactly 0.0.
- **D-014**: DNABERT-2 runs on eager attention; Triton is the defective path.
- **Guard observation: Δ = −74.3%** (masked-LM perplexity 687.9 → 176.9, same weights, same
  inputs, kernel the only difference). **This guard did not pass and is not described as
  passed.** It is reported because the kernel decision rests on the other two facts below.
- **Mechanism**: the Triton path casts qkv *and* the attention bias to **fp16**
  (`convert_dtype` in `BertUnpadSelfAttention.forward`), so DNABERT-2's attention never ran
  in fp32 at inference; the fp32 parameter dtype was cosmetic for attention.
- **Eager is bit-reproducible**: perplexity spread 0.000% across three repeats, against
  8.48% for Triton (720.6 / 681.1 / 662.2).

## E. Evo1

**Primary dose complete. Secondary dose (α = 1.0) FAILED — CUDA out of memory.**

Run inside `evo2.sif`, fp32, `use_flash_attn=False` set before `StripedHyena(config)`
(3 of 32 blocks change kernel; `attn_layer_idxs = [8, 16, 24]`), seed 42, n_controls 5.

**Noise floor: exactly 0.0** (`impulse_determinism_check --model evo1`, verdict
DETERMINISTIC). ε = 0.01 × std AC 3.3271e+05 = **3.3271e+03**.

### Primary endpoint (α = 0.01) — this is a measurement, not a null

| layer | T | C | headroom |
|---|---|---|---|
| 12 | 6.7586e-03 | +0.9516 | 1.66e+03× |
| 13 | 1.1995e-01 | +0.7045 | **26×** |
| 15 | 1.1996e-01 | +0.7044 | 26× |
| 31 | 1.1997e-01 | +0.7044 | 26× |

**0 of 20 downstream layers under-powered.** The headroom of 26× at L13+ matches D-013's
pre-run projection of ~26× at the max coordinate under fp32. Both halves of the null rule
are satisfied: headroom ≥ 4× everywhere, and the noise floor (0.0) is below the signal.

**Branch C is excluded.** The precision floor did not persist: the assay has dynamic range
on Evo1 under fp32, the random-coordinate arm is non-flat, and no layer is unmeasured.
Branch D was already excluded by the STEP 2 gate.

KL at the primary dose: 7.8678e-08.

### Control arms — peak T

| arm | n | peak T (mean) | KL (mean) |
|---|---|---|---|
| **SW row** | 1 | **1.1997e-01** | 7.868e-08 |
| random-coordinate | 5 | 1.1041e-01 | 2.940e-08 |
| random-dense-direction | 5 | 1.1639e-01 | 3.296e-08 |
| matched-norm (15 rows in band) | 5 | 1.1696e-01 | 6.885e-08 |

Reported as measured: the SW arm's peak T (1.1997e-01) and the three control arms'
(1.10–1.17e-01) are within ~9% of each other. The prereg's control requirement — that the
random-coordinate arm be **non-flat** — is satisfied. What that similarity means is not
interpreted here.

### The failed dose, recorded not hidden

The α = 1.0 secondary dose raised `CUDA out of memory` (tried to allocate 172 MiB with
131 MiB free of 39.49 GiB) after the primary dose had already run in the same process.

**No retry was performed.** The single provenance-preserving retry the run spec allows would
need the secondary dose to run in a fresh process, which the harness cannot do without a
per-dose flag — a code change, which is not a provenance-preserving retry. The failure is
therefore written into the results rather than silently excluded, and Evo1's KL column is
**not measured**, not zero.

## F. Prereg branch

**C and D are excluded. A vs B is not mechanically determined and is left open.**

- **Branch D** (precondition failed) — excluded at the STEP 2 gate.
- **Branch C** (precision floor persists, both arms flat) — excluded: 26× headroom, 0
  under-powered layers, controls non-flat.
- **Branch A vs B** turns on whether "Evo1 T is low relative to the other four". The peak T
  values are: EUK 3.79e-02, **Evo1 1.20e-01**, DNABERT-2 8.01e-01, PROK 6.18e+00, NTv3
  undefined. Evo1 sits above EUK and below DNABERT-2 and PROK. That is not the clean "low
  relative to the other four" that Branch A names, nor the "comparable to or greater than
  DNABERT-2's" that the v1 falsifier names.

**No branch is assigned.** Choosing between A and B here would require judging what counts
as "low", which is the interpretive step this report does not take.

## What is deliberately absent

No R3 thesis sentence. No statement about whether T predicts criticality or whether C is
necessary. No AC/DC decomposition. No claim about X-003 beyond the mechanical ledger rule
recorded in `CLAIMS_LEDGER.md` N-006.

## Artifacts

`results/sw_broadcast_impulse.json` (keys `<model>` = primary, `<model>__alpha1` = secondary;
`evo1_fixed_eps_SUPERSEDED` is the void pre-D-011 entry, retained);
`results/sw_broadcast_impulse_PRE_D011_fixed_eps.json` (preserved baseline);
`results/impulse_determinism_<model>.json`; `logs/e2_run_<model>.log`.
