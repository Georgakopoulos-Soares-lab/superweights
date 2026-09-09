# E2 — BLOCKED at STEP 3d

**Date:** 2026-08-12
**Prereg NOT locked. Five-model re-run NOT started.**
**Trigger:** the pre-committed STEP 3d stop condition fired — KL is ~0 for GENERator PROK
at the secondary dose as well as the primary.

---

## What was asked

> If KL is measurable at alpha=1.0 for PROK: lock as specified, dual dose stands.
> If KL is ~0 at alpha=1.0 for PROK too: do NOT escalate the dose to chase a signal.
> Stop and report.

## What was measured

`scripts/interpretability/secondary_dose_probe.py`, fp32, DNABERT-2 on eager (D-014).

### GENERator PROK — STOP CONDITION MET

| | primary α=0.01 | secondary α=1.0 |
|---|---|---|
| ε | 1.685e-2 | 1.685 |
| KL (protocol definition) | **0.0** | **9.940e-7** |
| KL @ SW token position | −1.704e-7 † | 1.064e-6 |
| max abs Δ logit at that position | 2.708e-4 | 3.144e-2 |
| top-10 mean rank shift | **0.000** | **0.000** |
| top-10 max rank shift | 0 | 0 |
| top-1 token changed | False | **False** |
| top-10 set overlap | 1.00 | 1.00 |

† Negative by float cancellation. KL cannot be negative; the value is numerically zero.

A 100× dose increase moved KL from 0 to 1e-6 and moved **nothing** in the output ranking:
not one of the top-10 tokens changed rank, and the top-1 prediction is unchanged.

### DNABERT-2 (eager) — measurable

| | primary α=0.01 | secondary α=1.0 |
|---|---|---|
| ε | 3.767e-3 | 3.767e-1 |
| KL (protocol definition) | 1.824e-8 | **4.627e-5** |
| KL @ SW token position | 3.184e-7 | **1.325e-3** |
| max abs Δ logit at that position | 3.420e-3 | 3.305e-1 |
| top-10 mean rank shift | 0.000 | 0.200 |
| top-1 token changed | False | **True** |

DNABERT-2 alone satisfies the dual-dose design. PROK does not, and PROK is the gating model
named in the stop condition.

## Why this blocks rather than degrades gracefully

Locking and running now would produce an R3 KL column in which the only model with a
measurable functional effect is DNABERT-2 — the model whose kernel was just replaced and
whose previous KL value has been retired (X-006). A KL comparison resting on that single
column is exactly the fragility the dual dose was introduced to remove.

## The candidate replacement metrics were measured, and they do not rescue PROK

Both candidates named in the instruction were recorded alongside KL, so the proposal is
grounded rather than speculative. **Neither separates PROK's SW perturbation from nothing:**

- **KL restricted to the SW token position** — 1.064e-6 at α=1.0. Same order as the
  sequence-mean definition. No improvement.
- **Top-k rank displacement** — exactly 0.000 mean shift, 0 max shift, top-1 unchanged,
  top-10 overlap 1.00, at **both** doses.

Reported as measurements only. What it means — whether the metric is wrong, or the PROK SW
perturbation genuinely has no small-signal functional effect — is not decided here, and the
two are not distinguished by this probe.

## Not done, deliberately

No dose escalation. No metric substituted. No prereg lock. No five-model run. No thesis
sentence, no AC/DC decomposition, no Phase 1b item.

## Exact next command, once a metric decision is made

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights
python manuscript/src/prereg_lock.py lock manuscript/docs/prereg/PREREG_evo1_broadcast.md
# then, per model:
ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python scripts/interpretability/run_sw_broadcast_impulse.py \
    --model all --n_controls 5 --seed 42 --device cuda --out_dir results
```

Evo1 must run inside `evo2.sif`; see `docs/ENVIRONMENT.md` for the invocation.

## Artifacts

`results/secondary_dose_probe_generator_prokaryote.json`,
`results/secondary_dose_probe_dnabert2.json`
