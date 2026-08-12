> # ⚠ RECONSTRUCTION — NOT A PREREGISTRATION RECORD
>
> This is **not** a recovered artifact. No commit, blob, hash or timestamp for prereg v1
> exists — see `../PREREG_evo1_broadcast.md` §"Result of that recovery: v1 was never
> committed". The `paper-salvage/` tree was untracked until commit `e705497`, and v1 was
> overwritten in the working tree before that commit was made.
>
> The text below was transcribed from the working session in which the v1 file was read,
> immediately before it was replaced. It is offered so the prior protocol is legible, and
> for no other purpose.
>
> **Do not cite this as evidence that anything was predicted in advance.** Do not lock it.
> Do not reference it from Methods except as a disclosed absence.
>
> Reconstructed: 2026-08-12. Original file: 2,757 bytes, mtime 2026-08-12 12:58.

---

# PREREG — E2, Evo1 standardized broadcast

**Lock before running:** `python src/prereg_lock.py lock docs/prereg/PREREG_evo1_broadcast.md`

---

## Question

Does the Evo1 candidate row show low total broadcast transmission (T), and does that
explain its null ablation phenotype?

## Background

| Model | T | C | Criticality |
|---|---|---|---|
| GENERator EUK | high | 1.0 end-to-end | catastrophic (+23,026% PPL) |
| GENERator PROK | grows toward output | 0.92 → ~0 by L14 | catastrophic (+25,975% PPL) |
| DNABERT-2 | largest KL (≈0.31) | ≈0 immediately | strongest encoder phenotype |
| Evo1 | **unknown** | attenuates/redistributes | null (ΔPPL 0.0%) |

Across the three measured positives, criticality tracks T while C varies from 1.0 to ~0.
Evo1 is therefore the decisive single-variable test.

## Design

- Injection: ε = 1.0 into the residual stream at the SW row (3776) / SW token position at
  the source layer.
- Tracking: all 32 StripedHyena blocks. bf16.
- Metrics: T, C, output KL — identical definitions and code path as the other four models.
- Control arms, all four, matched to the other models:
  1. random coordinate
  2. neighbouring row
  3. random dense direction
  4. matched-norm

**Nothing about the protocol may be adjusted after seeing Evo1's numbers.** If a protocol
change is unavoidable, log it in DECISIONS.md and re-run the other four models identically.

## Predictions (state before running)

**Primary:** Evo1 T will be low relative to GENERator EUK/PROK and DNABERT-2, consistent
with the conv mixer redistributing the signal across coordinates rather than transmitting
it. _(Confidence: ___ / 5)_

**Secondary:** C will decay faster than PROK's, consistent with the observed ~44%
per-block survival in the existing residual trace.

**Falsifier:** Evo1 T is comparable to or greater than DNABERT-2's while ablation remains
null. This falsifies "T predicts criticality."

## Pre-committed reporting

**If T is low →** R3 thesis sentence:
> Total broadcast gain T, not coordinate preservation C, predicts criticality; C describes
> routing style.

Two-axis taxonomy; R3 becomes mechanistic; the genomic nulls become correct predictions.

**If T is high and ablation is null →** R3 thesis sentence:
> Neither T nor C alone is sufficient; criticality additionally requires that the routed
> signal reach a readout the task depends on.

This is reported, not buried. It needs a logit-attribution follow-up — log it as a Phase 1b
candidate; **do not start it inside Phase 1.**

**If the run fails technically →** report as not completed in Limitations, with the reason.
Do not substitute the existing informal trace and present it as the standardized protocol.

---

_Locked: (filled by prereg_lock.py)_ — never filled; see the reconstruction notice above.
