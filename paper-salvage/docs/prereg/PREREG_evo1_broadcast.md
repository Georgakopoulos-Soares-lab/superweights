# PREREG — E2, Evo1 standardized broadcast — **v2 (AMENDED)**

**Amendment notice.** This supersedes v1, which was locked before the first run and whose
protocol proved to have no dynamic range on Evo1. The v1 prediction is preserved verbatim
below and is **not** revised in light of what was observed. See D-011, D-012.

**Lock before running:** `python src/prereg_lock.py lock docs/prereg/PREREG_evo1_broadcast.md`
`prereg_lock.py` will warn that the file changed after a previous lock. That warning is
correct and the relock must be disclosed in Methods.

---

## What v1 predicted, and what happened

> **v1 primary prediction:** Evo1 T will be low relative to GENERator EUK/PROK and
> DNABERT-2, consistent with the conv mixer redistributing the signal across coordinates
> rather than transmitting it.
>
> **v1 falsifier:** Evo1 T is comparable to or greater than DNABERT-2's while ablation
> remains null.

**Outcome:** neither. Every downstream layer (12–31) returned exactly 0.0 for the SW row
**and** for the random-coordinate control. Evo1's residual is effectively unnormalised —
median |h| across 4,096 dims ~2.8e5 at L11, ~4.2e6 from L13 onward, max ~1.29e9. bf16 ULP
at 4.2e6 is ~3.3e4. A fixed ε = 1.0 is rounded away at injection for any coordinate. The
result is uninterpretable: the control is as flat as the treatment.

A separate defect was fixed first — the harness forced fp16 autocast for Evo1, producing
`KL_sw = nan` throughout (D-012).

## Precondition (GATE — settle before spending compute)

The residual was reported as bit-for-bit identical from layer 13 onward. If that holds of
the **hidden state** rather than of a summary statistic, the output is input-independent
from L13, which is incompatible with Evo1's PPL of 3.11.

**Test:** forward two clearly different input sequences; diff the layer-20 hidden state.

- **Differ** → the median statistic is stable while per-coordinate values vary. An AC
  component exists. Proceed to the amended protocol.
- **Identical** → the trace is broken. **Stop.** No measurement through this trace is
  interpretable, including the residual-attribution numbers already in the manuscript.
  Log as a blocker in PROJECT_STATUS.md and fix the trace before any re-run.

## Amended protocol

Injection magnitude scaled to the AC component at the injection layer:

    ε_m = α · std( h_ℓ − mean(h_ℓ) ),   α = 0.01

α is fixed at 0.01 for all models and is not tuned per model.

Everything else is unchanged from v1: injection at the SW row / SW token / **source layer**
(Evo1 row 3776, L11); tracked across all downstream layers; metrics T, C, KL; all four
control arms — random-coordinate, neighbouring-row, random-dense-direction, matched-norm.

**Applied to all five models.** GENERator EUK, GENERator PROK, DNABERT-2, NTv3, Evo1. Only
re-run numbers are reported in R3.

**Rejected alternatives, recorded so they are not revisited:**
- ε scaled to |h[k]| or to median |h| — tracks the DC offset, the component carrying no
  input-dependent signal.
- fp32 or fp64 injection alone — fp32 recovers ~4 orders of magnitude but ε = 1.0 still
  rounds away at the SW coordinate's scale; fp64 measures in a regime the model does not
  compute in.
- Injection before the layer-10 scale explosion — Evo1's source layer *is* L11, so this
  measures a non-source layer and breaks source-layer symmetry with the other four models.
  Reconsider only if the amended protocol also returns flat.

> Note on the second rejected alternative: it rejects *fp32 alone*, i.e. fp32 while keeping
> ε = 1.0. D-013 adopts fp32 **together with** AC-relative ε. These are not in conflict —
> the rejection stands as written and is not revised.

## Numerical precision (added per D-013)

The impulse trace runs in **fp32** for all five models. α remains fixed at 0.01 and is not
tuned per model. Predictions and reporting branches are unchanged.

Every output record carries a per-layer headroom column:

    headroom_ℓ = ε_effective_ℓ / ULP( max |h_ℓ| )

- headroom ≥ 4× — the layer is adequately powered; a null there is a measurement.
- headroom < 4× — the layer is under-powered; report as **unmeasured**, not as zero.

A flat result is interpretable only where headroom is adequate. This column is required for
every model and every control arm, and it is what allows Branch C to be earned rather than
asserted.

## v1 provenance (added per D-013)

v1 of this preregistration was version-controlled but not machine-locked; `LOCKS.jsonl` did
not exist at the time. Its provenance is the git commit that introduced it:

    commit: ____________________   date: ____________________
    (git log --follow --diff-filter=A -- docs/prereg/PREREG_evo1_broadcast.md)

This is disclosed in Methods. No retroactive LOCKS.jsonl entry is synthesised for v1.

### Result of that recovery: v1 was never committed

**The block above cannot be filled, and the premise that v1 "was version-controlled" is
false.** The recovery command returns no commits:

```
$ git log --follow --diff-filter=A -- paper-salvage/docs/prereg/PREREG_evo1_broadcast.md
(no output)

$ git log --format="%h %ad %s" --date=short -- \
      paper-salvage/docs/prereg/PREREG_evo1_broadcast.md \
      paper-salvage/PREREG_evo1_broadcast.md
e705497 2026-08-12 E2 bookkeeping: void the fixed-eps protocol, install prereg v2, …
```

`e705497` is the only commit that has ever contained this path, and it already contains
**v2**. The entire `paper-salvage/` tree was untracked until that commit. v1 existed only
as an untracked working-tree file and was overwritten in place by the STEP 1 instruction
"replace `PREREG_evo1_broadcast.md` with the v2 text" before any commit was made. No blob
of v1 exists in the object store.

**What this costs.** v1's status as a *preregistration* rests on nothing verifiable. There
is no hash, no commit, and no timestamp establishing that its prediction was fixed before
the first Evo1 run. Methods must not describe v1 as preregistered.

**What survives, and why it is weaker.** v1's primary prediction and falsifier are quoted
verbatim in §"What v1 predicted, and what happened" above — but that quotation lives inside
v2, which was written *after* the first run. It is a post-hoc transcription of a
pre-hoc claim, and it cannot be distinguished from one by any external check.

A reconstruction of the full v1 text, recovered from the working session in which it was
read before being overwritten, is filed at `archive/PREREG_evo1_broadcast_v1_RECONSTRUCTED.md`.
It is labelled as a reconstruction. **It is not evidence and must not be cited as a
preregistration record.**

**Bearing on v2.** None. v2 is being locked *before* its confirming run, with `LOCKS.jsonl`
now in place, which is the property v1 turned out to lack. The v1 failure is disclosed, not
repaired.

## Predictions (state before running)

**Sanity, all models except Evo1:** α = 0.01 should land near ε = 1.0 for GENERator,
DNABERT-2 and NTv3, whose activations are O(1–100) under normalisation. T/C/KL should
closely reproduce the fixed-ε values. Any material divergence is itself a finding and must
be reported, not silently absorbed.

**Evo1 primary:** _(unchanged from v1)_ T low relative to the other four.
_(Confidence: ___ / 5)_

**Evo1 secondary:** C decays faster than PROK's, consistent with the ~44% per-block
survival in the existing residual trace.

**Evo1 falsifier:** T comparable to or greater than DNABERT-2's while ablation remains null.

**Control requirement:** the random-coordinate arm must now be non-flat. If treatment and
control are both flat again, the assay still lacks dynamic range and the result is again
uninterpretable — report as such, do not present it as a null finding about Evo1.

## Pre-committed reporting

**Branch A — Evo1 T low, controls non-flat.** R3 thesis sentence:
> Total broadcast gain T, not coordinate preservation C, predicts criticality; C describes
> routing style.

**Branch B — Evo1 T high, ablation still null.** R3 thesis sentence:
> Neither T nor C alone is sufficient; criticality additionally requires that the routed
> signal reach a readout the task depends on.

Requires a logit-attribution follow-up. Log as a Phase 1b candidate; **do not start it
inside Phase 1.**

**Branch C — precision floor persists (both arms flat under AC-relative ε).**
Report as an architecture-level result, not an assay failure, but **only if the precondition
passed**:
> In Evo1 the residual stream carries a DC offset several orders of magnitude above the
> input-dependent component, such that channel-aligned perturbation at the super-weight
> coordinate falls below the representable precision of the model's own numerical format.

This requires an AC/DC decomposition to earn: AC magnitude at the SW coordinate relative to
DC, and evidence that downstream computation depends on the AC component. That is a figure
panel, not a footnote, and it is a new experiment — log it as Phase 1b, do not improvise it
during the re-run.

**Branch D — precondition failed.** No branch. Fix the trace; re-scope.

---

_v1 locked: (see LOCKS.jsonl)_
_v2 locked: (filled by prereg_lock.py)_
