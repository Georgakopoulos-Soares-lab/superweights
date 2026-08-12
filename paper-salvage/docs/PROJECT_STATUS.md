# PROJECT_STATUS.md

**Last updated:** 2026-08-12
**Current phase:** Phase 1 — Blocking experiments (E2 halted at STEP 3)
**Blocking on:** a decision about DNABERT-2's impulse noise floor. **The prereg was not
locked and the five-model re-run was not started** — see the blocker below.

> ## ⛔ BLOCKER — DNABERT-2 impulse assay is noise-dominated
>
> Two identical DNABERT-2 forward passes with **no injection** differ by KL = 0.305; the SW
> injection gives KL = 0.277. Per-layer SNR 0.93–1.00. C-014's published "largest impulse
> KL ≈ 0.31" matches the noise floor to two significant figures.
>
> GENERator EUK/PROK and NTv3 have an exactly-zero noise floor on the same harness and path,
> so this is DNABERT-2-specific (likely its Triton flash-attn kernels).
>
> Running the five-model table now would place one column of run-to-run noise beside four
> real ones, certified as adequately powered by a headroom column that measures
> representability rather than determinism. That is the D-011 error class again.
>
> Full analysis: `experiments/E2_evo1_broadcast/INSTRUMENT_VALIDATION.md`. Ledger: N-004, N-005.
> No fix attempted, nothing interpreted, no thesis sentence drafted.

---

## One-paragraph state of the project

v15 of the manuscript is not submittable. The problem is framing and internal consistency,
not absence of results. The restructure keeps the closed-form ‖U_k‖_F predictor as the
lead methodological contribution, promotes the previously-unwritten broadcast/impulse
experiment to a central section, demotes roughly half of v15 to supplement, and adds three
blocking experiments. See `PAPER_OUTLINE.md` for the frozen skeleton.

## Phase board

| Phase | Doc | State |
|---|---|---|
| 0 — Triage | `PHASE_0_TRIAGE.md` | in progress |
| 1 — Blocking experiments | `PHASE_1_BLOCKING.md` | not started |
| 2 — Writing (Evo1-independent) | `PHASE_2_WRITING.md` | not started |
| 3 — Assembly & submission | `PHASE_3_ASSEMBLY.md` | not started |

Phases 1 and 2 run **in parallel**. R1, R2, R4, R6, R7 do not depend on the Evo1 result.

## Blocking experiments (Tier 0)

| ID | Experiment | Status | Owner | Prereg |
|---|---|---|---|---|
| E1 | NLP cold-weight validation (row + scalar) + prospective lock | not started | | `prereg/PREREG_nlp_prospective.md` |
| E2 | Evo1 standardized broadcast | **HALTED at STEP 3 — see blocker** | | `prereg/PREREG_evo1_broadcast.md` **v2, NOT locked** |
| E3 | Bidirectional steering w/ degradation controls | not started | | `prereg/PREREG_steering.md` |
| E4 | c_{k,i} granularity decomposition, all genomic gated FFNs | not started | | free alongside E1 |

**Stop rule:** finish E1–E4, then re-assess before starting anything else.

## E2 status detail

**v1 protocol is void.** Two defects, one fixed:

1. *Fixed (D-012).* The harness forced `torch.autocast(float16)` for Evo1. fp16 cannot hold
   Evo1's layer-10→13 residual excursion; the run returned `KL_sw = nan` at every layer.
   Now casts everything except `poles`/`residues` to bf16. Committed separately (`0cbca69`).
2. *Not fixed by (1) — motivated D-011.* After the bf16 fix the run returned exactly `0.0`
   at every downstream layer (12–31) for the SW row **and** for the random-coordinate
   control. Fixed ε = 1.0 is below the bf16 representable increment at Evo1's residual
   scale. Instrument null, not a finding about Evo1.

**STEP 2 gate: PASSED (2026-08-12).** Layer-20 hidden states differ across inputs — 99.6%
of the 504 × 4,096 coordinates, max |Δ| = 4.4e8. The output is *not* input-independent, so
the trace is sound and the manuscript's existing residual-attribution numbers stand.
Evidence: `results/evo1_input_sensitivity_gate.json`, `results/evo1_layer_freeze_check.json`,
`logs/evo1_gate.log`, `logs/evo1_layer_freeze.log`.

The reported "bit-for-bit identical from layer 13" is a property of a *summary statistic*,
not of the hidden state. No layer is bitwise identical to any other. What is real is a
**collapse in relative change**: after the L12→L13 magnitude explosion (max |h| → 1.29e9),
consecutive-layer change falls to ~1e-6 of the residual magnitude, and blocks 21–31 alter
< 0.2% of coordinates.

**Open risk carried into STEP 3 (flagged, not acted on).** α = 0.01 gives ε ≈ 3.3e3 at the
L11 injection layer, above the bf16 increment there (≈2.0e3). But downstream of the L13
explosion the increment is ≈3.3e4 — 10× larger than the injected ε. Unless the explosion
amplifies the perturbation, the assay may return flat a second time. The prereg already
pre-commits to reporting that outcome as residual lack of dynamic range rather than as a
null about Evo1 (§Control requirement, Branch C). α is **not** to be tuned to avoid it.

## Open gaps

- **Phase 0 §0.2 migration not done.** `results/keep/` now exists but is empty. The
  provenance discipline starts with the E2 re-run; the pre-existing artifacts cited in
  `docs/CLAIMS_LEDGER.md` have not been copied in with `PROVENANCE.md` files, so several
  ledger rows still cite paths that live only in the old tree. Backfill is Phase 2 work.
- **Prereg v1 text is not recoverable from git.** See `docs/prereg/PREREG_evo1_broadcast.md`
  §v1 provenance. v1 was untracked when it was replaced, so no commit ever contained it.

## Submission blockers (independent of science)

- [ ] Figure 2 panels A–D are absent; v15 caption admits they are "pending GENERator JSONs"
- [ ] Abstract/Results contradiction on pruning tolerance ("shadow redundancy")
- [ ] "Eight models" oversell vs. actual coverage asymmetry
- [ ] NTv3 post-hoc MCC metric choice
- [ ] Verify ref [11] Sun et al. arXiv ID (2603.05498 does not match the known
      massive-activations paper 2402.17762)
- [ ] Typo sweep ("extremelly")

## Open questions

1. Does T predict criticality? — resolved by E2.
2. Is the genomic causal object a scalar, a row, or an ensemble, and does it differ
   between GENERator and DNABERT-2? — resolved by E4.
3. Does the SW track the raw corpus prior or deviation from a Markov expectation? —
   conditional, only if R4's interpretation stays unclear after E3.

## Session log

Append one line per working session: date, what changed, what's next.

```
YYYY-MM-DD  |  scaffold created; outline frozen  |  next: run PHASE_0 inventory
2026-08-12  |  E2 v1 declared void (D-011, D-012); prereg v2 installed; bf16 harness fix
            |  committed separately (0cbca69); C-012/13/14 -> pending-rerun; evo1 results
            |  key -> evo1_fixed_eps_SUPERSEDED; container hash recorded in ENVIRONMENT.md;
            |  STEP 2 input-sensitivity gate PASSED
            |  next: relock prereg v2, then re-run all five models at alpha=0.01
2026-08-12  |  STEP 2b restructure done; harness amended to fp32 + AC-eps + headroom +
            |  matched-norm arm; instrument validation found DNABERT-2 impulse assay is
            |  noise-dominated (N-004). Prereg NOT locked, re-run NOT started.
            |  next: decide DNABERT-2 determinism before locking
```
