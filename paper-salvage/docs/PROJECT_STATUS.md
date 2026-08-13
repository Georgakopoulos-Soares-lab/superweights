# PROJECT_STATUS.md

**Last updated:** 2026-08-12
**Current phase:** Phase 1 — Blocking experiments (E2 halted at STEP 3)
**Blocking on:** a decision about which attention kernel is DNABERT-2. **The prereg was not
locked and the five-model re-run was not started** — see the blocker below.

> ## ⛔ BLOCKER — DNABERT-2 attention kernel: determinism fixed, guard failed
>
> **Determinism: solved.** Attempt 1 of the STEP 3a ladder worked — forcing the eager
> PyTorch attention path gives an **exactly zero** noise floor. Attempts 2 and 3 not needed.
> The real DNABERT-2 impulse signal is ‖Δh‖ = 2.4e-3, KL = 1.8e-8 — about 1,800× smaller
> than the 4.50 / 0.277 previously measured, which was entirely noise. C-014's "largest
> impulse KL ≈ 0.31" does not survive.
>
> **Guard: FAILED.** MLM perplexity, same weights and inputs, kernel the only difference:
> Triton/fp16 = 687.9 (nondeterministic, 8.48% spread across repeats) vs eager/fp32 = 176.9
> (0.000% spread). Δ = **−74.3%** against ±1%. Per the standing rule this is reported, not
> adopted, so the fix is **not** in force for the re-run.
>
> The Triton path silently casts qkv to fp16 and its own perplexity varies by 8.5%, so it is
> not a fixed quantity to match against; the eager path is 3.9× better and bit-reproducible.
> That suggests Triton is the defective path — but deciding which path *is* DNABERT-2 is a
> substantive call and is not made here.
>
> **Unaudited scope:** the loaded config has `attention_probs_dropout_prob = 0.0`, so Triton
> was the *default* inference path. Every inference-time DNABERT-2 analysis went through it.
> Whether C-002, C-010, C-027 or C-028 are affected has not been checked.
>
> Ledger: N-004, N-006, N-007, N-008. Detail:
> `experiments/E2_evo1_broadcast/INSTRUMENT_VALIDATION.md`.
> Everything else in STEP 3b/3c is complete; the prereg is ready to lock once this is decided.

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
2026-08-12  |  STEP 3a: eager attention gives zero noise floor, but perplexity guard FAILS
            |  (-74.3%, N-007). STEP 3b/3c complete: matched-norm redefined as equal-norm
            |  random row, dual dose (a=0.01 T/C, a=1.0 KL), sanity expectation corrected,
            |  v1 declared not-preregistered, reconstruction deleted (N-008 fp32 mislabel).
            |  next: decide DNABERT-2 kernel, then lock and run
```
