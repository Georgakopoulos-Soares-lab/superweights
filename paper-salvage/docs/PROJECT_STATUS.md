# PROJECT_STATUS.md

**Last updated:** 2026-08-12
**Current phase:** Phase 1 — Blocking experiments (E2 halted at STEP 3)
**Blocking on:** a functional-metric decision for R3. KL is ~0 for GENERator PROK at both
doses, so the STEP 3d stop condition fired. **Prereg NOT locked, re-run NOT started** —
see `experiments/E2_evo1_broadcast/BLOCKED.md`.

> ## 🔒 E2 PREREGISTRATION LOCKED — five-model run authorised
>
> **sha256 `3d7515d0b7889f65038acd8479e85976248e7dbd0e5a701303de3a1b37dbfdc3`**
> **UTC 2026-08-13T02:59:20** · git `695eb92` · ledger `docs/prereg/LOCKS.jsonl`
>
> `verify --all` → OK. The locked file is **deliberately not edited after locking** — the
> Methods `<hash>`/`<date>` placeholders stay as written, because filling them in would
> change the file's hash and break its own verification. LOCKS.jsonl is authoritative.
>
> Pre-lock checks, all clean: no result under the new protocol had been generated
> (`sw_broadcast_impulse.json` held only pre-D-011 fixed-ε entries); `verify --all` reported
> "no locks recorded"; the diff was reviewed. The lock tool warned the working tree was
> dirty — that is `paper/main.tex` and two untracked manuscript files which predate this
> session and are not part of E2.
>
> **STEP 3d blocker resolved by D-015:** T and C primary, KL secondary/descriptive against
> each model's noise floor, no dose escalation, no metric substitution. Amendment made
> after the planned pre-lock probe and before the lock.
>
> DNABERT-2 on eager (D-014). C-014 retired (X-006), never to be resurrected. X-003 stays
> **live**. Null rule binding: headroom ≥ 4× **and** noise floor below signal.

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
| E1 | NLP cold-weight validation (row + scalar) + prospective lock | **retrospective DONE (3/3 both levels)**; prospective not started | | `prereg/PREREG_nlp_prospective.md` |
| E2 | Evo1 standardized broadcast | **LOCKED — run authorised** | | `prereg/PREREG_evo1_broadcast.md` v2, **locked `3d7515d0b788…`** |
| E3 | Bidirectional steering w/ degradation controls | not started | | `prereg/PREREG_steering.md` |
| E4 | c_{k,i} granularity decomposition, all genomic gated FFNs | **DONE — 5/5 models** | | free alongside E1 |

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
2026-08-12  |  D-014 adopt eager for DNABERT-2; C-014 retired as X-006; X-003 live.
            |  STEP 3d probe: KL flat for PROK at alpha=1.0 too (9.9e-7, zero rank shift);
            |  stop condition fired -> BLOCKED.md written, prereg NOT locked, run NOT started
            |  next: functional-metric decision for R3; meanwhile queue items 3 (E1) and 4 (E4)
2026-08-12  |  E1 retrospective arm COMPLETE. Self-test passes. Llama-7B/Mistral-7B/OLMo-7B:
            |  row rank 1/4096 all three (Level 1), published scalar index rank 1 and top-1
            |  contributor all three (Level 2). Scalar recovery claimable. C-004, C-005 ->
            |  established. Prospective arm NOT run (needs model choice + lock).
            |  next: queue item 4 (E4 granularity)
2026-08-12  |  E4 COMPLETE, 5/5 models, all shape-verified. PR: DNABERT-2 3.64, EUK 4.56,
            |  NTv3 22.7, Evo1 122, PROK 2192; NLP published SWs 1.02-1.24. C-032 supported.
            |  Two flags: PROK ranks 1289/3072 at L2 (C-001 on hold, N-009), and the
            |  uk_frobenius ntv3 adapter is wrong -- NTv3 is packed like DNABERT-2 (N-010).
            |  next: E2 metric decision (BLOCKED.md); E1 prospective arm needs model + lock
2026-08-12  |  D-015 recorded (T/C primary, KL secondary, pre-lock). Queue item 1: N-009
            |  resolved -- C-001's PROK layer IS 2 (layer-mismatch hypothesis was wrong),
            |  same checkpoint name; stored 2648.48/rank 1 does NOT reproduce (exact 5.51
            |  rank 1277, decomposed 5.47 rank 1289). EUK reproduces exactly. C-001 stays
            |  on hold. No layer search, no tuning.
            |  next: queue item 2 (N-010 shared adapter fix + tests)
2026-08-12  |  Item 2: N-010 FIXED in src/uk_frobenius.py (adapter_ntv3, shape-guarded).
            |  7 new tests, all failing under the old Llama adapter; suite green + E1
            |  self-test. NTv3 E4 recomputed via shared impl -- reproduces EXACTLY.
            |  next: queue item 3 (amend + lock E2 prereg)
2026-08-13  |  Item 3: E2 prereg amended (D-015 endpoints, confidence 3/5) and LOCKED.
            |  sha256 3d7515d0b7889f65... utc 2026-08-13T02:59:20. verify OK.
            |  next: queue item 4 (five-model run)
2026-08-13  |  Item 4 partial: FOUR models run under the locked protocol (EUK, PROK,
            |  DNABERT-2 eager, NTv3), both doses, 0 under-powered layers, noise floors all
            |  exactly 0.0. C-012 and C-013 CONFIRMED. N-006's rule fired -> C-015 restored,
            |  X-003 stays retired (N-011). matched-norm arm empty for EUK/DNABERT-2 (0 rows
            |  in +/-10% band) and failed for NTv3; band NOT widened. NTv3 T/C undefined
            |  (no downstream layers). Evo1 still running. RESULTS.md written; no branch.
            |  next: finish Evo1, then items 6 and 8
2026-08-13  |  Evo1 COMPLETE at primary dose: noise floor 0.0, headroom 26x at L13+ (matches
            |  D-013's projection), 0/20 under-powered. C +0.9516 -> +0.7044 sustained,
            |  T 6.76e-03 -> 1.20e-01. Branch C and D EXCLUDED; A vs B not assigned.
            |  Secondary dose FAILED (CUDA OOM) -- recorded, no retry (would need a code
            |  change, not a provenance-preserving retry). C-016 -> supported.
            |  next: items 6 and 8; author to choose branch A vs B
```
