# PROJECT_STATUS.md

**Last updated:** 2026-08-14 (E6 Stage A — fourth session, direct continuation of the E5 Gate-0
session)
**Current phase:** A second new experiment (E6 — independent confirmation of exact
cross-pathway interaction geometry, `experiments/E6_cross_geometry/`) was opened this session,
built on E5's finding but tested on a panel frozen from evidence that predates E5 entirely
(additional Yu et al. Table 2 rows, this repo's own pre-existing activation-detection sweep).
Its Stage A ran to completion and was assigned **Branch C** (no independent replication) by
its own locked, mechanical decision rule — but the miss was on effect-size *magnitude* only
(0.28% short of the pre-committed margin), not on *direction* (complete separation, exact
p = 0.0035). Full nuance: `experiments/E6_cross_geometry/RESULTS.md`. Per Branch C's governing
rule, E6 stops here — no causal follow-up, no new metric, no added models. This does not
change anything from the 2026-08-13 reconciliation pass below: R1/R6 are still the draftable
sections, R2–R5 are still blocked, and the manuscript was not touched by either E5 or E6.

**Blocking on:** author review of three flagged provenance conflicts (N-013, N-014, N-015 in
`CLAIMS_LEDGER.md`) and author decisions on the pre-existing open items below (Evo1 Branch A
vs B, N-009/C-001, the missing Evo1 secondary-dose KL, empty matched-norm arms) — none of
which changed this session. **New this session:** E6 recommends C-002/C-003 be **retained**
(row-ranking survives independently under the exact form on 12 of 13 confirmation-panel rows)
and C-032 be **held** (E5's measurement-validity flag is neither confirmed nor refuted by
E6's near-miss result) — both are recommendations only; no `CLAIMS_LEDGER.md` row was edited.

> ## 🔒 E2 PREREGISTRATION LOCKED — unchanged, re-verified this session
>
> **sha256 `3d7515d0b7889f65038acd8479e85976248e7dbd0e5a701303de3a1b37dbfdc3`**
> **UTC 2026-08-13T02:59:20** · git `695eb92` · ledger `docs/prereg/LOCKS.jsonl`
>
> `python3 src/prereg_lock.py verify --all` → **OK**, re-checked at the start and end of this
> session. The lock is untouched by this pass; no impulse/broadcast work was performed.

---

## What changed this session (2026-08-13, reconciliation pass)

A downstream review asserted a substantially revised scientific state (new headline results:
a DNABERT-2 redundant pair, a GENERator attention-sink phenotype, causal GC steering, an Evo1
numerical-saturation diagnosis, an NTv3 truncation-bug retraction, a PROK contamination
diagnosis). Every one of these was traced against the actual repository and **none has a
supporting artifact** — three are additionally in direct conflict with artifacts that already
exist. Full trace: `docs/NEW_DIRECTION_EVIDENCE_AUDIT.md`. Consequences, in order:

1. `CLAIMS_LEDGER.md` reconciled: three claims (C-020, C-021's PROK half, C-029) downgraded
   to a new `contested` status (real evidence exists, but a later instruction calls it into
   question with no artifact of its own — flagged for the author, not resolved either way).
   One new claim added (C-033, quantization scale-endpoint math, established directly from
   code with no run). See `CLAIMS_LEDGER.md` N-012–N-015 for the reasoning behind each.
2. `DECISIONS.md` D-016–D-023 appended (append-only; nothing rewritten). New thesis and
   R1–R6 outline adopted; U_k demoted from headline to calibration signature; broadcast/T-C
   work confirmed supplementary-only; Evo1, NTv3, and PROK's contested status recorded with
   rationale; no-new-experiments instruction recorded.
3. `PAPER_OUTLINE.md` rewritten (v2) around the new R1–R6 skeleton. R2–R5 are marked
   **BLOCKED, no artifact** and must not be drafted until real evidence exists. Only R1
   (reframed, U_k as calibration) and R6 (quantization granularity, C-033) can be drafted now.
4. `paper-salvage/CLAUDE.md` rewritten: new thesis, new hard-constraint section listing every
   claim that must not be written without a real artifact.
5. `docs/MANUSCRIPT_MIGRATION_MAP.md` created: every major section/figure in `paper/main.tex`
   (still the v6 manuscript, untouched by this pass) mapped to KEEP / REWRITE / MOVE TO
   SUPPLEMENT / DELETE / BLOCKED BY PROVENANCE.
6. `docs/PAPER_READINESS.md` created: sufficiency call for R1–R6. Verdict on whether any
   further experiment is justified before writing — see that file; short answer is no.
7. No provenance migration into `results/keep/` this session — no new claim met the
   established-with-a-real-evidence-path bar except C-033, which is a code fact, not a
   results artifact, so there is nothing to copy into `results/keep/` for it.

**Nothing about the pre-2026-08-13 state changed except the four ledger-status edits above.**
E1/E2/E4 results, the prereg lock, and every previously-established claim are untouched.

---

## What changed this session (2026-08-13, E5 Gate 0 — third session of the day)

A new, narrower hypothesis was scoped: does the *within-row* structural dimensionality of a
gated-FFN high-gain row (top-1 share / participation ratio of the diagonal `c_{k,i}`
decomposition) reflect an unusually strong multiplicative alignment between the down-
projection term and the gated-amplifier term in NLP specifically, versus the clean genomic
models? This is explicitly **not** a re-test of row-level recovery, which the immediately
preceding feasibility audit (this session, earlier) already showed is strong in both groups.

New, self-contained experiment directory: `paper-salvage/experiments/E5_dimensionality/`
(own README with the premise correction, own prereg, own library + tests, does not modify
`results/`, `CLAIMS_LEDGER.md`, `PAPER_OUTLINE.md`, or `CLAUDE.md`).

1. Premise correction recorded in `experiments/E5_dimensionality/README.md`: row-level
   recovery is rank 1/N in 5 of 6 primary-panel models (Evo1 48/4096, excluded from this
   panel anyway); the real contrast is within-row granularity; NLP has published scalar
   ground truth genomic models lack; DNABERT-2's L7 706/768 finding is a separate local
   result, not cited here.
2. `docs/prereg/PREREG_dimensionality_gate0.md` locked (sha256 `7e508315...`, UTC
   2026-08-13T18:13:24Z) after `dimensionality_lib.py`'s 12 synthetic-tensor tests passed
   green (cross-validated exactly against the existing `uk_frobenius` predictor and a
   from-scratch brute-force `U_k` materialization). No real checkpoint weight was touched
   before the lock.
3. Gate 0 ran on the six-model primary panel (Llama-7B, Mistral-7B, OLMo-7B, GENERator EUK,
   DNABERT-2, NTv3 — PROK and Evo1 excluded per the prereg's fixed exclusion list) and
   **stopped at §0B**, the preregistered exactness-vs-diagonal robustness check: cross terms
   in the exact quadratic form exceed the locked 20% threshold for all three genomic models
   (GENERator EUK 82.9%, DNABERT-2 50.7%, NTv3 20.7%) and none of the three NLP models
   (6.7%–19.3%). Per the prereg's own mechanical stop rule, §0C (factor decomposition) and
   §0D (permutation test) were never run on any real weight. Full numbers and the interpretive
   flag for existing claims C-002/C-003/C-032: `experiments/E5_dimensionality/GATE0_RESULTS.md`.
4. **Gate 1 is not authorized.** No replacement hypothesis was generated in this pass, per
   the governing instruction for this experiment.

---

## What changed this session (2026-08-14, E6 Stage A — fourth session, continuation)

E5's discovery-panel cross-term split (six rows, chosen as the E5 primary panel) suggested a
narrower, testable hypothesis: are NLP high-gain rows approximately coordinate-separable
while genomic ones derive real strength from coherent cross-pathway interactions? Because the
hypothesis was generated *after* seeing E5's six rows, those six do not count as evidence for
it — a genuinely independent panel was required.

New, self-contained experiment: `paper-salvage/experiments/E6_cross_geometry/`.

1. **Task 1 — panel frozen before any cross-term value was computed.**
   `experiments/E6_cross_geometry/CONFIRMATION_PANEL.md` selects 13 candidate rows using only
   evidence on record in git history *before* E5's lock: three additional OLMo-7B rows from
   Yu et al.'s own Table 2 (layers 2, 7, 24, all sharing output row 269 — the E5 primary was
   layer 1), and ten genomic rows from this repository's pre-existing activation-detection
   sweep (`results/super_weight_index.json`, added May 2026) — nine additional DNABERT-2 rows
   and one additional GENERator EUK row. GENERator PROK (both checkpoints) and Evo1 are
   mechanically excluded, one of them (the 1.2B PROK variant) doubly so — its checkpoint
   isn't even locally cached. The panel's own limitation (the NLP arm is one model's
   recurring channel across layers, not three independent models) is disclosed in the panel
   document itself, before any measurement.
2. **Task 2 — `cross_geometry_lib.py`** implements the exact pairwise cross-term
   decomposition (`X_ij`, positive/negative cross mass, a single preregistered `kappa`
   coherence statistic, and a layer-wide `f_cross` reference distribution), reusing E5's
   `exact_uk_all_rows` and `uk_frobenius.uk_frobenius` unmodified. 7/7 synthetic tests green
   (orthogonal / constructive / cancellation cases matched to hand computation), plus a
   pre-lock regression check reproducing all six of E5's published `f_cross` values to
   ~1e-14 relative error.
3. **Task 3 — `docs/prereg/PREREG_cross_geometry_stageA.md` locked** (sha256 `4cbe4416...`,
   UTC 2026-08-14T12:08:49+00:00) with three primary endpoints and a three-step mechanical
   decision tree (Q1 replication → Q2 layer-specificity → Q3 coherence), each threshold
   (1.5× margin, 75th-percentile, kappa ≥ 0.3) chosen and justified before any panel weight
   was loaded.
4. **Task 4 — Stage A ran to completion** on all 13 candidates, weight-only.
5. **Task 5 — mechanical decision: Branch C.** Complete separation held (every genomic
   `f_cross` exceeds every NLP `f_cross`; exact non-asymptotic rank-sum p = 0.0035) but the
   pre-committed 1.5× effect-size margin narrowly failed — the genomic minimum (DNABERT-2
   L7/r603, `f_cross` 0.1134) fell 0.28% short of the required 0.1137. That specific row was
   already flagged as structurally atypical in this repository's own ledger (C-010,
   "propagator not source") before E6 existed. Per Branch C's rule, **the row was not
   excluded and the decision was not recomputed** — the near-miss is reported, not rescued.
   Full numbers, the descriptive (non-decision) Q2/Q3 values, and the reasoning: `experiments/
   E6_cross_geometry/RESULTS.md`.
6. **Task 6 (causal-feasibility audit) was not written** — it is conditional on Branch A
   only. **E6 stops here per its own governing rule:** no alternative metric, no threshold
   change, no added models, no causal experiment.
7. **Claim-ledger recommendations (not enacted):** C-002/C-003 (row-ranking) — **retain**,
   with a Methods clarification distinguishing row-level from scalar-level recovery;
   row-ranking under the exact form held on 12 of 13 confirmation-panel rows, not just E5's
   six. C-032 (granularity/PR) — **hold**; E5's flag is neither confirmed nor refuted by a
   near-miss result, and this pass does not invent a cross-term-aware replacement metric.

---

## One-paragraph state of the project

The manuscript (v6, `paper/main.tex`) is written around a thesis (`‖U_k‖_F` as headline
predictor) this project has now retired. It has not been rewritten to match the new thesis —
this pass produced the audit, ledger, decisions, outline, and migration map that a writing
session needs, but performed no prose rewrite itself (out of scope; see
`docs/MANUSCRIPT_MIGRATION_MAP.md` Task note). Two of six new-outline sections (R1, R6) have
enough real evidence to draft; four (R2–R5) do not and must not be drafted until their
artifacts exist. See `PAPER_OUTLINE.md` for the frozen v2 skeleton.

## Phase board

| Phase | Doc | State |
|---|---|---|
| 0 — Triage (v1) | `PHASE_0_TRIAGE.md` | historical; superseded by the 2026-08-13 reconciliation for framing purposes, but its mechanical items (figure JSONs, ref [11]) are still open and unaffected |
| 1 — Blocking experiments (v1) | `PHASE_1_BLOCKING.md` | E1 retrospective, E2, E4 done; E1 prospective is **no longer required** (D-017); E3 not started and not blocking under the new outline (R5 is simply not drafted without it) |
| Reconciliation (this session) | `docs/NEW_DIRECTION_EVIDENCE_AUDIT.md`, `docs/MANUSCRIPT_MIGRATION_MAP.md`, `docs/PAPER_READINESS.md` | **done** |
| Writing (v2, next) | not yet created | R1 and R6 draftable now; R2–R5 blocked on evidence |

## Tier-0 experiments (v1 framing, retained for historical accuracy — no longer the active
Tier-0 list)

| ID | Experiment | Status | Prereg |
|---|---|---|---|
| E1 | NLP cold-weight validation (row + scalar) + prospective lock | retrospective DONE (3/3 both levels); prospective **retired as a requirement**, D-017 | `prereg/PREREG_nlp_prospective.md` |
| E2 | Evo1 standardized broadcast | LOCKED, run, reported; results supplementary-only under the new outline (D-018) | `prereg/PREREG_evo1_broadcast.md` v2, locked |
| E3 | Bidirectional steering w/ degradation controls | not started; not run this session; R5 stays BLOCKED without it | `prereg/PREREG_steering.md` (unlocked) |
| E4 | c_{k,i} granularity decomposition | DONE — 5/5 models | free alongside E1 |

**No Tier-0 list is active for the next session.** Do not treat E3, an Evo1 secondary dose, a
PROK layer search, or an NTv3 retrain as queued — see `NEXT_SESSION.md`.

## Open gaps (unchanged from before this session, still open)

- Phase 0 §0.2 migration: `results/keep/` has E1/E2/E4 batches (Phase 0 backfill, prior
  session) plus nothing new from this session (see item 7 above).
- Prereg v1 text is not recoverable from git (unchanged, historical).

## Submission blockers (independent of science, unchanged this session — none of these were
touched)

- [ ] Figure 2 panels A–D — source JSONs located; render blocked on N-009 (PROK panel would
  come from the unreproducible artifact). See `docs/REFERENCE_AUDIT.md`.
- [ ] Abstract/Results still assert the retired "shadow redundancy" claim verbatim
  (`paper/main.tex:111-113`) — flagged again in `docs/MANUSCRIPT_MIGRATION_MAP.md`, not fixed
  this session (no manuscript prose was rewritten).
- [ ] "Eight models" oversell language is still present in `paper/main.tex` (abstract line 76,
  intro line 140/171/176, Figure 1 caption line 238) — flagged in the migration map.
- [ ] NTv3 post-hoc MCC metric choice (unchanged) — now compounded by C-029's `contested`
  status (N-014); both need author attention together.
- [ ] Verify ref [11] Sun et al. arXiv ID — still flagged, needs external lookup, not
  resolved this session (`docs/REFERENCE_AUDIT.md`).
- [x] Typo sweep (prior session, unchanged).

## Open questions

### Pre-existing (unchanged this session)

1. Does T predict criticality? — E2 measured; Branch A vs. B is an open author call, now
   explicitly **not to be decided in a writing-focused session either** (D-018) unless a
   future instruction reopens it.
2. Is the genomic causal object a scalar, a row, or an ensemble? — E4 measured
   (`experiments/E4_granularity/CANONICAL_TABLE.md`); DNABERT-2's real, established ensemble
   answer (C-027/C-028) stands; whether a *pair* sub-structure exists within it is now an
   explicitly open, unevidenced question (R2/R3, blocked).
3. Does the SW track raw corpus prior or deviation from a Markov expectation? — conditional,
   only if R4 (old) is ever revived; not applicable to the current R1–R6 outline as drafted.

### New this session

4. **N-013/N-014/N-015** — three claims (`contested`) where a later instruction and this
   repository's own evidence directly disagree. Not resolved by this pass. Each needs either
   (a) the missing artifacts located/added, or (b) an explicit author decision to retract the
   existing claim on the strength of the instruction alone, logged as a new DECISIONS.md
   entry.
5. Whether R2–R5 are ever written depends entirely on whether their underlying experiments
   get run in a future, explicitly-instructed session. This pass does not queue them.

## Session log

Historical entries (through 2026-08-13, first session of the day) are preserved verbatim
below, unchanged.

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
2026-08-13  |  Item 6 DONE: canonical E4 table at experiments/E4_granularity/
            |  CANONICAL_TABLE.md, built from stored artifacts only (no model loads).
            |  PROK row included at its own layer 2 and marked CONTESTED / C-001 ON HOLD;
            |  no other layer examined. Unavailable cells marked n/s, not recomputed.
            |  next: item 7c (mechanical cleanup), item 8 (provenance backfill)
2026-08-13  |  Items 7c and 8 DONE. 7c: "extremelly" fixed; Methods precision disclosure
            |  added for Evo1 bf16 and DNABERT-2's Triton fp16 attention cast (D-012/D-014/
            |  N-008); PHASE_1 E2 design annotated as superseded; stale PROJECT_STATUS state
            |  corrected (X-003 retired, not live). Manuscript BUILD NOT RUN -- no LaTeX
            |  toolchain on this node. 8: 5 established claims migrated to results/keep/
            |  with PROVENANCE.md; 12 logged in UNMIGRATED.md (empty evidence paths).
            |  QUEUE COMPLETE. Remaining work is author judgment -- see NEXT_SESSION.md 5.
2026-08-13  |  RECONCILIATION PASS (second session of the day). A downstream review handed
            |  down six "new headline results" and six "rejected claims" against a claimed
            |  new scientific state. Audited every one against actual repo artifacts
            |  (docs/NEW_DIRECTION_EVIDENCE_AUDIT.md): none of the six new headline results
            |  has any supporting artifact; three (Evo1 2^24, NTv3 p=0.48, PROK contamination)
            |  directly conflict with existing measured evidence. No experiment run. Ledger
            |  reconciled (3 claims -> contested, 1 new claim C-033 from direct code
            |  inspection); DECISIONS D-016-D-023 appended; PAPER_OUTLINE.md rewritten to
            |  v2 (R1-R6, U_k demoted, R2-R5 marked BLOCKED pending evidence, R6 draftable
            |  now); CLAUDE.md rewritten; MANUSCRIPT_MIGRATION_MAP.md and PAPER_READINESS.md
            |  created. Prereg lock re-verified OK, untouched.
            |  next: see NEXT_SESSION.md -- writing R1/R6, or resolving N-013/014/015.
2026-08-13  |  E5 Gate 0 opened and STOPPED at 0B. New experiment dir
            |  experiments/E5_dimensionality/; prereg locked (sha256 7e508315...); 12/12
            |  synthetic tests green pre-lock. Confirmatory run on 6-model panel (Llama-7B,
            |  Mistral-7B, OLMo-7B, GENERator EUK, DNABERT-2, NTv3): exact-vs-diagonal
            |  cross-term fraction f_cross exceeds the locked 0.20 stop threshold for all
            |  three genomic models (EUK 0.829, DNABERT-2 0.507, NTv3 0.207) and none of the
            |  three NLP models (0.067-0.193). 0C/0D never run on real weights, per the
            |  prereg's own mechanical rule. Gate 1 NOT authorized. Interpretive flag (not
            |  acted on): existing C-002/C-003/C-032 granularity numbers rest on the same
            |  diagonal decomposition now shown unreliable specifically for the genomic side.
            |  next: author call on whether E5 continues (a cross-term-aware granularity
            |  metric would need its own new preregistration) or the R1/R6 writing /
            |  N-013/014/015 resolution work from the prior session resumes.
2026-08-14  |  E6 Stage A: STOPPED at Branch C (near-miss, not clean fail). New experiment
            |  dir experiments/E6_cross_geometry/; panel of 13 rows frozen from pre-E5
            |  evidence (Yu Table 2 + this repo's own pre-existing detection sweep); prereg
            |  locked (sha256 4cbe4416...); 7/7 synthetic tests + 6/6 E5-regression checks
            |  green pre-lock. Confirmatory run: complete separation held (genomic f_cross
            |  0.11-0.83 all exceed NLP f_cross -0.004-0.076; exact p=0.0035) but the locked
            |  1.5x margin missed by 0.28% (DNABERT-2 L7/r603, already flagged atypical by
            |  C-010, pulled the genomic minimum down). Row not excluded, decision not
            |  recomputed, per Branch C's own rule. No Task 6, no new metric, no added
            |  models. Recommend (not enacted): C-002/C-003 retain w/ Methods clarification;
            |  C-032 hold.
            |  next: author call on whether a freshly-independent, separately-preregistered
            |  E6 continuation (larger/more independent NLP arm, or a principled cross-term-
            |  aware granularity metric) is worth running, or the R1/R6 writing /
            |  N-013/014/015 resolution work resumes instead.
```
