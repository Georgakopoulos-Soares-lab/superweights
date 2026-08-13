# NEXT_SESSION.md

**Session:** 2026-08-13, reconciliation pass (second session of the day).
**Status: this is a writing/consolidation handoff, not an experiment queue.** No model was
loaded, no GPU job was launched, no script was run this session. Everything below is either
"write prose against evidence that already exists" or "close a provenance gap by finding or
producing the missing artifact" — not "run a new analysis."

---

## 1. What happened this session

A downstream review asserted a substantially revised scientific state (six new headline
results, six claim retractions). Every one of the six new headline results was traced against
the actual repository and **none has a supporting artifact** — three (Evo1's "2^24
saturation," NTv3's "p≈0.48 retrain," PROK's "eukaryotic-probe contamination") are additionally
in direct conflict with artifacts that already exist. Full trace:
`paper-salvage/docs/NEW_DIRECTION_EVIDENCE_AUDIT.md`.

Consequences, all mechanical/documentary, no science performed:

- `paper-salvage/docs/CLAIMS_LEDGER.md`: 3 claims downgraded to a new `contested` status
  (C-020, C-021 PROK half, C-029); 1 new claim added from direct code inspection (C-033,
  quantization scale-endpoint math — no run needed, no run performed).
- `paper-salvage/docs/DECISIONS.md`: D-016–D-023 appended (append-only).
- `paper-salvage/docs/PAPER_OUTLINE.md`: rewritten to v2 (R1–R6). R1 and R6 are draftable
  now; R2–R5 are marked **BLOCKED, no artifact** and must not be drafted.
- `paper-salvage/CLAUDE.md`: rewritten with an explicit list of claims that must not be
  written without a real artifact.
- `paper-salvage/docs/MANUSCRIPT_MIGRATION_MAP.md`: new. Maps every major section/figure of
  `paper/main.tex` (still the old v6 manuscript, **not rewritten this session**) to KEEP /
  REWRITE / MOVE TO SUPPLEMENT / DELETE / BLOCKED BY PROVENANCE.
- `paper-salvage/docs/PAPER_READINESS.md`: new. Section-by-section sufficiency call for
  R1–R6, and an explicit verdict on whether any further experiment is justified before
  writing (short answer: no).
- `paper-salvage/docs/PROJECT_STATUS.md`: updated with this session's log entry (history
  preserved, nothing rewritten).

## 2. Blockers

**None mechanical.** Everything remaining is either (a) prose writing against evidence that
already exists, or (b) a provenance question only the author (or a future session with new
information) can resolve:

- **N-013 (`CLAIMS_LEDGER.md`)** — PROK SAE's fp16-clamp mechanism is real and disclosable;
  the "98% variance destroyed / pathological features" characterization is not measured
  anywhere and conflicts with the manuscript's own recorded SAE health diagnostics (97.4%
  features active). Needs either a variance measurement from the stored SAE shards, or the
  claim dropped in favor of the disclosed clamp mechanism alone.
- **N-014** — C-029 (NTv3 splice replication, p=0.008, real 5-seed controls) directly
  contradicts an instruction to retire it in favor of a truncation-bug/p≈0.48 story with zero
  local artifact. Needs either the truncation-bug-fix and retrained-checkpoint artifacts
  located/added, or an explicit author decision to retract C-029 anyway, logged as a new
  DECISIONS.md entry.
- **N-015** — C-020/C-021 (PROK sign-convention and shuffle claims) are `contested` because
  they are keyed to the same artifact N-009 already found non-reproducible (C-001, on hold
  since 2026-08-12) — not because a "wrong probe" mechanism was verified (it wasn't). Needs
  either the probe-provenance check and an L8/r260 detection-sweep output located/added, or
  an explicit author decision.

## 3. Canonical report paths (this session)

| What | Path |
|---|---|
| **New-direction evidence audit** | `paper-salvage/docs/NEW_DIRECTION_EVIDENCE_AUDIT.md` |
| **Manuscript migration map** | `paper-salvage/docs/MANUSCRIPT_MIGRATION_MAP.md` |
| **Paper readiness / R1–R6 sufficiency** | `paper-salvage/docs/PAPER_READINESS.md` |
| Reconciled outline (v2) | `paper-salvage/docs/PAPER_OUTLINE.md` |
| Reconciled operating rules | `paper-salvage/CLAUDE.md` |
| Superseding decisions | `paper-salvage/docs/DECISIONS.md` (D-016–D-023) |
| Ledger reconciliation notes | `paper-salvage/docs/CLAIMS_LEDGER.md` (N-012–N-015) |

Everything from the prior session (E1/E2/E4 results, prereg lock) is unchanged — see that
session's paths still listed in `paper-salvage/docs/PROJECT_STATUS.md`.

## 4. What this session did NOT do (deliberately, per its own instructions)

- Did not run E3 (steering), retrain NTv3, re-run Evo1's secondary dose, search PROK layers
  for L8/r260, redo matched-norm controls, start the E1 prospective model, or rerun C-010 on
  eager.
- Did not rewrite `paper/main.tex`. It still asserts the old (retired) thesis, the "eight
  models" language, and the "shadow redundancy" claim verbatim — all flagged in
  `MANUSCRIPT_MIGRATION_MAP.md`, none fixed. **The manuscript prose is stale relative to
  `PAPER_OUTLINE.md` v2 and must not be cited as reflecting the current state.**
- Did not invent numbers, controls, or mechanisms for any of the six new headline results.
  Where no artifact existed, the claim is documented as missing, not filled in.

## 5. What a future session should actually do — pick one lane, do not mix them

**Lane A — write.** R1 and R6 have enough real, established evidence to draft now
(`PAPER_READINESS.md` confirms this). Draft those two sections of a new manuscript (or a new
draft file — do not overwrite `paper/main.tex` without a plan for the rest of it) directly
against `CLAIMS_LEDGER.md` and `PAPER_OUTLINE.md` v2. This does not require any new
experiment.

**Lane B — close one provenance gap.** Pick exactly one of N-013/N-014/N-015, and either (a)
locate the missing artifact (it may exist outside this repository, e.g. on another node or in
another session's untracked scratch space, and simply needs to be copied in with provenance),
or (b) run the minimum experiment needed to produce it, under a full preregistration if it's
a confirmatory claim, with an explicit go-ahead from the author first. Do not do this for more
than one item without stopping to reassess — this project has scattered before.

**Do not** start a new Tier-0 experiment list, and do not treat R2–R5 as implicitly queued —
they stay BLOCKED until a future session is explicitly told to produce their evidence.

## 6. Unresolved scientific decisions carried over from the prior session (still open, still
not decided by this or any pass since)

1. **Evo1 Branch A vs B** — not to be decided in a writing-focused session either (D-018)
   unless a future instruction reopens it. Peak T: EUK 3.79e-02, Evo1 1.20e-01, DNABERT-2
   8.01e-01, PROK 6.18e+00.
2. **Whether T has any relationship to criticality** — C-017, `pending`, explicitly not a
   headline-thesis candidate under the new outline.
3. **C-001 / PROK rank discrepancy (N-009)** — unchanged, still on hold.
4. **Evo1's missing secondary-dose KL** — unchanged, unmeasured, not to be retried without
   instruction.
5. **Empty matched-norm arms** — unchanged.
6. **Whether/when to re-run C-010 on eager** — unchanged, not started.
7. Ref [11] Sun et al. arXiv ID — still needs external lookup (`paper-salvage/docs/REFERENCE_AUDIT.md`).

## 7. Exact next commands

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights

# verify the prereg lock still holds (should print OK — nothing in this session touched it)
python3 paper-salvage/src/prereg_lock.py verify --all

# re-run the shared-library tests if you touch uk_frobenius.py (untouched this session)
ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$PWD/paper-salvage/src \
  $ENV/bin/python -m pytest paper-salvage/src -q            # expect: 7 passed
```

## 8. Environment traps (unchanged from prior session)

- Container has **no `python`**, only `python3`.
- Host `SSL_CERT_FILE` points outside the container; use `env -u SSL_CERT_FILE -u
  REQUESTS_CA_BUNDLE` for any `huggingface_hub` network call.
- `grlm` needs `LD_LIBRARY_PATH=$ENV/lib`.
- No `evo` conda env; Evo1/StripedHyena work needs the `evo2.sif` Apptainer container.
- `paper-salvage/results/` is gitignored via the `results/` pattern — use `git add -f`.
- No LaTeX toolchain on this node.
