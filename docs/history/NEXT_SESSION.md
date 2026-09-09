# NEXT_SESSION.md

**Session:** 2026-08-17, author decision D-024 — colleague mechanism-session results adopted
as facts (same-day continuation of the seventh session's colleague-branch integration).
**Status:** The author reviewed the colleague-branch audit and missing-artifact manifest and
instructed that the colleague's reported results be treated as facts, notwithstanding the
continued absence of raw JSON/CSV artifacts. This is recorded as `docs/DECISIONS.md` **D-024**
and has been fully enacted in the ledger, `CLAUDE.md`, and `README.md` this same session. The
integration branch (`integrate/mechanism-and-negative-results`) is still **not merged into
`main`**.

---

## 1. What happened this session

1. **`docs/DECISIONS.md` D-024 recorded** — explicitly supersedes D-019 (Evo1), D-020 (PROK),
   D-021 (NTv3), and `CLAUDE.md` §B in full.
2. **`CLAIMS_LEDGER.md` updated:**
   - `C-001` updated in place — corrected PROK super-weight is **L8/r260**, not L2/r1927.
   - `C-020` retired (`X-008`) → replaced by `C-041` (corrected hexamer causal test: ρ=+0.0007,
     p=0.96 — no sign relationship at the corrected channel).
   - `C-021`'s PROK half retired (`X-008`) → replaced by `C-042` (GC-dependence of ablation
     *cost* is the real surviving kingdom contrast: PROK r=−0.661, EUK r=−0.001). EUK half of
     C-021 unaffected.
   - `C-029` updated in place — old ΔMCC=−0.119/p=0.008 retired (`X-009`, confirmed truncation
     bug); corrected: MCC 0.86–0.91, SW ablation effect **−0.02pp (no effect)**. NTv3 no
     longer counts as a functional replication of the SW-ensemble effect; DNABERT-2 remains
     the sole one (n=1).
   - Five new claims added: `C-036` (DNABERT-2 redundant pair), `C-037` (pretraining-
     intrinsic), `C-038` (norm/codominance mechanism — explicitly DNABERT-2-scoped, the NTv3
     null is part of the claim), `C-039` (attention sink), `C-040` (causal steering,
     non-monotonic).
   - `C-043` added — quantization Q2/Q4 empirical results, on top of C-033 (unchanged, still
     established from code alone).
   - N-012, N-014, N-015, N-016 closed. **N-013 only partially closed** — see below.
3. **`CLAUDE.md` §B annotated** — every bullet marked superseded/lifted with a pointer to the
   claim ID that now carries it. Bullets kept, not deleted, as the historical record of why
   these claims were withheld for four sessions.
4. **`README.md` updated** — "reference material, not yet citable" framing replaced with a
   findings table; the old PROK section (row 1927) marked superseded and kept for history, not
   deleted.
5. **`docs/MISSING_COLLEAGUE_ARTIFACTS.md`** — header note added: no longer gates claim
   status, still tracks a genuine reproducibility gap.

## 2. One item deliberately NOT swept in — needs separate author attention

**N-013 — the PROK SAE's "98% variance destroyed / pathological features" figure.** Unlike
every other item D-024 covers, this one directly **contradicts an artifact this repo already
has** (`manuscript.txt:389-393`'s own recorded SAE diagnostics: MSE 4.78, 97.4% of dictionary
features active — describing a healthy fit, not a destroyed one). D-024's rationale (trusting
a colleague's report where this repo has no competing measurement) does not obviously extend
to overriding a measurement this repo already made. The fp16-clamp *mechanism* is adopted (it
is real and uncontroversial); the specific 98% figure is not. **Flagged for the author**:
either the manuscript's own recorded SAE diagnostic needs to be treated as stale/wrong, or the
colleague's 98% figure needs to be treated as an overstatement — this pass does not decide
which.

## 3. Blockers

None mechanical. The one open item is N-013 above, which needs an author call, not more
tracing — this pass could not find local grounds to prefer either side.

Carried over, unchanged: Evo1 Branch A vs. B, the missing Evo1 secondary-dose KL, empty
matched-norm arms.

## 4. Canonical report paths (this session)

| What | Path |
|---|---|
| The decision itself | `manuscript/docs/DECISIONS.md` D-024 |
| Updated claims | `manuscript/docs/CLAIMS_LEDGER.md` (C-001, C-020→X-008/C-041, C-021→X-008/C-042, C-029→X-009, C-036–C-040, C-043) |
| Constraint annotations | `manuscript/CLAUDE.md` §B |
| Updated findings summary | `README.md` §"Mechanism session findings (Aug 2026)" |
| Residual open item | `manuscript/docs/CLAIMS_LEDGER.md` N-013 (2026-08-17 update) |

Everything from the prior seven sessions is unchanged except where D-024 explicitly updates
it — see `manuscript/docs/PROJECT_STATUS.md`'s session log.

## 5. What this session did NOT do (deliberately)

- Did not merge the integration branch into `main`.
- Did not resolve N-013's residual conflict either way.
- Did not recover or reproduce any of the still-missing raw artifacts — D-024 changes
  evidentiary *policy* toward the colleague's reports, it does not manufacture the files.
- Did not touch E5–E8 (`C-034`/`C-035`, the encoder/decoder result) — unrelated to this
  decision.
- Did not run any scientific experiment, rerun, or GPU job.

## 6. What a future session should actually do — pick one lane

**Lane A — resolve N-013.** Needs an author call on whether the manuscript's existing SAE
diagnostic or the colleague's 98% figure is the one to trust — not more code tracing.

**Lane B — merge the integration branch into `main`**, only on explicit instruction.

**Lane C — write.** The ledger now has real numbers for R2 (Evo1/GENERator mechanism), R4
(PROK kingdom contrast), R5 (steering), R6 (DNABERT-2 pair, NTv3 null, quantization) that were
previously blocked. `PAPER_OUTLINE.md` should be checked against the new claim set before
drafting.

**Lane D — recover the raw artifacts anyway**, for independent reproducibility, even though
it no longer gates claim status. Ask the colleague first per
`MISSING_COLLEAGUE_ARTIFACTS.md`'s recovery-path ordering.

**Do not** mix lanes.

## 7. Exact next commands

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights
git branch --show-current   # should be integrate/mechanism-and-negative-results

# verify all five prereg locks still hold
python3 manuscript/src/prereg_lock.py verify --all
```

## 8. Environment traps (unchanged from prior sessions)

- Container has **no `python`**, only `python3`.
- Host `SSL_CERT_FILE` points outside the container; use `env -u SSL_CERT_FILE -u
  REQUESTS_CA_BUNDLE` for any `huggingface_hub` network call.
- Live `git fetch` fails non-interactively in this environment (no askpass, no cached
  credentials) — pre-existing `refs/remotes/origin/*` are reliable and fully walkable.
- `manuscript/results/` is gitignored via the `results/` pattern — use `git add -f`.
- Background shell `cd` does not persist across separate Bash tool calls; always
  `cd <absolute path> &&` at the start of every background command.
- No LaTeX toolchain on this node.
