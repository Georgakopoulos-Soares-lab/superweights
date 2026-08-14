# NEXT_SESSION.md

**Session:** 2026-08-14, colleague branch (`mechanism-and-negative-results`) audit +
integration (seventh session, continuation of E5–E8).
**Status:** A colleague's independent mechanism/negative-results branch was audited, then —
on explicit author approval — integrated onto a **separate, not-yet-merged** branch,
`integrate/mechanism-and-negative-results`. `main` is untouched by the integration itself
(only by the audit's own documentation commits, which are ordinary, safe additions). No
scientific experiment was run or reproduced this session.

---

## 1. What happened this session

**Pass 1 — audit** (`paper-salvage/docs/COLLEAGUE_BRANCH_AUDIT.md`, on `main`, no merge):

- Identified `origin/mechanism-and-negative-results` (single commit `5b0220c`, diverged
  2026-06-17, authored 2026-08-13) as the only candidate colleague branch.
- Confirmed `paper-salvage/` — and therefore every E5–E8 file — does not exist on that branch
  or at the merge-base: zero path collision.
- Audited all 9 claims (A–I) the colleague's reports make, back to producing scripts and raw
  artifacts. **Every raw JSON/CSV artifact the reports cite is absent from the pushed
  branch** — only `.md` reports and `.py` scripts were committed. Classified essentially every
  claim PARTIAL, with two exceptions resolved by careful, specific tracing (as instructed):
  - **Evo1 "2^24"**: the colleague's own early (2026-08-03) hypothesis, retracted by their own
    later (2026-08-07) fp64 adjudication (real plateau = 1.75×2^24), which falls inside this
    repo's own N-001 range. **Not a contradiction.**
  - **NTv3 truncation bug**: independently re-derived from this repo's own code (not
    colleague prose) — `run_gue_multiseed.py`'s `_MAX_LEN["reconstructed"]=80` fallback fires
    on C-029's own launch script. **Confirmed real, applies to C-029's own backing artifact.**
  - **PROK**: colleague's independently-computed contaminated-row rank (1289/3072) matches
    this repo's own N-009 rank exactly — corroborating, not resolving, the "cause not
    determined" gap. The claimed `super_weight_index.json` correction is **not actually
    present** in the pushed branch (byte-identical to the pre-existing, uncorrected file).

**Pass 2 — integration** (`integrate/mechanism-and-negative-results`, off this session's
`main` HEAD, not merged into `main`):

1. Committed the audit doc on `main` (`a43020a`), verified all 5 prereg locks (`OK`).
2. Branched `integrate/mechanism-and-negative-results` from `main`.
3. Cherry-picked `5b0220c` — **zero unresolved conflicts**; git auto-merged both
   flagged files. Hand-verified (not just trusted) against pre-integration `main`:
   `_GeneratorClassifier` byte-identical, only hunk touched in `run_gue_ablation.py` is the
   `_NTv3Classifier` padding-to-multiple fix.
4. Rescoped the cherry-picked `README.md` section from an "established findings" table down
   to reference material (pointers to the audit + manifest, explicit "not yet citable"
   warning), kept only the code-verified NTv3-bug summary as a real finding, fixed one
   directory-tree comment that incorrectly implied JSON/CSV artifacts were present. Added a
   caveat beside the pre-existing PROK write-direction-convention material (from an earlier,
   independent `main` commit, `4ba1686`) flagging that a colleague report disputes the channel
   identity beneath that convention — without resolving which is right.
5. Wrote `paper-salvage/docs/MISSING_COLLEAGUE_ARTIFACTS.md` — every missing raw artifact,
   grouped by claim, producing script, and priority (P0/P1/P2). **Nothing in it was
   reproduced.**
6. Enacted ledger changes that were justified **independent of the missing artifacts**:
   - `C-032` → `X-007` (retired; this repo's own E7 evidence, not colleague-attributed).
   - `C-034`, `C-035` added (E7+E8's exact-dimensionality/encoder-decoder result, folded from
     "proposed" into live rows — this was already fully decided pre-integration, just not yet
     enacted).
   - `C-029` → `pending-rerun` (old ΔMCC/p=0.008 invalidated by the independently-confirmed
     bug; **no replacement number inserted** — it has no artifact yet).
   - `N-014`, `N-015` updated in place (dated addenda, originals preserved); `N-016` added
     (Evo1, no headline change); `C-033`'s evidence cell got one corroborating addendum
     (independent second implementation, same formula) — no status change.
7. Updated `paper-salvage/docs/PROJECT_STATUS.md` (new top summary + new session-log block).

Full detail: `paper-salvage/docs/COLLEAGUE_BRANCH_AUDIT.md`,
`paper-salvage/docs/MISSING_COLLEAGUE_ARTIFACTS.md`.

## 2. Blockers

**The integration branch has not been merged into `main`.** No further merge, cherry-pick, or
rebase happens without separate, explicit author approval — this was the standing instruction
for this entire task.

The single largest blocker for turning any of this into manuscript prose: **recovering or
reproducing the colleague's raw artifacts**, starting with P0 (DNABERT-2 redundant-pair,
pretrained-intrinsic-pair, and norm/codominance mechanism — see
`MISSING_COLLEAGUE_ARTIFACTS.md`). Ask the colleague first (most likely explanation: `results/`
is gitignored and the `.md`-only carve-out didn't extend to JSON/CSV/PNG). Only plan a bounded
reproduction, with its own preregistration, if the colleague confirms the artifacts are
permanently unrecoverable.

Carried over, still open: N-013 (unchanged), the still-open half of N-015 (PROK — corroborated
but not artifact-complete), Evo1 Branch A vs. B, the missing Evo1 secondary-dose KL, empty
matched-norm arms.

## 3. Canonical report paths (this session)

| What | Path |
|---|---|
| Colleague branch provenance audit | `paper-salvage/docs/COLLEAGUE_BRANCH_AUDIT.md` |
| Missing-artifact manifest (P0/P1/P2) | `paper-salvage/docs/MISSING_COLLEAGUE_ARTIFACTS.md` |
| Integration branch | `integrate/mechanism-and-negative-results` (cherry-pick `3d3bbb0`, README reconciliation, bookkeeping commits on top) |
| Colleague's own reports (imported, reference only) | `results/mechanism/*.md` on the integration branch |
| Updated ledger | `paper-salvage/docs/CLAIMS_LEDGER.md` (`C-032`→`X-007`, `C-034`, `C-035`, `C-029`, `N-014`/`N-015`/`N-016`) |

Everything from the prior six sessions is unchanged — see `paper-salvage/docs/
PROJECT_STATUS.md`'s session log.

## 4. What this session did NOT do (deliberately)

- Did not merge, rebase, or cherry-pick anything into `main`.
- Did not rewrite either branch's history.
- Did not run any scientific experiment, rerun, or GPU job.
- Did not reproduce or recreate any of the colleague's missing raw artifacts.
- Did not update any manuscript claim from colleague prose alone — every ledger change made
  this session traces to either this repo's own prior E5–E8 work (C-032/C-034/C-035) or an
  independently-performed code trace by this integration itself (C-029/N-014), not to the
  colleague's reports as such.
- Did not enact the L8/r260 PROK replacement or any corrected NTv3 number.
- Did not touch `CLAUDE.md`'s hard constraints (Section B) — every one of them remains fully
  in force; nothing that surfaced this session clears any of their "no artifact" bars.

## 5. What a future session should actually do — pick one lane

**Lane A — recover P0 artifacts.** Ask the colleague for the raw `results/mechanism/*.json`
outputs backing the DNABERT-2 pair/pretraining/norm-codominance chain. If recovered, audit
them before writing anything into the ledger.

**Lane B — bounded reproduction of one P0 claim**, only if Lane A is exhausted and the
colleague confirms the artifacts are gone. Needs its own preregistration; do not reuse E5–E8's
locks.

**Lane C — write** (unchanged from prior sessions). R1 and R6 have enough real evidence to
draft now, independent of anything in this session.

**Lane D — merge the integration branch into `main`**, only on the author's explicit
instruction — this session's mandate was integrate-and-report, not merge.

**Do not** mix lanes.

## 6. Unresolved scientific decisions carried over (unchanged, still open)

Identical to the prior `NEXT_SESSION.md` versions' §6 (Evo1 Branch A vs B, C-017,
C-001/N-009, Evo1's missing secondary-dose KL, empty matched-norm arms, C-010 on eager,
ref [11]) — nothing in this session touched any of them.

## 7. Exact next commands

```bash
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights

# verify all five prereg locks still hold (E2, E5 Gate-0, E6 Stage-A, E7, E8)
python3 paper-salvage/src/prereg_lock.py verify --all

# inspect the integration branch without switching off main's working tree
git log --oneline --decorate -n 8 integrate/mechanism-and-negative-results
```

## 8. Environment traps (unchanged from prior sessions, plus one new note)

- Container has **no `python`**, only `python3`.
- Host `SSL_CERT_FILE` points outside the container; use `env -u SSL_CERT_FILE -u
  REQUESTS_CA_BUNDLE` for any `huggingface_hub` network call.
- `grlm` needs `LD_LIBRARY_PATH=$ENV/lib`, and both `HF_HOME=/work/11034/atzanakak/ls6/
  huggingface/.hf-cache` and `TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/nonbdna/cache/hf`
  set simultaneously — a full-model download and a targeted `hf_hub_download` call can land
  in **different** cache directories in the same session.
- MosaicBERT ships no tokenizer files of its own — load `BertTokenizer.from_pretrained(
  "bert-base-uncased")` instead of `AutoTokenizer`.
- Background shell `cd` does not persist across separate Bash tool calls; always
  `cd <absolute path> &&` at the start of every background command.
- `paper-salvage/results/` is gitignored via the `results/` pattern — use `git add -f`.
- **New — live `git fetch` fails non-interactively in this environment** (no askpass, no
  cached HTTPS credentials). Pre-existing `refs/remotes/origin/*` from an earlier successful
  fetch are reliable and fully walkable (verified non-shallow) — use those rather than
  blocking on a live fetch; just record that a live fetch could not be completed.
- No LaTeX toolchain on this node.
