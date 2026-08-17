# CLAUDE.md — operating rules for this folder (v2, 2026-08-13)

You are working on the salvage and restructure of a manuscript on super-weights in genomic
language models. As of 2026-08-13 the manuscript is being rebuilt around a **second** new
thesis — the v1 U_k-led thesis below is retired (`DECISIONS.md` D-016). Read
`docs/PAPER_OUTLINE.md` (v2) before doing anything substantive, and read
`docs/NEW_DIRECTION_EVIDENCE_AUDIT.md` before writing any sentence that sounds like a
headline result — most of the results a downstream review described as the new state turned
out to have **no artifact anywhere in this repository**. Check before you write, not after.

## The thesis (frozen — do not re-litigate without a new DECISIONS.md entry)

**What survives — and what fails — when the NLP "single super-weight" concept is transferred
to genomic foundation models?** Super-weight-like concentration transfers, but the causal
object is not universal (canonical decoder super-activation vs. distributed encoder ensemble
vs. structurally-fired-but-non-load-bearing candidate); this exposes limits of single-weight
detection and precision-preservation heuristics carried over from NLP.

Old thesis, retired (`D-016`), kept here only so you recognize it if you see it in old prose:
*"Gated MLPs create structurally predictable high-gain channels, but architectures differ
fundamentally in how these signals are routed and recruited for biological computation."*

## Hard constraints on claims

These are not style preferences. Section A is retained from v1 (still true, still binding).
Section B is new as of 2026-08-13. Violating either reintroduces exactly the kind of
provenance failure this project has already been burned by twice (v15's "shadow redundancy,"
and now the un-auditable "new headline results" this reconciliation pass found).

### A. Retained from v1

- **C (coordinate preservation) is NOT necessary for criticality.** DNABERT-2 has C ≈ 0
  (re-measured on the clean eager kernel, N-011) and the strongest *uncontested* encoder
  ablation phenotype (C-027/C-028). Never write that (‖U_k‖_F, C) jointly predict
  criticality.
- **"Recovers the published super-weight output rows"** unless the scalar index *i* also
  ranks highly, in which case you may say scalar recovery. Be pedantic here.
- **Never write "shadow redundancy"** or "SW-neighbourhood pruning tolerance." The pruning
  data contradict it (X-001). This is now doubly true: the specific "layer-depth control
  reproduces the far-SW effect" claim floated in the 2026-08-13 review also has no artifact
  (`NEW_DIRECTION_EVIDENCE_AUDIT.md`, cross-cutting notes) — do not invent one to fill the gap.
- **Never write "eight models"** as though they form a uniform benchmark. Use the coverage
  table in `docs/PAPER_OUTLINE.md`.
- **Do not assert causal arrows between routing and context-sensitivity.**
- Assertive about the **method**. Hedged about the **mechanism**.

### B. New as of the 2026-08-13 reconciliation — see `NEW_DIRECTION_EVIDENCE_AUDIT.md` for the
full trace behind each of these

> **SUPERSEDED 2026-08-17 — see `docs/DECISIONS.md` D-024.** The author reviewed the located
> colleague branch (`mechanism-and-negative-results`, commit `5b0220c`,
> `docs/COLLEAGUE_BRANCH_AUDIT.md`) and instructed that its reported results be treated as
> facts, notwithstanding the continued absence of raw JSON/CSV artifacts. Every bullet below
> is therefore lifted, with a pointer to the claim ID that now carries it. **The bullets are
> kept, not deleted** — they record real, sound reasoning for four sessions of withholding
> these claims, and remain the correct default the next time a claim shows up as prose with no
> local artifact and no author adjudication.

- ~~Do not write a DNABERT-2 "redundant pair," a base-model/pretrained pair-superadditivity
  comparison, or a layer-9 joint-residual-norm-carriage claim.~~ **Lifted — see C-036
  (pair), C-037 (pretraining-intrinsic), C-038 (norm carriage, DNABERT-2-scoped).** The
  existing 10-row ensemble (C-027/C-028) is still a separate, adjacent result — do not
  conflate the two.
- ~~Do not write a GENERator BOS/attention-sink phenotype.~~ **Lifted — see C-039.**
- ~~Do not write a GENERator EUK causal GC-steering result, "38.6×" or otherwise.~~ **Lifted —
  see C-040.** C-025 (the separately preregistered E3 steering experiment, never locked or
  run) stays `pending` in its own right — C-040 is the colleague's independent measurement,
  not a retroactive confirmation of C-025.
- ~~Do not claim Evo1's SW value was pinned at 2^24, or that a rescue experiment was run.~~
  **Narrowed, not fully lifted.** The literal "pinned at exactly 2^24" claim was never true —
  it was the colleague's own early hypothesis, which their own later work retracted (real
  plateau 1.75×2²⁴). That corrected account is now accepted as supplementary detail on
  C-009/C-011 (see N-016, closed). C-009/C-011's headline status is unchanged.
- ~~Do not claim NTv3 was retrained after a truncation-bug fix, or cite p≈0.48.~~ **Lifted —
  see C-029 (updated in place; corrected figures are MCC 0.86–0.91 / −0.02pp, not p≈0.48 —
  check the ledger row, not this old bullet, for the exact numbers).** Old result retired as
  X-009.
- ~~Do not claim the old PROK super-weight was identified via a "eukaryotic probe on the
  prokaryotic model," and do not cite an "L8/r260" replacement SW.~~ **Lifted — see C-001
  (updated), C-041, C-042.** Old channel (L2/r1927) and everything keyed to it retired as
  X-008.
- ~~Do not claim the PROK SAE's correlations were "driven by pathological/single-active
  features destroying 98% of variance."~~ **Partially lifted only.** The fp16-clamp mechanism
  is accepted. The specific "98%/pathological" figure is **not** lifted by default — unlike
  every other bullet here, it directly contradicts an artifact this repo already has
  (`manuscript.txt:389-393`'s healthy SAE diagnostics), not merely an absent one. See N-013's
  2026-08-17 update — flagged for separate, explicit author attention.
- ~~Do not generalize the quantization scale-preservation math (C-033) beyond the per-row rule
  actually implemented.~~ **Lifted for the empirical Q2/Q4 results — see C-043.** C-033 itself
  was never affected (it was always established from code alone); C-043 adds the empirical
  confirmation and the destructive-regime null as accepted fact.
- **Do not restore C-017 as a headline thesis, and do not adjudicate Evo1 Branch A vs. B.**
  Unaffected by D-024 — this bullet was never about the colleague branch. Broadcast/T-C work
  (C-012, C-013, C-015, C-016) is supplementary only (`D-018`).
- **If a claim resurfaces with a real artifact in a future session**, or if a future author
  decision revisits D-024 itself, update this section and the ledger together, in the same
  commit — this is the same discipline that produced this update.

## Working discipline

- Every claim that will appear in the manuscript gets a row in `docs/CLAIMS_LEDGER.md` with
  its evidence file and status. If you add a claim to a draft section, add the row in the
  same commit. If you cannot fill the evidence path, the claim is not ready to be written —
  full stop, regardless of how confidently it was described to you.
- Every decision that changes scope gets an entry in `docs/DECISIONS.md`. Append only; never
  rewrite history.
- Update `docs/PROJECT_STATUS.md` at the end of any session that changes state.
- Predictions must be locked *before* the confirming run: `python3 src/prereg_lock.py lock
  <file>`. Never edit a locked prereg file after locking.
- Results we are keeping get copied into `results/keep/` with provenance recorded (`git add
  -f`, since `results/` is gitignored). Do not reference paths that live only in the old tree.
- **Before writing any sentence that states a numerical result, find its evidence path.** If
  you cannot find one in under a few minutes of targeted search, it does not exist yet — say
  so, do not write the sentence anyway.

## What not to do

- Do not add new analyses that aren't explicitly instructed. This project has now failed
  twice by scattering / by writing ahead of its evidence — once as v15, once as the
  2026-08-13 "new headline results" that turned out to have no artifacts.
- Do not silently drop a negative result. Null steering, null Evo1, contested NTv3, contested
  PROK — all get reported, not smoothed over in either direction.
- Do not delete anything from the old tree. Cuts are *demotions* to supplement; see
  `docs/CUT_LIST.md` and `docs/MANUSCRIPT_MIGRATION_MAP.md`.
- **Do not launch GPU jobs, load large checkpoints, or start any new experiment** (including
  E3, an Evo1 secondary dose, a PROK layer search, or an NTv3 retrain) without an explicit,
  separate instruction to do so. See `NEXT_SESSION.md` for exactly what is and is not queued.
