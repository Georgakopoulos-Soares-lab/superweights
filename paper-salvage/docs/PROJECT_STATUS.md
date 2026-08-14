# PROJECT_STATUS.md

**Last updated:** 2026-08-14 (colleague branch integration — seventh session, continuation of
E5–E8)
**Current phase:** A colleague's independent mechanism/negative-results branch
(`origin/mechanism-and-negative-results`, diverged mid-June, worked in parallel with this
project's whole E5–E8 arc) was audited (`docs/COLLEAGUE_BRANCH_AUDIT.md`), then — on explicit
author approval — integrated onto a dedicated branch, `integrate/mechanism-and-negative-
results`, **not onto `main`**. The colleague's single commit (`5b0220c`) cherry-picked
cleanly; `paper-salvage/` never overlapped with it (zero path collision, confirmed by the
audit), so **nothing in E5–E8 was touched by the integration**. Two files needed manual
reconciliation: `README.md` (the colleague's imported "current status" section was rescoped
down to reference material — pointers to `results/mechanism/` reports, a script inventory, and
an explicit warning that the underlying JSON/CSV artifacts are absent — rather than left to
assert unaudited numeric findings; a caveat was added next to the pre-existing PROK
write-direction-convention material flagging that a colleague report disputes the channel
identity itself, without adjudicating between them) and `scripts/evaluation/
run_gue_ablation.py` (both sides' additions — current main's `_GeneratorClassifier`, the
colleague's `_NTv3Classifier` padding-to-multiple fix — coexist with zero actual line overlap,
confirmed by diffing the reconciled file against pre-integration `main`). A
`docs/MISSING_COLLEAGUE_ARTIFACTS.md` manifest records every raw result artifact the
colleague's reports cite that is not present anywhere in the pushed branch or this filesystem,
grouped P0–P2 for future recovery/reproduction (**no reproduction was attempted this
session**). Ledger bookkeeping enacted only where independently justified without the missing
artifacts: **C-032 retired** (moved to `X-007`, on this repo's own E7 evidence, not colleague
work), **C-034/C-035 added** (E7+E8's exact-dimensionality/encoder-decoder result, previously
only "proposed"), **C-029 downgraded from `contested` to `pending-rerun`** (the NTv3
truncation bug is now independently confirmed by a direct code trace performed by this
integration — not taken from colleague prose — as applying to C-029's own backing artifact;
no corrected replacement number was inserted, since none has its own artifact yet), and three
notes appended in place (N-014, N-015 updated; N-016 added for Evo1) recording what the
colleague branch resolves, narrows, or leaves open for N-009/N-013/N-014/N-015 — no claim's
headline status beyond C-029 changed. Full detail: `docs/COLLEAGUE_BRANCH_AUDIT.md`,
`docs/MISSING_COLLEAGUE_ARTIFACTS.md`. **`main` itself was not merged into** — the integration
branch is a separate, explicitly-not-yet-merged branch, per the author's instruction; the
former "sixth session" work below is unchanged.

The prior phase (E8 encoder-vs-decoder dimensionality, sixth session) tested the
one clean hypothesis E7's post-hoc observation generated: does exact operator dimensionality
track encoder-vs-decoder organization rather than text-vs-genomic domain? Two independently
selected, previously unmeasured text encoders (MosaicBERT — shares DNABERT-2's exact FFN
class; ModernBERT — an independent codebase) were audited, detected, and measured under a
locked prereg. **Both produced clean high-gain candidates (detection ratios 289x and 561x)
and both fell below the decoder-cluster floor on the pre-registered criterion — mechanical
Branch A** — though MosaicBERT's shift was dramatic (`q1`=0.477 vs. a 0.899 floor) while
ModernBERT's was thin (`q1`=0.897, a 0.21% relative margin). A striking secondary finding:
in both encoders, the detected high-gain row is far *more* concentrated than its own layer's
ordinary rows (control `q1` ≈ 0.03–0.05), the reverse of what a naive "encoders are generally
distributed" reading would suggest. Full 2x2 table, both-metric decision trace, and a
design-only causal-feasibility note (concluding the needed intervention tooling does not yet
exist): `experiments/E8_encoder_decoder/RESULTS.md`. **C-032 was separately, decisively
recommended for retirement** independent of E8's outcome
(`experiments/E7_exact_dimensionality/C032_RETIREMENT_RECOMMENDATION.md`), and an E7
provenance addendum resolved immutable checkpoint hashes for all four E7 models
(`experiments/E7_exact_dimensionality/PROVENANCE_ADDENDUM.md`). None of this touches the
manuscript. The third new experiment before this one (E7 — exact bilinear-operator
dimensionality,
`experiments/E7_exact_dimensionality/`) reformulated E5/E6's question around the exact `U_k`
singular spectrum rather than the diagonal approximation, and around model/checkpoint as the
primary unit rather than row (fixing E6's OLMo/DNABERT-2 pseudoreplication). A four-model
confirmatory panel (Phi-3, Qwen2.5-7B; Evo 2 7B, GenomeOcean-4B) was audited, downloaded, and
run — but **Evo 2 returned a Phase-1 detection null** (no down-projection spike anywhere in
32 layers cleared the preregistered 5x threshold under the frozen ACTB probe), degrading the
genomic group to one surviving model and disqualifying the panel from a formal Branch A/B/C/D
call, per the prereg's own rule. The Phase-6 legacy reanalysis (run only after Phase 5 froze)
then showed why forcing a branch would have been wrong regardless: **GENERator EUK's exact
q1 (0.969) exceeds one of the four NLP models' (OLMo-7B, 0.965)**, and GenomeOcean-4B sits
essentially on top of Phi-3 — no clean NLP-vs-genomic separation survives under the exact
metric across either panel. NTv3 remains a genuine, substantial outlier. Full picture, an
unconfirmed but visually clean encoder-vs-decoder observation, and claim recommendations:
`experiments/E7_exact_dimensionality/RESULTS.md`. This does not change anything from the
2026-08-13 reconciliation pass below: R1/R6 are still the draftable sections, R2–R5 are still
blocked, and the manuscript was not touched by E5, E6, E7, or E8.

**Blocking on:** recovering or reproducing the colleague's missing raw artifacts
(`docs/MISSING_COLLEAGUE_ARTIFACTS.md`, P0 = DNABERT-2 redundant pair / pretrained-intrinsic /
norm-codominance mechanism) before any of claims A–E, the corrected NTv3 retrain, or the
corrected PROK detection can be written as established; and author review of N-013 and the
remaining, still-open half of N-015 (PROK: colleague diagnosis located and corroborated by an
exact rank match with this repo's own N-009, but still unverified at the artifact level) —
plus the pre-existing open items (Evo1 Branch A vs B, the missing Evo1 secondary-dose KL,
empty matched-norm arms). **This session enacted, rather than merely recommended, four
ledger changes** (the only prior session's edits were the C-033 row itself): C-032 → `X-007`
retired; C-034/C-035 added (E7+E8's exact-dimensionality/encoder-decoder result, folded from
"proposed" into a live row); C-029 → `pending-rerun` (old ΔMCC/p=0.008 invalidated by an
independently code-confirmed truncation bug — see N-014's 2026-08-14 update — no replacement
number inserted); N-015 narrowed (the "wrong-probe" mechanism it flagged as unaudited now has
located, partially-corroborating colleague code, but is still not artifact-complete); N-016
added (Evo1 "2^24": a colleague's own early claim, retracted by their own later work, found
consistent with — not contradicting — this repo's existing N-001/N-002, no headline claim
changed). C-002/C-003 unchanged (retain).

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

## What changed this session (2026-08-14, E7 exact operator dimensionality — fifth session,
continuation)

E6's row-level confirmatory near-miss motivated a reformulated hypothesis about the exact
`U_k` operator (not its diagonal approximation) and a model-level (not row-level) confirmatory
design, explicitly to fix the pseudoreplication E6 disclosed in its own panel (3 OLMo-7B rows,
9 DNABERT-2 rows). New, self-contained experiment: `paper-salvage/experiments/
E7_exact_dimensionality/`.

1. **Phase 0 model audit** (`MODEL_PANEL.md`): Gene42 — **UNAVAILABLE**, no public checkpoint
   exists anywhere on HuggingFace despite the paper's claim (only empty placeholder finetune
   repos found). Evo 2 7B — initially appeared broken (`transformer_engine` ABI mismatch from
   a stale host-side package); **the user corrected this** — the repo's own existing sbatch
   convention (`--cleanenv --env PYTHONNOUSERSITE=1`) resolves it, confirmed working.
   GenomeOcean-4B selected as Gene42's one authorized replacement (serves the same
   LLaMA-style-on-genomic-data bridge-diagnostic role; `MistralForCausalLM`, confirmed
   compatible). Phi-3-mini-4k-instruct and Qwen2.5-7B confirmed ELIGIBLE. Gemma 2 —
   UNAVAILABLE, gated, no valid HF token in this environment.
2. **Phase 2/3**: `spectral_lib.py` implements `q1`/`PR_spec` from `U_k`'s exact singular
   spectrum. 6/6 synthetic tests green pre-lock, including a cancellation case showing the old
   diagonal PR reports near-maximal apparent dimensionality on a construction whose exact
   operator is the zero matrix.
3. **Phase 4**: `docs/prereg/PREREG_exact_operator_dimensionality.md` locked (sha256
   `7da30f81...`) — 4-model confirmatory panel, a frozen activation-detection protocol for
   the three models without published coordinates, model-level aggregation, and a mechanical
   decision tree requiring both members of a 2-vs-2 group to separate.
4. **Phase 5**: Phi-3 (6 rows, median `q1`=0.903) and Qwen2.5-7B (`q1`=0.953) detected/measured
   cleanly. GenomeOcean-4B detected cleanly (`q1`=0.899). **Evo 2 7B returned a Phase-1 null**
   — no layer's down-projection spike cleared the preregistered 5x ratio anywhere across 32
   blocks. Per the locked rule this degrades the genomic group to one surviving model,
   disqualifying Branch A outright — **mechanical result: EXPLORATORY, not A/B/C/D.**
5. **Phase 6** (run only after Phase 5 froze, per the binding ordering rule): reanalyzed
   E5/E6's six legacy rows under the identical exact metric. **GENERator EUK's exact `q1`
   (0.969) exceeds OLMo-7B's (0.965)** — a genomic model is more rank-1 than an NLP model.
   NTv3 remains a substantial outlier (`q1`=0.389). No clean domain split survives.
6. **No causal-feasibility document was written** (conditional on Branch A only; did not
   occur). No replacement geometry hypothesis invented; a visually clean but explicitly
   **unconfirmed** encoder-vs-decoder pattern is recorded descriptively in `RESULTS.md`, not
   asserted as a finding.
7. **Claim-ledger recommendations (not enacted):** C-002/C-003 unchanged (retain). **C-032
   recommended RETIRE** — the exact metric does not reproduce its domain-general qualitative
   claim. A new claim ID (**C-034, proposed, not added**) is drafted in `RESULTS.md` for the
   actual pattern found (heterogeneous exact dimensionality within the genomic group).

---

## What changed this session (2026-08-14, E8 encoder-vs-decoder — sixth session, continuation)

Two housekeeping items, then the new experiment:

1. **E7 provenance addendum** (`experiments/E7_exact_dimensionality/PROVENANCE_ADDENDUM.md`,
   `EVO2_SHA256.txt`): resolved immutable HF commit hashes and weight-shard SHA256 for all
   four E7 checkpoints (Phi-3, Qwen2.5-7B, GenomeOcean-4B, Evo2-7B). Documentation only — no
   E7 result changed or rerun.
2. **C-032 retirement, decided now, independent of E8**
   (`experiments/E7_exact_dimensionality/C032_RETIREMENT_RECOMMENDATION.md`): GENERator EUK's
   exact `PR_spec` already sits inside the published-NLP range, and every genomic model's
   diagonal PR overstates its exact value in the same direction — sufficient on its own,
   without waiting for E8.

New experiment: `paper-salvage/experiments/E8_encoder_decoder/`, testing the one clean
hypothesis E7's post-hoc observation generated (encoder/decoder organization, not domain).

3. **Model audit** (`MODEL_AUDIT.md`): MosaicBERT (`mosaicml/mosaic-bert-base`) confirmed to
   use the **identical** `BertGatedLinearUnitMLP` class DNABERT-2 uses (verified from the
   actual remote-code source) — the strongest possible architecture bridge to DNABERT-2 while
   changing domain. ModernBERT (`answerdotai/ModernBERT-base`) confirmed independent
   (`ModernBertMLP`, a different codebase, verified from installed `transformers` source).
   Both ELIGIBLE; hard gate (≥2 independent bidirectional encoder families) passes.
4. **Prereg locked** (`docs/prereg/PREREG_encoder_decoder_dimensionality.md`, sha256
   `7a5dc793...`): reuses E7's frozen WikiText-2 detection protocol and 5.0x threshold
   verbatim; fixes the Branch-A criterion as `q1` strictly below 0.8989 (the decoder
   cluster's own observed floor, GenomeOcean-4B, used unadjusted).
5. **Confirmatory run:** both models detected cleanly (ratios 289x, 561x). **MosaicBERT
   `q1`=0.4766 (dramatically below the floor); ModernBERT `q1`=0.8970 (below the floor by
   only 0.21% relative — thin but genuine).** Both shift on both `q1` and the secondary
   `PR_spec` check. **Mechanical result: Branch A.**
6. **Secondary finding, reported prominently:** in both encoders, the detected candidate row
   is far *more* concentrated than 5 seeded same-layer control rows (control `q1` ≈
   0.03–0.05 vs. candidates' 0.48/0.90) — the opposite of a naive "encoder rows are generally
   distributed" reading; the high-gain phenotype, where present, is a local exception.
7. **Causal-feasibility note written** (Branch A triggers this; design-only, no run):
   concludes the needed activation-space intervention tooling does not yet exist in this
   repository — a genuinely new engineering effort, not a quick reuse of existing scalar/row
   ablation scripts (which are keyed to the diagonal decomposition this whole arc has been
   moving away from).
8. **Claim-ledger note:** C-034's proposed wording is narrowed with E8's result (encoder/
   decoder tracking, not domain) rather than restated as a new headline claim — still not
   added to `CLAIMS_LEDGER.md`.

---

## What changed this session (2026-08-14, colleague branch integration — seventh session,
continuation)

Two sequential passes on the same colleague branch, `origin/mechanism-and-negative-results`
(a single commit, `5b0220c`, diverged from this project at the 2026-06-17 merge-base and
worked entirely independently, in parallel, through the whole E5–E8 arc):

**Pass 1 — provenance audit only** (no merge, no rerun): `docs/COLLEAGUE_BRANCH_AUDIT.md`.
Established the branch relationship (39 commits unique to current vs. 1 colleague commit;
`paper-salvage/` does not exist on the colleague branch or at the merge-base — zero path
collision with any of E5–E8); audited all 9 claims the colleague's reports make (A–I) back to
their producing scripts and raw artifacts; found every raw JSON/CSV output the reports cite
absent from the pushed branch (only `.md` reports and `.py` scripts were committed —
`results/`'s gitignore evidently didn't carve out the JSON/CSV siblings the way it did the
markdown). Two findings were resolved with specific care, as instructed: **Evo1's "2^24"**
claim was the colleague's own early (2026-08-03), since-self-retracted hypothesis — their
later (2026-08-07) fp64 adjudication found the real plateau at 1.75×2^24, which falls inside
this repo's own independently-measured N-001 range — **not a contradiction**. **The NTv3
truncation bug** was independently re-derived from this repo's own code (not colleague
prose): `run_gue_multiseed.py`'s `_MAX_LEN["reconstructed"]=80` fallback fires on C-029's own
launch script, which never passes `--max_length` — **confirmed real and load-bearing on
C-029's own backing artifact.**

**Pass 2 — integration, on explicit author approval** (branch, cherry-pick, reconcile,
bookkeeping only — no scientific rerun): new branch `integrate/mechanism-and-negative-
results`, off this session's HEAD, **not `main`**. `5b0220c` cherry-picked with zero conflicts
git couldn't auto-merge; both flagged files (`README.md`, `run_gue_ablation.py`) were then
hand-checked (not just trusted) against pre-integration `main` to confirm nothing current
main added was lost. `README.md`'s imported colleague section was rescoped from an
"established findings" table down to reference material with an explicit "not yet citable"
warning and pointers to the audit/manifest; a caveat was added beside the pre-existing PROK
write-direction-convention material (itself from an earlier, independent reconciliation,
commit `4ba1686`) noting a colleague report disputes the channel identity at a level below
that convention, without resolving which is right. `docs/MISSING_COLLEAGUE_ARTIFACTS.md`
catalogues every missing artifact by claim, script, and priority (P0: DNABERT-2 pair/
pretrained-pair/norm-codominance; P1: steering, attention sink; P2: corrected NTv3, corrected
PROK, destructive-quantization controls, Evo1 supplements) — **nothing in it was reproduced
this session.**

**Ledger changes actually enacted** (not just recommended, unlike prior sessions' C-032/C-034
notes): C-032 → `X-007` (retired, on this repo's own E7 evidence — not attributed to
colleague work); C-034 and C-035 added (folding E7's proposed claim and E8's confirmatory
narrowing into live rows, exactly as E8's own `RESULTS.md` asked for); C-029 → `pending-rerun`
(old ΔMCC/p=0.008 invalidated by the independently-confirmed bug; no replacement number
written in — none has its own artifact); N-014 and N-015 each got a dated update appended in
place (originals preserved, not rewritten) narrowing what each is still waiting on; N-016
added for the Evo1 note; C-033's evidence cell got one corroborating addendum (a second,
independent implementation uses the identical scale-endpoint formula) with no status change.

Provenance: `docs/COLLEAGUE_BRANCH_AUDIT.md`, `docs/MISSING_COLLEAGUE_ARTIFACTS.md`, commits
on `integrate/mechanism-and-negative-results` (cherry-pick `3d3bbb0`, README reconciliation,
manifest+bookkeeping). **The integration branch has not been merged into `main`** and no
further action (merge, further cherry-pick, artifact reproduction) proceeds without separate
explicit author approval.

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
2026-08-14  |  E7 exact operator dimensionality: EXPLORATORY (not A/B/C/D). New experiment
            |  dir experiments/E7_exact_dimensionality/; prereg locked (sha256 7da30f81...).
            |  Model audit: Gene42 UNAVAILABLE (no public checkpoint anywhere); Evo2 initially
            |  looked broken (transformer_engine ABI mismatch) but user corrected -- fixed
            |  via this repo's own --cleanenv --env PYTHONNOUSERSITE=1 convention; GenomeOcean
            |  -4B selected as Gene42's replacement. 4-model confirmatory panel (Phi-3,
            |  Qwen2.5-7B / Evo2-7B, GenomeOcean-4B): Phi-3 q1=0.903 (median of 6 rows),
            |  Qwen2.5 q1=0.953, GenomeOcean q1=0.899 all measured cleanly; Evo2 returned a
            |  Phase-1 detection NULL (max spike ratio 2.22 < 5.0x threshold, all 32 layers)
            |  -- degrades genomic group to 1 survivor, disqualifies Branch A per the prereg's
            |  own rule. Phase 6 legacy reanalysis (run only after Phase 5 froze): GENERator
            |  EUK exact q1=0.969 EXCEEDS OLMo-7B's 0.965; NTv3 remains a genuine outlier
            |  (q1=0.389). No clean NLP-vs-genomic separation survives under the exact metric
            |  across either panel. No causal-feasibility doc (Branch A only, not reached).
            |  Recommend (not enacted): C-032 RETIRE; new claim C-034 proposed (not added) for
            |  the heterogeneous-within-genomic-group pattern actually found.
            |  next: author call on Evo2 (retry with a different native input distribution?
            |  needs its own new prereg either way), or resume R1/R6 writing / N-013/014/015.
2026-08-14  |  E7 provenance addendum (hashes/revisions for all 4 E7 checkpoints, no result
            |  change) and C-032 retirement recommendation (decided now, independent of E8:
            |  GENERator EUK's exact PR_spec already sits inside the NLP range). E8 encoder-
            |  vs-decoder: prereg locked (sha256 7a5dc793...), reusing E7's frozen WikiText-2
            |  detector + 5.0x threshold verbatim. MosaicBERT (identical FFN class to
            |  DNABERT-2) and ModernBERT (independent codebase) both audited ELIGIBLE, both
            |  detected cleanly (ratios 289x, 561x). MosaicBERT q1=0.4766 (dramatically below
            |  the 0.8989 decoder floor); ModernBERT q1=0.8970 (below by only 0.21% relative
            |  -- thin but genuine, reported as such). Mechanical result: BRANCH A. Secondary
            |  finding: both candidates are far MORE concentrated than 5 seeded same-layer
            |  control rows (control q1 ~0.03-0.05) -- the high-gain phenotype is a local
            |  exception, not typical of encoder rows generally. Design-only causal-
            |  feasibility note written (Branch A triggers it): concludes the needed
            |  activation-space intervention tooling does not exist yet, new engineering
            |  effort required. C-034's proposed wording narrowed with this result, not
            |  restated as new. No CLAIMS_LEDGER.md edit.
            |  next: author review of C-032 retirement + C-034 proposal; separately, whether
            |  to scope the causal-feasibility gap as new infrastructure work, or resume
            |  R1/R6 writing / N-013/014/015.
```
