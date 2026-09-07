# Colleague branch provenance audit — `mechanism-and-negative-results`

**Provenance-first audit only. No merge, rebase, cherry-pick, or scientific rerun was
performed. Both branches are exactly as they were before this document was written.** A
separate worktree was used throughout; the current branch's working tree was never checked
out over.

---

## 0. Branch identification

Three remote branches exist: `origin/main`, `origin/feat/genomic-super-weights-import`
(an old import branch, ancestor of everything, last touched 2026-05-05), and
`origin/mechanism-and-negative-results`. The name, single-commit structure, and commit date
(2026-08-13, the day of the reconciliation pass that went looking for exactly this work)
identify `origin/mechanism-and-negative-results` unambiguously as **COLLEAGUE_BRANCH**. No
ambiguity, no candidates to report.

**Live `git fetch --all --prune` failed in this session** (interactive HTTPS askpass, no
credential available in this environment — `fatal: could not read Username for
'https://github.com'`). The remote-tracking refs used throughout this audit were already
present locally from an earlier, successful fetch (confirmed via `git reflog show
refs/remotes/origin/mechanism-and-negative-results`, one entry, `storing head`) and were
verified non-shallow and fully walkable (`git rev-list`, `git ls-tree -r` both succeed
completely). The data audited here is real and complete; only a *live* re-fetch was
unavailable this session.

**Worktree:** `/work/11034/atzanakak/glm_super_weight/worktrees/mechanism-and-negative-results`
(detached at `5b0220c`).

---

## 1. Branch relationship

| | |
|---|---|
| Current HEAD | `e1a23c305b98b2adeccb489cb33eaabe314b7bc2` — "E8: BRANCH A..." (2026-08-14) |
| COLLEAGUE_BRANCH HEAD | `5b0220c513b1cd7db379f3023ae04ad3fd5801aa` — "Mechanism experiments, confound controls, and retraction of NTv3 results" (2026-08-13T16:25:18Z, Aris Karatzikos) |
| Merge-base | `15309002c8793d645aace79637f3717f98acf07b` — "added tests" (**2026-06-17**) |
| Commits unique to current | **39** |
| Commits unique to colleague | **1** (a single, large, well-documented commit) |

**COLLEAGUE_BRANCH was based nearly two months before the reconciliation pass and the entire
E5–E8 arc.** It diverged in mid-June, before `paper-salvage/` existed in any form (see §2).
Its single commit was authored 2026-08-13 — the same day as the reconciliation pass that went
looking for it — meaning the colleague was working independently and in parallel with this
session's E5–E8 work, not building on it.

**`paper-salvage/` does not exist on COLLEAGUE_BRANCH, and did not exist at the merge-base.**
Verified: `git ls-tree origin/mechanism-and-negative-results -- paper-salvage` and the same at
the merge-base both return nothing. Every file COLLEAGUE_BRANCH touches is in the old tree
(`scripts/`, `sae/`, `results/`, `README.md`). **This means the entire diff is structurally
incapable of overwriting any E5–E8 file** — there is no path collision at all with
`paper-salvage/experiments/E5_dimensionality/` through `E8_encoder_decoder/`, the prereg
locks, `CLAIMS_LEDGER.md`, `PROJECT_STATUS.md`, or anything else under `paper-salvage/`.

`git diff --stat HEAD...origin/mechanism-and-negative-results`: **38 files changed, 6,676
insertions(+), 18 deletions(-)**. Two files are touched by **both** branches since the
merge-base and need explicit reconciliation (not overwrite) — see §5.

---

## 2. The provenance ceiling that applies to every claim below

**Only two classes of file are actually tracked on COLLEAGUE_BRANCH: markdown reports under
`results/mechanism/*.md`, and the producing scripts under `scripts/mechanism/`,
`scripts/compression/`, `scripts/evaluation/`, and `sae/`.** The commit message states this
explicitly: *"Reports in `results/mechanism/` (markdown tracked by exception to the `results/`
ignore rule...)"* — the exception covers the `.md` files only.

**No raw JSON/CSV/PNG output artifact referenced by any report is present in the pushed
branch.** Verified exhaustively: `git ls-tree -r origin/mechanism-and-negative-results
--name-only` lists zero files matching any of the ~40 distinct artifact filenames the reports
cite (`channel_survival_evo1.json`, `mixer_intervention.json`,
`epistasis_pretrained_vs_finetuned.json`, `generator_prokaryote_sw_corrected.json`,
`evo1_fp64_adjudication.json`, `attention_sink_implicit_bias.json`,
`sw_steering_generation.json`, `scale_preservation_grouped.json`, and every other name in the
reports' own artifact indices). A broad filesystem search for these filenames outside git
(the shared repo working directory, home directory) also found nothing — `results/mechanism/`
does not exist at all outside the colleague's own worktree checkout. **This is a real,
consistently-observed gap, not a one-off omission**, and it is the single fact that governs
every classification in §3: no claim below can reach AUDITED in the strict sense ("reproduced/
traced from actual artifacts... without relying on prose") purely from this branch, because
the actual output data does not exist in it.

**This is charitably, not accusatorially, read.** The reports are unusually careful and
self-critical — they record their own errors, retract their own premature conclusions (Evo1,
§3.F), state explicit caveats and limits on almost every finding, and the methodology
described (seeds, controls, decision rules stated before runs, "kill" verdicts accepted when
data contradicted the hypothesis) reads as genuine completed work, not fabrication. The most
likely explanation is that `results/` is gitignored by default in this repository and the
`.md`-only carve-out in the commit did not extend to the JSON/CSV/PNG siblings the same
session produced — a fixable gap, not a red flag about the work itself. It is nonetheless a
gap that this audit's strict evidentiary standard (matching this project's own discipline
throughout E5–E8) must not paper over.

**One claim was independently verified by this audit directly from code, not from colleague
prose** — see §3.G. That one is AUDITED in the full sense.

---

## 3. Claim audit table (A–I)

| | Status | Why |
|---|---|---|
| **A** — redundant pair | **PARTIAL** | Detailed methodology + real 220-line producing script (`scripts/evaluation/run_sw_pairwise_epistasis.py`); no raw artifact |
| **B** — pretraining-intrinsic | **PARTIAL** | Real 341-line script (`scripts/mechanism/run_pretrained_epistasis.py`); no raw artifact |
| **C** — joint norm carriage | **PARTIAL** | Detailed, self-limiting methodology; no raw artifact; colleague's *own* later work (§3.C, `FALSIFICATION_NORM_VS_SUPERWEIGHT.md`) already narrows this claim significantly |
| **D** — attention sink | **PARTIAL** | Real 209-line script (`scripts/mechanism/run_attention_sink.py`); no raw artifact |
| **E** — causal steering | **PARTIAL** | Real scripts (`run_sw_steering.py`, `run_steering_biological.py`); no raw artifact |
| **F** — Evo1 "2^24" | **CONTRADICTION DISSOLVES ON INSPECTION** — see §4.1. The literal claim (exactly 2²⁴) was tested and retracted by the colleague's *own* later checkpoint. The corrected finding is consistent with, not contradictory to, this branch's N-001/N-002 |
| **G** — NTv3 truncation bug | **Bug mechanism: AUDITED, by this audit directly (not colleague prose).** Corrected retrained result: MISSING (no artifact) — see §4.3 |
| **H** — PROK wrong probe | **PARTIAL, with one concrete discrepancy** — see §4.2. The single tracked shared artifact (`results/super_weight_index.json`) that the reports say was corrected is, in fact, **byte-identical** to current main's uncorrected version |
| **I** — quantization Q1–Q4 | **Q1–Q3: AUDITED** (directly code-verifiable, and matches this branch's own C-033 exactly). **Q4: PARTIAL** (detailed numbers, no raw artifact) — see §5 |

### A. DNABERT-2 redundant pair (`SUPERADDITIVITY_AND_COMPOSITION_REPORT.md` §1, `HANDOFF` §1.1)

Splice all-10-row ablation: **−26.84 ± 2.56 pp** (n=5 seeds) vs. sum-of-individual-parts
**−3.51 ± 2.15 pp**. Seed-0 critical pair (L9/r264, L9/r294): separate **−0.02, −0.11 pp**;
joint **−33.76 pp**; random-pair control **+0.006 to +0.007 pp**. Top-7 epistatic pairs are
7/7 "structurally related" (same layer or same row index) across all 3 tasks tested
(splice p=0.00014, promoter p=0.0035, histone p=0.0035, n=5 seeds). Producing script:
`scripts/evaluation/run_sw_pairwise_epistasis.py` (220 lines, real, not a stub — confirmed by
direct inspection). Commit: `5b0220c`. **Caveat the colleague states themselves**: their own
later falsification test (§3.C below) shows the SW pair is empirically indistinguishable from
"the two largest-norm channels" in this specific model — the label "super weight" vs.
"dominant norm carrier" is not separable here.

### B. Pretraining-intrinsic (`HANDOFF` §1.2, `CHECKPOINT_1_PRETRAINED_EPISTASIS.md`)

Same critical pair, pretrained MLM loss (no task head, held-out hg38, seed 42):
epistasis **+2.0118**. Enrichment: top-3 3/3 (p=0.032), top-5 5/5 (p=0.0025), top-7 6/7
(p=0.0035). Random-pair floor sd **0.0000147**; top pair is **136,521×** it. k-of-N cliff at
k=5. Pretrained-vs-finetuned epistasis rank correlation ρ=+0.316, p=0.034. Script:
`scripts/mechanism/run_pretrained_epistasis.py` (341 lines). No raw artifact.

### C. Mechanism: joint norm carriage (`HANDOFF` §1.3, `FALSIFICATION_NORM_VS_SUPERWEIGHT.md`,
`E1_E2_CODOMINANCE_AND_CONFOUND.md`)

Layer-9 residual norm: baseline **17.20** → ablate both **7.14 (−58%)**. Two competing
mechanistic geometries explicitly tested and rejected by the colleague themselves (depth
redundancy: 91.9% of readout survives ablating all four upstream writes to the channel;
asymmetric head-readout: one channel read at percentile 98, its partner at percentile 31).

**The colleague's own later work already scopes this claim tightly, unprompted — read this
before citing C anywhere:**
- `FALSIFICATION_NORM_VS_SUPERWEIGHT.md` (2026-08-10): norm dominance does **not** generalize.
  NTv3's SW channel is **more** norm-dominant than DNABERT-2's pair (29.4× gap to the next
  channel vs. ~4.5×) yet ablating it costs **−0.02 pp** vs. DNABERT-2's −17.72 pp for the
  matched test. Explicit verdict: *"The joint-norm-carriage mechanism explains DNABERT-2 and
  does not generalise to NTv3."*
- `E1_E2_CODOMINANCE_AND_CONFOUND.md` (2026-08-10): a genuinely well-controlled intervention
  (redistributing magnitude between the two channels while holding total norm and baseline
  accuracy constant) shows co-dominance is **necessary** for DNABERT-2's ensemble phenotype
  (epistasis migrates −33.63 → −0.66 as the split becomes lopsided, while the single-channel
  effect rises −0.02 → −33.03 — the *same* total criticality becomes single-point instead of
  joint). But imposing the identical co-dominance ratio onto NTv3 (`MAKE-PAIR`) **produces
  nothing** — neither joint nor single criticality. The colleague's own conclusion: *"an
  additional architecture-level factor"* is required, and it is explicitly **[UNTESTED]**.

No raw artifact for any of A/B/C.

### D. GENERator attention-sink (`HANDOFF` §1.4, `scripts/mechanism/run_attention_sink.py`)

EUK: 37.96% of attention mass at position 0 (33.0× uniform); 78.3% of heads have their argmax
there; SW activation at pos 0 is 45,585× other positions; argmax at token 0 in 60/60 (100%)
of probes. PROK replicates directionally (28.72%, 25.0×). DNABERT-2 (encoder, contrast case):
argmax at token 0 in **0/40 (0%)**. Colleague-disclosed caveats: MosaicBERT does not expose
attention maps in this pipeline, so the encoder side of the dissociation rests on activation
data only, not attention; and the shuffle-ratio-at-pos-0 result is architecturally guaranteed
in a causal decoder once BOS-anchoring is established, so it corroborates rather than
independently confirms. No raw artifact.

### E. GENERator EUK causal steering (`HANDOFF` §1.5)

Scaling the SW row's write by {0.0, 0.5, 1.0, 2.0, 5.0}: generated GC moves 0.2944 → 0.3961 →
0.3549 (non-monotonic — saturates by 2.0, reverses at 5.0). Random-row scaling: flat
0.396–0.399. Span ratio **38.59×** (PROK replicates directionally, weaker, 17.87×). Colleague's
own caveat: *"Monotone on the suppression side only... do not present as a linear knob."* No
raw artifact.

### I. Quantization — see the dedicated §5 below (kept separate per the governing instruction).

---

## 4. Resolving the three specifically-flagged items with care

### 4.1 Evo1 — the contradiction dissolves once the exact tensor is traced

**Claim as originally reported (item F):** "apparent SW is a numerical saturation false
positive... a relevant quantity pinned exactly at 2²⁴... a rescue forcing it to survive
output had no functional information."

**What the colleague's own timeline actually shows, traced commit-internally:**

1. `CHECKPOINT_REPORT.md` (2026-08-03, the *earlier* checkpoint): first observes "an exact
   `2^24` value appears in the existing validated trace, and it 'freezes' for 19 blocks,"
   and (§T1.2) concludes row 3776 is "best described as an inert numerical artifact."
2. `CHECKPOINT_A_REPORT.md` (2026-08-07, the *later*, more careful checkpoint, §A2) **goes
   back and explicitly tests the earlier premise, and retracts it**:
   > *"Frozen at exactly 2²⁴? **No.** Plateau = 29,360,128 = **1.75 × 2²⁴**; no block hits
   > 2²⁴ exactly."*
   >
   > *"A2 — Evo1 fp64 adjudication: **REAL_BUT_DISPERSED**, not an artifact... The expected
   > answer was 'numerical artifact.' The data says otherwise, and I'm reporting the result
   > rather than the expectation."*

   fp64 recompute of the L11 down-proj write at row 3776: **1,247,245.4**, vs. the bf16 value
   **1,245,184.0** actually used — a 0.165% gap that is **exactly bf16's grid spacing (ULP =
   8192) at that magnitude**. bf16 is faithfully rounding a genuine large value, not
   saturating garbage. Mechanistic decomposition: the MLP writes **1.25e6** at block 11; the
   **mixer** (not the MLP) amplifies it ~24× over the next two blocks (block 12: **+14.68e6**;
   block 13: **+13.43e6**) — summing to the observed plateau, ≈29.36e6. Ablation ΔPPL under
   this fp64-verified value: **+0.026%**, still null. The rescue/forced-survival intervention
   (from the earlier checkpoint) gave rescued ΔPPL 0.0026% ≈ natural 0.0032% — no change, so
   *"no causal-gating claim is licensed."*

**Are these the same tensor as this branch's N-001/N-002?** Yes, materially — same model
(Evo1), same coordinate (row 3776), same layer region (L11 write, L12–13 amplification/
settling). This branch's own N-001 already measured (independently, months earlier, via a
completely different code path — `scripts/interpretability/run_sw_broadcast_impulse.py`, not
the colleague's new `hooks/ablation_trace.py`): *"median |h| across the 4,096 dims reaches
~2.8e5 at layer 11 and settles at ~4.2e6–3e7 from layer 13 onward (max ~1.29e9)."* **The
colleague's fp64-verified plateau (2.94e7) falls squarely inside this branch's own
independently-measured settling range (4.2e6–3e7).** This branch's N-002 already established
the mechanism-adjacent fact that the apparent "freeze" is a summary-statistic artifact of a
saturating `max|Δ|`, not a literal frozen tensor — fully consistent with, and now given a
concrete per-block mechanistic explanation by, the colleague's contribution decomposition.

**Verdict: no contradiction survives.** The "exactly 2²⁴" framing was the colleague's own
early, since-retracted hypothesis — not something this branch ever needs to reconcile against.
The corrected finding (real, fp64-verified, mechanistically decomposed, causally null under
both natural ablation and a forced-survival rescue) is **independent corroboration** of this
branch's own C-009/C-011 (*"Evo1 is structurally concentrated... ablation gives ΔPPL = 0.0%...
structural concentration does not imply functional criticality"*), measured through a
genuinely different instrument, and it **adds** mechanistic depth (the MLP-then-mixer
decomposition; the fp64-vs-bf16 fidelity test) that this branch does not currently have.
**Recommend this be treated as new corroborating/supplementary detail for C-009/C-011 once
its own artifact (`evo1_fp64_adjudication.json`, not currently present) is recovered or
reproduced — not as a contradiction to resolve.**

### 4.2 GENERator PROK — old sign result and new candidate are not simply "both real"

**Old selection procedure:** `scripts/detection/run_detection_generator_prokaryote.sbatch`
invoked with `--probe human_promoter` — a human ACTB sequence — against the
prokaryote-specialized 3B model. Result: `layer 2, row 1927` (`out_max=506,014`).

**Corrected selection procedure (per the colleague's reports):** a fresh detection sweep using
`pseudomonadota`, an existing prokaryotic probe already present in `probes/dna_probes.py`
(confirmed present in this repository — this is not a new resource the colleague introduced).
Reported result: `layer 8, row 260` (secondary: `row 1325`), `out_max=30,167.07`, content-
invariant across all 4 canonical probes, `‖U_k‖_F` rank **1/3072**.

**Which old downstream results become invalid, per the colleague's own D1 rerun report:** the
`r = −0.710` kingdom sign-reversal (`main.tex:95,434`) — on the corrected row, Spearman
ρ = **+0.0007, p=0.96** (no correlation; the naive Pearson +0.958 is explicitly flagged by the
colleague as a two-hexamer artifact, "do not report... as a sign flip"). The PROK SAE
(`main.tex:447–465,752–756,925`) — trained on the contaminated layer, no checkpoint exists
anywhere, and the corrected row has essentially zero activation variance (621±2 across all
4096 hexamers) to decompose in the first place. The `+9.47` log-PPL ablation figure — becomes
`+1.25 ± 0.54` (still highly significant vs. random rows, just 7.6× smaller). One thing that
**strengthens** under correction, reported as hypothesis-level by the colleague themselves:
GC-dependence of ablation *cost* (not activation), r = −0.520 → **−0.661**.

**Not called a "contradiction" by this audit** — per the governing instruction. The old
Pearson r=−0.710 was a real, internally-consistent computation on the channel it was run on;
it is invalidated by the *selection* of that channel being contaminated, not by an inconsistency
in the arithmetic itself.

**One more thread, found independently by this audit, not previously connected:** current
main's own history already tried to reconcile the −0.710 number once before, on a *different*
axis. Commit `4ba1686` ("Align merged main.tex to canonical: unified magnitude-scaled SW
mechanism (PROK -0.710 is write-direction convention, +0.710 on magnitude)...") reinterprets
the sign as a write-direction convention **while keeping L2/r1927 as the channel**. The
colleague's finding — if the missing artifact is recovered and confirms it — would make that
reconciliation moot at a deeper level: not "the sign was read wrong," but "the whole channel
was never the real super weight." **These are two independent, unreconciled explanations for
the same anomalous number**, and only one of them (the sign-convention patch) currently has
any artifact behind it in this repository (main.tex itself, on current main).

**The concrete discrepancy this audit found, not asserted by the colleague's own reports:**
the reports state, repeatedly and specifically, that `results/super_weight_index.json` was
updated (*"generator_prokaryote now holds L8/r260... Old entry archived → `results/
super_weight_index_generator_prokaryote_ORIGINAL_SUSPECT.json`"*). **`git diff HEAD...
origin/mechanism-and-negative-results -- results/super_weight_index.json` is empty — the file
is byte-identical to current main's, and still shows the old, contaminated `layer: 2, row:
1927` entry.** The archived `_ORIGINAL_SUSPECT.json` file, `generator_prokaryote_sw_
corrected.json`, and all nine rerun files under `results/prokaryote/` that the reports
describe as written are likewise absent from the tracked tree (§2). This is the single
clearest instance in this audit of the general artifact gap: a specific, checkable claim
("the index file was corrected") that is directly falsifiable against the actual pushed
content, and does not hold as pushed.

This directly and productively engages this branch's own **N-009** (`CLAIMS_LEDGER.md`): the
stored ‖U_k‖_F value for L2/r1927 (2648.4773, rank 1/3072) was already independently found,
by this branch's own re-computation, **not to reproduce** at its own claimed layer (recomputed
5.51/5.47, rank 1277/1289) — *"cause not determined"* at the time N-009 was written. The
colleague's wrong-probe diagnosis is a specific, credible, independently-motivated candidate
explanation for exactly the gap N-009 left open, consistent with — not contradicting — what
this branch already knew was broken about that entry. **N-015's own text already anticipated
this exact situation**: *"if the eukaryotic-probe diagnosis and the L8/r260 relocation are
real findings from work done outside this repository, the artifacts... need to be added to
the repo before either can be written into the manuscript."* That work has now been located.
Its numeric artifacts still have not been added.

### 4.3 NTv3 — the most direct conflict, and this audit independently confirms the bug

**Old configuration, traced exactly:** `results/gue_multiseed_ntv3_splice.json` (backing
C-029, `dMCC = −0.119 ± 0.054`, `p = 0.0083`, 5 seeds) was produced by
`scripts/evaluation/submit_ntv3_splice_multiseed.sh`, which invokes `run_gue_multiseed.py`
with `--model ntv3 --task splice/reconstructed ...` and **does not pass `--max_length` at
all** — verified by reading the launch script directly, on this session's own filesystem, not
from colleague prose.

**Traced forward through the actual code, independently, by this audit:**
`scripts/evaluation/run_gue_multiseed.py:269` resolves
`max_length = args.max_length or _MAX_LEN.get(tkey, 512)`. With no `--max_length` passed,
`args.max_length` is `None`, so this falls through to `_MAX_LEN.get("reconstructed", 512)`.
`scripts/evaluation/run_gue_ablation.py:54–68` (imported by `run_gue_multiseed.py:57`) defines
`_MAX_LEN["reconstructed"] = 80  # splice`. **The resolved max_length for this exact run was
80 tokens.** NTv3 tokenizes at nucleotide level (1 bp = 1 token; confirmed architecturally —
NTv3 is a conv/deconv U-Net over a per-nucleotide vocabulary, distinct from DNABERT-2's BPE
tokenizer, for which 80 tokens covers roughly a 400bp window). **80 tokens on NTv3 is 80bp —
the first 20% of a ~400bp splice window, truncated before the splice junction the task is
about.** `results/gue_multiseed_ntv3_splice.json` itself records no `max_length` field, so
this was not self-documented in the output and would not have been visible without tracing the
launch script and the two-file default-resolution chain by hand.

**This directly, concretely resolves this branch's own N-014**, which had left the truncation-
bug claim as *"flagged for the author... nothing in the repository... states an intended
larger window"* — that assessment did not trace through `_MAX_LEN`'s task-key fallback the way
this audit just did. **The bug is real, and it applies to the exact artifact behind C-029.**
This is the one finding in this entire audit classified **AUDITED without qualification** —
it was independently reproduced from code by this audit, not taken from colleague prose.

**What is NOT yet auditable:** the corrected numbers. The colleague reports MCC 0.86–0.91 on
a refit model with the SW ablation effect reduced to −0.02 pp, but no retrained checkpoint,
eval JSON, or log for this specific corrected run exists in the pushed branch or anywhere
findable on this filesystem. **Per the governing instruction, C-029 is not replaced by this
audit.** The correct status update is: *the bug that would invalidate C-029 is now
independently, code-level confirmed real and applicable to C-029's own backing artifact; the
specific corrected replacement numbers remain unverified pending a recovered or reproduced
artifact.* This is a stronger, more specific position than N-014's prior "flagged, not
resolved either way" — the bug diagnosis itself is no longer in doubt; only the corrected
number is still missing.

---

## 5. Quantization, decomposed (Q1–Q4)

**Q1 — Mathematical** (per-row symmetric RTN quantizer): confirmed **AUDITED**, directly from
code, independent of the colleague's framing. `scale = max|w| / qmax` (symmetric, no explicit
zero-point — implied zero-centered by "symmetric"); the max-magnitude element of a row maps to
`round(max|w|/s) = qmax` and dequantizes to `qmax·s = max|w|` exactly, by construction, at any
bit width. **This is the identical formula this branch's own C-033 already established by
direct code inspection** (`scripts/compression/run_quantization_ablation.py`,
`run_whole_model_quantization.py:300-303`). No new information here relative to C-033 — this
is independent, empirical corroboration of a claim this branch already holds as `established`.

**Q2 — Empirical** (the SW value defines its row's scale): the colleague measures SW-element
quantization error of **0.000e+00** at INT8/4/3/2 for the two rows directly tested (L9/r264,
L9/r294) — a direct empirical confirmation that these SW values are literally their row's
max-magnitude element, not merely a mathematical possibility. Reported separately: 5 of
DNABERT-2's 10 SW rows are additionally the **global tensor max** (`row_max/tensor_max =
1.000`), meaning they are exempt under per-tensor as well as per-row quantization; the other 5
range 0.35–0.83 of the tensor max. **PARTIAL** — the two-row zero-error result is a specific,
plausible, well-explained measurement; the broader 10-row table has no raw artifact behind it.

**Q3 — Consequence** (explicit protection is a no-op under this quantizer): **AUDITED as a
direct logical consequence of Q1+Q2** — if the quantizer already maps the SW to itself exactly
(Q1), and the SW value already *is* the element that triggers that exact mapping (Q2), then
explicitly "protecting" it changes nothing by construction. The colleague's own dose-response
comparison makes this concrete: INT8→INT2 quantization of the critical pair moves epistasis by
~0.00 pp at every precision, while zeroing the same pair moves it by −11.56 pp — the quantizer
literally cannot damage what it already preserves exactly.

**Q4 — Experimental** (destructive-regime benefit): **PARTIAL**. Per-tensor SW exemption
(where the SW is *not* automatically protected, since the scale comes from the global max):
mean SW-specific gain **+0.048 pp**, median **−0.035 pp**, range **[−2.13, +2.51]**, 9 negative
vs. 5 positive cells, **t ≈ +0.15** over 14 informative cells (3 tasks × 2 granularities ×
{INT4,INT3,INT2}, minus saturated cells). The colleague's own read: *"super-weight exemption
confers no reliable benefit at any granularity or precision tested."* Symmetric/zero-point:
symmetric, zero-point implicit at 0. Group sizes tested: per-row (full row), group-wise g=64
and g=16, per-tensor. Rounding: RTN (round-to-nearest) throughout — **never described or
tested as stochastic rounding.** No claim of bit-exactness is made beyond the specific,
measured 0.000e+00 element-level result (Q2) — the colleague does not generalize "bit-exact"
beyond that. No raw artifact for the destructive-regime numbers
(`scale_preservation_grouped.json`, `exemption_all_granularities_*.json`,
`compression_destructive_and_activation.json`, none present).

**Net effect on this branch's own claims:** Q1–Q3 substantially **strengthen** C-033 with
independent empirical measurement; nothing here contradicts it. Q4 is new, additional
supplementary detail (a destructive-regime null result) this branch does not currently have,
consistent with C-031's existing empirical null.

---

## 6. Supersession map — every relevant current-branch claim

| Current claim | Colleague evidence | Status |
|---|---|---|
| C-002/C-003 (row-ranking) | Not touched by colleague work | **Unchanged** |
| C-009/C-011 (Evo1 structural/causal) | §4.1 — corroborated, mechanistically deepened | **Unchanged, supplementary corroboration available once artifact recovered** |
| C-029 (NTv3 splice) | §4.3 — bug independently confirmed by this audit | **Still `contested`; bug diagnosis now resolved in the colleague's favor; corrected number still unresolved** |
| N-009 / C-001 (PROK ‖U_k‖_F non-reproduction) | §4.2 — wrong-probe diagnosis is a credible explanation | **Still `on hold`; a candidate cause now exists but its own artifact is unpushed** |
| N-015 (PROK wrong-probe allegation) | §4.2 | **The external work N-015 asked for has been located; its numeric artifacts have not** |
| C-032 (diagonal-PR granularity, already recommended RETIRE by this branch's own E7 work) | Colleague's T0.2 (`uk_precision_recall.csv`, prose): *"‖U_k‖_F... a necessary structural signature, not a universal predictor"* | **Directionally consonant with this branch's own E5–E8 conclusion (no clean domain-general predictor), not a new consideration** |
| C-033 (quantization scale-endpoint math) | §5, Q1–Q3 | **Strengthened by independent empirical measurement — no change to status, stronger evidentiary footing if imported** |
| C-020/C-021 PROK halves (`contested`, N-015) | §4.2 | **Same as N-015 above** |
| Nothing in E5/E6/E7/E8 | Colleague branch cannot touch these (§1 — `paper-salvage/` does not exist on it) | **Unaffected, structurally** |

**No current-branch claim is contradicted by anything on COLLEAGUE_BRANCH once traced
carefully.** Every apparent tension (Evo1, PROK) resolves into either "different quantity,
consistent once compared correctly" (Evo1) or "a credible candidate explanation for an
already-known gap, itself still missing its own artifact" (PROK, NTv3).

---

## 7. Integration safety map

### SAFE SCIENTIFIC IMPORT (result artifacts, scripts, tests — preserve as-is)

- `scripts/mechanism/*.py` (9 files: `run_attention_sink.py`, `run_codominance.py`,
  `run_compensation_circuit.py`, `run_direction_vs_magnitude.py`, `run_ensemble_encoding.py`,
  `run_norm_matched_control.py`, `run_pretrained_epistasis.py`, `run_steering_biological.py`,
  `run_sw_steering.py`) — new directory, no collision.
- `scripts/compression/run_{destructive_sw_protection,group_scale_preservation,
  pair_aware_compression,per_tensor_sw_exemption,proximity_confound_control}.py` — new files,
  no collision.
- `scripts/evaluation/run_sw_pairwise_epistasis.py` — new file, no collision.
- `sae/collect.py`, `sae/model.py`, `sae/train.py`, `sae/analyze.py` bug fixes (fp16 clamp →
  `--store_dtype float32`; `--standardize`; `data_scale` persistence) and
  `sae/analyze_real_sequence.py` (new) — current main has not touched `sae/` since the
  merge-base (verified, §1) — pure addition/fix, no collision.
- `scripts/detection/run_detection.py`'s `--pad_to_multiple` flag (NTv3 U-Net padding fix) —
  current main has not touched this file since the merge-base — pure addition.
- The `results/mechanism/*.md` reports themselves — genuinely valuable, self-critical
  methodology documents, safe to bring in as reference material regardless of the missing raw
  artifacts.

### NEEDS MANUAL RECONCILIATION

- **`README.md`** — both branches edited it since the merge-base (colleague +138/−X lines,
  current main +21/−13 lines independently). The colleague's version presents an entire
  "current status" narrative built around Paper B/mechanism findings, with no awareness of
  paper-salvage's later thesis pivot (Frobenius/dimensionality → exact operator geometry) or
  the reconciliation pass. A naive merge would produce two competing, uncoordinated
  "current state of the project" sections. Needs a human decision on which narrative — or
  what combination — the top-level README should present.
- **`scripts/evaluation/run_gue_ablation.py`** — both branches touched it since the
  merge-base. Structurally, the changes appear to target different, non-overlapping regions
  (current main added a new `_GeneratorClassifier` class; the colleague's change is entirely
  inside the existing `_NTv3Classifier.forward`, fixing the padding-multiple bug). This audit
  did **not** attempt a trial merge (out of scope, per instruction) — this read is a structural
  inference from the diffs, not a confirmed conflict-free result, and should be verified
  mechanically before any integration.
- **The PROK correction itself** (§4.2) — needs the missing `super_weight_index.json` edit
  (and its provenance block) either recovered from wherever the colleague actually has it, or
  reproduced from scratch using the already-present `pseudomonadota` probe and the unmodified
  `run_detection.py`, before anything citing L8/r260 can be written anywhere.
- **The NTv3 corrected retrain** (§4.3) — same situation: needs the actual corrected
  checkpoint/eval artifact recovered or reproduced before C-029 can be updated.
- **Any `CLAIMS_LEDGER.md` / `DECISIONS.md` / `PAPER_OUTLINE.md` entries** that would
  eventually reference A–I — none exist yet (this audit does not create them), and per this
  project's standing discipline, none should be written until each claim's own artifact is
  actually present.

### DO NOT IMPORT

- Nothing on this branch would overwrite or stale-ify current E5–E8 conclusions, because
  nothing on this branch touches `paper-salvage/` at all (§1). There is no file here that
  needs to be *excluded* to protect E5–E8 — the two trees simply do not overlap in that
  region.
- The colleague's `README.md` section should not be merged **wholesale** (see "needs manual
  reconciliation" above) — not because it is wrong, but because merging it silently would
  present it as this project's single current narrative without reconciling it against the
  later thesis pivot.

---

## 8. The DNABERT-pair → distributed-operator chain — explicitly assessed, not asserted

The prompt asks whether the DNABERT-2 pair/norm result can support:

> more distributed exact encoder operator → multi-parameter/redundant causal implementation →
> joint norm carriage / normalization failure

**This chain is not supported as a general statement, and the colleague's own evidence is why.**

- **Arrow 1 (distributed operator → redundant causal implementation):** holds for DNABERT-2
  (E8: exact `q1`=0.793, among the two most distributed operators measured; colleague: a
  genuine redundant-pair causal phenotype, joint epistasis −33.76 pp). **Fails for NTv3**:
  NTv3's exact operator is *more* distributed than DNABERT-2's under E8's own metric
  (`q1`=0.389 vs. 0.793), yet the colleague's E2 `MAKE-PAIR` intervention shows **imposing the
  identical co-dominance structure on NTv3 produces no causal effect at all**. Distributedness
  under the exact metric does not predict which models develop a redundant causal
  implementation.
- **Arrow 2 (redundant implementation → joint norm carriage):** holds, well-demonstrated, for
  DNABERT-2 specifically (§3.C). Not tested as a general claim anywhere in the colleague's
  work — n=1.
- **Arrow 3 (norm carriage → normalization failure generalizes):** explicitly **falsified** by
  the colleague's own `FALSIFICATION_NORM_VS_SUPERWEIGHT.md`: NTv3's SW channel is *more*
  norm-dominant (29.4× gap vs. DNABERT-2's ~4.5×) yet functionally inert.

**The chain holds, n=1, for DNABERT-2 alone, and the colleague's own second test case (NTv3)
breaks it at two of its three links — using the same architecture-adjacent reasoning E8 uses.**
This is consistent with, and adds real mechanistic texture to, this branch's own standing
discipline (`CLAUDE.md`: *"Do not assert causal arrows between routing and context-
sensitivity"*; C-011: *"structural concentration does not imply functional criticality"*). It
should not be written up as a confirmed or even a soft general chain — at most, as an n=1
case study with an explicit, evidenced n=2 counter-case, exactly as the colleague themselves
already frames it.

---

## 9. Recommended integration strategy

**C — create a new integration branch and manually reconcile both histories.**

Reasoning: a clean whole-branch merge (A) is not demonstrably safe — `README.md` carries two
independently-developed "current status" narratives that would silently collide, and every
scientific claim on the branch is at best PARTIAL by this project's own evidentiary standard,
meaning none of it is ready to enter `CLAIMS_LEDGER.md` without either recovering the missing
artifacts or re-running the (real, inspectable) scripts to produce them. A pure selective
cherry-pick (B) undersells the actual complexity here: the two touched-by-both files need real
reconciliation, not a mechanical pick, and the scientific content needs an explicit "these
claims are PARTIAL, do not cite as established" holding pattern that a bare cherry-pick would
not itself enforce. **D is too pessimistic** — the branch is not "too provenance-incomplete to
integrate scientifically" in general; it is one systematic, fixable gap (the `results/`
gitignore not carrying its JSON/CSV siblings) away from being very well-provenanced, and one
claim (NTv3, §4.3) is already fully audited independent of the colleague's own reports.

**Because COLLEAGUE_BRANCH is a single commit, "create an integration branch" in practice
means: branch from current `main`, cherry-pick the single commit `5b0220c` onto it (not onto
`main` directly), then do the manual reconciliation work (README, the one shared script,
recovering or re-deriving the missing artifacts) inside that integration branch before any of
it touches `main` or `paper-salvage/`.**

### Exact commit to import

- `5b0220c513b1cd7db379f3023ae04ad3fd5801aa` — the entire colleague contribution is this one
  commit. Cherry-picking it (onto a new integration branch, not `main`) is mechanically
  simpler than the "several distinct commits" framing Option B implies, precisely because
  there is only one.

### Exact files requiring manual reconciliation before/during that cherry-pick

- `README.md` (both sides edited since merge-base — see §7)
- `scripts/evaluation/run_gue_ablation.py` (both sides edited since merge-base — appears
  structurally non-overlapping but not mechanically verified — see §7)

### Everything else in the commit

Applies cleanly as new files with no current-branch counterpart (§7, "SAFE SCIENTIFIC
IMPORT") — mechanically safe to bring in on the integration branch; **still not safe to cite
scientifically** until each claim's own artifact is recovered or reproduced.

---

## 10. What this audit explicitly did not do

No merge, rebase, or cherry-pick was executed. No file on either branch was modified. No GPU
job, retrain, rerun, or new candidate search was performed. No colleague result was "repaired"
or patched. No `CLAIMS_LEDGER.md`, `DECISIONS.md`, `PROJECT_STATUS.md`, or `PAPER_OUTLINE.md`
entry was added or edited by this document. The worktree at `/work/11034/atzanakak/
glm_super_weight/worktrees/mechanism-and-negative-results` remains available for the next
session's use and was left untouched (detached HEAD, no local changes).
