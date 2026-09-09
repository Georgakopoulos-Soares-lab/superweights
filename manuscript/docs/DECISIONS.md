# DECISIONS.md

Append-only. Never rewrite an entry — supersede it with a new one that references the old ID.

Format: `## D-nnn — <title>` / Date / Decision / Rationale / Consequences / Supersedes.

---

## D-001 — Restructure v15 rather than patch it
**Date:** Phase 0
**Decision:** Rebuild the manuscript around a new causal progression rather than revising
v15 section by section.
**Rationale:** v15's problem is organization, not results. It reads as "here are many
things we found about super-weights." Patching preserves that.
**Consequences:** Roughly half of v15 moves to supplement. Three new experiments become
blocking.

## D-002 — Lead with the method, not the survey
**Date:** Phase 0
**Decision:** ‖U_k‖_F becomes R1, validated against published NLP super-weights before any
genomic result is presented.
**Rationale:** It is the cleanest standalone contribution (closed form, zero forward
passes; Yu et al. require one). Leading with known ground truth buys credibility before
spending it on unfamiliar models.
**Consequences:** Adds the NLP validation as a blocking experiment (E1).

## D-003 — Mechanistic-interpretability spine, computational-biology venue
**Date:** Phase 0
**Decision:** Frame as mechanism/method; submit to a comp-bio methods venue.
**Rationale:** The biology finding ("encodes GC composition, no annotated regulatory
grammar") cannot carry a biology-discovery paper, but can carry the "what does the channel
compute" section of a mechanism paper. At an ML venue the panel is a liability; at a
comp-bio venue it is the asset.
**Consequences:** Intro needs an explicit "why genomics" paragraph. See PAPER_OUTLINE.

## D-004 — C is routing geometry, not a criticality predictor
**Date:** Phase 0
**Decision:** Reject the framing that (‖U_k‖_F, C) jointly predict criticality.
**Rationale:** DNABERT-2 has C ≈ 0 with the largest impulse KL (0.31) and the strongest
encoder phenotype (−25.5 pp splice). The joint framing is falsified by our own data.
**Consequences:** Three-stage structure adopted: structural amplification → routing
regime → functional recruitment.

## D-005 — Hold the T claim pending Evo1
**Date:** Phase 0
**Decision:** R3's thesis sentence remains a slot with two pre-drafted variants until the
Evo1 standardized broadcast run completes.
**Rationale:** In all three positives measured so far, criticality tracks T while C varies
wildly. That makes Evo1 the decisive single-variable test. Asserting it now would be
post-hoc on n=3.
**Consequences:** E2 is blocking for R3 only. R1, R2, R4, R6, R7 proceed in parallel.

## D-006 — Prospective prediction moves to NLP, not the genomic panel
**Date:** Phase 0
**Decision:** Run the timestamped prospective test on an NLP model Yu et al. did not
cover, not on a held-out genomic model.
**Rationale:** Every ablatable genomic model is already unblinded (GENERator EUK/PROK,
DNABERT-2, Evo1, HybridNA, MegaDNA). The only untouched one is Caduceus, which is the
CUDA-broken one — choosing it would trade a clean prospective test for a month of
infrastructure debugging.
**Consequences:** Folds into E1 at near-zero marginal cost.

## D-007 — Keep the shuffle controls
**Date:** Phase 0
**Decision:** Retain a compact shuffle experiment in main text (R4).
**Rationale:** The shuffle result and the broadcast result corroborate each other from
independent methods. EUK: no shuffle sensitivity at any order + C = 1.0. PROK:
sensitive at every order + diffusion by ~L14. This makes the EUK/PROK contrast a
mechanistic result rather than a sign-convention footnote.
**Consequences:** Phrase as joint difference, not a causal arrow.

## D-008 — Null steering is reported, not dropped
**Date:** Phase 0
**Decision:** Pre-commit to reporting a failed steering result in two sentences within R4.
**Rationale:** "Activation correlates with composition" and "the channel controls
composition" are different claims. A null steering result says the first holds without the
second, which makes R4 more credible, not less.
**Consequences:** Pre-committed language lives in `prereg/PREREG_steering.md`.

## D-009 — Corpus-prior test is conditional and two-way only
**Date:** Phase 0
**Decision:** Run the corpus-prior fork only if R4's interpretation remains unclear after
E3. Limit to two comparisons: raw marginal k-mer frequency vs. deviation from a
lower-order Markov expectation.
**Rationale:** The data (4,096-hexamer activation table) already exist, so cost is hours.
But the six-variant battery is exactly how this project scattered the first time.
**Consequences:** Not in Tier 0. Gated on E3's outcome.

## D-010 — Granularity is an axis, not a section
**Date:** Phase 0
**Decision:** Scalar-vs-row-vs-ensemble findings distribute across R2, R3 and R6. No
standalone "single-neuron analysis" section.
**Rationale:** Making it its own section re-creates the disease — a paper that is a list
of analyses. Scalar-vs-row falls out of the c_{k,i} decomposition; row-vs-ensemble falls
out of impulse spreading.
**Consequences:** Apple-style neuron results slot into existing sections.


## D-011 — Fixed-ε impulse protocol replaced by AC-relative ε; all five models re-run
**Date:** Phase 1
**Supersedes:** the protocol fixed in `PREREG_evo1_broadcast.md` v1 (locked, see LOCKS.jsonl)

**Decision:** Replace the fixed ε = 1.0 injection with ε scaled to the AC component of the
residual stream at the injection layer:

    ε_m = α · std( h_ℓ − mean(h_ℓ) ),   α = 0.01

Re-run all five models (GENERator EUK, GENERator PROK, DNABERT-2, NTv3, Evo1) under the
new protocol. Report only the re-run numbers in R3.

**Trigger:** The locked Evo1 run returned exactly 0.0 at every downstream layer (12–31) for
the SW row **and** for the random-coordinate control. Evo1's residual stream is effectively
unnormalised: median |h| across 4,096 dims reaches ~2.8e5 at layer 11 and ~4.2e6 from layer
13 onward (max ~1.29e9). bf16 ULP at 4.2e6 is ~3.3e4 and at 3e7 is ~2.6e5, so a fixed
ε = 1.0 is rounded away at injection for any coordinate. The assay had no dynamic range at
that scale. The null is a property of the instrument, not of the model.

**Rationale for AC rather than raw magnitude.** Scaling ε to |h[k]| or to median |h| tracks
the DC offset, which is precisely the component that carries no input-dependent signal.
Scaling to the AC component targets the quantity downstream computation actually depends on.
fp32 alone is not a fix: it recovers ~4 orders of magnitude but ε = 1.0 still rounds away at
the SW coordinate's own scale. fp64 would measure in a numerical regime the model does not
compute in.

**Rationale for re-running all five.** `PREREG_evo1_broadcast.md` v1 states: "Nothing about
the protocol may be adjusted after seeing Evo1's numbers. If a protocol change is
unavoidable, log it in DECISIONS.md and re-run the other four models identically." A
relative-ε Evo1 number sitting next to fixed-ε numbers for the other four reintroduces
exactly the protocol asymmetry that sank the v15 "eight models" claim.

**Expected cost.** Low. GENERator / DNABERT-2 / NTv3 activations are O(1–100) under
LayerNorm/RMSNorm, so α = 0.01 lands near ε = 1.0 for them. The re-run should reproduce the
existing T/C/KL values closely. If it does not, that is itself informative and must be
reported.

**Gate.** The re-run is conditional on the input-sensitivity check in
`PREREG_evo1_broadcast.md` v2 §Precondition passing. The reported bit-for-bit identity of
the residual from layer 13 onward is incompatible with Evo1 achieving PPL 3.11 if it holds
of the hidden state rather than of a summary statistic. If two clearly different inputs
produce an identical layer-20 hidden state, the trace is broken and no measurement through
it is interpretable — including the residual-attribution numbers already in the manuscript.

**Consequences:**
- `PREREG_evo1_broadcast.md` revised to v2 and re-locked; the v1 prediction is preserved
  verbatim and the relock is disclosed in Methods.
- C-012, C-013, C-014 move from `supported` to `pending-rerun` in CLAIMS_LEDGER.md.
- C-016, C-017 remain `pending`. R3's thesis sentence remains a slot.
- A third reporting branch is added to the prereg for the precision-floor / AC–DC outcome.

---

## D-012 — fp16 autocast removed from the impulse harness for StripedHyena
**Date:** Phase 1

**Decision:** `scripts/interpretability/run_sw_broadcast_impulse.py` no longer forces
`torch.autocast(dtype=torch.float16)` for Evo1. It casts everything except `poles` and
`residues` to bf16, matching the pattern already used in
`run_evo1_residual_attribution_fp32.py`.

**Rationale:** The blanket fp16 cast corrupts StripedHyena — the wrapper's own docstring
warns about it — and produced `KL_sw = nan` at every layer. This is a repo-level fix, not a
one-off workaround, and it affects any future StripedHyena run through this harness.

**Consequences:**
- Commit separately from the ε work, referencing `run_evo1_residual_attribution_fp32.py`
  as the reference pattern.
- Audit whether any existing Evo1 artifact in `results/keep/` was generated through the
  fp16 path. If so, mark it in its PROVENANCE.md as superseded.
- Environment note for reproducibility: run performed inside `evo2.sif` (Apptainer) with
  `evo-model` and `stripedhyena==0.2.2` installed to `~/.local`; there is no `evo` conda env
  on this node. Record the container hash in Methods.

**Independent consistency check worth keeping:** the measured residual freeze at
~4.2e6–3e7 reproduces the manuscript's "residual freezes at ≈3×10⁷" claim for row 3776 from
a different script. Log this in CLAIMS_LEDGER.md as corroboration of C-009's supporting trace.

## D-012a — Correction to D-012: reference script path
**Date:** Phase 1
**Corrects:** D-012 (not edited; append-only)

D-012 cites the reference fp32 pattern as
`scripts/interpretability/run_evo1_residual_attribution_fp32.py`. The actual path is
`scripts/analysis/run_evo1_residual_attribution_fp32.py`. Everything else in D-012 stands.

---

## D-013 — Impulse assay runs in fp32; α remains fixed at 0.01; per-layer headroom reported
**Date:** Phase 1
**Builds on:** D-011 (AC-relative ε). Does **not** supersede it.

**Decision:** Run the impulse trace in fp32 for all five models. Keep α = 0.01. Add a
per-layer headroom diagnostic, ε_effective / ULP(|h_ℓ|), to every output record.

**Trigger.** The STEP 2 gate measured Evo1's AC scale: std 3.33e5 at L11, 5.53e7 at L13+.
α = 0.01 gives ε ≈ 3.3e3 at injection. The bf16 increment is ≈2.0e3 at L11 but ≈3.3e4 from
L13 onward, so under bf16 the injected perturbation sits ~10× below representable precision
downstream unless the L12→L13 explosion amplifies it proportionally.

**Why this is disqualifying, not merely risky.** Whether that amplification occurs *is* T.
Under bf16 the instrument's validity is conditional on the quantity being measured, and a
flat result cannot be distinguished from T = 0. Branch C of the prereg would also be
unearnable, since it requires showing the perturbation was representable in the first place.

**Why fp32 rather than a larger α.** fp32 ULP is 0.5 at 4.2e6 and 128 at 1.29e9, giving
6,600× and 26× headroom respectively at ε = 3.3e3. It also leaves ε at ~6e-5 of the local AC
component, i.e. a genuine small-signal linear-response probe. Beating bf16 by raising α
would require α ≈ 0.1–0.5 — injecting 10–50% of the AC component, no longer small-signal and
at risk of nonlinearity. Precision is a property of the instrument; α is a property of the
protocol. Changing the former is calibration, changing the latter after seeing results is
tuning.

**Why all five and not Evo1 alone.** At magnitude ~100 the bf16 increment is ~0.78, so the
original ε = 1.0 was only ~1.3 ULP for GENERator / DNABERT-2 / NTv3 as well. If those runs
were bf16 they were precision-marginal and their existing T/C/KL values are suspect. Running
all five in fp32 is both the symmetric choice and the one that removes precision as a
confound everywhere. **Record the precision used by the original runs before re-running** —
if it was not fp32, note it as a defect in the superseded numbers.

**The headroom diagnostic is what makes a null reportable.** "The perturbation was 26× above
representable precision and still produced no downstream change" is a finding. "Everything
was zero" is not. This column must be present in every output record, for every model and
every control arm.

**Consequences:**
- Prereg v2 amended (see below) and locked before the re-run.
- α is **not** revised. Predictions and branches A–D are **not** revised.
- Branch C becomes earnable, conditional on the headroom column showing adequate margin.
- New pass/fail criterion: if headroom < 4× at any downstream layer for any model, the
  assay is under-powered at that layer and results there are reported as unmeasured, not
  as zero.

---

# Append to docs/prereg/PREREG_evo1_broadcast.md (v2), under "Amended protocol"

## Numerical precision (added per D-013)

The impulse trace runs in **fp32** for all five models. α remains fixed at 0.01 and is not
tuned per model. Predictions and reporting branches are unchanged.

Every output record carries a per-layer headroom column:

    headroom_ℓ = ε_effective_ℓ / ULP( max |h_ℓ| )

- headroom ≥ 4× — the layer is adequately powered; a null there is a measurement.
- headroom < 4× — the layer is under-powered; report as **unmeasured**, not as zero.

A flat result is interpretable only where headroom is adequate. This column is required for
every model and every control arm, and it is what allows Branch C to be earned rather than
asserted.

## v1 provenance (added per D-013)

v1 of this preregistration was version-controlled but not machine-locked; `LOCKS.jsonl` did
not exist at the time. Its provenance is the git commit that introduced it:

    commit: ____________________   date: ____________________
    (git log --follow --diff-filter=A -- docs/prereg/PREREG_evo1_broadcast.md)

This is disclosed in Methods. No retroactive LOCKS.jsonl entry is synthesised for v1.

---

## D-014 — DNABERT-2 runs on eager attention; Triton is the defective path
**Date:** Phase 1
**Builds on:** D-013. **Supersedes in part:** the STEP 3 finding recorded in
`experiments/E2_evo1_broadcast/INSTRUMENT_VALIDATION.md` §4 that "the three normalised
models were already fp32" — that holds for GENERator EUK/PROK and NTv3, **not** for
DNABERT-2's attention.

**Decision:** DNABERT-2 runs the eager PyTorch attention path for all impulse work.
`_dnabert2_force_eager_attention()` sets the remote module's `flash_attn_qkvpacked_func`
to `None`, selecting the branch the remote code already provides.

**Rationale.**

1. **The guard was inapplicable, not failed.** The Triton arm's masked-LM perplexity varies
   **8.48%** run to run (720.6 / 681.1 / 662.2). A ±1% tolerance cannot protect a quantity
   that has no fixed value. The comparison was ill-posed from the start.
2. **Triton never ran fp32.** `convert_dtype` in `BertUnpadSelfAttention.forward` casts qkv
   and the attention bias to **fp16** because the kernel accepts only fp16/bf16. DNABERT-2's
   attention has never run in fp32 at inference; the fp32 parameter dtype was cosmetic.
3. **Eager is better on both axes at once.** PPL 176.9 vs 687.9 (3.9× better) **and**
   bit-reproducible (0.000% spread). A change that had damaged the model would not improve
   perplexity and eliminate nondeterminism simultaneously.

**Disclosure — required, verbatim, in Methods.** The guard outcome is reported as it
happened and is **not** presented as a passed guard:

> Forcing DNABERT-2 onto its eager attention path changed masked-LM perplexity by
> Δ = −74.3% (687.9 → 176.9). The mechanism is the Triton kernel's fp16 cast of the query,
> key, value and bias tensors; the same kernel also made the forward pass nondeterministic,
> with perplexity varying 8.5% across identical repeats.

**Consequences:**
- C-014 is **retired**, not revised (see CLAIMS_LEDGER X-006). Its "largest impulse
  KL ≈ 0.31" was the noise floor of a nondeterministic kernel; the clean primary-dose value
  is 1.8e-8.
- X-003 is **live** again pending the re-measured DNABERT-2 C. `CLAUDE.md`'s "C is NOT
  necessary for criticality" rested on C-014 and is unsupported until the re-run reports.
- Phase 1b queue (logged, not started): C-010's L7/r603 source-vs-propagator resolution ran
  through Triton at inference and needs re-verification on eager. C-002 is weights-only and
  unaffected. C-027/C-028 are fine-tuning with dropout > 0 and already took the eager branch.

---

## D-015 — T and C are the primary impulse endpoints; KL is secondary/descriptive
**Date:** Phase 1
**Builds on:** D-011, D-013. **Resolves:** the STEP 3d blocker recorded in
`experiments/E2_evo1_broadcast/BLOCKED.md`.

**Timing — material to how this may be described.** STEP 3d was an *explicitly planned
pre-lock metric-validation probe*. This amendment therefore occurs **before** preregistration
lock, not after seeing confirmatory results. It must be documented as such wherever the
prereg is cited, and it must not be presented as a post-hoc rescue.

**Decision:**
- **T and C are the primary impulse-response measurements.**
- **KL is secondary and descriptive**, reported only where it is measurable above the
  model-specific noise floor.
- A model is **not** required to show measurable KL for the T/C routing assay to be valid.
- KL is **not** replaced by top-k displacement or token-local KL. Neither becomes a primary
  metric.
- The perturbation dose is **not** increased beyond the already-specified dual-dose protocol.
- Functional criticality continues to be measured independently by the existing
  ablation/downstream experiments, not by the impulse KL.

**Trigger.** STEP 3d found GENERator PROK's output KL below useful resolution at both
preregistered doses: 0.0 at α = 0.01 and 9.94e-7 at α = 1.0, with top-10 mean rank shift
0.000 and top-1 unchanged at both. Both candidate replacement metrics were measured and were
also flat for PROK.

**Consequences:**
- E2 proceeds to lock and to the five-model run.
- R3's endpoints are T and C; KL appears as a descriptive column with each model's noise
  floor beside it.
- The null rule is unchanged and still binding: no result is reported as a null unless
  headroom ≥ 4× at that layer **and** that model's noise floor is below the observed signal.
- No manuscript thesis is drawn from this decision.

---

# New-direction reconciliation, 2026-08-13

The decisions below were made in a single bounded reconciliation pass, triggered by a
downstream review that asserted a substantially revised scientific state relative to the
overnight-queue work above. That review's headline claims were traced against this
repository's actual artifacts before any of D-016–D-025 was written; the trace is
`docs/NEW_DIRECTION_EVIDENCE_AUDIT.md` and the corresponding ledger reconciliation is in
`CLAIMS_LEDGER.md` (N-012–N-015). **No experiment was run to produce these decisions.** Where
the review's claim had no local artifact, the decision below says so explicitly rather than
adopting the claim.

## D-016 — The v6 framing ("closed-form structural predictor" as the headline contribution)
is retired; a new thesis and R1–R6 outline are adopted
**Date:** 2026-08-13
**Supersedes:** D-001, D-002, D-003 (in framing; their factual content is not disputed)

**Decision:** The manuscript's organizing question moves from "here is a closed-form
predictor of genomic super-weights" to **"what survives, and what fails, when the NLP
single-super-weight concept is transferred to genomic foundation models?"** `‖U_k‖_F` moves
from headline method to a calibration/confirmatory structural signature (see D-017). The
outline in `PAPER_OUTLINE.md` is rewritten around six results (R1–R6); see that file for the
frozen skeleton and `docs/MANUSCRIPT_MIGRATION_MAP.md` for how existing prose maps onto it.

**Rationale.** `‖U_k‖_F` is a leaky, non-universal structural signature (DNABERT-2 already
contains a row ranked 706/768 under this repo's own audited C-010), and the construction is
substantially related to prior massive-activation/amplifier work (Sun et al. — see
`docs/REFERENCE_AUDIT.md`, still unresolved as a citation, not newly resolved here). Leading
a manuscript with it as *the* novel contribution overclaims relative to what the repo's own
audited results support. The retrospective NLP validation (C-004, C-005: rank 1/4,096 row and
scalar recovery, Llama/Mistral/OLMo) remains genuinely useful — as calibration evidence that
the detection apparatus works, not as the paper's central claim.

**Consequences:**
- `PAPER_OUTLINE.md` rewritten; old R1–R7 skeleton retired (content preserved via the
  migration map, not deleted).
- Manuscript title/abstract in `paper/main.tex` need rewriting in a future session — this
  pass does not rewrite manuscript prose (Task 5 is a migration map, not a rewrite).
- `CLAUDE.md`'s frozen thesis line is superseded; see D-024 for its replacement.

## D-017 — U_k/Frobenius is a confirmatory/calibration signature, not this paper's novel
methodological contribution
**Date:** 2026-08-13
**Supersedes:** D-002 ("lead with the method")

**Decision:** Retain the NLP retrospective validation (C-004, C-005) and the genomic
cold-weight ranks (C-002, C-003, C-010) as supporting/calibration material within R1. Do not
present `‖U_k‖_F` as a novel universal predictor anywhere in the manuscript.

**Rationale:** see D-016. Additionally, the E1 **prospective** arm (predicting an unseen NLP
model cold, then confirming) was never run or locked (`C-006` stays `pending`) and is not
required to make this point — R1's job under the new framing is to establish where the
transfer holds (three encoder/decoder gated-FFN architectures, cold weights) and where it
already visibly strains (DNABERT-2 rank 706/768; Evo1 fires on a structural candidate that
turns out non-functional), not to add a fourth confirmation.

**Consequences:**
- E1 prospective is **not** a blocking experiment for this manuscript. `PHASE_1_BLOCKING.md`
  and `PROJECT_STATUS.md`'s Tier-0 table are updated accordingly (Task 4).
- C-006 remains `pending` in the ledger, untouched, with no expectation it is filled before
  submission.

## D-018 — Broadcast/impulse (T/C/KL) work is supplementary; Evo1 Branch A/B and the
T-criticality thesis are not adjudicated in this pass
**Date:** 2026-08-13
**Supersedes:** D-004 (in part — the empirical finding "C is not necessary for criticality"
is retained), D-005 (the R3-thesis-slot mechanism is retired, not filled)

**Decision:** The E2 broadcast/impulse results (C-012, C-013, C-015, C-016) remain valid,
locked-protocol measurements and may appear as supplementary/methodological material or
model-specific descriptive evidence. They do **not** become a main-text section, and no
attempt is made in this pass to assign Evo1's T value to Branch A or B, or to decide whether
T predicts criticality (C-017 stays `pending`, explicitly not eligible as a headline thesis —
see the C-017 ledger note).

**Rationale:** the new central thesis (D-016) does not require a resolved T-criticality
claim, and forcing that adjudication now would be exactly the kind of "invent a new analysis
to rescue an old framing" the current pass is instructed not to do. The E2 results are real
and stay in the repository at full strength (C-012/013/015/016 strengths are unchanged) —
only their manuscript position changes.

**Consequences:**
- `PAPER_OUTLINE.md`'s R3 (old) has no direct successor section; its content is available for
  supplement or as local support inside another section if a specific new-headline claim
  needs it (none currently does — see `docs/MANUSCRIPT_MIGRATION_MAP.md`).
- Branch A vs. B remains an open, explicitly deferred author call, now with no manuscript
  deadline attached to it.

## D-019 — Evo1: the existing structural/systemic account stands; no numerical-saturation
"false positive" narrative is adopted without evidence
**Date:** 2026-08-13

**Decision:** Evo1 continues to be described exactly as the existing, audited claims already
describe it: structurally concentrated by `‖U_k‖_F` (top 1.2–4.1%) but non-load-bearing under
ablation (C-009, ΔPPL=0.0%), attributed to the StripedHyena mixer redistributing the
candidate channel rather than preserving it (`paper/main.tex:325-341`, already in the
manuscript). A downstream review's more specific diagnosis — the SW value pinned exactly at
2^24, invalidated by a failed "rescue" experiment — is **not adopted**: no artifact in this
repository names 2^24 (16,777,216) at any point in Evo1's trace, the repo's own documented
saturation points are quantitatively different (fp16 overflow at ~65,504, or a continued
climb toward ~1.29×10⁹ under the corrected fp32/bf16 trace), "rescue" appears nowhere in
connection with Evo1, and the "frozen residual" framing the new diagnosis would need was
already re-examined and attributed to a summary-statistic artifact in this repo's own N-002,
not a hard numeric ceiling.

**Rationale:** adopting a specific, differently-reasoned mechanistic story with zero local
support — and one that conflicts with an existing, already-corrected finding (N-002) — is
exactly the "reinterpret failed experiments" and "invent new analyses to rescue old claims"
behavior this pass is instructed against, just pointed at a different conclusion.

**Consequences:**
- `CLAIMS_LEDGER.md` C-009 is unchanged. No new claim row is added for the 2^24 saturation
  story.
- If the 2^24 diagnosis and rescue experiment are real work done outside this repository,
  the artifacts need to be committed before either enters the manuscript.

## D-020 — GENERator PROK: the C-001 hold is generalized to claims keyed to the same
contested artifact; the eukaryotic-probe contamination diagnosis and the L8/r260 relocation
are not adopted without evidence
**Date:** 2026-08-13

**Decision:** C-001 (PROK `‖U_k‖_F` rank at layer 2/row 1,927 — on hold since N-009, 2026-08-12,
because the stored artifact does not reproduce at its own layer) is unchanged. C-020 (EUK/PROK
r=±0.710 sign-convention claim) and C-021's PROK half (shuffle sensitivity) are downgraded from
`established` to `contested` because they are keyed to the identical layer-2/row-1927
artifact, not because a eukaryotic-probe contamination mechanism has been verified — it has
not (see N-015). The proposed replacement SW location (≈L8/r260) is **not** adopted as a
claim anywhere; no artifact names it.

**Rationale:** this achieves the practical outcome the downstream review's guardrails
require — the old PROK kingdom/sign/composition story is not presented as supported going
forward — through a reason this repository can actually audit (N-009's reproducibility
failure), rather than through an unaudited mechanism. It avoids both errors the current pass
is instructed against: defending a now-doubted old claim, and fabricating support for its
replacement.

**Consequences:**
- The PROK SAE result and the old kingdom-composition narrative are cut from main text (they
  were already flagged for demotion in `CUT_LIST.md`); the disclosed reason is the SAE's fp16
  clamp at ±60,000 against the PROK SW's own out_max of 506,014 (N-013), stated as a methods
  caveat, not as a "98% variance destroyed" measured figure.
- No layer search for an alternative PROK SW is performed in this or any pass unless
  explicitly instructed in a future session (this would require a forward-pass sweep, which
  is out of scope here).
- Flagged for the author in `docs/NEW_DIRECTION_EVIDENCE_AUDIT.md` and `CLAIMS_LEDGER.md`
  N-015.

## D-021 — NTv3: C-029 is neither confirmed nor retired
**Date:** 2026-08-13

**Decision:** C-029 (NTv3 5-seed splice replication, p=0.008) moves from `supported` to
`contested`. It is not presented as a confirmed positive replication in any new manuscript
material. It is also not retired, deleted, or replaced with a p≈0.48 null, because no
artifact in this repository supports the truncation-bug diagnosis or the retrained-checkpoint
null offered for retiring it — the only NTv3 splice p-value that exists anywhere in the repo
is C-029's own p=0.008, with real 5-seed controls.

**Rationale:** the downstream review's guardrails forbid presenting NTv3 as a positive
replication; this pass honors that without retracting a controlled result on zero evidence.
This is a direct conflict between instruction and existing measured evidence and is
deliberately left open rather than resolved by fiat.

**Consequences:**
- `PAPER_OUTLINE.md`/R2 (new) states DNABERT-2 as the currently-uncontested strong functional
  encoder example; NTv3's splice result is not cited as corroborating evidence pending
  resolution of N-014.
- A future session should either locate the truncation-bug fix and retrained-checkpoint
  artifacts, or explicitly decide (with the author) to retract C-029 on the strength of the
  claim alone — this pass does neither.

## D-022 — Compression/pruning narrative retired; quantization granularity (C-033) adopted
as its replacement, scoped to per-row only
**Date:** 2026-08-13
**Supersedes:** the "shadow redundancy" framing wherever it still appears in manuscript prose
(it was already retired at the claim level as X-001; this decision addresses the prose, which
had not caught up — see `docs/MANUSCRIPT_MIGRATION_MAP.md`)

**Decision:** R7 (old, "compression, one paragraph") is replaced by R6 (new): the finding
that per-row RTN quantization already implemented in this repo (`scale = max|row| / maxval`)
maps a row's max-magnitude element exactly to the quantizer endpoint by construction (C-033,
new, established directly from code — no run needed), which is consistent with (not proven to
cause) the existing C-031 null result that explicit SW exemption under INT4 makes no
measurable difference. This is stated strictly for the per-row rule as implemented; no
group-wise quantization scheme exists in this repo, and no "deliberately destructive regime"
result exists beyond the standard INT4 condition C-031 already covers (a more aggressive INT2
script exists but has no output artifact — see N-012/`NEW_DIRECTION_EVIDENCE_AUDIT.md` item 9).

**Rationale:** this is the one new-direction item where a genuine, code-verifiable
contribution is available without running anything, and it is a materially better R6 than the
retired pruning-tolerance story (already contradicted by the project's own pruning sweep,
X-001).

**Consequences:**
- `paper/main.tex`'s abstract line 111–113 ("The SW neighbourhood is extremely tolerant to
  pruning and INT4 quantization... suggesting refined SW-aware compression schemes are
  possible") still asserts the retired X-001 claim verbatim and must be rewritten in a future
  writing session — flagged in `docs/MANUSCRIPT_MIGRATION_MAP.md`, not rewritten in this pass
  (Task 5 is a map, not a rewrite).
- The full pruning sweep (near-SW/far-SW/random) stays in supplement per `CUT_LIST.md`,
  unchanged.

## D-023 — No further experiments before writing; NEXT_SESSION.md becomes a consolidation
handoff
**Date:** 2026-08-13

**Decision:** This pass is a bounded audit/reconciliation/documentation pass. No model was
loaded, no GPU job was launched, no script was run. `NEXT_SESSION.md` is rewritten as a
writing/consolidation handoff (Task 4), not an experiment queue. The next session's job is to
either (a) write manuscript prose against the reconciled ledger and outline, or (b) resolve
one of the explicitly flagged provenance conflicts (N-013, N-014, N-015) by locating or
reproducing missing artifacts — not to start new analyses.

**Rationale:** per the governing instruction for this pass, and because most of the
"headline" results handed down for this reconciliation turned out to have no artifact — the
highest-value next step is closing that gap or writing around it, not adding new open threads.

**Consequences:**
- `PROJECT_STATUS.md` and `NEXT_SESSION.md` updated (Task 4).
- Any future session that wants to run E3 (steering), retrain NTv3, re-run Evo1's secondary
  dose, search PROK layers, or investigate the 2^24 saturation claim needs its own explicit
  instruction to do so — none of that is queued by this pass.

## D-024 — Author decision: colleague mechanism-session results are adopted as established
facts, notwithstanding continued absence of raw JSON/CSV artifacts in this repository
**Date:** 2026-08-17
**Supersedes:** D-019 (Evo1), D-020 (PROK), D-021 (NTv3); `CLAUDE.md` §B in full; N-012's
blanket "no artifact, do not write" holding; the `contested`/`pending-rerun`/`on hold`
statuses this pass had assigned to C-001, C-020, C-021, C-029 pending artifact recovery.

**Decision:** The author reviewed `docs/COLLEAGUE_BRANCH_AUDIT.md` and
`docs/MISSING_COLLEAGUE_ARTIFACTS.md` and instructed: treat the colleague's mechanism-session
reports (branch `mechanism-and-negative-results`, commit `5b0220c`, integrated onto
`integrate/mechanism-and-negative-results`) as results facts. This is explicit author
adjudication of exactly the question `contested` status and N-012/N-014/N-015 were reserving
for the author ("neither 'still established' nor 'retired' is warranted without author
adjudication"). It is adopted as-is: the specific numbers and conclusions each report states
are treated as ground truth for this project going forward, including the reports'
self-disclosed caveats, limitations, and negative/null sub-results (e.g. the falsification
that norm dominance does not predict criticality in NTv3; ModernBERT-style hedges are not
applicable here but the same principle — report the caveats as part of the fact, do not
smooth them away — is retained).

**What does not change:** the raw JSON/CSV/checkpoint artifacts backing these reports are
still not physically present in this repository (`docs/MISSING_COLLEAGUE_ARTIFACTS.md`
remains accurate as an inventory of what is not committed). This decision changes the
project's *evidentiary policy* toward the colleague's reports specifically — their prose and
tables are now treated as sufficient — it does not retroactively manufacture the missing
files. A future session recovering those files should use them to double-check, not to
"re-decide," what this entry already settles.

**Consequences (enacted in the same pass as this entry, not deferred):**
- `CLAIMS_LEDGER.md`: C-001 resolved (L8/r260 adopted as the corrected PROK super-weight,
  L2/r1927 retired as probe-contaminated); C-020/C-021 PROK halves replaced with the
  corrected D1 rerun findings; C-029 updated to the corrected NTv3 result (MCC 0.86–0.91, SW
  ablation effect −0.02pp, old ΔMCC=−0.119 stays retired as the truncation-bug artifact it
  was); N-016/Evo1 addendum promoted from "not cited as evidence" to accepted supplementary
  detail on C-009/C-011; five new claims added (C-036 DNABERT-2 redundant pair, C-037
  pretraining-intrinsic, C-038 norm/codominance mechanism incl. the NTv3 non-generalization,
  C-039 attention sink, C-040 causal steering); C-033/C-031 gain the Q2/Q4 empirical results
  as established detail. N-012, N-013, N-014, N-015 marked CLOSED, pointing here.
- `CLAUDE.md` §B: every bullet superseded in place, dated, pointing to this entry and its
  replacement claim IDs — not deleted, so the project's own reasoning for why it withheld
  these claims for four sessions remains legible.
- `README.md`: the "reference material, not yet citable" framing for the mechanism-session
  section is replaced with a findings summary; the PROK caveat box is replaced with a
  supersession notice (L8/r260 adopted, the pre-existing write-direction-convention material
  on L2/r1927 is superseded, not merely disputed).
- `docs/MISSING_COLLEAGUE_ARTIFACTS.md`: header note added — recovery is no longer a
  blocker for claim status, only for independent reproducibility.

**Rationale:** this is a call only the author can make (whether to trust a collaborator's
reported results absent independently-verifiable raw data is a research-judgment call, not
something this pass's evidence discipline can resolve on its own), and it was made explicitly,
in response to a direct summary of exactly what is and is not independently verifiable. The
prior caution (D-019/D-020/D-021, N-012) was the correct default in the absence of that
adjudication; it is not treated as having been wrong, only as now superseded.

## D-025 — E9 opened: mechanistic tomography of high-gain FFN causal response
**Date:** 2026-08-22

**Decision:** Open E9 per an explicit, separate, detailed instruction (`next_prompt.md`,
2026-08-22) — this satisfies D-023's condition that any new experiment needs its own
explicit go-ahead, and `manuscript/CLAUDE.md`'s "do not launch GPU jobs... without an
explicit, separate instruction" constraint. E9 supersedes the never-run E3/C-025 steering
prereg (`docs/prereg/PREREG_steering.md`) as the operative GENERator causal-response design;
C-025 stays `pending` in its own right, not retroactively resolved. Does not reopen E5-E8,
does not add model families, does not touch structural `U_k` work.

**Provenance/environment findings this pass (recorded in full in
`experiments/E9_mechanistic_tomography/PROVENANCE_AND_BASELINES.md`):**
- **GENERator EUK basis is thin.** Only two candidate high-gain rows exist anywhere in this
  repository (layer 4, row 2371 rank 1; row 1522 rank 2 — both writing to the same output
  column 2536). The target 6-12 component basis Phase 1 asks for cannot be built from
  genuinely pre-existing detected candidates. Padding to 6-12 with structurally-arbitrary
  same-layer rows was considered and **rejected** — it would mix a real detected high-gain
  pair with rows that were never flagged as high-gain, undermining exactly the "declared
  high-gain basis" the experiment is supposed to probe, and risks the forbidden
  "invent another structural metric" / "search for a prettier super-weight" moves. **Per
  Phase 1's own written contingency, the GENERator arm is scoped down to a one-dimensional
  primary dose-response analysis on row 2371, with row 1522 as a real (not synthetic)
  secondary consistency check — not a multi-component observer-complexity comparison.**
  DNABERT-2 tomography proceeds at full scope (`n_D=10`, the pre-existing canonical
  ensemble).
- **DNABERT-2 revision pinning gap.** `run_pretrained_epistasis.py` and
  `run_gue_multiseed._load_model` do not pass the pinned HF revision
  (`7bce263b15377fc15361f52cfab88f8b586abda0`) that `models/dnabert2_wrapper.py` uses —
  fixed for E9 by passing `--code_revision` explicitly in every E9 invocation.
- **hg38 FASTA missing on this filesystem.** Every mechanism script defaults to
  `/data/nvidia/data/hg38/hg38.fa` (the colleague's original machine); this path, and any
  equivalent, is absent here. Downloaded the standard UCSC hg38 reference
  (`hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz`) to
  `data/reference/hg38/hg38.fa` (not committed — too large, gitignored like `results/`).
  GUE dataset (`/data/nvidia/data/gue/GUE`) is likewise absent and **not** fixed, since E9's
  primary DNABERT-2 endpoint (pretrained MLM loss) does not require it — the splice/GUE
  endpoint is out of scope for E9.
- **DNABERT-2 layer-9 norm hook inconsistency.** `run_compensation_circuit.py` hooks the
  *last* encoder layer generically; `run_codominance.py` /
  `run_direction_vs_magnitude.py` / `run_norm_matched_control.py` hook layer 9 explicitly.
  E9 standardizes on the explicit layer-9 hook for its own residual-norm secondary endpoint.

**Rationale:** every one of these is a documentation/scope call the E9 instruction itself
anticipated and pre-authorized (Phase 0's "record exact... previous intervention semantics,"
Phase 1's explicit GENERator scope-limitation clause) or a plain reproducibility-gap fix
(revision pin, missing input data) rather than a new scientific judgment call requiring
separate author adjudication.

**Consequences:**
- `experiments/E9_mechanistic_tomography/PROVENANCE_AND_BASELINES.md` and
  `INTERVENTION_BASIS.md` created this pass.
- GENERator EUK will not receive an F0-F3 observer-complexity ladder or a pairwise-lifted
  design in E9 — only a dose-response curve. This must not be silently generalized later
  into "GENERator is additive" language beyond what a 1D curve supports.
- Baseline regression (Phase 0) proceeds next against the newly-fixed environment.

## D-026 — E9 results frozen: DNABERT-2 requires pair terms at both scales; bootstrap bug
caught and fixed before trusting the CI
**Date:** 2026-08-22

**Decision:** E9's mechanical decision is frozen as **PAIR_TERMS_REQUIRED** for DNABERT-2
at both `epsilon=0.5` and `epsilon=1.0` (held-out F2→F3 relative MAE improvement 54.7% and
24.4% respectively, bootstrap 95% CI excluding zero at both scales: (0.0038, 0.0186) and
(0.0614, 0.1144); known critical pair L9r264+r294 ranks #1 of 45 pairs by |Gamma| at both
scales — H5 confirmed). GENERator EUK's row 2371 shows a large (~390×random-control),
mostly-monotonic dose-response; row 1522 is non-monotonic (GC rises at partial suppression,
falls at full ablation) — reported as observed, not smoothed. Full numbers, F0-F3 tables,
and the Branch determination are in `experiments/E9_mechanistic_tomography/RESULTS.md`.

**Bug caught before trusting results**: the first `run_fit_observers.py` pass produced a
bootstrap CI with identical upper/lower bounds (zero width) at both epsilons. Root cause:
`bootstrap_mae_diff` reconstructed the held-out response from stored per-batch data as a
**raw aggregate MLM loss** (~4.7-5.2) but compared it against F2/F3 predictions fit on the
**baseline-subtracted `dloss` scale** (~0.01-0.5) — a ~4.7 constant offset that swamped any
real signal and produced a degenerate, wrongly-signed result (point estimate said F3 was
24% better at epsilon=1.0; the buggy bootstrap's mean was negative, implying the opposite).
Caught by comparing the bootstrap's zero-resampling limit against the independently-computed
point-estimate MAEs, which did not match. Fixed by resampling the baseline with the *same*
batch-index draw as the held-out response (paired resampling) before subtracting. Re-run
confirms the point estimates were always correct (only the bootstrap CI was wrong) and both
epsilons now show a correctly-signed, sensible-width, zero-excluding CI. No prereg
threshold, mask, or fitted coefficient was touched by this fix — it corrects only the
uncertainty quantification around an already-computed point estimate.

**Consequences:**
- `run_fit_observers.py`'s `bootstrap_mae_diff` now takes `baseline_per_batch` and resamples
  it jointly with the held-out per-batch data.
- `experiments/E9_mechanistic_tomography/RESULTS.md`, `HVP_FEASIBILITY.md` written this pass.
- New claims for `CLAIMS_LEDGER.md`: DNABERT-2 pair-requirement under finite forward
  intervention (both scales, bootstrap-confirmed) and GENERator's scale-limited dose-response
  finding — added in the same commit as this entry, per the ledger-discipline rule.

## D-027 — E10/E10b reconciled: architecture-level headline shifts from q1-structural split
(C-034) to causal-response complexity; C-040/C-045 flagged as an unreconciled pair pending E12
**Date:** 2026-08-23
**Supersedes:** none directly; narrows C-034's role established by D-016/D-017.

**Decision:** E10 (`experiments/E10_nlp_architecture_causal/ARCHITECTURE_SYNTHESIS.md`, commit
`626cddd`, locked prereg `PREREG_E10_nlp_architecture_causal_v2.md`) and E10b
(`experiments/E10b_phi3_tomography/E10B_SYNTHESIS.md`, locked prereg
`PREREG_E10b_phi3_tomography.md`) both completed 2026-08-22/23 under explicit prior
authorization, but neither was folded into this ledger, `PAPER_OUTLINE.md`, or the
`PROJECT_STATUS.md` session log before now. That gap is closed by this entry and the two new
claims below (C-046, C-047). `PAPER_OUTLINE.md` itself is **not** rewritten in this pass — it
still reads as of 2026-08-13 and needs its own update to home these results in an R-section;
that is flagged as a `next` item, not done here.

Net finding: across the 7-model E10 panel, 4/5 decoders (Llama, Mistral, OLMo, Qwen2.5)
mechanically resolve SINGLE_COMPONENT_DOMINANT while 2/2 encoders (MosaicBERT, ModernBERT)
resolve PAIR_TERMS_REQUIRED at both intervention scales — but Phi-3 (decoder) is an explicit
counterexample (MULTI_COMPONENT_CANDIDATE, C1=0.333), and E10b's follow-up tomography on Phi-3
finds the causal decision itself splits by intervention strength (PAIR_TERMS_REQUIRED at
ε=0.5, MIXED_OR_UNRESOLVED at ε=1.0, driven by a specific three-row layer-2 redundancy break).
OLMo additionally shows structural-rank/causal-rank dissociation within its own row set (the
causally dominant row is not the structurally largest one). This causal-response-complexity
result — not q1 magnitude — is now the paper's most developed architecture-level finding.

This narrows, but does not retire, C-034 (q1 tracks encoder/decoder organization more than
domain). C-034 was already hedged at birth (2026-08-14: "not a clean universal domain split")
and D-017 already established that structural signatures generally are confirmatory/
calibration, not novel headline material. What changes here is explicit: with E10/E10b's
causal-complexity result now the stronger, more developed architecture-level claim, C-034 is
confirmed as secondary/calibration only, and any pending scale-confound check on it (E11) is a
check on calibration material, not on the manuscript's headline.

Separately, this entry flags an unreconciled tension between **C-040** (colleague-adopted,
5-point grid, GC saturates then *reverses* at 5×, span ratio 38.59×) and **C-045** (E9's own
locked measurement, 3-point grid restricted to α∈[0,1], ~390× span, "mostly monotonic" in that
range). Both are marked `established` without being checked against each other — C-045 was
never run at amplification (α>1), so it cannot confirm or refute C-040's reversal. This is
left open pending E12 (see N-017), not resolved here.

**Rationale:** Per `CLAUDE.md`'s working discipline ("update `PROJECT_STATUS.md` at the end of
any session that changes state," "every decision that changes scope gets an entry in
`DECISIONS.md`"), a completed, artifact-backed experiment sitting unlogged for a session is
exactly the kind of untracked-state failure mode this project has already been burned by twice
(v15, the 2026-08-13 reconciliation). Surfaced now, before E11/E12 preregs are drafted on top
of a ledger that doesn't reflect current evidence.

**Consequences:**
- `CLAIMS_LEDGER.md`: new rows C-046 (E10 architecture-level causal-complexity synthesis) and
  C-047 (E10b Phi-3 epsilon-split result), added in this same commit; C-034's row annotated to
  point to C-046 as the current primary architecture finding; new ledger note N-017 on the
  C-040/C-045 tension.
- `PROJECT_STATUS.md`: session log entry added for this reconciliation pass.
- `PAPER_OUTLINE.md` is explicitly **not** updated this pass — still stale as of 2026-08-13,
  R-section homes for C-046/C-047 undecided. Flagged as a `next` item.
- E11's protocol (drafted next) is framed as testing C-034 specifically, with a decision rule
  that allows cutting C-034 outright, not just softening it — consistent with its
  calibration-only status confirmed here.
- E12's protocol (drafted next) treats resolving the C-040/C-045 tension (via a row-2371
  amplification arm) as a secondary, separately-reported question from the primary
  damage-matched-control comparison.

## D-028 — E10 decoder-arm protocol correction relocated to `docs/history/`; pointer added
because it was the only surviving record of the change

**Date:** 2026-08-26 (relocated; original correction dated 2026-08-22 by file mtime, prior to
any E10 decoder causal measurement).

**Decision:** A root-level, untracked file (`e10_prompt_correction.md`) contained the actual
instruction that changed E10's decoder arm from a single-row alpha-sweep design to the
structural-top-K-freeze / singleton-causal-concentration design (Steps D1–D5: freeze
structurally ranked top-K rows + 5 random controls before measurement; measure each row's
singleton causal effect; compute concentration statistics C1/C2; apply a preregistered
Case A/B/C decision rule — top-1-dominant stops at the singleton spectrum, comparable-effect
rows justify tomography, a weak/null top structural row is reported as dissociation, not
rescued post hoc). Round-2 audit (`audit_2_prompt.md`) found this correction's substance is
**not** duplicated in `experiments/E10_nlp_architecture_causal/PROTOCOL_CORRECTION_01.md` (0
matches for "decoder arm" / "Revised decoder") — it was an orphaned, uncommitted decision
record. Moved to `docs/history/e10_prompt_correction.md` and committed so it survives a fresh
clone; this entry is the pointer so a future reader finds it from the ledger rather than by
accident.

**Consequences:** none to current claims — this documents pre-existing E10 design history,
it does not change any result. No `CLAIMS_LEDGER.md` rows affected.
