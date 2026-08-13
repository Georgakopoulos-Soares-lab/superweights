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
