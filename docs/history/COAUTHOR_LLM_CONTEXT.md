# Coauthor LLM Context — "Structure Is Not Mechanism"

> This document is an explanatory companion to the current manuscript. If wording here
> conflicts with the newest manuscript, the manuscript governs the paper's current
> presentation; if an implementation detail conflicts, the repository artifact/script
> governs what was actually run.

**How to use this file.** Upload this file together with the current manuscript PDF (source
file `new v9(1).pdf`; a converted, searchable copy lives in the repository at
`manuscript/actual_manuscript.md`) to an LLM, and ask it anything about the paper's design,
terminology, controls, provenance, or interpretation. This document is not a summary of the
paper — the paper does that job well enough on its own. It exists to carry the *scaffolding
around* the paper: why it looks the way it does now, what it used to claim, which numbers are
load-bearing versus illustrative, and where the manuscript's own text is still imprecise or
under-specified. Everything quantitative below was traced to a specific script or artifact in
the repository (`https://github.com/Georgakopoulos-Soares-lab/superweights`, commit
`e89968f4059f17f03fb6ea93d053397e2a367f75` at the time this document was written); nothing
here should be treated as more authoritative than the manuscript's own Results/Methods text
for what the paper *says*, or than the cited artifact for what was *run*.

---

## 1. Executive overview

The paper studies "high-gain rows" — output rows of gated feed-forward (SwiGLU/GEGLU-style)
blocks that behave, structurally, like the "super weights" Yu et al. described in large
language models — and asks whether an analogous phenomenon recurs in genomic foundation
models, and if so, whether its structural signature tells you anything about its functional
importance. The honest answer the paper now gives is: **the structural phenomenon recurs
broadly across text and genomic models, but structural prominence, functional criticality, and
the causal complexity of a model's response to intervention are three separate quantities that
do not reduce to one another.** That three-way dissociation — not a new detector, not a new
universal mechanism — is the paper's actual contribution, and it is why the title is now
*"Structure Is Not Mechanism"* rather than anything promising a predictor of causal importance.

Concretely, the paper does three things. First, it defines an **exact, weight-only, bilinear
operator** `U_k` associated with each candidate output row (no forward pass, no gating
nonlinearity), computes its exact Frobenius norm and full singular spectrum, and uses the
leading singular-value share `q1` as a concentration metric. Applied to 23 foundation models
(11 text decoders, 6 text encoders, 4 genomic decoders, 2 genomic encoders — Evo2-7B was
tested but produced no accepted candidate), this shows that high-gain rows are, almost without
exception, the highest-norm row in their own layer by a wide margin, and are typically
near-rank-one in spectral concentration — but concentration itself does not cleanly separate
by domain or architecture, and once you control for the fact that these rows are simply the
largest rows in the layer, most of their apparent structural "specialness" disappears. Second,
it runs a **frozen, 22-model causal census**: one candidate row and five same-layer control
rows per model, evaluated under 50% suppression and full ablation, measured as a signed change
in each model's own native loss. The candidate usually beats the controls — including a much
stricter control set built purely from weight magnitude, with no forward pass at all — but the
*size* of that advantage ranges from negligible to a nearly seven-fold loss increase, and
neither the candidate's spectral concentration nor its operator magnitude predicts which regime
a given model falls into. Third, two **focused mechanistic case studies** — DNABERT-2 and
GENERator — show that even "the causal effect exists" does not mean the same thing twice:
DNABERT-2's effect is a genuine, held-out-validated *pairwise interaction* between two rows
that is invisible to either row alone; GENERator's effect is a large, real compositional shift
that tracks *how much damage was done* to the model rather than *which specific learned
direction* was removed.

The reason this is no longer simply "super weights in genomic models" is that the original
framing implied a transferable, closed-form, weight-only predictor of causal importance — the
kind of thing you could compute once and trust everywhere. Roughly a year of iteration on this
project (see §2) found, repeatedly and from several independent angles, that this does not
hold: the same weight-space geometry that looks exceptional against arbitrary controls looks
unremarkable against the five next-largest rows in the same layer; a model's most
spectrally-concentrated row can be the *least* causally important row it has; and two models
that both show "the candidate matters" can be causally important for structurally
unrelated reasons. The paper now reports this as the finding, rather than as a limitation to
be minimized. A coauthor coming back to this project after a few weeks away should expect the
emphasis to have shifted from *"we found a predictor"* to *"we characterized where a
plausible-looking predictor breaks, and by how much."*

---

## 2. How the project changed

The manuscript has gone through more than one distinct scientific framing, and the current text
is the result of several deliberate, documented pivots — not organic drift. The project's
internal decision log (`manuscript/docs/DECISIONS.md`, append-only, entries D-001 through
D-028) and its removed-claims ledger (`manuscript/docs/CUT_LIST.md`) record these explicitly,
and are worth knowing about because **old drafts, old prose, and even the file `paper/main.tex`
on disk still reflect an earlier, retired framing** — if you or an LLM ever consults that file
directly, treat it as historical, not current.

**Framing 1 (earliest, "v15" and the file `paper/main.tex`).** The paper was organized as a
survey: "here are many things we found about super-weights" across roughly eight genomic
models, built around `‖U_k‖_F` (at the time, a *diagonal approximation* to the operator, not
today's exact bilinear form) as the headline, novel, closed-form predictor of super-weight
identity. This framing carried several claims that did not survive later scrutiny and were
formally retired rather than quietly dropped: a "shadow redundancy" / SW-neighborhood pruning
tolerance claim (contradicted by the project's own pruning sweep, X-001); the phrase "eight of
the best genomic language models," used as though this were a uniform, representative
benchmark (replaced by an explicit coverage table); a claim that `(‖U_k‖_F, C)` jointly predict
causal criticality (rejected once DNABERT-2 showed near-zero coordinate-preservation `C` with
the strongest measured encoder ablation phenotype in the panel — see D-004); and a claim that
INT4 quantization / pruning results "suggest refined SW-aware compression schemes are
possible" (the underlying results were mostly null at the tested fractions and did not support
this).

**The D-016 pivot (2026-08-13).** A reconciliation pass explicitly retired the "closed-form
structural predictor" framing as the paper's headline contribution. The stated reason: `‖U_k‖_F`
is a "leaky, non-universal structural signature" — the project's own audited results already
contained a counterexample (a DNABERT-2 row ranked 706th of 768 in its own layer under an
earlier version of this metric) — and leading a manuscript with it as *the* novel contribution
overclaimed relative to what the data supported. The organizing question moved to: **"what
survives, and what fails, when the NLP single-super-weight concept is transferred to genomic
foundation models?"** The retrospective NLP validation (recovering the published Llama/Mistral/
OLMo super-weight rows) was retained, but reframed as *calibration evidence that the detection
apparatus works*, not as the paper's central claim (D-017).

**A colleague-branch adoption, then independent re-verification (D-024, then E9–E13).** In the
interim, a separate branch of mechanism-session work reported several striking findings — a
DNABERT-2 redundant/epistatic pair, a GENERator BOS attention-sink phenotype, a GENERator causal
GC-steering effect — without the underlying raw JSON/CSV artifacts being present in this
repository. On 2026-08-17 the author made an explicit, documented judgment call (D-024) to treat
those reported results as facts pending independent verification, rather than either discarding
them for lack of local evidence or silently trusting them forever. That verification then
happened: the E9 mechanistic-tomography experiment (2026-08-22) independently re-derived the
DNABERT-2 pairwise interaction from a from-scratch environment and reproduced a previously
reported epistasis figure to four decimal places; the GENERator BOS phenotype and GC
dose-response were independently re-measured with real artifacts (`results/mechanism/
attention_sink_implicit_bias.json`, the E9/E12 damage-matching pipeline). **Everything about
DNABERT-2 and GENERator now stated in the manuscript rests on these independently reproduced
measurements, not on the original unverified colleague reports.** A few of the colleague
branch's more specific claims — a downstream splice-accuracy magnitude for the DNABERT-2 pair,
a "codominance" ratio-sweep magnitude — turned out, on fresh single-seed reproduction, to
materially conflict with the earlier reported numbers; those specific magnitudes are **not**
in the current manuscript and should not be resurrected (see §15).

**The scale-up from case studies to a 22/23-model census (E10 → E13, mid-to-late August).**
The paper's architecture-level claims used to rest on a handful of hand-picked models (five
decoders, two encoders). A deliberate escalation (D-025 through D-027) built the frozen,
pre-registered 23-model structural panel and 22-model causal census that the current manuscript
actually reports. This is the single biggest structural change since D-016: the paper's
headline numbers (18/20 and 20/19 models exceeding controls, the 69-fold q1-gap collapse
against top-norm controls, the near-zero q1/effect correlation) all come from this expanded
census, not from the earlier five-model architecture study. The five-model study's own findings
— Phi-3 as a decoder that needs pairwise interaction terms, OLMo's structural/causal rank
dissociation within its own four rows — are qualitatively consistent with, and were an early
warning sign for, the census's headline dissociation result, and remain in the paper as
illustrative material even though they predate the full census.

**What survived across all of this:** the core empirical fact that high-gain rows recur
structurally across text and genomic gated-FFN models, the DNABERT-2 pairwise-interaction
result, and the GENERator damage-tracking result. **What did not survive:** the idea that any
of the paper's structural descriptors (`q1`, `‖U_k‖_F`, or a diagonal proxy for either) function
as a general-purpose, transferable predictor of causal importance. **The current title —
"Structure Is Not Mechanism" — is the direct, intentional statement of that outcome, not a
hedge.** Do not resurrect language implying a validated closed-form causal predictor, a clean
architecture/domain split in structural concentration, or that any single case study (DNABERT-2
or GENERator) establishes a universal mechanism.

---

## 3. Current claim hierarchy

| Claim | Evidence | Scope | What it does NOT imply |
|---|---|---|---|
| High-gain rows recur structurally across text and genomic gated-FFN models | 23-model structural panel, exact `q1` range 0.389 (NTv3) to 0.9996 (EuroBERT-610M); near-rank-1 candidates in both encoders and decoders | All 23 models with an accepted activation-based candidate | A clean domain or architecture split; that concentration level itself means anything about function |
| Candidates are structurally extreme relative to *random* same-layer rows | 22-model census, candidate beats all 5 random controls' `q1` in every one of 12/22 models directly measured this way; median gap 0.935 (22-model) / mean 0.884 (12-model, a distinct earlier statistic) | 22-model causal cohort (12-model subset for the mean+range statistic) | That the candidate is exceptional *among large rows*, not merely among typical ones |
| `q1` is not independently distinctive once conditioned on magnitude | Against the 5 highest-*Frobenius-norm* rows in the same layer (no forward pass), median `q1` gap collapses 69-fold, from 0.935 to 0.014; reverses sign in 7/22 models | Same 22-model cohort | That magnitude alone determines concentration, or that the candidate is never the most concentrated row locally — it usually still is, just not by a wide margin |
| Activation-derived candidates are functionally enriched relative to same-layer controls | Candidate exceeds median random control in 18/22 (ε=0.5) and 20/22 (ε=1.0) models; exceeds median top-norm control in 20/22 (ε=0.5) and 19/22 (ε=1.0) models | 22-model singleton causal census | A universal effect (NTv3 and EuroBERT-2.1B are non-positive at full ablation; four other models show a sign flip against the stricter top-norm control) |
| Simple Frobenius-magnitude ranking does not reproduce causal enrichment | Top-norm controls — the largest available weight-only alternatives — have median causal effect +0.035%/+0.110% (ε=0.5/1.0), close to inert, versus candidate effects reaching several hundred percent | Same census | That magnitude is causally irrelevant, or that no nonlinear magnitude threshold exists at the extreme (rank-1) end |
| `q1` and operator magnitude do not robustly calibrate causal effect *size* | `q1` vs. signed full-ablation effect: Spearman ρ=0.074 (p=0.744, 95% CI −0.391 to 0.513, n=22); layer-relative Frobenius ρ=0.441 (p=0.040) at ε=1.0 only, not significant at ε=0.5, and reverses sign within the text-decoder subset | Full 22-model panel and text-decoder subgroup | A validated predictive relationship between either descriptor and effect size, at either scale, robustly |
| DNABERT-2 has strong pairwise dependence on its pretrained objective | Direct joint ablation of L9/r264+L9/r294 (epistasis +2.0118 at ε=1.0, +0.0528 at ε=0.5); held-out F2→F3 MAE improves 54.7%/24.4% with bootstrap CI excluding zero; the pair ranks #1/45 by fitted coefficient at both scales | DNABERT-2, pretrained masked-LM endpoint only | Anything about DNABERT-2's fine-tuned/downstream task performance; that DNABERT-2's intrinsic causal dimension is exactly 2 (basis-dependent) |
| GENERator has damage-linked compositional fragility at row L4/r2371 | Ablating row 2371 drops generated GC from 0.4204 to 0.3065; a random unit direction at matched damage reproduces the same low-GC phenotype; composition tracks damage magnitude monotonically as damage is relaxed | GENERator-EUK-3B, this one row and its 5 non-2371 controls | That the row's *specific learned direction* is required (direction-independence is not fully established either — see §9); that BOS attention co-localization is causal |
| No universal mechanism is established | Singleton criticality, pairwise interaction, and damage-tracking fragility are all observed, in different models, for structurally similar candidates | Whole panel | That any one case study generalizes to the others, or to models not tested |

This table is the paper's actual backbone; nearly every FAQ answer in §14 traces back to one row
of it.

---

## 4. Model accounting and candidate provenance

Three different model counts appear in the manuscript, and they are all internally consistent —
but the manuscript's own Methods paragraph never states this in one place, which is the single
most common point of confusion for a first-time reader (it was, in fact, question #4 of an
internal audit that this section directly answers).

- **23-model structural panel** (Fig. 1B; "23 foundation models with accepted activation-based
  candidates"): 11 text decoders + 6 text encoders + 4 genomic decoders + 2 genomic encoders.
  Evo2-7B was tested with the same detector but is **excluded** — its maximum activation ratio
  was 2.22, below the acceptance threshold of 5.0, so no candidate was accepted and no `q1`
  value exists for it. It appears in Fig. 1B only as an annotated null.
- **22-model causal census** (Fig. 2; Methods, "Cross-model singleton functional-criticality
  census"): the same 23-model list **minus Phi-3-mini**. Phi-3 is excluded from this cohort by
  explicit scope definition, because its pre-existing basis is six rows, not one comparable
  primary row — it would not be a like-for-like singleton intervention. Phi-3's tomography
  remains a separate mechanistic case study (see §14, "Why is Phi-3 tomography still in
  Methods?").
- **"16 ratio-detector + 6 grandfathered-activation = 22" provenance breakdown** (Methods,
  "Candidate provenance"): this paragraph's scope is the **22-model causal cohort**, not the
  23-model structural panel, and the paragraph never says so explicitly — that omission is what
  makes the arithmetic look off if you check it against the 23-model list. Phi-3 is the missing
  model; it was selected by a *third*, undocumented-in-that-paragraph protocol (see below).

**Full 23-model structural list, by group:**

- *Text decoders (11):* Llama-7B, Mistral-7B, OLMo-7B, Phi-3-mini, Qwen2.5-7B, Qwen2.5-0.5B,
  Qwen2.5-1.5B, Qwen2.5-3B, SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B
- *Text encoders (6):* MosaicBERT, ModernBERT-base, ModernBERT-large, EuroBERT-210M,
  EuroBERT-610M, EuroBERT-2.1B
- *Genomic decoders (4):* GENERator-v2-eukaryote-3B, GENERator-v2-prokaryote-1.2B,
  GENERator-v2-prokaryote-3B, GenomeOcean-4B
- *Genomic encoders (2):* DNABERT-2, NTv3
- *Tested, no accepted candidate:* Evo2-7B

**22-model causal cohort** = the same list minus Phi-3-mini (10 text decoders, 6 text encoders,
4 genomic decoders, 2 genomic encoders).

**Candidate selection protocols (three, not one):**

1. **Ratio-detector selection (16 models).** A candidate row is accepted if
   `channel_max[layer,row] / median_row(channel_max[layer,:]) ≥ 5.0`, where `channel_max` is
   the maximum absolute activation over all token positions in a single forward pass over a
   fixed probe (first 20 non-empty WikiText-2 raw-test lines, truncated to 512 tokens, for text
   models; a fixed 504-bp ACTB-derived sequence for genomic models). This is the "current" /
   "final" detector and is what accepted every model not named below.
2. **Grandfathered activation-magnitude selection (6 models): Llama-7B, Mistral-7B, OLMo-7B,
   DNABERT-2, NTv3, GENERator-EUK-3B.** These coordinates were fixed under an earlier
   activation-magnitude-only convention (no ratio computed the same way) and retained rather
   than reselected. For Llama/Mistral/OLMo, the coordinates are literally the ones published by
   Yu et al. and serve as calibration; the ratio detector *independently* recovers the same
   coordinate in Llama and Mistral, but **not** in OLMo (the ratio-maximum falls at L2/r269, the
   activation-maximum at L30/r269 — both part of a "row-269 family" recurring across depth, not
   a single detector-determined coordinate). DNABERT-2 and NTv3 also show detector/activation
   disagreement (see below).
3. **Phi-3's own protocol (1 model, structural panel only).** Phi-3's six frozen rows were
   selected by ranking rows by exact `‖U_k‖_F` divided by the layer median and taking the top 6
   — the same style of top-K-by-norm rule used for MosaicBERT/ModernBERT-base's earlier 10-row
   bases, not the ratio detector and not the single-row activation-magnitude rule. Fig. 1B's
   single Phi-3 point is the **median `q1` across all six of these rows** (0.9028), not a single
   primary coordinate — this is self-documented in the underlying artifact
   (`results/E11/scale_ladder.csv`), which also leaves Phi-3's Frobenius-norm field blank
   because no single norm is meaningful for a six-row median.

**Special per-model provenance notes:**

- **DNABERT-2.** The frozen candidate L5/r603 is the global *activation* maximum under
  canonical discovery preprocessing (tokenizer-default special tokens; `out_max=944.5556`,
  ratio 152.73) — but it is **not** the global *ratio* maximum, which falls at L8/r603 under a
  different preprocessing (special tokens disabled). The two candidates disagree substantially:
  the ratio-selected coordinate has `q1=0.379` (below the panel minimum), while the frozen,
  retained candidate has `q1=0.793`. The paper reports this discrepancy rather than reconciling
  it, and keeps L5/r603 for continuity with prior work. Separately, DNABERT-2's *causal*
  evaluation uses an entirely different preprocessing pipeline (256 seed-42 hg38 windows) from
  its *discovery* preprocessing (the fixed ACTB probe) — this is intentional, not an
  inconsistency, but it means the discovery number and the causal number should never be
  described as coming from "the same measurement."
- **NTv3.** The ratio-detector and activation-magnitude rules select L11/r1472 and L6/r1472
  respectively — same row index, different layer, `q1` differing by less than 0.01. NTv3's
  original model-weight revision was also never recorded exactly and is not recoverable; every
  NTv3 result in the paper carries this caveat.
- **Llama/Mistral/OLMo.** Serve primarily as calibration (recovering a published ground truth),
  not as detector output in the same sense as the other 19 models.
- **GENERator-EUK-3B.** Grandfathered activation-magnitude selection; its structural candidate
  (L4/r2371) is the same row used in the GENERator mechanistic case study (§9).
- Resolved Hugging Face checkpoint revisions are available for 21 of the 22 causal-cohort
  models; **NTv3 is the sole exception.**

---

## 5. Structural analysis

**The row-associated bilinear weight operator.** For output row `k` of a gated feed-forward
block, let `g_i` and `u_i` be the gate- and up-projection row vectors for hidden unit `i`, and
let `d_{k,i}` be the corresponding down-projection coefficient. The paper defines

```
U_k = Σ_i d_{k,i} · g_i · u_iᵀ
```

This is a pure **weight-space** object: it is a sum of outer products of *weight* vectors,
scaled by a *weight* coefficient. It never evaluates the gate nonlinearity `φ(g_iᵀ x)` for any
input `x`, and it does not depend on activations at all. **`U_k` is explicitly not the full
nonlinear SwiGLU/GEGLU computation** — the manuscript is careful about this and consistently
calls it a "bilinear weight operator" (never "the gated-FFN operator" as if it captured the
gate). The only place the word "bilinear" is dropped is a Methods section *header*
("Exact gated-FFN operator and structural metrics"); every sentence beneath it uses the correct
term. If you see "row-associated bilinear weight operator" used as the preferred term, that is
exactly the manuscript's own usage — it is the precise name for this object, and it should be
used instead of "the exact gated-FFN operator" whenever precision matters.

**Exact Frobenius norm, without materializing cross terms.** The squared Frobenius norm can be
computed exactly via a Gram-matrix identity, without ever forming the full `d_model × d_model`
matrix `U_k`:

```
‖U_k‖_F² = d_kᵀ K d_k,     K_ij = (g_iᵀ g_j)(u_iᵀ u_j)
```

i.e., `K` is the Hadamard (elementwise) product of the gate- and up-projection Gram matrices.
This is what makes it feasible to compute exact norms for *every* row in a layer at once (used
for the top-norm control selection in §6), without ever materializing an individual operator.

**Singular spectrum and concentration metrics.** Singular values `{σ_j}` of `U_k` are computed
in float64. From them:

- **`q1` (leading singular-energy share):** `q1 = σ1² / Σ_j σ_j²`. This is the paper's primary
  concentration metric — the fraction of the operator's squared singular-value mass sitting in
  its single leading direction. `q1 → 1` means the operator behaves almost like a rank-1 outer
  product.
- **`PR_spec` (spectral participation ratio):** `PR_spec = (Σ_j σ_j²)² / Σ_j σ_j⁴`. Defined in
  Methods and reported per-row in the supplementary structural table, but **it is not used in
  any Results claim, correlation, or figure anywhere in the current manuscript** — a genuine
  gap between what Methods defines and what the paper actually uses (see §16).
- **`stable_rank = 1/q1`**, reported occasionally but never as a primary quantity — algebraically
  redundant with `q1`.
- **`‖U_k‖_F = sqrt(Σ_j σ_j²)`**, the operator's overall magnitude. Because different models
  parameterize their weights at very different absolute scales, **raw Frobenius norms are not
  interpreted as comparable across models** — only within-model, layer-relative comparisons are
  used for causal correlations.

**The retired diagonal proxy — used only once, for calibration.** An earlier, much cheaper
approximation retains only per-hidden-unit ("diagonal") terms and drops all cross-terms between
distinct hidden units:

```
c_{k,i} = W_down[k,i]² · ‖W_gate[i,:]‖² · ‖W_up[i,:]‖²
```

This is **only** used for the retrospective Llama/Mistral/OLMo calibration in Fig. 1A, never
for any cross-model structural claim. It underlies two distinct diagonal-only metrics shown in
that one panel: a **row rank** (where the published super-weight row ranks among all
`d_model` down-projection rows in its layer, by `sqrt(Σ_i c_{k,i})` — all three models rank
1st, i.e., 100th percentile) and a separate, within-row **scalar top-1 share**
(`max_i(c_{k,i}) / Σ_i(c_{k,i})`, measuring how much of *that one row's* diagonal energy sits
in a single hidden unit: 0.89 / 0.99 / 0.96 for Llama/Mistral/OLMo). These are two different
rankings computed on the same diagonal proxy, both distinct from `q1`, which is computed on the
*exact* operator. The diagonal proxy was retired for cross-model structural comparisons because
it materially overstates concentration in genomic models specifically — an earlier calibration
found the fraction of squared mass carried by dropped cross-terms exceeded a locked 20%
threshold in every genomic model tested and none of the text models tested, with the diagonal
concentration estimate overstating the exact spectral estimate by 2.4–4.3× in genomic models.
This asymmetric bias is the concrete reason the exact operator, not the diagonal proxy, is used
everywhere except Fig. 1A.

**Scientific interpretation of Fig. 1.** Panel A is calibration only (does the detection
apparatus recover known ground truth? yes). Panel B shows that `q1` varies substantially
(0.389–0.9996) and does not cleanly separate by architecture or domain — a regression of `q1`
on log-parameter-count and decoder-status explains almost none of the variance (R²=0.158,
neither term significant). Panel C plots exact `‖U_k‖_F` on a log scale, explicitly kept
separate from `q1` because they measure different things (magnitude vs. concentration) and
because absolute magnitudes are not cross-model comparable. Panel D is the local-control
comparison discussed in depth in §6 — this is where the paper's most important structural
qualification lives.

---

## 6. Top-norm-control issue: exact current interpretation

This is the part of the paper that changed most recently and needs the most care, because a
naive reading ("candidates are structurally exceptional") is *not* what the full analysis
supports once you look at the stricter control.

**Two control sets, defined without using any causal measurement:**

1. **Random same-layer controls** — 5 rows sampled without replacement from the candidate's
   layer (deterministic `SeedSequence(42)` stream, excluding the candidate).
2. **Top-norm same-layer controls** — the 5 rows ranked immediately below the candidate by
   exact Frobenius norm, in the same layer, computed for every row at once via the Gram
   identity (no individual operator ever materialized, no forward pass at all). The candidate
   is rank 1 by exact norm in every model tested. **Norm-matched controls were considered and
   found to be unconstructible**: the candidate exceeds its layer's median norm by 12–32×, and
   no row in any candidate's layer falls within 20% of the candidate's own norm in any model
   examined. This is why "top-norm" (the closest available weight-space competitors), not
   "norm-matched," is the correct description of this control set.

**The canonical facts, from the repository's own audited artifacts (22-model cohort, 110
individual top-norm-control measurements = 22 models × 5 controls):**

- Random-control `q1` gap (candidate minus **mean** of 5 random controls), median across 22
  models = **0.935** (unrounded: 0.9346752...). A separate, earlier, *not-directly-comparable*
  statistic exists for a 12-model subset: **mean** (not median) of the same per-model gap =
  0.884, range 0.544–0.977. These are two genuinely different summaries over two different
  panels (12 models vs. 22, mean+range vs. median) — the 12-model number is not a stale subset
  of the 22-model number, but the two numbers landing near each other (0.935 in both, by
  coincidence) is a real readability hazard the manuscript does not currently disambiguate.
- Top-norm-control `q1` gap (candidate minus the **maximum**, i.e. closest/most-concentrated, of
  the 5 top-norm controls), median across 22 models = **0.014** (unrounded: 0.0135382...).
  **This is a different aggregation rule than the random-control gap** (max-of-5 vs.
  mean-of-5) — the manuscript's own supplementary Table S2 caption already flags this as an
  open, unreconciled choice ("carries two candidate-versus-top-norm gap definitions... pick one
  for the main text"). The max-based (most conservative) definition is what Fig. 1C actually
  plots and is the recommended canonical choice going forward.
- **The "69-fold collapse" is correct**, computed from full-precision values
  (0.9346752/0.0135382 = 69.04). Dividing the rounded text values (0.935/0.014) instead gives
  ≈66.8 — a rounding artifact, not an error in the reported number.
- **In 7 of 22 models, at least one top-norm neighbor is *more* spectrally concentrated than
  the candidate itself:** SmolLM2-360M, EuroBERT-210M, EuroBERT-2.1B, ModernBERT-large,
  DNABERT-2, GENERator-EUK-3B, and GenomeOcean-4B.

**The strongest legitimate conclusion:** structural selection identifies rows that are the
highest-norm row in their layer by a wide margin and are near-rank-1 in concentration — but
*conditional on already being high-norm*, their concentration is not distinctive. Put another
way: **simple weight-magnitude ranking generally does not recover the same rows that
activation-based selection recovers as structurally exceptional**, once the comparison is made
against other large rows rather than arbitrary ones.

**What this experiment does NOT establish, and should never be paraphrased as establishing:**

- It does **not** show that magnitude is irrelevant to concentration or to causal importance —
  it shows only that magnitude *alone*, at the "second-through-sixth-largest" scale, does not
  reproduce the *most*-concentrated row.
- It does **not** mean the candidate has been "norm-matched" against controls — norm-matching
  was explicitly attempted and found impossible (no candidate in this panel has any same-layer
  row within 20% of its own norm).
- It does **not** exclude a **nonlinear magnitude threshold** specific to the single most
  extreme (rank-1) row in a layer — the top-norm experiment only tests rows ranked 2nd through
  6th, and cannot speak to whether being *the* largest row (rather than merely *among* the
  largest) crosses some threshold the next-largest rows do not.

**Causally, the top-norm controls behave almost identically to inert rows**, despite being the
largest weight-space alternatives available: median effect +0.035% (ε=0.5) and +0.110%
(ε=1.0), against candidate effects reaching several hundred percent. A paired Wilcoxon
signed-rank test does distinguish the top-norm-control median from the random-control median
(p=0.0009 at ε=0.5, p=0.0001 at ε=1.0) — but the magnitude of that difference is roughly
one-tenth of one percentage point, i.e., statistically real but practically negligible.
Consequently, candidate-versus-control separation is essentially unchanged whether the stricter
or the looser control set is used.

**Close/exception cases, summarized from source data rather than intuition:** the candidate
exceeds the *median* top-norm control in 20/22 models at ε=0.5 and 19/22 at ε=1.0. Fourteen
models show decisive, stable separation. Four show a small but consistently positive gap where
both candidate and controls are nearly inert (EuroBERT-210M, EuroBERT-610M, EuroBERT-2.1B,
Qwen2.5-1.5B). **Four models show an actual sign flip** — the candidate's gap against the
*median* top-norm control is zero or negative at one or both intervention strengths: MosaicBERT
(ε=0.5 only, gap −0.0019), Qwen2.5-0.5B (ε=1.0 only, gap −0.0030), Qwen2.5-7B (ε=1.0 only, gap
−0.0014), and **NTv3 (both strengths, and the only model negative on every comparison against
both control sets at both epsilons)**. This "four exception models" statistic is specifically a
*median-of-five* criterion (`G_ε = R_candidate − median(5 top-norm control R's) ≤ 0`); a laxer
criterion counting any single one of the five top-norm controls exceeding the candidate would
in general flag a different (larger) set of models, since it only takes one control out of five
to be close by chance, whereas failing the median requires roughly half the control set to be
competitive. The manuscript's headline "four exception models" is the median-based statistic,
not the any-of-five statistic; a reader who wants the any-of-five count can reconstruct it from
Supplementary Table S3, which stores all five individual top-norm-control effects per model per
epsilon. Notably, the candidate's own margin of concentration over its top-norm neighbors does
**not** predict which models are exceptions (Spearman ρ=0.311, p=0.159 at ε=0.5; ρ=0.355,
p=0.105 at ε=1.0) — several of the *most* decisively separated models (DNABERT-2,
GENERator-EUK-3B, ModernBERT-large) have small or even negative structural margins, meaning
weight-space descriptors and causal response rank these particular rows in *opposite* orders.

---

## 7. Causal census

**Design.** For each of the 22 cohort models, one frozen candidate row and 5 same-layer control
rows (evaluated separately, never jointly) were perturbed at two strengths:
`w_ε = (1 − ε) w`, so **ε=0.5** halves the row (partial suppression) and **ε=1.0** zeroes it
(full ablation). Effect is the signed relative change in the model's own native loss,
`R_ε = (L_ε − L0)/L0`, reported as `100·R_ε` percent — positive means the native objective got
worse (loss increased) after perturbation.

**Model-level comparator.** For each model, `G_ε = R_ε,candidate − median(R_ε,control,1..5)`.
The manuscript's "18/22," "20/22," and "19/22" headline counts are all counts of models with
`G_ε > 0` — a *median-of-five* criterion, computed separately against the random-control set and
the top-norm-control set (see §6 for why this differs from an "any of five" criterion).

**Random-control results.** Median `G_0.5` = **+0.60%** (95% model-bootstrap CI +0.23% to
+3.06%), candidate exceeds median control in 18/22 models. Median `G_1.0` = **+0.88%** (95% CI
+0.61% to +111.78%), candidate exceeds median control in 20/22 models. The unusually wide upper
bound on the full-ablation CI reflects the heterogeneous, heavy-tailed distribution of candidate
effects across this panel (from −0.79% in NTv3 to +698.87% in SmolLM2-1.7B): a percentile
bootstrap over 22 model-level gaps will, in some resamples, draw disproportionately from the
handful of models with very large gaps (OLMo +111.78%, ModernBERT-base +357.04%, SmolLM2-1.7B
+698.87%), pulling the upper tail far out even though the *median* estimate itself is stable
and modest.

**Top-norm-control results.** See §6 for the full discussion — candidate exceeds the median
top-norm control in 20/22 models at ε=0.5 and 19/22 at ε=1.0, with the same practical separation
as against random controls because the top-norm controls are themselves nearly inert.

**Structural correlates of effect size — both are weak or non-robust.**

- `q1` vs. signed full-ablation effect: Spearman ρ=0.074 (two-sided p=0.744; 95% model-bootstrap
  CI −0.391 to 0.513; n=22). Essentially no relationship — highly concentrated candidates can be
  weakly or catastrophically causal, and less concentrated candidates are not uniformly inert.
- Layer-relative Frobenius magnitude vs. the same effect: ρ=0.441 (p=0.040, CI 0.062 to 0.717)
  at ε=1.0 — nominally significant — but the same association at ε=0.5 is not (ρ=0.348,
  p=0.112), no correction was applied across the two predictors × two intervention strengths
  tested, and restricting to the text-decoder subgroup (where the evaluation endpoint is at
  least uniform) makes both associations vanish and flip sign (Frobenius ρ=−0.042; q1
  ρ=−0.333). **The manuscript explicitly does not treat this nominal Frobenius association as a
  transferable law** — no leave-one-model-out, leave-one-family-out, or covariate-adjusted
  robustness check for it exists anywhere in the repository, and it is deliberately kept out of
  the main correlation figure for that reason. Note also a real Methods-text defect here: the
  manuscript's Methods sentence describes this predictor as "the candidate norm divided by the
  median norm of the five same-layer control rows," but the actual code
  (`backfill_layer_median.py`) divides by the median Frobenius norm across **every** row in the
  layer, not just the 5 sampled controls — a materially different (larger) reference
  population. The reported correlation values themselves are unaffected (they were always
  computed from the actual, full-layer-median column); only the prose description is wrong and
  needs a Methods correction.
- **Nonlinear-threshold caveat applies to both:** neither correlation result excludes the
  possibility of a nonlinear effect specific to extreme values of either descriptor — the panel
  is small (n=22) and heterogeneous, and Spearman correlation is not designed to detect a sharp
  threshold effect.

**Descriptive subgroup medians** (full ablation): text decoders +85.47%, text encoders +0.65%,
genomic decoders +0.68%, genomic encoders +0.02%. These groups differ simultaneously in
objective, tokenizer, architecture, scale, and panel composition, so **the manuscript does not
interpret this as evidence that architecture or domain determines causal importance** — it is
reported strictly as description.

**Figure 2, panel by panel.** (A) Signed effect at ε=0.5 for candidate (diamond) and 5
individual random controls (gray points) per model, with a short bar marking the within-model
control median; models ordered by their full-ablation candidate effect; symmetric-log axis to
show negative, sub-percent, and catastrophic responses together. (B) The same layout at ε=1.0.
(C) Candidate `q1` (x-axis) against signed full-ablation effect (y-axis) — the null correlation
scatter. (D) Candidate-minus-top-norm-control gap at both intervention strengths on a
symmetric-log axis, with the four sign-flip models labeled.

---

## 8. DNABERT-2 mechanistic case study

**Setup.** A fixed 10-row high-gain basis (L5/r603, L3/r86, L3/r399, L9/r264, L9/r294, L3/r603,
L3/r641, L7/r603, L6/r603, L5/r86) is evaluated on DNABERT-2's own **pretrained masked-language-
model objective** — 256 hg38 windows of 600 bp, a single fixed 15% masking realization (seed
42, 4,453 masked tokens reused across every condition), eager attention, the revision-pinned
checkpoint.

**The direct pair-ablation experiment.** L9/r264 and L9/r294 were ablated individually and
jointly. Individually, both effects are small (`d_A=+0.0218`, `d_B=+0.0064` at ε=1.0). Jointly,
loss increases by `d_AB=+2.0400` — far larger than the sum of the parts. Epistasis
(`d_AB − d_A − d_B`) = **+2.0118** at ε=1.0 and **+0.0528** at ε=0.5, both positive (superadditive)
at both scales. This reproduces a previously reported epistasis figure to four decimal places
from an independent, from-scratch environment. The manuscript's own text calls this a
"standalone regression check" (meaning: a reproduction/regression-*test* against a prior
result, in the software-testing sense) — but this label sits uncomfortably close to the paper's
*statistical* regression language for F2/F3 (below) and is worth reading carefully: **this pair
condition is a directly measured joint intervention, not a value predicted by any fitted
model**, and it is explicitly **absent** from the 118 measured non-singleton mask conditions
used to fit the observer families described next.

**The observer-family ladder (F0–F3), applied to combinations of the same 10 rows.** This
tests whether the pair result is an isolated anomaly or part of a broader interactional
pattern, using the mechanistic-tomography framework (Erramilli, ref. 10 in the manuscript):
`F0` sums measured singleton effects additively; `F1` applies one fitted calibration scalar to
that sum; `F2` jointly fits additive main effects by ridge regression; `F3` adds all 45 pairwise
interaction terms and refits by ridge regression. Models were fit and calibrated on disjoint
mask pools (78 fit, 20 calibration masks) and evaluated on 20 **genuinely held-out** masks at
each intervention strength — this fit/calibration/held-out separation is the whole point of the
design; F3's held-out performance is not a training-fit artifact.

- Held-out R² at ε=0.5: F0 −0.023, F1 0.324, F2 0.517, **F3 0.888**.
- Held-out R² at ε=1.0: F0 −0.040, F1 0.659, F2 0.581, **F3 0.790**.
- F2→F3 relative reduction in held-out MAE: **54.7%** (ε=0.5, 95% bootstrap CI [0.0038, 0.0186],
  excluding zero) and **24.4%** (ε=1.0, 95% CI [0.0614, 0.1144], excluding zero). Confidence
  intervals use 5,000 paired bootstrap resamples over batch indices, resampling the held-out
  condition and its baseline jointly before subtracting.
- The known L9/r264×L9/r294 pair re-emerges **without any special treatment during fitting**:
  among all 45 fitted pairwise coefficients, it ranks **#1 by |Γ|** at both scales (Γ=0.077 at
  ε=0.5, Γ=0.739 at ε=1.0; runner-up 0.054 / 0.502).
- **Resplit robustness:** 100 role-permutation resplits of the same fixed 118-condition pool
  (re-assigning which measured conditions land in fit/calibration/held-out, not sampling new
  combinations) show F3 beating F2 on held-out MAE in **100 of 100 resplits at both epsilons**.
  Median [2.5th, 97.5th percentile] held-out R² across resplits: at ε=0.5, F0 −0.21
  [−0.45, 0.06], F1 0.30 [−0.35, 0.57], F2 0.55 [0.26, 0.70], F3 0.92 [0.79, 0.96]; at ε=1.0,
  F0 −0.45 [−1.14, −0.05], F1 0.31 [−0.33, 0.69], F2 0.56 [0.33, 0.70], F3 0.76 [0.49, 0.88].
  The selected ridge penalty λ is unstable across resplits (F2: 0.001–100 at ε=0.5; F3:
  0.001–10 at ε=0.5, 0.1–100 at ε=1.0), meaning the exact point-estimate coefficients depend
  somewhat on which conditions happen to land in the fit set — but the qualitative F3>F2 ranking
  itself does not change under any resplit tested.

**What this establishes:** a real, triangulated (direct measurement + held-out predictive
tomography + coefficient ranking), pretraining-objective-level pairwise interaction — one of the
strongest positive mechanistic results in the whole panel, precisely because three independent
lines of evidence agree. **What it does not establish:** anything about downstream, fine-tuned
task performance (an earlier, differently sourced claim about a downstream splice-accuracy
magnitude for this same pair was contested by a fresh, conflicting, single-seed reproduction and
is **not** part of the current manuscript — do not resurrect it); that DNABERT-2's causal
dimension is intrinsically 2 (this is a property of the declared 10-row basis, not a
model-wide statement); or anything about NTv3, which has no comparable current-result artifact.

---

## 9. GENERator mechanistic case study

**The candidate.** GENERator-EUK-3B's row L4/r2371 has exact `q1=0.969`, among the most
spectrally concentrated operators in the genomic panel.

**BOS attention/activation phenotype (association only).** A reproduced measurement (40 hg38
windows, seed 42) found position 0 receives 37.9% of mean incoming attention (33.0× the uniform
expectation), 78.8% of all (layer, head) observations have their attention maximum at position
0, and the row's own maximum activation is also at position 0. **This establishes co-occurrence,
not causation**: no experiment independently manipulated the high-gain row and the attention
sink separately, and — critically — in a causal decoder, position 0's hidden state can, by
construction, only ever attend to itself, which architecturally guarantees part of this
co-occurrence regardless of any particular row's behavior. Do not describe this as the row
"causing" the sink.

**Damage-matching design.** Baseline generated GC = 0.4204; full ablation of row 2371 drops it
to 0.3065 (a 0.1139 decrease), alongside a native-loss increase from 6.3854 to 8.7543. **Five
non-2371 control rows**, swept across α∈[0,8], moved native loss by only 0.001–0.002 and left GC
essentially unchanged (~0.420–0.421) — these controls are not merely under-damaged, they are
causally insensitive at this location, so they cannot themselves demonstrate specificity.

**The random-direction control — exactly one direction, not several.** A single, fixed random
unit vector (`np.random.default_rng(20260823 + 2371)`, drawn once, unit-normalized) *replaces*
(not adds to) the weight vector at the row-2371 location, scaled by `c ∈ {0.0125, 0.5, 1.0, 3.0,
8.0}`. Because replacement rather than additive perturbation is used, `c=0` is definitionally
identical to ablation, and damage *decreases* as `c` increases (the opposite convention from the
α-sweep). At `c ≤ 1.0` (98–100% of ablation's native-loss damage retained), GC stays at
0.3068–0.3078 — essentially the same low-GC phenotype as true ablation, despite the direction
being random rather than the model's learned one. At `c=8.0` (84% of ablation damage retained),
GC recovers to 0.3402, closing 29.6% of the gap back to the intact baseline; loss itself recovers
more slowly (16%) than composition does over the same range.

**The correct, narrow interpretation:** composition tracks the *magnitude of damage* done at
this location, not the *specific learned direction* removed — a random direction at matched
damage reproduces the phenotype, which argues against a naive "must preserve the learned
direction" story. **This experiment does not establish full direction-independence**: even at
`c=8.0` — eight times the row's own original norm — the random replacement still retains 84% of
ablation's damage, meaning the tested random directions never actually escape being
functionally close to removal. This is a real limitation of the control, not a subtlety to
gloss over: a control that cannot be made simultaneously large *and* low-damage cannot cleanly
separate "direction matters" from "this location is just generally sensitive to disruption."
Note also the manuscript's own **Discussion text has a minor plural/singular slip** — it once
refers to "the tested random directions" (plural) when in fact only one direction, swept across
five scales, was tested; the Methods text correctly uses the singular ("a fixed random unit
direction"). Treat "one direction, five scales" as the fact and the Discussion's plural as a
wording error, not a design change.

**Secondary checks.** The GC decrease is not explained by a single degenerate homopolymer event
(the longest homopolymer accounts for only 3.9% of the GC decrease), and enriched short motifs
after ablation are broadly AT-rich rather than one specific repeat — consistent with a broad
compositional shift rather than a narrow artifact. (A historical bug in the damage-matching
aggregation pipeline — a reachability/self-match boundary error that let `c=0` trivially "match"
its own ablation-definitional target — was found and fixed before the current manuscript numbers
were produced; the current values were independently re-derived from raw per-condition records
in a later audit pass and confirmed correct.)

**Figure 4, panel by panel.** (A) `q1` callout for row 2371 (cross-references Fig. 1B). (B) BOS
attention/activation phenotype bars, explicitly captioned as association-only. (C) Left: GC vs.
random-direction scale `c`, with ablation and intact-baseline reference lines. Right: GC vs.
native-loss damage on shared axes for the row-2371 α-sweep, the five inert controls, and the
random-direction grid together — the point of this panel is that composition tracks damage
monotonically regardless of which manipulation produced that damage.

---

## 10. Evaluation endpoints and datasets

| Group | n | Dataset / windows | Masking / tokenization | Objective | Bootstrap unit |
|---|---:|---|---|---|---|
| Text decoders | 10 | WikiText-2-raw-v1 test; lines shuffled `random.Random(42)`; 100 windows, exactly 512 tokens each, per model | Teacher-forced, no masking | Shifted-label mean-token NLL | Stored batches (25 units if >3B params/batch 4; else 13 units/batch 8) |
| Text encoders | 6 | Same WikiText construction; 256 windows × 512 tokens | Fixed 15% MLM mask, seed 42, reused across all conditions; CLS/SEP/PAD excluded | Summed cross-entropy over masked tokens ÷ masked-token count | 16 fixed batches |
| Genomic decoders | 4 | hg38, `random_262kb.bed`, seed-42 disjoint partition; 100 windows × 512 bp, <1% N | GENERator: trim leftmost `len mod 6` bp, prepend BOS, special tokens disabled; GenomeOcean: no trim/forced BOS, special tokens disabled | Teacher-forced shifted causal-LM mean-token NLL | Individual windows (100 units) |
| Genomic encoders | 2 | hg38, 256 windows × 600 bp, <1% N, padded/truncated to 256 tokens | Fixed 15% MLM mask, seed 42; special/PAD excluded | Masked-nucleotide summed loss ÷ masked-token count | 16 fixed batches |

Structural-discovery probes are **separate** from causal-evaluation datasets: text discovery
uses the first 20 non-empty WikiText-2 lines (512-token truncation, batch size 1); genomic
discovery uses one fixed 504-bp ACTB-derived sequence per model. This is why, e.g., DNABERT-2's
discovery preprocessing and causal-evaluation preprocessing can legitimately disagree about
which coordinate looks most extreme (§4) — they are different assays by design, not an error.

**Case-study datasets that differ from the cohort census:** DNABERT-2's mechanistic tomography
(§8) uses the same 256-window/600-bp/seed-42-mask construction as the genomic-encoder cohort
endpoint. GENERator's mechanistic case study (§9) uses two additional, disjoint hg38 pools not
used elsewhere: 96 windows × 170 bp for generation (first 120 bp used as the prompt), and 100
windows × 512 bp for teacher-forced native-loss measurement — plus a separate 40-window pool
(seed 42) for the BOS attention/activation reproduction.

---

## 11. Statistical analysis

- **Exact operator spectra are deterministic**, single-checkpoint calculations (`q1`,
  `PR_spec`, `‖U_k‖_F` are computed once, exactly, via SVD) — they carry **no sampling error
  bars** anywhere in the paper; this is intentional, not an omission.
- **Paired bootstrap** (5,000 resamples over batch indices, resampling the held-out condition
  and its baseline jointly before subtracting) backs the DNABERT-2 F2→F3 confidence intervals.
- **Model bootstrap** (5,000 resamples of the 22 model-level values, with replacement) backs the
  cohort candidate-minus-control gap CIs (seeds 47 for ε=0.5, 52 for ε=1.0) and the two
  structural-vs-causal correlation CIs (seed 43 for `q1`, seed 44 for layer-relative
  Frobenius).
- **Spearman correlations** are computed on the full 22-model panel and, separately, within
  architecture subgroups; any subgroup with fewer than 8 models is explicitly flagged as
  underpowered and is not used to support any claim.
- **Nothing here is treated as population-level inference.** The model panel is small,
  heterogeneous, and not a random sample from a defined population of foundation models —
  coefficient tests and correlations are reported and interpreted descriptively.
- **No multiple-comparison correction is applied** anywhere — not across models, not across the
  two structural predictors (`q1`, layer-relative Frobenius), and not across the two
  intervention strengths. This matters directly for interpreting the one nominally significant
  correlation in the paper (layer-relative Frobenius vs. full-ablation effect, p=0.040): with
  two predictors × two intervention strengths tested and no correction, a single p≈0.04 result
  that also fails to replicate at the other intervention strength and reverses sign within a
  subgroup should be read as weak, not as a confirmed effect.
- **The heavy-tailed +0.88% [+0.61%, +111.78%] interval** arises because the underlying
  distribution of per-model gaps is strongly right-skewed (a handful of models show gaps in the
  hundreds of percent); a percentile bootstrap over the median of 22 such values will, in some
  resamples, happen to draw more of the large-gap models, stretching the upper tail far beyond
  the point estimate even though the estimate itself (the sample median) is stable.

---

## 12. Figure-by-figure guide

**Figure 1 — Structural geometry across the 23-model panel.**
- *(A)* Retrospective diagonal-proxy calibration: recovers the published Llama/Mistral/OLMo
  super-weight rows at rank 1/d_model. **Intended claim:** the detection apparatus works.
  **Do not infer:** that this diagonal metric is the same quantity as `q1` elsewhere in the
  figure (it is numerically distinct by construction).
- *(B)* Exact `q1` across all 23 models, sorted, marker shape = architecture, color = domain.
  **Intended claim:** near-rank-1 concentration recurs in both encoders and decoders; no clean
  domain/architecture split. **Subtle point:** Evo2-7B is annotated separately as a detection
  null (ratio 2.22 < 5.0), not plotted with a `q1` value.
- *(C)* Two panels sharing a y-axis: candidate `q1` vs. random controls (left) and vs. top-norm
  controls (right), across all 22 causal-cohort models. **Intended claim:** the dramatic
  candidate-vs-random separation (median gap 0.935) collapses to near-nothing (median gap 0.014)
  against the stricter control. **Subtle point:** the two panels use different within-model
  aggregation rules (mean-of-5 for random, max-of-5/closest-neighbor for top-norm) — this is
  disclosed in the manuscript's own supplementary caption but easy to miss on a first read.
- *(D)* Detected row vs. 5 ordinary same-layer rows in MosaicBERT and ModernBERT specifically.
  **Do not infer:** generalization beyond these two models — no same-layer controls exist for
  DNABERT-2 or NTv3 in this panel.

**Figure 2 — Causal census across the 22-model cohort.**
- *(A, B)* Signed relative native-loss change at ε=0.5 and ε=1.0 respectively; candidates as
  diamonds, 5 individual controls as gray points, within-model control median as a short bar.
  **Do not infer:** a universal positive effect — several models are near zero or negative.
- *(C)* `q1` vs. signed full-ablation effect. **Intended claim:** essentially no correlation
  (ρ=0.074). **Subtle point:** this null result is itself a headline finding, not a negative
  control to skim past.
- *(D)* Candidate-minus-top-norm-control gap at both strengths, four sign-flip models labeled.
  **Do not infer:** that the labeled models share a common mechanism — the manuscript explicitly
  could not find a structural predictor of which models these are.

**Figure 3 — DNABERT-2.**
- *(A, B)* Singleton vs. joint ablation of the L9/r264×L9/r294 pair; epistasis at both scales.
  **Subtle point:** these values come from a standalone, directly measured joint intervention,
  not from the F0–F3 fit pools, and this exact pair condition never appears among the 118
  measured non-singleton conditions used to fit those models.
- *(C)* Held-out R² for F0–F3 across 20 truly held-out mask conditions. **Intended claim:**
  additive/calibrated models fail held-out adequacy; the pairwise-lifted model materially
  improves it.
- *(D)* Fitted pairwise coefficients, ranked; the known pair ranks #1/45 at both scales.
  **Do not infer:** that this ranking reflects a basis-selection procedure — the pair was known
  before F3 was ever fit.

**Figure 4 — GENERator.**
- *(A)* `q1` callout for row 2371 (cross-references Fig. 1B; not a restatement of the full
  comparison).
- *(B)* BOS attention/activation phenotype. **Do not infer:** causation — explicitly
  co-occurrence only, and partly architecturally guaranteed in a causal decoder.
- *(C)* GC vs. `c` (random-direction scale) and GC vs. native-loss damage on shared axes.
  **Intended claim:** composition tracks damage magnitude, not direction identity, within the
  range tested. **Do not infer:** full direction-independence (the largest tested scale still
  retains 84% of ablation's damage) or a validated linear/graded steering axis (only 3–5 dose
  points collected by design).

**Supplementary Figure S1 — Structural and causal control detail.** (A) All six Phi-3 row-level
`q1` values (0.558–0.953) behind the single median value plotted in Fig. 1B. (B) Individual
same-layer random-control causal effects for all decoder models in the causal audit. (C) Local
spectral exceptionalness (candidate `q1` minus mean control `q1`) vs. non-embedding parameter
count, for the 12-model common local-control panel specifically — not the full 22-model cohort.

---

## 13. Terminology glossary

- **Super weight.** The original NLP concept (Yu et al.): an individual parameter whose ablation
  catastrophically degrades a language model's generation quality. This paper's "candidates" are
  the genomic/cross-domain analogue under test, not an assumed equivalent.
- **High-gain row.** This paper's general term for an output row of a gated FFN block with
  unusually large activation/weight-space signature — the object under study across all models,
  regardless of whether it turns out to be causally critical.
- **Candidate.** The one frozen, pre-registered high-gain row per model used in the structural
  panel and/or causal census. Selected *before* any causal measurement.
- **Activation-derived / activation-based.** Umbrella term covering *both* candidate-selection
  protocols used in this paper (the current ratio detector and the earlier grandfathered
  activation-magnitude rule) — both require a forward pass over real input, as opposed to a
  weight-only rule. Correctly used as an umbrella in the manuscript; "activation-selected" (used
  once, in the Abstract) is slightly less precise since 6/22 candidates were retained rather than
  freshly selected by any activation rule.
- **Activation-ratio detector.** The specific, current selection rule:
  `channel_max[layer,row] / median_row(channel_max[layer,:]) ≥ 5.0`, computed from the maximum
  absolute activation over token positions in one forward pass over a fixed probe.
- **Row-associated bilinear operator (`U_k`).** The exact, weight-only object
  `Σ_i d_{k,i} g_i u_iᵀ` associated with output row `k` — captures the bilinear (gate × up)
  structure but not the gate nonlinearity.
- **Frobenius norm (`‖U_k‖_F`).** The operator's overall magnitude; computed exactly via a Gram
  identity, not comparable in absolute terms across models with different weight
  parameterizations.
- **`q1`.** Leading singular-energy share of `U_k`'s exact spectrum, `σ1²/Σσ_j²` — the paper's
  primary concentration metric.
- **Near-rank-one.** Informal description of a `q1` value close to 1 — the operator behaves
  almost like a single outer product.
- **Random control.** A same-layer row sampled independently of any magnitude or activation
  criterion (fixed `SeedSequence(42)` stream), used as the "arbitrary row" baseline.
- **Top-norm control.** One of the five same-layer rows ranked immediately below the candidate
  by exact Frobenius norm, selected with no forward pass — the stricter, magnitude-only
  baseline.
- **Random `q1` gap.** Candidate `q1` minus the **mean** of five random-control `q1` values.
- **Top-norm `q1` margin.** Candidate `q1` minus the **maximum** (closest competitor) of five
  top-norm-control `q1` values — a different aggregation rule than the random gap, by design
  (see §6).
- **Absolute structural margin.** The unsigned version of the top-norm `q1` margin, used when
  correlating structural exceptionalness against the (signed) causal gap.
- **Functional criticality.** Whether a *single* row's perturbation measurably changes a
  model's native loss — what the 22-model census measures.
- **Causal response complexity.** Whether a model's finite-intervention response to a *set* of
  rows is captured by an additive model or requires higher-order (pairwise or beyond) terms —
  what the F0–F3 tomography ladder measures. Distinct from, and not implied by, functional
  criticality.
- **Epistasis.** In the DNABERT-2 case study, `d_AB − d_A − d_B`: the joint ablation effect minus
  the sum of the two singleton effects. Positive epistasis here means superadditive damage.
- **Observer families F0–F3.** The mechanistic-tomography response-model ladder: F0 additive
  singleton sum, F1 single calibration scalar, F2 ridge-fit additive main effects, F3 F2 plus all
  pairwise interaction terms.
- **Damage tracking.** The GENERator interpretation that a downstream effect (GC composition
  shift) scales with *how much native-loss damage* a perturbation causes, regardless of which
  specific weight direction produced that damage.
- **Native objective.** Each model's own pretraining loss (causal-LM NLL for decoders,
  masked-LM cross-entropy for encoders) — the endpoint used for every causal measurement in this
  paper, as opposed to any downstream fine-tuned task.

---

## 14. FAQ / likely coauthor questions

**Why 23 models in Fig. 1 but 22 in Fig. 2?** Fig. 1 (structural) includes Phi-3-mini; Fig. 2
(causal) excludes it, because Phi-3's pre-existing basis is six rows rather than one comparable
primary row, so a like-for-like singleton intervention isn't possible for it. See §4.

**Why is Phi-3 treated differently?** Its six candidate rows were chosen by a top-6-by-exact-norm
rule (the same style used for MosaicBERT/ModernBERT-base's earlier bases), not by the ratio
detector or the grandfathered single-row rule used for everything else. Its Fig. 1B point is the
*median* `q1` across those six rows, not one primary coordinate.

**Why didn't Evo2 get a `q1`?** Its maximum activation ratio (2.22) fell below the acceptance
threshold (5.0), so no candidate was accepted under the detector — it is a structural null, not
a missing measurement.

**Are all candidates selected by the same detector?** No — 16 by the current ratio detector, 6
retained from an earlier activation-magnitude-only protocol (Llama, Mistral, OLMo, DNABERT-2,
NTv3, GENERator-EUK-3B), and Phi-3 by a separate top-6-by-norm multi-row rule. See §4.

**Is `U_k` the actual SwiGLU/GEGLU computation?** No. It omits the gate nonlinearity entirely —
it is a pure weight-space bilinear form. "Row-associated bilinear weight operator" is the
correct, manuscript-consistent term; avoid calling it "the gated-FFN operator" without
qualification.

**Why isn't `q1` enough to identify causally important rows?** Because `q1` and signed
full-ablation causal effect are essentially uncorrelated across the 22-model panel
(ρ=0.074, CI spanning −0.39 to +0.51). A row can be near-rank-1 and either weakly or
catastrophically causal.

**What does "rank one" mean?** Loosely, that nearly all of an operator's squared singular-value
mass sits in a single leading direction (`q1 → 1`) — i.e., `U_k` behaves almost like one outer
product, `σ1 · v1 w1ᵀ`, rather than a genuinely high-rank object.

**Why use random controls?** To establish a baseline: is the candidate exceptional relative to
an arbitrary row in the same layer? (Answer: usually yes, and dramatically so.)

**Why use top-norm controls?** Because "exceptional relative to an arbitrary row" is a weak
claim if the candidate is simply the biggest row in its layer — top-norm controls test whether
the candidate is exceptional relative to other *large* rows specifically. (Answer: mostly no —
the apparent structural gap collapses 69-fold.)

**Are the top-norm controls norm-matched?** No — norm-matching was attempted and found
impossible; no row in any candidate's layer comes within 20% of the candidate's own norm. They
are the closest available alternatives, not matched controls.

**Does the top-norm experiment prove magnitude is irrelevant?** No. It shows that magnitude
*alone*, among rows ranked 2nd–6th by norm, does not predict causal importance — it says nothing
about whether crossing into the single most extreme (rank-1) magnitude regime matters
nonlinearly.

**Why can a control have greater Frobenius norm than the candidate?** In one model
(GENERator-PROK-1.2B, per earlier structural work) and in the seven models where a top-norm
control is more spectrally *concentrated* than the candidate (§6), the candidate is still the
single highest-*norm* row by construction of the control-selection rule — but concentration
(`q1`) and magnitude (`‖U_k‖_F`) are different quantities, and a control lower in norm can still
exceed the candidate's `q1`.

**Why do some individual controls exceed the candidate but only four models count as
exceptions?** Because the "four exception models" statistic requires the *median* of the five
top-norm controls to meet or exceed the candidate, not merely any single one of the five. A
laxer any-of-five criterion would generally flag more models. See §6.

**Why are negative causal effects retained?** Because a negative effect (loss *improving* after
perturbation) is itself informative — it means the row was, if anything, mildly harmful to the
native objective — and dropping it would bias the reported distribution toward more dramatic
results than actually occurred.

**Why is the full-ablation CI so asymmetric ([+0.61%, +111.78%])?** The underlying distribution
of per-model gaps is strongly right-skewed (a few models show gaps in the hundreds of percent);
bootstrap resamples occasionally draw disproportionately from those models, stretching the upper
percentile far beyond the stable median point estimate.

**What exactly does the Frobenius correlation mean?** A nominal association (ρ=0.441, p=0.040)
between layer-relative operator magnitude and full-ablation causal effect that holds only at one
of two intervention strengths, is uncorrected for multiple comparisons, and reverses sign within
the text-decoder subgroup — reported, but explicitly not treated as a robust or transferable
relationship.

**Why do you still mention a nonlinear threshold?** Because the top-norm experiment tests only
rows ranked 2nd–6th by magnitude; it cannot rule out a threshold effect that only the single
most extreme row in a layer crosses. Absence of a linear relationship among near-top rows is not
evidence against a nonlinear one at the extreme.

**Why does DNABERT singleton ablation miss the important pair?** Because the interaction is
genuinely superadditive: each row's individual ablation effect is small (+0.02, +0.006 at full
ablation), and the joint effect (+2.04) is far larger than their sum — the important quantity
only appears when both are perturbed together.

**Is the DNABERT pair cherry-picked?** No — it was a pre-existing, independently known pair
being *re-derived* (not searched for) by the F0–F3 fitting procedure; it ranked #1 of 45
possible pairs by fitted coefficient at both intervention scales without receiving any special
treatment during fitting.

**Is the direct pair condition included in tomography fitting?** No. The joint-ablation
condition for exactly this pair is confirmed absent from the 118 measured non-singleton mask
conditions used for the F0–F3 fits — the epistasis numbers come from a separate, standalone
direct measurement.

**Why is Phi-3 tomography unresolved?** The same F0–F3 approach was applied to Phi-3's six-row
basis, but no third-order (triple-interaction) model was fit, so its full-ablation response is
reported as unresolved rather than modeled with higher-order terms after the fact — a
deliberate, disclosed negative result, not an orphaned analysis (it still supports Phi-3's
structural `q1` value and Supplementary Fig. S1A).

**Does GENERator row 2371 control GC content?** Disrupting it produces a large GC shift, but the
evidence supports "this location is unusually load-bearing and disruption biases composition,"
not "this row is a dedicated, direction-specific GC controller" — a random direction at matched
damage reproduces the same shift.

**Does the random-direction experiment prove the learned direction is irrelevant?** No — it
shows the phenotype is not uniquely tied to the *learned* direction at matched damage, but even
the least-damaging tested random-direction scale (8× the row's own norm) still retains 84% of
ablation's damage, so the control never actually demonstrates a low-damage, direction-specific
comparison point.

**Why is c=0 equivalent to ablation?** Because the random-direction control *replaces* the
row-2371 weight vector rather than adding to it; at scale `c=0`, "replace with zero" is
identical to zeroing the row outright, which is exactly what full ablation does.

**Are the GENERator non-2371 controls damage-matched?** No — despite being swept across a wide
range (α∈[0,8]), they never reach comparable native-loss damage to row 2371's ablation; they are
causally insensitive at this location, not merely under-tested.

**Does row 2371 cause the BOS attention sink?** Not established — only co-occurrence is shown,
and part of the co-occurrence (position 0 attending only to itself) is architecturally
guaranteed in any causal decoder, independent of any particular row's behavior.

**Why are the DNABERT discovery and causal datasets different?** Candidate *discovery* uses a
fixed short ACTB probe with tokenizer-default special tokens (to reproduce the historically
established activation maximum); causal *evaluation* uses an independent, larger hg38 sampling.
These are intentionally distinct assays, not an inconsistency.

**What exactly is novel relative to Yu et al.?** Yu et al. established super weights in NLP
decoders. This paper extends the question to genomic and encoder architectures with an exact
(not diagonal-approximate) operator, and — more importantly — runs the systematic structural-vs-
causal dissociation test that Yu et al.'s original framing did not need to address at the same
scale.

**What is the strongest defensible one-sentence conclusion?** High-gain gated-FFN rows are a
real, recurring structural phenomenon across text and genomic foundation models, but structural
prominence is only an enrichment signal for functional importance — not a calibrated predictor
of effect size or of causal organization — and even genuinely causal high-gain rows realize
their importance through qualitatively different mechanisms (singleton dominance, pairwise
interaction, damage-tracking fragility) in different models.

**What claims should we avoid in a rebuttal or presentation?** See §15 in full — in short,
anything implying a universal mechanism, a validated closed-form causal predictor, perfect
norm-matching, or that either case study (DNABERT-2, GENERator) generalizes beyond its own
model.

---

## 15. Claims we should NOT make

| Retired / dangerous formulation | Preferred safe formulation |
|---|---|
| "High-gain rows are a universal mechanism" | "High-gain rows are a recurring structural phenotype whose causal realization is architecture- and model-specific" |
| "Near-rank-one geometry determines causal importance" | "Near-rank-one geometry is common among causally important rows but does not predict effect size (ρ≈0.07, CI crossing zero)" |
| "Frobenius magnitude has been ruled out as relevant" | "Simple magnitude ranking among near-top rows does not reproduce activation-derived causal enrichment; a threshold effect at the single most extreme row is not excluded" |
| "Top-norm controls are fully norm-matched" | "Top-norm controls are the closest available weight-space alternatives; true norm-matching was attempted and found unconstructible in this panel" |
| "The candidate is the highest-norm row in every model" | "The candidate is the highest-norm row in its own layer in every model tested — a within-layer, not cross-model, statement" |
| "GENERator row 2371 specifically controls GC content" | "Disrupting this location produces a large, damage-tracking GC shift; the effect is not shown to require the row's specific learned direction, nor is direction-independence fully established" |
| "The random-direction experiment proves direction-independence" | "A random direction at matched damage reproduces the phenotype, but the tested random directions never escape being functionally close to removal, so the control cannot cleanly separate direction-specificity from general location-sensitivity" |
| "The high-gain row causes the attention sink" | "The high-gain row and a BOS-centered attention/activation pattern co-occur; no causal link between them has been established, and part of the co-occurrence is architecturally guaranteed" |
| "DNABERT-2 pairwise structure establishes downstream biological function" | "DNABERT-2's pairwise interaction is established on the pretrained masked-LM objective only; no claim is made about fine-tuned or downstream task consequences" |
| "The DNABERT-2 pair has a downstream splice-accuracy magnitude of X pp" (any specific earlier-reported magnitude) | Not stated at all in the current manuscript — a fresh, single-seed reproduction materially conflicted with the earlier number, and neither figure is used |
| "Encoder/decoder architecture determines structural concentration or causal organization" | "Architecture is descriptively associated with structural and causal patterns in this panel but is not shown to determine them, and named exceptions exist on both sides (NTv3, Phi-3, OLMo)" |
| "Shadow redundancy" / SW-neighborhood pruning tolerance | Retired outright — the project's own pruning sweep contradicts it; do not reintroduce in any form |
| "We scanned eight of the best genomic language models" (implying uniform, representative coverage) | Use the explicit coverage/provenance tables (§4, §10) instead of a round headline number |
| "`(‖U_k‖_F, C)` jointly predict criticality" | Retired — DNABERT-2's own data falsify the joint framing (near-zero coordinate preservation with the strongest measured encoder ablation phenotype) |
| Any claim resting on the NTv3 truncation-bug-affected splice result | That result is retired; NTv3's only current, live causal-adjacent fact in this manuscript is its structural `q1` outlier value and its full-negative status in the causal census |

---

## 16. Remaining manuscript housekeeping

These are copy/formatting/internal-consistency issues only — none of them affects any reported
number, and none should be conflated with a scientific finding. All were identified against the
current manuscript text (`manuscript/actual_manuscript.md`, source `new v9(1).pdf`); verify
against whatever version you are looking at before acting, since some may already be fixed.

- **Empty end-matter.** The Acknowledgments, Author contributions, and Competing interests
  sections are present as headers with no body text — these need to be filled in before
  submission.
- **A genuine Methods error, not just imprecision:** the sentence describing the layer-relative
  Frobenius predictor ("divided by the median norm of the five same-layer control rows") does
  not match the code, which divides by the median norm across *all* rows in the layer. The
  reported correlation numbers are unaffected; only this one sentence needs correcting.
- **Two unreconciled statistic definitions**, both already flagged in the manuscript's own
  Supplementary Table S2 caption: (1) the top-norm `q1` gap has both a max-based and a
  mean-based column, with the caption itself asking the authors to pick one for the main text;
  (2) the "structural margin" used in the ρ=0.311/0.355 correlation (§6, §7) is never explicitly
  defined in Methods as being the same statistic as the Fig. 1C/1D top-norm gap, just unsigned —
  worth one added defining sentence.
- **A coincidental numeric collision.** The 12-model mean-based structural gap example value
  ("GENERator-PROK-3B had q1=0.935...") and the unrelated 22-model median random-control gap
  statistic ("0.935") land on the identical rounded number a few sentences apart — real, not an
  error, but worth one extra decimal place or a different illustrative example to avoid a
  reviewer misreading them as duplicated.
- **Provenance-paragraph scope.** The "16 ratio-detector + 6 grandfathered = 22" paragraph never
  states that its scope is the 22-model causal cohort (not the 23-model structural panel); one
  clarifying sentence would remove the apparent arithmetic mismatch a careful reader would
  otherwise flag (§4).
- **A terminology collision, not an error.** "Standalone regression check" (DNABERT-2 pair,
  §8) is technically defensible but sits uncomfortably close to the paper's *statistical*
  "ridge regression" language for F2/F3 in the same section; renaming to "standalone direct
  pair-ablation experiment" or "reproduction check" would remove the ambiguity. Relatedly, the
  identical sentence about the 118 non-singleton conditions is duplicated verbatim across two
  Methods subsections and could be trimmed to one place.
- **A missing cross-reference.** The Methods sentence disclosing Phi-3's unresolved
  full-ablation tomography result has no pointer to Supplementary Fig. S1A, where its supporting
  structural values actually live.
- **`PR_spec` is defined in main Methods but used nowhere** in Results, Discussion, or any
  figure/correlation table — consider relegating its definition to supplementary methods, or
  adding one sentence noting it is reported descriptively only.
- **A minor plural/singular wording slip** in the Discussion's GENERator paragraph ("the tested
  random directions" should be singular — only one direction, swept across five scales, was
  tested; Methods already gets this right).
- **NTv3's checkpoint revision remains unrecoverable** — already disclosed in Methods and Data
  Availability, but worth keeping visible in any cover letter or response to reviewers, since it
  is the one model in the panel without a resolvable Hugging Face revision.
- No TMLR-specific or other venue anonymization requirements were found recorded anywhere in
  this repository as of this writing — if a specific target venue's formatting rules apply,
  they have not yet been captured here and should be checked directly against the venue's
  current author guidelines.
