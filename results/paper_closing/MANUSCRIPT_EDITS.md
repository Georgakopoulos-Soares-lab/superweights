# Manuscript edits — paper-closing round (Sep 2026)

Publication-ready prose for *"Structure Is Not Mechanism: High-Gain Gated-FFN Rows Across
Text and Genomic Foundation Models."*

**Format-agnostic by design.** Each edit is keyed to a **section name** and an exact **quoted
anchor** rather than a line number, so it applies equally to
`paper-salvage/actual_manuscript.md` and to the pandoc LaTeX conversion of it. Nothing here
has been written into the manuscript file itself — see the note at the end.

Every number below is quoted from a committed artifact, named per edit. No number is
transcribed from prose.

---

## E1 — Abstract: one added sentence

**Anchor** (current final sentence of the Abstract):

> …could be reproduced by a random-direction replacement at comparably severe damage, arguing
> against a simple direction-specific interpretation.

**Append:**

> Finally, sweeping 36 rows spanning the full activation-ratio range within a single layer, in
> one text and one genomic decoder, showed that the ratio's relationship to causal damage is
> two-regime rather than graded: among rows below the detector's acceptance threshold the
> ratio carried no positive information about damage (Spearman ρ=0.040, p=0.86 in the genomic
> decoder; ρ=−0.584, p=0.005 in the text decoder), while above it the ratio ordered rows
> strongly (ρ=0.639 and ρ=0.975). Within one such layer a row ranked fourth by activation
> ratio was 130-fold more damaging than the row ranked second, and its damage was abolished by
> additionally ablating a third, individually near-inert row — so the ratio identifies where
> high-gain rows are without determining which of them matters most.

*Evidence:* `within_model_slope_comparison.json`, `smollm2_second_row_epistasis.json`.

---

## E2 — Results: new subsection

Place **immediately after** "Top-norm rows do not reproduce the candidate's causal
enrichment" and **before** the DNABERT-2 pairwise-interaction subsection: it refines the
enrichment result before the paper moves to mechanism.

### Activation ratio identifies high-gain rows without grading their severity

Every comparison to this point is either between models — one candidate per model, so model
identity is confounded with the candidate's ratio — or within a model but binary, contrasting
the candidate against five controls that sit near ratio 1 by construction. Neither design can
show whether damage *scales* with the ratio. We therefore measured both quantities for many
rows inside a single layer, holding architecture, scale, endpoint and baseline loss fixed.

In each of two decoders — GENERator-EUK-3B at L4 (the layer carrying its frozen candidate
r2371) and SmolLM2-1.7B at L7 (r227) — we recorded the per-row maximum absolute activation at
the `down_proj` output, formed each row's layer-relative ratio, and selected 36 rows
log-spaced by ratio **rank** so that the full ratio range was represented rather than only its
extremes. Selection used the activation ratio alone and never a causal outcome. Each selected
row was then ablated (α=0) and the model's native loss re-evaluated on its own frozen
evaluation pool.

Pooled across all 36 rows the ratio and the relative loss increase were positively associated
in both models (GENERator ρ=0.454, p=0.005; SmolLM2 ρ=0.625, p=4.6×10⁻⁵), and the association
survived removing the frozen candidate itself (ρ=0.406 and ρ=0.592), so it is not an artifact
of a single extreme point. The pooled coefficient is nevertheless misleading, because the
relationship is not uniform across the ratio range. Partitioning the swept rows at the
detector's own pre-existing acceptance threshold of 5.0 separates two regimes. Among the 21
rows below threshold the ratio carried no positive information about damage in either model:
the association was indistinguishable from zero in GENERator (ρ=0.040, p=0.86) and
significantly *negative* in SmolLM2 (ρ=−0.584, p=0.005), at relative-damage magnitudes of
order 10⁻⁵ — physically negligible, though approximately three orders of magnitude above the
harness's own numerical reproducibility. Among the 15 rows at or above threshold the ratio
ordered rows strongly (ρ=0.639, p=0.010 in GENERator; ρ=0.975, p=7.1×10⁻¹⁰ in SmolLM2), and
again survived removing the candidate (ρ=0.556 and ρ=0.969). The frozen candidate's damage
exceeded the median sub-threshold row's by factors of 6.1×10⁴ and 5.4×10⁵.

We report this partition as an exploratory analysis. The split point is the detector's
acceptance threshold, fixed before these measurements and not fitted to them, but the decision
to analyse the two regimes separately was made after inspecting the swept data.

The consistent reading across both domains is therefore that the activation ratio functions as
a **detector** and, above its own threshold, as an **ordering**, but not as a dose-response
measure of severity. This is the within-model counterpart of the cohort-level dissociation
reported above, and it supports rather than revises the description of the ratio as an
enrichment signal for functional importance rather than a calibrated measure of effect size.

*Evidence:* `within_model_slope.{json,tsv}`, `within_model_slope_smollm2_1.7b.{json,tsv}`,
`within_model_slope_comparison.json`. Figure: `fig_within_model_slope_2panel.pdf`.

---

## E3 — Results: new subsection

Place **directly after** E2.

### A single layer can carry two critical rows whose damage is not independent

Sweeping many rows rather than one also revealed structure that a one-candidate-per-model
census cannot see. In SmolLM2-1.7B layer 7, two rows were individually catastrophic under
ablation: the frozen candidate r227 (activation ratio 3181.7, rank 1 of 2048; relative loss
increase +698.9%) and r161 (ratio 63.5, rank 4; +287.2%). A row the detector ranks *above*
r161 was close to inert: r749 (ratio 358.6, rank 2) cost +2.2%. A row ranked fourth by
activation ratio was thus 130-fold more damaging than the row ranked second, and r161 appears
in no census artifact because the census retains a single candidate per model.

Ablating rows in pairs on the same evaluation pool showed that these effects are not
independent. Because the endpoint is a loss increase, a positive interaction term denotes
super-additivity. Joint ablation of r227 and r161 cost +866.6% against a sum of singles of
+986.1% (interaction −119.5 percentage points, 87.9% of the sum), i.e. mildly sub-additive.
Joint ablation of r227 and r749 was super-additive (+772.9% against +701.1%). The third pair
was qualitatively different: ablating r161 together with r749 cost **+2.07%**, against a sum
of singles of +289.4% — 0.7% of the additive expectation. Removing r749, which is nearly
harmless on its own, therefore **abolishes** r161's catastrophic effect; removing both leaves
the model essentially intact. Because a cancellation of this form is also the signature of a
weight-restoration error, all six conditions were re-measured with an independent
implementation that zeroes rows by explicit indexing, restores from separately held copies,
and re-evaluates the intact loss after every restoration; every value reproduced exactly, the
result was invariant to ablation order, and the recovered intact loss was identical to 10
significant figures.

The three rows are also the geometric extremes of their layer. Referenced against all 19,900
pairs among the layer's 200 highest-norm rows (mean cosine −0.0001, s.d. 0.0224), r161 and
r749 are the most **aligned** pair in the layer (cosine +0.395, z=+17.6, 100th percentile)
and r227 and r749 the most **anti-aligned** (cosine −0.398, z=−17.8, 0th percentile), while
r227 and r161 are unremarkable (cosine −0.032). The two rows whose ablations cancel are
therefore positively aligned rather than opposed, which excludes the simplest
mutual-cancellation account; their operator norms differ threefold (‖r749‖=24.61,
‖r161‖=7.91, ‖r227‖=26.84).

We report the causal and geometric observations as measurements and do not propose a mechanism
linking them. Establishing why an aligned, higher-norm partner is required for r161's damage to
manifest would require activation-level mediation experiments that we have not performed.

*Evidence:* `smollm2_second_row_epistasis.json`, `smollm2_161_749_geometry.json`.

---

## E4 — Methods → "Candidate provenance": status upgrade, text unchanged

The existing paragraph is **correct as written** and its coordinate claims are now measured
rather than asserted. Do not rewrite it. Two edits only.

**E4a.** Anchor:

> For Llama-7B, Mistral-7B, and OLMo-7B the coordinates are those published by Yu et al. and
> serve as calibration rather than as detector output; the ratio detector independently
> recovers the published coordinate in Llama-7B and Mistral-7B. In OLMo-7B it does not: the
> ratio maximum falls at L2/r269 and the activation maximum at L30/r269…

**Append to that sentence's paragraph:**

> These three coordinates were re-derived for this report under the discovery protocol stated
> below. The ratio maximum coincided with the published coordinate in Llama-7B (L2/r3968,
> ratio 3959.2, rank 1 of 131,072 layer–row coordinates) and Mistral-7B (L1/r2070, ratio
> 1714.3, rank 1 of 131,072); in OLMo-7B it fell at L2/r269 with the published coordinate at
> rank 2, and the absolute-activation maximum fell at L30/r269, as stated. Repeating the
> measurement over 24 independent WikiText-2 windows rather than the single discovery input
> returned the same coordinate in every case, with the ratio maximum recovered in 100% of
> windows for Llama-7B and Mistral-7B and 83% for OLMo-7B, and was unchanged under both
> special-token conventions.

**E4b.** Anchor (the OLMo sentence's conclusion):

> …so OLMo's headline coordinate is one instance of a depth-recurring row-269 family rather
> than a coordinate uniquely determined by the detector.

**Append:**

> Llama-7B shows the same pattern: its ratio maximum is L2/r3968 while its absolute-activation
> maximum is L30/r3968, the same row index at a much later depth. Mistral-7B is the exception,
> with both maxima at L1/r2070. Depth-recurring row families are therefore the majority pattern
> among these three models rather than an OLMo-specific irregularity.

*Evidence:* `detector_published_protocol.json`, `uniform_detector_text_{llama,mistral,olmo}.json`,
`audit/detector_provenance_exp2_resolution.csv`.

---

## E5 — Results or Methods: the causal cost of the two selection rules

Add where the DNABERT-2 rule disagreement is discussed, so both directions appear together.

> Where the two rules disagree, the coordinate preferred by the ratio detector is not
> systematically the more consequential one. In OLMo-7B, ablating the published L1/r269
> increased loss by 111.8% while ablating the ratio maximum L2/r269 increased it by 2.4% — a
> 47-fold difference in favour of the coordinate the ratio rule rejects — evaluated on the same
> pool, with five seeded same-layer controls at L2 giving a median effect of −2.5×10⁻⁷.
> DNABERT-2 runs the other way: ablating the ratio maximum L8/r603 raised masked-language-model
> loss (+10.7%) whereas ablating the frozen L5/r603 slightly lowered it (−13.2%). Neither rule
> is therefore uniformly closer to causal importance; they are different statistics that
> disagree about depth along a shared row.

*Evidence:* `uniform_detector_text_olmo.json`, `uniform_detector_dnabert2.json`.

---

## E6 — Discussion: one added paragraph

Place after the paragraph beginning "The structural panel does not support a simple
architecture- or scale-based partition…".

> These results also bound how the activation-ratio criterion should be described. Within a
> single layer the ratio is informative about which rows are candidates and, above its
> acceptance threshold, about their relative ordering, but it is uninformative below that
> threshold and it does not recover the most consequential coordinate. A row ranked fourth by
> ratio was 130-fold more damaging than the row ranked second in SmolLM2-1.7B, and in OLMo-7B
> the coordinate the ratio selects is 47-fold less damaging than the one it rejects. The
> criterion is best specified as a reproducible **detector** for high-gain candidates — a
> global maximum of the layer-relative ratio with an acceptance threshold of 5.0 — and not as a
> definition of the functionally critical row. The SmolLM2-1.7B layer-7 result makes the same
> point at the finest available granularity: within one layer, one gated-FFN block and one set
> of controls, structural prominence, individual criticality and joint behaviour come apart,
> with a near-inert row required for another row's damage to appear at all.

---

## E7 — New main figure

**Artwork:** `results/paper_closing/fig_within_model_slope_2panel.{pdf,png}` (2 panels,
generated by `scripts/paper_closing/plot_within_model_slope_2panel.py`).

**Caption:**

> **Figure N. The activation ratio detects high-gain rows and orders them above its acceptance
> threshold, but does not grade severity below it.** Relative native-loss increase after
> ablating a single `down_proj` row (α=0), against that row's layer-relative activation ratio,
> for 36 rows log-spaced by ratio rank within one layer. (**A**) GENERator-EUK-3B, layer 4
> (genomic decoder; frozen candidate r2371, star). (**B**) SmolLM2-1.7B, layer 7 (text decoder;
> frozen candidate r227, star). Rows were selected by activation ratio alone and never by causal
> outcome; both axes are logarithmic, with the vertical axis symmetric-log below 10⁻⁵ so that
> sign changes remain visible. Shading marks the detector's acceptance region (ratio ≥ 5). Inset
> Spearman coefficients are computed over all rows, excluding the frozen candidate, over
> sub-threshold rows only, and over supra-threshold rows only; the last is an exploratory
> partition (see Results). n=36 rows per panel, of which 21 fall below and 15 at or above
> threshold.

---

## E8 — Supplementary

**S1 (Model panel and provenance).** For Llama-7B, Mistral-7B and OLMo-7B, change the
selection-rule verification status from unverified to verified and cite
`audit/detector_provenance_exp2_resolution.csv`. Add columns for the ratio maximum, the
absolute-activation maximum, and the frozen coordinate's global ratio rank. No coordinate
changes.

**New S-table (within-model graded sweep).** 72 rows — 36 per model — with columns
`model, layer, row, ratio_rank, activation_ratio, nll, delta_nll, rel_delta,
is_frozen_candidate`. Sources: `within_model_slope.tsv`,
`within_model_slope_smollm2_1.7b.tsv`.

**New S-table (SmolLM2-1.7B layer-7 interactions).** Three singles and three pairs with
joint effect, sum of singles, interaction, and joint-over-sum ratio, plus the cosine/norm
geometry and its 19,900-pair reference distribution. Sources:
`smollm2_second_row_epistasis.json`, `smollm2_161_749_geometry.json`.

**Robustness note.** Substituting OLMo-7B's ratio-maximum coordinate for its published one
moves the cohort correlation between activation ratio and full-ablation effect from ρ=0.766 to
ρ=0.755 across all 22 models, and from ρ=0.770 to ρ=0.673 (p=0.033) within the 10 text
decoders; omitting OLMo-7B entirely gives ρ=0.769 and ρ=0.783. The text-decoder subgroup's
95% interval reaches +0.032 under substitution, so that n=10 subgroup should not be reported
as independently robust. Note that the substituted ratio was measured here while the other 21
values derive from the census discovery input; the omission analysis is convention-free and
agrees. Source: `exp2_olmo_exposure_sensitivity.json`.

---

## Do NOT apply: one superseded recommendation

`PAPER_CLOSING_REPORT.md` §7 row 3 recommends replacing the enrichment language with
*"activation **ratio** predicts causal severity (ρ = 0.766, CI [0.53, 0.88])"*. **That
upgrade is not supported and should not be made.** It rests on a between-model correlation in
which each model contributes one row, so model identity is confounded with the ratio. The
within-model sweeps in E2 test the implied claim directly and refute it below the detector
threshold in both domains. The manuscript's existing wording — an enrichment signal for
functional importance, not a calibrated measure of effect size — is correct and should stand
unchanged.

---

## Note on the target file

These edits have **not** been written into `paper-salvage/actual_manuscript.md`. At
2026-09-08 18:39 UTC that file was overwritten in the working tree with pandoc LaTeX output
(`\documentclass…`, 1245 lines) by a different process; two other Claude Code sessions were
running against this repository at the time. The conversion is faithful — all 20 sections and
every claim phrase checked are present — and the original Markdown remains intact in git at
`HEAD:paper-salvage/actual_manuscript.md` (860 lines, 67,425 bytes), so nothing is lost either
way. Applying these edits requires deciding which of the two is the source of truth; every
edit above is keyed to section names and quoted anchors so that it applies to either without
modification.
