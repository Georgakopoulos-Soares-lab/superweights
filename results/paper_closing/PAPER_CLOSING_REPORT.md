# PAPER CLOSING REPORT
### *Structure is not mechanism: high-gain gated-FFN rows across text and genomic foundation models*

Generated 2026-09-01. All statistics from `results/paper_closing/`; provenance in
`audit_manifest.tsv`; one row per experiment in `paper_closing_summary.tsv`.

---

## 1. Executive summary

**What survived.** The cross-model phenotype and the enrichment claim survive unchanged.
Frozen candidates are same-layer **rank 1 on 528/528 (22 models × 24 inputs)**, exactly as the
manuscript states. Neither weight-space predictor calibrates causal severity.

**What changed — the strongest new result.** The paper tested q1 and Frobenius against causal
damage but never tested the *selection* variable. Doing so shows **activation ratio predicts
severity strongly and stably: ρ = +0.766 (95% CI [+0.532, +0.884], p < 1e-4)**, replicating on
24 independent inputs (ρ = +0.768) and surviving family de-duplication (ρ = +0.766,
[+0.629, +0.881]). Critically it must be the *ratio*: absolute activation is non-significant
(ρ = +0.375, CI crosses zero). So activation selection does not merely enrich — **it
calibrates**, and the layer-relative denominator is doing the work.

**Three manuscript claims must change.**
1. **The GENERator effect follows BOS *token identity*, not position 0.** Moving BOS to
   position 1 moves the activation with it (argmax@BOS = 1.00, argmax@0 = 0.00); a neutral
   token at position 0 receives **43.09** versus BOS's **354,530** — an 8,228× gap.
2. **L4/r2371 does not *support* the BOS sink — it attenuates it.** Ablation *increases* BOS
   attention 30% (fold 7.90 → 10.24). The manuscript's directional claim is not supported.
3. **The DNABERT pair interaction is not compensatory.** Ablating either row leaves the
   other's activation at ×1.0000. The rows are significantly **anti-aligned** (cos = −0.1876,
   z = 3.50) yet co-activate almost perfectly (r = +0.9986).

**Biggest unresolved limitation.** The non-SW / non-genuine contrast rests on a single model
(Evo1), and the 22 models are ~12 families, so every cross-model ρ is a robustness statement,
not population inference. Experiment 2 (uniform detector re-selection across all executable
models) was **not run** — see §6.

---

## 2. Cross-model activation vs function

Endpoint `R_cand_eps1.0` (signed relative native-loss change), Spearman ρ, 10k bootstrap, seed 42.

| predictor | all (n=22) | 95% CI | p | text-decoder (n=10) |
|---|---|---|---|---|
| **activation ratio** | **+0.766** | [+0.532, +0.884] | <1e-4 | **+0.770** (p=0.009) |
| 24-input median ratio | +0.768 | [+0.534, +0.889] | <1e-4 | +0.770 |
| absolute activation max | +0.375 | [−0.063, +0.713] | 0.085 | +0.224 |
| ratio CV across inputs | +0.130 | [−0.372, +0.585] | 0.565 | −0.406 |
| q1 | +0.074 | [−0.393, +0.524] | 0.744 | −0.333 |
| layer-relative Frobenius | +0.441 | [+0.068, +0.721] | 0.040 | **−0.042** |

**Stability across cohort definitions is the discriminator.** Activation ratio: +0.766 pooled,
+0.720 one-per-family (deterministic), +0.766 resampled, +0.770 text-decoder — stable. q1:
+0.074 / +0.175 / +0.441 / −0.333 — sign-unstable. Frobenius: +0.441 / +0.392 / +0.617 /
−0.042 — its pooled significance vanishes in the homogeneous subset, indicating cross-
architecture heterogeneity rather than prediction.

`activation_vs_causality.tsv`, `family_clustered_sensitivity.tsv`,
`fig_predictor_comparison_4panel.pdf`.

---

## 3. Uniform detector sensitivity — PARTIALLY RUN (see §3b)

Experiment 2 requires re-running the ratio-argmax detector over all eligible layers/rows for
every executable model and causally evaluating each changed coordinate. This was not executed
in this session. **Mitigating evidence already on file:** `audit/detector_provenance.csv`
records, per model, whether the frozen candidate is the ratio-argmax and the activation-argmax
(both fields populated 22/22). That file should be used to bound the exposure, and the
experiment should be run before submission. Reported as an open reviewer risk, not as a
result.

---

## 3b. Uniform detector sensitivity — RUN for the executable violators

The provenance audit found **exactly 5 of 22** models whose frozen candidate is not verified
to be the ratio-argmax. Two were cached and have now been re-selected under the uniform rule
(`uniform_detector_sensitivity.tsv`):

| model | condition | frozen | uniform argmax | same? | frozen global rank | causal frozen | causal uniform |
|---|---|---|---|---|---|---|---|
| DNABERT-2 | default | L5/r603 | **L8/r603** | no | **11** | **−0.132** | **+0.107** |
| DNABERT-2 | no-special | L5/r603 | **L9/r264** | no | 5 | −0.132 | −0.034 |
| NTv3 | default | L11/r1472 | L11/r1472 | **yes** | 1 | — | — |
| NTv3 | no-special | L11/r1472 | L11/r1472 | **yes** | 1 | — | — |

**DNABERT-2's frozen coordinate is confirmed mis-selected** — layer-rank 1 but global-rank 11.
The uniform candidate is *more* causally damaging (+0.107 vs −0.132 on identical probes and a
fixed mask), so adopting the uniform rule **strengthens** the enrichment claim. Note the frozen
row's own effect is negative on this probe set.

**NTv3 does not violate under multi-probe detection** (frozen = uniform, global rank 1, both
special-token settings), whereas the round-2 audit recorded it as ratio-rank 3 with L6/r1472
winning. The discrepancy is **probe dependence** — which coordinate wins depends on the input.
That is direct evidence for the multi-input requirement in the rule below, and it means NTv3's
recorded "violation" was a single-probe artifact, not a mis-selection.

**Causal numbers here use 6 short synthetic probes, not the census eval pool**, so they are not
comparable to census `R_cand` values; they are internally paired (frozen vs uniform, identical
inputs) which is the valid contrast.

### The universal selection rule (as specified after this audit)

1. **Statistic: layer-relative ratio, never absolute activation.** Absolute activation fails as
   a predictor (ρ = +0.375, ns); 6/22 candidates were chosen by it.
2. **Scope: global argmax over all layers × rows** — this is what DNABERT-2 violates.
3. **Special-token handling declared, and reported both ways** (see §4b: it changes both the
   magnitude *and* the winning coordinate).
4. **Multiple diverse inputs with a rank-1 stability requirement** (≥90%). Already met at
   528/528, so it costs nothing and forecloses regression-to-the-mean.
5. **Record rank and ratio of the frozen coordinate** so "is it the argmax?" is answerable from
   the artifact — precisely what could not be answered for 3 models.
6. **Drop >5 as a selection threshold**; keep it as a reported diagnostic (all 22 exceed >10).

### Remaining exposure — ✅ RESOLVED 2026-09-08 (run here, all 3 models)

Llama-7B, Mistral-7B and OLMo-7B were the last unverified legacy candidates. All three were
downloaded (51.8 GB) and run on this machine. Full write-up:
[`EXP2_LEGACY_DETECTOR_RESOLUTION.md`](EXP2_LEGACY_DETECTOR_RESOLUTION.md); machine-readable
per-model rows: `audit/detector_provenance_exp2_resolution.csv` (additive — no census or audit
file was modified).

Each run passed **four** reproduction gates from `census_master.csv` before interpretation
(`baseline_loss`, `R_cand` at ε=1.0 and ε=0.5, and the 5 stored control rows redrawn from
`SeedSequence(42).spawn(23)[panel_index]`): observed relative errors 2.1e-09 – 3.1e-06, control
rows exact. The detector positive control on SmolLM2-1.7B also passed (rank 1/49152, 100 %
stable).

| model | frozen | ratio-argmax | agree? |
|---|---|---|---|
| **Llama-7B** | L2 / r3968 | L2 / r3968 | ✅ rank **1/131072**, **100 %** input-stable, identical under all 3 tokenizations |
| **Mistral-7B** | L1 / r2070 | L1 / r2070 | ✅ rank **1/131072**, **100 %** input-stable, identical under all 3 tokenizations |
| **OLMo-7B-0724-hf** | L1 / r269 | **L2 / r269** | ❌ frozen is rank **2/131072**; robust disagreement (same row, one layer deeper) |

**2 of 3 confirm the frozen choice exactly.** For OLMo the disagreement is real but runs
*against* the ratio rule: ablating the frozen L1/r269 costs **+1.1178** while ablating the
ratio-argmax L2/r269 costs **+0.0237** — the coordinate the ratio rule prefers is **47× less
damaging**. DNABERT-2 goes the other way (argmax +0.1072 vs frozen −0.1322), so the ratio rule
is **neither uniformly better nor worse** than the legacy rule; it is a different statistic and
neither reliably lands on the most causal coordinate.

**Exposure to the headline ρ is bounded** (`exp2_olmo_exposure_sensitivity.json`): the 22-model
result is insensitive (+0.766 published → +0.769 dropping OLMo → +0.755 substituting the
argmax coordinate). The text-decoder subgroup moves +0.770 → +0.673 (p=0.033) in the worst
case — still significant, but its CI lower bound falls to +0.032, so that **n=10 subgroup was
already fragile and should not carry weight alone** regardless of this substitution.

**Consequence for §3's "universal selection rule".** Describe it as **the detector**, not as
the definition of the super row. "Global argmax of the layer-relative ratio, accept if ≥5" is
a reproducible way to *find* candidates and should be specified as such; it must **not** be
presented as identifying the most causally important coordinate — OLMo-7B (47×) and
SmolLM2-1.7B L7 (130×) falsify that in two different geometries.

---

## 2b. Within-model calibration sweep — enrichment yes, linear calibration no

Every earlier result is either between-model (n=22, one row per model, confounded by model
identity) or within-model but *binary* (candidate vs 5 random same-layer controls, which sit at
ratio ≈ 1 by construction). Neither can show a slope. This measures both variables for 36 rows
inside GENERator-EUK layer 4, holding every model-level factor fixed
(`within_model_slope.tsv`):

| | ρ | p | 95% CI |
|---|---|---|---|
| all 36 rows | **+0.454** | 0.0054 | [+0.077, +0.760] |
| excluding the frozen candidate | **+0.406** | 0.0156 | — |

The slope survives removal of the extremum, so it is not one outlier driving a rank statistic.
But the shape is **threshold-like, not linear**:

| rank | 1 | 2 | 3 | 4 | 5 | … | 3072 |
|---|---|---|---|---|---|---|---|
| ratio | 7124.1 | 2730.7 | 531.3 | 498.4 | 377.7 | … | 0.03 |
| rel. damage | **+0.368** | **+0.107** | −0.0001 | +0.0025 | +0.0012 | ~0 | +0.00001 |

Only **10 of 36** rows exceed |damage| = 1e-4. The top two carry essentially everything: a 3×
drop from rank 1 to 2, then a ~40× collapse to rank 3. A row at ratio 531 is indistinguishable
from a row at ratio 0.03.

> **Wording consequence.** "Activation extremeness *calibrates* causal severity" is too strong.
> Supported: activation ratio **orders** rows correctly and **identifies the small set that
> matters**. Not supported: damage scaling smoothly with ratio. The between-model ρ = 0.766
> should be read as "models with a more extreme top row have a more damaging top row", not as a
> dose-response law.

### Replication in a text decoder — RUN 2026-09-08 (closes the one-model limit)

The sweep was repeated in **SmolLM2-1.7B** (text decoder, L7/r227) with identical logic and
that model's own frozen census endpoint — `within_model_slope_smollm2_1.7b.{json,tsv}`,
`within_model_slope_comparison.json`. Three stored census quantities reproduced first as
gates: `activation_ratio` 4439.34 (exact), `baseline_loss` to 1.5e-08 relative,
`R_cand_eps1.0` to 1.6e-07 relative. Both sweeps are n=36 with n=21 sub-threshold rows.

| | GENERator-EUK-3B (genomic) | SmolLM2-1.7B (text) |
|---|---|---|
| ρ, all rows | +0.454 (p=5.4e-3) | **+0.625** (p=4.6e-5) |
| ρ, excl. frozen candidate | +0.406 (p=1.6e-2, n=35) | **+0.592** (p=1.8e-4, n=35) |
| ρ, ratio < 5 only | +0.040 (p=0.86, n=21) | **−0.584** (p=5.4e-3, n=21) |
| ρ, ratio ≥ 5 only *(post hoc)* | +0.639 (p=1.0e-2, n=15) | **+0.975** (p=7.1e-10, n=15) |
| ρ, ratio ≥ 5 excl. candidate *(post hoc)* | +0.556 (p=3.9e-2, n=14) | +0.969 (p=1.2e-8, n=14) |
| candidate ÷ median sub-threshold damage | 6.1e4 × | 5.4e5 × |

**The relation is two-regime, and it replicates across domains.** Below the detector's own
≥5 accept rule the ratio carries no positive graded signal — flat in the genomic decoder,
mildly *inverted* in the text decoder, at damage magnitudes ~1e-5 (physically negligible,
though ~1e3× above the harness's 1.5e-8 float reproducibility, so the inversion is a real
ordering of trivial effects and not noise). Above the threshold the ratio **orders** rows
well and survives removing the extremum. The mediocre pooled ρ reported above is partly a
dilution artifact of averaging a noise regime with a graded one.

⚠️ The sub-/supra-threshold decomposition is **post hoc** — added after the pre-specified
statistics (ρ all, ρ excl. candidate, ρ ratio<5) were computed and plotted. The split point
is the detector's pre-existing ≥5 rule and was not fitted to these data, but the decision to
decompose came after seeing the scatter. Labelled as such everywhere it appears.

**The §2b wording consequence stands, and is now better supported.** Ratio ordering is good;
ratio-as-dose-response is still not: see the next subsection for a case where the row ranked
**4th** by ratio is 130× more damaging than the row ranked **2nd**.

## 2c. ★ A second critical row, and a masking interaction the census cannot see

The graded sweep was not designed to find this; it fell out of sweeping 36 rows instead of 1.
SmolLM2-1.7B layer 7 contains **two** rows whose single-row ablation is catastrophic, and a
higher-ratio row that is nearly inert (`smollm2_second_row_epistasis.json`):

| row | activation ratio | ratio rank | rel. NLL increase, α=0 |
|---|---|---|---|
| **227** (frozen census candidate) | 3181.70 | 1 | **+6.9887** |
| **161** (never reported anywhere) | 63.54 | 4 | **+2.8718** |
| 749 | 358.56 | 2 | +0.0221 |

A row ranked **4th** by activation ratio is **130× more damaging** than the row ranked 2nd.
The census reports one candidate per model, so row 161 has never appeared in any artifact.

Pairwise ablations, same endpoint. **Sign convention: this is a loss-increase endpoint, so
epistasis > 0 is super-additive synergy — the OPPOSITE sign to the project's accuracy-endpoint
epistasis numbers (e.g. DNABERT-2's −33.63 pp), which must be converted before comparison.**

| pair | joint | sum of singles | epistasis | joint ÷ sum | verdict |
|---|---|---|---|---|---|
| 227 + 161 | +8.6655 | +9.8605 | −1.1950 | 87.9 % | sub-additive |
| 227 + 749 | +7.7289 | +7.0108 | +0.7181 | 110.2 % | super-additive |
| **161 + 749** | **+0.0207** | +2.8939 | **−2.8732** | **0.7 %** | **masking** |

**Ablating row 749 — which costs +2.2 % on its own — completely abolishes row 161's +287 %
catastrophe.** Row 161's criticality is contingent on row 749 being present; remove both and
the model is essentially intact.

*Verified independently.* All six conditions were re-measured by a separate implementation
that zeroes rows by explicit indexing rather than `e10_lib.masked`, restores from separately
held clones, and re-measures the intact loss after every restore: every value reproduced
exactly, order-independent, weight drift 0.00e+00. (The cancellation has exactly the shape a
weight-aliasing bug would produce, hence the check; `_save_row` does `.clone()`.)

*Geometry* (`smollm2_161_749_geometry.json`), data-free, referenced against all 19,900 pairs
among the 200 highest-norm rows of this layer (mean −0.0001, sd 0.0224):

| pair | cosine | z | percentile |
|---|---|---|---|
| 161 · 749 | **+0.3952** | +17.6 | **100.00** (the maximum) |
| 227 · 749 | **−0.3980** | −17.8 | **0.00** (the minimum) |
| 227 · 161 | −0.0323 | −1.4 | 6.91 |

The three rows are the most extreme aligned and anti-aligned pair in the layer, at ~18σ. Note
161 and 749 are strongly **aligned**, not opposed — an "opposed pair" reading of the masking
result is ruled out by this measurement.

> **Status.** The causal facts and the geometric facts are both **[MEASURED]** and
> independently verified. The **link between them is [UNTESTED]** — why positive alignment
> plus a 3× norm asymmetry (‖749‖=24.61, ‖161‖=7.91, ‖227‖=26.84) produces near-total masking
> is a hypothesis, not a result. An activation-level mediation test is the obvious next step
> and has not been run. Do not write a mechanism sentence for 2c yet.

**Why this matters for the manuscript's thesis.** "Structure is not mechanism" currently
rests on cross-model dissociations. This is the same dissociation *inside a single layer*:
the activation ratio ranks 749 above 161, yet 749 is nearly inert alone while 161 is
catastrophic alone — and 161's catastrophe is a property of the *pair*, not of either row.
It is a sharper instance of the thesis than anything currently in the draft, and it also
bounds the one-candidate-per-model census design: the census undercounts critical rows.

---

## 4. GENERator mechanism

**Position (EXP4).** Full ablation, then restoring the intact contribution at exactly one
position:

| p | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| rescue | **0.9996** | 0.0111 | −0.0014 | 0.0003 | −0.0013 | −0.0011 | 0.0005 | −0.0002 |
| accessibility (T−p)/T | 1.000 | 0.988 | 0.977 | 0.953 | 0.907 | 0.814 | 0.628 | 0.256 |

**90× drop from p=0 to p=1 while accessibility falls only 1.2%**; max |rescue| for p ≥ 2 is
0.0014. Generic early-position broadcasting is excluded. p=1 is small but its CI excludes zero
([+0.0013, +0.0213]) — a real ~1% residual. Provenance reproduced bit-for-bit before running
(intact 6.385380 vs historical 6.385381; ablated 8.754320 vs 8.754320).

**Token identity (EXP5).** With `add_special_tokens=False` (essential — the tokenizer has
`add_bos_token=True` and otherwise auto-prepends BOS to every condition):

| condition | act @ pos0 | argmax@0 | argmax@BOS |
|---|---|---|---|
| standard (BOS@0) | 375,361 | 1.00 | 1.00 |
| no_bos (content@0) | 374,347 | 0.88 | — |
| **shifted_bos** (neutral@0, BOS@1) | **43.09** | **0.00** | **1.00** |
| no_bos_pad (neutral@0) | 43.09 | 0.00 | — |

Activation follows BOS. Position 0 is neither necessary nor sufficient. In BOS's absence the
first *contentful* token inherits the role.

**Attention sink (EXP6).** Against the causal-mask-aware baseline mean_i[1/(i+1)] = 0.0475
(not the naive 1/T): S_BOS = 0.3753, **fold = 7.90×**, max 21.03×, BOS top-attended for 55.8%
of queries, 82.9% of layer-heads above 2×. BOS is a genuine sink.

**Causal relation to the sink (EXP7).** fold: intact 7.90 → full ablation **10.24** → BOS-only
ablation 10.14 → preserve-BOS 7.89 → restore-at-BOS 7.89 → restore-at-non-BOS 10.24. The
conditions track the NLL rescue pattern exactly, so the coupling is real and BOS-mediated —
but **ablation strengthens the sink by 30%**. The row attenuates rather than creates it.

**Direction specificity (EXP8).** 20 random unit directions, scale-searched to full-ablation
damage: **20/20 matched** (tolerance 15%) and **20/20 reproduced the GC shift**
(GC 0.3505 ± 0.0070 vs ablated 0.3455, intact 0.4592). The GC phenotype is **generic
damage-linked**, not learned-direction-specific.

---

## 4b. Special-token dependence (EXP3) — an architecture dissociation

Matched sequence content, tokenizer default vs `add_special_tokens=False`:

| model | row | default ratio | no-special ratio | collapse | max at special token? |
|---|---|---|---|---|---|
| GENERator-EUK | L4/r2371 | 7124.1 | **7089.0** | **1.00×** | default yes, no-special no |
| GENERator-PROK | L8/r260 | 257.3 | **257.1** | **1.00×** | default yes, no-special no |
| DNABERT-2 | L9/r264 | 238.6 | **1.85** | **129×** | always (1.00) |
| DNABERT-2 | L9/r294 | 228.0 | **9.05** | **25×** | always (1.00) |
| DNABERT-2 | L3/r86 | 61.4 | **1.46** | **42×** | always (1.00) |
| DNABERT-2 | L5/r603 | 65.2 | **2.16** | **30×** | always (1.00) |

**The decoders' phenotype is not special-token-dependent; the encoder's is entirely so.**
GENERator's ratio is unchanged when special tokens are removed, because the first
*contentful* token inherits the role (consistent with EXP5's `no_bos` condition, 374,347 at
position 0). Every DNABERT-2 candidate collapses to ratio 1.46–9.05 — barely above an
ordinary channel — and its activation maximum sits on a special token in 100% of default
measurements (at median positions 12–17, i.e. SEP-like, **not** position 0).

This reframes the DNABERT special-token sensitivity flagged in the manuscript: it is not an
implementation nuisance to be normalised away but a **mechanistic property that differs by
architecture**. It should be reported, not suppressed.

---

## 4c. Structurally matched replacements (EXP9)

Replacing L4/r2371 with vectors sharing progressively more of its structure, all renormalised
to the original row norm (`generator_matched_replacements.tsv`):

| family | cos to row | ΔNLL @c=0.5 (× ablation) | ΔNLL @c=2.0 (× ablation) | ΔGC @c=2.0 |
|---|---|---|---|---|
| random unit | −0.008 | +2.237 (0.98×) | +1.977 (0.86×) | −0.035 |
| random same-layer row | −0.005 | +2.300 (1.00×) | +2.319 (1.01×) | −0.111 |
| top-Frobenius row | −0.506 | +2.346 (1.02×) | +1.141 (0.50×) | −0.059 |
| permuted entries | −0.027 | +2.274 (0.99×) | +2.207 (0.96×) | −0.094 |
| **sign-flipped** | −0.033 | +2.232 (0.97×) | **+0.348 (0.15×)** | **−0.029** |
| orthogonalised | +0.000 | +2.250 (0.98×) | +2.048 (0.89×) | −0.068 |

At matched norm every replacement is as damaging as deletion (0.97–1.02× ablation damage) —
**no learned direction is recoverable from structure alone.** But damage is **non-monotonic in
scale**: the sign-flipped vector at 2× norm avoids **85% of the damage** (ΔNLL +0.348 vs
ablation +2.292) and recovers 74% of the GC shift, despite cosine −0.033 to the original.

Read with EXP8 (20/20 damage-matched random directions reproduce the GC shift), the mechanism
depends on **delivering sufficient magnitude with roughly the right magnitude profile**, not on
the specific learned direction. This is the paper's title claim in its strongest form:
*structure is not mechanism.*

---

## 5. DNABERT-2 mechanism

| measurement | value |
|---|---|
| cosine(r264, r294) | **−0.1876** (random same-layer 0.0008 ± 0.0538, z = 3.50) |
| per-token activation correlation | **+0.9986** (n = 20,305) |
| r264 activation after ablating r294 | **×1.0000** |
| r294 activation after ablating r264 | **×1.0000** |
| accuracy: base / −A / −B / −AB | 0.9288 / 0.9285 / 0.9277 / **0.5912** |

The interaction is robust (**claim 12 SUPPORTED**) but is **not compensatory** and **not
redundant duplication**: the rows co-activate almost perfectly while writing in significantly
opposing directions, and neither changes when the other is removed. The joint effect is
therefore **downstream/convergent**. No further mechanism label is asserted.

---

## 6. Reviewer-risk audit

1. **Experiment 2 not run** (§3) — mixed candidate provenance remains unaddressed empirically.
2. **Experiment 2 remains the only unrun analysis.** EXP3 (special-token dependence) and
   EXP9 (structurally matched replacements) were completed — see §4b/§4c. EXP3 covers 2
   decoders and 4 DNABERT-2 candidates; NTv3 could not be measured in the same pass (device
   placement error under the shared wrapper) and is outstanding.
3. **n = 22 across ~12 families.** All ρ are robustness statements; CIs are bootstrap over
   models, not population inference.
4. **Endpoint heterogeneity.** Causal-LM NLL and MLM loss are pooled in the "all" cohort;
   the text-decoder subset is the homogeneous comparison and is reported alongside.
5. **EXP7 direction is opposite to the manuscript's hypothesis** and must be rewritten, not
   softened.
6. **A tokenisation error was made and caught** in the first EXP5 run (auto-BOS made all four
   conditions identical). The corrected run is the reported one; the error is recorded here
   because the same trap applies to any future special-token experiment.
7. **EXP8 GC generation used 12 prompts / 48 new tokens**, smaller than the frozen GC assay.

---

## 7. Exact manuscript changes

| # | current claim | recommended replacement | reason |
|---|---|---|---|
| 3 | activation extremeness may only enrich | "activation **ratio** predicts causal severity (ρ = 0.766, CI [0.53, 0.88]); absolute activation does not (ρ = 0.375, ns)" | EXP1 |
| 5 | Frobenius poorly predicts | keep, and add: "its pooled association (ρ = 0.44) vanishes within text-decoders (ρ = −0.04)" | EXP1/EXP12 |
| 6 | mediated primarily through BOS | keep; strengthen with the p-curve (99.96% at p=0) | EXP4 |
| 7 | BOS-specific vs early-position geometry | "rescue is confined to p = 0 (90× drop to p = 1 while accessibility falls 1.2%)" | EXP4 |
| 8 | (conflated) | "the effect follows **BOS token identity**: moving BOS to position 1 moves the activation with it; a neutral token at position 0 receives 43 vs 354,530" | EXP5 |
| 9 | BOS is an attention sink | keep, but state the **causal-mask-aware** baseline: 7.90× mean_i[1/(i+1)], not 1/T | EXP6 |
| 10 | r2371 supports the sink | **"ablating r2371 increases BOS attention 30% (7.90→10.24); the row attenuates rather than creates the sink"** | EXP7 |
| 11 | GC phenotype is damage-linked | keep, now supported by 20/20 damage-matched random directions | EXP8 |
| 13 | interaction is compensatory/redundant | **"not compensatory: cross-ablation leaves the survivor at ×1.0000; the rows are anti-aligned (cos −0.19) yet co-activate (r 0.999)"** | EXP10 |

---

## 8. Figure plan

**Main.** (F1) 4-panel predictor comparison — activation ratio / absolute activation / q1 /
Frobenius vs Δloss. (F2) GENERator position-rescue curve with the accessibility baseline
overlaid. (F3) BOS identity-vs-position bar panel. (F4) attention-sink fold under the six
interventions.

**Supplementary.** input-stability heatmap (22 × 24 rank + ratio); activation-ratio
distribution with the >3/>5/>10 thresholds; random-direction ΔGC vs ΔNLL scatter;
family-clustered resampling distribution.

---

## 9. Statistical appendix

- Spearman ρ, 10,000-resample percentile bootstrap, `numpy.random.default_rng(42)`.
- EXP4 rescue: paired over 100 windows, 5,000 bootstrap resamples; rescue = (L_abl − L_res)/(L_abl − L_int).
- EXP6/7: 24 windows, all layers × heads, causal baseline mean_i[1/(i+1)] over i>0.
- EXP8: 20 directions, scale grid {0.25,0.5,1,1.5,2,3}, 15% damage tolerance, 40 damage windows, 12 GC prompts.
- EXP10: 8 batches × 32, 20,305 tokens; 500 random same-layer pairs for the cosine null.
- EXP12: 2,000 one-per-family resamples over 12 families.
- **Multiple testing:** 7 predictors × 4 endpoints × 4 cohorts = 112 tests were computed and
  are reported in full without correction. The headline (activation ratio, p < 1e-4) survives
  Bonferroni at 112 tests (α = 4.5e-4); q1 and absolute activation would not have been
  significant under any correction.
