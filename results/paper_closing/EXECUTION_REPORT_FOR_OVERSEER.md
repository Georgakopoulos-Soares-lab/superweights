# Execution report — extra-experiment program
### Report back to the framing agent. Maps each specified item to what was run, what it found, and what residual risk remains.

Generated 2026-09-01. Repo `superweights`, branch `mechanism-and-negative-results` (merged with
`integrate/`). All artifacts in `results/paper_closing/` (16 deliverables, 22 summary rows).
Provenance in `audit_manifest.tsv`. Machine-readable: `paper_closing_summary.tsv`.

**Tagging:** [MEASURED] = observed in this session · [INFERRED] = reasoned from measurements ·
[UNTESTED] = stated but not run · [NOT RUN] = specified but not executed.

---

## 0. Headline for the overseer

The program did four things worth acting on:

1. **Closed the central gap.** The selection variable had never been tested against causal
   damage. It predicts: activation *ratio* ρ = **+0.766** (CI [+0.532, +0.884], p < 1e-4),
   replicating on 24 independent inputs (+0.768) and surviving family de-duplication (+0.766).
   Absolute activation does **not** (+0.375, ns) — the layer-relative denominator carries it.
2. **Falsified three manuscript claims** (8, 10, 13) and **weakened one wording choice**
   ("calibrates" → "orders and identifies").
3. **Found that one specified experiment (EXP2) was the real exposure**, and partially ran it:
   exactly 5/22 models violate the uniform rule; 2 resolved, 3 blocked on downloads.
4. **Added one experiment the spec did not request** — a within-model graded sweep — which
   turned out to be necessary, because everything else was either between-model (confounded) or
   within-model but binary. It is the reason the "calibrates" wording had to change.

**Two self-caught errors are documented in §6.** Both were mine, both would have produced
wrong published numbers, and both were found by cross-checking rather than by the pipeline.

---

## 1. Priority-1 items (all 5 complete)

### EXP1 — activation extremeness vs causal damage [MEASURED]
Spec asked for activation predictors vs causal endpoints, Spearman + bootstrap, text-decoder
subset, and a 4-predictor comparison. Runnable with **no model execution**: the census had
`activation_ratio` populated **0/22** while q1/Frobenius/endpoints were 22/22, but
`results/E13_candidate_stability/*.json` held 22 models × 24 inputs with `activation_max`,
`activation_ratio`, `rank`, `percentile`, `max_position`. Joined on `(layer,row)`, matched 22/22.

Endpoint `R_cand_eps1.0`, 10k bootstrap, seed 42:

| predictor | all (n=22) | 95% CI | p | text-dec (n=10) | text-enc (n=6) | genomic (n=6) |
|---|---|---|---|---|---|---|
| **activation ratio** | **+0.766** | [+0.532,+0.884] | <1e-4 | **+0.770** (0.009) | +0.543 (0.27) | +0.600 (0.21) |
| 24-input median ratio | +0.768 | [+0.534,+0.889] | <1e-4 | +0.770 | — | — |
| absolute activation | +0.375 | [−0.063,+0.713] | 0.085 | +0.224 | +0.771 (0.07) | +0.086 |
| ratio CV over inputs | +0.130 | [−0.372,+0.585] | 0.565 | −0.406 | — | — |
| q1 | +0.074 | [−0.393,+0.524] | 0.744 | −0.333 | −0.371 | +0.486 |
| layer-rel. Frobenius | +0.441 | [+0.068,+0.721] | 0.040 | **−0.042** | +0.543 | +0.486 |

**Stability across cohort definitions is the discriminator** [INFERRED]: activation ratio
+0.766 / +0.720 / +0.766 / +0.770 (pooled / one-per-family deterministic / one-per-family
resampled / text-decoder). q1 flips sign (+0.074 / +0.175 / +0.441 / −0.333). Frobenius is
significant **only pooled** and in **no** subgroup — a between-group heterogeneity effect, not
prediction.

**Cohort composition matters for interpretation** [MEASURED]: 10 text-dec / 6 text-enc /
4 gen-dec / 2 gen-enc. In non-decoder cohorts most candidates have effects ≈ 0 (encoders 2/6
above |0.05|; genomic 1/6), so ρ there is rank-ordering near-ties. **Recommend reporting
non-decoder cohorts descriptively only.**

Bonus [MEASURED]: `rank_median_24` returned NaN because rank is **constant** — the frozen
candidate is same-layer rank 1 on **528/528** observations, substantiating the stability claim
exactly as written. Max activation sits at position 0 in **61.2%** of observations.

Files: `activation_vs_causality.tsv`, `..._per_model.tsv`, `fig_predictor_comparison_4panel.*`,
`fig_activation_ratio_vs_{full_ablation,partial}.*`

### EXP4 — positional rescue curve [MEASURED]
Provenance gate reproduced **bit-for-bit** before measurement (intact 6.385380 vs historical
6.385381; ablated 8.754320 vs 8.754320). Reused the frozen harness (revision `7dc01bcc`,
tokenizer, corpus seed 42, 100 damage windows, hook class, bootstrap); added only a
parameterised restore-at-p condition.

| p | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| rescue | **0.9996** | 0.0111 | −0.0014 | 0.0003 | −0.0013 | −0.0011 | 0.0005 | −0.0002 |
| accessibility (T−p)/T | 1.000 | 0.988 | 0.977 | 0.953 | 0.907 | 0.814 | 0.628 | 0.256 |

**90× drop from p=0 to p=1 while accessibility falls 1.2%**; max |rescue| for p≥2 = 0.0014.
Generic early-position broadcasting is excluded. p=1 is small but its CI excludes zero
([+0.0013,+0.0213]) — a real ~1% residual, reported.

### EXP5 — BOS identity vs position 0 [MEASURED]
| condition | act @ pos0 | argmax@0 | argmax@BOS |
|---|---|---|---|
| standard (BOS@0) | 375,361 | 1.00 | 1.00 |
| no_bos (content@0) | 374,347 | 0.88 | — |
| **shifted_bos** (neutral@0, BOS@1) | **43.09** | **0.00** | **1.00** |
| no_bos_pad (neutral@0) | 43.09 | 0.00 | — |

Activation follows BOS to position 1; a neutral token at position 0 receives 43.09 vs BOS's
354,530 — an **8,228× gap**. Position 0 is neither necessary nor sufficient. In BOS's absence
the first *contentful* token inherits the role.

### EXP6 — attention sink, causal-mask-aware [MEASURED]
Baseline mean_i[1/(i+1)] = 0.0475 (not naive 1/T). S_BOS = 0.3753 → **fold 7.90×**, max 21.03×,
BOS top-attended for 55.8% of queries, **82.9%** of layer-heads above 2×.

### EXP7 — does the row support the sink? [MEASURED — direction opposite to hypothesis]
| condition | fold |
|---|---|
| intact | 7.90 |
| **full ablation** | **10.24** |
| BOS-only ablation | 10.14 |
| preserve-BOS-only | 7.89 |
| restore at BOS | 7.89 |
| restore at matched non-BOS | 10.24 |

Conditions track the NLL rescue pattern exactly, so the coupling is real and BOS-mediated —
but **ablation *increases* the sink 30%**. The row **attenuates** rather than creates it.

---

## 2. Priority-2 items

### EXP2 — uniform detector sensitivity [PARTIALLY RUN — the key residual risk]
Provenance audit result: **exactly 5/22** models are not verified ratio-argmax.

| model | domain/arch | status | resolution |
|---|---|---|---|
| DNABERT-2 | gen-enc | **VIOLATES** | **resolved**: frozen L5/r603 is layer-rank 1 but **global-rank 11**; uniform picks L8/r603, which is **more** damaging (+0.107 vs −0.132) |
| NTv3 | gen-enc | audit said VIOLATES | **resolved differently**: frozen = uniform, global rank 1, both special-token settings → the recorded violation was a **single-probe artifact** |
| Llama-7B | text-dec | UNVERIFIED | **blocked** (uncached) |
| Mistral-7B | text-dec | UNVERIFIED | **blocked** |
| OLMo-7B | text-dec | UNVERIFIED | **blocked** |

The 17 compliant models are ratio-argmax (16) plus GENERator-EUK (legacy but verified
coincident, ratio 7124, rank 1/1). Adopting the uniform rule **strengthens** enrichment where
tested. **Residual risk:** the 3 blocked models are all text decoders — the cohort carrying
ρ = +0.770 — and all were selected by absolute activation, the statistic EXP1 showed does not
predict (+0.375, ns). If any is not the ratio-argmax, 3/10 points in the headline correlation
were chosen by the wrong feature. Cost ≈ 2.5–4 h, **download-bound not GPU-bound** (4/22 cached).

Proposed universal rule (6 clauses) is in `PAPER_CLOSING_REPORT.md` §3b.

### EXP8 — multiple damage-matched random directions [MEASURED]
20 directions, scale-searched, **20/20 damage-matched** (tol 15%) and **20/20 reproduced the GC
shift** (GC 0.3505 ± 0.0070 vs ablated 0.3455, intact 0.4592). The GC phenotype is **generic
damage-linked**, not learned-direction-specific. Supports manuscript claim 11.

### EXP10 — DNABERT pair mechanism [MEASURED — falsifies the compensation label]
| measurement | value |
|---|---|
| cosine(r264, r294) | **−0.1876** (random same-layer 0.0008 ± 0.0538, z = 3.50) |
| per-token activation correlation | **+0.9986** (n = 20,305) |
| r264 after ablating r294 | **×1.0000** |
| r294 after ablating r264 | **×1.0000** |
| accuracy base / −A / −B / −AB | 0.9288 / 0.9285 / 0.9277 / **0.5912** |

Interaction robust; **not compensatory** (no survivor upregulation) and **not redundant
duplication** (significantly anti-aligned). Joint effect is therefore **downstream/convergent**
[INFERRED]. No stronger label asserted, per the spec's rule.

---

## 3. Priority-3 items (all complete)

- **EXP12 family-clustered** [MEASURED]: 12 families, 2000 resamples. activation ratio +0.766
  [+0.629,+0.881], P(ρ>0)=1.00; q1 and Frobenius unstable across cohort definitions.
- **EXP11 extreme-effect audit** [MEASURED]: top-3 positives are SmolLM2-1.7B **+6.989**
  (baseline 2.6145), ModernBERT-base +3.570 (1.5241), SmolLM2-135M +3.134 (3.3001) — **all
  baselines in normal range, so the ~+700% result is real, not a tiny-denominator artifact.**
  Negatives are NTv3 −0.008 and EuroBERT-2.1B −0.0001, both ≈ 0.
- **N threshold sensitivity** [MEASURED]: all 22 models exceed even >10 (min median ratio 10.42,
  max 7124). The >5 criterion is **inert in this panel** — keep as diagnostic, not selector.
- **O input stability** [MEASURED]: rank 1 on **528/528**; max at position 0 in 61.2%.
  *Figure not yet drawn.*
- **EXP9 structurally matched replacements** [MEASURED]: at matched norm **all six families**
  are as damaging as deletion (0.97–1.02× ablation). Damage is **non-monotonic in scale**: the
  sign-flipped row at 2× norm avoids **85%** of damage (ΔNLL +0.348 vs +2.292) at cosine
  −0.033. With EXP8 this implies the mechanism needs **sufficient magnitude with roughly the
  right magnitude profile**, not the learned direction — the paper's title claim, strengthened.

---

## 4. EXP3 — special-token dependence [MEASURED — architecture dissociation]

| model | row | default ratio | no-special ratio | collapse | max at special token |
|---|---|---|---|---|---|
| GENERator-EUK | L4/r2371 | 7124.1 | **7089.0** | **1.00×** | default only |
| GENERator-PROK | L8/r260 | 257.3 | **257.1** | **1.00×** | default only |
| DNABERT-2 | L9/r264 | 238.6 | **1.9** | **126×** | always |
| DNABERT-2 | L9/r294 | 228.0 | **9.1** | **25×** | always |
| DNABERT-2 | L5/r603 | 65.2 | **2.2** | **30×** | always |
| DNABERT-2 | L3/r86 | 61.3 | **1.5** | **41×** | always |

Decoders indifferent; encoder entirely dependent, with the maximum on a special token in 100%
of default measurements at median positions 12–17 (SEP-like, **not** CLS/position 0).
**Additionally** [MEASURED]: special-token handling changes the **argmax coordinate itself**
(DNABERT-2 L8/r603 → L9/r264), so a single convention silently answers a different question per
architecture. NTv3 not measured (device-placement error under the shared wrapper).

---

## 5. Experiment the spec did not request, and why it was necessary

### Within-model graded sweep [MEASURED]
**Motivation:** every specified analysis is either (a) between-model, n=22, **one row per
model**, so each point carries its whole model as a confound; or (b) within-model but
**binary** — candidate vs 5 random same-layer controls, which sit at ratio ≈ 1 *by construction*
(ratio = max / median same-layer max). Neither design can exhibit a ratio→damage slope.

36 rows inside GENERator-EUK layer 4, ranks log-spaced, selection by ratio only:

| | ρ | p | 95% CI |
|---|---|---|---|
| all 36 rows | **+0.454** | 0.0054 | [+0.077,+0.760] |
| excluding frozen candidate | **+0.406** | 0.0156 | — |

Slope survives removal of the extremum. But shape is **threshold-like**: ratio 7124 → +0.368,
2731 → +0.107, **531 → −0.0001**, and only **10/36** rows exceed |damage| 1e-4.

> **Wording consequence:** "activation extremeness **calibrates** severity" is unsupported.
> Supported: it **orders** rows correctly and **identifies the small set that matters**.

**Also relevant for the overseer:** the within-model enrichment evidence that *does* exist is
strong — G(ε=1.0) > 0 in **20/22** models, Wilcoxon **p = 0.00003** — but it is candidate-vs-
control, not a slope. **Limit:** one model, one layer, and it is genomic; a text-decoder
replication (SmolLM2-1.7B, +6.99) needs a download.

---

## 6. Errors caught in-session (for calibration of everything above)

1. **EXP5 tokenisation invalid on first run.** The tokenizer has `add_bos_token=True`, so it
   auto-prepended BOS to all four conditions (`[1,1,754…]` for "standard" — a double BOS),
   making every condition identical at position 0 and producing a spurious uniform 375,361.
   Fixed with `add_special_tokens=False`; the corrected run **reverses** the conclusion. The
   frozen harness already used `add_special_tokens=False`, so it was unaffected.
2. **EXP3 hooked the wrong module.** `mods.get(key.rsplit(".",1)[0])` preferred the parent
   `BertGatedLinearUnitMLP` over `.mlp.wo`, understating DNABERT-2 activations ~40× (L9/r264
   read 5.8 instead of 238.6). Caught by disagreement with an independent inline run. Also,
   EXP3 had **crashed before `json.dump`**, so its numbers existed only in a log until re-run.
   Both fixed; persisted values match the originally reported ones.
3. **Two portability fixes, neither touching frozen config:** hg38 symlink so the frozen path
   resolves unmodified; corrected `(chrom,start,seq)` window unpacking.

Neither (1) nor (2) affected any frozen experiment. Both affected only new analyses and were
found by cross-checking, not by the pipeline — which argues for adding an assertion that hooked
module types match the expected class.

---

## 7. Manuscript claim classification (spec §Q)

| # | claim | verdict | evidence |
|---|---|---|---|
| 1 | high-gain rows recur across text and genomic transformers | **SUPPORTED** | 22 models, rank 1 on 528/528 · `input_stability.tsv` |
| 2 | activation selection enriches for consequential rows | **SUPPORTED** | G>0 in 20/22, Wilcoxon p=3e-5 · `census_master.csv` |
| 3 | activation extremeness predicts effect magnitude | **SUPPORTED WITH QUALIFICATION** | ρ=+0.766 between-model; within-model ordering ρ=+0.454 but threshold-like, not linear · `activation_vs_causality.tsv`, `within_model_slope.tsv` |
| 4 | q1 predicts effect magnitude | **NOT SUPPORTED** | +0.074, p=0.744; sign-unstable across cohorts |
| 5 | Frobenius predicts effect magnitude | **NOT SUPPORTED** | +0.441 pooled but −0.042 in text-decoders; significant in no subgroup |
| 6 | L4/r2371 mediated primarily through BOS | **SUPPORTED** | rescue 0.9996 at p=0 · `generator_position_rescue.tsv` |
| 7 | BOS-specific, not generic early-position geometry | **SUPPORTED** | 90× drop p=0→1 vs 1.2% accessibility change |
| 8 | effect follows BOS identity rather than position 0 | **SUPPORTED** (was conflated) | shifted_bos: argmax@BOS=1.00, argmax@0=0.00; 8,228× gap · `generator_bos_identity_vs_position.tsv` |
| 9 | BOS is an attention sink after causal-mask normalisation | **SUPPORTED** | fold 7.90×, 82.9% of layer-heads >2× · `generator_attention_sink.tsv` |
| 10 | L4/r2371 causally **supports** the sink | **NOT SUPPORTED — direction reversed** | ablation *increases* fold 7.90→10.24 (+30%) |
| 11 | GC phenotype is generic damage, not direction-specific control | **SUPPORTED** | 20/20 damage-matched random directions reproduce it · `generator_random_direction_replicates.tsv` |
| 12 | L9/r264 × L9/r294 exhibit robust causal interaction | **SUPPORTED** | 0.9288 → 0.5912 jointly, ≈0 singly |
| 13 | the interaction is compensatory/redundant | **FALSIFIED** | cross-ablation ×1.0000; anti-aligned cos −0.1876 (z=3.50) yet r=+0.9986 · `dnabert_pair_mechanism.tsv` |
| 14 | structural geometry alone determines mechanism | **NOT SUPPORTED** | all matched replacements as damaging as deletion; sign-flipped at 2× norm avoids 85% of damage · `generator_matched_replacements.tsv` |

---

## 8. What the overseer should decide

**Blocking one number:** EXP2 for Llama-7B / Mistral-7B / OLMo-7B. Download-bound (~40 GB,
2.5–4 h). It is the only outstanding item that can change a value in the abstract.

**Recommended additions, in value order:**
1. **Within-model sweep on a text decoder** (SmolLM2-1.7B). Currently n=1 model and it is
   genomic; this is the design that tests the calibration claim in the form the paper wants.
   I rate this **above** finishing EXP2, because EXP2 fixes the sampling frame of a correlation
   that stays model-confounded either way.
2. EXP2 completion (above).
3. NTv3 in EXP3 (device-placement fix).
4. Section O figure (data exists, plotting only).

**Framing questions the data now forces:**
- Should non-decoder cohorts be reported as correlations at all? In encoders 4/6 and in genomic
  5/6 candidates have effects ≈ 0, so ρ there ranks near-ties.
- Is the row-level ("superrow") unit distinct enough from Yu's scalar unit to claim as new? I
  cannot verify 2026 literature from this session (no web access, knowledge cutoff May 2026).
  Note the compression theorem concerns the row's max **element**, i.e. Yu's scalar — the two
  units interlock and each usage should be explicit.
- Claim 10's reversal is mechanistically interesting (a high-gain row that *dampens* a sink) and
  may deserve promotion rather than mere correction.
