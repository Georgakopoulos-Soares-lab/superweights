# FIGURE_PROVENANCE.md — panel-by-panel provenance manifest (v3)

Governing figure specification: `experiments/figures/_figstyle.py` (repo root) — the only
codified figure-style module in this repository, applied via `apply_style()` /
`panel_label()` in every script below. See the final report for the full rules trace.
Shared model-class visual encoding: `experiments/figures/_paper_encoding.py`.

**Every plotted number in every panel below is loaded programmatically from a raw JSON
file at render time — none is transcribed from prose or hardcoded from memory.** Where a
narratively-relevant number has no raw artifact, the panel is *not rendered* (see
"Not producible" table at the end) rather than approximated from `.md` prose.

All raw JSON paths are relative to the repository root
`/work/11034/atzanakak/glm_super_weight/genomic-super-weights/`.

---

## Figure 1 — Cross-domain structural organization (`figures/main/fig1_structural.{png,pdf}`)

Script: `figures/fig1_structural.py`. Source data snapshot: `figures/source_data/fig1_structural.json`.

| Panel | Plotted variable | Artifact | Keys | Transformation | Claim supported | Caveat |
|---|---|---|---|---|---|---|
| A | Published-SW row rank, scalar top-1 share | `results/e1_nlp_retrospective.json` | `.models[].{level1.row_rank, level1.percentile, level2.top1_share}` | none (direct read) | Cold-weight predictor recovers published NLP super-weights (calibration only) | Diagonal-approximation metric (`uk_frobenius.py`), numerically distinct from the exact metric in B/C (confirmed: Llama-7B `uk_norm=123.64` vs exact `frob_norm=137.62` for the same row) |
| B | Exact `q1` per model | `results/experiments/E7/e7_legacy_reanalysis.json` (Mistral/Llama/GENERator EUK/OLMo/DNABERT-2/NTv3); `results/experiments/E7/e7_phase1_detection_qwen25.json` (Qwen2.5); `results/experiments/E7/e7_phase1_detection_genomeocean.json` (GenomeOcean); `results/experiments/E7/e7_phi3_spectral.json` (Phi-3, model-level median); `results/e8_detection_{mosaicbert,modernbert}.json` (encoders); `results/experiments/E7/e7_phase1_detection_evo2.json` (Evo2, explicit null) | `.q1` / `.spectral.q1` / `.model_level.q1_median` | none | q1 (exact spectral concentration) organizes more by architecture than domain, descriptively | q1 is NOT a magnitude or causal-importance axis (enforced by never plotting it against effect size in this or any other figure) |
| C | Exact `‖U_k‖_F` (frob_norm), log scale | same files as B, `.frob_norm` / `.spectral.frob_norm` | none | Magnitude kept visually distinct from q1 | Raw values span 0.39–836 and are explicitly NOT cross-model comparable (different weight scales); caption states this |
| D | MosaicBERT/ModernBERT candidate q1 vs. 5 same-layer control rows | `results/e8_detection_{mosaicbert,modernbert}.json` | `.spectral.q1`, `.control_rows[].q1` | none | Detected high-gain row is far more concentrated than ordinary same-layer rows | Only measured for these 2 encoders; no same-layer controls exist in this repo for DNABERT-2/NTv3 |

**Verified during audit:** E8's `control_rows` prose numbers in `RESULTS.md` match the raw JSON exactly (no discrepancy). Evo2-7B has zero measured q1/PR_spec/frob_norm (Phase-1 detection null, ratio 2.22 < 5.0× threshold) — rendered as an explicit "structural null" marker in panel B, never omitted or imputed.

---

## Figure 2 — Cross-model causal organization (`figures/main/fig2_causal.{png,pdf}`)

Script: `figures/fig2_causal.py`. Source data snapshot: `figures/source_data/fig2_causal.json`.

| Panel | Plotted variable | Artifact | Keys | Transformation | Claim supported | Caveat |
|---|---|---|---|---|---|---|
| A | Per-row `\|relative_pct_change\|` at full ablation, all 5 decoders; control median | `results/experiments/E10/e10_decoder_concentration.json` | `.models.{llama,mistral,olmo,phi3,qwen25}.{decision,K,target_rows[].relative_pct_change,control_median_abs_delta,baseline_nll}` | `control_median_abs_delta / baseline_nll * 100` for the control line | Decoder singleton causal spectra; 4/5 SINGLE_COMPONENT_DOMINANT, Phi-3 MULTI_COMPONENT_CANDIDATE | Llama/Mistral/Qwen are K=1 by construction (no candidate set exists) — plotted as single points, never implying an unmeasured spread |
| B | OLMo structural rank (exact ‖U_k‖_F) vs. causal rank (\|%NLL change\|) | `results/experiments/E10/e10_exact_uknorm_olmo.json` (`.layers.{L}.layer_max_uk`) + `results/experiments/E10/e10_decoder_concentration.json` (`.models.olmo.target_rows[]`) | see above | rank ordering computed at plot time from raw values | L24/r269 structurally largest (53.4× layer median) but causally weakest (+1.4%); L1/r269 causally dominant (+111.8%) despite ranking 3rd structurally | Both structural and causal numbers independently confirmed against raw JSON during audit |
| C | F2→F3 held-out MAE relative improvement (%) + bootstrap 95% CI, both epsilons, MosaicBERT/ModernBERT (encoders) **and Phi-3 (decoder, E10b)** | `results/experiments/E10/e10_encoder_fit_results.json`; `results/experiments/E10/e10b_phi3_fit_results.json` | `.models.{mosaicbert,modernbert}.by_epsilon.{0.5,1.0}.{F2_vs_F3_relative_MAE_improvement, F2_vs_F3_bootstrap.{ci_2.5,ci_97.5}, F2.metrics.mae}`; `.by_epsilon.{0.5,1.0}.{same keys}` | CI bounds converted from absolute MAE-difference scale to % via division by F2 MAE | Both encoders require pair terms at both scales (bootstrap CI excludes zero, positive). Phi-3 (E10b) **splits by epsilon**: +36.4% at eps=0.5 (CI excludes zero, positive — PAIR_TERMS_REQUIRED) but −2.5% at eps=1.0 (CI excludes zero, negative — pairwise model reliably *worse*, MIXED_OR_UNRESOLVED). E10B_SYNTHESIS.md diagnoses this as a three-way redundancy break among Phi-3's 3 layer-2 rows (`L2/r525`, `L2/r1693`, `L2/r1113`) — visible independently in raw per-mask responses and in the fitted pair coefficients, not chased with a higher-order model | Statistics unit = MODEL (n=2 encoders + 1 decoder); not presented as independent mask-level replicates. Phi-3 is n=1 among the 5 E10 decoders — does not establish decoders in general require pair terms |

**E10b status (updated 2026-08-22): now frozen.** Commits `1128fea`…`fafd44d` on the same integration branch; raw artifacts `results/experiments/E10/e10b_phi3_fit_results.json` (F0–F3 coefficients/metrics/bootstrap/retrospective pairs) and `results/experiments/E10/e10b_phi3_tomography_responses.json` (raw per-mask, per-batch responses, both epsilons). Governing doc: `experiments/E10b_phi3_tomography/E10B_SYNTHESIS.md`.

**Data-integrity note (not plotted, recorded for the record):** `ARCHITECTURE_SYNTHESIS.md`'s retrospective same-layer-pair-clustering prose ("20-40%... vs 33%"; "40-50%... vs 64%") mislabels `frac_same_output_row_*` as same-layer fractions; the true `frac_same_layer_*` values are lower (MosaicBERT 0.20/0.089, ModernBERT 0.10/0.044). The qualitative conclusion is unaffected; flagged so it is not silently propagated into manuscript text.

---

## Figure 3 — DNABERT-2 mechanistic organization (`figures/main/fig3_dnabert.{png,pdf}`)

Script: `figures/fig3_dnabert.py`. Source data snapshot: `figures/source_data/fig3_dnabert.json`.
All four panels are backed by E9's own from-scratch 2026-08-22 measurement — none by the
unreproduced colleague-branch narrative.

| Panel | Plotted variable | Artifact | Keys | Transformation | Claim supported | Caveat |
|---|---|---|---|---|---|---|
| A | Individual vs. joint pretrained-MLM Δloss for the critical pair (L9/r264, L9/r294), epsilon=1.0 | `experiments/E9_mechanistic_tomography/baseline_regression_results.json` | `.dnabert2.epsilon_1p0.{d_A,d_B,d_AB,epistasis}` | none | Individual channel effects small; direct joint (2-channel) ablation much larger than either alone | This is a **directly measured** 2-channel intervention (not a fitted prediction) on pretrained MLM loss — a different endpoint/unit than the colleague's unreproduced splice-accuracy pair-ablation number (C-036), stated explicitly in the caption |
| B | Epistasis (`d_AB - d_A - d_B`) at both epsilons | same file | `.dnabert2.{epsilon_0p5,epsilon_1p0}.epistasis` | none | Super-additivity present at both intervention scales — not an artifact of one ablation strength | Independently reproduces the colleague-reported pretraining epistasis figure (+2.0118) to 4 decimals, from a fresh environment — genuinely stronger evidence than "adopted report," stated in caption |
| C | F0–F3 held-out predicted-vs-actual Δloss, both epsilons | `experiments/E9_mechanistic_tomography/{fit_results_dnabert2.json,dnabert2_mask_responses.json}` | `.by_epsilon.{eps}.{singleton_effects_x_i,F1.gain_g,F2.beta,F3.{beta_main,gamma_pairs}}`, held-out pool `a` vectors | exact prediction arithmetic from `run_fit_observers.py` (F0=a·x, F1=g·F0, F2=a·beta, F3=lifted(a)·coef) reproduced inline | F0/F1/F2 fail held-out adequacy; F3 materially improves held-out MAE (54.7%/24.4%), bootstrap CI excludes zero | Declared residual-channel basis — not proof of intrinsic causal dimension 2 (E9 RESULTS.md §18) |
| D | F3 pairwise `Γ` coefficients ranked, critical pair highlighted, both epsilons | same fit_results file | `.by_epsilon.{eps}.F3.{gamma_pairs, critical_pair_rank_by_abs_gamma, critical_pair_gamma}` | top-12 by \|Γ\| | Known critical pair ranks #1/45 without special treatment during fitting | Retrospective validation, not a basis-selection procedure (stated in caption) |

**Panels NOT rendered — UPDATED 2026-08-22, status changed from "no artifact" to "contested":**

A same-day reproduction pass produced real raw artifacts for the codominance and
splice-accuracy pair-ablation claims. Both **conflict materially** with the previously-adopted
colleague-report magnitudes:

| Claim | Previously adopted (prose only, no artifact) | Now measured (real artifact, single seed) | New artifact |
|---|---:|---:|---|
| Critical pair (L9/r264+r294) joint splice-accuracy effect | −33.76 pp | **−0.42 pp** | `results/mechanism/pairwise_epistasis_dnabert2_reconstructed.json` |
| Strongest epistatic pair | (asserted to be the named pair) | **L3/r86 + L3/r399, −2.10 pp** (named pair ranks lower, −0.53 pp) | same file |
| Codominance ratio sweep (co-dominant → imbalanced) | epistasis −33.63 → −0.66 pp | epistasis **−0.53 → −0.04 pp** (same direction, ~60× smaller) | `results/mechanism/codominance_break_dnabert2.json` |

Both new measurements come from `scripts/evaluation/run_sw_pairwise_epistasis.py` and
`scripts/mechanism/run_codominance.py` respectively, run 2026-08-22 on this repo's own GPU
after fixing two hardcoded absent-path bugs (GUE data root; see script diffs) — real
forward-pass measurements, not transcribed or estimated. Both used only **1 seed** (the only
fine-tuned DNABERT-2 splice checkpoint present in this repository), versus the n=5 protocol
the old numbers claim, so this is not a fully powered replication either way.

**Per explicit instruction, neither the old nor the new number is plotted in any figure or
presented as established.** C-036 and C-038 are downgraded to **contested** — a real,
code-verified artifact now exists for both and disagrees with the adopted prose by roughly
two orders of magnitude; this is disclosed as an open discrepancy for author adjudication,
not resolved here in either direction. Figure 3 remains centered on panels A-D (all on the
pretrained-MLM endpoint, unaffected by this dispute).

**Still not rendered (no artifact of any kind):**
- **NTv3 falsification (corrected, C-029):** the only NTv3-splice JSON in this repo (`results/experiments/gue/gue_multiseed_ntv3_splice.json`) is the *retired* truncation-bug result (X-009). It is **not shown anywhere in this figure set** (the transparency-only supplement panel that previously displayed it, labeled RETIRED, has been removed per explicit instruction — an invalid result is not plotted even for transparency). The corrected replacement number has no artifact anywhere. NTv3's MAKE-PAIR construction likewise has no artifact.

---

## Figure 4 — GENERator functional realization (`figures/main/fig4_generator.{png,pdf}`)

Script: `figures/fig4_generator.py`. Source data snapshot: `figures/source_data/fig4_generator.json`.

| Panel | Plotted variable | Artifact | Keys | Transformation | Claim supported | Caveat |
|---|---|---|---|---|---|---|
| A | GENERator EUK row 2371 exact q1 | `results/experiments/E7/e7_legacy_reanalysis.json` | `["GENERator EUK"].q1` | none | Minimal structural callout, cross-referencing Fig. 1B | Deliberately minimal — no activation-lifecycle artifact exists in this repo for GENERator |
| B | BOS-centered attention/activation phenotype (C-039) | `results/mechanism/attention_sink_implicit_bias.json` — **UPDATED 2026-08-22: now a real artifact**, produced by `scripts/mechanism/run_attention_sink.py` after fixing a hardcoded hg38 path absent on this filesystem (no science changed, only an environment fix, documented inline in the script) | `.generator.attention.{sink_share_pos0,uniform_expectation,sink_over_uniform,frac_heads_argmax_pos0}`, `.generator.implicit_bias.pos0_real_mean` | none | 37.9% of incoming attention mass at position 0 (33.0× uniform), 78.75% of (layer,head) pairs argmax at position 0, activation maximum also at position 0 — **closely reproduces** the previously-cited colleague-report numbers (37.96%, 33.0×, 78.3%) | **Strictly associative wording only** — "co-occurs with," never "causes." Single run (seed 42, 40 windows), not yet seed-replicated. Unlike the splice/codominance reproduction below, this one is *consistent with*, not contradicting, the prior claim. |
| C | Generated GC fraction vs. row scale alpha, rows 2371/1522/random control | `experiments/E9_mechanistic_tomography/baseline_regression_results.json` | `.generator.results.{primary_2371,secondary_1522,random_control}[alpha].{gc_mean,gc_sd,n}`, `.generator.{gc_span_primary,gc_span_random}` | SEM = sd/sqrt(n) | Row 2371 causally moves GC composition far more than random control (~390×); row 1522 non-monotonic | E9's own measurement (C-045) — explicitly **not** the colleague's unreproduced "38.6×"/"17.9×" figure (C-040), which used a different, never-reproduced 5-point alpha grid (stated in caption). "Causal sensitivity," not "validated graded steering" (only 3 dose points exist by design) |

**Panels NOT rendered (no raw artifact of any kind):**
- **Sequence-quality/specificity over the intervention range:** confirmed entirely absent — not even as prose numbers. `PROVENANCE_AND_BASELINES.md` states outright that no raw output exists for the designed quality-metric script, and E9's own `baseline_regression_results.json` measures only `gc_mean`/`gc_sd`/`n`.

---

## Supplement S1 (`figures/supplement/fig_s1_structural_detail.{png,pdf}`)

Script: `figures/supplement/fig_s1_structural_detail.py`.

| Panel | Plotted variable | Artifact | Keys |
|---|---|---|---|
| a | Phi-3's 6 individual rows' exact q1 | `results/experiments/E7/e7_phi3_spectral.json` | `.rows[].{layer,row,q1}`, `.model_level.q1_median` |
| b | All 5 decoders' same-layer control-row causal effects | `results/experiments/E10/e10_decoder_concentration.json` | `.models.{*}.controls[].relative_pct_change` |

## Supplement S2 — REMOVED

Previously showed the retired (X-009) NTv3 splice-ablation result, explicitly labeled
RETIRED, for methodological-history transparency. **Removed 2026-08-22 per explicit
instruction**: an invalid (truncation-bug-affected) result is not plotted in the figure set
even labeled as retired. `results/experiments/gue/gue_multiseed_ntv3_splice.json` remains the artifact of
record for X-009 in prose (`CLAIMS_LEDGER.md`); it is no longer rendered as a figure.

---

## Summary: panels not producible or contested (updated 2026-08-22)

| Panel (candidate location) | Claim | Status | Artifact |
|---|---|---|---|
| Fig 3 — DNABERT-2 splice-accuracy pair ablation | C-036 | **Contested** (real artifact conflicts with adopted prose by ~80×) | `results/mechanism/pairwise_epistasis_dnabert2_reconstructed.json` |
| Fig 3 — Codominance mechanism | C-038 | **Contested** (real artifact conflicts with adopted prose by ~60×) | `results/mechanism/codominance_break_dnabert2.json` |
| Fig 3 — NTv3 falsification (corrected) | C-029 (corrected) | No artifact | none (retired X-009 artifact no longer shown anywhere, see Supplement S2 removal) |
| Fig 4 — GENERator sequence quality/specificity | (unclaimed, design note only) | No artifact | none |

**Resolved (no longer in this table):** Fig 4B GENERator BOS/attention-sink (C-039) — now
real, reproduced, plotted (see Figure 4 section above); Fig 2 E10b Phi-3 tomography — now
real, integrated into Figure 2 panel C.

Authoritative absence record for the colleague-branch items still missing: `experiments/docs/MISSING_COLLEAGUE_ARTIFACTS.md`.
