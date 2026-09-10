# FIGURE_PLAN_v3.md — main + supplementary figure plan (FINALIZED, rev. 2026-08-22b)

**Status: FINAL, updated after a second provenance pass.** A first pass found that a large
fraction of the DNABERT-2/GENERator "mechanism-session" narrative had zero raw JSON/CSV
artifact anywhere in this repository (only `.md` prose, adopted as fact by author decision
D-024 with producing scripts that never wrote their declared output). Three of those gaps
were then closed by fresh, real, code-verified reproduction runs on this repo's own GPU
(2026-08-22, same day):

- **GENERator BOS/attention-sink (C-039): reproduced cleanly, matches the previously-cited
  numbers closely.** Now a real Figure 4 panel (B).
- **DNABERT-2 splice-accuracy pair ablation (C-036) and codominance ratio sweep (C-038):
  reproduced, but the real numbers materially CONFLICT with the previously-adopted
  magnitudes** (roughly 60-80x smaller; the old -33.76pp splice number is, on its face,
  larger than this repo's own established full-10-row-ensemble effect, -26.84pp). Per
  explicit instruction, **neither the old nor the new number is plotted or presented as
  established** — both are flagged CONTESTED in `FIGURE_PROVENANCE.md`, and Figure 3 stays
  centered on its four pretrained-MLM-endpoint panels, none of which are affected by this
  dispute.
- **E10b (Phi-3 pair tomography): completed and integrated into Figure 2** (see Figure 2
  section) — result is `MIXED_OR_UNRESOLVED`, reported as split by epsilon, not forced into
  one label.

GENERator's sequence-quality/specificity claim and NTv3's corrected falsification number
(C-029) remain entirely without any artifact and are still not producible. The retired NTv3
splice supplement (previously shown for transparency) has been **removed** per instruction —
an invalid result is not plotted even labeled as retired.

Governing figure specification: `experiments/figures/_figstyle.py` (see
`FIGURE_PROVENANCE.md` for the full rules trace — this is the only codified,
repo-wide figure style module found in this repository; there is no separate
prose "figure spec" document). Shared model-class visual encoding:
`figures/_paper_encoding.py`.

Figure logic (per governing prompt): (1) structural organization -> (2) cross-model
causal organization -> (3) DNABERT-2 mechanistic deep dive + falsification ->
(4) GENERator functional/biological deep dive.

Placeholder rows marked `[PENDING AUDIT]` will be filled in once the three
background provenance-audit passes (E7/E8, E10/E10b, GENERator/NTv3/colleague-prose)
return. Nothing in this file is final until those markers are gone.

---

## Figure 1 — Cross-domain structural organization of high-gain FFN pathways

Claim: high-gain FFN structures occur across text and genomic models, but their exact
geometry (q1, PR_spec) is organized more by encoder/decoder architecture than by
text-vs-genomic domain — descriptively, not as a clean binary split (NTv3 outlier,
DNABERT-2 intermediate, ModernBERT thin margin, GENERator EUK inside the NLP range).

| Panel | Question | Claim | Source artifact(s) | Script | Output | Main/Supp | Caveat |
|---|---|---|---|---|---|---|---|
| A | Does the cold-weight predictor recover published NLP super-weights? | Calibration only: row rank 1/N and scalar top1-share 0.89-0.99 in Llama/Mistral/OLMo | `results/e1_nlp_retrospective.json` (`.models[].{level1,level2}`) | `fig1_structural.py::panel_a` | `figures/main/fig1_structural.{pdf,png}` | Main | Diagonal-approx metric (`level1.uk_norm`, `level2.participation_ratio`, computed by `uk_frobenius.py`'s diagonal `c_{k,i}` decomposition), **numerically distinct from** the exact `frob_norm`/`q1`/`pr_spec` used in B/C (confirmed non-interchangeable: Llama-7B `uk_norm=123.64` vs exact `frob_norm=137.62` for the identical row). E5 showed the diagonal approximation is acceptable for NLP models specifically (cross-term fraction 6.7-19.3%), not for genomic ones. Never plot on the same axis as B/C. |
| B | How rank-1/concentrated is each model's exact high-gain operator? | q1 (exact) across the combined E7+E8 panel (11 measured + 1 structural-null), colored by domain, shaped by architecture | `results/e7_legacy_reanalysis.json`, `e7_phase1_detection_{qwen25,genomeocean,evo2}.json`, `e7_phi3_spectral.json`, `e8_detection_{mosaicbert,modernbert}.json` — full table below | `fig1_structural.py::panel_b` | same file | Main | Never use q1 as a magnitude/selection axis. No causal meaning. Evo2-7B has **no q1** (Phase-1 detection null, ratio 2.22<5.0 threshold) — render as an explicit null marker, never omit or impute. |
| C | Separately, how large is the high-gain operator? | `\|\|U_k\|\|_F` (exact `frob_norm`), log scale, kept visually distinct from q1 | same files as B, `.frob_norm` / `.spectral.frob_norm` keys | `fig1_structural.py::panel_c` | same file | Main | Raw `frob_norm` spans 0.39-836 across models and is **not cross-model comparable in absolute terms** (different weight scales/parameterizations/dimensions) — caption must state this explicitly; the panel exists to keep magnitude visually distinct from q1, not to rank "how strong" each model's pathway is. Phi-3 has no model-level frob_norm (per-row only, 6 values). |
| D | Is the encoder high-gain row still unusual relative to ordinary rows at the same layer? | MosaicBERT candidate q1=0.4766 vs. 5 controls 0.035-0.047 (PR_spec 4.12 vs 140-170); ModernBERT candidate q1=0.8970 vs. 5 controls 0.024-0.049 (PR_spec 1.23 vs 90-121) | `results/e8_detection_{mosaicbert,modernbert}.json` (`.control_rows[]`, verified exactly against RESULTS.md prose, no mismatch) | `fig1_structural.py::panel_d` | same file | Main | Only measured for these 2 encoder models — no same-layer controls exist in this repo for DNABERT-2 or NTv3 (confirmed absent); do not imply otherwise. |

**Panel B/C confirmed model table** (q1 / PR_spec / frob_norm, domain, architecture, exact source):

| Model | Domain | Arch | q1 | PR_spec | frob_norm | Source |
|---|---|---|---:|---:|---:|---|
| Mistral-7B | text | decoder | 0.9922 | 1.0158 | 0.388 | `e7_legacy_reanalysis.json["Mistral-7B"]` |
| Llama-7B | text | decoder | 0.9888 | 1.0227 | 137.62 | `e7_legacy_reanalysis.json["Llama-7B"]` |
| GENERator EUK | genomic | decoder | 0.9689 | 1.0653 | 522.13 | `e7_legacy_reanalysis.json["GENERator EUK"]` |
| OLMo-7B | text | decoder | 0.9646 | 1.0747 | 0.911 | `e7_legacy_reanalysis.json["OLMo-7B"]` |
| Qwen2.5-7B | text | decoder | 0.9529 | 1.0995 | 29.69 | `e7_phase1_detection_qwen25.json.spectral` |
| Phi-3-mini (median, 6 rows) | text | decoder | 0.9028 | 1.2249 | n/a (per-row only) | `e7_phi3_spectral.json.model_level` |
| GenomeOcean-4B | genomic | decoder | 0.8989 | 1.2243 | 2.883 | `e7_phase1_detection_genomeocean.json.spectral` |
| DNABERT-2 | genomic | encoder | 0.7933 | 1.5033 | 74.01 | `e7_legacy_reanalysis.json["DNABERT-2"]` |
| ModernBERT | text | encoder | 0.8970 | 1.2337 | 836.05 | `e8_detection_modernbert.json.spectral` |
| MosaicBERT | text | encoder | 0.4766 | 4.1159 | 20.00 | `e8_detection_mosaicbert.json.spectral` |
| NTv3 | genomic | encoder | 0.3889 | 6.4803 | 441.52 | `e7_legacy_reanalysis.json["NTv3"]` |
| Evo2-7B | genomic | decoder | **NULL** (detection null) | NULL | NULL | `e7_phase1_detection_evo2.json` (`.null_reason`, ratio 2.22<5.0) |

Domain/architecture labels for MosaicBERT/ModernBERT and FFN-type for every model are
**prose-only** (no JSON key) — sourced from `E8_encoder_decoder/MODEL_AUDIT.md` and
`E7_exact_dimensionality/MODEL_PANEL.md`, verified from actual model source code. This is
fine to hardcode into the plotting script's model-metadata table (already done in
`_paper_encoding.py::MODEL_CLASS`) since it is a verified architectural fact, not a
measurement in need of its own provenance row.

Do NOT imply: perfect encoder/decoder separation; a genomic/text separation; q1 as gain
magnitude; q1 as a causal-importance predictor.

---

## Figure 2 — Cross-model causal organization

Claim: in the tested panel, decoder high-gain systems tend toward single-component causal
concentration (4/5 decoders: Llama, Mistral, OLMo, Qwen2.5), while both tested encoders
(MosaicBERT, ModernBERT) require pairwise interaction terms — but this is not universal
(Phi-3 is a decoder counterexample, flagged not resolved) and structural ranking does not
reliably predict which component dominates causally (OLMo: structural top L24/r269 vs.
causal top L1/r269).

Governing prereg: `docs/prereg/PREREG_E10_nlp_architecture_causal_v2.md` (decision rule:
SINGLE_COMPONENT_DOMINANT iff C1>0.5 AND top1 control-normalized effect >3.0;
MULTI_COMPONENT_CANDIDATE if top fails but >=2 rows individually clear >3.0).
Endpoint: causal-LM NLL, WikiText-2-raw-v1 test split, N=100 windows, 512 tokens,
`alpha = 1 - epsilon*a` row-scale intervention on `mlp.down_proj` (`ENDPOINTS.md`,
`DECODER_INTERVENTION_FREEZE.md`).

| Panel | Question | Claim | Source artifact(s) | Script | Output | Main/Supp | Caveat |
|---|---|---|---|---|---|---|---|
| A | Decoder singleton causal spectrum, all 5 models | Llama/Mistral/Qwen (K=1) single dominant row; OLMo (K=4) C1=0.608 dominant but NOT on the structurally-largest row; Phi-3 (K=6) C1=0.333, distributed, MULTI_COMPONENT_CANDIDATE | `results/e10_decoder_concentration.json` (`.models.{llama,mistral,olmo,phi3,qwen25}.{C1,C2,decision,target_rows[]}`) | `fig2_causal.py::panel_a` | `figures/main/fig2_causal.{pdf,png}` | Main | Do NOT visually imply multi-row concentration was measured for Llama/Mistral/Qwen (K=1 by construction, no candidate set exists). Mistral is non-monotonic (alpha=0.5 dNLL 6.652 EXCEEDS full-ablation 6.338) — must be visible, not smoothed. |
| B | OLMo dissociation detail | L24/r269 structurally largest (1.6369, 53.4x layer median) but weakest causal effect (dNLL=0.0347, +1.4%, rank 4/4); L1/r269 causally dominant (dNLL=2.789, +111.8%) despite ranking 3rd structurally (0.911, 39.2x) | `results/e10_exact_uknorm_olmo.json` (structural) + `results/e10_decoder_concentration.json.models.olmo.target_rows[]` (causal) — both independently confirmed | `fig2_causal.py::panel_b` | same file | Main | Single most important structure-vs-causality dissociation in the whole figure set — plotted as a paired dumbbell/rank-swap, not folded into panel A's aggregate view. |
| C | Encoder side: is the additive description adequate? | MosaicBERT F2→F3 held-out MAE improves 84.2%/54.5% (eps 0.5/1.0); ModernBERT 58.8%/46.4%; bootstrap CI excludes zero at all 4 (model,epsilon) cells — replicates DNABERT-2's (E9, Fig 3C) interactional phenotype in 2 independent text encoders | `results/e10_encoder_fit_results.json` (`.models.{mosaicbert,modernbert}.by_epsilon.{0.5,1.0}`) — all figures verified exactly against raw JSON | `fig2_causal.py::panel_c` | same file | Main | Statistics unit = MODEL (n=2 encoders); do not present as many independent replicates. |
| C (extended) | E10b Phi-3 pair tomography | **UPDATE 2026-08-22: E10b is now frozen** (commit `fafd44d`, `results/e10b_phi3_fit_results.json` + `results/e10b_phi3_tomography_responses.json`). Result is `MIXED_OR_UNRESOLVED`: F2->F3 improves 36.4% at eps=0.5 (PAIR_TERMS_REQUIRED, bootstrap CI excludes zero on the positive side) but *worsens* -2.5% at eps=1.0 (bootstrap CI excludes zero on the negative side — F3 reliably worse than F2). Diagnosed (not rescued) as a three-way redundancy break among the 3 layer-2 rows. Integrated into panel C as a third bar-group (hatched) alongside MosaicBERT/ModernBERT, directly visualizing the split. | `results/e10b_phi3_fit_results.json` (`.by_epsilon.{0.5,1.0}.{F2,F3,F2_vs_F3_relative_MAE_improvement,F2_vs_F3_bootstrap,mechanical_decision}`) | `fig2_causal.py::panel_c` (updated) | `figures/main/fig2_causal.{pdf,png}` | Main | Reported honestly as split by epsilon, per E10B_SYNTHESIS.md's own framing — not forced into one label. This is n=1 decoder (Phi-3 was the only one E10 flagged eligible); does not establish decoders in general require pair terms. |

**Data-integrity note (from audit):** `ARCHITECTURE_SYNTHESIS.md`'s retrospective pair-structure
prose mislabels two field names — the quoted "20-40% same-layer... vs 33% baseline"
(MosaicBERT) and "40-50%... vs 64% baseline" (ModernBERT) percentages are actually
`frac_same_output_row_*`, not `frac_same_layer_*`. The true same-layer fractions are much
lower (MosaicBERT 0.20 top / 0.089 all; ModernBERT 0.10 top / 0.044 all) — the qualitative
conclusion ("not same-layer-dominated") still holds, but any figure caption quoting this
descriptive aside must use the corrected field, not the mislabeled prose number. This
supplementary aside is not plotted as a data panel in this figure set (kept as a
`RESULTS.md`-level correction note only) — flagged here so it is not silently propagated.

Do NOT claim: architecture universally determines mechanism; all decoders are
single-component; all encoders are interactional; q1 causes the causal regime.

---

## Figure 3 — DNABERT-2 mechanistic organization and falsification

Claim: DNABERT-2's high-gain system is a redundant causal circuit whose interaction
structure is mechanistically meaningful (fitted, held-out-validated pair requirement,
E9/C-044), but analogous structural prominence elsewhere (NTv3) is not sufficient for
criticality.

Two-tier provenance note (load-bearing for this whole figure, UPDATED 2026-08-22 after a
fresh reproduction attempt): panels A-D below are ALL backed by E9's own from-scratch raw
JSON (`baseline_regression_results.json`, `fit_results_dnabert2.json`,
`dnabert2_mask_responses.json`) on the **pretrained-MLM endpoint**, independently measured
2026-08-22 with full held-out/bootstrap validation where applicable. This endpoint is not in
dispute.

A separate, later attempt to reproduce the colleague-branch narrative on the **fine-tuned
splice-accuracy endpoint** (C-036) and the **codominance ratio sweep** (C-038) now *does*
have a real raw artifact — but that artifact materially conflicts with the previously-adopted
magnitudes: measured joint pair effect on splice accuracy is **-0.42pp** (vs. the previously
cited -33.76pp — a number that is, on its face, larger than this repo's own established
10-row full-ensemble effect, -26.84pp), and the fitted "critical pair" is not even the
strongest epistatic pair in this remeasurement (L3/r86+L3/r399 is, at -2.10pp epistasis vs.
the named pair's -0.53pp). The codominance ratio sweep reproduces the same *qualitative*
direction (epistasis shrinks toward zero as the ratio moves away from co-dominance) but at
roughly 60x smaller magnitude (-0.53->-0.04pp vs. the previously-cited -33.63->-0.66pp). Both
new measurements are single-seed only (the only checkpoint available), weaker evidence than
the 5-seed protocol the old numbers claimed.

**Per explicit instruction, neither the old (now-contested) magnitude nor the new
single-seed value is plotted as an established or headline result.** C-036 and C-038 are
downgraded to **contested** in this figure set (real, code-verified raw artifacts now exist
for both, and they conflict with the previously-adopted colleague-report numbers by roughly
two orders of magnitude — this is a genuine open discrepancy, not a rounding difference,
and is disclosed rather than resolved by picking a side). Figure 3 therefore stays centered
on the panels below, none of which are affected by this dispute:

| Panel | Question | Claim | Source artifact(s) | Script | Output | Main/Supp | Caveat |
|---|---|---|---|---|---|---|---|
| A | Singleton vs. pair causal effect | Critical pair (L9/r264, L9/r294 = basis indices 3,4) individual pretrained-MLM dloss effects are small relative to other singletons; the fitted pairwise interaction term is the single largest of 45 | E9 own data: `fit_results_dnabert2.json` `by_epsilon.{eps}.singleton_effects_x_i[3,4]` (all 10) + `F3.gamma_pairs["(9, 264)x(9, 294)"]`, `F3.critical_pair_rank_by_abs_gamma=1` | `fig3_dnabert.py::panel_a` | `figures/main/fig3_dnabert.{pdf,png}` | Main | Endpoint is pretrained MLM loss (dloss), not the now-contested fine-tuned splice-accuracy claim (C-036) — this panel is the fully-provenanced, held-out-adjacent analogue, on a different endpoint that is not itself in dispute. |
| B | Is the interaction pretraining-intrinsic? | DNABERT-2 pretrained-MLM epistasis at full ablation = **+2.011803** (E9's own from-scratch reproduction, matching the previously-reported colleague figure +2.0118 to 4 decimals) — the effect exists at the pretraining stage, no fine-tuning/task head required; individual-channel effects (`d_A`, `d_B`) are small relative to the joint effect (`d_AB`) | `baseline_regression_results.json.dnabert2.{epsilon_0p5,epsilon_1p0}.{d_A,d_B,d_AB,epistasis}` — real, independently measured, NOT copied from prose | `fig3_dnabert.py::panel_b` | same file | Main | This is a genuine independent replication (different environment, different run, same result to 4 decimals) — the interpretation is now stated explicitly as: L9/r264+r294 is strongly supported as a pretrained-MLM interaction; the previously-claimed downstream splice/codominance magnitudes are a *separate, contested* claim (see note above), not confirmed by this panel. |
| C | E9 mechanistic tomography (F0-F3 observer ladder) | F0/F1/F2 fail held-out adequacy at both scales; F3 (pair-lifted) materially improves held-out MAE (54.7%/24.4%), bootstrap CI excludes zero | `fit_results_dnabert2.json`, `dnabert2_mask_responses.json` | `fig_e9_observer_ladder.py` (existing, spec-compliant — reused as-is) | `figures/main/fig3_dnabert.{pdf,png}` (composited) | Main | None beyond §18 basis-dependence caveat (declared residual-channel basis, not proof of intrinsic causal dimension 2). |
| D | Pair interaction map | Known critical pair ranks #1/45 pairs by \|Gamma\| at both scales, without special treatment during fitting | `fit_results_dnabert2.json` (`F3.gamma_pairs`) | `fig_e9_pair_interaction_map.py` (existing, spec-compliant — reused as-is) | same file | Main | Retrospective validation, not a basis-selection procedure (stated explicitly in caption). |
| — | Codominance mechanism (E1/E2 BREAK-PAIR ratio sweep, C-038) | **Contested, not plotted.** | `results/mechanism/codominance_break_dnabert2.json` (real, new, single-seed) vs. `results/mechanism/E1_E2_CODOMINANCE_AND_CONFOUND.md` (prose, previously adopted) | — | — | — | Real artifact now exists but conflicts with the previously-adopted magnitude by ~60x (see note above). Neither number is plotted per explicit instruction; flagged as a discrepancy requiring author adjudication, not resolved here. |
| — | DNABERT-2 splice-accuracy pair ablation (C-036) | **Contested, not plotted.** | `results/mechanism/pairwise_epistasis_dnabert2_reconstructed.json` (real, new, single-seed) vs. `results/mechanism/SUPERADDITIVITY_AND_COMPOSITION_REPORT.md` (prose, previously adopted) | — | — | — | Same discrepancy as above (-0.42pp measured vs. -33.76pp previously cited). Not plotted. |
| — | NTv3 falsification (corrected, C-029) | **Not producible as a current-result panel.** | — | — | — | — | The **only** NTv3-splice raw JSON in this repo (`results/gue_multiseed_ntv3_splice.json`) is the retired, truncation-bug-affected X-009 result — not shown anywhere in this figure set (see Supplement note below: the retired-result supplement panel has been removed per explicit instruction, not just left out of main text). The corrected replacement number (C-029) still has no raw artifact. NTv3's real structural data (q1=0.3889, genuine outlier) already appears in Figure 1. |

Do NOT claim: codominance is universally sufficient; norm dominance predicts criticality;
distributed geometry causes redundancy; the contested splice/codominance magnitudes as
established in either direction.

**Resulting main-text Figure 3 remains 4 panels (A-D), all fully provenanced to raw JSON
and none affected by the C-036/C-038 dispute.**

---

## Figure 4 — GENERator: functional and biological realization of a decoder high-gain pathway

Claim: a structurally concentrated decoder high-gain pathway (row 2371, layer 4) shows a
BOS-centered attention/activation phenotype (association only) and can exert strong,
directionally consistent causal control over generated genomic composition (GC fraction),
though only over a coarse 3-point dose grid and not as a validated linear/graded steering
axis.

| Panel | Question | Claim | Source artifact(s) | Script | Output | Main/Supp | Caveat |
|---|---|---|---|---|---|---|---|
| A | Structural phenotype (minimal) | GENERator EUK q1=0.9689 (near-rank-1) plotted as a small callout referencing Figure 1B, not a full restatement | `results/e7_legacy_reanalysis.json["GENERator EUK"]` | `fig4_generator.py::panel_a` | `figures/main/fig4_generator.{pdf,png}` | Main | Kept deliberately minimal — no separate activation-lifecycle artifact exists in this repo for GENERator, so this panel does not claim more than the structural number already established. |
| B | BOS-centered attention/activation phenotype (C-039) | **Now reproduced with a real artifact** (2026-08-22): 37.9% of incoming attention mass at position 0 (33.0x uniform expectation), 78.75% of (layer,head) pairs argmax at position 0, row 2371's activation maximum also at position 0 | `results/mechanism/attention_sink_implicit_bias.json` (`.generator.attention.{sink_share_pos0,uniform_expectation,sink_over_uniform,frac_heads_argmax_pos0}`, `.generator.implicit_bias.pos0_real_mean`) — produced by `scripts/mechanism/run_attention_sink.py` after fixing a hardcoded hg38 path absent on this filesystem (no science changed) | `fig4_generator.py::panel_b` | same file | Main | **Strictly associative wording**: "BOS-centered attention/activation co-occurs with the high-gain pathway" — does NOT claim the high-gain row causes the attention sink. Single run (seed 42, 40 windows), not yet replicated across seeds. |
| C | GC causal dose-response | Row 2371: monotonic decrease, span ≈390x random-control span (0.1017 vs 0.0003); row 1522: non-monotonic (rises then falls); random control flat | `baseline_regression_results.json.generator` (E9's own from-scratch measurement, C-045 — **explicitly not** the colleague's prose-only C-040 "38.6×"/"17.9×" number, which used a different, never-reproduced 5-point alpha grid per `BASELINE_REGRESSION.md`'s own disclosure) | `fig_e9_generator_dose_response.py` (existing, spec-compliant — dotted guide-the-eye connectors already implemented per the correction that a fitted curve would overclaim — reused as-is) | same file | Main | "Causal sensitivity with a directionally consistent primary row," not "validated graded steering" (E9 RESULTS.md §13's own wording) — caption repeats this framing verbatim. |
| — | Quality/specificity over the intervention range | **Not producible.** | — | — | — | — | Confirmed entirely absent, not even as prose numbers — `PROVENANCE_AND_BASELINES.md` states outright no raw output exists for the designed quality-metric script. Recorded as a "could not be produced" item. |

**Resulting main-text Figure 4 is now 3 panels (A structural callout, B BOS phenotype
[reproduced], C GC dose-response)** — the BOS panel closes the gap that made this figure
thin in the previous pass; only the quality/specificity panel remains unproducible.

---

## Supplementary figures (updated)

### Supplement S1 — Full E7/E8 structural table + decoder control-row distributions
- Panel S1a: full q1/PR_spec table for all 12 models in Figure 1's panel, including
  per-row Phi-3 detail (6 rows) not shown in the main-figure aggregate.
  Source: `results/e7_phi3_spectral.json.rows[]`.
- Panel S1b: all 5 decoders' same-layer control-row distributions (the random-control rows
  behind Figure 2A's control-normalized thresholds).
  Source: `results/e10_decoder_control_rows.json`.
- Output: `figures/supplement/fig_s1_structural_detail.{pdf,png}`.

### Supplement S2 — REMOVED
The retired NTv3 splice-ablation figure (previously shown explicitly labeled RETIRED, for
methodological transparency) has been **removed** per explicit instruction: an invalid
(truncation-bug-affected) result is not plotted even for transparency purposes. The retired
status of that result (X-009) remains documented in `CLAIMS_LEDGER.md`/`FIGURE_PROVENANCE.md`
text; it no longer has a rendered figure anywhere in this figure set.

### Not produced as figures (main or supplement) — current state
- GENERator sequence-quality-during-steering: no artifact of any kind.
- DNABERT-2 codominance ratio sweep (C-038) and splice-accuracy pair ablation (C-036): real
  artifacts now exist but are contested (see Figure 3 note above) — not plotted pending
  author adjudication of the discrepancy.
- NTv3 MAKE-PAIR and corrected splice ablation (C-029): still no artifact.
- E10b Phi-3 tomography: **now produced** — integrated into Figure 2 panel C (no longer a
  gap; see Figure 2 section above).

---

## Panel count summary

| Figure | Main panels | Supplement panels | Panels not producible / contested |
|---|---|---|---|
| 1 — Structural | 4 (A-D) | — | — |
| 2 — Cross-model causal | 3 (A-C, C now includes E10b/Phi-3) | 2 (S1a-b) | — |
| 3 — DNABERT-2 mechanism | 4 (A-D) | 0 (S2 removed) | Codominance + splice pair-ablation (C-036/C-038, contested, not plotted either direction); NTv3 corrected falsification (no artifact) |
| 4 — GENERator | 3 (A-C, B now reproduced) | — | quality/specificity panel (no artifact) |

**All rendered panels (main and supplement) are backed by a raw, machine-readable artifact
with a traceable file path and JSON key — none are transcribed from prose, and none plot a
contested or retired number.** See `FIGURE_PROVENANCE.md` for the panel-by-panel manifest.
