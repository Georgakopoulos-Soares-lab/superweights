# MANUSCRIPT_SOURCE_OF_TRUTH.md

**Status: COMPLETE.** All 12 sections are final, built from (a) this project's own verified
figure-provenance work (`manuscript/figures/FIGURE_PLAN_v3.md`, `FIGURE_PROVENANCE.md`,
`FIGURE_CAPTIONS_v3.md`, and the exact JSON values saved in
`manuscript/figures/source_data/*.json`, each of which is itself loaded live from the raw
experiment artifact at figure-render time — not transcribed by hand), and (b) two dedicated
repository fact-extraction passes covering exact model provenance/mathematical
definitions/intervention procedures (§4) and dataset/reference/manuscript-logistics facts
(§5, §10, §11), each independently quote- and path-verified against the actual source files.

This document is a **reference package**, not manuscript prose. Do not treat any sentence
here as ready-to-publish text except where explicitly marked as a draft claim statement or
figure legend. Prefer raw machine-readable evidence over prose summaries; where two sources
conflict, both are shown, not reconciled by picking one.

---

## 0. Governing conceptual framing (as given, confirmed consistent with repo state)

> High-gain gated-FFN structures recur across text and genomic foundation models, but
> structural prominence, functional criticality, and finite-intervention causal response
> complexity are distinct quantities. Encoder/decoder architecture organizes structural
> geometry more consistently than it organizes causal mechanism.

```
STRUCTURAL GEOMETRY != FUNCTIONAL CRITICALITY != CAUSAL RESPONSE COMPLEXITY
```

This is confirmed as the repo's live thesis: `manuscript/CLAUDE.md`'s frozen thesis section
and `manuscript/docs/DECISIONS.md` D-016 retire the older "predictable-routing" thesis in
favor of this one; the E7-E10b experiment arc (2026-08-14 through 2026-08-22) was run
specifically to test it. **This document reports the state of that thesis as of 2026-08-22,
including one newly-discovered contested discrepancy (DNABERT-2 splice/codominance magnitude,
see §7) that was not yet known when D-016 was written.**

---

## 1. Paper framing and outline

### 1.1 Title options (draft, not finalized in the repository — no committed title found)

1. *Structure Is Not Mechanism: High-Gain Gated-FFN Pathways Across Text and Genomic
   Foundation Models*
2. *Encoder/Decoder Architecture Organizes Structural Geometry, Not Causal Mechanism, in
   High-Gain FFN Pathways*
3. *Beyond the Super Weight: Structural, Functional, and Causal Dissociation in Gated-FFN
   High-Gain Channels*
4. *One Structure, Many Mechanisms: Cross-Architecture Analysis of High-Gain Pathways in
   Language and Genomic Models*
5. *Super Weights Across Domains: When Structural Concentration Does Not Predict Causal
   Complexity*

`paper/main.tex` does carry a working title — *"A Structural Predictor of Super-Weights
Across Genomic Language Model Architectures"* (`paper/main.tex:46-47`) — but this reflects
the older, explicitly-retired v1 thesis (§1.4), not the current framing, and should not be
reused without revision. Recommend (1) or (2) above instead: both foreground the paper's
actual current novel claim (the STRUCTURE != FUNCTION != CAUSAL-COMPLEXITY trichotomy) rather
than the retired "closed-form structural predictor" framing alone.

### 1.2 Central claim (concise)

High-gain rows in gated/GLU-style FFN blocks are a structural phenomenon that recurs across
text and genomic transformer models and is organized more consistently by encoder-vs-decoder
architecture than by text-vs-genomic domain — but this structural organization does **not**
transfer cleanly to functional criticality or to causal response complexity: decoders are
usually (not always) single-component causally dominant, encoders tested so far require
pairwise interaction terms to predict finite causal response, structural magnitude ranking
does not reliably predict which component is causally dominant (OLMo), and a structurally
prominent channel can be causally inert (NTv3, descriptively — see §7 for artifact status).

### 1.3 Final recommended four-part Results structure

This matches the repository's own current figure structure exactly (`FIGURE_PLAN_v3.md`,
finalized 2026-08-22) — **the latest figure-generation scripts already reflect this
structure**; no reorganization is needed before writing.

| # | Section | Figure | What it establishes | What it explicitly does NOT establish |
|---|---|---|---|---|
| R1 | Structural organization across text and genomic models | Fig. 1 | Exact `q1`/`PR_spec`/`‖U_k‖_F` across a 12-model panel; encoder/decoder is a better organizing axis than domain, descriptively; NLP cold-weight calibration recovers published super-weights (Llama/Mistral/OLMo); detected high-gain rows are far more concentrated than same-layer ordinary rows (MosaicBERT, ModernBERT) | A clean binary domain or architecture split (NTv3 is a genomic-encoder outlier below the "encoder" pattern set by MosaicBERT/ModernBERT/DNABERT-2's own spread; GENERator EUK's `q1` sits inside the NLP-decoder range; ModernBERT's separation from the decoder floor is thin, 0.21% relative); `q1` as a magnitude or causal-importance measure; cross-model comparability of raw `‖U_k‖_F` |
| R2 | Causal organization is heterogeneous despite structural trends | Fig. 2 | Decoder singleton causal spectra for 5 models (4/5 SINGLE_COMPONENT_DOMINANT); OLMo's structural-vs-causal rank dissociation (structurally-largest row is causally weakest); 2/2 tested encoders (MosaicBERT, ModernBERT) require pair terms at both intervention scales; Phi-3 (E10b), the one decoder flagged eligible for its own tomography, gives a **split** result (pair terms help at ε=0.5, reliably hurt at ε=1.0 — diagnosed as a three-way redundancy break, not resolved) | "All decoders are single-component" (false — Phi-3); "all encoders are interactional" (n=2, both agree, not universal); architecture *determines* causal organization (only *associates* with it in this panel); `q1` causes the causal regime |
| R3 | DNABERT-2 mechanistic deep dive | Fig. 3 | The known critical pair (L9/r264, L9/r294) is a real, held-out-validated pairwise interaction on the pretrained-MLM endpoint (epistasis +2.0118, reproduced independently to 4 decimals; F2→F3 held-out MAE improves 54.7%/24.4% with bootstrap CI excluding zero at both scales; the pair ranks #1/45 without special treatment during fitting) | The previously-claimed catastrophic downstream splice-accuracy pair effect (−33.76pp) or codominance magnitude (−33.63→−0.66pp epistasis) as established — a fresh, real, single-seed reproduction materially conflicts with both (see §7); that DNABERT-2's causal dimension is intrinsically 2 (only within the declared 10-row residual-channel basis); anything about NTv3 as a falsification case (no current-result artifact exists, see §7) |
| R4 | GENERator functional/biological deep dive | Fig. 4 | GENERator EUK row 2371's near-rank-1 structural phenotype; a BOS-centered attention/activation phenotype, now reproduced with a real artifact (37.9% attention mass at position 0, 33.0× uniform, 78.8% of heads argmax there, activation max also at position 0); a large, directionally consistent causal dose-response of generated GC composition for row 2371 (span ≈390× random control) | That the high-gain row *causes* the BOS attention sink (association/co-occurrence only, and a causal-decoder structural fact — position 0 attends only to itself — partly guarantees some of this co-occurrence architecturally); a validated linear/graded steering axis (only 3 dose points by design; secondary row 1522 is non-monotonic); sequence-quality stability under intervention (no artifact) |

### 1.4 Old vs. current manuscript state (factual, not reconciled here)

- **`paper/main.tex` exists on disk** (55,408 bytes, header "v6", last modified 2026-08-13)
  and reflects the **old, explicitly-retired v1 thesis**. Its Abstract states the panel
  scanned is "eight of the best genomic language models" and frames the central contribution
  as a "closed-form, weight-only structural predictor of candidate super-weight rows... the
  Frobenius bound `‖U_k‖_F`" that "ranks the empirical super-rows at the top of their layer
  in three independent gated-FFN transformer models (GENERator SwiGLU decoder, DNABERT-2 GLU
  encoder, NTv3 encoder)," with Evo1 as a structural-positive/causal-null case. **No edit to
  `paper/main.tex` postdating the 2026-08-13 thesis-retirement decision (D-016) was found** —
  `PAPER_OUTLINE.md` itself notes "Figure numbering is not re-derived in this pass (no
  manuscript rewrite performed)."
- `manuscript/CLAUDE.md`'s frozen thesis statement (v2, 2026-08-13) explicitly retires this
  v1 framing: *"the v1 U_k-led thesis below is retired (`DECISIONS.md` D-016)"* — and records
  the second-generation thesis (super-weight-like concentration transfers, but the causal
  object is not universal). This is the closest existing repository document to the *current*
  framing in §0-1.2 above, but predates the E7-E10b work (2026-08-14 through 2026-08-22) that
  produced most of §3's headline results.
- `manuscript/docs/PAPER_OUTLINE.md` (v2, 2026-08-13) is organized around a *different*,
  intermediate R1-R6 skeleton, of which R2/R3/R4/R5 are explicitly marked "BLOCKED, no
  artifact" in the outline text itself, with only R1 and R6 stated as draftable at that time.
  This too predates and does not reference E7-E10b.
- **Root `README.md`** does not itself state a manuscript thesis in prose; its "Mechanism
  session findings (Aug 2026)" section presents a findings table broadly consistent with the
  second-generation thesis (not the v1 `‖U_k‖_F`-led framing `paper/main.tex` still contains).
- **Net conclusion: three internally-inconsistent framings currently coexist in this
  repository** — `paper/main.tex` (v1, retired), `PAPER_OUTLINE.md` v2 (intermediate,
  R1-R6, itself now stale), and this document's §0-1.3 (current, matching the E7-E10b/E9
  figure work actually completed). **No manuscript rewrite is undertaken by this document** —
  see the UNRESOLVED FACTS section for what a future writing pass needs to reconcile, and
  §11 for which `paper/main.tex` logistics fields (author list, affiliations, corresponding
  author) can likely be reused as-is despite the thesis mismatch.

### 1.5 Recommended Discussion-level takeaways

- The paper's central conceptual contribution is the three-way dissociation (structural
  geometry / functional criticality / causal response complexity) — every other finding is in
  service of demonstrating this dissociation empirically, not of re-establishing the older
  "super weight" transfer story on its own.
- Encoder/decoder architecture is a *better* organizing axis than text/genomic domain for
  structural geometry in this panel, but this is descriptive, not a clean binary law — NTv3,
  ModernBERT's thin margin, and GENERator EUK sitting inside the NLP-decoder `q1` range are
  all genuine complications, not noise to explain away.
- The same architecture pattern does *not* reliably transfer to causal mechanism: Phi-3 is a
  decoder that requires pair terms (partially); OLMo shows structural magnitude ranking and
  causal dominance ranking can diverge within a single model's own row set.
- DNABERT-2 is the strongest positive mechanistic case in the whole panel: a real,
  independently-reproduced, held-out-validated pairwise interaction on the pretrained
  objective. It is *not*, currently, backed by an established downstream/fine-tuned magnitude
  claim — that part of the story needs author adjudication before publication (§7).
- GENERator demonstrates that a single near-rank-1 decoder pathway can have large, real,
  non-trivial causal control over a downstream biological property (GC composition) without
  that control being linear, graded, or (yet) causally linked to its attention-sink-like
  co-occurrence pattern.
- Sample sizes throughout are small (5 decoders, 2-3 encoders per tier) — architecture
  association is a descriptive finding in this tested panel, not a universal law, and the
  paper's own language should reflect that throughout Discussion.

---

## 2. Final figure map

Governing figure specification: `scripts/analysis/_figstyle.py` (repo root) — the only
codified, repo-wide figure style module found in this repository (`apply_style()`/
`panel_label()`: no suptitles/per-panel titles, compact bold corner panel labels, sans-serif
9pt, no top/right spines, `savefig.dpi=220`, editable-text vector PDF). Shared model-class
visual encoding: `manuscript/figures/_paper_encoding.py` (marker shape = architecture,
color = domain). **The latest figure-generation scripts (`manuscript/figures/fig1_structural.py`
through `fig4_generator.py`, plus `supplement/fig_s1_structural_detail.py`) already reflect
the exact structure below** — this is not a proposal, it is a description of what already
renders. Full per-panel JSON-key-level provenance: `manuscript/figures/FIGURE_PROVENANCE.md`.

### Figure 1 — Cross-domain structural organization of high-gain FFN pathways

| Panel | Purpose | Artifact | Metric | Model(s) | N | Key result | Safe interpretation | Must NOT claim |
|---|---|---|---|---|---|---|---|---|
| A | NLP cold-weight calibration | `results/e1_nlp_retrospective.json` | row rank, percentile, scalar top-1 share (diagonal approx.) | Llama-7B, Mistral-7B, OLMo-7B | 3 models, 1 checkpoint each | Row rank 1/N (100th pctile) all 3; top1_share 0.890/0.988/0.956 | Cold-weight predictor recovers published NLP super-weights | That this diagonal-approx metric is the same quantity as `q1`/`PR_spec` in panels B/C (confirmed numerically distinct, e.g. Llama `uk_norm`=123.64 diag vs `frob_norm`=137.62 exact) |
| B | Exact spectral concentration across domains/architectures | `results/e7_legacy_reanalysis.json`, `e7_phase1_detection_{qwen25,genomeocean,evo2}.json`, `e7_phi3_spectral.json`, `e8_detection_{mosaicbert,modernbert}.json` | exact `q1` (top singular-value share of exact bilinear operator `U_k`) | 12 models: Mistral-7B, Llama-7B, GENERator EUK, OLMo-7B, Qwen2.5-7B, Phi-3-mini, GenomeOcean-4B, DNABERT-2, ModernBERT, MosaicBERT, NTv3, Evo2-7B (null) | 1 checkpoint/model | `q1` range 0.9922(Mistral)→0.3889(NTv3); Evo2-7B has no measured `q1` (Phase-1 detection null, ratio 2.22<5.0×) | Encoder/decoder is a better organizing axis than domain, descriptively, with named exceptions | `q1` as magnitude; `q1` as causal-importance predictor; a clean domain split |
| C | Exact operator magnitude, kept separate from `q1` | same files as B | exact `‖U_k‖_F` (Frobenius norm), log scale | same 11 measured models | 1/model | Spans 0.39 (Mistral) to 836.05 (ModernBERT) | Magnitude is a distinct axis from concentration | Cross-model magnitude comparability (different weight scales/parameterizations — explicitly not comparable in absolute terms) |
| D | Encoder high-gain row vs. ordinary same-layer rows | `results/e8_detection_{mosaicbert,modernbert}.json` | candidate `q1`/`PR_spec` vs. 5 seeded control rows' `q1`/`PR_spec` | MosaicBERT, ModernBERT | 1 candidate + 5 controls/model | MosaicBERT: 0.4766 vs 0.035-0.047; ModernBERT: 0.8970 vs 0.024-0.049 | Detected row is far more concentrated than ordinary rows, even in "distributed" encoders | Generalization beyond these 2 models (no same-layer controls exist for DNABERT-2/NTv3) |

### Figure 2 — Cross-model causal organization

Endpoint: causal-LM NLL (decoders) / masked-LM loss (encoders); row-scale intervention
`alpha = 1 - epsilon*a` on `mlp.down_proj`.

| Panel | Purpose | Artifact | Metric | Model(s) | N | Key result | Safe interpretation | Must NOT claim |
|---|---|---|---|---|---|---|---|---|
| A | Decoder singleton causal spectrum | `results/e10_decoder_concentration.json` | `\|relative %NLL change\|` at full ablation per row; control median | Llama(K=1), Mistral(K=1), OLMo(K=4), Phi-3(K=6), Qwen2.5(K=1) | 5 models; 1-6 rows/model + 5 control rows/model | 4/5 SINGLE_COMPONENT_DOMINANT (C1>0.5, control-norm.>3.0); Phi-3 MULTI_COMPONENT_CANDIDATE (C1=0.333); Mistral non-monotonic (α=0.5 effect exceeds full ablation) | Decoders tend toward single-component dominance, with a real exception | "All decoders are single-component" (false); that K=1 models' single row was "compared" to alternatives (none exist) |
| B | OLMo structural-vs-causal dissociation | `results/e10_exact_uknorm_olmo.json` + `.../e10_decoder_concentration.json` | structural rank (exact `‖U_k‖_F`/layer median) vs. causal rank (`\|%NLL change\|`) | OLMo-7B | 4 rows, same model | L24/r269 structurally largest (53.4× layer median) but causally weakest (+1.4%); L1/r269 causally dominant (+112%) but ranks 3rd structurally | Structural magnitude ranking and causal-effect ranking can dissociate within one model's own row set | That this generalizes beyond OLMo or invalidates the structural criterion generally (it correctly measures a different quantity) |
| C | Pair-requirement across encoders + the one eligible decoder | `results/e10_encoder_fit_results.json`, `results/e10b_phi3_fit_results.json` | F2→F3 held-out MAE relative improvement (%) + bootstrap 95% CI, both ε | MosaicBERT, ModernBERT (encoders); Phi-3 (decoder, E10b) | 2 encoders × 2 ε; 1 decoder × 2 ε | MosaicBERT +84.2%/+54.5%; ModernBERT +58.8%/+46.4% (both CI excl. zero, both ε); Phi-3 **splits**: +36.4% (ε=0.5, PAIR_TERMS_REQUIRED) vs. −2.5% (ε=1.0, MIXED_OR_UNRESOLVED, pairwise model reliably worse) | 2/2 tested encoders require pair terms at both scales; interactional causal organization is not encoder-specific (Phi-3 partially replicates it, then breaks down at full ablation via a diagnosed three-way redundancy) | "All encoders are interactional" (n=2); "decoders never need pair terms" (Phi-3 partially disproves this); that Phi-3's ε=1.0 result validates a higher-order (triple) model — no such model was fit |

### Figure 3 — DNABERT-2 mechanistic organization

All panels on the **pretrained-MLM endpoint**, from E9's own from-scratch 2026-08-22
measurement (`experiments/E9_mechanistic_tomography/{baseline_regression_results.json,
fit_results_dnabert2.json,dnabert2_mask_responses.json}`). None of these four panels are
affected by the §7 splice/codominance dispute (different endpoint entirely).

| Panel | Purpose | Artifact | Metric | Model(s) | N | Key result | Safe interpretation | Must NOT claim |
|---|---|---|---|---|---|---|---|---|
| A | Singleton vs. pair effect | `baseline_regression_results.json.dnabert2.epsilon_1p0` | individual vs. joint Δ pretrained-MLM loss | DNABERT-2 | 256 hg38 windows, 1 seed | `d_A`=+0.0218, `d_B`=+0.0064, `d_AB`=+2.040 (joint ≫ sum of parts) | Individually small, jointly large — real, directly-measured 2-channel intervention | Conflation with the (contested, unplotted) splice-accuracy pair effect — different endpoint/unit |
| B | Pretraining-intrinsic super-additivity | same file, both ε | epistasis = `d_AB - d_A - d_B` | DNABERT-2 | same | +2.0118 (ε=1.0), +0.0528 (ε=0.5), both positive; reproduces a prior figure to 4 decimals from a from-scratch environment | Interaction present at pretraining, not scale-dependent artifact | The downstream splice-accuracy or codominance magnitude (separate, contested claim) |
| C | Observer-family (F0-F3) held-out prediction | `fit_results_dnabert2.json`, `dnabert2_mask_responses.json` | held-out R², MAE, normalized MAE | DNABERT-2 | 10-row basis; 78 fit + 20 calib + 20 held-out masks, 2 ε | F0 R²=−0.023/−0.040; F1 0.324/0.659; F2 0.517/0.581; F3 0.888/0.790 (ε=0.5/1.0) | Additive families fail held-out adequacy; pair-lifted family materially improves it | That F3's fit constitutes proof of intrinsic causal dimension 2 (basis-dependent, stated explicitly) |
| D | Pair-interaction map | `fit_results_dnabert2.json` (`F3.gamma_pairs`) | fitted pairwise coefficient `Γ`, ranked | DNABERT-2 | 45 pairs, 2 ε | Known pair (L9/r264,r294) ranks #1/45 at both ε (Γ=0.077 / 0.739) without special treatment during fitting | Retrospective validation of an independently-known pair | That this was a basis-selection procedure (it was not — pair was already known before F3 was fit) |

**Not plotted, contested (see §7):** DNABERT-2 splice-accuracy pair ablation (C-036) and
codominance ratio sweep (C-038) — both now have real, single-seed, code-verified artifacts
that materially conflict with previously-adopted magnitudes. Neither number is used anywhere.
**Not plotted, no artifact:** NTv3 falsification (corrected C-029; only the retired,
truncation-bug X-009 artifact exists, and it is not shown even labeled retired, per explicit
instruction removing the prior transparency-only supplement panel).

### Figure 4 — GENERator functional/biological realization

| Panel | Purpose | Artifact | Metric | Model(s) | N | Key result | Safe interpretation | Must NOT claim |
|---|---|---|---|---|---|---|---|---|
| A | Minimal structural callout | `results/e7_legacy_reanalysis.json["GENERator EUK"]` | exact `q1` | GENERator EUK | 1 row (L4/r2371) | `q1`=0.9689 | Cross-references Fig. 1B; near-rank-1 | Restating Fig. 1's full comparison |
| B | BOS-centered attention/activation phenotype | `results/mechanism/attention_sink_implicit_bias.json` (reproduced 2026-08-22) | mean incoming attention @pos0 vs. uniform; frac. heads argmax@pos0; activation max location | GENERator EUK | 40 hg38 windows, seed 42, 1 run | 37.9% attention @pos0 (33.0× uniform); 78.75% of (layer,head) argmax @pos0; activation max also @pos0 | BOS-centered attention/activation **co-occurs** with the high-gain pathway | That the high-gain row **causes** the sink (association only; causal-decoder position-0 self-attention structurally guarantees part of this) |
| C | GC causal dose-response | `experiments/E9_mechanistic_tomography/baseline_regression_results.json.generator` | generated GC fraction vs. row-scale α∈{0,0.5,1.0} | GENERator EUK, row 2371 (primary), row 1522 (secondary), 5-row random control | 24 hg38-window prompts/condition (120 for random control, 5 rows×24) | Row 2371 span 0.1017 (≈390× random-control span 0.00026); monotonic; row 1522 non-monotonic | Large, directionally consistent causal sensitivity for row 2371 | Validated linear/graded steering (only 3 dose points by design); that row 1522 behaves the same way |

**Not plotted, no artifact:** sequence-quality/specificity over the intervention range —
confirmed entirely absent, not even as prose numbers.

### Supplement S1 — Full structural/control-row detail

| Panel | Purpose | Artifact | Metric |
|---|---|---|---|
| a | Phi-3's 6 individual rows' exact `q1` | `results/e7_phi3_spectral.json.rows[]` | per-row `q1`, spread 0.558-0.953 vs. model-level median 0.9028 used in Fig. 1B |
| b | All 5 decoders' same-layer control-row causal effects | `results/e10_decoder_concentration.json.models.{*}.controls[]` | `\|relative %NLL change\|`, log scale — the individual points behind Fig. 2A's control-median line |

**Supplement S2 removed** (2026-08-22, explicit instruction): previously showed the retired,
truncation-bug-affected NTv3 splice result (X-009), explicitly labeled RETIRED, for
transparency. An invalid result is not plotted in this figure set even labeled as retired.

### Draft figure legends (evidence-supported only)

**Figure 1.** *Exact structural geometry of high-gain gated-FFN rows across a 12-model panel.*
(A) Cold-weight calibration: the exact `U_k` predictor recovers the published NLP
super-weight row at rank 1 (100th percentile) in Llama-7B, Mistral-7B, and OLMo-7B, with
scalar top-1 share 0.89-0.99 (diagonal approximation; retrospective, E1). (B) Exact spectral
concentration (`q1`) across 5 text decoders, 2 genomic decoders, 2 text encoders, 2 genomic
encoders, plus one structural null (Evo2-7B, ×). Marker shape = architecture, color = domain.
Dashed line: pre-registered decoder floor (GenomeOcean-4B, `q1`=0.8989). (C) Exact operator
magnitude (`‖U_k‖_F`, log scale), kept on a separate axis from `q1` by design; raw magnitudes
are not cross-model comparable in absolute terms. (D) For the two encoders with same-layer
controls measured, the detected row (filled) is far more concentrated than 5 ordinary rows
at the same layer (open). No error bars: single-checkpoint, exact (non-stochastic)
measurements. Descriptive, not a formal statistical test of domain-vs-architecture
separation.

**Figure 2.** *Cross-model causal organization.* (A) Decoder singleton causal spectrum at
full ablation (`ε=1.0`), 5 models; dotted line = same-layer random-control median; color =
mechanical decision (blue single-component-dominant, red multi-component-candidate). Mistral
is non-monotonic. (B) OLMo's structural-vs-causal rank dissociation across its own 4-row set.
(C) F2→F3 held-out MAE improvement, both intervention scales, with bootstrap 95% CI (5,000
resamples, paired over held-out masks), for MosaicBERT, ModernBERT (encoders) and Phi-3
(decoder, E10b, hatched); Phi-3 splits by scale, reported honestly rather than forced into
one label. Sample unit: model.

**Figure 3.** *DNABERT-2 mechanistic organization on the pretrained-MLM endpoint.* (A)
Individually masking either channel of the critical pair (L9/r264, L9/r294) produces small
effects; a direct joint 2-channel intervention produces a much larger effect than the sum of
parts (`ε=1.0`). (B) The same super-additivity holds at both tested scales; independently
reproduces a prior pretraining-epistasis figure to 4 decimals. (C) Held-out prediction
quality for four observer families (F0 singleton-additive through F3 pairwise-lifted) on 20
genuinely held-out masks; only F3 clears or approaches adequacy. (D) F3's fitted pairwise
coefficients (top 12/45), critical pair ranked #1 at both scales without special treatment
during fitting. A related, previously-adopted claim about downstream splice-accuracy and
codominance magnitude is now contested by a fresh, conflicting, single-seed reproduction and
is not plotted (see manuscript text / §7 of this document for the exact discrepancy).

**Figure 4.** *GENERator: functional realization of a decoder high-gain pathway.* (A)
GENERator EUK row 2371's exact `q1`=0.969 (see Fig. 1B). (B) BOS-centered
attention/activation phenotype, reproduced 2026-08-22: 37.9% mean incoming attention at
position 0 (33.0× uniform), 78.8% of (layer,head) pairs argmax there, activation maximum
also at position 0 — association only, not shown to be caused by the high-gain row. (C)
Causal dose-response of generated GC composition under row-scale suppression (`α`∈{0,0.5,1});
row 2371 span ≈390× random-control span; row 1522 non-monotonic. Dotted connectors are
guide-the-eye, not fitted curves — only 3 dose points collected by design.

---

## 3. Headline results table

Every quantitative result likely to appear in Abstract/Results/Discussion, with exact values
loaded from the raw artifact (via the figure `source_data/*.json` snapshots, each of which is
itself generated by loading the cited raw JSON live — not hand-transcribed). Status column:
`MAIN-TEXT SAFE` / `SUPPLEMENT ONLY` / `PROVISIONAL` / `RETIRED`.

### 3.1 Structural analysis (exact metric, E7/E8)

| Model | Domain | Arch | `q1` | `PR_spec` | `‖U_k‖_F` | N (seeds/ckpts) | Artifact | Script | Status |
|---|---|---|---:|---:|---:|---:|---|---|---|
| Mistral-7B | text | decoder | 0.9922 | 1.0158 | 0.388 | 1 | `results/e7_legacy_reanalysis.json` | `E7_exact_dimensionality/run_legacy_reanalysis.py` | MAIN-TEXT SAFE |
| Llama-7B | text | decoder | 0.9888 | 1.0227 | 137.62 | 1 | same | same | MAIN-TEXT SAFE |
| GENERator EUK | genomic | decoder | 0.9689 | 1.0653 | 522.13 | 1 | same | same | MAIN-TEXT SAFE |
| OLMo-7B | text | decoder | 0.9646 | 1.0747 | 0.911 | 1 | same | same | MAIN-TEXT SAFE |
| Qwen2.5-7B | text | decoder | 0.9529 | 1.0995 | 29.69 | 1 | `results/e7_phase1_detection_qwen25.json` | `E7_exact_dimensionality/run_phase1_detection.py` | MAIN-TEXT SAFE |
| Phi-3-mini | text | decoder | 0.9028 (median, 6 rows) | 1.2249 (median) | n/a (per-row only) | 1 | `results/e7_phi3_spectral.json` | `E7_exact_dimensionality/run_phi3_spectral.py` | MAIN-TEXT SAFE |
| GenomeOcean-4B | genomic | decoder | 0.8989 | 1.2243 | 2.883 | 1 | `results/e7_phase1_detection_genomeocean.json` | `run_phase1_detection.py` | MAIN-TEXT SAFE |
| DNABERT-2 | genomic | encoder | 0.7933 | 1.5033 | 74.01 | 1 | `results/e7_legacy_reanalysis.json` | `run_legacy_reanalysis.py` | MAIN-TEXT SAFE |
| ModernBERT | text | encoder | 0.8970 | 1.2337 | 836.05 | 1 | `results/e8_detection_modernbert.json` | `E8_encoder_decoder/run_detection_and_spectral.py` | MAIN-TEXT SAFE |
| MosaicBERT | text | encoder | 0.4766 | 4.1159 | 20.00 | 1 | `results/e8_detection_mosaicbert.json` | same | MAIN-TEXT SAFE |
| NTv3 | genomic | encoder | 0.3889 | 6.4803 | 441.52 | 1 | `results/e7_legacy_reanalysis.json` | `run_legacy_reanalysis.py` | MAIN-TEXT SAFE |
| Evo2-7B | genomic | decoder | **NULL** | NULL | NULL | 1 | `results/e7_phase1_detection_evo2.json` | `run_phase1_detection.py` | MAIN-TEXT SAFE (as a structural null) |

Phi-3 per-row detail (SUPPLEMENT ONLY, `results/e7_phi3_spectral.json.rows[]`):

| Layer | Row | `q1` | `PR_spec` | `‖U_k‖_F` |
|---:|---:|---:|---:|---:|
| 2 | 525 | 0.9529 | 1.1010 | 120.77 |
| 2 | 1693 | 0.9137 | 1.1931 | 105.86 |
| 2 | 1113 | 0.7505 | 1.7691 | 97.52 |
| 4 | 525 | 0.9278 | 1.1615 | 72.12 |
| 4 | 1113 | 0.5584 | 3.1989 | 61.70 |
| 4 | 1693 | 0.8919 | 1.2567 | 46.49 |

Same-layer ordinary-row controls (MAIN-TEXT SAFE, `results/e8_detection_{mosaicbert,modernbert}.json.control_rows[]`, seed `SeedSequence(42)`):

| Model | Candidate `q1` | Candidate `PR_spec` | Control `q1` (5 rows) | Control `PR_spec` (5 rows) |
|---|---:|---:|---|---|
| MosaicBERT | 0.4766 | 4.116 | 0.0472, 0.0351, 0.0393, 0.0348, 0.0445 | 139.7, 169.1, 154.8, 163.7, 146.5 |
| ModernBERT | 0.8970 | 1.234 | 0.0485, 0.0316, 0.0426, 0.0241, 0.0328 | 90.4, 116.3, 91.2, 120.9, 101.1 |

NLP calibration (SUPPLEMENT-adjacent, cited in Fig. 1A; diagonal approximation, `results/e1_nlp_retrospective.json`):

| Model | Row rank | Percentile | Scalar top-1 share |
|---|---:|---:|---:|
| Llama-7B | 1/N | 100.0 | 0.8903 |
| Mistral-7B | 1/N | 100.0 | 0.9884 |
| OLMo-7B | 1/N | 100.0 | 0.9558 |

### 3.2 E10 decoder causal audit (exact, `results/e10_decoder_concentration.json`)

Endpoint: causal-LM NLL, row-scale intervention, full ablation (`α=0`) unless noted. Decision
thresholds: `C1>0.5` AND top-row control-normalized effect `>3.0` ⇒ SINGLE_COMPONENT_DOMINANT.

| Model | K | Row(s) (layer/row) | `dNLL` | `%change` | Control-normalized | `C1` | `C2` | Decision | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| Llama | 1 | L2/r3968 | 6.7536 | +300.8% | — | 1.0 | 1.0 | SINGLE_COMPONENT_DOMINANT | MAIN-TEXT SAFE |
| Mistral | 1 | L1/r2070 | 6.3380 | +287.1% (α=0.5: 6.6521, +301.4%, **exceeds full ablation**) | — | 1.0 | 1.0 | SINGLE_COMPONENT_DOMINANT (non-monotonic) | MAIN-TEXT SAFE |
| OLMo | 4 | L24/r269, L7/r269, L1/r269, L2/r269 | 0.0347 / 1.7054 / 2.7892 / 0.0593 | +1.4% / +68.3% / +111.8% / +2.4% | — | 0.6079 | 0.9795 | SINGLE_COMPONENT_DOMINANT (dominant row ≠ structural top) | MAIN-TEXT SAFE |
| Phi-3 | 6 | L2/r525, L2/r1693, L2/r1113, L4/r525, L4/r1113, L4/r1693 | 0.1102 / 0.1052 / 0.0218 / 0.0582 / 0.0073 / 0.0282 | +4.62% / +4.41% / +0.91% / +2.44% / +0.31% / +1.18% | 1044.5× / 996.9× / 206.3× / 551.2× / 68.9× / 267.4× | 0.3332 | 0.6511 | MULTI_COMPONENT_CANDIDATE | MAIN-TEXT SAFE |
| Qwen2.5 | 1 | L26/r458 | 0.01807 | +0.74% | — | 1.0 | 1.0 | SINGLE_COMPONENT_DOMINANT (weak absolute magnitude) | MAIN-TEXT SAFE |

OLMo structural-vs-causal dissociation (MAIN-TEXT SAFE, `results/e10_exact_uknorm_olmo.json` +
above):

| Layer | Structural `‖U_k‖_F`/layer-median | Structural rank | Causal `dNLL` | Causal rank |
|---:|---:|---:|---:|---:|
| 24 | 53.4× | 1st | 0.0347 | 4th (weakest) |
| 7 | 39.7× | 2nd | 1.7054 | 2nd |
| 1 | 39.2× | 3rd | 2.7892 | 1st (dominant) |
| 2 | 20.8× | 4th | 0.0593 | 3rd |

### 3.3 Encoder tomography (F0-F3, exact, `results/e10_encoder_fit_results.json`, `experiments/E9_mechanistic_tomography/{fit_results_dnabert2,baseline_regression_results}.json`, `results/e10b_phi3_fit_results.json`)

| Model | ε | F0 R² | F1 R² | F2 R² | F3 R² | F2→F3 rel. MAE improvement | Bootstrap 95% CI | Decision | Status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| DNABERT-2 | 0.5 | −0.023 | 0.324 | 0.517 | 0.888 | 54.7% | (0.0038, 0.0186), excl. 0 | PAIR_TERMS_REQUIRED | MAIN-TEXT SAFE |
| DNABERT-2 | 1.0 | −0.040 | 0.659 | 0.581 | 0.790 | 24.4% | (0.0614, 0.1144), excl. 0 | PAIR_TERMS_REQUIRED | MAIN-TEXT SAFE |
| MosaicBERT | 0.5 | — | — | 0.832 | 0.994 | 84.2% | (0.0809, 0.0895), excl. 0 | PAIR_TERMS_REQUIRED | MAIN-TEXT SAFE |
| MosaicBERT | 1.0 | — | — | 0.836 | 0.969 | 54.5% | (0.4607, 0.4845), excl. 0 | PAIR_TERMS_REQUIRED | MAIN-TEXT SAFE |
| ModernBERT | 0.5 | — | — | 0.529 | 0.916 | 58.8% | (0.1933, 0.2070), excl. 0 | PAIR_TERMS_REQUIRED | MAIN-TEXT SAFE |
| ModernBERT | 1.0 | — | — | 0.813 | 0.933 | 46.4% | (0.9950, 1.0509), excl. 0 | PAIR_TERMS_REQUIRED | MAIN-TEXT SAFE |
| Phi-3 (E10b) | 0.5 | — | — | 0.278 | 0.574 | +36.4% | (0.0057, 0.0109), excl. 0 (improving) | PAIR_TERMS_REQUIRED | MAIN-TEXT SAFE |
| Phi-3 (E10b) | 1.0 | — | — | −0.196 | −0.103 | **−2.5%** | (−0.0500, −0.0450), excl. 0 (**worsening**) | **MIXED_OR_UNRESOLVED** | MAIN-TEXT SAFE (report as split, not a clean class) |

Note: DNABERT-2's F0/F1 R² are populated (from E9's own fit); MosaicBERT/ModernBERT/Phi-3's
F0/F1 R² were not extracted into the figure source-data snapshot — available in the raw JSON
(`.by_epsilon.{eps}.F0.metrics.r2`, `.F1.metrics.r2`) if needed for a full table; not required
for any current figure or headline claim.

Phi-3 E10b diagnosis (descriptive, not a fitted higher-order model — MAIN-TEXT SAFE,
explicitly labeled descriptive per instruction): at ε=1.0, masks that fully ablate all three
of Phi-3's layer-2 rows together (`L2/r525`, `L2/r1693`, `L2/r1113`) show catastrophic damage
(`dloss` +9.25 to +10.24) while every mask containing only 2-of-3 stays in the +0.02 to +1.1
range. The top 3 of 15 fitted pairwise `Γ` coefficients (ε=1.0) are exactly the 3 pairwise
combinations of this same triplet, at 1.6-1.7× the next-ranked pair — independent
corroboration from raw responses and from fitted coefficients, but **no triple-interaction
term was fit**; this is reported as a limit of the licensed F0-F3 ladder on this basis, not
rescued with a higher-order model. Source: `experiments/E10b_phi3_tomography/E10B_SYNTHESIS.md`,
`results/e10b_phi3_tomography_responses.json`.

Known pair ranking (MAIN-TEXT SAFE, `fit_results_dnabert2.json.by_epsilon.{eps}.F3`):

| ε | Critical pair rank (of 45) | `Γ` | Runner-up `Γ` |
|---:|---:|---:|---:|
| 0.5 | 1 | 0.0772 | 0.0544 |
| 1.0 | 1 | 0.7387 | 0.5019 |

### 3.4 DNABERT-2 pretrained-MLM interaction (MAIN-TEXT SAFE, `experiments/E9_mechanistic_tomography/baseline_regression_results.json.dnabert2`)

| ε | `d_A` (L9/r264 alone) | `d_B` (L9/r294 alone) | `d_AB` (joint) | epistasis |
|---:|---:|---:|---:|---:|
| 0.5 | +0.00342 | −0.00100 | +0.05520 | +0.05278 |
| 1.0 | +0.02182 | +0.00640 | +2.04002 | **+2.01180** |

Baseline pretrained MLM loss: 4.6953 (256 hg38 windows, 16 batches, 4453 masked tokens, 1
seed). This reproduces a previously-reported figure (+2.0118) to 4 decimals from a
from-scratch environment (revision-pinned checkpoint `7bce263b15377fc15361f52cfab88f8b586abda0`,
freshly-downloaded hg38) — a genuine independent replication.

Singleton effects, all 10 basis rows, ε=1.0 (`fit_results_dnabert2.json.by_epsilon.1.0.singleton_effects_x_i`, basis order from `BASIS_FREEZE.md`):

| Index | Layer/Row | `x_i` (dloss) |
|---:|---|---:|
| 0 | L5/r603 | 0.03895 |
| 1 | L3/r86 | 0.00302 |
| 2 | L3/r399 | 0.00042 |
| 3 | L9/r264 | 0.02182 |
| 4 | L9/r294 | 0.00640 |
| 5 | L3/r603 | 0.07643 |
| 6 | L3/r641 | 0.00116 |
| 7 | L7/r603 | 0.15284 |
| 8 | L6/r603 | 0.39954 |
| 9 | L5/r86 | 0.02626 |

### 3.5 GENERator (MAIN-TEXT SAFE unless noted)

GC dose-response (`experiments/E9_mechanistic_tomography/baseline_regression_results.json.generator`):

| Row | α=0.0 | α=0.5 | α=1.0 | Span | N/condition |
|---|---:|---:|---:|---:|---:|
| 2371 (primary) | 0.2944 ± 0.0421 | 0.3863 ± 0.0475 | 0.3961 ± 0.0701 | 0.1017 | 24 |
| 1522 (secondary) | 0.3466 ± 0.0562 | 0.4131 ± 0.0743 | 0.3961 ± 0.0701 | 0.0665 | 24 |
| random control (5 rows) | 0.39599 | 0.39625 | 0.39605 | 0.00026 | 120 |

Row 2371 span ≈ 390× random-control span. Row 1522 is non-monotonic (rises at partial
suppression, falls below baseline at full ablation).

BOS/attention-sink (`results/mechanism/attention_sink_implicit_bias.json`, reproduced
2026-08-22, 40 hg38 windows, seed 42, 1 run):

| Metric | Value |
|---|---:|
| Attention mass @ position 0 | 37.92% |
| Uniform expectation | 1.15% |
| Fold over uniform | 33.0× |
| Fraction of (layer,head) pairs argmax @ pos 0 | 78.75% (of 38,400 layer-head observations) |
| Activation value @ pos 0 (real vs. dinuc-shuffled) | 375,361.3 both (ratio 1.0000 — **structurally guaranteed at pos 0 in a causal decoder, not independent evidence**; script's own caveat) |

Status: **MAIN-TEXT SAFE, association-only wording required.** Single run, not yet
seed-replicated — note this explicitly if used as a headline number.

### 3.6 NTv3

| Result | Value | Artifact | Status |
|---|---|---|---|
| Exact `q1` (structural outlier) | 0.3889 | `results/e7_legacy_reanalysis.json` | MAIN-TEXT SAFE (structural claim only) |
| Splice ablation ΔMCC (5 seeds) | −0.1185 ± 0.054, p=0.008 | `results/gue_multiseed_ntv3_splice.json` | **RETIRED (X-009)** — truncation-bug artifact (task-key fallback resolved `max_length=80` for a nucleotide-level tokenizer, truncating every splice window to its first 80bp). **Do not cite as current.** |
| Corrected splice ablation (MCC 0.86-0.91, SW-ablation effect −0.02pp) | claimed, not measured here | none | **PROVISIONAL / cannot support a plotted or main-text quantitative claim — no raw artifact of any kind exists in this repository for the corrected number.** |
| MAKE-PAIR construction | claimed "no effect" | none | **PROVISIONAL**, same reason |

**Per instruction: NTv3 currently cannot support ANY plotted or main-text quantitative causal
claim beyond the structural `q1` outlier fact.** The only causal-adjacent NTv3 artifact in
this repository is the retired X-009 result, which must not be presented as current.

---

## 4. Exact methods needed for manuscript writing

### 4.1 Models

| Model | HF repo ID | Pinned revision | Params | Enc/Dec | Architecture / FFN class | Down-proj module pattern | FFN packing | Tokenizer |
|---|---|---|---|---|---|---|---|---|
| Llama-7B | `huggyllama/llama-7b` | **NOT FOUND — unpinned anywhere in repo** | 7B | Decoder | `LlamaForCausalLM`/`LlamaMLP` (SwiGLU) | `model.layers.{i}.mlp.down_proj` | separate gate/up/down | UNKNOWN |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | **NOT FOUND — unpinned** | 7B | Decoder | `MistralForCausalLM`/`MistralMLP` (SwiGLU), identical layout to Llama | `model.layers.{i}.mlp.down_proj` | separate gate/up/down | UNKNOWN |
| OLMo-7B | `allenai/OLMo-7B-0724-hf` | **NOT FOUND — unpinned** | 7B | Decoder | Llama-style tensor path; no dedicated adapter in `uk_frobenius.py` | `model.layers.{layer}.mlp.{gate_proj,up_proj,down_proj}.weight` | separate gate/up/down | UNKNOWN |
| Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` | `f39ac1d28e925b323eae81227eaba4464caced4e` (shard SHA256 `b7492726c01287bf6e13c3d74c65ade3d436d50da1cf5bb6925bc962419d6610`) | mini (`hidden_size=3072`, `intermediate_size=8192`, 32 layers) | Decoder | `Phi3MLP` | `model.model.layers[i].mlp.gate_up_proj` (packed) / `...mlp.down_proj` | **packed** `gate_up_proj`, `.chunk(2,dim=-1)`, first half gate/second half up | UNKNOWN |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` (shard SHA256 `b5a2298dddcf228129975a9a271912a9f8dc817957deecde523e3154481ec3fb`) | 7B | Decoder | `Qwen2ForCausalLM`/`Qwen2MLP` (Llama-pattern) | `model.model.layers[i].mlp.{gate_proj,up_proj,down_proj}` | separate gate/up/down | UNKNOWN |
| GenomeOcean-4B | `DOEJGI/GenomeOcean-4B` (config still says `pGenomeOcean/GenomeOcean-4B`, org moved, redirect confirmed) | `2bed2fc3ed47c5f6955ba3e64563512c9b338dfb` (shard SHA256 `570ae8543a5b23850965d1812830c589d1eb16ae145482723333ea46104d89f3`) | 4B (`hidden_size=3072`, `intermediate_size=16384`, `num_hidden_layers=24`) | Decoder | `MistralForCausalLM` (`hidden_act: silu`) | `model.layers.{i}.mlp.down_proj` | separate gate/up/down | character/BPE, no fixed-length alignment needed |
| Evo2-7B | `arcinstitute/evo2_7b` | `bda0089f92582d5baabf0f22d9fc85f3588f6b58` (checkpoint SHA256 `c66645929dc1b9c631f5be656da8726f38946315dc9167000a615dd626fcecf4`) | 7B (`hidden_size=4096`, `inner_mlp_size=11008`, 32 layers) | Decoder (SSM/Hyena) | StripedHyena2 `ParallelGatedMLP`; for layers>0, activation replaced with `nn.Identity()` (no elementwise nonlinearity) | `blocks.{i}.mlp.l3` (down); `l1`=gate, `l2`=up | separate l1/l2/l3 | UNKNOWN |
| GENERator EUK | `GenerTeam/GENERator-v2-eukaryote-3b-base` | not globally pinned; one E9 run resolved `main`→`7dc01bccce5b65e15141170538afdc2ff09d8dde` (first time a revision was recorded for E9's purposes) | 3B, 30 layers | Decoder | `AutoModelForCausalLM`, Llama-shaped MLP | `model.layers.{i}.mlp.down_proj` | separate gate/up/down | **6-mer tokenizer** — sequence length must be a multiple of 6bp |
| DNABERT-2-117M | `zhihan1996/DNABERT-2-117M` | `7bce263b15377fc15361f52cfab88f8b586abda0` | 117M, 12 layers | Encoder (bidirectional, ALiBi) | custom remote-code `BertForMaskedLM` (Mosaic-BERT-derived) | `bert.encoder.layer.{i}.mlp.wo` (in=3072/out=768) | **packed** `gated_layers` (`Linear(hidden,2*intermediate)`), `BertGatedLinearUnitMLP` class | BPE (inferred from `_MAX_LEN` tuning comment) |
| NTv3 | `InstaDeepAI/NTv3_650M_pre` | **NOT FOUND — unpinned** | 650M, 12 layers (indices 0-11) | Encoder (bidirectional MLM) | custom `core.transformer_blocks`; no `mlp` submodule | `core.transformer_blocks.{i}.fc2` (down); `fc1` packed gate+up | **packed** `fc1` (`[12288,1536]`, 2×`fc2`'s 6144 d_ffn) — "which half is gate/up cannot be read off the shapes, and does not matter" (symmetric in the exact `c_{k,i}` formula) | **nucleotide-level, 1bp=1token** (confirmed) |
| MosaicBERT | `mosaicml/mosaic-bert-base` | `c89bbadc24278928f22bcdd7de6b61a5a2d08553` (found only in the raw detection JSON's `resolved_revision`, never written into any markdown) | 768/3072/12 layers | Encoder (bidirectional, ALiBi) | custom remote-code `BertForMaskedLM`, identical `BertGatedLinearUnitMLP` class to DNABERT-2 | `model.bert.encoder.layer[i].mlp.{gated_layers,wo}` | packed `gated_layers` | `bert-base-uncased` tokenizer (itself unpinned) |
| ModernBERT | `answerdotai/ModernBERT-base` | `8949b909ec900327062f0ebf497f51aef5e6f0c8` (same status: JSON-only, not in markdown) | 768/1152/22 layers | Encoder (bidirectional, alternating local/global attn + RoPE) | native `transformers` `ModernBertForMaskedLM` | `model.model.layers[i].mlp.{Wi,Wo}` | packed `Wi` (`Linear(hidden,2*intermediate)`, no bias), activation on first chunk half | native ModernBERT tokenizer |

Tokenizer type is genuinely undocumented in this repository for Llama, Mistral, OLMo,
Qwen2.5, and Evo2 — left UNKNOWN rather than assumed. No pinned HF revision exists anywhere
in the repo for Llama-7B, Mistral-7B, or OLMo-7B (repeatedly disclosed as an open gap in
`E5_dimensionality/GATE0_RESULTS.md`, `E6_cross_geometry/RESULTS.md`,
`MODEL_AND_BASIS_AUDIT.md`) — a manuscript Methods section should either pin these before
submission or explicitly disclose the gap.

### 4.2 Structural analysis — exact mathematical definitions

**Retired diagonal approximation** (`manuscript/src/uk_frobenius.py`), weights-only, no
forward pass:
```
||U_k||_F = sqrt( sum_i  W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2 )
c_{k,i}   = W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2      # per-hidden-unit contribution
```
with `W_gate: [d_ffn, d_model]`, `W_up: [d_ffn, d_model]`, `W_down: [d_model, d_ffn]`.
Granularity metrics from this decomposition: `top1_share = max_i c_i / sum_i c_i`,
`participation_ratio = (sum_i c_i)^2 / sum_i c_i^2`.

**Exact bilinear operator** (`E7_exact_dimensionality/spectral_lib.py`) — the object the
diagonal form above approximates, materialized in closed form, no cross-terms dropped:
```
U_k = sum_i  d_i * outer(g_i, u_i),   d_i = W_down[k,i],  g_i = W_gate[i,:],  u_i = W_up[i,:]
U_k = (d[:, None] * W_gate)^T @ W_up          # [d_model, d_model], float64
```
Its Frobenius norm equals `sqrt(sum_j sigma_j^2)` of `U_k`'s exact singular spectrum and is
asserted (by design and by test) to equal `||U_k||_F` above exactly — the exact and diagonal
forms report the *same* row-magnitude quantity, but the exact form additionally exposes the
*spectrum*, which the diagonal form cannot.
```
q1      = sigma_1^2 / sum_j sigma_j^2                              # top singular-energy share
PR_spec = (sum_j sigma_j^2)^2 / sum_j sigma_j^4                    # spectral participation ratio
stable_rank = 1.0 / q1                                             # reported, never primary
```
Computed via `torch.linalg.svdvals(U_k)` (singular values only, no vectors needed).

**What the diagonal form drops — cross-terms** (`E6_cross_geometry/cross_geometry_lib.py`):
```
K[i,j] = (W_gate[i,:].W_gate[j,:]) * (W_up[i,:].W_up[j,:])         # [d_ffn, d_ffn]
X[i,j] = d_i * d_j * K[i,j]
f_cross = (sum(X) - sum(diag(X))) / sum(X)          # fraction of squared mass in off-diagonal terms
```
`f_cross` is the fraction of the exact quadratic form's mass carried by cross-terms the
diagonal `c_{k,i}` decomposition ignores entirely. **Confirmed asymmetric bias**: E5's Gate-0
result found `f_cross` exceeds a locked 20% stop threshold for all 3 genomic models tested
(GENERator EUK 82.9%, DNABERT-2 50.7%, NTv3 20.7%) and none of the 3 NLP models tested
(6.7-19.3%) — every genomic model's diagonal PR overstates its exact `PR_spec` in the same
direction (EUK 4.3×, DNABERT-2 2.4×, NTv3 3.5×), which is why the diagonal approximation is
retired for structural claims (§7 item 4) but remains an acceptable low-cost calibration
proxy specifically for the NLP panel, where the bias is small.

**Candidate-row selection (activation detection)**: hooks every layer's down-proj-equivalent
module during a single forward pass over a fixed probe, records `out_max` (global max
activation) and each layer's median channel-max; `ratio = out_max / layer_median`; accepted
only if `ratio ≥ 5.0` (`RATIO_THRESHOLD = 5.0`), else an explicit structural null
(`null_reason = "global-max ratio {ratio} < threshold 5.0"`) — this is exactly how Evo2-7B
was ruled a structural null (ratio 2.22). Probe: ACTB CDS (504bp/84 tokens) for genomic
decoder models (GenomeOcean, Evo2); WikiText-2-raw-v1 test split (first 20 non-empty lines,
512-token truncation) for NLP-panel models measured this way (Qwen2.5, MosaicBERT,
ModernBERT) — no DNA probe used for the NLP-architecture-comparison arm (E8).

**Ordinary-row controls**: 5 seeded random rows per model at the candidate's own layer,
excluding the candidate; seed `numpy.random.SeedSequence(42).spawn(N)[panel_index]` with a
fixed per-model panel index (E8: MosaicBERT=0, ModernBERT=1; E10 decoders: Llama=0,
Mistral=1, OLMo=2, Phi-3=3, Qwen2.5=4).

### 4.3 Causal interventions

**Intervention object and formula** (identical code path for ablation and partial
suppression — `_scale_rows`/`scale_rows`, reused from `run_gue_ablation.py`'s
`_resolve_module`/`_save_row`/`_restore_row`):
```
alpha_i = 1 - epsilon * a_i
mod.weight.data[row_i, :] = saved_row_i * alpha_i
```
- `a_i ∈ {0,1}`: binary mask indicator — is row `i` targeted by this mask.
- `epsilon`: continuous intervention scale/dose, frozen values `{0.5, 1.0}`.
- `alpha_i=1.0`: untouched. `alpha_i=0.0` (`epsilon=1.0,a_i=1`): **full ablation**. `alpha_i=0.5`
  (`epsilon=0.5,a_i=1`): **partial suppression** — mechanically the *same* scaling operation
  at a different multiplier, not a qualitatively different procedure.
- Intervention point: at the down-projection matmul itself, on the row defining the linear
  map from the post-activation intermediate to one downstream coordinate — after the
  nonlinearity (SwiGLU/GLU), before the residual-stream addition and next norm layer.

**Output metrics**:
- Decoders (causal-LM NLL): `nll = F.nll_loss(log_softmax(shift_logits), shift_labels, reduction="sum")`,
  reported as mean per-token NLL over all windows. Secondary: perplexity `exp(mean NLL)`,
  logit KL vs. baseline, next-token entropy.
- Encoders / DNABERT-2 (masked-LM "dloss"): `dloss = MLM_loss(intervened) − MLM_loss(baseline)`,
  token-weighted mean over masked positions; **positive = worse** (ablation-positive
  convention, inverted vs. accuracy-based GUE scripts elsewhere in the repo).
- Epistasis (confirmed exact code): `epistasis = d_AB − (d_A + d_B)`.

**Baseline**: the unmodified-weights forward pass, computed via the identical measurement
function before any row is scaled.

**Corpus / windows / sequence length**:
- E10 decoders (ARM A): `wikitext`/`wikitext-2-raw-v1`/`test`; N=100 windows (shuffled
  concatenated non-empty lines up to `max_length` tokens), 512-token context, batch size 8,
  float32, `torch.manual_seed(42)`, window-shuffle seed 42.
- E10 encoders (ARM B): same WikiText-2 corpus/split, N=256 windows, `mask_prob=0.15`, seed
  42 (one fixed mask realization reused across every condition), 512-token max length, batch
  size 16, float32.
- E9 DNABERT-2: hg38 (not WikiText) — 256 windows, 600bp each, `random_262kb.bed`,
  `mask_prob=0.15`, seed 42, 16 batches, `max_len=256`.

**Aggregation / statistics**: token/batch-weighted mean (not naive batch mean — per-batch
`(sum,n)` pairs retained so bootstrap resampling can reweight exactly without rerunning the
model). Bootstrap: `N_BOOT=5000`, resampling **batch indices with replacement**, applied
*jointly/paired* to both the held-out response and the baseline before baseline-subtraction
(the exact fix documented in D-026 — an earlier version resampled baseline and held-out
response independently, producing a spurious ~4.7-unit offset from mixing raw-loss and
baseline-subtracted scales; point estimates were unaffected, only the CI was wrong).

### 4.4 Tomography (F0-F3) exact procedure

**Frozen basis** (DNABERT-2, n=10): the pre-existing high-gain ensemble underlying prior
ablation work, `results/super_weight_index.json["dnabert2"]["results"]`, ordered by `out_max`
descending — no row added/dropped/reordered based on any E9 measurement. Basis order:
`(5,603),(3,86),(3,399),(9,264),(9,294),(3,603),(3,641),(7,603),(6,603),(5,86)` — indices 3,4
are the critical pair (L9/r264, L9/r294), included with no special treatment during fitting.
GENERator EUK's basis has only 2 real candidates (rows 2371, 1522) — too few for F0-F3;
scoped instead to a 1D dose-response (padding rejected as "fishing-adjacent").

**Mask pools** (disjoint, seed `20260822`): 10 singletons (exhaustive one-hot); 78 fit-pool
masks (densities ρ≈0.25:22, ρ≈0.5:34, ρ≈0.75:22); 20 calibration; 20 held-out (same density
split, 6/8/6). An initial 40-mask fit pool gave a rank-deficient (40/55) lifted design;
corrected to 78 before any response was measured. Final lifted design: 78×55, full rank,
`cond(XᵀX)=4521`.

**Lifted design matrix** (exact code):
```python
PAIRS = list(itertools.combinations(range(N), 2))     # N=10 -> 45 pairs
def lifted(a):
    inter = np.array([a[:, i] * a[:, j] for i, j in PAIRS]).T
    return np.hstack([a, inter])                       # 10 main + 45 pair = 55 columns
```

**Observer families** (each fit separately per epsilon, no pooling across scales):
- **F0** (singleton additive): `y_hat(a) = sum_i a_i * x_i`, `x_i` = measured singleton effect, no fitting.
- **F1** (scalar-calibrated): `y_hat(a) = g * sum_i a_i * x_i`, `g` fit by least squares on the calibration pool only.
- **F2** (jointly-fit additive): `y_hat(a) = sum_i beta_i * a_i`, ridge on the fit pool, penalty λ selected by minimizing MSE on the calibration pool from a fixed grid `{1e-3,...,1000}`.
- **F3** (lifted main+pair): `y_hat(a) = sum_i beta_i a_i + sum_{i<j} Gamma_ij a_i a_j`, ridge on the 55-column lifted design, same fit/calibration/held-out split.

**Metrics** (exact code):
```
MAE  = mean(|resid|);  RMSE = sqrt(mean(resid^2))
R²   = 1 - SS_res/SS_tot
normalized_MAE = MAE / (max(y_held) - min(y_held))       # MAE normalized by held-out response span
```

**Bootstrap CI for F2→F3 MAE improvement**: 5,000 resamples of batch indices (with
replacement), applied paired to held-out response and baseline before subtraction; CI =
[2.5th, 97.5th] percentile of `MAE_F2 − MAE_F3` across resamples; "excludes zero" = both
percentiles share the same sign.

**PAIR_TERMS_REQUIRED decision rule** (locked prereg, exact thresholds):
- ADDITIVE_ADEQUATE: held-out R² ≥ 0.90 **and** normalized MAE ≤ 0.10.
- CALIBRATION_SUFFICIENT: F1 meets that bar **and** F3 gives no meaningful improvement.
- PAIR_TERMS_REQUIRED: F0/F1/F2 fail adequacy or show structured residual **and** F3
  improves held-out MAE by ≥10% relative to F2 **and** that improvement's bootstrap CI
  excludes zero **and** the lifted design is confirmed identifiable (full rank).
- Otherwise: explicit intermediate case, not forced into the nearest bucket (this is exactly
  Phi-3's ε=1.0 MIXED_OR_UNRESOLVED outcome — F3 is reliably *worse* than F2, a genuinely
  different case from either adequacy class).

### 4.5 DNABERT-2 exact pretrained-MLM evaluation setup

- Checkpoint: `zhihan1996/DNABERT-2-117M` @ `7bce263b15377fc15361f52cfab88f8b586abda0`, no
  fine-tuning, no task head (`AutoModelForMaskedLM`); Triton flash-attention incompatibility
  patched by forcing `flash_attn_qkvpacked_func=None` (eager attention).
- Masking: this project's own fixed-realization masking (not DNABERT-2's native pipeline),
  `mask_prob=0.15`, one fixed mask realization reused across every condition so deltas are
  paired; selected positions replaced with `mask_token_id`, labels `-100` elsewhere.
- Data: hg38 (`data/reference/hg38/hg38.fa`, freshly downloaded from UCSC),
  `data/regions/hg38/random_262kb.bed`; 256 windows × 600bp, 16 batches, 4,453 masked tokens
  total (seed 42).
- Epistasis definition (confirmed exact): `epistasis = d_AB − (d_A + d_B)`; reproduces a
  prior figure (+2.0118) to 4 decimals from this from-scratch environment.
- E9's 10-row basis and critical-pair indices: see §4.4.

### 4.6 NTv3 architecture facts

- `InstaDeepAI/NTv3_650M_pre`, no pinned revision anywhere in the repo.
- `down_proj` equivalent: `core.transformer_blocks.{i}.fc2`; paired gate/up module is `fc1`
  at the same block, packed (`[12288,1536]`, 2×`fc2`'s 6144 `d_ffn`) — "which half is gate/up
  cannot be read off the shapes, and does not matter" (symmetric in the exact `c_{k,i}`
  formula).
- 12 transformer blocks (indices 0-11).
- Tokenizer: nucleotide-level, 1bp=1token (confirmed).
- Truncation bug — exact `_MAX_LEN` dict (`scripts/evaluation/run_gue_ablation.py`):
```python
_MAX_LEN = {
    "EMP": 128, "EPI": 128, "fungi": 512, "mouse": 30, "tf": 30,
    "prom_core_all": 20, "prom_core_notata": 20, "prom_core_tata": 20,
    "prom_300_all": 70, "prom_300_notata": 70, "prom_300_tata": 70,
    "reconstructed": 80,   # splice — the value responsible for X-009
    "covid": 256, "species_40": 512, "species_20": 512,
}
```
  The splice task path resolves to key `"reconstructed"` → 80 tokens; the launch script
  (`submit_ntv3_splice_multiseed.sh`) never passed `--max_length`, so 80 fired instead of the
  intended 512 default or the corrected `max_length=400`. Corrected retrain figures (adopted
  per D-024, no local artifact): MCC 0.86-0.91, SW-ablation effect −0.02pp.

---

## 5. Dataset and corpus inventory

### 5.1 hg38 reference genome

- **Source**: UCSC, `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz`,
  standard reference build, no login required.
- **Local path**: `data/reference/hg38/hg38.fa` (gitignored, not committed — 2,980,020,224
  bytes decompressed, 455 FASTA records, full primary assembly + alt/random contigs). Index:
  `data/reference/hg38/hg38.fa.fai`.
- **Note**: original colleague scripts default to `/data/nvidia/data/hg38/hg38.fa`, confirmed
  absent on this filesystem; every current script invocation explicitly passes the local path
  instead (documented fix, no science changed).

### 5.2 Bed files (`data/regions/hg38/`, per `data/regions/hg38/manifest.json`)

| File | Purpose | Rows | Generation |
|---|---|---:|---|
| `random_262kb.bed` | random window sampling (E9 GENERator/DNABERT-2 measurements) | 50,000 | `seq_len_bp=262144`, seed=1, `exclude_chroms=["chrM"]`, 97 eligible chromosomes |
| `promoters_262kb.bed` | promoter windows | 218,702 | from GENCODE v44 GTF (`transcript` feature), dedup=true |
| `enhancers_ccre_262kb.bed` | enhancer windows | 960,225 | from ENCODE cCREs (`GRCh38-cCREs.bed`), classes `ELS,dELS,pELS`, dedup=true |

Sources: GENCODE v44 (`https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz`);
ENCODE cCREs (mirror URL not resolved to one specific link in the manifest). Manifest
timestamp 2026-02-04. None of the promoter/enhancer bed files are used by the current
4-figure E7-E10b/E9 scope (`random_262kb.bed` is the only one in active use for the headline
results in §3) — flagged in case Discussion/Methods references a broader context set.

### 5.3 WikiText-2 (E10 decoder + encoder measurements)

Exact identifier: `load_dataset("wikitext", "wikitext-2-raw-v1", split="test")`.

- **ARM A (decoders, causal-LM NLL)**: N=100 windows (shuffled, concatenated non-empty test-
  split lines up to `max_length` tokens), 512-token context, batch size 8, float32,
  `torch.manual_seed(42)`, window-shuffle seed 42.
- **ARM B (encoders, MLM loss)**: N=256 windows, `mask_prob=0.15`, mask-generator seed 42
  (one fixed mask realization reused across every condition), 512-token max length, batch
  size 16, float32. Tokenizers: `bert-base-uncased` for MosaicBERT, native tokenizer for
  ModernBERT.
- E7/E8's own activation-detection forward passes for the NLP-panel models measured this way
  (Qwen2.5, MosaicBERT, ModernBERT) use the same corpus/split: first 20 non-empty test-split
  lines, newline-joined, truncated to 512 tokens. **Not** used for GenomeOcean/Evo2, which
  use the ACTB DNA probe instead (§5.5).
- Labels: not used anywhere (unsupervised LM-loss objectives only, both arms).

### 5.4 GUE benchmark — splice/reconstructed task

- **Location**: `/work/11034/atzanakak/GUE/GUE/splice/reconstructed/{train,dev,test}.csv`.
  `test.csv`: **4,562 examples** (4,563 lines minus header), confirmed by direct line count.
- **Label distribution** (3 classes): label 0 → 956, label 1 → 1,025, label 2 → 2,581.
- Format: `sequence,label` CSV, 400nt sequences.
- **Caveat (important, load-bearing for §7's contested claims)**: E9's own provenance note
  states the GUE path its scripts *default to* (`/data/nvidia/data/gue/GUE`) is absent on
  this filesystem, and that the fine-tuned **multiseed** splice checkpoints
  (`results/gue_checkpoints_multiseed/dnabert2_reconstructed/seed_{0,1,2}/model_state.pt`) do
  not exist — only a single non-multiseed checkpoint
  (`results/gue_checkpoints/dnabert2_reconstructed/model_state.pt`) is present. This is
  exactly why the fresh splice/codominance reproduction runs referenced in §7 are single-seed
  (n=1), not the originally-claimed n=5 protocol. The `/work/11034/atzanakak/GUE/GUE/...`
  path found and used in this session's reproduction runs is a *different* filesystem
  location than what the scripts default to internally — used via an explicit path override.
- **`_MAX_LEN` truncation dict** (`scripts/evaluation/run_gue_ablation.py`) — quoted in full
  in §4.6. Splice/reconstructed resolves to 80 tokens (correct for DNABERT-2's BPE tokenizer
  at this task; the same value is a bug for NTv3's nucleotide-level tokenizer, §7 item 3).

### 5.5 GENERator generation-prompt setup (GC dose-response, §3.5)

- **Prompt count**: 24, drawn from real hg38 windows via `read_windows(HG38_FASTA, HG38_BED, 24, 120+50, rng)` — 170bp windows drawn, first 120bp used as the prompt.
- **Generation**: `max_new_tokens=64`, `do_sample=True, top_k=50, temperature=1.0`,
  `torch.manual_seed(seed+i)` per prompt `i`, `seed=42`.
- **Random controls**: 5 non-SW rows, `random.Random(42).sample(...)` → rows
  `[2621, 456, 102, 3039, 1126]`.
- **GC-fraction computation**: computed **over the generated continuation only** (tokens
  after the prompt, decoded via `o[0][ids.shape[1]:]`), filtered to A/C/G/T characters only,
  uppercased, minimum-length filter of 10 characters applied. `gc_frac(s) = (count(G)+count(C)) / max(count(ACGT chars), 1)`.
  **The prompt itself is excluded from the GC calculation** — a methodologically important
  detail for the Methods section.
- Model/coordinate: `GenerTeam/GENERator-v2-eukaryote-3b-base`, layer 4, primary row 2371,
  secondary row 1522, module `model.layers.{i}.mlp.down_proj`.

### 5.6 Probe sequences (`probes/dna_probes.py`)

| Probe key | Length | Source | Used for |
|---|---:|---|---|
| `actb_500` (= `DEFAULT_PROBE`, also aliased `human_promoter`/`atg_context`) | 504bp / 84 tokens | NCBI NM_001101.5 CDS (human ACTB β-actin), first 504bp | Default detection/sanity checks; used for GenomeOcean-4B and Evo2-7B's E7 Phase-1 activation-detection sweep |
| `actb_full` | 1128bp / 188 tokens | full ACTB CDS | destruction/ablation tests, eukaryote |
| `pseudomonadota` | 1056bp / 176 tokens | NC_002947.4 (*Pseudomonas putida* KT2440), region 5070808-5071866 | destruction/ablation tests, prokaryote (not in current 4-figure scope) |
| `phage` | 2178bp / 363 tokens | T4 phage rIIA CDS | not in current 4-figure scope |
| `poly_a` | 504bp | homopolymer | sanity-check outlier (expect high perplexity) |

All sequences constrained to be divisible by 6bp for GENERator's 6-mer tokenizer.
**Qwen2.5-7B, MosaicBERT, and ModernBERT do NOT use a DNA probe** — their activation-detection
sweeps use the WikiText-2 construction in §5.3 instead (they are NLP-architecture-comparison
models, not genomic detection targets).

---

## 6. Provenance and preregistration

### 6.1 Preregistration versions (confirmed from `manuscript/docs/prereg/LOCKS.jsonl` and experiment RESULTS.md files read this session)

| Experiment | Prereg file | Locked | Notes |
|---|---|---|---|
| E7 (exact operator dimensionality) | `PREREG_exact_operator_dimensionality.md` | yes | 4-model confirmatory panel (Phi-3, Qwen2.5, Evo2, GenomeOcean); Evo2 returned Phase-1 null |
| E8 (encoder/decoder) | `PREREG_encoder_decoder_dimensionality.md` | yes, sha256 `7a5dc793...` | Reuses E7's frozen detection protocol; MosaicBERT/ModernBERT |
| E9 (DNABERT-2/GENERator tomography) | `PREREG_mechanistic_tomography_E9.md` | yes, `10de1694...`, 2026-08-22T20:55:26Z | DNABERT-2 full F0-F3 ladder; GENERator scoped to 1D dose-response only (basis too thin, D-025) |
| E10 v1 (decoder causal, ranked by `q1`) | superseded, see below | — | **v1 ranked OLMo/Phi-3 multi-row sets by `q1` — an invalid selection criterion** (see §6.2) |
| E10 v2 (decoder + encoder causal, corrected) | `PREREG_E10_nlp_architecture_causal_v2.md` | yes, `3e4b991d...` | Ranks decoder rows by exact `‖U_k‖_F`/layer-median instead; decision thresholds `C1>0.5`, control-normalized `>3.0` |
| E10b (Phi-3 pair tomography) | `PREREG_E10b_phi3_tomography.md` | yes, `098fd52c...`, 2026-08-23T03:15:53Z | Only decoder E10 flagged eligible (MULTI_COMPONENT_CANDIDATE); result MIXED_OR_UNRESOLVED |

### 6.2 The E10 `q1`-selection mistake and its correction (confirmed, load-bearing methodological fact)

E10's first version (v1) ranked OLMo's and Phi-3's multi-row structural candidate sets by
`q1` to decide which row to treat as the "top" structural candidate for the causal
intervention design. **This was methodologically invalid**: `q1` is scale-invariant
(rank-1-ness), not a magnitude/gain measure, so ranking by it does not identify the
structurally largest row. `PROTOCOL_CORRECTION_01.md` documents the fix: v2 ranks by exact
`‖U_k‖_F` relative to layer median instead. Consequences documented:
- **No causal measurement was ever run under v1** — the correction was made before any
  causal intervention was measured, so no v1-tainted causal result exists to retract.
- OLMo's top-1 row by the corrected criterion is still L24/r269 (53.4× layer median vs. the
  v1 `q1`-based ordering, which also happened to pick L24 — same top pick, invalid
  justification). The *ordering* of the remaining 3 rows changed: L7 moves above L1 under
  the corrected criterion, where `q1` had placed it last.
- Phi-3's top-1 row is unchanged (L2/r525), but v1's `q1`-based ranking placed L4/r1113
  *last* despite it being the 5th-largest (of 6) by exact magnitude, and L4/r1693 4th despite
  being the *smallest* — both corrected in v2.
- **All causal measurements in the current Figure 2 were performed only after this
  correction** (E10 v2, prereg locked before any row's causal effect was measured).

### 6.3 Provenance table for headline claims (see §3 for full numeric detail; this table adds prereg/decision linkage)

| Claim (§3 ref) | Artifact | Generating script | Analysis script | Prereg/Decision ID | Checkpoint revision | Status |
|---|---|---|---|---|---|---|
| Structural `q1`/`PR_spec`/`‖U_k‖_F`, 12-model panel | `results/e7_*.json`, `e8_*.json` | `run_phase1_detection.py`, `run_legacy_reanalysis.py`, `run_detection_and_spectral.py` | `spectral_lib.py` | E7/E8 preregs (§6.1) | see §4 (pending) | MAIN-TEXT SAFE |
| Decoder singleton causal spectrum | `results/e10_decoder_concentration.json` | E10 measurement scripts | `e10_lib.py` | `PREREG_E10_nlp_architecture_causal_v2.md`, D-025/D-026 (E9-adjacent) | see §4 (pending) | MAIN-TEXT SAFE |
| Encoder F0-F3 (MosaicBERT/ModernBERT) | `results/e10_encoder_fit_results.json` | `run_encoder_tomography.py` | `fit_encoder_observers.py` | same prereg | — | MAIN-TEXT SAFE |
| Phi-3 E10b tomography | `results/e10b_phi3_fit_results.json`, `..._responses.json` | `run_phi3_tomography.py` | `fit_phi3_observers.py` | `PREREG_E10b_phi3_tomography.md` | — | MAIN-TEXT SAFE, reported as split |
| DNABERT-2 pretrained-MLM epistasis + F0-F3 | `experiments/E9_mechanistic_tomography/{baseline_regression_results,fit_results_dnabert2,dnabert2_mask_responses}.json` | `run_baseline_regression.py`, `run_dnabert2_measurements.py` | `run_fit_observers.py` | `PREREG_mechanistic_tomography_E9.md`, D-025/D-026 | `7bce263b15377fc15361f52cfab88f8b586abda0` | MAIN-TEXT SAFE |
| GENERator GC dose-response | same `baseline_regression_results.json` (`.generator`) | same | same | same | — | MAIN-TEXT SAFE |
| GENERator BOS/attention-sink | `results/mechanism/attention_sink_implicit_bias.json` | `scripts/mechanism/run_attention_sink.py` (hardcoded hg38 path patched 2026-08-22, no science changed) | — | not separately preregistered (reproduction of a colleague-branch analysis) | — | MAIN-TEXT SAFE, association-only wording |
| DNABERT-2 splice-accuracy pair ablation | `results/mechanism/pairwise_epistasis_dnabert2_reconstructed.json` (single seed) | `scripts/evaluation/run_sw_pairwise_epistasis.py` (GUE_ROOT path patched 2026-08-22) | — | not preregistered | `dnabert2_reconstructed` checkpoint (single, no seed variants) | **CONTESTED — see §7** |
| DNABERT-2 codominance | `results/mechanism/codominance_break_dnabert2.json` (single seed) | `scripts/mechanism/run_codominance.py` | — | not preregistered | same checkpoint | **CONTESTED — see §7** |
| NTv3 splice ablation | `results/gue_multiseed_ntv3_splice.json` | `run_gue_multiseed.py` (truncation bug) | — | not preregistered | — | **RETIRED (X-009)** |

Results still backed only by prose, not raw data, as of 2026-08-22 (authoritative absence
record: `manuscript/docs/MISSING_COLLEAGUE_ARTIFACTS.md`): GENERator sequence-quality/
specificity during GC intervention; NTv3 corrected splice ablation and MAKE-PAIR; PROK
corrected hexamer/GC-cost results (not in current 4-figure scope); quantization Q2/Q4
empirical results (not in current 4-figure scope).

---

## 7. Retired, corrected, disputed, or unsafe claims

| # | Claim | Why unsafe | Corrected replacement |
|---|---|---|---|
| 1 | Old DNABERT-2 catastrophic splice-accuracy pair magnitude (critical pair L9/r264+r294 joint effect **−33.76pp**) | Previously adopted from a colleague report with **no raw artifact** (D-024). A fresh, code-verified, single-seed reproduction now exists (`results/mechanism/pairwise_epistasis_dnabert2_reconstructed.json`) and gives **−0.42pp** — ~80× smaller. The old number is also internally implausible on its face: it exceeds this repo's own established full-10-row-ensemble effect (−26.84pp), i.e. it claims 2 of 10 rows damage the model more than all 10 removed together. The new measurement's strongest epistatic pair is a *different* pair (L3/r86+L3/r399, −2.10pp); the named pair ranks lower (−0.53pp) in this remeasurement. | **None currently established.** Neither the old nor the new (single-seed) number should be presented as settled — flagged CONTESTED, pending either a multi-seed reproduction or explicit author adjudication. The pretrained-MLM-endpoint version of the same pair (epistasis +2.0118, held-out validated via F0-F3) IS safely established and is a *different* claim on a *different* endpoint — do not conflate the two in manuscript text. |
| 2 | Old DNABERT-2 codominance magnitude (ratio sweep epistasis **−33.63→−0.66pp**) | Same status as #1: no raw artifact when adopted; fresh single-seed reproduction (`results/mechanism/codominance_break_dnabert2.json`) gives **−0.53→−0.04pp**, ~60× smaller, though the *qualitative direction* (epistasis shrinks toward zero as the ratio moves away from co-dominance) is reproduced. | **Direction reproduced, magnitude contested.** Do not cite the old magnitude. The new single-seed magnitude is real but weak evidence (n=1 vs. the claimed n=5 protocol) — cite only with that caveat, or omit the magnitude and state the qualitative direction only. |
| 3 | NTv3 splice-ablation 5-seed result (ΔMCC=−0.119±0.054, p=0.008) | **RETIRED (X-009).** Produced under a confirmed truncation bug: task-key fallback resolved `max_length=80` for NTv3's nucleotide-level tokenizer (tuned for DNABERT-2's BPE tokenizer instead), truncating every splice window to its first 80bp — before the splice junction the task is about. | A corrected re-fit is *claimed* (MCC 0.86-0.91, SW-ablation effect −0.02pp, i.e. a null result) but has **no raw artifact anywhere in this repository**. Cannot currently support any plotted or main-text quantitative claim — see §3.6. |
| 4 | Old approximate/diagonal `U_k` (Hadamard `c_{k,i}` decomposition) as the dimensionality metric | E5's own Gate-0 result showed the diagonal approximation's cross-term fraction exceeds a 20% stop threshold for all 3 genomic models tested (GENERator EUK 82.9%, DNABERT-2 50.7%, NTv3 20.7%) and none of the 3 NLP models (6.7-19.3%) — i.e. the approximation specifically overstates dimensionality for genomic models, non-uniformly across domain. The old diagonal PR numbers (DNABERT-2 3.64, EUK 4.56, NTv3 22.7, Evo1 122, PROK 2192 vs. NLP 1.02-1.24) are retired as **X-007**. | Superseded by the exact `q1`/`PR_spec` metric (E7/E8, §3.1) — a genuine SVD of the constructed bilinear operator, not a per-coordinate approximation. **The diagonal approximation is not used for any final structural claim in the current figure set.** It remains acceptable *only* for the NLP calibration panel (Fig. 1A, E1), where E5 showed the approximation's cross-term fraction is low enough (6.7-19.3%) to be a reasonable proxy — and even there it must not be plotted on the same axis as the exact metric (confirmed numerically distinct: Llama-7B diagonal `uk_norm`=123.64 vs. exact `frob_norm`=137.62 for the identical row). |
| 5 | Evo1 numerical-saturation / "pinned at exactly 2^24" claim | The literal "pinned at 2^24" claim was the colleague's own early, since-self-retracted hypothesis; their own later fp64 adjudication found the real plateau at 1.75×2^24. Not in current 4-figure scope (Evo1/PROK are not part of the E7-E10b/E9 four-figure structure), but flagged here per instruction to inspect it. | The corrected 1.75×2^24 value is accepted as supplementary detail (falls inside this repo's own independently-measured range, ~4.2e6-3e7). Not a headline claim in the current 4-figure structure. |
| 6 | Any retracted NTv3/Evo1 mechanism claims beyond #3/#5 | See `CLAIMS_LEDGER.md` X-001 (shadow redundancy — contradicted by this repo's own pruning sweep), X-003 (DNABERT-2 C≈0 stays retired, unaffected by the KL-noise-floor retraction), X-006 (a nondeterministic-Triton-kernel noise floor mistaken for a measurement). None of these are in the current 4-figure scope. | See `CLAIMS_LEDGER.md` for each item's specific corrected status. |
| 7 | Any structural/systemic decomposition that no longer survives | X-007 (see #4). C-032 (the old "genomic vs. NLP granularity" domain-general contrast) is retired — the exact metric shows GENERator EUK's `PR_spec` sits inside the published-NLP range, contradicting the old domain-general claim. | Superseded by C-034 (the heterogeneous, encoder/decoder-organized pattern in §3.1) — explicitly NOT a clean domain split (NTv3 outlier, DNABERT-2 intermediate, ModernBERT thin margin, GENERator EUK inside the NLP range). |
| 8 | Any claim that `q1` predicts causal importance | Never established by any experiment in this repository; explicitly tested and not supported — Phi-3 has a *lower* median `q1` than Llama/Mistral yet shows *weaker* structural concentration in a direction that, if anything, argues against a naive "q1 causes concentration" story (though n is far too small to treat this as positive evidence either). | No replacement claim — `q1` is a concentration/rank-1-ness measure only, never a causal-importance predictor, in any figure or claim in this repository. |
| 9 | Any claim that architecture fully determines causal organization | Explicitly disproven within this repo's own panel: Phi-3 (decoder) requires pair terms at ε=0.5, contradicting "decoders are additive"; OLMo shows structural-vs-causal rank dissociation within one model. | Replacement framing: architecture *associates with* causal organization in the tested panel (4/5 decoders single-component, 2/2+1-split encoders/eligible-decoder pair-requiring) — descriptive, not deterministic. |
| 10 | Any claim of GENERator causal control over the BOS attention sink | Only association/co-occurrence has been measured (Fig. 4B) — no intervention was performed that manipulates the sink independently of the high-gain row's own activation, and the source analysis itself notes a causal-decoder structural confound (position 0 attends only to itself). | Safe replacement: "BOS-centered attention and activation co-occur with the high-gain pathway" — never "causes." |
| 11 | Any implied continuous/linear GC steering claim (GENERator) | Only 3 dose points exist by design (α∈{0,0.5,1.0}); row 1522 (secondary) is explicitly non-monotonic, direct evidence against a simple steering-knob reading. | Safe replacement: "large, directionally consistent causal sensitivity for row 2371," not "validated graded/linear steering." |

---

## 8. Controls and falsification evidence

| Control | Where used | What it rules out | What it does NOT rule out |
|---|---|---|---|
| Random-row controls | E9 GC dose-response (5 rows), E10 decoder same-layer controls (5/model), attention-sink implicit-bias dinuc-shuffle check | That the observed effect is a generic consequence of perturbing *any* row/position rather than the specific detected one (GC span ≈390-4200× random-control span across models; decoder target rows clear control-normalized threshold by 69-1045×) | Whether the *specific* detected row is the uniquely optimal one to have detected (no exhaustive alternative search was performed) |
| Same-layer ordinary-row controls | Fig. 1D (E8, MosaicBERT/ModernBERT); Fig. 2/S1b (E10 decoders) | That the detected high-gain row is unremarkable relative to its own layer (control `q1` 0.024-0.049 vs. candidate 0.48-0.90; control causal effects cluster tightly near the control median, well-separated from target rows) | Generalization to models without measured same-layer controls (DNABERT-2, NTv3) |
| Held-out mask splits (tomography) | E9 DNABERT-2 F0-F3, E10 encoder F0-F3, E10b Phi-3 F0-F3 | That an observer family's apparent fit quality reflects overfitting to the same masks it was calibrated on (fit/calibration/held-out pools are disjoint, seeded) | Basis-independence (all tomography is stated as being *in the declared residual-channel basis*, not a claim about intrinsic causal dimension) |
| Bootstrap 95% CIs | F2→F3 MAE improvement, all tomography experiments | Whether an apparent improvement could plausibly be zero/noise (CI excludes zero in every reported PAIR_TERMS_REQUIRED case, including Phi-3's negative-direction ε=1.0 result) | Multiple-comparisons correction across models (not applied; each model's CI is reported on its own terms) |
| Noise-floor/determinism checks | DNABERT-2 impulse assay (N-004, N-007 — not in current 4-figure scope but methodologically load-bearing for the repo's broader discipline) | That a measured effect isn't actually a nondeterministic-kernel artifact (Triton fp16 attention path found noise-dominated; fixed by forcing eager attention) | Nothing directly in the current 4-figure results (this check predates and is orthogonal to E7-E10b/E9) |
| Exact-operator correction (vs. diagonal approximation) | All of Fig. 1, §7 item 4 | That the old diagonal-PR domain-general contrast was an artifact of the approximation dropping cross-terms (E5 Gate-0: cross-term fraction 20.7-82.9% for genomic models vs. 6.7-19.3% for NLP) | Whether the exact metric itself has its own unmeasured biases (not tested) |
| Structural-positive / causal-negative examples | OLMo (Fig. 2B, within-model dissociation); NTv3 (structural `q1` outlier, but no causally-established phenotype in current scope, §3.6) | That structural prominence guarantees causal importance (OLMo: structurally-largest row is causally weakest of its own set) | A general claim that structure never predicts causal importance (only that it doesn't reliably, in these specific cases) |
| Independent architecture replications | 2/2 encoders (MosaicBERT, ModernBERT) both PAIR_TERMS_REQUIRED at both scales | That the DNABERT-2-style interactional phenotype is a one-off (n=1) result — it replicates in 2 architecturally distinct text encoders | That it generalizes to encoders broadly (n=2, both text; no genomic encoder besides DNABERT-2 itself has been tomographed) |

---

## 9. Limitations (evidence-grounded)

- **Limited number of architectures.** The causal-organization panel (Fig. 2) covers 5
  decoders and 2 encoders (+1 decoder with its own deeper tomography, Phi-3/E10b). This is
  not enough to support a population-level claim about "decoders" or "encoders" in general —
  every finding is reported as a descriptive pattern in this tested panel, per §7 item 9.
- **Incomplete structural/causal candidate coverage for K=1 models.** Llama, Mistral, and
  Qwen2.5 each had exactly one pre-existing legitimate high-gain candidate row in this
  repository — there is no alternative candidate set to compare against, so their
  SINGLE_COMPONENT_DOMINANT classification is not itself informative about whether a
  multi-row structure was considered and rejected; it reflects the absence of any known
  alternative.
- **Finite intervention strengths.** All causal interventions use a small, fixed grid
  (`ε∈{0.5,1.0}` for tomography; `α∈{0,0.5,1.0}` for GENERator's GC dose-response) — no dense
  dose-response curve exists for any model, and GENERator's steering claim is explicitly
  scoped to "causal sensitivity," not a validated continuous function.
- **Tomography only up to second order.** F3 fits main effects plus pairwise interaction
  terms; no triple-or-higher-order model was fit anywhere in this repository, per explicit
  methodological constraint. Phi-3's ε=1.0 MIXED_OR_UNRESOLVED result is a direct
  consequence: a diagnosed three-way redundancy break among its 3 layer-2 rows is visible in
  the raw data and in the fitted pairwise coefficients, but no model in the licensed ladder
  can represent it, and none was fit to try.
- **Phi-3 failure at full ablation.** The one decoder given its own deeper tomography
  produces a genuinely unresolved result at ε=1.0 (every observer family has negative
  held-out R², and the pairwise model is reliably *worse* than the additive one) — this
  should be reported as a real finding about the limits of the tested model family, not
  smoothed into either PAIR_TERMS_REQUIRED or ADDITIVE_SUFFICIENT.
- **No causal proof linking GENERator's BOS attention sink to the high-gain pathway.** Only
  association/co-occurrence has been measured; a causal-decoder structural fact (position 0
  attends only to itself) partially confounds even the co-occurrence measurement's
  interpretability, as the source analysis itself discloses.
- **Disputed downstream DNABERT-2 biological-phenotype magnitudes.** The splice-accuracy
  pair-ablation and codominance claims are currently contested (§7 items 1-2) — a real,
  single-seed reproduction conflicts with the previously-adopted magnitude by roughly two
  orders of magnitude, and this discrepancy is unresolved as of this document's writing.
- **Structural measures do not predict causal importance universally.** OLMo's own row set
  is the clearest within-repository demonstration: the structurally largest row is the
  causally weakest of its own 4-row set.
- **Architecture association is descriptive, not a universal law.** Repeated throughout §1,
  §2, §7 — Phi-3 is a decoder counterexample to "decoders are additive"; NTv3, ModernBERT's
  thin margin, and GENERator EUK's in-NLP-range `q1` all complicate a clean domain/
  architecture split at the structural level too.
- **Single-seed measurements in several places.** The DNABERT-2 splice/codominance
  reproduction (§7), and the GENERator BOS/attention-sink reproduction (§3.5), are each a
  single run — neither has been replicated across seeds within this repository.

Do not invent limitations beyond what the project's own design and results support — this
list is exhaustive of what the current evidence base actually shows, not a generic
boilerplate limitations section.

---

## 10. References

No `.bib` file exists anywhere in the repository. The only bibliography is an inline,
numbered `\begin{enumerate}` list in `paper/main.tex:1039-1078` (11 entries, quoted exactly):

| Key | Citation | Supports |
|---|---|---|
| [1] | Avsec, Ž. et al. Effective gene expression prediction from sequence by integrating long-range interactions. *Nat. Methods* 18, 1196-1203 (2021). | intro, generic genomic-LM background |
| [2] | Dalla-Torre, H. et al. Nucleotide Transformer: building and evaluating robust foundation models for human genomics. *Nat. Methods* 22, 287-297 (2025). | intro background (NOT the in-text NTv3 citation — see [5]) |
| [3] | Linder, J., Srivastava, D., Yuan, H., Agarwal, V. & Kelley, D. R. Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation. *Nat. Genet.* 57, 949-961 (2025). | intro background |
| [4] | Wu, W. et al. GENERator: A Long-Context Generative Genomic Foundation Model. Preprint (2025). | **GENERator model citation** |
| [5] | Boshar, S. et al. A foundational model for joint sequence-function multi-species modeling at scale for long-range genomic prediction. bioRxiv 2025.12.22.695963 (2025). | **NTv3 model citation** (mapped in-text, despite ref [2] also being Nucleotide-Transformer-adjacent — resolve which is the intended NTv3 citation before finalizing) |
| [6] | Zhou, Z. et al. DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome. Preprint (2023). | **DNABERT-2 model citation** |
| [7] | Brixi, G. et al. Genome modelling and design across all domains of life with Evo 2. *Nature* 652, 1349-1361 (2026). | **Evo2 (and, by the same shared reference, Evo1 — no separate Evo1 citation exists anywhere in the repo)** |
| [8] | Ma, M. et al. HybriDNA: A Hybrid Transformer-Mamba2 Long-Range DNA Language Model. Preprint (2025). | HybridNA (not in current 4-figure scope) |
| [9] | Shao, B. & Yan, J. A long-context language model for deciphering and generating bacteriophage genomes. *Nat. Commun.* 15, 9392 (2024). | MegaDNA (not in current 4-figure scope) |
| [10] | Yu, M., Wang, D., Shan, Q., Reed, C. J. & Wan, A. The Super Weight in Large Language Models. Preprint arXiv:2411.07191 (2024). | **Core method citation** — the data-free, single-forward-pass super-weight detection method this whole project builds on |
| [11] | Sun, P. et al. The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks. Preprint arXiv:2603.05498 (2026). | **UNRESOLVED — see below** |

**Ref [10] cross-check**: root `README.md` states the same paper twice with a first-initial
discrepancy — `README.md:3-5` gives no first initial ("Yu et al. (2024)"), but
`README.md:1293` gives "Yu, T. et al." while `paper/main.tex` ref [10] gives "Yu, M." — one
of these initials is wrong; not resolved by this document (arXiv:2411.07191 is confirmed
correct and consistent everywhere).

**Ref [11] — flagged unresolved, per `manuscript/docs/REFERENCE_AUDIT.md`** (read in full):
this reference is **never cited in the body text** of `paper/main.tex` (`grep` for "Sun"
returns only the bibliography line itself). The repo's own local copy of the Yu et al. 2024
paper (`docs/superweight_paper.txt`) cites a *different*, related paper in its own reference
list: "Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large
language models. ICLR 2024 Workshop... URL https://openreview.net/forum?id=1ayU4fMqme" (no
arXiv ID given there). `REFERENCE_AUDIT.md` explicitly declines to assume `arXiv:2603.05498`
is a typo for the massive-activations paper's actual ID (`2402.17762`), citing five reasons:
different author first-name/initial (P. vs. Mingjie), different title, different year (2026
vs. 2024), no arXiv ID in the one local candidate, and `2603.05498` being "a plausible
March-2026 identifier and may well be a real, different paper." **An external arXiv lookup of
both `2603.05498` and `2402.17762` is needed before this reference can be used — flag this to
the author/ChatGPT explicitly rather than guessing.** A 2026-08-13 addendum to the same audit
notes this ambiguity is now load-bearing for the paper's positioning claim, since an earlier
manuscript revision (D-017) frames `‖U_k‖_F` as "substantially related to prior work by Sun
et al." Separately, attention-sink prior art (Xiao et al., StreamingLLM) is **not cited
anywhere** in the current bibliography — needed if Figure 4's BOS/attention-sink panel (§3.5)
is discussed in Discussion, flagged as a gap.

### Per-model citation status (12-model structural/causal panel)

No `citations.md`-style file exists; the table below is built from `paper/main.tex`'s 11-entry
bibliography (which itself reflects the *old* v1 8-genomic-model panel, §1.4) cross-referenced
against the current 12-model E7-E10b panel.

| Model | Citation available? |
|---|---|
| GENERator | Yes — [4] |
| DNABERT-2 | Yes — [6] |
| NTv3 | Ambiguous — [5] (Boshar et al., bioRxiv) is mapped in-text, but [2] (Dalla-Torre et al., *Nat. Methods* 2025, "Nucleotide Transformer") is also plausible; resolve before finalizing which is the actual NTv3-v3 citation |
| Evo1 | No separate citation — shares [7] with Evo2 |
| Evo2 | Yes — [7] |
| GenomeOcean-4B | **NOT FOUND** anywhere in the repo (postdates the v1 8-model panel) |
| HybridNA | Yes — [8] (not in current 4-figure scope) |
| MegaDNA | Yes — [9] (not in current 4-figure scope) |
| Caduceus | **NOT FOUND** — mentioned by name only, no citation (not in current 4-figure scope) |
| MosaicBERT | **NOT FOUND** anywhere in the repo |
| ModernBERT | **NOT FOUND** anywhere in the repo |
| Phi-3 | **NOT FOUND** anywhere in the repo (only the HF checkpoint/commit is cited, no technical report) |
| Qwen2.5 | **NOT FOUND** anywhere in the repo |
| OLMo | **NOT FOUND** anywhere in the repo |
| Llama / Mistral | **NOT FOUND** anywhere in the repo (cited only by HF repo path; used solely as Yu-et-al.-coordinate recovery targets) |

Eight of twelve current-panel models (GenomeOcean, MosaicBERT, ModernBERT, Phi-3, Qwen2.5,
OLMo, Llama, Mistral) have **no citation anywhere in this repository** and will need one
added before submission; do not fabricate one here.

---

## 11. Manuscript logistics

`paper/main.tex` exists (55,408 bytes, last modified 2026-08-13; header comment reads "v6"),
alongside `paper/main_old.tex` and a `paper/media/` figure-asset directory. **Everything
below is quoted from the actual manuscript file — it reflects the OLD, retired v1 thesis
(§1.4) and should be treated as a starting point for logistics fields only, not as evidence
that the current four-figure structure is reflected in this file's prose.**

| Field | Value | Source |
|---|---|---|
| Working title (old thesis) | "A Structural Predictor of Super-Weights Across Genomic Language Model Architectures" | `paper/main.tex:46-47` — **this is the v1/retired-thesis title; see §1.1 for new draft options matching the current thesis** |
| Author list | Alexandros Tzanakakis¹, Aris Karatzikos¹,², Ilias Georgakopoulos-Soares¹,* | `paper/main.tex:51-54` |
| Affiliations | ¹Division of Pharmacology and Toxicology, College of Pharmacy, The University of Texas at Austin, Dell Paediatric Research Institute, Austin, TX, USA. ²Department of Computer Science, College of Natural Sciences, The University of Texas at Austin, Austin, TX, USA. | `paper/main.tex:58-64` |
| Corresponding author | Ilias Georgakopoulos-Soares, ilias@austin.utexas.edu | `paper/main.tex:63-64` |
| Document class / formatting | `\documentclass[11pt]{article}` — plain LaTeX article, NOT a journal-specific template (no Nature/bioRxiv/PLOS class file); packages: `fontenc, lmodern, inputenc[utf8], geometry[margin=1in], amsmath/amssymb, graphicx, booktabs/array/longtable, caption, float, hyperref, xcolor, enumitem, ulem` | `paper/main.tex:1-20` (approx.) |
| Intended venue | Not stated inside `paper/main.tex` itself. Target list recorded separately: "Positioning: Mechanistic-interpretability spine, computational-biology venue. Targets unchanged: **Genome Biology, Nature Methods, NAR Genomics & Bioinformatics, Patterns.**" | `manuscript/docs/PAPER_OUTLINE.md:25-26` |
| Abstract word limit | UNKNOWN — USER INPUT REQUIRED | not found anywhere in repo |
| Section requirements | UNKNOWN — USER INPUT REQUIRED (no external template found; current file follows an informal Abstract/Intro/Results/Discussion+Limitations/Methods/References structure of its own devising) | not found |
| Figure limits | UNKNOWN — USER INPUT REQUIRED | not found |
| Acknowledgments | UNKNOWN — USER INPUT REQUIRED | confirmed absent — `grep -i "acknowledg"` over `paper/main.tex` returns nothing |
| Funding statement | UNKNOWN — USER INPUT REQUIRED | confirmed absent |
| Code/data availability statement | UNKNOWN — USER INPUT REQUIRED | confirmed absent — no such section exists in the file |
| Author contributions | UNKNOWN — USER INPUT REQUIRED | confirmed absent |
| Competing interests | UNKNOWN — USER INPUT REQUIRED | confirmed absent |

A combined search (`grep -i "acknowledg|funding|competing interest|data availability|code
availability|author contribution|conflict of interest|word limit|journal|submitted to|venue"`
over `paper/main.tex`) returned **zero matches** — every UNKNOWN row above is a confirmed
absence in the manuscript file itself, not merely an unlabeled section.

**Recommendation for the manuscript writer**: the author list, affiliations, and
corresponding-author fields are real and likely still current — reuse them. The **title**
reflects the retired thesis and should be replaced with one of §1.1's options (or a new one
matching the current framing) before submission. The four target venues (Genome Biology,
Nature Methods, NAR Genomics & Bioinformatics, Patterns) are a decision record, not a
confirmed submission target — confirm with the author before assuming a specific venue's
formatting template.

---

## UNRESOLVED FACTS FOR CHATGPT

### Manuscript-blocking
- **DNABERT-2 splice-accuracy pair-ablation and codominance magnitudes are contested**
  (§7 items 1-2). The manuscript cannot state either the old or the new number as
  established fact without author adjudication — this blocks any Results/Discussion sentence
  that would otherwise cite "the DNABERT-2 pair costs −33.76pp on splice accuracy" or the
  codominance magnitude.
- **Reference [11] (Sun et al., arXiv:2603.05498) cannot be verified** and is never cited in
  the manuscript body despite appearing in the bibliography (§10). If Figure 4's
  BOS/attention-sink phenotype or the `‖U_k‖_F`/massive-activations connection is discussed
  in Discussion, an external arXiv lookup of both `2603.05498` and `2402.17762` is needed
  before citing either — do not guess which is correct.
- **Eight of twelve current-panel models have no citation anywhere in the repository**
  (GenomeOcean-4B, MosaicBERT, ModernBERT, Phi-3, Qwen2.5, OLMo, Llama, Mistral — §10). A
  Methods/References section citing these models needs external citations added; none exist
  to draw from locally.
- **The NTv3 citation is ambiguous** — ref [5] (Boshar et al.) is mapped in-text to NTv3, but
  ref [2] (Dalla-Torre et al., "Nucleotide Transformer") is also plausible and the two are not
  obviously the same underlying model/paper. Resolve which is correct before citing NTv3 in
  Methods.
- **No pinned HF revision exists for Llama-7B, Mistral-7B, or OLMo-7B anywhere in the
  repository** (§4.1) — a rigorous Methods section should either have these pinned before
  submission or explicitly disclose the reproducibility gap.
- **Working title in `paper/main.tex` reflects the retired v1 thesis** and must not be reused
  without revision (§1.1, §11) — a title decision from §1.1's options (or a new one) is
  needed before a full draft can be finalized, though this does not block drafting the body.

### Can safely remain as placeholders
- Formatting/word-limit/section-requirement/figure-limit fields (§11) — all confirmed
  genuinely absent from the repository, not merely unlabeled. A first full draft can be
  written in plain structure and reformatted once a specific venue is confirmed.
- Acknowledgments, funding, code/data availability, author contributions, competing
  interests (§11) — all confirmed absent; standard placeholder boilerplate is acceptable
  until the author supplies real text.
- NTv3's corrected splice-ablation number (§3.6, §7 item 3) — the manuscript can state "not
  currently established, no artifact" as a placeholder sentence without blocking the rest of
  Results/Discussion, since NTv3 is not central to any of the four Results sections.
- Tokenizer type for Llama, Mistral, OLMo, Qwen2.5, Evo2 (§4.1) — genuinely undocumented in
  the repository; a Methods table can cite "standard [architecture-family] tokenizer" as a
  reasonable placeholder pending author confirmation.

### Optional/nonessential
- GENERator sequence-quality/specificity stability during GC intervention — absent
  entirely, not referenced by any main-text claim, safe to omit or mention as a stated gap.
- E10b's three-way redundancy diagnosis mechanism (why layer 2 specifically) — explicitly
  out of scope per the source experiment's own stated limits; not needed for any main-text
  claim beyond reporting the split result itself.
- The promoter/enhancer bed files (§5.2) not used by any current headline result — safe to
  omit from Methods entirely unless a future supplementary analysis uses them.
- Intended-venue confirmation (§11: Genome Biology / Nature Methods / NAR Genomics &
  Bioinformatics / Patterns) — a decision record, not a confirmed target; a first draft can
  be written venue-agnostically.

---

## Checklist: is there enough evidence to write each section?

| Section | Status | Note |
|---|---|---|
| Title | READY WITH PLACEHOLDER | 5 draft options provided (§1.1); `paper/main.tex`'s existing title reflects the retired v1 thesis and should not be reused as-is |
| Abstract | READY | Central claim (§1.2) and 4 headline results (§3) are sufficient; must incorporate the §7 contested-claims caveat |
| Introduction | READY | Framing (§0-1.2) and prior-work distinction (structure≠function≠causal-complexity) are well-supported |
| Results 1 (Structural) | READY | Full 12-model table, §3.1, all MAIN-TEXT SAFE |
| Results 2 (Causal organization) | READY | Full decoder/encoder/E10b table, §3.2-3.3, all MAIN-TEXT SAFE |
| Results 3 (DNABERT-2) | READY WITH PLACEHOLDER | Pretrained-MLM panels (§3.4) fully safe; the downstream splice/codominance claim must be written as explicitly contested (§7), not omitted or resolved |
| Results 4 (GENERator) | READY WITH PLACEHOLDER | GC dose-response and BOS phenotype (§3.5) are safe; sequence-quality claim must be stated as absent, not fabricated |
| Discussion | READY | §1.5 takeaways are evidence-grounded |
| Methods | READY WITH PLACEHOLDER | §4 provides exact model/math/intervention/tomography definitions; placeholders needed only for unpinned NLP-model revisions and undocumented tokenizer types (both flagged explicitly, not silently filled) |
| Figure legends | READY | §2 provides full draft legends for all 4 main figures + Supplement S1 |
| References | READY WITH PLACEHOLDER | §10 provides the full 11-entry bibliography and exact per-model citation coverage; 8 of 12 current-panel models and 1 flagged-ambiguous reference (Sun et al. [11]) still need real citations added — do not fabricate them |
| Supplementary notes | READY | §2 (Supplement S1), §7, §9 cover the supplementary material fully |
| Manuscript logistics (title page, venue, statements) | READY WITH PLACEHOLDER | Author list/affiliations/corresponding author (§11) are real and reusable; venue, word limits, and all administrative statements are confirmed absent from the repository — `UNKNOWN, USER INPUT REQUIRED` |
