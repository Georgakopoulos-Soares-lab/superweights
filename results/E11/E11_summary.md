# E11 — scale-ladder measurement: does architecture survive conditioning on scale?

**What this experiment tests.** E11 checks whether C-034 (`CLAIMS_LEDGER.md`: exact
bilinear-operator `q1` tracks encoder/decoder organization more than text-vs-genomic
domain) is a scale confound rather than a real architecture effect — every decoder in the
original 9-model panel was ≥3B params, every encoder ≤650M. **This is explicitly a test of
secondary, already-hedged calibration material (D-027, 2026-08-23), not the manuscript's
headline finding.** The headline architecture-level claim is causal-response complexity
(C-046/C-047, from E9/E10/E10b's causal tomography); nothing here bears on that claim in
either direction. Per the prereg's own three-branch design, outright deletion of C-034 was
an explicitly legitimate outcome of this experiment, not a result requiring a rescue
paragraph.

**Scope trim, disclosed up front.** This run measured a 12-model reduced ladder, not the
full 15-model prereg'd set: the OLMo-2 family (1B/7B/13B) and three bonus data points
(GENERanno-PROK/EUK-0.5B, GTE-large-en-v1.5) were dropped by author instruction on
2026-08-23 for infrastructure time constraints — each OLMo-2 rung took 20–30+ minutes to
download and the bonus points were already flagged in the prereg as optional
regression-leverage extras, not required for the decision rule. **This is not a scientific
exclusion.** The prereg's own minimum bar — "≥2 decoder families, ≥3 sizes each" — is met
without OLMo-2: Qwen2.5 alone contributes 4 sizes (0.5B/1.5B/3B/7B) and SmolLM2 contributes
3 (135M/360M/1.7B), both fully independent decoder ladders. The encoder side is anchored by
EuroBERT's 3 sizes (210M/610M/2.1B), plus GENERator-PROK's 2-size genomic decoder ladder and
ModernBERT's 2-size (base/large) text-encoder ladder as secondary regression leverage. All
12 new models were measured with the identical, unmodified E7/E8/E10 pipeline
(`DownProjRecorder` detection at `RATIO_THRESHOLD=5.0`, `spectral_lib.row_spectral_metrics`,
`SeedSequence(42)`-seeded 5-row same-layer controls), verified prereg-locked
(`0a36ac01...`, confirmed OK before and after measurement) before any forward pass.

## Result

All 12 measured models produced a valid, accepted detection candidate (no nulls) and were
independently confirmed **gated** from actual `named_modules()` inspection at load time —
every Qwen2.5/SmolLM2/GENERator-PROK size uses separate `gate_proj`/`up_proj`/`down_proj`
(SwiGLU); every EuroBERT size uses the identical separate-projection pattern under
`model.model.layers[i].mlp`; ModernBERT-large uses ModernBERT-base's packed `Wi`/`Wo` GeGLU
pattern. No expected-gated model failed verification. All measured total-parameter counts
matched the prereg's own pre-computed values exactly (e.g. Qwen2.5-3B: 3,085,938,688;
GENERator-PROK-3B: 2,998,262,784), confirming the pinned revisions resolved correctly.

**The encoder-range contingency did *not* trigger.** EuroBERT-2.1B carries 1.81B
non-embedding parameters — comfortably above the ~1B threshold the prereg flagged as the
condition under which Branch 3 becomes the forced default. A real gated bidirectional
encoder family above 1B was measured at all three of its sizes.

**OLS `q1 ~ log10(non_embedding_params) + is_decoder`** (23 models: 11 cited + 12 measured;
descriptive, not a powered test, per the manuscript's standing statistical stance):

- `is_decoder` coefficient = **0.1162**, SE = **0.0860**, **|coef|/SE = 1.351**
- `log10(non_embed_params)` coefficient = see `results/E11/regression_summary.json`
- R² = 0.158, partial correlation of `is_decoder` with `q1` controlling for scale = **0.289**
- Raw (unconditioned) architecture gap, median decoder `q1` − median encoder `q1` = **0.031**
- Conditioned-gap estimate (the fitted `is_decoder` coefficient itself, since the
  specification is additive/parallel-slopes in log10(params)) = **0.116** — this is
  *larger* in magnitude than the raw gap, not smaller (a −275% "shrink," i.e. the opposite
  of shrinkage)

**Decision-rule outcome: Branch 3 — demote to a one-line supplementary footnote.**
`1 ≤ |coefficient|/SE ≤ 2` is the literal trigger (1.351 falls in that band), so neither
Branch 1 (keep, `|coef|/SE > 2`) nor Branch 2 (cut, `|coef|/SE < 1` or gap shrinks >50%) is
met. Note this is a *different* route into Branch 3 than the prereg's anticipated default
(the "cannot sample encoders above ~1B" fallback) — that fallback did not fire here; Branch
3 instead follows directly from the coefficient's own ambiguous magnitude relative to its
standard error, with 23 model-level data points.

## The more interesting fact underneath the ambiguous coefficient

The raw architecture gap (0.031) is not small because the effect vanished — it is small
because **the EuroBERT ladder, added specifically for regression leverage, scores
essentially at ceiling on `q1` at every one of its three sizes**: EuroBERT-210m 0.984,
EuroBERT-610m 0.9996, EuroBERT-2.1B 0.998. These are indistinguishable from — and in two of
three cases, higher than — every decoder in the original panel except Mistral-7B. A
genuinely gated, bidirectional, >1B-parameter text encoder turns out to be *more* rank-1
than most of the decoders C-034 was built on. This is a direct, measured counterexample to
"encoders are less concentrated than decoders" from exactly the family this experiment
added to give the encoder side real scale range, not a hypothetical edge case. By contrast,
the original discovery-panel encoders (MosaicBERT 0.477, DNABERT-2 0.793, NTv3 0.389) remain
low, and the new decoder ladders are *not* uniformly high either (Qwen2.5-3B 0.817,
GENERator-PROK-1.2B 0.847) — so the picture is not "encoders now equal decoders" so much as
"both groups turn out to be far more heterogeneous internally than the original 9-model
panel suggested," which is itself the reason the coefficient lands in the ambiguous zone
rather than cleanly surviving or cleanly failing.

**Within-family slopes** (Δq1 per decade of log10(non-embedding params), not pooled):

| Family | n | slope/decade | predicted Δq1 over observed range |
|---|---:|---:|---:|
| Qwen2.5 | 4 | −0.075 | −0.094 |
| GENERator-PROK | 2 | +0.213 | +0.088 |
| ModernBERT | 2 | +0.149 | +0.073 |
| SmolLM2 | 3 | +0.032 | +0.038 |
| EuroBERT | 3 | +0.012 | +0.014 |

All five slopes are small in absolute magnitude, and EuroBERT's is the flattest of all
(consistent with it saturating near `q1`≈1 already at its smallest, 210M size) — within-family
scale alone does not explain much of anything, in either direction, for any ladder measured.

## Secondary analyses

**Depth control.** Correlation of `q1` with relative candidate depth (candidate layer / total
layers) across all 23 models = **−0.383** — a moderate but not dominant negative
relationship. Candidates at deeper relative depth tend to run somewhat lower `q1`, but this
does not track cleanly with either scale or architecture (e.g. Qwen2.5-1.5B sits at depth
0.93 with `q1`=0.995, while MosaicBERT sits at depth 0.75 with `q1`=0.477) — flagged as a
real but secondary, non-explanatory factor, not a second confound that overturns the scale
analysis above.

**Both `‖U_k‖_F` rankings disagree substantially.** Absolute `‖U_k‖_F` and
ratio-to-layer-median `‖U_k‖_F` (candidate frob-norm ÷ median of its own 5 seeded
same-layer control rows' frob-norms) produce very different orderings — e.g. EuroBERT-610m
ranks 15th of 20 by absolute magnitude but **1st** by ratio-to-median; Qwen2.5-0.5B ranks
14th absolute but 2nd by ratio. This confirms absolute `‖U_k‖_F` is dominated by raw
model/layer scale (larger `d_model`, larger weight norms) and is not a valid cross-model
comparison; the ratio-to-layer-median version is the metric that should be used whenever
`‖U_k‖_F` magnitude is compared across models of different size.

**Candidate-vs-control gap is stable and large across all 12 measured models**, independent
of architecture or scale: mean gap (candidate `q1` − mean of 5 control-row `q1`) = 0.884,
range 0.544 (ModernBERT-large) to 0.977 (Qwen2.5-1.5B). Every measured model's detected
high-gain row is dramatically more concentrated than its own layer's typical rows, with no
sign of this gap shrinking as scale increases. This generalizes E8's C-035 finding
(MosaicBERT/ModernBERT-base only) to 12 more models spanning 106M–3.0B non-embedding
parameters and both architectures — and, per the prereg's own framing, is the most
defensible remaining version of an architecture-*adjacent* structural effect, precisely
because it does not depend on the disputed OLS specification above. It is, however, a
statement about the row-level phenomenon's robustness, not a restatement of C-034's
domain/architecture claim.

## Recommendation

**File under Branch 3.** C-034 should be demoted to a one-line supplementary footnote,
pointing to `results/E11/`, stating that the scale confound was checked with a 23-model
ladder spanning ~110M to ~7.6B non-embedding parameters across 2 decoder families
(Qwen2.5, SmolLM2) and 1 encoder family (EuroBERT) with ≥3 sizes each, and that the result
was inconclusive by the prereg's own pre-committed threshold (`|coef|/SE` = 1.35, in the
1–2 ambiguous band) — not because encoders above 1B could not be found (they were), but
because the architecture coefficient's magnitude relative to its standard error simply did
not clear either bright line. It should not appear in the main text. This does not affect
C-046/C-047 in any way.

## Provenance

| | |
|---|---|
| Prereg | `docs/prereg/PREREG_E11_scale_ladder.md`, sha256 `0a36ac01...`, locked 2026-08-23T16:03:20+00:00, verified OK before and after measurement |
| Producing scripts | `experiments/E11_scale_ladder/run_model.py` (per-model measurement), `build_csv.py` (assembly), `regression.py`, `plot.py` |
| Shared library | `experiments/E7_exact_dimensionality/spectral_lib.py` (reused unmodified) |
| Raw outputs | `results/E11/raw/{key}.json` (12 files), `results/E11/logs/{key}.log` |
| Deliverables | `results/E11/scale_ladder.csv` (23 rows), `results/E11/scale_ladder_controls.csv` (60 rows, 12 models × 5), `figures/E11/q1_vs_scale.pdf`, `results/E11/regression_summary.json` |
| Dtype | float32 for all 12 new forward passes and weight extractions (no dtype fallback needed — all 12 fit comfortably on a 40GB A100); float64 for all spectral/SVD computation |
| Device | CUDA, single A100-PCIE-40GB, one model at a time |
| Cited (not re-measured) | Llama-7B, Mistral-7B, OLMo-7B-0724-hf, Phi-3-mini-4k-instruct, Qwen2.5-7B, MosaicBERT, ModernBERT-base, NTv3, DNABERT-2, GENERator-EUK-3B, GenomeOcean-4B — transcribed from `results/e7_legacy_reanalysis.json`, `results/e7_phase1_detection_{qwen25,genomeocean}.json`, `results/e8_detection_{mosaicbert,modernbert}.json`, `experiments/E7_exact_dimensionality/RESULTS.md`, and (for NTv3's architecture only — `d_model`=1536, `d_ffn`=6144, 12 layers) `docs/MANUSCRIPT_SOURCE_OF_TRUTH.md` |
| Scope trim | OLMo-2-{0425-1B,1124-7B,1124-13B} and 3 bonus points (GENERanno-PROK/EUK-0.5B, GTE-large-en-v1.5) dropped for infrastructure time, not scientific reasons — see top of this document |
| All artifacts | written under `/work/11034/atzanakak/glm_super_weight/genomic-super-weights/`, never under session-scoped `/tmp` |
