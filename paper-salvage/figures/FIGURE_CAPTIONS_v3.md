# FIGURE_CAPTIONS_v3.md — draft captions for main and supplementary figures

Descriptive only; every quantity here is traceable to `FIGURE_PROVENANCE.md`. These are
draft captions for figure files, not manuscript prose — do not treat as a section rewrite.

---

## Figure 1. Cross-domain structural organization of high-gain FFN pathways

**(A)** Calibration: the exact `U_k` cold-weight predictor recovers the published NLP
super-weight row at rank 1 (100th percentile) in Llama-7B, Mistral-7B, and OLMo-7B, with
the published scalar index carrying 89–99% of the row's diagonal contribution
(retrospective analysis, diagonal approximation; E1). **(B)** Exact spectral concentration
(`q1`, the fraction of squared singular mass on the first singular vector of the exact
bilinear operator `U_k`) across a combined panel of 5 text decoders, 2 genomic decoders, 2
text encoders, and 2 genomic encoders. Marker shape denotes architecture (circle=decoder,
square=encoder); color denotes domain. Evo2-7B is marked as a structural null (no
down-projection spike cleared the pre-registered 5× activation-detection threshold in 32
layers; ×). Dashed line: the pre-registered decoder floor (GenomeOcean-4B, `q1`=0.8989)
used for the E8 encoder-vs-decoder branch decision. **(C)** Exact operator magnitude
(`‖U_k‖_F`, log scale) for the same panel, kept on a separate axis from `q1` by design —
raw magnitudes span three orders of magnitude across models with different weight scales
and are not directly comparable in absolute terms; this panel exists only to keep magnitude
and concentration visually distinct. **(D)** For the two encoders with same-layer control
rows measured (MosaicBERT, ModernBERT), the detected high-gain row (filled) is far more
concentrated than 5 seeded ordinary rows at the same layer (open), the reverse of a naive
"encoders are generally distributed" reading. Sample unit: model (panel B/C, n=12) or row
(panel D, 1 candidate + 5 controls per model). No error bars: single-checkpoint, exact
(non-stochastic) measurements throughout. Descriptive, not a formal statistical test of
domain vs. architecture separation.

---

## Figure 2. Cross-model causal organization

Endpoint: causal-LM negative log-likelihood (decoders) / masked-LM loss (encoders) over a
frozen WikiText-2-raw-v1 test-split window set (N=100 windows, 512 tokens, decoders; N=256
windows, encoders), row-scale intervention `alpha = 1 - epsilon*a` on `mlp.down_proj`.
**(A)** Singleton causal spectrum for all 5 decoders at full ablation (`epsilon=1.0`,
`alpha=0`): `|relative %NLL change|` per structurally-selected row (log scale), dotted line
= same-layer random-control median. Llama/Mistral/Qwen have exactly one pre-existing
high-gain candidate (K=1); OLMo has 4 (K=4); Phi-3 has 6 (K=6). Color denotes the
pre-registered mechanical decision (`C1`>0.5 and control-normalized effect >3.0 ⇒
single-component-dominant; else, ≥2 rows individually clearing the threshold ⇒
multi-component-candidate). Mistral is non-monotonic (its `alpha=0.5` effect, +301%,
exceeds full ablation, +287%) — visible directly in the raw values, not smoothed. **(B)**
OLMo's structural-vs-causal rank dissociation: the structurally largest row (L24/r269, 53.4×
its layer's median exact `‖U_k‖_F`) is the causally *weakest* of its own 4-row set (+1.4%
NLL change), while the causally *dominant* row (L1/r269, +112%) ranks only 3rd
structurally. **(C)** Encoder side: relative held-out MAE improvement from adding pairwise
interaction terms (F2→F3) for MosaicBERT and ModernBERT, both intervention scales, with
bootstrap 95% CIs (5,000 resamples, paired over held-out masks); both clear the
pre-registered 10%-improvement-with-CI-excluding-zero adequacy bar at both scales,
replicating DNABERT-2's interactional phenotype (Fig. 3C) independently in two
architecturally distinct text encoders. **Panel C also shows Phi-3 (hatched bars) — the one
decoder E10 flagged as a multi-component candidate and therefore eligible for its own
tomography (E10b).** Phi-3's result splits by intervention scale: pair terms materially help
at `epsilon=0.5` (+36.4%, bootstrap CI excludes zero) but *reliably hurt* at `epsilon=1.0`
(−2.5%, bootstrap CI excludes zero on the negative side) — diagnosed as a three-way
redundancy break among its 3 layer-2 rows (damage appears only when all three are removed
together, not predictable from any pairwise combination), not rescued with a higher-order
model. Reported as split, not forced into either class. Sample unit: model (n=5 decoders,
n=2 encoders, n=1 additionally-tomographed decoder) — descriptive model-level counts, not a
powered cross-model statistical test.

---

## Figure 3. DNABERT-2 mechanistic organization and falsification

**(A)** Individually masking either channel of the critical row pair (L9/r264, L9/r294) in
DNABERT-2's pretrained masked-LM loss (no fine-tuning, 256 hg38 windows) produces small
effects; a direct, jointly-masked 2-channel intervention produces a much larger effect than
either channel alone (`epsilon=1.0`). This is a directly measured joint ablation, not a
fitted prediction, and is reported on the pretrained MLM-loss scale — a different
endpoint/unit from the previously-reported splice-accuracy pair-ablation figure, which has
no raw artifact in this repository (see caveat below). **(B)** The same super-additivity
(`epistasis = Δloss_joint − Δloss_A − Δloss_B`) is positive at both tested intervention
scales, confirming the interaction is not an artifact of one specific ablation strength;
this independently reproduces a previously-reported pretraining-epistasis figure to 4
decimals from a from-scratch environment and revision-pinned checkpoint. **(C)** Held-out
prediction quality for four increasingly expressive "observer" models of the finite causal
response to 10-row masks (F0 singleton-additive, F1 scalar-calibrated, F2 jointly-fit
additive, F3 pairwise-lifted ridge regression), on 20 genuinely held-out masks never used
for fitting, at both intervention scales; F0–F2 fail held-out adequacy (R²<0.90 or worse),
F3 clears or approaches it. **(D)** F3's fitted pairwise coefficients (top 12 of 45 by
magnitude), both scales; the independently-known critical pair ranks #1 at both scales
without having been given special treatment during fitting — a retrospective validation,
not the basis-selection procedure. Error bars/CIs: bootstrap 95% CI on the F2→F3 MAE
improvement, reported in the main text accompanying this figure, not drawn on panel C/D
directly to avoid clutter (see source data). **Caveat (all panels):** the interaction
structure found here describes causal response *in the declared 10-row residual-channel
intervention basis* — not a proof of an intrinsic causal dimension of 2. **A related claim —
that the same critical pair produces a large joint effect on fine-tuned splice accuracy, and
a separate codominance/norm-carriage mechanism — was previously cited from an adopted prior
report (−33.76pp joint splice effect; codominance epistasis −33.63→−0.66pp). A fresh,
code-verified reproduction (single seed, the only checkpoint available) now exists and
materially conflicts with those magnitudes: measured joint splice effect −0.42pp (the named
pair is not even the strongest epistatic pair in this remeasurement), codominance epistasis
range −0.53→−0.04pp. L9/r264+r294 is therefore strongly supported as an interaction in
pretrained-MLM tomography (panels A–D above), but the previously claimed catastrophic
downstream splice effect and large codominance mechanism are not currently established.
Neither the old nor the new number is plotted in this figure or presented as settled — this
is disclosed as an open, contested discrepancy pending author adjudication, not resolved
here.** NTv3's role as a falsification case is discussed in the main report but not plotted
in any figure: the only NTv3-splice raw artifact in this repository is a retired,
truncation-bug-affected result and is not shown, even labeled as retired.

---

## Figure 4. GENERator: functional realization of a decoder high-gain pathway

**(A)** GENERator EUK's layer-4/row-2371 high-gain channel has exact `q1`=0.969 (see Fig.
1B for the full cross-model panel this is drawn from). **(B)** BOS-centered
attention/activation phenotype, now reproduced with a real raw artifact (2026-08-22, this
repo's own run, eager attention, 40 hg38 windows, seed 42): 37.9% of mean incoming attention
mass concentrates at position 0 (33.0× the uniform expectation across the window), 78.8% of
(layer, head) pairs have their attention argmax at position 0, and row 2371's activation
maximum is also at position 0. These numbers closely match a previously-cited, at-the-time
unreproduced report. **Wording is strictly associative**: BOS-centered attention and
activation *co-occur* with the high-gain pathway; this does not establish that the row
*causes* the sink (a causal-decoder structural fact — position 0 can only attend to itself —
means some of this co-occurrence is architecturally guaranteed rather than a discovered
effect, as the source analysis itself notes). Single run, not yet seed-replicated. **(C)**
Causal dose-response of generated sequence GC composition under row-scale suppression
(`alpha` ∈ {0, 0.5, 1.0}; 24 hg38-window prompts, paired generation seeds), for the primary
row (2371), a secondary row (1522) at the same output column, and a 5-row random-seeded
control. Error bars: SEM. Connectors are dotted guide-the-eye lines, not fitted curves — only
3 dose points were collected by design (the pre-registered protocol froze exactly these 3
alphas), so this figure supports "large, directionally consistent causal sensitivity for row
2371" (span ≈390× the random-control span), not a validated linear or graded steering axis;
row 1522's non-monotonicity (rises under partial suppression, falls under full ablation) is
direct evidence against reading this as a simple steering knob. **Sequence-quality stability
over this intervention range has been discussed elsewhere but has no raw artifact anywhere
in this repository and is not plotted.**

---

## Supplement S1. Full structural / control-row detail behind Figures 1 and 2

**(a)** Phi-3's 6 individually published high-gain rows' exact `q1` — Figure 1B shows only
the model-level median (dashed line here); this panel shows the underlying spread (0.56–0.93
across the 6 rows). **(b)** All 5 decoders' same-layer random-control rows' individual
`|relative %NLL change|` (log scale) — the raw points behind Figure 2A's control-median
dotted line, showing the control distributions are tight and well-separated from the
target-row effects at every model.

## Supplement S2 — REMOVED

Previously showed the retired (X-009) NTv3 splice-ablation result, explicitly labeled
RETIRED, for methodological-history transparency. Removed per explicit instruction: an
invalid, truncation-bug-affected result is not plotted in this figure set even labeled as
retired. For the record (not plotted): **RETIRED (X-009).** Per-seed ΔMCC (SW-ablated −
baseline) for NTv3 on the splice/reconstructed GUE task, 5 seeds, was produced under a
since-confirmed truncation bug (the launch script never passed `--max_length`, so a
task-key fallback resolved to 80 tokens for NTv3's nucleotide-level tokenizer, truncating
every splice window to its first 80bp — before the splice junction the task is about); see
`CLAIMS_LEDGER.md` for the retired claim's status. **The corrected re-fit
(reported MCC 0.86–0.91, SW-ablation effect −0.02pp, i.e. a null result) has no raw
JSON/CSV artifact anywhere in this repository and cannot be plotted.**
