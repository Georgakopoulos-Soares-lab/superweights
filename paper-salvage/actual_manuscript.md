# Structure Is Not Mechanism: High-Gain Gated-FFN Rows Across Text and Genomic Foundation Models

> Coding-agent version converted from the manuscript PDF. The full manuscript text and figure captions are preserved in searchable Markdown. The plotted figure artwork itself is not embedded here; consult the source PDF when visual inspection of a figure is required. Mathematical notation is text-extracted from the PDF and should be checked against the source before editing equations.

Source: `new v9(1).pdf`

Alexandros Tzanakakis¹, Aris Karatzikos¹,², Ilias Georgakopoulos-Soares¹,*
¹Division of Pharmacology and Toxicology, College of Pharmacy, The University of Texas at Austin, Dell Paediatric Research
Institute, Austin, TX, USA
²Department of Computer Science, College of Natural Sciences, The University of Texas at Austin, Austin, TX, USA
*Correspondence: ilias@austin.utexas.edu

## Abstract

A small number of unusually high-gain parameters can exert disproportionate effects in transformer language
models, but whether analogous structures recur in genomic foundation models, and whether their structural
geometry determines their functional importance, is unknown. We analyzed high-gain rows in gated
feed-forward networks across text and genomic foundation models using an associated bilinear weight
operator whose exact Frobenius norm and singular spectrum separate operator magnitude from spectral
concentration. Against ordinary same-layer rows, high-gain candidates were strongly enriched for spectral
concentration; however, this separation largely disappeared when comparison was restricted to the
highest-magnitude rows in the same layer. In a frozen 22-model causal census, one activation-selected
candidate and five same-layer controls were evaluated under partial suppression and full ablation. The
candidate exceeded the within-model median control in 18 of 22 models at half strength and 20 of 22 at full
ablation; the cohort-median candidate-minus-control gaps were +0.60% and +0.88%, respectively. Against a
stricter control set — the five highest-Frobenius-norm rows in the candidate’s own layer, selected without
any forward pass — the candidate retained its advantage in 20 of 22 models at half strength and 19 of 22 at
full ablation, while these high-norm rows were themselves causally close to inert (median effects +0.035%
and +0.110%). Yet full-ablation candidate effects ranged from −0.79% to +698.87%, and spectral
concentration was essentially unrelated to signed full-ablation effect across the panel (Spearman ρ=0.074,
95% model-bootstrap interval −0.391 to 0.513). Operator magnitude likewise failed to provide a robust
effect-size predictor. Focused mechanistic analyses showed that DNABERT-2 contains a robust pairwise
interaction on its pretrained masked-language-model objective, whereas disruption of a concentrated
GENERator row produced a large shift in generated nucleotide composition that tracked native-loss damage
and could be reproduced by a random-direction replacement at comparably severe damage, arguing against a
simple direction-specific interpretation.

## Introduction

Transformer models often contain small subsets of parameters or activation channels with effects far
larger than their numerical prevalence would suggest. In large language models, Yu et al. identified
individual "super weights" whose ablation can severely degrade generation quality, and related work has
described massive activations and attention sinks that recur at fixed positions across prompts [1-4]. These
observations raise a broader mechanistic question: are such high-gain structures peculiar to particular
language-model families, or are they a more general architectural phenotype that also appears in foundation
models trained on other symbolic sequences?
Prior super-weight studies established that individual parameters can be catastrophically important in
several text decoders. Here we ask a different question: when activation-derived high-gain structures recur
across architectures and domains, which properties of those structures, if any, determine their functional
importance? This distinction turns the problem from detecting extreme parameters into testing whether
structural extremeness constitutes a transferable mechanism.
Genomic foundation models provide a useful setting in which to separate architecture from data domain.
Current genomic models span decoder-only generative transformers, bidirectional masked-language
encoders, and hybrid sequence models, while operating on nucleotide rather than natural-language corpora
[5-9]. If high-gain structures are fundamentally tied to gated feed-forward computation rather than linguistic
semantics, related structural phenotypes should recur across text and genomic models. At the same time, their
downstream consequences need not be shared: a high-gain coordinate could act as a nearly singular
bottleneck in one model, participate in a redundant or interaction-dependent circuit in another, or be
structurally conspicuous yet functionally weak.
A central difficulty is therefore definitional. Structural magnitude, spectral concentration, and causal
importance are different quantities. A diagonal approximation to the gated-FFN bilinear operator neglects
cross terms that can be substantial, particularly in genomic models, making it unsuitable for cross-model
dimensionality claims. We therefore define a bilinear weight operator associated with each candidate output
row and compute its Frobenius norm and singular spectrum exactly, using these quantities as weight-space
descriptors of magnitude and concentration. For multi-row case studies, we evaluate progressively richer
response models on held-out finite interventions, asking whether additive predictions suffice or whether
calibration and pairwise interaction terms are required [10]. Candidate rows are suppressed and evaluated
through changes in the model's native language-model loss or, for a generative genomic case study,
nucleotide composition.
Across the tested panel, high-gain gated-FFN rows form a recurrent structural phenotype across both
text and genomic foundation models, but the causal census shows that structural prominence is only an
enrichment signal for functional importance, not a calibrated measure of effect size. Across 22 models,
frozen activation-selected candidates usually produced larger native-loss changes than five same-layer
controls under both partial suppression and full ablation, yet their effects ranged from negligible or negative

changes to several-hundred-percent loss increases. Neither the leading spectral concentration q1 nor operator
magnitude predicted the signed full-ablation effect within architecture class, and the highest-norm rows in a
candidate’s own layer were themselves causally close to inert. We therefore treat singleton functional
criticality and multi-component causal response complexity as separate questions. Focused case studies show
that DNABERT-2 contains a robust pretrained pair interaction, whereas disruption of a spectrally
concentrated GENERator row produced a large compositional shift that tracked damage magnitude and did
not require the row’s learned direction. These results support a three-way distinction between structural
geometry, functional criticality, and causal response complexity, and argue against treating the super-weight
phenomenon as a single universal mechanism.

## Results

### High-gain rows are extreme in magnitude, but spectral concentration is not independently distinctive

We first asked whether frozen activation-derived high-gain gated-FFN rows share a structural signature
across foundation models trained on text and genomic sequence. As a calibration, retrospective cold-weight
analysis recovered the published super-weight row at rank 1 in Llama-7B, Mistral-7B, and OLMo-7B (Fig.
1A). Because the diagonal decomposition used for this calibration omits cross terms between hidden units,
all subsequent cross-model structural analyses used the exact gated-FFN bilinear operator 𝑈𝑘, with its
Frobenius norm measuring operator magnitude and its singular spectrum measuring spectral concentration.
Among the 23 models with frozen activation-derived structural candidates, the leading singular-energy
fraction 𝑞1 varied substantially, from 0.389 in NTv3 to 0.9996 in EuroBERT-610M (Fig. 1B). Near-rank-1
candidates occurred in both decoder and encoder architectures: for example, EuroBERT-210M,
EuroBERT-610M, EuroBERT-2.1B, and ModernBERT-large had 𝑞1 = 0. 984, 0.9996, 0.998, and 0.970,
respectively. Conversely, several decoders were less concentrated, including Qwen2.5-3B (𝑞1 = 0. 817) and
GENERator-PROK-1.2B (𝑞1 = 0. 847). A cross-model regression,

(

)

𝑞1 = β0 + β1𝑙𝑜𝑔10 𝑁𝑛𝑜𝑛𝑒𝑚𝑏 + β2𝐼𝑑𝑒𝑐𝑜𝑑𝑒𝑟 + ε,
2

explained little of the observed variation (𝑅 = 0. 158). Neither the decoder term (β2 = 0. 1162 ± 0. 0860,
𝑝 = 0. 192) nor the log-parameter term (β1 = 0. 0143 ± 0. 0637, 𝑝 = 0. 824) provided convincing
evidence for a simple architecture- or scale-based explanation of 𝑞1. Text and genomic models also
overlapped substantially in spectral concentration. Evo2-7B was evaluated with the same detector, but its
maximum activation ratio was 2.22, below the predefined threshold of 5.0; no candidate was therefore
accepted and no 𝑞1 value was assigned.
A much more consistent pattern emerged from within-layer controls. In 12 models analyzed with a common
same-layer control protocol, the detected candidate was compared with five distinct ordinary rows from the
same layer. The candidate had higher 𝑞1 than every one of its five controls in all 12 models (Fig. 1C). The
candidate-minus-mean-control gap averaged 0.884 and ranged from 0.544 to 0.977. For example,
Qwen2.5-1.5B had 𝑞1 = 0. 995 versus a control mean of 0.018, EuroBERT-610M had 𝑞1 = 0. 9996 versus
0.032, GENERator-PROK-3B had 𝑞1 = 0. 935 versus 0.009, and Qwen2.5-3B had 𝑞1 = 0. 817 versus

0.027. MosaicBERT and ModernBERT-base, analyzed with the same substantive control design but a
separate fixed seed allocation, showed compatible candidate-control separation (0. 477 versus 0. 190 and
0. 897 versus 0. 168, respectively) and were not pooled into the 12-model summary statistic.
Operator magnitude remained conceptually distinct from spectral concentration. The exact Frobenius norm
‖ 𝑈𝑘‖𝐹 measures the magnitude of the row-associated operator, whereas 𝑞1 measures the fraction of its
squared singular-value mass concentrated in the leading direction. Because absolute Frobenius norms are
parameterization-dependent, they were not interpreted as directly comparable cross-model scores. This local
comparison, however, is against ordinary rows carrying 3–8% of the candidate’s operator norm, and it does
not survive a stricter control. Repeating the comparison against the five highest-norm rows in the candidate’s
own layer, the median candidate-minus-control q1 gap falls from 0.935 to 0.014 — a 69-fold collapse — and
in 7 of 22 models at least one high-norm neighbour is more spectrally concentrated than the candidate
(SmolLM2-360M,
EuroBERT-210M,
EuroBERT-2.1B,
ModernBERT-large,
DNABERT-2,
GENERator-EUK-3B, and GenomeOcean-4B; Fig. 1C). The transferable structural statement is therefore
narrower than it first appears: activation-selected candidates are the highest-norm row in their layer by a wide
margin and are near-rank-1 in concentration, but conditional on high norm their concentration is not
distinctive. The absolute degree of concentration also varies substantially across model families and does not
form a clean architecture, domain, or scale partition (Fig. 1B).

Figure 1. High-gain gated-FFN rows are the highest-norm rows in their layer, but are not spectrally
distinctive among high-norm rows. (A) Retrospective cold-weight calibration recovers the published
super-weight row at rank 1 in Llama-7B, Mistral-7B, and OLMo-7B; the scalar top-1 share is shown
only as the diagonal calibration metric. (B) Exact spectral concentration q₁ for all 23 models with
accepted activation-based candidates, sorted by q₁. Marker shape denotes encoder/decoder
architecture and colour denotes text/genomic domain. Evo2-7B is annotated separately because no
candidate passed the predefined detector threshold (maximum ratio 2.22 < 5.0), and no q₁ value is
assigned. (C) Candidate q₁ against two same-layer control sets across the 22-model census cohort,
stacked on a shared x-axis of models. Top: five randomly sampled same-layer controls; the candidate
exceeds all five in every model and the median candidate-minus-control gap is 0.935. Bottom: the five
rows ranked immediately below the candidate by exact Frobenius norm in the same layer; the median
gap falls to 0.014, and in seven models at least one high-norm neighbour is more concentrated than the
candidate (labelled). Both control sets are defined without reference to any causal measurement.

Frozen structural candidates are functionally enriched relative to same-layer
controls, but functional effects are heterogeneous
We next asked whether the structurally selected rows identified before causal evaluation were functionally
consequential across the broader model panel. The completed census contained 22 models: 10 text decoders,
six text encoders, four genomic decoders, and two genomic encoders. For each model, one frozen primary
candidate was compared with five independently evaluated same-layer control rows. Each row was tested
separately at two intervention strengths: partial suppression (ε=0.5, corresponding to a 50% row scale) and
full ablation (ε=1.0). Functional effects were measured as the signed relative change in the model's native
language-model loss, so positive values indicate loss degradation and negative values indicate improved loss
after perturbation.
At partial suppression, the candidate produced a larger signed loss change than the within-model median
control in 18 of 22 models (Fig. 2A). Across models, the median candidate-minus-control gap was +0.60%,
with a 95% model-bootstrap interval of +0.23% to +3.06%. The median control effect itself was
approximately zero. Candidate effects were nevertheless highly heterogeneous, ranging from -0.84% in
EuroBERT-610M and -0.50% in NTv3 to +10.93% in ModernBERT-base and +301.38% in Mistral-7B.
Thus, partial suppression already separated candidates from ordinary rows at the cohort level, but did not
support a universal positive effect in every model.
Full ablation strengthened the cohort-level separation while preserving this heterogeneity. The candidate
exceeded the median same-layer control in 20 of 22 models, and the median candidate-minus-control gap
increased to +0.88% (95% model-bootstrap interval +0.61% to +111.78%; Fig. 2B). The unusually wide
upper bound reflects the strongly heavy-tailed distribution of effects across this heterogeneous model panel.
Full-ablation candidate effects were highly heterogeneous, ranging from -0.79% in NTv3 and approximately
zero in EuroBERT-2.1B to +111.78% in OLMo-7B, +300.79% in Llama-7B, +357.04% in
ModernBERT-base, and +698.87% in SmolLM2-1.7B. NTv3 and EuroBERT-2.1B had nonpositive
candidate-minus-control gaps under full ablation, so the data do not support a universal criticality claim.
Instead, the common pattern is enrichment: the frozen structural candidate is usually more functionally
consequential than ordinary rows from the same layer, while the magnitude and even sign of the response
remain model dependent.

### Top-norm rows do not reproduce the candidate’s causal enrichment

Randomly sampled same-layer rows are a weak comparison. Across the panel, the frozen candidate was the
highest-norm row in its layer by a wide margin, and the five random controls carried between 3% and 8% of
the candidate’s exact operator norm. An enrichment measured against such rows could reflect nothing more
than the difference between a large row and a typical one.
We therefore repeated the census against a second, stricter control set defined without reference to any causal
measurement: the five rows ranked immediately below the candidate by exact Frobenius norm within the
candidate’s own layer. This selection is purely weight-space — it requires the model’s frozen parameters and
no forward pass — whereas the activation-based detector that fixed the candidate requires evaluation on real
input sequences. The comparison therefore asks whether data-dependent selection identifies rows that a
magnitude-based criterion does not.

The top-norm control rows were causally close to inert. Their median relative loss change was +0.035% at
partial suppression and +0.110% at full ablation, against candidate effects reaching several hundred percent.
Although these medians were distinguishable from the random-control medians by a paired test across
models (Wilcoxon signed-rank, p=0.0009 at ε=0.5 and p=0.0001 at ε=1.0), the difference is negligible in
magnitude: selecting the highest-norm rows in a layer rather than arbitrary rows changed the median control
effect by roughly one tenth of one percent. Consequently, candidate–control separation was qualitatively
unchanged under the stricter control.
The candidate exceeded the median top-norm control in 20 of 22 models at partial suppression and 19 of 22
at full ablation (Fig. 2D). Fourteen models showed a decisive separation with no instability at either strength;
four showed consistently positive but negligible gaps in which candidate and controls were both nearly inert
(EuroBERT-210M, EuroBERT-610M, EuroBERT-2.1B, Qwen2.5-1.5B); and four showed a sign flip, in
which a top-norm control row was causally competitive with or exceeded the candidate at one or both
strengths (MosaicBERT at ε=0.5, Qwen2.5-0.5B and Qwen2.5-7B at ε=1.0, and NTv3 at both). NTv3 is the
only model with a negative candidate-minus-control gap against both control sets at both intervention
strengths.
We could not identify a structural property that predicts which models fall into this minority. In particular, the
candidate’s own margin of spectral concentration over its top-norm neighbours does not: the rank correlation
between the absolute structural margin and the causal gap was ρ=0.311 (p=0.159) at ε=0.5 and ρ=0.355
(p=0.105) at ε=1.0. Several models with small or negative structural margins — DNABERT-2,
GENERator-EUK-3B, ModernBERT-large — were among the most decisively separated causally. We
therefore report the exceptions as an unexplained minority rather than as a predictable regime.
The result of substantive interest is the dissociation this comparison exposes within individual models. In
DNABERT-2 and GENERator-EUK-3B, the frozen candidate is less spectrally concentrated than at least one
of the five highest-norm rows in its own layer, yet causes one to three orders of magnitude more damage
when ablated. Weight-space descriptors and causal response rank these rows in opposite orders.

The wide range of causal effects allowed a direct test of whether structural descriptors predict the signed
full-ablation effect. Neither concentration nor magnitude does. Across the 22 frozen candidates, q1 was
essentially uncorrelated with the signed full-ablation relative loss change (Spearman ρ=0.074, two-sided
p=0.744; 95% model-bootstrap CI -0.391 to 0.513; Fig. 2C). Highly concentrated candidates could therefore
be either weakly or catastrophically causal, and less concentrated candidates were not uniformly inert.
Operator magnitude showed a nominal cross-panel association at full ablation that did not survive scrutiny.
Layer-relative Frobenius magnitude correlated with the signed full-ablation effect at ρ=0.441 (p=0.040, 95%
bootstrap CI 0.062 to 0.717), but the same association at partial suppression was not significant (ρ=0.348,
p=0.112), and no correction was applied across the two predictors and two intervention strengths tested.
Restricting to text decoders, where the evaluation endpoint is uniform, both associations vanish and change
sign (Frobenius ρ=−0.042, p=0.907; q1 ρ=−0.333, p=0.347 at ε=1.0). Consequently, The top-norm control
experiment argues against a simple monotonic relationship between operator magnitude and causal
importance: rows ranked immediately below the candidate by magnitude are generally close to inert despite
being the largest available weight-space controls. This experiment does not exclude a nonlinear threshold
effect specific to the extreme rank-1 row.–control separation was qualitatively unchanged under the stricter
control. The cross-panel association is not robust to restriction to the endpoint-homogeneous text-decoder
subset and therefore should not be interpreted as evidence for a general magnitude–effect relationship.

Descriptively, text decoders had a larger median full-ablation effect (+85.47%) than text encoders (+0.65%),
genomic decoders (+0.68%), or genomic encoders (+0.02%), but these groups differ in objective, tokenizer,
architecture, scale, and panel composition; we therefore do not interpret the subgroup medians as evidence
that architecture or domain determines causal importance.
This census changes the level at which the structural-function relationship can be stated. Structural selection
identifies rows that are usually enriched for functional importance relative both to arbitrary local controls and
to the highest-magnitude rows in the same layer, but neither q1 nor operator magnitude specifies how large
the causal effect will be. The census is intentionally a singleton experiment: because one primary row was
intervened per model, it does not determine whether multi-row high-gain systems combine additively or
through interactions. We therefore treat causal response complexity as a separate mechanistic question and
analyze it only in focused case studies.

Figure 2. Frozen structural candidates are usually more functionally consequential than same-layer
controls, including the highest-norm ones, but neither structural descriptor predicts effect size. (A)
Signed relative native-loss change after partial suppression (ε=0.5; row scale α=0.5) for one frozen
candidate and five random same-layer controls in each of 22 models. Diamonds denote candidates,
grey points individual controls, and short horizontal bars within-model control medians. Models share
one order determined by the full-ablation candidate effect. The inset reports the number of models in
which the candidate exceeded the median control and the across-model median
candidate-minus-control gap with its 95% model-bootstrap interval. (B) The same analysis under full
ablation (ε=1.0; α=0). Signed symmetric-log axes retain negative, sub-percent, and catastrophic

responses on the same scale. (C) Candidate spectral concentration q₁ against the signed full-ablation
relative native-loss change. Spectral concentration was not associated with the signed full-ablation
effect (Spearman ρ=0.074, 95% bootstrap CI −0.391 to 0.513, p=0.744, n=22). (D)
Candidate-minus-top-norm-control gap at both intervention strengths, on a symmetric-log axis. The
candidate exceeds the median top-norm control in 20 of 22 models at ε=0.5 and 19 of 22 at ε=1.0; the
four models with a negative gap at either strength are labelled. Top-norm controls are themselves
causally close to inert (median effects +0.035% and +0.110%), so this comparison leaves the
candidate-minus-control gap essentially unchanged relative to random controls.

### DNABERT-2 contains a robust pairwise interaction on its pretrained objective

The cohort-wide singleton census establishes whether frozen structural candidates are functionally
enriched, but it does not determine how multiple high-gain components combine. DNABERT-2 provides a
focused setting in which to test that separate mechanistic question on the pretrained masked-language-model
objective. We analyzed a fixed 10-row high-gain basis on the pretrained masked-language-model objective,
using 256 hg38 windows and a fixed masking realization. Two identified rows, L9/r264 and L9/r294, had
small singleton effects at full ablation (delta loss +0.0218 and +0.0064), but their joint ablation increased loss
by +2.0400. The resulting epistasis, defined as d_AB-d_A-d_B, was +2.0118 (Fig. 3A,B). At partial
suppression (epsilon=0.5), epistasis remained positive (+0.0528).
To test whether this pair reflected a broader interactional response rather than an isolated two-row
anomaly, we applied the observer-family progression motivated by the mechanistic tomography framework
[10] to combinations of the 10 fixed rows. F0 used measured singleton effects additively; F1 applied a single
fitted calibration scalar; F2 jointly fit additive main effects; and F3 added all 45 pairwise interaction terms.
This progression tests whether finite-response mismatch can be repaired by calibration or instead requires
interaction features. Models were fit and calibrated on disjoint mask pools and evaluated on 20 held-out
masks at each intervention strength.
At epsilon=0.5, held-out R2 increased from -0.023 (F0), 0.324 (F1), and 0.517 (F2) to 0.888 for F3.
Pairwise lifting reduced held-out MAE by 54.7% relative to F2, with the bootstrap confidence interval
excluding zero. At epsilon=1.0, F0, F1, F2, and F3 achieved R2 values of -0.040, 0.659, 0.581, and 0.790,
respectively, and F3 improved MAE by 24.4%, again with a confidence interval excluding zero (Fig. 3C).
Pairwise terms therefore materially improve prediction at both scales, although they do not fully account for
every finite-response pattern under full ablation.
The known L9/r264-L9/r294 interaction also re-emerged without special treatment during fitting. Among
all 45 pairwise F3 coefficients, this pair ranked first by absolute coefficient at both intervention strengths
(Gamma=0.077 at epsilon=0.5 and 0.739 at epsilon=1.0; Fig. 3D). Thus, DNABERT-2 provides a robust
example in which a high-gain system is genuinely interaction-dependent on the model's pretrained objective.

Figure 3. DNABERT-2 mechanistic organization on the pretrained masked-language-model endpoint.
(A) Full-ablation effects of L9/r264 and L9/r294 individually and jointly. (B) Pair epistasis at ε=0.5 and
ε=1.0, defined as the joint loss change minus the sum of the two singleton loss changes. These values
derive from a standalone regression check rather than from the mask pools used for the
observer-family fits. (C) Held-out predictive performance of observer families F0–F3 across 20
held-out row-subset conditions at each intervention strength. (D) Pairwise F3 coefficients;
L9/r264–L9/r294 ranks first among all 45 pairs at both intervention strengths despite receiving no
special treatment during fitting. F3 outperformed F2 on held-out MAE in 100 of 100 partition resplits
at both strengths.

### Disrupting a high-gain GENERator location produces a damage-linked compositional shift

GENERator EUK provides a complementary genomic-decoder case. Its primary candidate, L4/r2371, has
an exact q1 of 0.969, placing it among the most spectrally concentrated operators in the genomic panel (Fig.
4A). A separate reproduction identified a strong beginning-of-sequence (BOS) phenotype: position 0
received 37.9% of mean incoming attention, 33.0 times the uniform expectation, and 78.8% of layer-head
observations had their maximum attention at position 0. The maximum high-gain activation was also
observed at position 0 (Fig. 4B). These observations establish co-occurrence, not causation; no experiment
independently manipulated the high-gain row and the attention sink, and the causal-decoder attention mask
itself makes position 0 a special case.
We next tested whether the large compositional effect associated with row 2371 reflected the specific learned
direction of that row or a more general sensitivity of its location. In the damage-matching experiment, mean
GC fraction was 0.4204 at baseline and 0.3065 after full ablation of row 2371, a decrease of 0.1139. Five
non-2371 control rows produced essentially no GC shift even under strong perturbation, and none
approached the native-loss damage of row 2371. These rows were not merely under-damaged but causally
insensitive: scaling each across α ∈ [0, 8] moved native loss by 0.001–0.002, against a 2.37-unit gap to the
row-2371 ablation target of 8.754.
We therefore tested a second specificity axis by replacing the weight vector at the row-2371 location with a
fixed random unit direction scaled by a factor c, so that c=0 recovers ablation and increasing c restores
progressively more of the model’s original loss. Across c ∈ {0.0125, 0.5, 1.0, 3.0, 8.0}, generated GC
composition tracked native-loss damage monotonically: at c ≤ 1.0, where the perturbation retains 98–100%
of the ablation damage, GC was 0.3068–0.3078 against 0.3065 for ablation itself, and at c=8.0, where
damage falls to 84%, GC recovered to 0.3402, closing 29.6% of the gap to the intact baseline of 0.4204. Loss
recovered more slowly than composition over the same range (16% versus 29.6%). Instead, the evidence
identifies this row location as an unusually load-bearing point whose disruption strongly biases generated
composition. Because the non-2371 controls did not reach comparable native-loss damage, however, these
experiments do not exclude the possibility that an equally catastrophic perturbation elsewhere could produce
a similar shift.
A random-direction replacement at the row-2371 location therefore reproduces the low-GC phenotype when
native-loss damage remains close to ablation. This argues against a simple interpretation in which
preservation of the learned row-2371 direction is required for the phenotype. However, the experiment does
not establish full direction-independence: even at c=8.0, eight times the row’s original norm, the random
replacement retains 84% of the ablation damage. The tested random directions therefore remain functionally
close to removal, limiting the ability of this control to distinguish direction-specific effects from a more
general sensitivity to disrupting this location. The evidence is best interpreted as identifying row 2371 as an
unusually load-bearing location whose disruption is associated with a large shift in generated composition.
The GC decrease was also not explained primarily by a single degenerate homopolymer event: the longest
homopolymer accounted for only 3.9% of the GC decrease, and enriched short motifs after ablation were
broadly AT-rich. Together, these observations support a broad compositional shift rather than a narrow
repeat-collapse artifact.

Together, the DNABERT-2 and GENERator case studies illustrate distinct functional realizations of
high-gain structure. DNABERT-2 exhibits strong pairwise dependence on its pretrained encoder objective,
whereas GENERator contains a spectrally concentrated, load-bearing location whose disruption strongly
biases generated composition without requiring its learned weight direction. Neither realization can be taken
as the universal mechanism of high-gain rows.

Figure 4. Generated composition tracks damage magnitude at a concentrated high-gain location in
GENERator EUK. (A) Exact q₁ of the primary high-gain row L4/r2371. (B) Reproduced
beginning-of-sequence (BOS)-centred attention/activation phenotype: mean attention mass at position
0, fold enrichment over the uniform expectation, fraction of layer-head observations with maximum
attention at position 0, and co-localized activation maximum. This panel establishes co-occurrence only
and does not imply that the high-gain row causes the attention sink. (C) Left: generated GC fraction
against the scale c of a fixed random unit direction replacing the weight vector at the row-2371
location, with the ablation and intact-baseline levels marked. Because the row is replaced rather than
perturbed additively, c=0 is equivalent to ablation and damage decreases with increasing c. Right:
generated GC fraction against native-loss damage for the row-2371 α-sweep, five non-2371 control
rows, and the random-direction grid on shared axes. Composition tracks damage monotonically: at
matched damage a random direction reproduces the low-GC phenotype, and at the largest tested scale
(c=8.0, 84% of ablation damage) GC recovers 29.6% of the way to baseline. The five control rows
moved native loss by 0.001–0.002 across α ∈ [0, 8] and are therefore causally insensitive rather than

merely under-damaged, so these experiments do not establish that the compositional shift is unique to
this location.

## Discussion

This study separates three quantities that are easily conflated when discussing super-weight-like
phenomena: structural geometry, functional criticality, and finite-intervention causal response complexity.
High-gain gated-FFN structures recur across both text and genomic foundation models, supporting the idea
that the phenomenon is not restricted to linguistic data. Yet the causal role of those structures is not fixed. A
row can be near-rank-1 and strongly causal, near-rank-1 but weakly causal, one of several interacting
components, or part of a regime whose response is not captured by second-order models.
The structural panel does not support a simple architecture- or scale-based partition of spectral
concentration. Near-rank-1 candidates occur in both encoders and decoders, while several decoders are
substantially less concentrated. Across the 23 models with accepted candidates, a regression of q1 on log10
non-embedding parameter count and decoder status explained little of the observed variation (R²=0.158),
with neither term providing convincing evidence of association. The apparent local result — the candidate
outscoring ordinary same-layer rows on q1 — is strongly attenuated after conditioning on operator
magnitude. Against the five highest-norm rows in the same layer, the median concentration gap collapses
69-fold and reverses sign in 7 of 22 models. Against the five highest-norm rows in the same layer the median
gap collapses 69-fold and reverses sign in 7 of 22 models. The transferable structural phenotype is therefore
that these rows are the highest-norm rows in their layer and are near-rank-1; conditional on that, their
concentration is unremarkable.
The 22-model intervention census provides a broader test of the relationship between structure and
function than the earlier causal case studies. At both intervention strengths, the frozen candidate usually
produced a larger loss change than five ordinary rows from the same layer, and the cohort-median
candidate-minus-control gap remained positive under model-level bootstrap resampling. Structural selection
therefore appears to enrich for functional importance. That enrichment is not explained by magnitude.
Repeating the census against the five highest-norm rows in each candidate’s layer left the
candidate-minus-control gap essentially unchanged, because those high-norm rows are themselves causally
close to inert. Because top-norm selection is purely weight-based whereas the candidate selection uses
forward-pass activations, the comparison shows that weight-only magnitude ranking does not recover the
same causal ordering as activation-derived selection. This does not hold universally: in 4 of 22 models a
high-norm control row was causally competitive with or exceeded the candidate at one or both intervention
strengths, and no structural property we tested — including the candidate’s own margin of concentration over
those neighbours — accounts for which models those are. That enrichment does not amount to a universal
effect-size law. Full-ablation responses ranged from slightly negative or near-zero changes to almost
seven-fold increases in native loss, and spectral concentration q1 was essentially unrelated to the signed
full-ablation effect. A row can therefore be strongly near-rank-1 yet weakly causal, or comparably
concentrated and catastrophically important. The transferable signal is the existence of a locally exceptional
high-gain structure; its functional severity must still be measured.
DNABERT-2 adds a particularly strong mechanistic result because the same row pair is supported by
direct joint intervention, held-out predictive tomography, and coefficient ranking. The pair was part of a fixed
basis and was not privileged during model fitting, yet ranked first of 45 pair interactions at both scales. This
triangulation supports a genuine interaction on the pretrained objective while remaining agnostic about its
magnitude on downstream tasks.

GENERator demonstrates a different type of functional realization. Disrupting row 2371 produces a large
shift toward lower GC content, and generated composition tracks the magnitude of loss damage at that
location rather than the identity of the direction removed: a random direction at matched damage reproduces
the phenotype, and restoring damage restores composition monotonically. The supported interpretation is
therefore damage-tracking fragility at this location rather than direction-specific control of GC composition,
with the caveat that a random direction cannot be made both large and low-damage, which limits how
sharply this control discriminates. The associated BOS attention and activation pattern is striking but should
not be overinterpreted: recent work has emphasized that massive activations and attention sinks can co-occur
for architectural reasons and need not share a single causal role [2-4]. Our data establish co-localization in
GENERator, not a causal link between the high-gain row and the sink. Because the non-2371 controls did not
reach equally severe native-loss damage, we also cannot exclude the possibility that similarly catastrophic
perturbations elsewhere would yield a related compositional shift.
Several limitations bound this interpretation. Although the singleton census spans 22 models, the panel is
heterogeneous and is not a random sample from a defined population of foundation models. Native endpoints
differ across causal decoders and masked-language encoders, as do tokenizers, scales, and sequence domains,
so raw subgroup differences cannot be assigned causally to architecture or training domain. The cohort
experiment perturbs one frozen primary row per model against five same-layer controls and therefore
measures singleton functional criticality, not interaction order or the causal organization of a multi-row basis.
Full ablation is also a strong finite intervention and can enter nonlinear or catastrophic regimes. The four
models in which a high-norm control row matched or exceeded the candidate remain unexplained; we tested
the candidate’s structural margin as a predictor and it does not account for them. Finally, NTv3's original
model-weight revision was not recorded exactly, and the DNABERT-2 discovery and causal-evaluation
preprocessing are intentionally different assays. These limitations motivate treating the cohort census as
evidence for functional enrichment and structure-function dissociation, while reserving mechanistic claims
for the focused intervention studies.
The broader implication is that high-gain structure is the more transferable object. Exact operator analysis
can identify a recurring structural phenotype across domains, but it does not by itself specify which
component matters most, whether components combine additively, or which biological or linguistic function
will be affected. Mechanistic interpretation therefore requires a second stage of causal measurement rather
than inference from geometry alone. The held-out, stepwise observer-testing logic of mechanistic
tomography [10] provides a principled way to perform that second stage without escalating model
complexity merely because richer interactions are available. This separation offers a more conservative
framework for studying extreme parameter pathways in both language and genomic foundation models.

## Methods

### Model panel and checkpoints

We analyzed gated feed-forward structures across 23 foundation models with accepted activation-based
candidates. The text-decoder panel comprised Llama-7B, Mistral-7B, OLMo-7B, Phi-3-mini, Qwen2.5-7B,
Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, SmolLM2-135M, SmolLM2-360M, and SmolLM2-1.7B. Text
encoders comprised MosaicBERT, ModernBERT-base, ModernBERT-large, EuroBERT-210M,
EuroBERT-610M, and EuroBERT-2.1B. Genomic decoders comprised GENERator-v2-eukaryote-3B,
GENERator-v2-prokaryote-1.2B, GENERator-v2-prokaryote-3B, and GenomeOcean-4B; genomic encoders
comprised DNABERT-2 and NTv3. Evo2-7B was evaluated with the same activation-based detection
procedure but did not yield an accepted candidate. Repository identifiers and checkpoint revisions were

recorded for all measurements where available. For models originally evaluated without an explicit revision,
the resolved Hub commit from the completed run is reported when recoverable; NTv3 remains an exception
because its original model-weight revision was not recorded exactly.

### Exact gated-FFN operator and structural metrics

For output row 𝑘 of a gated feed-forward block, let 𝑔𝑖 and 𝑢𝑖 denote the gate- and up-projection row vectors
for hidden unit 𝑖, and let 𝑑𝑘,𝑖 denote the corresponding coefficient in row 𝑘 of the down projection. We
defined the row-associated bilinear operator as
𝑇

𝑈𝑘 = ∑ 𝑑𝑘,𝑖 𝑔𝑖𝑢𝑖 .
𝑖

Its exact squared Frobenius norm can be evaluated without explicitly materializing all cross terms using the
Gram identity
2

𝑇

‖ 𝑈𝑘‖𝐹 = 𝑑𝑘 𝐾𝑑𝑘,
where

( )( )
𝑇

𝑇

𝐾𝑖𝑗 = 𝑔𝑖 𝑔𝑗 𝑢𝑖 𝑢𝑗 ,
that is, 𝐾 is the Hadamard product of the gate- and up-projection Gram matrices. Singular values {σ𝑗} were
computed in float64. Spectral concentration was quantified as
2

𝑞1 =

σ1
2

,

∑σ𝑗
𝑗

and the spectral participation ratio as
2

( ).
2

𝑃𝑅𝑠𝑝𝑒𝑐 =

∑σ𝑗
𝑗

4

∑σ𝑗
𝑗

Operator magnitude was quantified by

‖ 𝑈𝑘‖𝐹 =

2

∑ σ𝑗 .
𝑗

Thus, 𝑞1 measures concentration of operator energy in the leading singular direction, whereas ‖ 𝑈𝑘‖𝐹
measures operator magnitude. For within-layer magnitude comparisons, the layer-relative Frobenius
magnitude was defined as the candidate norm divided by the median norm of the five same-layer control
rows. A diagonal proxy that retains only per-hidden-unit terms omits interactions between distinct hidden
units; it was used only for the retrospective Llama/Mistral/OLMo calibration in Fig. 1A and not for
cross-model structural comparisons.

### Candidate provenance

Candidate coordinates in this panel derive from two protocols, and we distinguish them explicitly rather than
describing a single uniform rule. Sixteen models were selected under the activation-ratio detector, with the
selected coordinate’s ratio recorded at or above the acceptance threshold of 5.0. Six models carry coordinates
fixed under an earlier activation-magnitude protocol and retained rather than reselected: Llama-7B,
Mistral-7B, OLMo-7B, DNABERT-2, NTv3, and GENERator-EUK-3B.
For Llama-7B, Mistral-7B, and OLMo-7B the coordinates are those published by Yu et al. and serve as
calibration rather than as detector output; the ratio detector independently recovers the published coordinate
in Llama-7B and Mistral-7B. In OLMo-7B it does not: the ratio maximum falls at L2/r269 and the activation
maximum at L30/r269, both sharing the published coordinate’s row index, so OLMo’s headline coordinate is
one instance of a depth-recurring row-269 family rather than a coordinate uniquely determined by the
detector.
For DNABERT-2, the frozen coordinate L5/r603 is the global activation maximum under the canonical
discovery preprocessing (out_max = 944.5556, activation ratio 152.7345) but is not the global ratio
maximum, which falls at L8/r603. The two rules disagree substantially here: the ratio-selected coordinate has
q1 = 0.379, below the minimum of the structural panel, while the frozen candidate has q1 = 0.793. In NTv3
the two rules select L11/r1472 and L6/r1472 respectively, differing in layer but not in row index and in q1 by
less than 0.01. We retain the activation-based coordinates for these models and report the discrepancies rather
than reconciling them retrospectively.
Detector execution for DNABERT-2 depends on tokenizer special-token handling: the historical discovery
path uses default special tokens and reproduces the frozen activation maximum exactly, whereas the
structural census path disables them and assigns the same coordinate an activation ratio near 1.0. Attention
implementation has no effect on either result, and structural metrics for a given coordinate are identical
across all four combinations, so the discrepancy is one of input preprocessing and not of model state.
Model checkpoints were revision-pinned where recorded. Resolved Hub commits are available for 21 of 22
models; NTv3’s original model-weight revision was not recorded exactly and is not recoverable. Per-model
provenance is given in Supplementary Table S1.
Text ratio-based discovery used the first 20 non-empty lines of the WikiText-2 raw test split, joined and
tokenized with truncation to a maximum length of 512 tokens. Genomic discovery used the fixed 504-bp
ACTB-derived sequence with model-specific tokenizer preprocessing. DNABERT-2 L5/r603 was defined by
the maximum-activation discovery criterion using tokenizer-default special tokens; under this canonical
preprocessing it is the global activation maximum (out_max=944.5556, activation ratio 152.7345),
although it is not the global ratio maximum. Calibration used batch size 1. Evo2-7B was evaluated with the
ratio-based procedure and no candidate passed the predefined criterion (maximum ratio 2.22), so no
structural spectrum was assigned.
For the common 12-model local-control analysis, five distinct ordinary rows were sampled from the detected
candidate's layer, excluding the candidate, using a fixed NumPy SeedSequence(42) allocation. The same
exact spectral metrics were then computed for the candidate and each control row. MosaicBERT and
ModernBERT-base used the same substantive five-row, same-layer control design but a different fixed

SeedSequence(42) child allocation; their control results were therefore reported as compatible supporting
measurements but were not pooled into the 12-model control summary.

### Structural statistical analysis

Exact operator spectra are deterministic single-checkpoint calculations and were therefore shown without
sampling error bars. For each of the 12 models in the common local-control analysis, structural
exceptionalness was summarized as candidate 𝑞1 minus the mean 𝑞1 of the five same-layer controls; the
manuscript reports the mean and range of this quantity across models. To test whether a simple
architecture/scale model explained cross-model spectral concentration, we fit the ordinary least-squares
regression

(

)

𝑞1 = β0 + β1𝑙𝑜𝑔10 𝑁𝑛𝑜𝑛𝑒𝑚𝑏 + β2𝐼𝑑𝑒𝑐𝑜𝑑𝑒𝑟 + ε
across the 23 models with accepted candidates. Because the model panel is heterogeneous and not a random
sample from a defined population of foundation models, coefficient tests are interpreted descriptively rather
than as population-level architectural laws.
The paired comparison between top-norm and random control effects used a Wilcoxon signed-rank test over the
22 model-level median control effects at each intervention strength. Correlations between structural descriptors
and causal effect were computed on the full panel and, separately, within architecture class; panels with fewer
than eight models are reported as underpowered and are not used to support any claim. No
multiple-comparison correction was applied across models, predictors, or intervention strengths.

### Cross-model singleton functional-criticality census

The functional-criticality census included 22 models with one frozen primary candidate per model: 10 text
decoders, six text encoders, four genomic decoders, and two genomic encoders. Phi-3 was excluded from this
22-model singleton analysis by the frozen scope definition because its pre-existing causal basis comprised six
rows rather than one comparable primary row; its prior tomography remains a mechanistic case study rather
than part of the cohort statistic. Candidate coordinates were frozen before causal evaluation and were not
changed after observing causal responses. For each model, five distinct same-layer control rows were
sampled without replacement using a deterministic SeedSequence(42) child stream assigned by panel index,
excluding the frozen candidate set in that model/layer. Candidate and control rows were always evaluated
one at a time.
Two control sets were constructed per model, both defined without reference to any causal measurement.
Random same-layer controls were five distinct rows sampled without replacement from the candidate’s layer
using a deterministic SeedSequence(42) child stream assigned by panel index, excluding the frozen
candidate. Top-norm same-layer controls were the rows ranked 2 through 6 by exact Frobenius norm within
the same layer, computed for all rows simultaneously via the Gram identity and never materializing an
individual operator; the candidate was rank 1 in every model tested. Norm-matched controls were considered
and found to be unconstructible, since the candidate exceeded its layer median norm by factors of 12 to 32
and no row in the candidate’s layer fell within 20% of the candidate’s norm in any model examined. The
top-norm sweep used the same endpoints, window pools, masking realizations, batch constructions, and
evaluation harness as the original census, differing only in which row indices were perturbed.

### Row intervention

For a targeted down-projection row w, we cloned the original row, applied multiplicative suppression,
evaluated the model, and restored the cloned row in a context-manager/finally path. The intervention was
wε = (1 - ε) w , ε ∈ {0.5, 1.0}.
Thus ε=0.5 implemented 50% row scaling and ε=1.0 set the targeted row to zero. Models were placed in
evaluation mode and loaded/evaluated in float32; no autocast was declared. Candidate and control rows were
never perturbed jointly.

### Native-objective evaluation endpoints

Text decoders (n=10) were evaluated on WikiText-2-raw-v1 test text. Nonempty lines were shuffled with
random.Random(42), concatenated, and tokenized separately for each model into 100 exactly 512-token
windows. The endpoint was teacher-forced shifted-label mean token negative log-likelihood (NLL): logits at
positions 0…T-2 predicted labels at positions 1…T-1, and summed NLL was divided by the number of
predicted tokens. Models with more than 3B loaded parameters used batch size 4 (25 stored bootstrap units);
smaller models used batch size 8 (13 units, including the final partial batch).
Text encoders (n=6) used the same WikiText construction but 256 windows of 512 tokens, evaluated in
batches of 16. A single 15% masking realization was generated with seed 42 and reused for baseline and
every intervention condition. CLS, SEP, and PAD tokens were excluded from masking; selected tokens were
replaced by MASK and all unselected labels were set to -100. The endpoint was summed cross-entropy over
masked tokens divided by the masked-token count. MosaicBERT used the bert-base-uncased tokenizer; the
remaining text encoders used their checkpoint tokenizers.
Genomic decoders (n=4) were evaluated on hg38 sequence windows derived from random_262kb.bed. A
seed-42 disjoint partition supplied 100 damage windows of 512 bp with less than 1% N content. For
GENERator models, the leftmost len mod 6 bases were trimmed, a BOS token was prepended, and
tokenization used special tokens disabled; GenomeOcean used no 6-bp trim or forced BOS and also
tokenized with special tokens disabled. The endpoint was teacher-forced shifted causal-LM mean token NLL,
weighted over contributing tokens. Each genomic-decoder window was retained as an individual bootstrap
unit. The separately constructed 96-window prompt pool was not used for this endpoint.
Genomic encoders (n=2) were evaluated on 256 hg38 windows of 600 bp sampled from random_262kb.bed
with random.Random(42), requiring less than 1% N content. Inputs were padded/truncated to 256 tokens and
evaluated in batches of 16. A fixed 15% mask realization with seed 42 excluded special and padding tokens
and was reused for baseline and every intervention. The endpoint was masked-nucleotide summed loss
divided by the masked-token count. DNABERT-2 used the revision-pinned pretrained MLM checkpoint with
the eager compatibility loader; NTv3 used its required pinned remote-code loader.

### Effect definitions and control comparison

For evaluation unit j, the raw records stored summed loss Sj and the number of contributing tokens Nj.
Baseline and perturbed losses were recomputed as token-weighted means:
L0 = (Σj S0,j) / (Σj N0,j) ,

Lε = (Σj Sε,j) / (Σj Nε,j).

The signed relative effect was
Rε = (Lε - L0) / L0 ,

reported as 100Rε percent. Positive values therefore indicate degradation of the native objective after
suppression, whereas negative values indicate a lower loss after suppression. For each model, the same-layer
control reference was the median of the five independently measured control effects:
Gε = Rε,candidate - median(Rε,control,1, …, Rε,control,5).
The 18/22 and 20/22 summaries count models with Gε>0. The reported +0.60% and +0.88% cohort
summaries are medians of Gε across the 22 models. Figure 2A-B display the signed candidate and
individual-control Rε values, not Gε. Figure 2C uses the signed full-ablation candidate R1.0 on the y-axis.

### Finite-intervention tomography

Tomography followed the mechanistic-tomography framework of Erramilli [10], which treats
interventions as designed measurements, tests candidate observer families on held-out finite interventions at
the intended scale, and expands the measurement family when structured residuals remain. We evaluated four
observer families of increasing expressive capacity. F0 predicted a mask response by summing measured
singleton effects. F1 applied a single least-squares calibration scalar to that sum. F2 jointly fit additive main
effects by ridge regression. F3 augmented the design with all pairwise products a_i a_j and fit main and pair
coefficients jointly by ridge regression. Ridge strength was selected from a fixed grid on a calibration split;
all reported performance was measured on a disjoint held-out split.
For DNABERT-2, the fixed basis contained 10 high-gain rows: (L5/r603), (L3/r86), (L3/r399), (L9/r264),
(L9/r294), (L3/r603), (L3/r641), (L7/r603), (L6/r603), and (L5/r86). The lifted F3 design contained 10 main
effects and 45 pair terms. Using seed 20260822, the mask pools comprised 78 fit masks, 20 calibration
masks, and 20 held-out masks; the 78x55 lifted design had full column rank.
For each intervention strength, we reported held-out R2, MAE, root-mean-square error, and normalized
MAE. F2-to-F3 improvement was quantified as the relative reduction in held-out MAE. Confidence intervals
were obtained from 5,000 paired bootstrap resamples of batch indices, resampling the held-out condition and
its baseline jointly before baseline subtraction. Evidence for pair dependence was based on improved
held-out prediction, bootstrap uncertainty, and identifiability of the lifted design rather than on training fit
alone.
The same F0-F3 analysis was applied to a fixed six-row Phi-3 basis. No third-order model was fit; the
full-ablation response is therefore reported as unresolved rather than modeled with higher-order terms after
observing the data.
To assess sensitivity of the observer-family comparison to the specific fit/calibration/held-out partition, we
performed 100 resplits per intervention strength, re-permuting role assignment among the fixed pool of 118
measured non-singleton mask conditions; pool sizes match exactly what was measured, so this resampling
explores which subsets fall into held-out rather than new mask combinations. F3 outperformed F2 on held-out
MAE in 100 of 100 splits at both ε=0.5 and ε=1.0. At ε=0.5, median [2.5, 97.5] held-out R² was −0.21 [−0.45, 0.06]
for F0, 0.30 [−0.35, 0.57] for F1, 0.55 [0.26, 0.70] for F2, and 0.92 [0.79, 0.96] for F3; at ε=1.0 the corresponding
values were −0.45 [−1.14, −0.05], 0.31 [−0.33, 0.69], 0.56 [0.33, 0.70], and 0.76 [0.49, 0.88]. The selected ridge λ
was unstable across resplits — F2’s selected λ ranged from 0.001 to 100 at ε=0.5 and 0.001 to 30 at ε=1.0, and
F3’s from 0.001 to 10 at ε=0.5 and 0.1 to 100 at ε=1.0 — indicating that the point estimate depends on which
conditions fall into the fit set, although the qualitative F3 > F2 ranking did not.
The reported L9/r264 × L9/r294 epistasis values (ε=0.5: 0.053; ε=1.0: 2.012) come from a standalone regression
check rather than from the 78/20/20 fit, calibration, and held-out mask pools used for the F0–F3 fits; the

joint-ablation condition for this exact pair does not appear among the 118 measured non-singleton conditions in
those pools.

### DNABERT-2 pretrained masked-language-model experiment

DNABERT-2 was evaluated with the revision-pinned pretrained masked-language-model checkpoint
using eager attention. Sequence windows were sampled from the hg38 reference genome using
random_262kb.bed. We evaluated 256 windows of 600 bp, in 16 batches, with a fixed 15% mask realization
(seed 42), yielding 4,453 masked tokens. The same masks were reused across all intervention conditions.
Pair epistasis was defined as the joint loss change minus the sum of the two singleton loss changes. The
reported L9/r264 × L9/r294 epistasis values (ε=0.5: 0.053; ε=1.0: 2.012) come from a standalone regression
check rather than from the 78/20/20 fit, calibration, and held-out mask pools used for the F0–F3 fits; the
joint-ablation condition for this exact pair does not appear among the 118 measured non-singleton conditions
in those pools.
For DNABERT-2, candidate selection and causal evaluation were distinct assays. L5/r603 was defined using
the fixed 504-bp ACTB discovery probe tokenized with the model's default special tokens; this configuration
produces the characteristic activation maximum at that coordinate (out_max=944.5556; activation ratio
152.7345; activation rank 1). Functional evaluation instead used the independent seed-42 hg38
masked-language-model dataset described above. The ACTB discovery preprocessing and hg38
causal-evaluation preprocessing were therefore distinct.

### GENERator GC-composition and BOS analyses

For the GENERator composition-specificity analysis, two disjoint hg38 window pools were used: 96
windows of 170 bp for generation, with the first 120 bp used as prompts, and 100 windows of 512 bp for
teacher-forced native-loss measurement. The pools were drawn from non-overlapping regions of the same
reference BED file. Native-loss damage for row 2371 was measured over row scales
α∈{0,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0}, with the α=0 ablation defining the target damage. Five seeded
non-2371 control rows were each swept over α∈{0,0.1,...,8.0} to identify the closest achievable damage. A
second specificity control replaced the weight vector at the row-2371 location with a fixed random unit
direction scaled by c ∈ {0.0125, 0.5, 1.0, 3.0, 8.0}. Because the row is replaced rather than perturbed
additively, c=0 is equivalent to ablation and native-loss damage decreases monotonically with increasing c;
the maximum tested scale, eight times the row’s own norm, still retained 84% of the ablation damage.
Generated-sequence statistics were evaluated at each c over 96 prompts and three generation seeds, with
per-prompt GC retained. At each selected condition, generated-sequence statistics were evaluated over 96
prompts and three generation seeds. GC fraction was computed over the generated continuation after filtering
to A/C/G/T characters, and 95% confidence intervals were obtained from 5,000 bootstrap resamples paired
over prompts.
For the BOS attention/activation reproduction, 40 hg38 windows were evaluated with seed 42. We
measured mean incoming attention mass assigned to position 0, the corresponding fold over the uniform
expectation, the fraction of layer-head observations whose attention maximum occurred at position 0, and the
position of the maximum high-gain activation. Because position 0 has special status in a causal decoder and
no independent causal manipulation of the attention sink was performed, this analysis is interpreted as
co-occurrence only.

### Statistical analysis

Exact operator spectra are deterministic single-checkpoint calculations and therefore are shown without
sampling error bars. Tomography confidence intervals use 5,000 paired bootstrap resamples over batch
indices, as described above. The manuscript reports effect sizes and held-out predictive performance rather
than treating cross-model comparisons as a formal population test, because the model panel is small and
heterogeneous. No multiple-comparison correction was applied across models.
Per-condition 95% confidence intervals were obtained from 5,000 percentile bootstrap draws over the stored
evaluation units. In each draw, the same resampled index vector selected the paired baseline and perturbed
units, each loss was recomputed as total summed loss divided by total contributing tokens, and Rε was
recalculated. The stored units were actual batches for text decoders and both encoder groups, and individual
windows for genomic decoders. Condition-specific random streams were drawn sequentially from
SeedSequence(42).spawn(1000) in frozen panel/condition order.
For the cohort candidate-minus-control gap, the 22 model-level Gε values were resampled with replacement
and the median was recomputed for each of 5,000 draws. Percentile intervals used seed 47 for ε=0.5 and seed
52 for ε=1.0. To test the relationship between structural concentration and the full-ablation functional effect,
Spearman's rank correlation was computed between candidate q1 and the signed full-ablation candidate
effect across the 22 models. The reported p value is SciPy's default asymptotic two-sided test. A 5,000-draw
model bootstrap resampled the 22 model rows with replacement, recomputed ρ, discarded only nonfinite
replicates, and formed percentile limits (seed 43). Negative and zero effects were retained without
transformation.

### Data and code availability

Data and code availability. Code and non-model artifacts are available at
https://github.com/Georgakopoulos-Soares-lab/superweights (commit
e89968f4059f17f03fb6ea93d053397e2a367f75). The repository contains all analysis code, per-model
structural and causal measurement scripts, and result artifacts for every experiment reported here, with two
exceptions inherent to their size: the hg38 reference FASTA and WikiText-2 are not committed and must be
obtained independently. Reproducing the full 22-model census additionally requires downloading the pinned
Hugging Face checkpoint revisions listed in Supplementary Table S1.

## Acknowledgments

## Author contributions

## Competing interests

## References

1. Yu M, Wang D, Shan Q, Reed CJ, Wan A. The Super Weight in Large Language Models. arXiv:2411.07191 (2024).
2. Sun M, Chen X, Kolter JZ, Liu Z. Massive Activations in Large Language Models. arXiv:2402.17762 (2024).

3. Xiao G, Tian Y, Chen B, Han S, Lewis M. Efficient Streaming Language Models with Attention Sinks. International
Conference on Learning Representations (2024).
4. Sun S, Canziani A, LeCun Y, Zhu J. The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention
Sinks. arXiv:2603.05498 (2026).
5. Li Q, Zhan Z, Feng S, et al. GENERator-v2: Reconciling Coarse Tokenization with Single-Nucleotide Resolution in
Genomic Language Modeling. bioRxiv 2026.01.27.702015 (2026).
6. Zhou Z, Ji Y, Li W, Dutta P, Davuluri RV, Liu H. DNABERT-2: Efficient Foundation Model and Benchmark for
Multi-Species Genomes. International Conference on Learning Representations (2024).
7. Boshar S, Evans B, Tang Z, et al. A foundational model for joint sequence-function multi-species modeling at scale for
long-range genomic prediction. bioRxiv 2025.12.22.695963 (2025).
8. Brixi G, et al. Genome modelling and design across all domains of life with Evo 2. Nature 652, 1349–1361 (2026).
9. Zhou Z, Riley R, Kautsar S, et al. GenomeOcean: An Efficient Genome Foundation Model Trained on Large-Scale
Metagenomic Assemblies. bioRxiv 2025.01.30.635558 (2025).
10. Erramilli V. Mechanistic Tomography: Designed Measurement for Control-Oriented Interpretability. arXiv:2608.19338
(2026).
11. Touvron H, Lavril T, Izacard G, et al. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971
(2023).
12. Jiang AQ, Sablayrolles A, Mensch A, et al. Mistral 7B. arXiv:2310.06825 (2023).
13. Groeneveld D, Beltagy I, Walsh E, et al. OLMo: Accelerating the Science of Language Models. Proceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics, 15789–15809 (2024). DOI:
10.18653/v1/2024.acl-long.841.
14. Abdin M, Aneja J, Awadalla H, et al. Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone.
arXiv:2404.14219 (2024).
15. Yang A, Yang B, Zhang B, et al. Qwen2.5 Technical Report. arXiv:2412.15115 (2024).
16. Portes J, Trott A, Havens S, et al. MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining. Advances in
Neural Information Processing Systems 36 (2023).
17. Warner B, Chaffin A, Clavié B, et al. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory
Efficient, and Long Context Finetuning and Inference. Proceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics, 2526–2547 (2025).
18. Merity S, Xiong C, Bradbury J, Socher R. Pointer Sentinel Mixture Models. International Conference on Learning
Representations (2017).

### Supplementary Figure S1. Structural and causal control detail.

(A) Exact q1 values for all six Phi-3 candidate rows. (B) Individual same-layer random-control causal
effects for all five decoder models in the E10 causal audit. (C) Local spectral exceptionalness,
defined as candidate q1 minus mean same-layer-control q1, plotted against non-embedding parameter
count for the 12-model common local-control panel. Exact single-checkpoint spectral measurements are
shown without sampling error bars.

## Supplementary Tables

S1. Model panel and provenance (22 rows).
S2. Structural metrics (22 rows). Carries two candidate-versus-top-norm gap definitions (max-based
and mean-based). Pick one for the main text and define both in the legend.
S3. Full causal census (44 rows). all five individual top-norm control values per model per epsilon.
S4. Structure–function correlations (16 rows), underpowered-panel flag retained.
S5. GENERator conditions (106 rows). GC and CI columns are absent for NLL-only conditions
