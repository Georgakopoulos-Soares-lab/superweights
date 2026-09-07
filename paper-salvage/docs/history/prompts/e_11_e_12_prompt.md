# Task: two follow-up experiments for the "Structure Is Not Mechanism" manuscript

You are working in the repository for a manuscript on high-gain gated-FFN pathways across text and genomic
foundation models. Two reviewer-critical experiments need to be added. Both are cheap relative to the
existing work: Experiment A is almost entirely weight-space math plus one forward pass per model,
Experiment B is generation + loss evaluation on a model already in the panel.

Read this whole file before writing any code.

---

## 0. Orientation (do this first, do not skip)

1. Inventory the repo. Find and read:
   - the implementation of the **exact gated-FFN bilinear operator** `U_k` and the `q1` / `PR_spec` /
     `||U_k||_F` computation,
   - the **activation-based candidate detector** (global max output activation of the
     down-projection-equivalent module relative to the layer's median channel max; acceptance threshold
     ratio >= 5.0),
   - the **same-layer control sampling** code (five ordinary rows, `NumPy SeedSequence(42)` + fixed
     model-specific child seed),
   - the **GENERator GC generation** script (24 hg38-derived prompts, 170 bp windows, first 120 bp as
     prompt, 64 new tokens, top-k=50, temperature=1.0, seed 42 offset by prompt index),
   - the preregistration / experiment-freeze records (E7-E10b) and their format.

2. **Reuse these implementations. Do not reimplement them.** The whole value of these experiments is that
   they are measured with the identical operator definition and detector as the existing panel. If you find
   yourself writing a second copy of the singular-value code, stop and import the existing one.

3. Write a short `INVENTORY.md` listing what you found and the exact import paths you will use. Show it to
   me before proceeding.

4. Both experiments are **preregistered**. Follow the existing freeze convention: write the protocol file
   (analysis plan, model list, metrics, decision rules, seeds) and commit it **before** running any
   measurement. Name them `E11` (scale ladder) and `E12` (GENERator degradation control) unless the repo
   convention dictates otherwise.

---

## Experiment A (E11): decouple encoder/decoder from parameter scale

### Why

The manuscript's headline structural claim is that `q1` is organized by encoder-vs-decoder architecture
rather than by text-vs-genomic domain. As the panel currently stands, **architecture is perfectly
confounded with scale**: every decoder is >= 3B parameters (GENERator 3B, Phi-3 3.8B, GenomeOcean 4B,
Llama/Mistral/OLMo/Qwen 7B) and every encoder is <= 650M (MosaicBERT 110M, DNABERT-2 117M, ModernBERT-base
150M, NTv3 650M). A reviewer can explain the entire Fig. 1B pattern as "spectral concentration emerges with
scale," which is consistent with the massive-activation literature the paper already cites. NTv3 at 650M
having the lowest `q1` is the only evidence against pure scale and it is not sufficient.

The goal is **not** to prove architecture wins. It is to measure the scale axis honestly and report
whether the architecture term survives it. A result of "scale explains most of it" is a publishable and
useful correction — write it up either way.

### Critical constraint: the operator is only defined for gated FFNs

`U_k` is built from the gate and up projections. Models with a classical two-matrix MLP (no gate) do
**not** have this operator. Before adding any model, **inspect its config and module names
programmatically** and classify the FFN as gated (SwiGLU / GeGLU / GLU variants) or non-gated. Do not
assume from the model family name. Emit this classification into the results table as a column.

Non-gated models must be **excluded**, not approximated with a fallback. If exclusion leaves a hole in the
ladder, report the hole.

### Model selection

Build the ladder by **within-family scale sweeps**, which control for tokenizer, data, and training recipe
in a way that cross-family comparisons cannot:

- **Decoder ladders (need >= 2 families, >= 3 sizes each).** Search Hugging Face for gated-FFN decoder
  families with multiple released sizes and pin a revision for each. Qwen2.5 (0.5B / 1.5B / 3B / 7B) is a
  natural first ladder since Qwen2.5-7B is already in the panel. Find at least one more family with a
  genuine size sweep and a gated FFN. Prefer families where the small and large checkpoints share a
  training recipe.
- **Encoders at larger scale.** This is the scarce direction — large gated-FFN encoders are rare, and that
  scarcity is itself part of the finding. Enumerate candidates by config inspection rather than from
  memory. ModernBERT-large is the obvious first addition since ModernBERT-base is already in the panel.
  Check whether NTv3 has larger released variants. Look for other gated bidirectional encoders in the
  300M-1B+ range. **If you cannot find a gated encoder above ~1B, say so explicitly in the writeup** — the
  claim then becomes "we could not test encoders in the size range where decoder concentration is highest,"
  which is an honest limitation and much better than silently leaving the confound unaddressed.
- **Genomic ladder if available.** If any genomic family in the panel has multiple sizes, include them.

Pin a revision SHA for every new model. The manuscript already flags four models with no pinned revision as
an explicit reproducibility gap — do not add to it.

### Per-model measurement

For each model in the ladder, run the **existing** pipeline unchanged:

1. Activation-based candidate detection (ratio threshold 5.0). Record the ratio even when it fails, and
   record failures as detection nulls rather than dropping them silently.
2. Exact `U_k`, singular spectrum in float64, `q1`, `PR_spec`, `||U_k||_F`.
3. Five seeded same-layer control rows, same metrics.
4. Record: parameter count (total and non-embedding), hidden size `d_model`, FFN intermediate size, number
   of layers, candidate layer index, **relative depth** (candidate layer / total layers), layer median
   `||U_k||_F`, and the candidate's ratio to it.

### Analysis

The unit of analysis is one detected candidate per model.

1. **Primary plot**: `q1` vs `log10(non-embedding params)`, points colored by architecture class
   (decoder / encoder) and shaped by domain (text / genomic). Overlay within-family lines connecting sizes
   in the same ladder. This single figure is the deliverable that answers the reviewer.
2. **Within-family slope**: for each ladder, the change in `q1` per decade of parameters, with the
   direction and rough magnitude. If `q1` rises steeply with scale inside a single decoder family, scale is
   doing real work and the manuscript must say so.
3. **Architecture term after conditioning on scale**: fit `q1 ~ log10(params) + is_decoder` (OLS) and report
   coefficients, standard errors, and R². Also report the partial correlation between `is_decoder` and `q1`
   controlling for `log10(params)`. **Caveat this heavily in the writeup**: n will be on the order of
   15-25, the models are not exchangeable, and this is a descriptive summary, not a population test. That
   framing matches the manuscript's existing statistical stance — keep it.
4. **Report both magnitude rankings.** Recompute the candidate `||U_k||_F` both as an absolute value and as
   a ratio to the layer median, and note anywhere the two orderings disagree. (This also feeds the OLMo
   check in section C below.)
5. **Depth control**: plot `q1` against relative depth. If candidates in large models are systematically at
   different relative depths than in small models, that is a second confound and needs to be named.
6. **Candidate-vs-control gap by scale**: the manuscript's Fig. 1D argument is that a candidate can be
   exceptional relative to its own layer even when its `q1` is modest. Compute candidate `q1` minus mean
   control `q1` at every scale and check whether that gap is scale-dependent. If the gap is stable while
   absolute `q1` rises with scale, that is a genuinely interesting result and probably the strongest
   version of the architecture claim available.

### Decision rules (write into the protocol before running)

State in advance which outcome you will report:

- If the within-family scale slope is small and the architecture coefficient survives conditioning, the
  manuscript's claim stands and gets stronger.
- If the within-family slope is large and the architecture coefficient collapses, the structural claim must
  be **rewritten** as a scale effect. Do not soften this into "architecture and scale both contribute"
  unless the numbers actually support a mixture.
- If encoders cannot be sampled above ~1B, the claim becomes explicitly range-restricted.

### Deliverables

- `results/E11/scale_ladder.csv` (one row per model, all fields above)
- `results/E11/scale_ladder_controls.csv` (control rows)
- `figures/E11/q1_vs_scale.pdf`
- `results/E11/regression_summary.json`
- `results/E11/E11_summary.md`: 3-4 paragraphs stating what happened, written so it can be dropped into
  the Results section with light editing. Include the sentence that would go in Limitations if the encoder
  range is restricted.

---

## Experiment B (E12): degradation-matched control for the GENERator GC result

### Why

The manuscript reports that scaling row 2371 to alpha=0 moves mean generated GC from 0.3961 to 0.2944 (span
0.1017) while five random control rows span only 0.00026 — roughly 390-fold. As written, this is equally
consistent with two readings:

1. Row 2371 exerts specific control over generated nucleotide composition.
2. Ablating row 2371 breaks generation, and broken/degenerate genomic output happens to be AT-rich.

Reading 2 is the obvious reviewer objection and it is fatal to the strongest functional claim in the
genomic half of the paper. The random controls do not rule it out, because random rows barely perturb the
model at all — they are not matched on damage. **The fix is to compare row 2371 against control rows scaled
to produce an equivalent increase in language-model loss.**

### B1. Add generation-quality readouts to the existing dose-response

Re-run the existing GC protocol (same prompts, same decoding settings) with added per-condition metrics
computed over the **generated continuation only**, consistent with the existing GC computation:

- mean per-token predictive entropy during generation
- distinct-2 / distinct-3 / distinct-4 ratios (unique n-grams / total n-grams)
- longest homopolymer run, and mean homopolymer run length
- fraction of the continuation covered by the single most frequent 3-mer and 6-mer
- fraction of generated tokens that decode to non-ACGT characters (currently filtered out silently — this
  filtering rate is itself a degradation signal and must be reported)
- mean per-token NLL the model assigns to its own continuation

Also report **prompt GC** as a reference line: if ablated output drifts toward some fixed composition
regardless of prompt, that is a strong degeneracy signature.

### B2. Strengthen the dose-response itself

- Finer alpha grid: `{1.0, 0.75, 0.5, 0.25, 0.0}` rather than three points. The non-monotonicity of the
  secondary row 1522 makes the three-point curve hard to interpret; more points will either resolve it or
  confirm it.
- More prompts: raise from 24 to at least 96 hg38-derived prompts per condition. Keep the existing
  windowing (170 bp windows, first 120 bp as prompt, 64 new tokens).
- Multiple seeds: >= 3 base seeds, not just 42. Report between-seed variability.
- Add a **greedy decoding arm** (temperature=0 or top-k=1) at each alpha for at least one seed. If the GC
  effect survives greedy decoding, sampling stochasticity is eliminated as an explanation.
- Bootstrap 95% CIs on mean GC per condition, resampling prompts. Reuse the existing paired-bootstrap
  machinery (5,000 resamples) if the interface fits.

### B3. The degradation-matched control (this is the core of E12)

1. Define a damage metric: mean per-token causal-LM NLL on a held-out set of hg38 windows, **disjoint from
   the generation prompts**. Use >= 100 windows, fixed seed, float32.
2. Measure the damage curve for row 2371 across the alpha grid. Record `D(2371, alpha=0)`, the damage at
   full ablation.
3. For each of >= 5 seeded same-layer control rows, **find the scaling alpha_c such that the control row's
   damage matches `D(2371, alpha=0)`**. Since a random row scaled to alpha=0 may not reach that damage
   level at all, do this by:
   - sweeping the control row's scale over a range that includes **amplification** as well as suppression
     (e.g. multiplicative factors spanning strong suppression through several-fold amplification), and
   - selecting by bisection or nearest-match on the damage metric.
   - **If a control row cannot reach `D(2371, alpha=0)` anywhere in the sweep, record that** — it means row
     2371's damage is itself exceptional, which is a finding, and then fall back to matching at the
     highest damage the control can reach and note the mismatch explicitly.
4. Also construct a **random-direction matched control**: perturb the down-projection row by a random
   vector scaled to produce the same damage. This separates "this specific row" from "any perturbation of
   this magnitude at this location."
5. At matched damage, run the full generation protocol and compute GC plus all B1 quality metrics.

**The decisive comparison**: at equal LM damage, does row 2371 still move GC substantially more than
matched controls? Report the GC shift of row 2371 against the distribution of matched-control GC shifts,
with a bootstrap CI on the difference.

### B4. Degeneracy attribution

If ablating row 2371 does drop GC, determine *how*:

- Compare the 3-mer and 6-mer composition of ablated vs baseline continuations. A clean compositional shift
  looks different from collapse into poly-A or a short repeating motif.
- Report the top 5 most enriched k-mers at alpha=0 relative to baseline.
- Report what fraction of the GC drop is attributable to the single most frequent homopolymer run.

### Decision rules (write into the protocol before running)

- **Specific-control supported**: row 2371's GC shift exceeds matched controls by a clear margin with a
  bootstrap CI excluding zero, *and* generation-quality metrics at alpha=0 are not catastrophically
  degraded relative to matched controls. Then the manuscript's claim stands and gets much stronger.
- **Degradation-explained**: matched controls produce comparable GC shifts, or row 2371's alpha=0 output is
  degenerate by the quality metrics. Then the claim must be **downgraded** to "ablating this row degrades
  generation, and degraded genomic output is AT-biased." That is still reportable, but it is a different
  sentence and Fig. 4C needs relabeling.
- **Intermediate**: report the effect as partially specific with explicit quantification of how much
  survives damage-matching. Do not round this up.

### Deliverables

- `results/E12/dose_response_extended.csv` (per condition per seed per prompt: GC + all quality metrics)
- `results/E12/damage_matching.csv` (control row, scale sweep, damage, matched alpha, matched GC)
- `results/E12/kmer_attribution.csv`
- `figures/E12/gc_vs_damage.pdf` — GC shift on the y-axis against LM damage on the x-axis, with row 2371,
  row 1522, matched controls, and random-direction controls as separate series. **This is the figure that
  answers the reviewer**; the current Fig. 4C does not.
- `figures/E12/generation_quality.pdf`
- `results/E12/E12_summary.md`: 3-4 paragraphs, drop-in ready, stating which of the three decision-rule
  outcomes occurred.

---

## C. Small robustness check to fold in (cheap, do it alongside E11)

The OLMo structural-causal dissociation is the manuscript's cleanest counterexample, and it currently rests
on ranking by `||U_k||_F` **relative to the layer median**. Recompute the four OLMo candidate rows'
rankings by **absolute** `||U_k||_F` as well, and report both orderings. If the dissociation survives both
metrics, say so in the Results — it makes the counterexample substantially harder to dismiss as an artifact
of the normalization choice. If it does not survive, that must be stated plainly in the text.

Report the same dual ranking for the six Phi-3 candidate rows.

---

## Working rules

- **Protocol before measurement.** Commit `E11_protocol.md` and `E12_protocol.md`, including seeds, model
  lists with pinned revisions, metrics, and the decision rules above, before running anything that produces
  a number. Follow the repo's existing freeze convention.
- **Machine-readable artifacts are the source of truth.** Every number that will appear in the manuscript
  must be traceable to a committed CSV/JSON, matching the existing standard.
- **Do not resolve conflicts in favor of the interesting result.** The repo has an established practice of
  retaining unresolved discrepancies (the DNABERT-2 splice magnitudes, the NTv3 null). If a new measurement
  disagrees with an existing manuscript number, surface it prominently in the summary rather than adjusting
  either value.
- **Negative results are deliverables.** If E11 shows scale explains the structural pattern, or E12 shows
  the GC effect is degradation-driven, write it up with the same care as a positive result. Both outcomes
  are more publishable than an unaddressed confound.
- **Cost control**: before launching any run that requires loading a model above ~3B, print the planned
  model list, estimated memory, and estimated wall-clock, and ask me to confirm.
- Report progress after: (1) `INVENTORY.md`, (2) the two protocol files, (3) each experiment's completed
  measurement phase.
