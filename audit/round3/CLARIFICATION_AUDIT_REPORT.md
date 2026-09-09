# Response to `clarification.md` — audit of 17 manuscript ambiguities

**Ground truth used:** `manuscript/actual_manuscript.md` (source: `new v9(1).pdf`), per
explicit instruction that this supersedes every other manuscript copy in the repo
(`paper/main.tex`, `manuscript.txt`, `manuscript/docs/MANUSCRIPT_SOURCE_OF_TRUTH.md`,
etc. — all pre-date this version and were not used to answer any question below except
where noted as historical context).

**Method:** every question was answered by (1) reading the exact manuscript passage,
(2) tracing it to the underlying script/artifact, and (3) independently recomputing the
numeric claims from raw CSV/JSON where full precision mattered. Two prior audits already
exist in this repo (`audit/AUDIT_REPORT.md` round 1, `audit/round2/AUDIT_ROUND2_REPORT.md`
round 2) and are cited as corroborating evidence where they independently verified the same
artifacts; nothing here was taken on their word alone without checking the current manuscript
against the current artifact.

**No manuscript or code edits were made.** This is the audit only.

---

## 1. Why is there still a 12-model random same-layer structural-control analysis?

**They are not the same measurement, and neither is stale — but the manuscript states both
without ever saying so explicitly, which is the real defect.**

The manuscript's "12-model" and "22-model" numbers differ on two independent axes:

| | population | per-model control aggregation | across-model summary | control set(s) |
|---|---|---|---|---|
| "common 12-model local-control analysis" | 12 models (the original E11 cohort with full 5-control spectra) | mean of 5 random controls | **mean and range** (0.884, range 0.544–0.977) | random only — no top-norm control existed yet at this stage |
| Fig. 1C / the 22-model comparison | all 22 causal-census models | mean of 5 random controls (same rule) | **median** (0.935) | random **and** top-norm (0.014) |

Random-control q1 gaps were **recomputed for all 22 models**, not just retained for 12 —
confirmed directly from `manuscript/figures/source_data/fig1_panel_c_full22_source.csv`
(22 rows, `random_control_gap` column populated for every row). The 12-model number is not
a subset read of this file; it is a separately-reported, earlier statistic over a narrower
panel that (per Methods, "Structural statistical analysis") is *defined* as mean+range, not
median. `random_control_gap` in the 22-model file is itself candidate_q1 − **mean**(5
random controls) at the per-model level (verified by hand: Llama-7B candidate_q1 =
0.9888168037615837, 5 controls mean = 0.00956838781050271, difference =
0.97924841595108099, matching the stored `0.9792484159510809` to float precision) — so both
analyses use the *same* within-model rule (mean of 5 controls); they differ only in **which
models are included** (12 vs. 22) and **how the per-model gaps are then summarized across
models** (mean+range vs. median).

**Is the 12-model paragraph "simply obsolete and safe to delete"?** No — it carries
information the 22-model Fig. 1C panel does not (the mean+range statistic and four named
per-model illustrative examples), and it predates the top-norm control entirely, so it is
not a strict subset of the 22-model result. It is a **real, distinct design choice**, not
leftover text — but it creates a genuine readability hazard: one of its illustrative
examples, "GENERator-PROK-3B had q1 = 0.935," is **numerically identical, by coincidence, to
the unrelated 22-model median gap statistic** also stated as 0.935 a few sentences later.
A reviewer skimming for consistency could easily (mis)read these as the same number
duplicated. This is the same category of coincidence round 1's audit already flagged for
the unrelated "+111.78%" pair (`audit/AUDIT_REPORT.md`, Finding 5) — real, not an error, but
worth disambiguating (e.g., one extra decimal place, or moving the GENERator-PROK-3B example
to a different illustrative q1 value).

**Canonical 22-model statistic and exact definition:** median across 22 models of
(candidate q1 − mean of 5 random same-layer-control q1's) = **0.9346752031084058** (rounds
to 0.935). Source: `manuscript/figures/source_data/fig1_panel_c_full22_source.csv`,
column `random_control_gap`, aggregated by `manuscript/figures/fig1_structural.py:127`
(`med_rc = np.median(...)`).

**Citations:** `manuscript/actual_manuscript.md` lines 116–125 (12-model paragraph),
131–143 (22-model/top-norm paragraph), Fig. 1 caption panel C, Methods "Structural
statistical analysis"; `manuscript/figures/source_data/fig1_panel_c_full22_source.csv`;
`manuscript/figures/fig1_structural.py:59,71,81,127-129`; `audit/AUDIT_REPORT.md` Section
2 (independent confirmation that the 12-model coverage is exactly the E11 cohort).

**Classification: REAL DESIGN CHOICE**, with a **MANUSCRIPT STALE/AMBIGUOUS TEXT**
sub-issue (the coincidental 0.935/0.935 collision is not disambiguated in prose).

---

## 2. What exactly is the random-control q1 gap statistic?

**Resolved exactly, at full precision, from the Fig. 1C source data:**

- **Random-control gap (→ 0.935):** per model, `candidate_q1 − mean(5 random-control q1's)`;
  across models, the **median** of that quantity over 22 models. Verified by hand above.
  Column: `random_control_gap` in `fig1_panel_c_full22_source.csv`.
- **Top-norm gap (→ 0.014):** per model, `candidate_q1 − max(5 top-norm-control q1's)` —
  i.e., the gap to the single *closest* (most concentrated) of the five highest-norm
  neighbours, not their mean. Verified by hand: Llama-7B candidate_q1 = 0.9888168037615837,
  max of its 5 top-norm controls = 0.8210610887206907, difference = 0.167755715040893,
  matching the stored `topnorm_control_gap = 0.16775571504089293`. Across models, again the
  **median** of this max-based per-model gap. Column: `topnorm_control_gap` in the same
  file / `topk_by_norm_gap` in `audit/round2/fig1c_random_vs_topk_gaps_full22.csv`, read by
  `fig1_structural.py:128`.

**These two panels use inconsistent within-model aggregation** (mean-of-5 for the random
panel, max-of-5 / "closest neighbour" for the top-norm panel) — this is not an accident of
this audit's reading; the manuscript's own supplementary caption says so explicitly:
> "S2. Structural metrics (22 rows). Carries two candidate-versus-top-norm gap definitions
> (max-based and mean-based). Pick one for the main text and define both in the legend."
> (`actual_manuscript.md`, Supplementary Tables)

So the manuscript **already flags this as unresolved**, in its own words, in the S2 table
description — it is not merely an ambiguity this audit found for the first time. Methods
never states the top-norm gap formula at all (only "candidate norm divided by the median
norm of the five same-layer control rows" is given for the *magnitude* predictor, Q11 below
— a different quantity from the q1 *gap* used for Fig. 1C/1D).

**Recommendation for a single consistent definition:** use the **max-based** ("closest
neighbour") top-norm gap for the main text, since it is the more conservative statistic (it
reports the *smallest* structural advantage the candidate has over any of its five
strongest weight-space competitors, not an average that a single low outlier could inflate),
and it is already what Fig. 1C actually plots. Pair it with mean-of-5 for the random panel,
since random controls are not being tested for a "closest competitor" property. State both
formulas explicitly in the Fig. 1 legend and Methods, and drop (or clearly footnote) the
alternative mean-based top-norm column in S2 rather than leaving two live definitions.

**Citations:** `manuscript/figures/source_data/fig1_panel_c_full22_source.csv`;
`manuscript/figures/fig1_structural.py:127-129`; `audit/round2/tables/S2_structural_metrics.csv`
header (`random_control_gap`, `topk_by_norm_gap`, `mean_random_control_q1`,
`mean_topk_norm_control_q1` columns); `audit/round2/FINAL_CHECK_RESULTS.md` Section 4.

**Classification: AMBIGUOUS — NEEDS PI DECISION** (the manuscript's own supplementary
caption already asks for this decision explicitly; it has not yet been made in the main
text).

---

## 3. Check the "69-fold collapse."

**Verified correct from unrounded values — this is not a bug.**

```
median random_control_gap (unrounded) = 0.9346752031084058
median topnorm_control_gap (unrounded) = 0.013538223722150244
ratio = 0.9346752031084058 / 0.013538223722150244 = 69.03972207071442  →  "69-fold" ✓

For comparison, dividing the ROUNDED text values instead:
0.935 / 0.014 = 66.785714...  →  the ~66.8 the question computed
```

The apparent ~66.8-vs-69 discrepancy is purely an artifact of **rounding both operands to
three decimals before dividing**; the manuscript's "69-fold" is computed (correctly) from
the full-precision underlying values, not from the rounded text. Also independently
re-verified: the 7 negative-topnorm-gap models listed in the text
(SmolLM2-360M, EuroBERT-210M, EuroBERT-2.1B, ModernBERT-large, DNABERT-2, GENERator-EUK-3B,
GenomeOcean-4B) exactly match `topnorm_control_gap < 0` in the source CSV — all 7, no more,
no fewer.

**Citations:** recomputed directly this session from
`manuscript/figures/source_data/fig1_panel_c_full22_source.csv` (22 rows).

**Classification: no defect — VERIFIED CORRECT.** No manuscript change needed. (Optional
improvement: state one more significant figure for at least one of the two source numbers
in-text, so a reader who tries this same rounded-division check doesn't independently arrive
at 66.8 and flag a false discrepancy, exactly as this question's author did.)

---

## 4. Reconcile the model counts (23 structural vs. 22 causal vs. 16+6=22 provenance).

**The arithmetic is internally consistent; the missing model in the "16+6=22" provenance
paragraph is Phi-3, and the paragraph's scope (22-model causal cohort, not the 23-model
structural panel) is never stated explicitly — that is the actual defect.**

Counting the manuscript's own "Model panel and checkpoints" list directly:

- **23-model structural panel** (Fig. 1B, "23 foundation models with accepted
  activation-based candidates"): 11 text decoders (Llama-7B, Mistral-7B, OLMo-7B,
  **Phi-3-mini**, Qwen2.5-7B, Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, SmolLM2-135M,
  SmolLM2-360M, SmolLM2-1.7B) + 6 text encoders (MosaicBERT, ModernBERT-base,
  ModernBERT-large, EuroBERT-210M, EuroBERT-610M, EuroBERT-2.1B) + 4 genomic decoders
  (GENERator-v2-eukaryote-3B, GENERator-v2-prokaryote-1.2B, GENERator-v2-prokaryote-3B,
  GenomeOcean-4B) + 2 genomic encoders (DNABERT-2, NTv3) = **11+6+4+2 = 23.** ✓
  (Evo2-7B is evaluated but explicitly excluded — no accepted candidate.)
- **22-model causal census**: identical list **minus Phi-3** (excluded "by the frozen scope
  definition because its pre-existing causal basis comprised six rows rather than one
  comparable primary row") = 10 text decoders + 6 text encoders + 4 genomic decoders + 2
  genomic encoders = **22.** ✓ Matches the census breakdown given verbatim in Results.
- **"16 ratio-detector + 6 earlier-protocol = 22"** (Methods, "Candidate provenance"): this
  is the causal cohort's provenance, i.e. **22 models, not 23** — it is describing the same
  22, and Phi-3 is correctly absent from it, **but the paragraph never states that its scope
  is the 22-model causal cohort rather than the full 23-model structural panel.** A reader
  arithmetic-checking "23 structural candidates" against "16+6=22" (as this question's
  author did) has no textual cue that Phi-3 is the deliberately-excluded 23rd model — its
  own selection protocol (a separate top-6-by-exact-norm multi-row rule, not the ratio
  detector and not the single-row activation-magnitude protocol; see Q6) is never mentioned
  in the provenance paragraph at all.

**Missing category:** Phi-3, selected by a third, undocumented (in this paragraph) protocol.

**Exact 23-model structural list and exact 22-model causal list:** given verbatim above (23
list; causal list = same minus Phi-3-mini).

**Citations:** `actual_manuscript.md` Methods "Model panel and checkpoints" and "Candidate
provenance"; Results "The completed census contained 22 models: 10 text decoders, six text
encoders, four genomic decoders, and two genomic encoders"; Methods "Cross-model singleton
functional-criticality census" ("Phi-3 was excluded from this 22-model singleton analysis by
the frozen scope definition...").

**Classification: MANUSCRIPT STALE/UNDERSPECIFIED TEXT** (numbers are correct; the
provenance paragraph's implicit scope needs one clarifying sentence, e.g. "Of the 23
structural candidates, the following provenance breakdown covers the 22 models in the causal
census; Phi-3's coordinates were selected separately, see [X]").

---

## 5. Supplement tables vs. the 23-model structural panel.

**Intentional — S1/S2 are scoped to the 22-model causal cohort, not the 23-model structural
panel; Phi-3 is the omitted model, and it is omitted for the same reason given in Q4/Q6.**
The manuscript's own Supplementary Tables section states row counts explicitly: "S1. Model
panel and provenance (22 rows)" and "S2. Structural metrics (22 rows)" — both are causal-
census supplements, consistent with `audit/round2/tables/S1_model_panel_provenance.csv` and
`S2_structural_metrics.csv`, both confirmed 22 data rows. Fig. 1B is the only figure/table
covering the full 23-model structural panel (its own underlying data,
`results/E11/scale_ladder.csv`, has 23 rows including the Phi-3-mini row with `q1=0.9028`,
see Q6).

This is not stale — S1/S2 are deliberately causal-cohort supplements — but, as in Q4, the
supplementary captions never state *why* 22 rather than 23 (a one-line note "excludes Phi-3;
see Fig. 1B/S1-figure for its structural value" would close the loop for a reader jumping
straight to the supplement).

**Citations:** `actual_manuscript.md` Supplementary Tables section; `results/E11/scale_ladder.csv`
(23 rows, confirmed by round-1 audit `audit/AUDIT_REPORT.md` Section 2: "23 models with
accepted candidates — confirmed (23 rows in `scale_ladder.csv` with non-null q1)");
`audit/round2/tables/S1_model_panel_provenance.csv`, `S2_structural_metrics.csv` (22 rows
each).

**Classification: REAL DESIGN CHOICE**, with the same minor clarity gap as Q4.

---

## 6. Phi-3's role in the structural panel.

**Resolved precisely: Fig. 1B / S2's single Phi-3 value is the median q1 across its six
frozen rows, and this is recorded directly in the structural artifact itself.**

`results/E11/scale_ladder.csv`, Phi-3-mini-4k-instruct row:
```
layer: "2 & 4 (6 published rows, median)"
row:   "525/1693/1113 (L2), 525/1113/1693 (L4)"
q1:    0.9028
pr_spec: 1.2249
```
The `layer`/`row` fields are self-documenting: this is explicitly the **median q1 across
all six** frozen basis rows — (L2,r525), (L2,r1693), (L2,r1113), (L4,r525), (L4,r1113),
(L4,r1693) — not a single "primary" coordinate's value the way every other model in the
panel is represented. `frob_norm` is stored as blank/NaN for this row (no single Frobenius
norm makes sense for a six-row median), consistent with the manuscript's Fig. 1 caption
statement that "Exact Frobenius norms... were not interpreted as directly comparable
cross-model scores" being especially true here.

**Pre-frozen rule that produced these six rows:** exact `‖U_k‖_F` divided by the layer
median, top-6 by that ranking, computed and frozen **before** E10b tomography — i.e. the
same "top-K-by-exact-norm" selection logic used for MosaicBERT/ModernBERT-base's 10-row
bases (Q10's operator), not the activation-ratio detector and not the single-row
activation-magnitude protocol used for the other 5 grandfathered models. Source:
`manuscript/experiments/E10b_phi3_tomography/PHI3_BASIS_AUDIT.md`,
`results/e10_exact_uknorm_phi3.json`.

**Should Phi-3 count toward "23 models with accepted candidates"?** The manuscript already
answers yes — it is explicitly listed in the 23-model structural panel and Model-panel
Methods paragraph. This audit found nothing inconsistent with that choice; the only gap is
that its selection protocol is not mentioned in the Candidate-provenance paragraph (Q4).

**Citations:** `results/E11/scale_ladder.csv` (Phi-3-mini row);
`manuscript/experiments/E10b_phi3_tomography/PHI3_BASIS_AUDIT.md`;
`results/E13/PART2B_BASIS_RULE_PROPOSAL.md` (independently confirms the six-row top-K-by-
norm provenance and that Phi-3 is excluded from the 22-model causal cohort);
`actual_manuscript.md` Supplementary Fig. S1 caption ("(A) Exact q1 values for all six
Phi-3 candidate rows").

**Classification: REAL DESIGN CHOICE** — a pre-frozen, documented, non-arbitrary rule; only
the in-text disclosure (Q4) is missing.

---

## 7. What does "rank 1" in Fig. 1A actually mean?

**Two distinct rankings are combined in one panel, both over a diagonal (non-bilinear)
proxy, and the manuscript's own caption already flags this as diagonal-only.**

Implementation: `manuscript/src/uk_frobenius.py`. Per-hidden-unit diagonal contribution:
```
c_{k,i} = W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2        (uk_contributions, L40-66)
```
This is explicitly the **diagonal proxy** the manuscript's Methods describes ("A diagonal
proxy that retains only per-hidden-unit terms omits interactions between distinct hidden
units; it was used only for the retrospective Llama/Mistral/OLMo calibration in Fig. 1A")
— it is *not* the exact bilinear `U_k` used everywhere else in the paper.

- **"rank N" annotation next to each bar (Level 1, `row_rank`)**: the rank of the
  *published output row k* (e.g. Llama-7B's row 3968) among **all `d_model` down-projection
  output rows in that layer**, ordered by `sqrt(sum_i c_{k,i})` — i.e. this ranks **output
  rows within the layer**, exactly matching the manuscript's cross-model row-ranking logic
  elsewhere, just computed on the diagonal proxy instead of the exact operator.
  (`layer_report`, `uk_frobenius.py:140-169`.)
- **"published scalar top-1 share" (the plotted x-axis, Level 2, `top1_share`)**: a
  *different* ranking, computed only *within* the one published row k, across its `d_ffn`
  hidden units i: `top1_share = max_i(c_{k,i}) / sum_i(c_{k,i})`. This measures how much of
  that one row's diagonal energy sits in a single hidden unit — not a ranking of rows at
  all.

**Confirmed ranks and top1_share for the three calibration models**, from
`results/e1_nlp_retrospective.json` (STORED, computed by
`manuscript/experiments/E1_nlp_validation/run_e1_retrospective.py`):

| model | row_rank (Level 1) | top1_share (Level 2) | regime |
|---|---|---|---|
| Llama-7B (L2/r3968) | **1 / d_model=4096** (100th pct.) | 0.8902517404585564 | scalar-dominated |
| Mistral-7B (L1/r2070) | **1 / d_model=4096** (100th pct.) | 0.9884264554307004 | scalar-dominated |
| OLMo-7B (L1/r269) | **1 / d_model=4096** (100th pct.) | 0.9558374285814745 | scalar-dominated |

All three reproduce the published super-weight row exactly at rank 1, matching the
manuscript's claim ("recovers the published super-weight row at rank 1... in Llama-7B,
Mistral-7B, and OLMo-7B"). The caption's phrase "the scalar top-1 share is shown only as the
diagonal calibration metric" is accurate and already disambiguates this from the exact
bilinear-operator analyses used in Fig. 1B–D.

**Citations:** `manuscript/src/uk_frobenius.py:40-169`;
`manuscript/experiments/E1_nlp_validation/run_e1_retrospective.py`;
`results/e1_nlp_retrospective.json`; `manuscript/figures/fig1_structural.py:90-94`.

**Classification: no defect — VERIFIED, manuscript caption is already accurate.**

---

## 8. Define the activation-ratio detector exactly.

**Fully recovered from `run_rowwise_detector.py` (the E13 structural-census detector, "the
*unchanged* >=5 threshold" per its own docstring, applied identically to every candidate
row):**

```
detector_score[layer, row] = channel_max[layer, row] / median_row(channel_max[layer, :])
selected  iff  detector_score >= 5.0                                    (THRESHOLD = 5.0)
```

- **Numerator (`channel_max`)**: for a forward-hooked module (the down-projection, per
  `model_pattern`), `output.reshape(-1, d_model).abs().float().max(dim=0).values` — the
  **maximum absolute value**, over **all token positions in the single forward pass**
  (batch size 1, one frozen probe), of that output channel k.
- **Denominator**: the **median of `channel_max` across all `d_model` output channels in
  that same layer** — a per-layer, per-forward-pass median, recomputed fresh for every
  layer.
- **Reduction axes**: max over the sequence-length axis (tokens); no reduction over layers
  or across multiple probes — one probe, one forward pass, per model.
- **Absolute values**: yes — `.abs()` is applied before the max.
- **Token handling**: domain-dependent, both frozen. Text models: first 20 non-empty
  WikiText-2-raw-v1 test-split lines, joined, tokenized/truncated to 512 tokens (matches
  Methods "Candidate provenance" text: "first 20 non-empty lines of the WikiText-2 raw test
  split, joined and tokenized with truncation to a maximum length of 512 tokens").
  Genomic models: the fixed 504-bp ACTB-derived probe (`_ACTB_500`); most genomic models get
  a manually prepended BOS token with `add_special_tokens=False`; NTv3 uses a different,
  right-padded-to-multiple-of-256 convention with no added BOS (`ntv3_probe`, to preserve
  its stride-256 alignment).
- **Threshold**: `>= 5.0`, matching Methods and Evo2-7B's reported failure ("maximum
  activation ratio was 2.22, below the predefined threshold of 5.0").

**Which models were actually selected by this rule vs. retained from earlier provenance:**
exactly matches the manuscript's own explicit list (Methods, "Candidate provenance") —
**16 ratio-selected**: every model not named below. **6 retained/grandfathered** (never had
a ratio computed under this rule, selected instead under an earlier activation-magnitude-
only convention): **Llama-7B, Mistral-7B, OLMo-7B, DNABERT-2, NTv3, and GENERator-EUK-3B**.
This matches round 1's independent audit (`audit/AUDIT_REPORT.md` Section 6:
"6/22 models... sourced from `results/e7_legacy_reanalysis.json`... no ratio field at all").

**Citations:**
`manuscript/experiments/E13_full_cohort_causal_census/run_rowwise_detector.py:29,60-113`;
`actual_manuscript.md` Methods "Candidate provenance."

**Classification: no defect — VERIFIED, manuscript's Methods description is consistent
with the code**, though Methods never spells out the max/median-reduction detail at the
level of precision given here; worth inlining a version of the formula above into Methods
for full self-containedness.

---

## 9. Are "activation-selected" / "activation-derived" accurate labels for the whole cohort?

**Yes, as used — the manuscript is already careful about this and does not claim a single
uniform rule.** Methods, "Candidate provenance," opens with: *"Candidate coordinates in this
panel derive from two protocols, and we distinguish them explicitly rather than describing a
single uniform rule."* It then names the 6 grandfathered models individually (Q8's list) and
explains, model-by-model, exactly how Llama-7B/Mistral-7B/OLMo-7B (published-coordinate
calibration), DNABERT-2 (ratio-vs-activation-argmax disagreement, both reported), and NTv3
(same pattern) diverge from the ratio rule.

The umbrella terms actually used in the figure/abstract text — **"activation-based"**
(Fig. 1B caption: "23 models with accepted activation-based candidates") and
**"activation-derived"** (Introduction/Discussion) — are technically accurate umbrellas,
because *both* protocols (the ratio detector and the earlier magnitude-only rule) are
activation-based; neither protocol is weight-only. The Abstract's one use of
**"activation-selected"** ("one activation-selected candidate and five same-layer controls")
is slightly less precise, since 6/22 candidates were *retained*, not *freshly selected*, by
any activation rule in this study — but this is a minor wording point, not a substantive
mislabeling, and is fully resolved by the time a reader reaches Methods.

**Individual models not literally selected by the final ratio detector**: the same 6 named
in Q8 (Llama-7B, Mistral-7B, OLMo-7B, DNABERT-2, NTv3, GENERator-EUK-3B).

**Citations:** `actual_manuscript.md` Methods "Candidate provenance" (full paragraph);
Fig. 1 caption; Abstract.

**Classification: no defect — REAL DESIGN CHOICE, already handled correctly.** Optional
polish: replace "activation-selected" with "activation-derived" in the Abstract for
terminological consistency with the rest of the paper.

---

## 10. Confirm the bilinear-operator interpretation.

**Confirmed exactly as the question suspects: `U_k` contains no nonlinearity, and the
manuscript's own terminology is already precise in the body text — "exact gated-FFN
operator" without "bilinear" appears only as a section header, not as a substantive claim.**

Manuscript Methods, verbatim: `U_k = Σ_i d_{k,i} g_i u_i^T` — exactly the formula in the
question. Implementation, `manuscript/experiments/E7_exact_dimensionality/spectral_lib.py`:
```python
def materialize_Uk(W_gate, W_up, w_down_row, ...):
    """U_k = (d * W_gate)^T @ W_up, exactly."""
    ...
    return (d[:, None] * G).T @ U
```
Docstring: *"U_k = sum_i d_i * outer(g_i, u_i)... No cross terms are dropped here (unlike
the diagonal c_{k,i} decomposition)."* This is a pure weight-space bilinear form — a sum of
outer products of *weight* row vectors `g_i`, `u_i`, scaled by the (also fixed, weight-only)
coefficient `d_{k,i}` — it never evaluates `φ(g_i^T x)` for any input `x`, and does not
depend on activations at all. The manuscript is explicit about this omission: *"the actual
SwiGLU/GEGLU function contains the gate nonlinearity"* is never claimed to be captured;
instead the manuscript repeatedly and consistently calls this object a **"bilinear weight
operator"** (Abstract), **"exact gated-FFN bilinear operator"** (Results), and
**"row-associated bilinear operator"** (Methods body) — i.e., the terminology the question
itself proposes ("row-associated bilinear weight operator") is already what the manuscript
uses in every substantive sentence.

The one place the word "bilinear" is dropped is the **Methods section header itself**:
*"### Exact gated-FFN operator and structural metrics"* — a heading, not a claim, and every
sentence beneath it re-introduces "bilinear" correctly. This is a minor, low-stakes
inconsistency between header brevity and body precision, not a substantive misdescription —
but since a reader skimming only headers/captions could come away thinking "exact gated-FFN
operator" means the full nonlinear function, adding "bilinear" to the header would remove
any ambiguity at essentially zero cost.

**Implementation pointers:** exact `U_k` matrix + full singular spectrum:
`manuscript/experiments/E7_exact_dimensionality/spectral_lib.py` (`materialize_Uk`,
`singular_values`, `SpectralMetrics`). Vectorized Frobenius-norm-only computation (no
`U_k` ever materialized, used for the full census's per-row control-selection sweep):
`manuscript/experiments/E5_dimensionality/dimensionality_lib.py:92-111`
(`exact_uk_all_rows`, via the Gram identity `K = (W_gate W_gate^T) ⊙ (W_up W_up^T)`,
`‖U_k‖_F² = W_down[k,:] K W_down[k,:]^T` — algebraically identical to the manuscript's
`K_{ij} = (g_i^T g_j)(u_i^T u_j)`).

**Classification: no defect — VERIFIED**; header wording is a cosmetic
**MANUSCRIPT STALE/IMPRECISE TEXT** nit only.

---

## 11. What exactly is the layer-relative Frobenius predictor used in the correlations?

**Genuine mismatch between Methods text and the actual artifact: Methods says "the median
norm of the five same-layer control rows"; the code actually divides by the median
Frobenius norm across every row in the layer (the full population), independent of which
five rows were sampled as controls.**

Methods states: *"the layer-relative Frobenius magnitude was defined as the candidate norm
divided by the median norm of the five same-layer control rows."*

The artifact column actually used for the correlations is `frob_ratio_to_layer_median`
(`results/E13/part2_22_model_results.csv`, `part1_22_candidate_effects.csv`,
`figure2_structure_function_data.csv`; = S4's `layer_relative_frobenius`). Its source,
`manuscript/experiments/E13_full_cohort_causal_census/backfill_layer_median.py`, computes
`layer_median_frob_norm` as:
```python
median = frobs[n // 2] if n % 2 else 0.5 * (frobs[n // 2 - 1] + frobs[n // 2])   # L110
```
over `frobs`, the **full sorted array of exact `‖U_k‖_F` for every row in the layer**
(e.g. all `d_model` rows, via `run_confirmatory_spectral_lite`/`compute_llama_style`'s
whole-layer report) — explicitly documented in the same file as "median computed WITHIN
each layer, over that layer's own `d_model`=... rows" (line 305 comment, for the Phi-3
special case, but the same `LayerReport.median` mechanism is used for every model). This is
**not** a median of only the 5 sampled control rows (neither the random 5 nor the top-norm
5) — it is the median over the entire layer's row population, a materially larger and
different reference set.

**Which five controls, then?** Neither — the actual predictor uses **all rows in the
layer**, not a 5-row sample at all. This is a genuinely different quantity from what Methods
describes, though directionally similar (both are "the candidate's norm relative to a
typical same-layer row").

**Is this the exact predictor used for the reported correlations?** Yes — confirmed the
same `frob_ratio_to_layer_median` column drives both reported values:
ρ=0.4409937888198759 (p=0.0399, ε=1.0) and ρ=0.34839073969508755 (p=0.1121, ε=0.5),
both reproduced independently in `audit/round2/tables/S4_structure_function_correlations.csv`
and matching round 1's audit (`audit/AUDIT_REPORT.md` Section 3: "ρ=0.440994, p=0.039940").

**Citations:**
`actual_manuscript.md` Methods "Exact gated-FFN operator and structural metrics";
`manuscript/experiments/E13_full_cohort_causal_census/backfill_layer_median.py:97-118,304-306`;
`results/E13/part2_22_model_results.csv` (`layer_median_frob_norm`,
`frob_ratio_to_layer_median` columns); `audit/round2/tables/S4_structure_function_correlations.csv`.

**Classification: MANUSCRIPT STALE/INCORRECT TEXT — needs a Methods correction.** The
formula sentence should read "...divided by the median exact Frobenius norm of all other
rows in the candidate's layer" (or equivalent), not "the five same-layer control rows."
No regeneration needed — the artifact and the reported correlations are already computed
correctly under the *actual* full-layer-median definition; only the prose describing it is
wrong.

---

## 12. What is the "structural margin" used to predict the causal gap?

**Resolved exactly: "structural margin" = the absolute value of the same max-based top-norm
q1 gap used in Fig. 1C/1D (candidate q1 minus the closest/highest of its five top-norm
neighbours), correlated against the causal top-norm-control gap. It is neither a new
mean-based nor a separately-defined "concentration margin" — it reuses Fig. 1C's existing
statistic, just unsigned.**

Found verbatim in `audit/round2/AUDIT_ROUND2_REPORT.md` (line 1321): *"Spearman
ρ(|structural gap|, causal topK gap): ε=0.5: ρ=0.311, p=0.159. ε=1.0: ρ=0.355, p=0.105."*
The per-model "structural gap" values feeding this (e.g. Mistral-7B +0.0263, NTv3 +0.3694,
GenomeOcean-4B −0.0746) are numerically identical to the `topnorm_control_gap` /
`topk_by_norm_gap` column already verified in Q2 (Mistral-7B: 0.026282082064083223,
matching to displayed precision) — i.e., this is the **same max-based structural gap**
computed for Fig. 1C, simply taken in absolute value here (since 7/22 models have a
negative raw gap, an unsigned "margin" is required to test a monotone relationship without
sign confound). Source script: `audit/round2/scripts/structural_vs_causal_gap.py`
(column `structural_topk_gap_q1` in `audit/round2/structural_vs_causal_gap.csv`).

**Should the text say candidate-minus-mean or candidate-minus-max?** **Candidate-minus-max**
(closest top-norm neighbour), matching Fig. 1C, and the manuscript should state explicitly
that "the candidate's own margin of spectral concentration over its top-norm neighbours"
(Results/Discussion's exact phrase) is defined identically to the Fig. 1C/1D top-norm gap,
just unsigned — currently a reader has no way to know these are the same statistic without
independent code-tracing, since Methods never names a formula for "structural margin" at
all (it appears only in Results/Discussion prose).

**Citations:** `actual_manuscript.md` Results ("the rank correlation between the absolute
structural margin and the causal gap was ρ=0.311 (p=0.159) at ε=0.5 and ρ=0.355 (p=0.105) at
ε=1.0") and Discussion/Limitations ("we tested the candidate's structural margin as a
predictor and it does not account for them"); `audit/round2/AUDIT_ROUND2_REPORT.md` lines
1319–1352; `audit/round2/scripts/structural_vs_causal_gap.py`;
`audit/round2/structural_vs_causal_gap.csv`.

**Classification: MANUSCRIPT STALE/UNDERSPECIFIED TEXT** — values are correct and
independently reproduced; Methods should add one sentence defining "structural margin"
explicitly as `|candidate q1 − max(5 top-norm control q1's)|`, cross-referencing the same
quantity used in Fig. 1C.

---

## 13. Is spectral participation ratio (`PR_spec`) actually used?

**No — confirmed by direct search of the manuscript text and every statistical artifact:
`PR_spec` is defined once in Methods and never referenced again anywhere in Results,
Discussion, a figure, or a correlation table.**

- `grep -n "PR_spec\|participation ratio" actual_manuscript.md` returns exactly **one
  hit** — the Methods equation itself (`PR_spec = (Σσ_j²)² / Σσ_j⁴`). It appears in no
  Results sentence, no figure caption, no Discussion sentence, and is not one of S4's two
  correlation predictors.
- `audit/round2/tables/S4_structure_function_correlations.csv` (the manuscript's own
  16-row structure-function correlation table) tests exactly two predictors —
  `q1` and `layer_relative_frobenius` — at 2 epsilons × 4 panels; `PR_spec` is absent.
- It *is* computed and stored as a column everywhere the exact spectrum is computed
  (`S2_structural_metrics.csv`'s `PR_spec` column, `census_master.csv`,
  `run_rowwise_detector.py`, `E11_scale_ladder/run_model.py`, etc.) — it is fully populated
  data, just never analyzed or plotted.
- It was used **once**, historically, as a secondary sanity-check inside an earlier,
  superseded pipeline stage (`E7_exact_dimensionality/run_phase5_decision.py`'s "secondary
  consistency check on PR_spec"), not as evidence for any claim in the current manuscript.

**Is there scientific reason to keep it in main Methods?** It is algebraically related to
`stable_rank = 1/q1` (documented in `spectral_lib.py`'s own docstring as "Secondary,
algebraically redundant with q1, reported but never primary") and to `q1` itself (both are
functions of the same singular-value spectrum), so it adds a genuinely distinct
second-moment view of concentration (q1 measures only the *top* mode's share; PR_spec
measures the effective number of contributing modes overall) — there is a legitimate reason
a careful reader might want it defined. But as currently written it reads as load-bearing
Methods content for a quantity the paper never uses.

**Citations:** `grep` of `actual_manuscript.md`;
`audit/round2/tables/S2_structural_metrics.csv` header;
`audit/round2/tables/S4_structure_function_correlations.csv`;
`manuscript/experiments/E7_exact_dimensionality/spectral_lib.py:24-26`.

**Classification: MANUSCRIPT STALE TEXT (in the sense of unused Methods content)** —
recommend moving the `PR_spec` definition to a supplementary methods note (it is already
reported per-row in S2) and, if kept in main Methods, adding one sentence stating explicitly
that it is reported descriptively and does not drive any statistical claim in this paper.

---

## 14. GENERator random-direction control: one direction or multiple?

**Exactly one independently sampled random unit direction was tested, swept across 5 scale
values `c`; the manuscript's singular-form Methods language ("a fixed random unit
direction") is correct, and one Discussion sentence's plural ("random directions") is a
genuine, minor wording slip.**

Direction sampling: `manuscript/experiments/E12_generator_degradation_control/e12_lib.py`:
```python
def make_unit_random_direction(intermediate_size, row, dtype, device, base_seed=20260823):
    rng = np.random.default_rng(base_seed + row)
    ...   # single draw, unit-normalized
```
Called with `row=2371` → seed `20260823 + 2371 = 20263194`, **one call, one vector**, reused
unchanged across every tested `c ∈ {0.0125, 0.5, 1.0, 3.0, 8.0}` (confirmed: the
"directionfix" rerun trace in `audit/round2/AUDIT_ROUND2_REPORT.md` lines 199–207 shows
every one of the 14 grid-scale damage values is bit-identical before and after the fix that
changed only the reachability-matching logic, not the direction itself — i.e., the same
single `v̂` was used throughout both runs and the final GC-extension run).

**Seed/vector provenance:** `np.random.default_rng(20260823 + 2371)`, unit-normalized
(`np.linalg.norm`-divided per the function name and round-2's confirmation), then applied as
`new_row = c · v̂ · ‖orig_row‖` replacing (not adding to) the row-2371 weight vector.

**Strongest conclusion actually supported:** with n=1 sampled direction, the experiment can
only support a claim about *that one* direction, not about the population of random
directions in general — exactly the limitation the manuscript's Discussion already states
("The tested random directions therefore remain functionally close to removal, limiting the
ability of this control to distinguish direction-specific effects from a more general
sensitivity..."). The manuscript's actual conclusion ("a random-direction replacement...
reproduces the low-GC phenotype when native-loss damage remains close to ablation... does
not establish full direction-independence") is already correctly scoped to what a
single-direction experiment can show; it does not overclaim.

**The plural slip:** Discussion, "The tested random directions therefore remain functionally
close to removal..." — should read "the tested random-direction scales" or "the random
direction across the tested scales," since only the scale `c` varies (5 values), not the
direction itself (1 value). This is the exact ambiguity the question flagged, confirmed as a
real (if minor) wording inconsistency against the correctly-singular Methods text
("a fixed random unit direction scaled by a factor c").

**Citations:** `manuscript/experiments/E12_generator_degradation_control/e12_lib.py`
(`make_unit_random_direction`, `set_row_random_direction`); `actual_manuscript.md` Methods
"GENERator GC-composition and BOS analyses" (singular, correct) vs. Discussion paragraph on
GENERator (plural, "random directions" — the slip); `audit/round2/AUDIT_ROUND2_REPORT.md`
lines 183–230 (directionfix trace confirming the same single vector throughout).

**Classification: MANUSCRIPT STALE/IMPRECISE TEXT** — trivial fix (singular → the Discussion
sentence's noun), no data or regeneration issue.

---

## 15. Verify the GENERator damage-matching GC values and the "aggregation bug."

**The specific bug described in the prompt ("averaged c=0 and c=0.0125 because the writer
filtered by label but not selected scale") could not be located verbatim in this repo's
audit history — but a closely related, well-documented bug in the same artifact (a
reachability/self-matching boundary error at c=0) was found and fixed prior to the current
manuscript numbers, and the current numbers were independently re-derived from raw per-seed
generation records in round 2, not merely re-read from a possibly-stale summary file.**

**Current canonical values** (all independently `RECOMPUTED` this-session-equivalent, in
round 2, from `audit/round2/raw/gc_extension_records.jsonl` and
`results/E12/raw/damage_evals.jsonl`, not merely copied from `damage_matching.csv`):

| c | NLL | frac. of full-ablation damage | GC | source |
|---|---|---|---|---|
| 0.0 (= full ablation of row 2371) | 8.7543 | 1.000 | 0.3065 | STORED |
| 0.0125 (the manuscript's matched-damage point) | 8.7536 | 1.000 | 0.3068 | STORED |
| 0.5 | 8.7333 | 0.991 | 0.3071 | RECOMPUTED (round 2) |
| 1.0 | 8.7117 | 0.982 | 0.3078 | RECOMPUTED (round 2) |
| 3.0 | 8.6082 | 0.938 | 0.3130 | RECOMPUTED (round 2) |
| 8.0 | 8.3751 | 0.840 | 0.3402 | RECOMPUTED (round 2) |

Baseline (untouched) GC = 0.4204; baseline (untouched) NLL = 6.38538052380085; ablation
target NLL = 8.754319605827332; all 5 non-2371 control rows sit at GC ≈ 0.420–0.421 (inert)
across the full α∈[0,8] sweep. **These exactly match the current manuscript's numbers**
("GC was 0.4204 at baseline and 0.3065 after full ablation... c ≤ 1.0... GC was 0.3068–0.3078
against 0.3065... at c=8.0... GC recovered to 0.3402, closing 29.6% of the gap").

**The documented (not the prompt's exact-wording) bug**: `results/E12/raw/damage_matching.jsonl`
contains two full entries for the `random_direction` condition, written sequentially, not
overwritten. Entry 1 (pre-fix) reported `reachable=True, matched_scale=0.0` — because c=0
trivially equals its own definitional target (replacing the row with the zero vector *is*
full ablation, the same object used to define the target), so a naive `min(grid) <= target
<= max(grid)` reachability test accepted the tautological c=0 "match." Entry 2 (post-fix,
`run_e12_rerun_directionfix.log`, the one the manuscript now cites) correctly rejects that
self-match and reports the smallest genuinely-nonzero grid step, `c=0.0125`, as the closest
achievable match instead. Critically, **the underlying damage measurements themselves did
not change between the two runs** — every one of the 14 grid-scale NLL values is
bit-identical across both entries; only the *matching/decision logic* was corrected, not the
raw measurement pipeline. No script diff survives to confirm the exact original bug
mechanism (the pre-fix `e12_lib.py`/`run_e12_full.py` were never version-controlled), so
this audit reports the fix's *effect*, verified from the raw JSONL, as `STORED`/`RECOMPUTED`
fact rather than reconstructing unavailable code history.

**Has the bug been fixed in the pipeline, or is the figure manually using the correct raw
value?** **Fixed in the pipeline** — the current `damage_matching.csv`/
`dose_response_extended.csv` (mtime 2026-08-24 07:57) are the direct, unmodified output of
the corrected matcher (`run_e12_rerun_directionfix.log`), not a hand-patched value. Round 2
additionally re-derived the c=0.5/1.0/3.0/8.0 GC extension points from fresh generation
runs using the identical seed/direction convention (`base_seed=20260823`, row 2371),
independently confirming the whole curve rather than trusting the fixed matcher alone.

**Citations:** `results/E12/raw/damage_matching.jsonl`; `results/E12/run_e12_full.log`
(pre-fix); `results/E12/run_e12_rerun_directionfix.log` (post-fix, current);
`audit/round2/AUDIT_ROUND2_REPORT.md` lines 80–260;
`audit/round2/generator_random_direction_full.csv`;
`audit/round2/raw/gc_extension_records.jsonl`.

**Classification: BUG (historical) — CONFIRMED FIXED**, current manuscript values verified
against the post-fix, independently re-derived artifact. No regeneration needed. (Note for
the record: the specific "filtered by label not by selected scale, averaged c=0 with
c=0.0125" description in the prompt does not match any bug this audit could locate in the
repo's history — if that describes a different, not-yet-found issue, please point to where
it was originally reported so it can be checked separately; what *is* fully documented and
fixed is the reachability/self-match bug above, and it is the one the current manuscript
numbers reflect.)

---

## 16. DNABERT-2 direct pair experiment provenance.

**"Standalone regression check" is the manuscript's own already-used term, and it is
technically defensible (a software-testing-style "regression check" against a prior result)
but genuinely collides with the paper's other, unrelated use of "regression" for the F2/F3
ridge-regression observer families in the same section — this is a real terminology
ambiguity worth fixing, even though the underlying experiment and its scope are already
stated correctly.**

**What exactly was run:** L9/r264 and L9/r294 were ablated **individually and jointly** (a
direct, measured joint intervention, not a value predicted by any fitted model) on
DNABERT-2's pretrained masked-language-model objective, via a dedicated script,
`run_baseline_regression.py`, output `baseline_regression_results.json`. This is confirmed a
directly-measured joint ablation, not a regression-model output, by round 1's audit
(`audit/AUDIT_REPORT.md`): `d_A=+0.021818806, d_B=+0.006401925, d_AB(joint)=+2.040024024,
epistasis=+2.011803292` at ε=1.0 — `d_AB` is a stored raw joint-perturbation measurement,
matching the manuscript's own framing in Results ("their joint ablation increased loss by
+2.0400"). The script's purpose was to **reproduce a specific pre-existing older claim**
(`CHECKPOINT_1_PRETRAINED_EPISTASIS.md`, 2026-08-14, whose own raw backing JSON no longer
exists) under a fresh environment — this is the sense in which "regression check" is used:
a software-testing/reproduction "regression test," not a statistical curve fit.

**Is "regression check" accurate, or should it be "direct pair-ablation experiment"?**
Both descriptions are true simultaneously, describing different aspects: it *is* a direct,
raw joint-ablation measurement (Results correctly calls it "their joint ablation"), and it
*was run* as a reproduction/regression-test against an older result (hence "regression
check" in Methods/Fig. 3B caption). The genuine problem is that the same manuscript section
also uses "regression" extensively for the **statistical** ridge-regression observer
families (F2 additive, F3 pairwise-lifted) — so "standalone regression check" sitting one
paragraph away from "F2 jointly fit additive main effects by ridge regression" invites a
reader to (wrongly) assume the epistasis numbers are themselves a regression-model output
rather than a raw measurement. Recommend renaming to **"standalone direct pair-ablation
experiment"** or **"standalone joint-ablation reproduction check"**, as the question itself
suggests, to remove the collision — no change to what was actually done or to any number.

**Confirmed absent from the 118 non-singleton F0–F3 mask conditions:** yes, and the
manuscript already states this explicitly, twice, in near-identical language (Fig. 3B
caption and the Methods "Finite-intervention tomography" / "DNABERT-2 pretrained
masked-language-model experiment" sections): *"the joint-ablation condition for this exact
pair does not appear among the 118 measured non-singleton conditions in those pools."*
Independently reverified in round 1 (`audit/AUDIT_REPORT.md` Section 1: "not among the
78+20+20 tomography masks — no fit/calibration/held-out condition activates exactly {9,264}
and {9,294} alone").

**Citations:** `actual_manuscript.md` Results ("DNABERT-2 contains a robust pairwise
interaction..."), Fig. 3 caption, Methods "Finite-intervention tomography" and "DNABERT-2
pretrained masked-language-model experiment" (both containing the identical "standalone
regression check... 118 measured non-singleton conditions" sentence — itself duplicated
verbatim across two Methods subsections, a minor redundancy worth trimming to one place);
`manuscript/experiments/E9_mechanistic_tomography/run_baseline_regression.py`,
`baseline_regression_results.json`; `audit/AUDIT_REPORT.md` Section 1 epistasis-chain table.

**Classification: MANUSCRIPT STALE/AMBIGUOUS TEXT** (terminology collision, not a factual
or numerical error) — recommend the rename above, and de-duplicating the repeated sentence
across the two Methods subsections.

---

## 17. Why is Phi-3 tomography still in Methods?

**Intentionally retained as a documented negative/unresolved result tied to a live
supplementary figure — not orphaned, though its Methods mention has no explicit
forward-pointer to that figure.**

Methods, "Finite-intervention tomography," ends with: *"The same F0-F3 analysis was applied
to a fixed six-row Phi-3 basis. No third-order model was fit; the full-ablation response is
therefore reported as unresolved rather than modeled with higher-order terms after observing
the data."* This is explicitly a **null/negative disclosure** (F0–F3 fitting was attempted
and abandoned rather than silently dropped), consistent with this project's own working
discipline (`manuscript/CLAUDE.md`: "Do not silently drop a negative result... contested
NTv3, contested PROK — all get reported, not smoothed over in either direction").

**Does it support any current Results claim?** No dedicated Results subsection discusses
Phi-3 mechanistically (unlike DNABERT-2 and GENERator, which each get a full Results
subsection) — its only other manuscript appearances are: (1) the 23-model structural panel
(Fig. 1B, Model-panel Methods, Q6), (2) its explicit exclusion from the 22-model causal
census (Methods, "Cross-model singleton functional-criticality census," Q4), and (3)
Supplementary Fig. S1A ("Exact q1 values for all six Phi-3 candidate rows").

**Is it still intended to be part of the paper?** Yes, by design — as a disclosed limitation
of the tomography approach (it works for DNABERT-2's 10-row basis but was not successfully
extended to a third-order model for Phi-3's 6-row basis) and as one point in the 23-model
structural panel. It is not evidence-orphaned in the strict sense (every claim about it has
a live artifact: `results/e10_exact_uknorm_phi3.json` for the structural median-q1 value,
and the F0–F3 fit attempt itself, per round-1's audit map, under
`manuscript/experiments/E10b_phi3_tomography/`) — but the one Methods sentence about it
sits somewhat disconnected from Results, with no "(see Supplementary Fig. S1A)" pointer, so
a reader hitting that sentence in Methods has no cue where its supporting figure is.

**Recommendation:** keep it (it is honest, deliberate negative-result reporting consistent
with the project's stated discipline), but add a one-clause cross-reference from the Methods
sentence to Supplementary Fig. S1A so it does not read as an unsupported aside.

**Citations:** `actual_manuscript.md` Methods "Finite-intervention tomography" (Phi-3
sentence), Methods "Cross-model singleton functional-criticality census" (exclusion
rationale), Supplementary Fig. S1 caption; `manuscript/CLAUDE.md` ("do not silently drop
a negative result"); `results/E13/PART2B_BASIS_RULE_PROPOSAL.md` (independently confirms
Phi-3's tomography "remains a mechanistic case study... not carried forward" into any
cohort-level claim).

**Classification: REAL DESIGN CHOICE** (deliberate negative-result retention), with a minor
**MANUSCRIPT STALE/UNDERSPECIFIED TEXT** cross-referencing gap.

---

## Compact summary table

| # | Issue | Canonical truth | Manuscript change needed? | Regeneration needed? |
|---|---|---|---|---|
| 1 | 12-model vs. 22-model structural-control analyses | Two genuinely different statistics (mean+range on 12 models vs. median on 22, plus a top-norm control only the 22-model version has); coincidental numeric collision (both use "0.935") | Yes — disambiguate the coincidence, state scope of each explicitly | No |
| 2 | Random/top-norm q1 gap formulas | Random = candidate − mean(5); top-norm = candidate − max(5); aggregated by median across models. Manuscript's own S2 caption already flags this as unresolved | Yes — pick and state one top-norm definition (recommend max-based) | No |
| 3 | "69-fold collapse" | 0.9346752.../0.0135382... = 69.04-fold — **correct**; the ~66.8 reading comes from dividing pre-rounded numbers | No (verified correct) | No |
| 4 | 23 vs. 22 vs. "16+6=22" counts | Arithmetic is consistent; Phi-3 is the 23rd model, selected by a third (undocumented in that paragraph) protocol | Yes — one clarifying sentence on provenance-paragraph scope | No |
| 5 | S1/S2 = 22 rows vs. Fig. 1B = 23 | Intentional — S1/S2 are causal-cohort supplements, exclude Phi-3 by design | Minor — one clarifying note | No |
| 6 | Phi-3's Fig. 1B value | Median q1 across its 6 frozen rows (0.9028), self-documented in `scale_ladder.csv` | No | No |
| 7 | "Rank 1" / top1_share in Fig. 1A | Two diagonal-proxy rankings: row-rank across the layer (Level 1) + within-row scalar share (Level 2); all three models rank 1/d_model | No (verified correct) | No |
| 8 | Activation-ratio detector definition | `max_t|act_t,k| / median_k(that)` per layer, threshold ≥5.0, abs values, domain-specific probes | No (verified correct) | No |
| 9 | "Activation-selected/-derived" labels | Manuscript already distinguishes 16 ratio-selected vs. 6 grandfathered models explicitly | No (already correct) | No |
| 10 | Bilinear operator, no nonlinearity | Confirmed — `U_k` is pure weight-space bilinear form; body text already says "bilinear" everywhere; only a section header omits it | Cosmetic only | No |
| 11 | Layer-relative Frobenius predictor | Actually divides by the **full-layer** row-population median, not "the five same-layer control rows" as Methods states | **Yes — Methods sentence is factually wrong, needs correction** | No |
| 12 | "Structural margin" (ρ=0.311/0.355) | = \|candidate q1 − max(5 top-norm controls)\|, same statistic as Fig. 1C's top-norm gap, just unsigned; not separately defined in Methods | Yes — add one defining sentence | No |
| 13 | Is `PR_spec` used? | No — defined once in Methods, never used in any Result/figure/correlation | Yes — relegate to supplement or note it's descriptive-only | No |
| 14 | GENERator: one direction or many? | Exactly one (seed 20260823+2371), swept across 5 scales; Discussion has one plural-noun slip | Yes — trivial wording fix | No |
| 15 | GENERator damage-matching bug | A documented reachability/self-match bug at c=0 was found and fixed pre-manuscript; current numbers independently re-derived and verified. The prompt's exact "averaging" description was not located as-worded | No (already fixed; current numbers verified) — flag the wording mismatch for follow-up if it points to something else | No |
| 16 | DNABERT-2 pair "regression check" | A real, directly-measured joint ablation; "regression check" means reproduction-test, not statistical regression, but collides with the paper's other "regression" usage nearby | Yes — rename to "direct pair-ablation experiment/reproduction check"; de-duplicate the repeated sentence | No |
| 17 | Phi-3 tomography in Methods | Deliberate negative-result disclosure, tied to a live structural value and Supplementary Fig. S1A | Minor — add a cross-reference pointer | No |

**Bottom line:** no numerical claim checked in this audit was found to be wrong, and no
figure or table needs regeneration. Three items are genuine **prose-level defects** worth
fixing before submission — **#11** (the layer-relative Frobenius sentence factually
misdescribes its own denominator), **#2/#12** (two live, unreconciled gap/margin
definitions the manuscript's own supplementary caption already flags), and **#16**
(a terminology collision the authors can resolve with a rename). Everything else is either
already correct as written (#3, #6, #7, #8, #9, #10 body text, #15) or a real, deliberate
design choice that only needs one added clarifying sentence for a careful reviewer doing the
same arithmetic this prompt's author did (#1, #4, #5, #13, #14, #17).
