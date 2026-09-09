# Claude Code task: finalisation pass for "Structure Is Not Mechanism"

All experiments are complete. This task produces the remaining artifacts needed to write v8 and
submit. No new measurements except where Section 1 says otherwise.

Reference documents: `audit/AUDIT_REPORT.md`, `audit/round2/AUDIT_ROUND2_REPORT.md`, and the
manuscript edit list I'll paste separately. Same evidence discipline as before — `STORED` /
`RECOMPUTED` / `NOT FOUND` labels, provenance for every number, report conflicts rather than
resolving them silently.

Work through the sections in order and report after each.

---

## Section 1 — Close the one open number (BLOCKING)

The revised Abstract and Results will state that the candidate exceeded the median top-norm control
in **20 of 22 models at ε=0.5 and 19 of 22 at ε=1.0**. I derived those counts from the sign-flip
discussion, not from a file.

Compute both counts directly from `audit/round2/structural_vs_causal_gap.csv` as
`count(causal_topK_gap > 0)` per epsilon, list the models on the negative side of each, and confirm
or correct the two figures. Also report the same counts for the random-control comparison (expected
18/22 and 20/22) so all four appear in one place.

If either count differs from 20/22 or 19/22, say so plainly — it appears in the Abstract.

---

## Section 2 — Regenerate three figures

All three need source-data CSVs alongside them, in `audit/round2/figures/` and
`manuscript/figures/source_data/`. Match the existing figure style in
`manuscript/figures/` (fonts, marker conventions, encoder/decoder shape coding, text/genomic
colour coding) — find the existing plotting scripts and reuse their style helpers rather than
starting fresh.

**Fig. 1C → two panels.** Left: candidate q1 versus five random same-layer controls. Right:
candidate q1 versus five top-norm same-layer controls. Same y-axis on both so the collapse from a
median gap of 0.935 to 0.014 is visible. Label the 7 models with negative gaps in the right panel.
Full 22-model cohort in both panels, not the original 12.

**Fig. 2 → add a panel** showing candidate versus top-norm controls at both intervention strengths,
matching the existing 2A/2B symmetric-log treatment. Label the four sign-flip models (MosaicBERT,
Qwen2.5-7B, Qwen2.5-0.5B, NTv3). Keep the existing model ordering so panels are readable across.

**Fig. 4C → replace.** Two panels: GC versus c for the random-direction grid, and GC versus native
NLL with the row-2371 α-sweep, the five inert control rows, and the random-direction grid on shared
axes. Mark the intact baseline (GC 0.4204) and the ablation point (0.3065, NLL 8.754). The point of
the figure is that composition tracks damage monotonically, so make the damage axis the organising
one in the second panel.

Render each to PNG and PDF, then look at the rendered output and confirm it says what it should
before reporting.

---

## Section 3 — Tomography split-stability numbers for Methods

From the Section 2 work in round 2, give me the repeated-split result formatted as two or three
sentences I can drop into the tomography Methods: number of resplits, the fraction in which F3 beats
F2 on held-out MAE at each epsilon, the median and 2.5/97.5 percentiles of held-out R² per family,
and the ridge-λ instability. State explicitly whether the resplits were drawn from the wider
combination space or were role-permutations of the existing measured conditions — that distinction
has to be in the text.

Also draft one sentence for Methods stating that the L9/r264 × L9/r294 epistasis values come from
`baseline_regression_results.json` rather than the 78/20/20 mask pools, since that exact pair is not
among the sweep conditions.

---

## Section 4 — Supplementary tables

Produce publication-ready supplementary tables, as CSV plus a rendered version:

**S1 — Model panel and provenance.** One row per model: display name, HF repo, requested revision,
resolved revision, domain, architecture, non-embedding parameters, candidate layer/row, selection
protocol (current ratio detector / grandfathered activation / literature-derived), activation ratio
where recorded, and whether the frozen candidate is the ratio-argmax under the current detector.

**S2 — Structural metrics.** Per model: candidate q1, PR_spec, ‖U_k‖_F, layer median norm,
layer-relative Frobenius, mean random-control q1, mean top-norm-control q1, and both gaps.

**S3 — Causal census, full.** Per model per epsilon: candidate R, all five random control R values,
all five top-norm control R values, both medians, both gaps.

**S4 — Structure–function correlations.** The 16-row table from
`audit/round2/structure_function_correlations.csv`, with the underpowered flag retained.

**S5 — GENERator conditions.** The full α and c grids with NLL, GC, CI, and fraction-of-ablation-
damage.

Flag any cell you cannot fill from artifacts rather than leaving it blank.

---

## Section 5 — Data and code availability

That section of the manuscript is empty. Draft it.

First verify the claim you're about to make: clone the repository fresh into a temporary directory
from the current HEAD, and check that every artifact cited in `audit/verification_table.csv` is
present. Report anything missing — the round-2 hygiene pass force-added the results directories, but
I want this tested rather than assumed. Confirm `results/e7_phase1_detection_evo2.json` is tracked.

Then draft the section: repository URL, commit hash, what is in it, which external resources are
required (hg38, WikiText-2, the HF checkpoints with their pinned revisions), and how a reader
reproduces the census. Note NTv3's unrecoverable revision explicitly.

Also confirm the `implicit_bias` block removal in `run_attention_sink.py` is committed and that
`results/mechanism/attention_sink_implicit_bias.json` has been regenerated or the stale block
stripped — the file will be requested at review.

---

## Section 6 — Master numbers ledger

The thing I most need for writing. One CSV, `audit/round2/manuscript_numbers.csv`, listing every
numerical claim that will appear in v8, with columns:

```
section, claim_text, value, units, source_file, obtained_via, git_commit, notes
```

Include: all abstract numbers; the Part 1 structural values and the regression coefficients; both
control comparisons at both epsilons; the four subgroup medians; every named per-model extreme;
the correlation table; the tomography metrics; and every GENERator value.

Order it by position in the manuscript so I can check the draft against it top to bottom. Where a
number appears more than once (abstract and results), give it one row and list both locations.

This is the artifact I'll use to verify v8 line by line, so completeness matters more than brevity.

---

## Reporting

Report after each section. Section 1 first and on its own — it gates two sentences in the Abstract.

If anything in the audit reports conflicts with what you find now, say so rather than assuming the
newer number is right.
