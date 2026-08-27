# Results of the round-2 finalisation pass (`audit/round2/final_check.md`)

Everything below was produced by Claude Code working through `final_check.md` end to end,
then committed. Repo: `https://github.com/Georgakopoulos-Soares-lab/superweights`,
HEAD as of this writeup: `e89968f4059f17f03fb6ea93d053397e2a367f75` (re-check
`git rev-parse HEAD` if more commits land after this).

Commits this pass produced, in order:
```
8ecfcc3  Commit round-2 finalisation audit in full (audit/round2/*)
902f69e  Commit PART2_EVIDENCE_PACKET.md (Part 2 functional-criticality evidence)
50c87a5  Force-add results/E13/raw_4b/*.json (4b-queue per-model census raw records)
c8751d6  Strip stale implicit_bias block from attention_sink_implicit_bias.json
e89968f  Persist round-2 finalisation drop-in prose
```

Evidence discipline: `STORED` (copied verbatim from an artifact) / `RECOMPUTED` (derived
here from raw data) / `NOT FOUND` (flagged, not left blank). Every number below traces to
`audit/round2/manuscript_numbers.csv` (120 lines incl. header) unless stated otherwise.

---

## Section 1 — The one blocking number (CLOSED, confirmed correct)

Computed directly from `audit/round2/structural_vs_causal_gap.csv`
(`count(gap > 0)` per epsilon, 44 rows = 22 models × 2 epsilons):

| Comparison | ε=0.5 | ε=1.0 |
|---|---|---|
| candidate > median **top-norm** control | **20/22** ✅ | **19/22** ✅ |
| candidate > median **random** control | 18/22 ✅ | 20/22 ✅ |

**The abstract's "20 of 22 at ε=0.5 and 19 of 22 at ε=1.0" is correct as written** — this
was the gating number and it holds.

Negative-side models:
- Top-norm gap, ε=0.5 (2): MosaicBERT (−0.00190), NTv3 (−0.00498)
- Top-norm gap, ε=1.0 (3): NTv3 (−0.00796), Qwen2.5-0.5B (−0.00295), Qwen2.5-7B (−0.00136)
- Random-control gap, ε=0.5 (4): EuroBERT-2.1B, EuroBERT-610m, MosaicBERT, NTv3
- Random-control gap, ε=1.0 (2): EuroBERT-2.1B, NTv3

**NTv3 is negative on every comparison at both epsilons** — the one universally-negative
model in the cohort; worth a sentence in the text if not already there.

---

## Section 2 — Three regenerated figures (final, fixed, visually verified)

**Update after this report was first written:** the new panels were folded into the
unified `paper-salvage/figures/main/fig1_structural.png`, `fig2_causal.png`,
`fig4_generator.png` themselves (not left as standalone sub-panel files) — those are now
the complete, publication-ready figures, each committed with source data in
`paper-salvage/figures/source_data/`. The standalone sub-panel versions described below
still exist in `audit/round2/figures/` for the audit trail, but **use the `main/` files for
the manuscript.**

**`fig1c_full22_two_panel.{png,pdf}`** — Fig. 1C, full 22-model cohort, two panels sharing
one y-axis and one spanning "C" label (fixed from an earlier C1/C2 split; sub-panels now
distinguished by subtitle instead: "random same-layer controls" / "top-norm same-layer
controls"). Left: candidates sit far above random controls almost everywhere (median gap
0.935). Right: candidates sit at the top-norm control band (median gap 0.014); the 7
negative-gap models are text-labeled (SmolLM2-360M, EuroBERT-210M, EuroBERT-2.1B,
ModernBERT-large, DNABERT-2, GEN-EUK-3B, GenomeOcean-4B).

**`fig2_topnorm_panel.{png,pdf}`** — new Fig. 2 panel D, symlog y-axis, candidate-minus-
top-norm-control causal gap at both epsilons, all 22 models. 2/22 negative at ε=0.5, 3/22
at ε=1.0 (union of 4: MosaicBERT, NTv3, Qwen2.5-0.5B, Qwen2.5-7B), each labeled.
**Correction after this report was first written:** this panel was initially folded into
the wrong figure (`fig2_causal.py`, an older decoder-spectrum/OLMo-dissociation/tomography
figure that is supplementary/case-study material, not the manuscript's Figure 2 — see
`PART2_EVIDENCE_PACKET.md`'s own note to that effect). It has since been moved into the
real Figure 2, `build_part2_evidence_packet.py`'s `fig2_part2_functional_criticality.png`
(the 22-model functional-criticality census, panels A/B = candidate vs. random controls at
ε=0.5/1.0, C = q1-vs-effect correlation), as panel D there — verified against the target
caption: 20/22 exceed the median top-norm control at ε=0.5, 19/22 at ε=1.0, and the
top-norm controls' own median effect (+0.035% at ε=0.5, +0.110% at ε=1.0) is annotated on
the panel. **Use `paper-salvage/figures/main/fig2_part2_functional_criticality.png` for
the manuscript's Figure 2 — not `fig2_causal.png`.**

**`fig4c_replacement.{png,pdf}`** — Fig. 4C replacement, two panels. Left: GC vs.
random-direction scale *c*, x-axis now starts cleanly at 0 (fixed — previously padded into
an empty negative-*c* region that read as missing data), ablation/baseline reference
lines. Right: GC vs. NLL (damage pool) with row2371's own α-sweep, the 5 inert controls,
and the random-direction grid on shared axes, damage as the organizing axis — converges
near ablation at high damage, near baseline at low damage.

---

## Section 3 — Tomography Methods text (ready to paste)

> To assess sensitivity of the observer-family comparison to the specific
> fit/calibration/held-out partition, we performed 100 resplits per intervention strength,
> re-permuting role assignment among the fixed pool of 118 measured non-singleton mask
> conditions (pool sizes match exactly what was measured, so this resampling explores
> which subsets fall into held-out, not new mask combinations). F3 outperformed F2 on
> held-out MAE in 100/100 splits at both ε=0.5 and ε=1.0. Held-out R² was consistently
> higher for F3 than for the additive-only families: at ε=0.5, median [2.5, 97.5] R² was
> −0.21 [−0.45, 0.06] for F0, 0.30 [−0.35, 0.57] for F1, 0.55 [0.26, 0.70] for F2, and 0.92
> [0.79, 0.96] for F3; at ε=1.0, the corresponding values were −0.45 [−1.14, −0.05], 0.31
> [−0.33, 0.69], 0.56 [0.33, 0.70], and 0.76 [0.49, 0.88]. The selected ridge λ was
> unstable across resplits — F2's selected λ ranged from 0.001 to 100 (ε=0.5) and 0.001 to
> 30 (ε=1.0) across the 100 splits, and F3's from 0.001 to 10 (ε=0.5, 55% at λ=1.0) and 0.1
> to 100 (ε=1.0, 84% concentrated at λ∈{1.0, 3.0}) — indicating the point estimate depends
> on which conditions land in the fit set, though the qualitative F3>F2 ranking did not.

> The reported L9/r264 × L9/r294 epistasis values (ε=0.5: 0.053; ε=1.0: 2.012,
> superadditive) come from a standalone qualitative regression check
> (`baseline_regression_results.json`, `run_baseline_regression.py`) rather than the
> 78/20/20 fit/calibration/held-out mask pools used for the F0–F3 tomography fits —
> confirmed directly against `dnabert2_mask_responses.json` that the joint-ablation mask
> vector for this exact pair does not appear among the 118 measured non-singleton
> conditions in those pools.

---

## Section 4 — Five supplementary tables

`audit/round2/tables/`, CSV + rendered Markdown each: S1 (22 rows, model panel/provenance),
S2 (22 rows, structural metrics), S3 (44 rows, full causal census), S4 (16 rows,
structure-function correlations), S5 (106 rows, GENERator conditions).

**Notable finding:** S3's per-top-norm-control causal R values were thought to exist only
as a stored median. They don't — the raw per-model files `results/E13/raw_4b/<model>.json`
(now committed, all 22) each store the 5 individual top-norm-control conditions; extracted
and rank-ordered, and the median of the 5 reproduces the previously-stored median to
<1e-9 for all 44 rows. **Not a gap — just previously unaggregated.** S3 now has all 5
individual values per model per epsilon.

**Genuine NOT FOUND cells, flagged rather than blank:**
- S1 `activation_ratio` — absent for all 22 models by design (`census_master.csv` leaves
  it blank; the 6 "grandfathered activation" models' source JSON has no ratio field at all).
- S5 GC/CI columns — absent where the source condition has 0 prompts (all 78 alpha-grid
  rows are NLL-only) or lacks a bootstrap CI (14/20 GC-bearing c-grid rows report mean only).

**Definitional note (not an error):** S2 carries two different candidate-vs-top-norm gap
columns — `topk_by_norm_gap` (candidate minus the *max* of the 5 top-norm controls) vs. a
mean-based column — deliberately not reconciled, flagged in the table.

---

## Section 5 — Data and Code Availability (ready to paste, hash filled in)

> **Data and code availability.** Code and non-model artifacts are available at
> `https://github.com/Georgakopoulos-Soares-lab/superweights` (commit
> `e89968f4059f17f03fb6ea93d053397e2a367f75` — **re-verify this is still HEAD before
> submission**, since further commits will move it). The repository contains all analysis
> code, per-model structural and causal measurement scripts, and result artifacts
> (CSV/JSON) for every experiment reported in the manuscript, with two exceptions inherent
> to their size: the hg38 reference FASTA and WikiText-2 are not committed and must be
> obtained independently (see below). Reproducing the full 22-model census additionally
> requires downloading the pinned Hugging Face checkpoint revisions listed in
> Supplementary Table S1.
>
> *External resources required:* (1) the human reference genome hg38 (FASTA + BED file of
> sampled windows), used for all genomic-domain candidate detection and causal
> intervention; (2) WikiText-2 (`wikitext-2-raw-v1` test split, via the Hugging Face
> `datasets` library), used for text-domain candidate detection; (3) the 22 Hugging Face
> model checkpoints, each pinned to a specific revision hash recorded in Table S1. All
> commit hashes were resolved and confirmed except **NTv3**, whose checkpoint revision is
> **unrecoverable** — the original detection run (E5/E6) used an unpinned reference and no
> resolved commit hash was recorded; NTv3's results should be read with this caveat.
>
> *Reproducing the census:* candidate detection, structural spectrum computation (q1,
> PR_spec, Frobenius norm), and causal ablation at ε∈{0.5, 1.0} for each model are each
> driven by a per-domain detector/measurement script under `paper-salvage/experiments/`
> and `scripts/`; per-model provenance is recorded in Supplementary Table S1 and
> `audit/detector_provenance.csv`.

**What made this true (fixed this session, was not true before):** a fresh-clone check
originally found **23 artifacts cited in `audit/verification_table.csv` were untracked**
— all of `audit/round2/*`, all 9 cited `results/E13/raw_4b/*.json` files, and
`PART2_EVIDENCE_PACKET.md`. All 23 are now committed (see commit log above) and
**re-verified present in a fresh clone.**

Also fixed: `results/mechanism/attention_sink_implicit_bias.json` carried a stale
`implicit_bias` sub-block from before `scripts/mechanism/run_attention_sink.py` removed
that test on 2026-08-26 (the test is vacuous by construction on a causal decoder — position
0's hidden state cannot depend on any later token regardless of shuffling). The stale block
has been stripped so the artifact matches what the current script actually emits; the
`attention` block (the manuscript's real evidence) is untouched.

---

## Section 6 — Master numbers ledger

`audit/round2/manuscript_numbers.csv` — **119 rows** (+header), columns `section,
claim_text, value, units, source_file, obtained_via, git_commit, notes`, ordered Abstract →
R1-structural → R2-causal/correlations → R3-tomography → R4-generator → Methods/Availability
(matching `paper-salvage/docs/MANUSCRIPT_SOURCE_OF_TRUTH.md` §1.3's four-part Results
structure). This is the number-by-number check file — walk v8's draft against it top to
bottom before submission.

Covers: all 4 abstract numbers; the 12-model E11 structural table (q1/PR_spec/‖U_k‖_F/
layer-relative Frobenius); both control comparisons at both epsilons; all 4 subgroup
medians (text/decoder +85.47%, text/encoder +0.65%, genomic/encoder +0.02%,
genomic/decoder +0.68%, all descriptive/confounded — say so); named per-model extremes
(strongest/weakest per epsilon, the 4 sign-flip models, NTv3 as universal negative,
GenomeOcean-4B's worst rank); the full 16-row correlation table; every tomography
split-stability metric; and the full GENERator grid (c-scale, α-sweep, 5 controls,
dose-response, epistasis pair, damage fractions).

---

## Full artifact manifest for this pass

```
audit/round2/final_check.md                          -- the task spec this pass answered
audit/round2/manuscript_numbers.csv                   -- Section 6, master ledger
audit/round2/DRAFT_TEXT_FOR_V8.md                     -- Sections 3+5 prose, standalone
audit/round2/FINAL_CHECK_RESULTS.md                   -- this file
audit/round2/tables/S1_model_panel_provenance.{csv,md}
audit/round2/tables/S2_structural_metrics.{csv,md}
audit/round2/tables/S3_causal_census_full.{csv,md}
audit/round2/tables/S4_structure_function_correlations.{csv,md}
audit/round2/tables/S5_generator_conditions.{csv,md}
paper-salvage/figures/main/fig1_structural.{png,pdf}                    -- USE THIS: unified Fig 1 (A/B/C-new/D)
paper-salvage/figures/main/fig2_part2_functional_criticality.{png,pdf}  -- USE THIS: the real Fig 2 (A/B/C/D-new,
                                                                             22-model functional-criticality census)
paper-salvage/figures/main/fig4_generator.{png,pdf}                     -- USE THIS: unified Fig 4 (A/B/C-new)
paper-salvage/figures/main/fig2_causal.{png,pdf}                        -- NOT Fig 2 -- older decoder-spectrum/OLMo-
                                                                             dissociation/tomography-adequacy figure,
                                                                             supplementary/case-study material only
                                                                             (see PART2_EVIDENCE_PACKET.md's own note)
paper-salvage/figures/fig1_structural.py, fig4_generator.py             -- scripts for Fig 1 / Fig 4
paper-salvage/experiments/E13_full_cohort_causal_census/build_part2_evidence_packet.py  -- script for the real Fig 2
paper-salvage/figures/source_data/fig1_panel_c_full22_source.csv
paper-salvage/figures/source_data/fig2_panel_d_source.csv
paper-salvage/figures/source_data/fig4_panel_c_source.csv
audit/round2/figures/fig1c_full22_two_panel.{png,pdf}         -- standalone new-panel-only versions (audit trail)
audit/round2/figures/fig2_topnorm_panel.{png,pdf}              -- standalone new-panel-only versions (audit trail)
audit/round2/figures/fig4c_replacement.{png,pdf}               -- standalone new-panel-only versions (audit trail)
audit/round2/tables/S1-S5_*.tsv                                -- tab-separated versions for spreadsheet paste
PART2_EVIDENCE_PACKET.md                              -- now committed (was untracked)
results/E13/raw_4b/*.json (22 files)                  -- now force-added (was gitignored)
results/mechanism/attention_sink_implicit_bias.json   -- stale block stripped
```

Everything above traces to raw artifacts with `STORED`/`RECOMPUTED`/`NOT FOUND` labels in
`audit/round2/manuscript_numbers.csv` and the S1–S5 tables. No conflicts were found against
`audit/AUDIT_REPORT.md` or `audit/round2/AUDIT_ROUND2_REPORT.md` beyond what's already
flagged above.
