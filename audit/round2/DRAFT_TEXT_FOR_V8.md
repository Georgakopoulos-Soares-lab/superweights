# Drop-in prose from the round-2 finalisation pass

Produced closing out `audit/round2/final_check.md`. Not manuscript-polished, but ready to
paste into v8 and edit in place. Sourced/verified against artifacts in `audit/round2/`,
`PART1_STRUCTURAL_MANUSCRIPT_PACKET.md`, `PART2_EVIDENCE_PACKET.md` — see
`audit/round2/manuscript_numbers.csv` for full provenance per number.

## Methods — tomography split-stability

To assess sensitivity of the observer-family comparison to the specific fit/calibration/
held-out partition, we performed 100 resplits per intervention strength, re-permuting role
assignment among the fixed pool of 118 measured non-singleton mask conditions (pool sizes
match exactly what was measured, so this resampling explores which subsets fall into
held-out, not new mask combinations). F3 outperformed F2 on held-out MAE in 100/100 splits
at both ε=0.5 and ε=1.0. Held-out R² was consistently higher for F3 than for the
additive-only families: at ε=0.5, median [2.5, 97.5] R² was −0.21 [−0.45, 0.06] for F0,
0.30 [−0.35, 0.57] for F1, 0.55 [0.26, 0.70] for F2, and 0.92 [0.79, 0.96] for F3; at
ε=1.0, the corresponding values were −0.45 [−1.14, −0.05], 0.31 [−0.33, 0.69], 0.56 [0.33,
0.70], and 0.76 [0.49, 0.88]. The selected ridge λ was unstable across resplits — F2's
selected λ ranged from 0.001 to 100 (ε=0.5) and 0.001 to 30 (ε=1.0) across the 100 splits,
and F3's from 0.001 to 10 (ε=0.5, 55% at λ=1.0) and 0.1 to 100 (ε=1.0, 84% concentrated at
λ∈{1.0, 3.0}) — indicating the point estimate depends on which conditions land in the fit
set, though the qualitative F3>F2 ranking did not.

## Methods — L9/r264 × L9/r294 epistasis provenance

The reported L9/r264 × L9/r294 epistasis values (ε=0.5: 0.053; ε=1.0: 2.012,
superadditive) come from a standalone qualitative regression check
(`baseline_regression_results.json`, `run_baseline_regression.py`) rather than the
78/20/20 fit/calibration/held-out mask pools used for the F0–F3 tomography fits —
confirmed directly against `dnabert2_mask_responses.json` that the joint-ablation mask
vector for this exact pair does not appear among the 118 measured non-singleton
conditions in those pools.

## Data and Code Availability

**Data and code availability.** Code and non-model artifacts are available at
`https://github.com/Georgakopoulos-Soares-lab/superweights` (commit
`c8751d6b8dcef99d9d9a68909f887c0ecf5536ad`). The repository contains all analysis code,
per-model structural and causal measurement scripts, and result artifacts (CSV/JSON) for
every experiment reported in the manuscript, with two exceptions inherent to their size:
the hg38 reference FASTA and WikiText-2 are not committed and must be obtained
independently (see below). Reproducing the full 22-model census additionally requires
downloading the pinned Hugging Face checkpoint revisions listed in Supplementary Table S1.

*External resources required:* (1) the human reference genome hg38 (FASTA + BED file of
sampled windows), used for all genomic-domain candidate detection and causal
intervention; (2) WikiText-2 (`wikitext-2-raw-v1` test split, via the Hugging Face
`datasets` library), used for text-domain candidate detection; (3) the 22 Hugging Face
model checkpoints, each pinned to a specific revision hash recorded in Table S1. All
commit hashes were resolved and confirmed except **NTv3**, whose checkpoint revision is
**unrecoverable** — the original detection run (E5/E6) used an unpinned reference and no
resolved commit hash was recorded; NTv3's results should be read with this caveat, as the
exact checkpoint state cannot be guaranteed to reproduce bit-for-bit on a fresh download.

*Reproducing the census:* candidate detection, structural spectrum computation (q1,
PR_spec, Frobenius norm), and causal ablation at ε∈{0.5, 1.0} for each model are each
driven by a per-domain detector/measurement script under `paper-salvage/experiments/` and
`scripts/`; per-model provenance (repo, requested/resolved revision, selection protocol)
is recorded in Supplementary Table S1 and `audit/detector_provenance.csv`.

*(Note: verified by fresh-clone check at the commit above — all artifacts cited in
`audit/verification_table.csv` are present. `results/mechanism/attention_sink_implicit_bias.json`
was stripped of a stale `implicit_bias` block that no longer matches the current
`run_attention_sink.py` — see that file's commit message for why.)*
