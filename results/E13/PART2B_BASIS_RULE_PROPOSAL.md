# Part 2B basis audit and stop decision

> **Superseded for basis construction on 2026-08-25.** A subsequent source-code
> audit established that the frozen activation ratio is a per-row statistic:
> `channel_max[row] / median(channel_max)` within the same layer. E11 retained
> only the global winner after computing this vector. The follow-up structural
> census therefore applies its unchanged `>= 5.0` threshold to every row and is
> the governing basis audit. This document remains a record of why a norm-based
> top-K rule was not adopted.

This is a design conclusion, not evidence that the 22 models have no
interactions.  The existing E13 manifest contains one primary candidate per
cohort model by construction; a one-row basis cannot identify an interaction.

## Scope and inputs audited

The Part 2B cohort is the 22 models in `results/E13/PART1_CAUSAL_CENSUS_SUMMARY.md`.
Phi-3 is excluded from that cohort and is used below only as provenance for
the earlier tomography implementation.  All sources consulted here are
structural or methodological artifacts; no new combination-intervention
response was inspected or used for selection.

The relevant frozen input is `results/E13/candidate_manifest.json`, built by
`paper-salvage/experiments/E13_full_cohort_causal_census/build_candidate_manifest.py`
from `results/E11/scale_ladder_backfilled.csv`.  Its code asserts one candidate
for every non-Phi-3 model.  Thus it has structural quantities only for the
winner, not a candidate set from which a principled K can be determined.

## Stage 0: recovered prior multi-row bases

| Case | Coordinates and K | Selection information | Causal selection? | Primary provenance |
|---|---|---|---|---|
| DNABERT-2 | K=10: (5,603), (3,86), (3,399), (9,264), (9,294), (3,603), (3,641), (7,603), (6,603), (5,86) | The pre-existing `super_weight_index.json["dnabert2"]["results"]` high-gain ensemble, ordered by activation `out_max` descending.  This is activation magnitude, not exact `||U_k||_F`, q1, or a top-K layer-relative-norm ranking. | No. E9 froze the entire already-established list before E9 responses. | `paper-salvage/experiments/E9_mechanistic_tomography/BASIS_FREEZE.md`; `results/super_weight_index.json`; E9 freeze commit `2a76d99f1406adcf9838bc42add69235c6c7cdd3`. |
| MosaicBERT | K=10: (0,287), (0,444), (1,287), (11,287), (0,302), (5,287), (3,287), (10,666), (9,287), (10,79) | Fixed top 10 over all 9,216 output rows by exact `||U_k||_F` divided by that layer's median.  q1 is annotation only; the E8 activation-detector row (9,287) is rank 9. | No. The ranking was computed and frozen before E10 tomography. | `paper-salvage/experiments/E10_nlp_architecture_causal/ENCODER_BASIS_FREEZE.md`; `results/e10_encoder_row_ranking_mosaicbert.json`; `results/e10_selection_audit_mosaicbert.json`. |
| ModernBERT-base | K=10: (15,251), (0,251), (11,251), (0,67), (9,251), (4,251), (1,251), (5,251), (10,251), (1,67) | Fixed top 10 over all 16,896 output rows by exact `||U_k||_F` divided by the layer median.  q1 is annotation only; the E8 activation-detector row (15,251) is rank 1. | No. The ranking was computed and frozen before E10 tomography. | `paper-salvage/experiments/E10_nlp_architecture_causal/ENCODER_BASIS_FREEZE.md`; `results/e10_encoder_row_ranking_modernbert.json`; `results/e10_selection_audit_modernbert.json`; E10 final commit `626cddd`. |
| Phi-3-mini (external only) | K=6: (2,525), (2,1693), (2,1113), (4,525), (4,1113), (4,1693) | Fixed top 6 by exact `||U_k||_F` divided by layer median; q1 annotation only. | No for the E10 structural freeze.  Its later tomography was licensed by prior E10 singleton results, but that did not change membership. | `paper-salvage/experiments/E10b_phi3_tomography/PHI3_BASIS_AUDIT.md`; `results/e10_exact_uknorm_phi3.json`; E10b provenance commits `1128fea`, `507c5fd`, `13d2fa7`, `3374fb5`, `fafd44d`. |

The E13 one-primary rule is different again: E7/E8/E11's activation detector
(`DownProjRecorder`, ratio >= 5) supplied a single detected candidate per
model, after which E13 backfilled exact magnitude and layer-relative magnitude
for that already-fixed coordinate.  It neither ranked every row cohort-wide
nor defined a multi-row eligibility set.  `q1` is not an appropriate repair:
E10's protocol correction establishes it as a scale-invariant concentration
quantity, not a high-gain magnitude score.

## Why no universal rule can be frozen

Two superficially simple choices both violate the Part 2B requirements:

1. **Use the frozen E13 detector output.** It gives K=1 for all 22 models.
   This is a valid description of the existing artifact, but it produces no
   interaction-identifiable models and does not audit whether additional
   structurally prominent components exist.  It therefore cannot answer the
   requested census question.

2. **Select the top K by exact layer-relative `||U_k||_F`.** E10 used K=10 for
   two encoders and Phi-3 used K=6, but no project-wide, pre-existing rule sets
   either K or an eligibility cutoff.  Fixing K=10 everywhere would pad models
   with ordinary rows solely to enable tomography; choosing a cutoff now (for
   example importing 5x from the activation detector into weight space) would
   invent a new structural metric/threshold.  Either is disallowed.

Nor can the two approaches be silently combined: DNABERT-2's prior basis is an
activation-`out_max` ensemble, while the other three are norm-ranked; only
MosaicBERT and ModernBERT have complete all-row ranking artifacts.  The frozen
cohort inputs do not contain equivalent row-wise activation or exact-norm
distributions for the remaining 20 models.  The single winner's q1,
`||U_k||_F`, and layer-relative norm cannot determine how many other rows are
structurally exceptional.

## Requested K table: status

No proposed universal rule exists, so K and selected coordinates are **not
defined** for every Part 2B cohort model.  Reporting fabricated K values would
misrepresent a proposed rule as a frozen structural result.  Under the already
frozen E13 *one-primary* rule only, the audit result is 22/22 K=1, with each
coordinate listed in `results/E13/candidate_manifest.json`; that is not a
Part 2B multi-component basis and is not carried forward as one.

## Consequences

- Do not run Part 2B tomography or generate its requested figures/tables.
- Do not infer absent interactions from the current K=1 manifest.
- Existing DNABERT-2, MosaicBERT, ModernBERT, and Phi-3 results remain
  mechanistic case studies on their explicitly declared, basis-dependent
  selections; they cannot be converted into a full-cohort estimate.

A future cohort interaction census would first need a separately locked,
causal-blind structural study that computes the same complete row-level inputs
for every model and predefines an eligibility rule that permits variable K
without padding.  That would be a new experiment, not a continuation justified
by the current frozen E13 basis.
