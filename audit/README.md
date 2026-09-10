# audit/

The verification record. Its purpose is to be a *record*, so files here are retained even when
nothing currently cites them — an audit you can prune to taste is not an audit.

## What the passes were

The manuscript's numbers were checked three times, each pass adversarial against the state of
the draft at the time:

1. **Verification.** Every reported number re-read from its raw artifact. Produced the
   provenance tables at this directory's top level, including `verification_table.csv`.
2. **Re-derivation** (`rederivations/`, previously named `round2/`). The contested quantities
   were recomputed independently rather than re-read: the top-norm control comparison, the
   random-direction GC extension, the structure–function correlations, the tomography split
   stability, and the detector coordinates for NTv3 and GENERator-EUK. This is where a
   disagreement would have surfaced as a different number, not a different opinion.
3. **Clarification.** A follow-up pass on wording and remaining ambiguities.

The narrative reports from all three passes were removed on 2026-09-09 when the repository was
reduced to code, data and provenance; they are in git history. **The data those passes
produced is kept in full**, which is why `rederivations/` still holds 40-odd files that
nothing imports.

## The tables a reviewer will want

| file | contents |
|---|---|
| `census_master.csv` | **The canonical census table**: 22 models × 44 columns — coordinates, structural metrics, endpoint, baseline loss, candidate and control effects at both ε, resolved Hub revisions, control seed streams |
| `detector_provenance.csv` | How each candidate was selected, and whether that was the current ratio rule or the legacy activation rule |
| `detector_provenance_exp2_resolution.csv` | The 2026-09-08 resolution of every legacy candidate, with the paired causal comparison. **Additive** — it does not modify the file above |
| `census_controls_structural.csv` | Structural metrics for the control rows |
| `verification_table.csv`, `provenance.json` | Pass-1 number-by-number verification and artifact provenance |
| `tomography_pair_coeffs.csv`, `tomography_splits.csv` | F3 pairwise coefficients and the fit/calibration/held-out partitions (feed Supplementary S6) |
| `rederivations/manuscript_numbers.csv` | Pass 2's independent re-derivation of the numbers as printed |
| `rederivations/topk_norm_controls.csv`, `structure_function_correlations.csv`, `tomography_split_stability.csv` | The recomputed control, correlation and stability values |
| `rederivations/section3_*_detector_recheck.json` | Independent detector re-checks for NTv3 and GENERator-EUK |

`scripts/` holds the builders for the top-level tables; `rederivations/scripts/` holds pass 2's.

## Convention

No pass ever rewrites an earlier pass's output, and no audit file rewrites `census_master.csv`.
Corrections are added as new files that state what they supersede. That is why several tables
here look redundant: they are successive independent measurements of the same quantity, which
is the point.
