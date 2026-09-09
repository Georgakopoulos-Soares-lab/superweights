# audit/

The verification record. Its purpose is to be a *record*, so files here are retained even when
nothing currently cites them — an audit you can prune to taste is not an audit.

Three rounds, each adversarial against the state of the manuscript at the time:

| round | output | what it checked |
|---|---|---|
| 1 | `AUDIT_REPORT.md` + the top-level CSV/JSON here | Every manuscript number against its raw artifact; produced the provenance tables |
| 2 | `round2/AUDIT_ROUND2_REPORT.md`, `round2/FINAL_CHECK_RESULTS.md` | Re-derived the contested figures (top-norm controls, random-direction GC, detector coordinates) and drafted corrected text |
| 3 | `round3/` | Follow-up checks |

## The tables a reviewer will want

| file | contents |
|---|---|
| `census_master.csv` | **The canonical census table**: 22 models × 44 columns — coordinates, structural metrics, endpoint, baseline loss, candidate and control effects at both ε, resolved Hub revisions, control seed streams |
| `detector_provenance.csv` | How each candidate was selected, and whether that was the current ratio rule or the legacy activation rule |
| `detector_provenance_exp2_resolution.csv` | The 2026-09-09 resolution of every legacy candidate, with the paired causal comparison. **Additive** — it does not modify the file above |
| `census_controls_structural.csv` | Structural metrics for the control rows |
| `verification_table.csv`, `provenance.json` | Round-1 number-by-number verification and artifact provenance |
| `tomography_pair_coeffs.csv`, `tomography_splits.csv` | F3 pairwise coefficients and the fit/calibration/held-out partitions |
| `round2/manuscript_numbers.csv` | Round-2's re-derivation of the numbers as printed |

`scripts/` in each round holds the code that produced that round's artifacts.

## Convention

No round ever rewrites an earlier round's output, and no audit file rewrites `census_master.csv`.
Corrections are added as new files that state what they supersede. That is why several tables
here look redundant: they are successive independent measurements of the same quantity, which
is the point.
