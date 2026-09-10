# Supplementary tables

All eight tables in one place, as `.tsv` (machine-readable) and `.md` (human-readable with a
provenance header). S1–S5 additionally retain the `.csv` form they were produced in.

| table | rows | status | contents |
|---|---|---|---|
| **S1** | 22 | unchanged | Model panel and provenance: repo, requested/resolved revision, domain, architecture, non-embedding parameters, candidate coordinate, selection protocol, ratio-argmax flag |
| **S2** | 22 | **extended** | Structural metrics (q₁, PR_spec, ‖U_k‖_F, layer-relative Frobenius, control gaps) **+ the 24-input stability columns** |
| **S3** | 49 | **extended** | The 44-row causal census **+ a selection-rule column and the disagreement-coordinate rows** |
| **S4** | 16 | unchanged | Structure–function correlations for q₁ and layer-relative Frobenius, with leave-one-out range and an underpowered flag |
| **S5** | 106 | unchanged | GENERator conditions: the α grid and the random-direction c grid |
| **S6** | 98 | **new** | DNABERT-2 finite-intervention tomography — §A observer families F0–F3 with 100-resplit intervals, §B all 45 pairwise coefficients at both strengths |
| **S7** | 72 | **new** | Within-layer activation-ratio sweep, 36 rows in each of two decoders |
| **S8** | 13 | **new** | Two critical rows in SmolLM2-1.7B layer 7 — §A single and pairwise ablations, §B row geometry against a same-layer null |

## Layout

One file per table, named `S<n>_<what it is>`, in `.tsv` and `.md`. S1, S4 and S5 additionally
keep the `.csv` they were produced in.

`sources/` holds the two round-2 audit tables that the builder reads as input to S2 and S3.
They are reformatted verbatim from stored artifacts; the extension adds columns and rows to
them rather than recomputing what they already contain, so the verbatim provenance of those
values is preserved. They are inputs, not deliverables — cite S2 and S3, not `sources/`.

## Regenerating

S2, S3, S6, S7 and S8 are built from committed artifacts by

```bash
python scripts/census_analysis/build_supplementary_tables.py
```

No value in them is transcribed by hand. S1, S4 and S5 are reformatted verbatim from the
round-2 audit and are not regenerated here.

## Two things to read before comparing numbers across tables

**S3 mixes endpoints, deliberately.** The 44 census rows use each model's native-loss
endpoint. The appended DNABERT-2 disagreement rows use MLM loss on six fixed DNA probes —
a different scale. They are comparable to each other, not to the census rows, and the
`endpoint` column says so on every row. NTv3 contributes no disagreement row because under the
six-probe input its two selection rules pick the same coordinate.

**S8 uses the opposite epistasis sign convention to the rest of the project.** Its endpoint is
a loss *increase*, so a positive interaction term means super-additivity. The DNABERT-2
epistasis figures elsewhere (e.g. −33.63 pp) come from an accuracy endpoint, where synergy is
negative. Convert before comparing.
