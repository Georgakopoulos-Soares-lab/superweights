# E13 Stage-0 reproduction gate

**Gate decision: PASS.**

| Model | ε | New relative Δloss | Old relative Δloss | Ratio | Decision |
|---|---:|---:|---:|---:|---|
| llama | 0.5 | +3.855% | +3.855% | 1.000× | REPRODUCES |
| llama | 1.0 | +300.795% | +300.795% | 1.000× | REPRODUCES |
| mistral | 0.5 | +301.379% | +301.379% | 1.000× | REPRODUCES |
| mistral | 1.0 | +287.149% | +287.149% | 1.000× | REPRODUCES |
| olmo | 0.5 | +0.629% | — | — | NO_MATCHED_OLD_EPSILON_REFERENCE |
| olmo | 1.0 | +111.776% | +111.776% | 1.000× | REPRODUCES |
| phi3 | 0.5 | +1.653% | +1.653% | 1.000× | REPRODUCES |
| phi3 | 1.0 | +4.623% | +4.623% | 1.000× | REPRODUCES |
| modernbert-base | 0.5 | +10.926% | +10.926% | 1.000× | REPRODUCES |
| modernbert-base | 1.0 | +357.044% | +357.044% | 1.000× | REPRODUCES |
| dnabert2 | 0.5 | +0.311% | +0.311% | 1.000× | REPRODUCES |
| dnabert2 | 1.0 | +0.830% | +0.830% | 1.000× | REPRODUCES |

The gate uses the preregistered same-sign and approximate-magnitude rule. A missing historical epsilon cell is reported rather than fabricated; the available matched cell still governs reproduction for that model.

See `STAGE1_RESOURCE_ESTIMATE.md` for the pre-launch compute/disk/loading estimate.
