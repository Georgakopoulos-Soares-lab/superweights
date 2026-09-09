# Pipelines

One runner per experiment group. All are **idempotent**: a stage whose output artifact already
exists is skipped, so an interrupted run resumes rather than recomputing, and re-running a
finished pipeline costs nothing. Set `FORCE=1` to recompute regardless.

| pipeline | produces | notes |
|---|---|---|
| `fig5_within_layer.sh` | Figure 5, the second-critical-row result, Table S5/S6 inputs | 3 sweeps in parallel, then the plot. ~10 min on two idle A100s |
| `detector_provenance.sh` | EXP2 resolution for every legacy candidate | Runs the positive control **first** and stops if it fails |
| `census.sh` | The frozen 22-model singleton causal census | Restartable; pass slugs to subset. Cold checkpoint download dominates |

Shared helpers are in `_lib.sh`: a GPU worker pool that dispatches to devices with no resident
process, and artifact-based skipping. Logs go to `results/pipeline_logs/<stage>.log`.

Environment variables: `PY_GENERATOR`, `PY_DNABERT` (interpreter paths), `WORKERS` (census
parallelism, default 4), `HF_HOME`, `LOGDIR`, `FORCE`.

Two design choices worth knowing:

* **`detector_provenance.sh` gates on its own positive control.** SmolLM2-1.7B was selected by
  the current ratio rule, so `frozen == ratio-argmax` holds there by construction. If that run
  fails, the harness is broken and the 7B results would be meaningless, so the pipeline exits
  rather than continuing.
* **The plotting stage always re-runs.** It is seconds long and it carries its own check that no
  legend or annotation box covers a data point, which is worth re-verifying whenever the
  underlying sweep changes.
