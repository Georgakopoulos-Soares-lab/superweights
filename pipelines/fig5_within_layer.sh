#!/usr/bin/env bash
# Figure 5 + the second-critical-row result: the within-layer activation-ratio sweeps.
#
#   1. graded sweep in a text decoder   (SmolLM2-1.7B  L7/r227)
#   2. graded sweep in a genomic decoder (GENERator-EUK-3B L4/r2371)
#   3. the second critical row and its masking interaction (SmolLM2 L7)
#      plus the data-free row geometry behind the cosine figures
#   4. cross-domain comparison + the two-panel figure
#
# Stages 1-3 are independent and run in parallel across free GPUs; stage 4 depends on all of
# them. ~10 min wall clock on two idle A100s. Each sweep self-validates against stored census
# quantities and aborts on mismatch.
source "$(dirname "$0")/_lib.sh"
log "Figure 5 pipeline"

parallel_stage 3 <<EOF
results/within_layer_sweep/within_model_slope_smollm2_1.7b.json|fig5_sweep_text|$PY_GENERATOR|scripts/within_layer_sweep/run_within_model_slope_text.py
results/within_layer_sweep/within_model_slope.json|fig5_sweep_genomic|$PY_GENERATOR|scripts/within_layer_sweep/run_within_model_slope.py
results/within_layer_sweep/smollm2_second_row_epistasis.json|fig5_second_row|$PY_GENERATOR|scripts/within_layer_sweep/run_smollm2_second_row_epistasis.py
EOF

stage results/within_layer_sweep/smollm2_161_749_geometry.json fig5_geometry \
      "$PY_GENERATOR" scripts/within_layer_sweep/run_smollm2_row_geometry.py

FORCE=1 stage results/within_layer_sweep/fig_within_model_slope_2panel.pdf fig5_plot \
      "$PY_GENERATOR" scripts/within_layer_sweep/plot_within_model_slope_2panel.py
log "done -- figure at results/within_layer_sweep/fig_within_model_slope_2panel.{png,pdf}"
