#!/bin/bash
# scripts/detection/submit_evo1_detection_ablation.sh
#
# Closes the SW-claim asymmetry on the negative side.
#
# Current paper: Evo2 / HybridNA / MegaDNA were declared SW-negative on the
# basis of perplexity-level ablation only — the same metric we elsewhere
# argue can mask functional effects (DNABERT-2 +1.5% MLM PPL but −25.5%
# splice MCC). A reviewer will catch this asymmetry.
#
# This pilot runs the activation sweep + perplexity ablation on Evo1, then
# attempts a row-level ablation of its top activation-rich MLP row. Evo1 is
# StripedHyena — structurally similar to Evo2 — and importantly has a
# working environment on this machine (conda env `evo`). Evo2 cannot be
# loaded outside its custom singularity container on the cluster.
#
# Two-stage protocol (mirrors the GENERator / DNABERT-2 pipeline):
#   STAGE 1  Detection — find top-activating row in each block.mlp.l3
#             (Evo1 has both ParallelGatedConvBlock and AttentionBlock,
#              both expose .mlp with l1/l2/l3.)
#   STAGE 2  Ablation  — zero that row and measure ΔPPL on the standard
#             probe set; compare to 10 random-row controls.
#
# Predicted outcome
#   - If ΔPPL ≪ random control envelope at Evo1's top candidate, then
#     SSM-style architectures genuinely smear representational burden:
#     the negative claim holds upstream and downstream by construction
#     (no concentrated row to ablate). This is what we want for the
#     architecture-class argument.
#   - If ΔPPL >> random, Evo1 has a hidden SW the paper missed, and the
#     Evo2 negative result needs a stronger downstream check too.
#
# IMPORTANT: must run inside the `evo` conda env. The `models.evo1_wrapper`
# imports `from evo import Evo`, available only there.
#
# Outputs
#   results/super_weight_index.json   (updated with "evo1" entry)
#   results/ablation_results.json     (updated with "evo1" entry)
#
# Estimated wall-clock: ~25 minutes on a single A100-80GB.

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

GPU="${GPU:-0}"
PYTHON="${PYTHON:-/home/nvidia/miniconda3/envs/evo/bin/python}"
PROBE="${PROBE:-actb_500}"
ABL_PROBE="${ABL_PROBE:-actb_full}"
MODE="${MODE:-superrow}"

mkdir -p logs

echo "=== STAGE 1: detection ==="
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -u scripts/detection/run_detection.py \
  --model     evo1 \
  --probe     "$PROBE" \
  --threshold 0.05 \
  --mode      "$MODE" \
  --out       results/super_weight_index.json \
  2>&1 | tee logs/evo1_detection.log

echo
echo "=== STAGE 2: ablation ==="
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -u scripts/detection/run_ablation.py \
  --model    evo1 \
  --probe    "$ABL_PROBE" \
  --sw_index results/super_weight_index.json \
  --out      results/ablation_results.json \
  2>&1 | tee logs/evo1_ablation.log
