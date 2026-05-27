#!/bin/bash
# scripts/evaluation/submit_ntv3_splice_multiseed.sh
#
# Tier-1 follow-up #1: NTv3 GUE multi-seed splice/reconstructed.
#
# Confirms or refutes the single-seed −86.2% MCC pilot. If confirmed,
# NTv3 becomes the second encoder positive and strengthens the
# architecture-class claim.
#
# Runs three seeds (0, 1, 2). Fine-tuned checkpoints reused across the
# SW-ablation / random-row evaluation that follows. The existing
# run_gue_multiseed.py already supports --model ntv3 (see _load_model
# branch at line 90), and pulls SW coords from results/super_weight_index.json
# (NTv3 entry: layer 11, row 1472).
#
# Outputs
#   results/gue_multiseed_ntv3_splice.json
#   results/gue_checkpoints_multiseed/ntv3_reconstructed/seed{0,1,2}/
#
# Estimated wall-clock: ~1.5 h on a single A100-80GB.

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

GPU="${GPU:-0}"
SEEDS="${SEEDS:-0 1 2}"
GUE_ROOT="${GUE_ROOT:-/home/nvidia/data/gue/GUE}"
PYTHON="${PYTHON:-/home/nvidia/miniconda3/envs/generator/bin/python}"

mkdir -p logs

CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" -u scripts/evaluation/run_gue_multiseed.py \
  --model    ntv3 \
  --task     splice/reconstructed \
  --gue_root "$GUE_ROOT" \
  --seeds    $SEEDS \
  --ckpt_dir results/gue_checkpoints_multiseed/ntv3_reconstructed \
  --out      results/gue_multiseed_ntv3_splice.json \
  --device   cuda \
  2>&1 | tee logs/ntv3_splice_multiseed.log
