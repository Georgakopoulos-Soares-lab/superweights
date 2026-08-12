#!/bin/bash
# Run evo1 broadcast impulse on CUDA_VISIBLE_DEVICES=1.
# Launch with:
#   CUDA_VISIBLE_DEVICES=1 nohup bash scripts/interpretability/run_evo1_nohup.sh \
#       > logs/nohup_evo1.log 2>&1 &
set -uo pipefail

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "$REPO"

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh
conda activate evo

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub
export PYTHONPATH=$REPO:${PYTHONPATH:-}

mkdir -p logs results

echo "========================================="
echo " nohup_evo1 start: $(date)"
echo "========================================="

echo "[1/1] broadcast: evo1  $(date)"
python scripts/interpretability/run_sw_broadcast_impulse.py \
    --model evo1 --epsilon 1.0 --n_controls 5 --seed 42 \
    --device cuda --out_dir results

echo "========================================="
echo " nohup_evo1 DONE: $(date)"
echo "========================================="
