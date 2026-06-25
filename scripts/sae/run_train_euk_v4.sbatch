#!/bin/bash
#SBATCH -J sae-train-euk-v4
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-train-euk-v4.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-train-euk-v4.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A BCS25073

# EUK SAE v4: more aggressive anti-collapse
# - k=48 (less competition)
# - lambda_aux=0.5 (stronger penalty)
# - resample_every=5000 (very frequent)
# - resample_scale=0.3 (more aggressive resets)
# - steps=300000
#
# Usage:
#   sbatch scripts/sae/run_train_euk_v4.sbatch

set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

DICT_MULT="${DICT_MULT:-4}"
K="${K:-48}"
LR="${LR:-2e-4}"
STEPS="${STEPS:-300000}"
BATCH="${BATCH:-2048}"
LAMBDA_AUX="${LAMBDA_AUX:-0.5}"
LOG_EVERY="${LOG_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
RESAMPLE_EVERY="${RESAMPLE_EVERY:-5000}"
RESAMPLE_SCALE="${RESAMPLE_SCALE:-0.3}"

mkdir -p logs

echo "HOST=$(hostname)"
echo "dict_mult=${DICT_MULT}  k=${K}  lr=${LR}  steps=${STEPS}  batch=${BATCH}"
echo "lambda_aux=${LAMBDA_AUX}  resample_every=${RESAMPLE_EVERY}  resample_scale=${RESAMPLE_SCALE}"
date
nvidia-smi || true

echo "========================================"
echo "Training EUK SAE v4 (layer 4, aggressive resampling)"
echo "========================================"
python sae/train.py \
    --acts_dir       /scratch/11034/atzanakak/sae_acts/generator_euk_layer4 \
    --out_dir        "${REPO}/results/sae/generator_euk_layer4_v4" \
    --dict_mult      "${DICT_MULT}" \
    --k              "${K}" \
    --lr             "${LR}" \
    --steps          "${STEPS}" \
    --batch          "${BATCH}" \
    --lambda_aux     "${LAMBDA_AUX}" \
    --log_every      "${LOG_EVERY}" \
    --save_every     "${SAVE_EVERY}" \
    --resample_every "${RESAMPLE_EVERY}" \
    --resample_scale "${RESAMPLE_SCALE}"

echo "EUK SAE v4 done."
date
