#!/bin/bash
#SBATCH -J sae-train-euk-v3
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-train-euk-v3.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-train-euk-v3.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A BCS25073

# ──────────────────────────────────────────────────────────────────────────────
# EUK SAE retrain (v3) — adds Anthropic-style neuron resampling.
#
# v1 (k=64, λ=1/32):  63.5% dead — lambda_aux too weak
# v2 (k=32, λ=0.25):  83.0% dead — lower k made collapse worse
#
# v3 strategy:
#   k           = 64          back to v1 value (k=32 was worse)
#   lambda_aux  = 0.125       moderate (between v1 and v2)
#   resample_every = 25000    neuron resampling every 25k steps
#                             (12 events over 300k steps)
#   steps       = 300000      same extra budget as v2
#
# Neuron resampling (Anthropic, 2023):
#   Every resample_every steps, identify dead features and reset their
#   encoder/decoder weights to point toward inputs with high reconstruction
#   error.  Adam momentum is zeroed for those parameters.  This breaks the
#   feedback loop where dead features accumulate monotonically.
#
# Usage:
#   sbatch scripts/sae/run_train_euk_v3.sbatch
# ──────────────────────────────────────────────────────────────────────────────

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

# ── Hyperparameters ───────────────────────────────────────────────────────────
DICT_MULT="${DICT_MULT:-4}"
K="${K:-64}"                           # back to v1; k=32 made collapse worse
LR="${LR:-2e-4}"
STEPS="${STEPS:-300000}"
BATCH="${BATCH:-2048}"
LAMBDA_AUX="${LAMBDA_AUX:-0.125}"     # moderate (v1=1/32, v2=1/4)
LOG_EVERY="${LOG_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
RESAMPLE_EVERY="${RESAMPLE_EVERY:-25000}"
RESAMPLE_SCALE="${RESAMPLE_SCALE:-0.2}"

mkdir -p logs

echo "HOST=$(hostname)"
echo "dict_mult=${DICT_MULT}  k=${K}  lr=${LR}  steps=${STEPS}  batch=${BATCH}"
echo "lambda_aux=${LAMBDA_AUX}  resample_every=${RESAMPLE_EVERY}  resample_scale=${RESAMPLE_SCALE}"
date
nvidia-smi || true

echo "========================================"
echo "Training EUK SAE v3 (layer 4, + neuron resampling)"
echo "========================================"
python sae/train.py \
    --acts_dir       /scratch/11034/atzanakak/sae_acts/generator_euk_layer4 \
    --out_dir        "${REPO}/results/sae/generator_euk_layer4_v3" \
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

echo "EUK SAE v3 done."
date
