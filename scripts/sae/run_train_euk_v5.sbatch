#!/bin/bash
#SBATCH -J sae-train-euk-v5
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-train-euk-v5.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-train-euk-v5.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A BCS25073

# EUK SAE v5: fix the dict-too-large problem
#
# DIAGNOSIS: EUK layer-4 only has ~4500 usable features (v1 found this naturally).
# Forcing 12288 features (dict_mult=4) means 8000 are structurally dead.
# Resampling cycled 7386 slots in v4 but every rescued feature died again.
# FIX: dict_mult=1 → 3072 features. No resampling. Let the model find features naturally.
#
# Hyperparams:
#   dict_mult=1  → 3072 features (1x expansion, less than EUK's natural ~4500)
#   k=64         → same sparsity budget as v1/PROK
#   lambda_aux=0.03125  → light aux loss, prevent early collapse
#   resample_every=0    → DISABLED (it was making things worse)
#   steps=200000
#
# Usage:
#   sbatch scripts/sae/run_train_euk_v5.sbatch

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

DICT_MULT="${DICT_MULT:-1}"
K="${K:-64}"
LR="${LR:-2e-4}"
STEPS="${STEPS:-200000}"
BATCH="${BATCH:-2048}"
LAMBDA_AUX="${LAMBDA_AUX:-0.03125}"
LOG_EVERY="${LOG_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
RESAMPLE_EVERY="${RESAMPLE_EVERY:-0}"
RESAMPLE_SCALE="${RESAMPLE_SCALE:-0.2}"

mkdir -p logs

echo "HOST=$(hostname)"
echo "dict_mult=${DICT_MULT}  k=${K}  lr=${LR}  steps=${STEPS}  batch=${BATCH}"
echo "lambda_aux=${LAMBDA_AUX}  resample_every=${RESAMPLE_EVERY} (0=disabled)"
date
nvidia-smi || true

echo "========================================"
echo "Training EUK SAE v5 (layer 4, dict_mult=1, no resampling)"
echo "========================================"
python sae/train.py \
    --acts_dir       /scratch/11034/atzanakak/sae_acts/generator_euk_layer4 \
    --out_dir        "${REPO}/results/sae/generator_euk_layer4_v5" \
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

echo "EUK SAE v5 done."
date
