#!/bin/bash
#SBATCH -J sae-train-euk-v2
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-train-euk-v2.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-train-euk-v2.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A BCS25073

# ──────────────────────────────────────────────────────────────────────────────
# EUK SAE retrain (v2) — fixes 63.5% dead-feature collapse from v1.
#
# Root cause: lambda_aux=1/32 was too weak; dead features accumulated before
# the aux penalty could rescue them.
#
# Changes vs v1:
#   lambda_aux  0.03125 → 0.25   (8× stronger dead-feature penalty)
#   k           64      → 32     (fewer active features per token; less
#                                  competition means more features stay alive)
#   steps       200000  → 300000 (extra 100k to let aux loss finish its work)
#   out_dir     generator_euk_layer4_v2  (v1 checkpoint preserved)
#
# Usage:
#   sbatch scripts/sae/run_train_euk_v2.sbatch
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
K="${K:-32}"                        # v1=64; lower reduces per-token competition
LR="${LR:-2e-4}"
STEPS="${STEPS:-300000}"            # v1=200000; extra 100k for aux loss to work
BATCH="${BATCH:-2048}"
LAMBDA_AUX="${LAMBDA_AUX:-0.25}"   # v1=0.03125 (1/32); 8× stronger penalty
LOG_EVERY="${LOG_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-20000}"

mkdir -p logs

echo "HOST=$(hostname)"
echo "dict_mult=${DICT_MULT}  k=${K}  lr=${LR}  steps=${STEPS}  batch=${BATCH}  lambda_aux=${LAMBDA_AUX}"
date
nvidia-smi || true

echo "========================================"
echo "Training EUK SAE v2 (layer 4)"
echo "========================================"
python sae/train.py \
    --acts_dir   /scratch/11034/atzanakak/sae_acts/generator_euk_layer4 \
    --out_dir    "${REPO}/results/sae/generator_euk_layer4_v2" \
    --dict_mult  "${DICT_MULT}" \
    --k          "${K}" \
    --lr         "${LR}" \
    --steps      "${STEPS}" \
    --batch      "${BATCH}" \
    --lambda_aux "${LAMBDA_AUX}" \
    --log_every  "${LOG_EVERY}" \
    --save_every "${SAVE_EVERY}"

echo "EUK SAE v2 done."
date
