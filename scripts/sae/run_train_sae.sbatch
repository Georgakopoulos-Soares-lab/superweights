#!/bin/bash
#SBATCH -J sae-train
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-train.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-train.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A BCS25073

# ──────────────────────────────────────────────────────────────────────────────
# Train Batch-TopK SAE on pre-collected activation shards.
# Runs EUK (layer 4) then PROK (layer 2) sequentially.
#
# Dependencies:
#   data/sae_acts/generator_euk_layer4/   (from run_collect_euk.sbatch)
#   data/sae_acts/generator_prok_layer2/  (from run_collect_prok.sbatch)
#
# Outputs:
#   results/sae/generator_euk_layer4/sae_final.pt + training_log.json + plots
#   results/sae/generator_prok_layer2/sae_final.pt + ...
#
# Override defaults via --export on sbatch, e.g.:
#   sbatch --export=ALL,K=32,STEPS=100000 scripts/sae/run_train_sae.sbatch
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

# ── Hyperparameters (override via env vars) ───────────────────────────────────
DICT_MULT="${DICT_MULT:-4}"
K="${K:-64}"
LR="${LR:-2e-4}"
STEPS="${STEPS:-200000}"
BATCH="${BATCH:-2048}"
LAMBDA_AUX="${LAMBDA_AUX:-0.03125}"   # 1/32
LOG_EVERY="${LOG_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-20000}"

mkdir -p logs

echo "HOST=$(hostname)"
echo "dict_mult=${DICT_MULT}  k=${K}  lr=${LR}  steps=${STEPS}  batch=${BATCH}"
date
nvidia-smi || true

# ── EUK SAE ───────────────────────────────────────────────────────────────────
echo "========================================"
echo "Training EUK SAE (layer 4)"
echo "========================================"
python sae/train.py \
    --acts_dir   /scratch/11034/atzanakak/sae_acts/generator_euk_layer4 \
    --out_dir    "${REPO}/results/sae/generator_euk_layer4" \
    --dict_mult  "${DICT_MULT}" \
    --k          "${K}" \
    --lr         "${LR}" \
    --steps      "${STEPS}" \
    --batch      "${BATCH}" \
    --lambda_aux "${LAMBDA_AUX}" \
    --log_every  "${LOG_EVERY}" \
    --save_every "${SAVE_EVERY}"

echo "EUK SAE done."
date

# ── PROK SAE ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "Training PROK SAE (layer 2)"
echo "========================================"
python sae/train.py \
    --acts_dir   /scratch/11034/atzanakak/sae_acts/generator_prok_layer2 \
    --out_dir    "${REPO}/results/sae/generator_prok_layer2" \
    --dict_mult  "${DICT_MULT}" \
    --k          "${K}" \
    --lr         "${LR}" \
    --steps      "${STEPS}" \
    --batch      "${BATCH}" \
    --lambda_aux "${LAMBDA_AUX}" \
    --log_every  "${LOG_EVERY}" \
    --save_every "${SAVE_EVERY}"

echo "PROK SAE done."
date
echo "All SAE training complete."
