#!/bin/bash
# Train Batch-TopK SAE on pre-collected activation shards.
# Runs EUK (layer 4) then PROK (layer 2) sequentially.
# Run on an interactive GPU node (idev session).
#
# Dependencies:
#   data/sae_acts/generator_euk_layer4/   (from run_collect_euk.sh)
#   data/sae_acts/generator_prok_layer2/  (from run_collect_prok.sh)
#
# Override defaults via env vars:
#   K=32 STEPS=100000 bash scripts/sae/run_train_sae.sh
#   EUK_ONLY=1 bash scripts/sae/run_train_sae.sh

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

# ── Hyperparameters ───────────────────────────────────────────────────────────
DICT_MULT="${DICT_MULT:-4}"
K="${K:-64}"
LR="${LR:-2e-4}"
STEPS="${STEPS:-200000}"
BATCH="${BATCH:-2048}"
LAMBDA_AUX="${LAMBDA_AUX:-0.03125}"
LOG_EVERY="${LOG_EVERY:-1000}"
SAVE_EVERY="${SAVE_EVERY:-20000}"

EUK_ACTS="${EUK_ACTS:-${REPO}/data/sae_acts/generator_euk_layer4}"
EUK_OUT="${EUK_OUT:-${REPO}/results/sae/generator_euk_layer4}"
PROK_ACTS="${PROK_ACTS:-${REPO}/data/sae_acts/generator_prok_layer2}"
PROK_OUT="${PROK_OUT:-${REPO}/results/sae/generator_prok_layer2}"

EUK_ONLY="${EUK_ONLY:-0}"
PROK_ONLY="${PROK_ONLY:-0}"

# ── Logging ───────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sae-train-${TIMESTAMP}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "SAE training — Batch-TopK"
echo "============================================================"
echo "HOST=$(hostname)"
echo "LOG=${LOG_FILE}"
echo "dict_mult=${DICT_MULT}  k=${K}  lr=${LR}  steps=${STEPS}  batch=${BATCH}  lambda_aux=${LAMBDA_AUX}"
echo "EUK_ONLY=${EUK_ONLY}  PROK_ONLY=${PROK_ONLY}"
nvidia-smi || true
date
echo "------------------------------------------------------------"

# ── EUK SAE ───────────────────────────────────────────────────────────────────
if [[ "${PROK_ONLY}" != "1" ]]; then
    echo ""
    echo "======== EUK SAE (layer 4) ========================================"
    echo "acts_dir : ${EUK_ACTS}"
    echo "out_dir  : ${EUK_OUT}"
    date

    python sae/train.py \
        --acts_dir   "${EUK_ACTS}" \
        --out_dir    "${EUK_OUT}" \
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
fi

# ── PROK SAE ──────────────────────────────────────────────────────────────────
if [[ "${EUK_ONLY}" != "1" ]]; then
    echo ""
    echo "======== PROK SAE (layer 2) ======================================="
    echo "acts_dir : ${PROK_ACTS}"
    echo "out_dir  : ${PROK_OUT}"
    date

    python sae/train.py \
        --acts_dir   "${PROK_ACTS}" \
        --out_dir    "${PROK_OUT}" \
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
fi

echo "------------------------------------------------------------"
echo "All SAE training complete."
date
echo "Log written to: ${LOG_FILE}"
