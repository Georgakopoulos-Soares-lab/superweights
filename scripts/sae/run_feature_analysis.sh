#!/bin/bash
# SAE feature analysis — EUK then PROK.
# Run on an interactive GPU node (idev session).
#
# Dependencies:
#   results/sae/generator_euk_layer4/sae_final.pt  (from run_train_sae.sh)
#   results/sae/generator_prok_layer2/sae_final.pt
#   results/sw_hexamer_causal.json
#
# Override defaults via env vars:
#   TOP_MONO=30 bash scripts/sae/run_feature_analysis.sh
#   EUK_ONLY=1  bash scripts/sae/run_feature_analysis.sh

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_OFFLINE=1

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

# ── Parameters ────────────────────────────────────────────────────────────────
TOP_MONO="${TOP_MONO:-20}"
TOP_SW_DEP="${TOP_SW_DEP:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"

EUK_CKPT="${EUK_CKPT:-${REPO}/results/sae/generator_euk_layer4/sae_final.pt}"
EUK_OUT="${EUK_OUT:-${REPO}/results/sae/generator_euk_layer4}"
EUK_HEXAMER="${EUK_HEXAMER:-${REPO}/results/sw_hexamer_causal.json}"

PROK_CKPT="${PROK_CKPT:-${REPO}/results/sae/generator_prok_layer2/sae_final.pt}"
PROK_OUT="${PROK_OUT:-${REPO}/results/sae/generator_prok_layer2}"
# Use PROK-specific hexamer JSON if available, fall back to EUK one
if [ -f "${REPO}/results/sw_hexamer_causal_generator_prok.json" ]; then
    PROK_HEXAMER="${PROK_HEXAMER:-${REPO}/results/sw_hexamer_causal_generator_prok.json}"
else
    PROK_HEXAMER="${PROK_HEXAMER:-${REPO}/results/sw_hexamer_causal.json}"
    echo "WARNING: PROK hexamer causal JSON not found — using EUK hexamer file as fallback."
fi

EUK_ONLY="${EUK_ONLY:-0}"
PROK_ONLY="${PROK_ONLY:-0}"

# ── Logging ───────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sae-analyze-${TIMESTAMP}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "SAE feature analysis"
echo "============================================================"
echo "HOST=$(hostname)"
echo "LOG=${LOG_FILE}"
echo "top_mono=${TOP_MONO}  top_sw_dep=${TOP_SW_DEP}  batch_size=${BATCH_SIZE}"
echo "EUK_ONLY=${EUK_ONLY}  PROK_ONLY=${PROK_ONLY}"
nvidia-smi || true
date
echo "------------------------------------------------------------"

# ── EUK analysis ─────────────────────────────────────────────────────────────
if [[ "${PROK_ONLY}" != "1" ]]; then
    echo ""
    echo "======== EUK feature analysis (layer 4) ==========================="
    echo "sae_ckpt     : ${EUK_CKPT}"
    echo "hexamer_src  : ${EUK_HEXAMER}"
    echo "out_dir      : ${EUK_OUT}"
    date

    python sae/analyze.py \
        --model       generator \
        --layer       4 \
        --sae_ckpt    "${EUK_CKPT}" \
        --hexamer_src "${EUK_HEXAMER}" \
        --out_dir     "${EUK_OUT}" \
        --top_mono    "${TOP_MONO}" \
        --top_sw_dep  "${TOP_SW_DEP}" \
        --batch_size  "${BATCH_SIZE}"

    echo "EUK analysis done."
    date
fi

# ── PROK analysis ─────────────────────────────────────────────────────────────
if [[ "${EUK_ONLY}" != "1" ]]; then
    echo ""
    echo "======== PROK feature analysis (layer 2) =========================="
    echo "sae_ckpt     : ${PROK_CKPT}"
    echo "hexamer_src  : ${PROK_HEXAMER}"
    echo "out_dir      : ${PROK_OUT}"
    date

    python sae/analyze.py \
        --model       generator_prokaryote \
        --layer       2 \
        --sae_ckpt    "${PROK_CKPT}" \
        --hexamer_src "${PROK_HEXAMER}" \
        --out_dir     "${PROK_OUT}" \
        --top_mono    "${TOP_MONO}" \
        --top_sw_dep  "${TOP_SW_DEP}" \
        --batch_size  "${BATCH_SIZE}"

    echo "PROK analysis done."
    date
fi

echo "------------------------------------------------------------"
echo "All feature analysis complete."
date
echo "Log written to: ${LOG_FILE}"
