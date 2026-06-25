#!/bin/bash
#SBATCH -J sweep-evo2
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sweep-evo2.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sweep-evo2.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A BCS25073

# ─────────────────────────────────────────────────────────────────────────────
# Compression sweep: EVO2 7B  (prom/prom_core_notata)
#
# EVO2 runs inside the evo2 Apptainer container (evo2.sif).
# The sweep script no longer requires sklearn or matplotlib at import time,
# so the container environment is sufficient for the full run.
# The "superweights" are the *potential* SW rows identified in layer 29
# via iterative detection (results/hydra_test_evo2.json).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Apptainer ────────────────────────────────────────────────────────────────
set +u
module load tacc-apptainer
set -u

# ── Paths ────────────────────────────────────────────────────────────────────
SIF="/work/11034/atzanakak/ls6/containers/evo2.sif"
REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
HF_HOME="/work/11034/atzanakak/ls6/huggingface/.hf-cache"
GUE_ROOT="/work/11034/atzanakak/GUE/GUE"

MODEL="evo2"
TASK="prom/prom_core_notata"
CKPT_DIR="${REPO}/results/gue_checkpoints/${MODEL}_prom_core_notata"
OUT_JSON="${REPO}/results/compression_sweep_${MODEL}_prom_core_notata.json"
OUT_PNG="${REPO}/results/compression_sweep_${MODEL}_prom_core_notata.png"
EVO2_SW_SRC="${REPO}/results/hydra_test_evo2.json"

mkdir -p "${REPO}/logs" "${REPO}/results" "${CKPT_DIR}"

echo "============================================================"
echo "HOST=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
date
echo "MODEL  : EVO2 7B"
echo "TASK   : ${TASK}"
echo "SIF    : ${SIF}"
echo "============================================================"

# ── No fine-tuning checkpoint needed — sweep runs on base model ───────────────
if [ ! -f "${CKPT_DIR}/model_state.pt" ]; then
    echo ""
    echo "[warn] No EVO2 checkpoint at ${CKPT_DIR}/model_state.pt"
    echo "       Sweep will run on the base (un-fine-tuned) model."
    echo ""
fi

# ── Compression sweep (inside container) ─────────────────────────────────────
echo ""
echo "── Running EVO2 compression sweep ──────────────────────────────────"
PYTHONPATH="" apptainer exec --nv --cleanenv \
  --bind /work,/tmp \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_CACHE="${HF_HOME}/hub" \
  --env TRANSFORMERS_CACHE="/work/11034/atzanakak/ls6/huggingface/transformers" \
  --env HF_HUB_OFFLINE=1 \
  --env WANDB_MODE=offline \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONPATH="${REPO}" \
  "${SIF}" \
  python3 scripts/compression/run_compression_sweep.py \
    --model          "${MODEL}" \
    --task           "${TASK}" \
    --gue_root       "${GUE_ROOT}" \
    --ckpt_dir       "${CKPT_DIR}" \
    --sw_index       "${REPO}/results/super_weight_index.json" \
    --evo2_sw_source "${EVO2_SW_SRC}" \
    --fracs          0.5 1 2 5 10 15 20 30 \
    --n_rand_seeds   10 \
    --out            "${OUT_JSON}" \
    --plot           "${OUT_PNG}"

echo ""
echo "=== DONE ==="
echo "Results: ${OUT_JSON}"
echo "Plot:    ${OUT_PNG}"
date

