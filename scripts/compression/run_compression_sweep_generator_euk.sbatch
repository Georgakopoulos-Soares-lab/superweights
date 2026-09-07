#!/bin/bash
#SBATCH -J sweep-generator-euk
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sweep-generator-euk.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sweep-generator-euk.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A BCS25073

# ─────────────────────────────────────────────────────────────────────────────
# Compression sweep: GENERator EUK 3B  (prom/prom_core_notata)
#
# Workflow:
#   1. Fine-tune GENERator EUK on the GUE task (saves best checkpoint)
#   2. Run progressive row-pruning sweep (l1_low, prox_far, prox_near, random,
#      plus the new sw_only condition)
#   3. Generate a per-model plot
#
# After running all four model sweeps, combine with:
#   python scripts/analysis/plot_compression_multimodel.py \
#     --inputs \
#       dnabert2:results/compression_sweep_dnabert2_prom_core_notata.json \
#       generator:results/compression_sweep_generator_prom_core_notata.json \
#       generator_prokaryote:results/compression_sweep_generator_prokaryote_prom_core_notata.json \
#       evo2:results/compression_sweep_evo2_prom_core_notata.json \
#     --out results/compression_multimodel_prom_core_notata.png
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_MODE=offline

# ── Repo root ───────────────────────────────────────────────────────────────
REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "${REPO}"

echo "============================================================"
echo "HOST=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
date
echo "MODEL  : GENERator EUK 3B"
echo "TASK   : prom/prom_core_notata"
echo "============================================================"
TASK="prom/prom_core_notata"
MODEL="generator"
GUE_ROOT="/work/11034/atzanakak/GUE/GUE"
CKPT_DIR="${REPO}/results/gue_checkpoints/${MODEL}_prom_core_notata"
OUT_JSON="${REPO}/results/compression_sweep_${MODEL}_prom_core_notata.json"
OUT_PNG="${REPO}/results/compression_sweep_${MODEL}_prom_core_notata.png"

mkdir -p "${REPO}/logs" "${REPO}/results" "${CKPT_DIR}"

# ── Step 1: Fine-tune on GUE task (skipped if checkpoint already exists) ──────
if [ ! -f "${CKPT_DIR}/model_state.pt" ]; then
    echo ""
    echo "── Fine-tuning GENERator EUK on ${TASK} ────────────────────────"
    python scripts/evaluation/run_gue_ablation.py \
        --model     "${MODEL}" \
        --task      "${TASK}" \
        --gue_root  "${GUE_ROOT}" \
        --ckpt_dir  "${CKPT_DIR}" \
        --epochs    4 \
        --lr        3e-5 \
        --batch     4 \
        --out       results/gue_ablation_generator_prom_core_notata.json
    echo "── Fine-tuning complete ─────────────────────────────────────────"
else
    echo "Checkpoint found at ${CKPT_DIR}/model_state.pt — skipping fine-tuning."
fi

# ── Step 2: Compression sweep ─────────────────────────────────────────────────
echo ""
echo "── Running compression sweep ────────────────────────────────────────"
python scripts/compression/run_compression_sweep.py \
    --model      "${MODEL}" \
    --task       "${TASK}" \
    --gue_root   "${GUE_ROOT}" \
    --ckpt_dir   "${CKPT_DIR}" \
    --sw_index   results/super_weight_index.json \
    --fracs      0.5 1 2 5 10 15 20 30 \
    --n_rand_seeds 10 \
    --out        "${OUT_JSON}" \
    --plot       "${OUT_PNG}"

echo ""
echo "=== DONE ==="
echo "Results: ${OUT_JSON}"
echo "Plot:    ${OUT_PNG}"
date
