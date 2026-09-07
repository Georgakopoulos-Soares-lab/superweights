#!/bin/bash
#SBATCH -J sweep-euk-gue
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sweep-euk-gue.%j_%a.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sweep-euk-gue.%j_%a.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH -A MCB26038

# ─────────────────────────────────────────────────────────────────────────────
# Compression sweep: GENERator EUK 3B  — ALL GUE tasks (sequential loop)
#
# Each array element fine-tunes then sweeps one GUE task and writes
#   results/compression_sweep_generator_<TASK_SLUG>.{json,png}
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Task list (index → task path inside GUE_ROOT) ────────────────────────────
TASKS=(
    "EPI/K562"
    "EPI/NHEK"
    "fungi/species_20"
    "mouse/0"
    "mouse/1"
    "mouse/2"
    "mouse/3"
    "mouse/4"
    "prom/prom_300_all"
    "prom/prom_300_notata"
    "prom/prom_300_tata"
    "prom/prom_core_all"
    "prom/prom_core_tata"
    "splice/reconstructed"
    "tf/0"
    "tf/1"
    "tf/2"
    "tf/3"
    "tf/4"
    "virus/covid"
    "virus/species_40"
)

# ── Environment ───────────────────────────────────────────────────────────────
source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_MODE=offline

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "${REPO}"

MODEL="generator"
GUE_ROOT="/work/11034/atzanakak/GUE/GUE"

mkdir -p "${REPO}/logs" "${REPO}/results"

echo "============================================================"
echo "HOST=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
date
echo "MODEL  : GENERator EUK 3B"
echo "TASKS  : ${#TASKS[@]}"
echo "============================================================"

SCRATCH_CKPT_BASE="/scratch/11034/${USER}/genomic-super-weights/results/gue_checkpoints"
mkdir -p "${SCRATCH_CKPT_BASE}"

for TASK in "${TASKS[@]}"; do
    TASK_SLUG="${TASK//\//_}"
    CKPT_DIR="${SCRATCH_CKPT_BASE}/${MODEL}_${TASK_SLUG}"
    OUT_JSON="${REPO}/results/compression_sweep_${MODEL}_${TASK_SLUG}.json"
    OUT_PNG="${REPO}/results/compression_sweep_${MODEL}_${TASK_SLUG}.png"
    mkdir -p "${CKPT_DIR}"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "TASK : ${TASK}"
    date
    echo "════════════════════════════════════════════════════════════"

    # ── Step 1: Fine-tune (skipped if checkpoint already exists) ──────────────
    if [ ! -f "${CKPT_DIR}/model_state.pt" ]; then
        echo "── Fine-tuning GENERator EUK on ${TASK} ──"
        python scripts/evaluation/run_gue_ablation.py \
            --model     "${MODEL}" \
            --task      "${TASK}" \
            --gue_root  "${GUE_ROOT}" \
            --ckpt_dir  "${CKPT_DIR}" \
            --epochs    4 \
            --lr        3e-5 \
            --batch     4 \
            --out       "results/gue_ablation_${MODEL}_${TASK_SLUG}.json"
        echo "── Fine-tuning complete ──"
    else
        echo "Checkpoint found — skipping fine-tuning."
    fi

    # ── Step 2: Compression sweep (skipped if output already exists) ─────────
    if [ -f "${OUT_JSON}" ]; then
        echo "Sweep output found — skipping compression sweep."
        continue
    fi
    echo "── Running compression sweep ──"
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

    echo "Results : ${OUT_JSON}"
    echo "Plot    : ${OUT_PNG}"
done

echo ""
echo "=== ALL TASKS DONE ==="
date
