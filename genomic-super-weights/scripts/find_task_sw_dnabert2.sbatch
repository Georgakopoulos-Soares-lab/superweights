#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# find_task_sw_dnabert2.sbatch
#
# Runs find_task_superweights.py for all major GUE tasks with DNABERT-2.
# Requires fine-tuned checkpoints already present under
#   results/gue_checkpoints/dnabert2_<task_leaf>/model_state.pt
# (produced by run_gue_ablation.py).
#
# Submit:  sbatch scripts/find_task_sw_dnabert2.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J task-sw-dnabert2
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/task-sw-dnabert2.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/task-sw-dnabert2.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A BCS25073

set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_OFFLINE=1

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

echo "HOST=$(hostname)"
date
nvidia-smi || true

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd $REPO
export PYTHONPATH=$REPO/stubs:$REPO:$PYTHONPATH

GUE_ROOT=/work/11034/atzanakak/GUE/GUE
OUT=/work/11034/atzanakak/glm_super_weight/genomic-super-weights/results/task_superweights.json

# ── GUE tasks to scan ─────────────────────────────────────────────────────────
# Add / remove tasks here.  Format: "group/task_name"
TASKS=(
    "prom/prom_core_notata"
    "prom/prom_core_all"
    "prom/prom_core_tata"
    "prom/prom_300_notata"
    "prom/prom_300_all"
    "prom/prom_300_tata"
    "EMP/H3K4me3"
    "EMP/H3K4me1"
    "EMP/H3K4me2"
    "EMP/H3K14ac"
    "EMP/H3K36me3"
    "EMP/H3K79me3"
    "EMP/H3K9ac"
    "EMP/H3"
    "EMP/H4"
    "EMP/H4ac"
    "EPI/GM12878"
    "EPI/HeLa-S3"
    "EPI/HUVEC"
    "EPI/IMR90"
    "EPI/K562"
    "EPI/NHEK"
    "splice/reconstructed"
    "virus/covid"
    "virus/species_40"
    "fungi/species_20"
    "tf/0"
    "tf/1"
    "tf/2"
    "tf/3"
    "tf/4"
    "mouse/0"
    "mouse/1"
    "mouse/2"
    "mouse/3"
    "mouse/4"
)

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "──────────────────────────────────────────────"
    echo "  Task: $TASK"
    echo "──────────────────────────────────────────────"
    python scripts/find_task_superweights.py \
        --model    dnabert2 \
        --task     "$TASK" \
        --gue_root "$GUE_ROOT" \
        --topk     50 \
        --z_thresh 2.5 \
        --min_drop 0.005 \
        --out      "$OUT" \
        || echo "[warn] $TASK failed — continuing"
done

echo ""
echo "══════════════════════════════════════════════"
echo "  All tasks done.  Running overlap analysis …"
echo "══════════════════════════════════════════════"

python scripts/analyze_task_sw_overlap.py \
    --in    "$OUT" \
    --model dnabert2 \
    --heatmap /work/11034/atzanakak/glm_super_weight/genomic-super-weights/results/task_sw_overlap_dnabert2.png

echo "Done."
date
