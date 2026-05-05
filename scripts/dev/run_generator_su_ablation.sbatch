#!/bin/bash
# Super-weight ablation on GENERator using sequence_understanding.py.
#
# Runs three conditions per benchmark task:
#   1. Baseline   – fine-tune + evaluate normally
#   2. Prune SW   – zero super row (layer=4, row=2371) → re-evaluate
#   3. Rand ctrl  – zero 10 random rows → mean metric
# Plus an optional layer sweep (--sweep) to find task-specific super rows.
#
# Usage:
#   sbatch run_generator_su_ablation.sbatch                    # default: H3K4me3
#   # NT epigenetic mark tasks
#   TASK="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised:H3K4me3" sbatch run_generator_su_ablation.sbatch
#   TASK="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised:H3K27ac" sbatch run_generator_su_ablation.sbatch
#   # GENERator native tasks
#   TASK="GenerTeam/gener-tasks:gene_classification" sbatch run_generator_su_ablation.sbatch
#   TASK="GenerTeam/gener-tasks:taxonomic_classification" sbatch run_generator_su_ablation.sbatch
#
# To submit all tasks at once:
#   for TASK in "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised:H3K4me3" \
#               "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised:H3K27ac" \
#               "GenerTeam/gener-tasks:gene_classification"; do
#       TASK=$TASK sbatch run_generator_su_ablation.sbatch
#   done

#SBATCH -J generator-su-ablation
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/generator-su-ablation.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/generator-su-ablation.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A BCS25073

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

# ── Paths ─────────────────────────────────────────────────────────────────────
GENERATOR_SRC="/work/11034/atzanakak/ls6/GENERator/src"
SW_INDEX="/work/11034/atzanakak/glm_super_weight/genomic-super-weights/results/super_weight_index.json"
ABLATION_OUT="/work/11034/atzanakak/glm_super_weight/genomic-super-weights/results/generator_su_ablation.json"
CKPT_BASE="/work/11034/atzanakak/ls6/GENERator/results/ablation"
HF_CONFIG="/work/11034/atzanakak/ls6/GENERator/configs/hf_configs/sequence_understanding.yaml"
LOG_DIR="/work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs"
mkdir -p "${LOG_DIR}"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME="GenerTeam/GENERator-v2-eukaryote-3b-base"
DOWN_PROJ_PATTERN="model.layers.{i}.mlp.down_proj"
NUM_LAYERS=30

# ── Task (override via env or default to first task) ─────────────────────────
# TASK may also include a subset after a colon: "dataset_name:subset_name"
TASK="${TASK:-InstaDeepAI/nucleotide_transformer_downstream_tasks_revised:H3K4me3}"

# Split dataset:subset
DATASET_NAME="${TASK%%:*}"
SUBSET_NAME="${TASK##*:}"
[ "${SUBSET_NAME}" = "${TASK}" ] && SUBSET_NAME=""

# Safe name for directories (replace / with _)
SAFE_TASK=$(echo "${TASK}" | tr '/:' '__')
OUT_DIR="${CKPT_BASE}/${SAFE_TASK}"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "HOST=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
date
echo "MODEL  : ${MODEL_NAME}"
echo "TASK   : ${TASK}"
echo "DATASET: ${DATASET_NAME}"
echo "SUBSET : ${SUBSET_NAME:-<none>}"
echo "OUT_DIR: ${OUT_DIR}"
echo "============================================================"

# ── Build python args ─────────────────────────────────────────────────────────
PYTHON_ARGS=(
    "${GENERATOR_SRC}/tasks/downstream/sequence_understanding.py"
    --model_name       "${MODEL_NAME}"
    --dataset_name     "${DATASET_NAME}"
    --output_dir       "${OUT_DIR}"
    --hf_config_path   "${HF_CONFIG}"
    --sw_index         "${SW_INDEX}"
    --down_proj_pattern "${DOWN_PROJ_PATTERN}"
    --num_transformer_layers "${NUM_LAYERS}"
    --n_rand_controls  10
    --ablation_out     "${ABLATION_OUT}"
)

# Add subset if provided
[ -n "${SUBSET_NAME}" ] && PYTHON_ARGS+=(--subset_name "${SUBSET_NAME}")

# Re-use checkpoint if it already exists
BEST_MODEL_DIR="${OUT_DIR}/best_model"
[ -d "${BEST_MODEL_DIR}" ] && PYTHON_ARGS+=(--skip_training)

echo "Running: python3 ${PYTHON_ARGS[*]}"
echo "------------------------------------------------------------"

python3 "${PYTHON_ARGS[@]}"

echo "============================================================"
echo "Done: ${TASK}"
date
