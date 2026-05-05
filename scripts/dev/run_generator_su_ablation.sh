#!/bin/bash
# Super-weight ablation on GENERator — interactive GPU node version.
#
# Modes (set via env vars):
#   LINEAR_PROBE=1   — embedding extraction + sklearn LR (fast, ~10 min) [recommended]
#   NO_FINETUNE=1    — zero-shot ablation on base model (near-random baseline)
#   default          — full fine-tune via sequence_understanding.py
#
# Usage:
#   LINEAR_PROBE=1 TASK="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised:H3K27ac" bash scripts/run_generator_su_ablation.sh
#   LINEAR_PROBE=1 TASK="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised:H3K4me3" bash scripts/run_generator_su_ablation.sh
#   LINEAR_PROBE=1 TASK="GenerTeam/gener-tasks:gene_classification" bash scripts/run_generator_su_ablation.sh

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="/work/11034/atzanakak/evo2_env"

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_MODE=offline

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO="/work/11034/atzanakak/glm_super_weight/genomic-super-weights"
GENERATOR_SRC="/work/11034/atzanakak/ls6/GENERator/src"
SW_INDEX="${REPO}/results/super_weight_index.json"
ABLATION_OUT="${REPO}/results/generator_su_ablation.json"
PROBE_ABLATION_OUT="${REPO}/results/generator_probe_ablation.json"
CKPT_BASE="/work/11034/atzanakak/ls6/GENERator/results/ablation"
HF_CONFIG="/work/11034/atzanakak/ls6/GENERator/configs/hf_configs/sequence_understanding.yaml"
LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-GenerTeam/GENERator-v2-eukaryote-3b-base}"
DOWN_PROJ_PATTERN="${DOWN_PROJ_PATTERN:-model.layers.{i}.mlp.down_proj}"
NUM_LAYERS="${NUM_LAYERS:-30}"

# ── Task ──────────────────────────────────────────────────────────────────────
TASK="${TASK:-InstaDeepAI/nucleotide_transformer_downstream_tasks_revised:H3K4me3}"
DATASET_NAME="${TASK%%:*}"
SUBSET_NAME="${TASK##*:}"
[ "${SUBSET_NAME}" = "${TASK}" ] && SUBSET_NAME=""
SAFE_TASK=$(echo "${TASK}" | tr '/:' '__')
OUT_DIR="${CKPT_BASE}/${SAFE_TASK}"
mkdir -p "${OUT_DIR}"

# ── Log ───────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/generator-su-ablation-${SAFE_TASK}-${TIMESTAMP}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "HOST=$(hostname)"
echo "GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
date
echo "LOG    : ${LOG_FILE}"
echo "MODEL  : ${MODEL_NAME}"
echo "TASK   : ${TASK}"
echo "SUBSET : ${SUBSET_NAME:-<none>}"
echo "============================================================"

# ── Run ───────────────────────────────────────────────────────────────────────
if [[ "${LINEAR_PROBE:-0}" == "1" ]]; then
    echo "MODE: embedding-probe (probe_ablation.py)"
    PYTHON_ARGS=(
        "${REPO}/probe_ablation.py"
        --model_name             "${MODEL_NAME}"
        --dataset_name           "${DATASET_NAME}"
        --sw_index               "${SW_INDEX}"
        --down_proj_pattern      "${DOWN_PROJ_PATTERN}"
        --num_transformer_layers "${NUM_LAYERS}"
        --n_rand_controls        "${N_RAND_CONTROLS:-10}"
        --max_length             "${LINEAR_PROBE_MAXLEN:-512}"
        --ablation_out           "${PROBE_ABLATION_OUT}"
        --hf_cache               "${HF_HOME}"
    )
    [ -n "${SUBSET_NAME}" ] && PYTHON_ARGS+=(--subset_name "${SUBSET_NAME}")

elif [[ "${NO_FINETUNE:-0}" == "1" ]]; then
    echo "MODE: no-finetune (sequence_understanding.py --no_finetune)"
    PYTHON_ARGS=(
        "${GENERATOR_SRC}/tasks/downstream/sequence_understanding.py"
        --model_name             "${MODEL_NAME}"
        --dataset_name           "${DATASET_NAME}"
        --output_dir             "${OUT_DIR}"
        --hf_config_path         "${HF_CONFIG}"
        --sw_index               "${SW_INDEX}"
        --down_proj_pattern      "${DOWN_PROJ_PATTERN}"
        --num_transformer_layers "${NUM_LAYERS}"
        --n_rand_controls        10
        --ablation_out           "${ABLATION_OUT}"
        --no_finetune
    )
    [ -n "${SUBSET_NAME}" ] && PYTHON_ARGS+=(--subset_name "${SUBSET_NAME}")

else
    echo "MODE: full fine-tune (sequence_understanding.py)"
    PYTHON_ARGS=(
        "${GENERATOR_SRC}/tasks/downstream/sequence_understanding.py"
        --model_name             "${MODEL_NAME}"
        --dataset_name           "${DATASET_NAME}"
        --output_dir             "${OUT_DIR}"
        --hf_config_path         "${HF_CONFIG}"
        --sw_index               "${SW_INDEX}"
        --down_proj_pattern      "${DOWN_PROJ_PATTERN}"
        --num_transformer_layers "${NUM_LAYERS}"
        --n_rand_controls        10
        --ablation_out           "${ABLATION_OUT}"
    )
    [ -n "${SUBSET_NAME}" ] && PYTHON_ARGS+=(--subset_name "${SUBSET_NAME}")
    BEST_MODEL_DIR="${OUT_DIR}/best_model"
    [ -d "${BEST_MODEL_DIR}" ] && PYTHON_ARGS+=(--skip_training)
fi

echo "Running: python3 ${PYTHON_ARGS[*]}"
echo "------------------------------------------------------------"
python3 "${PYTHON_ARGS[@]}"

echo "============================================================"
echo "Done: ${TASK}"
echo "Log saved to: ${LOG_FILE}"
date
