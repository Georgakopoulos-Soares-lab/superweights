#!/bin/bash
set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd $REPO
export PYTHONPATH=$REPO:$PYTHONPATH

SCRIPT_NAME=$(basename "${1:-run}" .py)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_OUT="logs/${SCRIPT_NAME}.${TIMESTAMP}.out"
LOG_ERR="logs/${SCRIPT_NAME}.${TIMESTAMP}.err"

echo "Logging stdout → $LOG_OUT"
echo "Logging stderr → $LOG_ERR"

python "$@" > >(tee "$LOG_OUT") 2> >(tee "$LOG_ERR" >&2)
