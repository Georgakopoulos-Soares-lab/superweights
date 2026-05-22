#!/bin/bash
set -euo pipefail

set +u
module load tacc-apptainer
set -u

SIF="/work/11034/atzanakak/ls6/containers/evo2.sif"
REPO="/work/11034/atzanakak/glm_super_weight/genomic-super-weights"

export HF_HOME="/work/11034/atzanakak/ls6/huggingface/.hf-cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="/work/11034/atzanakak/ls6/huggingface/transformers"

MAX_ROUNDS="${MAX_ROUNDS:-10}"
DEGRADATION_THRESHOLD="${DEGRADATION_THRESHOLD:-5.0}"
PROBE="${PROBE:-actb_full}"
OUT="${OUT:-${REPO}/results/hydra_test_evo2.json}"

echo "HOST=$(hostname)"
echo "MAX_ROUNDS=${MAX_ROUNDS}  DEGRADATION_THRESHOLD=${DEGRADATION_THRESHOLD}%  PROBE=${PROBE}"
nvidia-smi || true
date

PYTHONPATH="" apptainer exec --nv --cleanenv \
  --bind /work,/tmp \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="${REPO}" \
  --env PYTHONNOUSERSITE=1 \
  "${SIF}" \
  python3 "${REPO}/scripts/detection/run_hydra_test_evo2.py" \
    --model evo2 \
    --probe "${PROBE}" \
    --max_rounds "${MAX_ROUNDS}" \
    --degradation_threshold "${DEGRADATION_THRESHOLD}" \
    --out "${OUT}"

echo "Done."
date
