#!/bin/bash
# Run multimodule detection for Evo2 interactively on a GPU node.
# Usage:  bash scripts/run_multimodule_detection_evo2.sh

set -euo pipefail

set +u
module load tacc-apptainer 2>/dev/null || true
set -u

SIF="/work/11034/atzanakak/ls6/containers/evo2.sif"
REPO="/work/11034/atzanakak/glm_super_weight/genomic-super-weights"
OUT="${REPO}/results/multimodule_detection.json"

export HF_HOME="/work/11034/atzanakak/ls6/huggingface/.hf-cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="/work/11034/atzanakak/ls6/huggingface/transformers"

echo "HOST=$(hostname)"
date
nvidia-smi || true

echo "── Dry run ──────────────────────────────────────────────────────────────"
PYTHONPATH="" apptainer exec --nv --cleanenv \
  --bind /work,/tmp \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="${REPO}/genomic-super-weights/stubs:${REPO}/genomic-super-weights" \
  --env PYTHONNOUSERSITE=1 \
  "${SIF}" \
  python3 "${REPO}/genomic-super-weights/scripts/run_multimodule_detection.py" \
    --model evo2 --list_patterns

echo "── Full scan ────────────────────────────────────────────────────────────"
PYTHONPATH="" apptainer exec --nv --cleanenv \
  --bind /work,/tmp \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="${REPO}/genomic-super-weights/stubs:${REPO}/genomic-super-weights" \
  --env PYTHONNOUSERSITE=1 \
  "${SIF}" \
  python3 "${REPO}/genomic-super-weights/scripts/run_multimodule_detection.py" \
    --model  evo2 \
    --probe  actb_full \
    --n_rand 10 \
    --out    "${OUT}"

echo "Done."
date
