#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_multimodule_detection_evo2.sbatch
#
# Runs run_multimodule_detection.py for Evo2 inside the apptainer container.
# Scans: down_proj (l3), mlp_l1, mlp_l2, attn_out, attn_qkv
#         (attn patterns restricted to layers [3,10,17,24,31])
#
# Submit:  sbatch scripts/run_multimodule_detection_evo2.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J multimodule-evo2
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/multimodule-evo2.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/multimodule-evo2.%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A BCS25073

set -euo pipefail

set +u
module load tacc-apptainer
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

# Dry-run first
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
    --model evo2 \
    --list_patterns

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
