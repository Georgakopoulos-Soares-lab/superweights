#!/bin/bash
#SBATCH -J superweight-ablation-genomeocean
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/ablation-genomeocean.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/ablation-genomeocean.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A BCS25073

set -euo pipefail

set +u
module load tacc-apptainer
set -u

SIF="/work/11034/atzanakak/genomeocean/apptainer/genomeocean.sif"
REPO="/work/11034/atzanakak/glm_super_weight/genomic-super-weights"

export HF_HOME="/work/11034/atzanakak/ls6/huggingface/.hf-cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="/work/11034/atzanakak/ls6/huggingface/transformers"

echo "HOST=$(hostname)"
date
nvidia-smi || true

PYTHONPATH="" apptainer exec --nv --cleanenv \
  --bind /work,/tmp \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="${REPO}" \
  --env PYTHONNOUSERSITE=1 \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_USE_V1=0 \
  "${SIF}" \
  python3 "${REPO}/scripts/run_ablation.py" \
    --model genomeocean \
    --probe actb_full \
    --sw_index "${REPO}/results/super_weight_index.json" \
    --out "${REPO}/results/ablation_results.json"

echo "Done."
date
