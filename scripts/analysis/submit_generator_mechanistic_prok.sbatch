#!/bin/bash
#SBATCH -J generator-mechanistic-prok
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/generator-mechanistic-prok.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/generator-mechanistic-prok.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A BCS25073

# Generate GENERator-PROK mechanistic data for Fig 2 panels B, D, E.
#
# Outputs:
#   results/sw_mechanistic_generator_prokaryote.json          (panels D, E)
#   results/activation_lifecycle_generator_prokaryote.json    (panel B)
#
# Usage:
#   sbatch scripts/analysis/submit_generator_mechanistic_prok.sbatch

set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_OFFLINE=1

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

mkdir -p logs results

echo "HOST=$(hostname)"
date
nvidia-smi || true

echo "========================================"
echo "Step 1 — ||U_k||_F audit (weights only, CPU)"
echo "========================================"
python scripts/analysis/run_generator_uk_audit.py --variant prok

echo ""
echo "========================================"
echo "Step 2 — Residual attribution (forward pass, GPU)"
echo "========================================"
python scripts/analysis/run_generator_residual_attribution.py --variant prok

echo ""
echo "GENERator-PROK mechanistic analysis done."
date
