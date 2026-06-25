#!/bin/bash
#SBATCH -J generator-lifecycle
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/generator-lifecycle.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/generator-lifecycle.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A BCS25073

# Generate GENERator activation lifecycle data for Figure 1 panel C.
#
# Outputs (required by plot_figure1.py):
#   results/activation_lifecycle_generator.json           (EUK, n_seqs=20)
#   results/activation_lifecycle_generator_prokaryote.json (PROK, n_seqs=20)
#
# Usage:
#   sbatch scripts/analysis/submit_generator_lifecycle.sbatch

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
echo "Step 1 — EUK lifecycle (GENERator 3B, 20 seqs)"
echo "========================================"
python scripts/analysis/run_activation_lifecycle.py --model generator --n_seqs 20

echo ""
echo "========================================"
echo "Step 2 — PROK lifecycle (GENERator 3B, 20 seqs)"
echo "========================================"
python scripts/analysis/run_activation_lifecycle.py --model generator_prokaryote --n_seqs 20

echo ""
echo "GENERator lifecycle done."
date
