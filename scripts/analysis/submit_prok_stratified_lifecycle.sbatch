#!/bin/bash
#SBATCH -J prok-stratified-lifecycle
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/prok-stratified-lifecycle.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/prok-stratified-lifecycle.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A BCS25073
#
# Tests the early-gating hypothesis for GENERator-PROK row 1927.
# Runs 200 forward passes (100 GC-rich Q1 + 100 AT-rich Q4 hexamers),
# tracks |H[:,1927]| across all 30 layers, and computes persistence ratios.
#
# Output: results/prok_stratified_lifecycle.json
#
# Usage:
#   sbatch scripts/analysis/submit_prok_stratified_lifecycle.sbatch

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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "========================================"
echo "PROK stratified lifecycle (gate hypothesis)"
echo "SW row 1927, layer 2"
echo "100 Q1 (GC-rich/high|act|) + 100 Q4 (AT-rich/low|act|)"
echo "========================================"

python scripts/analysis/run_prok_stratified_lifecycle.py \
    --n_sample 100 \
    --seq_len  504 \
    --seed     42

echo ""
echo "Done."
date
