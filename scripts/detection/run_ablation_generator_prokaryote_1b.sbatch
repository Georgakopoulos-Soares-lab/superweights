#!/bin/bash
#SBATCH -J superweight-ablation-generator-prokaryote-1b
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/ablation-generator-prokaryote-1b.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/ablation-generator-prokaryote-1b.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A BCS25073

set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

echo "HOST=$(hostname)"
date
nvidia-smi || true

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd $REPO
export PYTHONPATH=$REPO:$PYTHONPATH

python scripts/run_ablation.py \
  --model generator_prokaryote_1b \
  --probe pseudomonadota \
  --sw_index results/super_weight_index.json \
  --out results/ablation_results.json

echo "Done."
date
