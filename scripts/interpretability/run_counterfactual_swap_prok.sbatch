#!/bin/bash
#SBATCH -J sw-cfswap-prok
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sw-cfswap-prok.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sw-cfswap-prok.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A MCB26038

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
mkdir -p logs results
export PYTHONPATH=$REPO:$PYTHONPATH

python scripts/interpretability/run_sw_counterfactual_swap.py \
    --model      generator_prokaryote \
    --n_pairs    20 \
    --gc_frac    0.70 \
    --at_frac    0.30 \
    --n_controls 5 \
    --seed       42 \
    --out_dir    results

echo "Done."
date
