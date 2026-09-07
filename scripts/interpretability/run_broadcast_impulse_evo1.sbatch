#!/bin/bash
#SBATCH -J sw-broadcast-evo1
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sw-broadcast-evo1.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sw-broadcast-evo1.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A MCB26038

# NOTE: Evo1 requires the `evo` conda environment (not `grlm`).
#       Verify that `conda activate evo` gives you `from evo import Evo`.

set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate evo

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/evo
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

echo "HOST=$(hostname)"
date
nvidia-smi || true

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd $REPO
mkdir -p logs results
export PYTHONPATH=$REPO:$PYTHONPATH

python scripts/interpretability/run_sw_broadcast_impulse.py \
    --model      evo1 \
    --epsilon    1.0 \
    --n_controls 5 \
    --seed       42 \
    --device     cuda \
    --out_dir    results

echo "Done."
date
