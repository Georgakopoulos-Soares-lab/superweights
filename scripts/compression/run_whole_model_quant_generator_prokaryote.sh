#!/bin/bash
set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd $REPO
export PYTHONPATH=$REPO:$PYTHONPATH

mkdir -p logs results

echo "=== GENERator prokaryote: SW-exemption quantization comparison (INT4) ==="
date

python scripts/compression/run_whole_model_quantization.py \
    --model      generator_prokaryote \
    --scopes     full \
    --criteria   \
    --fracs      100 \
    --bits       4 \
    --n_rand_seeds 0 \
    --sw_index   results/super_weight_index.json \
    --out        results/whole_model_quant_generator_prokaryote_sw_comparison.json \
    --plot       results/whole_model_quant_generator_prokaryote_sw_comparison.png \
    --verify_quant

echo "Done."
date

echo "Done."
date
