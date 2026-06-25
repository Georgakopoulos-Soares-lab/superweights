#!/bin/bash
#SBATCH -J sae-analyze-euk-v5
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-analyze-euk-v5.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-analyze-euk-v5.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A BCS25073

# Feature analysis for EUK SAE v5 (dict_mult=1, k=64, no resampling).
#
# Usage:
#   sbatch scripts/sae/run_analyze_euk_v5.sbatch

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

TOP_MONO="${TOP_MONO:-20}"
TOP_SW_DEP="${TOP_SW_DEP:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"

mkdir -p logs

echo "HOST=$(hostname)"
echo "top_mono=${TOP_MONO}  top_sw_dep=${TOP_SW_DEP}"
date
nvidia-smi || true

echo "========================================"
echo "Analysing EUK SAE v5 features (layer 4)"
echo "========================================"
python sae/analyze.py \
    --model       generator \
    --layer       4 \
    --sae_ckpt    "${REPO}/results/sae/generator_euk_layer4_v5/sae_final.pt" \
    --hexamer_src "${REPO}/results/sw_hexamer_causal.json" \
    --out_dir     "${REPO}/results/sae/generator_euk_layer4_v5" \
    --top_mono    "${TOP_MONO}" \
    --top_sw_dep  "${TOP_SW_DEP}" \
    --batch_size  "${BATCH_SIZE}"

echo "EUK SAE v5 analysis done."
date
