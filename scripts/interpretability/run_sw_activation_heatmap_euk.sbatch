#!/bin/bash
#SBATCH -J sw-heatmap-euk
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sw-heatmap-euk.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sw-heatmap-euk.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
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

# Path to indexed hg38 FASTA — update if location differs
HG38_FASTA=/home/nvidia/data/hg38/hg38.fa

python scripts/interpretability/run_sw_activation_heatmap.py \
    --model     generator \
    --fasta     "$HG38_FASTA" \
    --contexts  promoters_262kb.bed:promoter \
                enhancers_ccre_262kb.bed:enhancer \
                random_262kb.bed:random \
    --bed_dir   data/regions/hg38 \
    --sw_layer  4 \
    --sw_rows   2371 1522 \
    --n_ctrl    10 \
    --n_seqs    30 \
    --window_bp 3072 \
    --device    cuda \
    --out   results/sw_activation_heatmap_generator.json \
    --plot  results/sw_activation_heatmap_generator.png

echo "Done."
date
