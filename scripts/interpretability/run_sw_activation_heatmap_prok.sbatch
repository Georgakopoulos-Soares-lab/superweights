#!/bin/bash
#SBATCH -J sw-heatmap-prok
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sw-heatmap-prok.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sw-heatmap-prok.%j.err
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

python scripts/interpretability/run_sw_activation_heatmap.py \
    --model     generator_prokaryote \
    --fasta     data/reference/ecoli/ecoli_k12.fna \
    --contexts  promoters_ecoli.bed:promoter \
                terminators_ecoli.bed:terminator \
                random_ecoli.bed:random \
    --bed_dir   data/regions/ecoli \
    --sw_layer  2 \
    --sw_rows   1927 \
    --n_ctrl    10 \
    --n_seqs    30 \
    --window_bp 3072 \
    --device    cuda \
    --out   results/sw_activation_heatmap_generator_prokaryote.json \
    --plot  results/sw_activation_heatmap_generator_prokaryote.png

echo "Done."
date
