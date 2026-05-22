#!/bin/bash
#SBATCH -J sae-collect-euk
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-collect-euk.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-collect-euk.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A BCS25073

# ──────────────────────────────────────────────────────────────────────────────
# Collect GENERator-EUK residual-stream activations for SAE training.
#
# Hooks layer 4 (post-residual) and writes float16 shards to:
#   data/sae_acts/generator_euk_layer4/
#
# Usage:
#   sbatch scripts/sae/run_collect_euk.sbatch
#   MAX_TOKENS=100000000 sbatch --export=ALL,MAX_TOKENS=100000000 scripts/sae/run_collect_euk.sbatch
# ──────────────────────────────────────────────────────────────────────────────

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

HG38_FASTA=/scratch/11034/atzanakak/grlm/data/reference/hg38/hg38.fa
BED="${REPO}/data/regions/hg38/random_262kb.bed"
OUT_DIR=/scratch/11034/atzanakak/sae_acts/generator_euk_layer4
MAX_TOKENS="${MAX_TOKENS:-50000000}"
SHARD_TOKENS="${SHARD_TOKENS:-500000}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"      # 512 tokens = 3 072 bp
BATCH_SEQS="${BATCH_SEQS:-32}"

mkdir -p logs

echo "HOST=$(hostname)"
echo "MAX_TOKENS=${MAX_TOKENS}"
date
nvidia-smi || true

python sae/collect.py \
    --model       generator \
    --layer       4 \
    --fasta       "${HG38_FASTA}" \
    --bed         "${BED}" \
    --out_dir     "${OUT_DIR}" \
    --max_tokens  "${MAX_TOKENS}" \
    --shard_tokens "${SHARD_TOKENS}" \
    --chunk_tokens "${CHUNK_TOKENS}" \
    --batch_seqs  "${BATCH_SEQS}"

echo "Done collecting EUK activations."
date
