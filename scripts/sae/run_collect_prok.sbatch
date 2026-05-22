#!/bin/bash
#SBATCH -J sae-collect-prok
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-collect-prok.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-collect-prok.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A BCS25073

# ──────────────────────────────────────────────────────────────────────────────
# Collect GENERator-PROK residual-stream activations for SAE training.
#
# Hooks layer 2 (post-residual) and writes float16 shards to:
#   data/sae_acts/generator_prok_layer2/
#
# Because E. coli K-12 is only ~770 K tokens per pass, we repeat 65 passes
# to accumulate ~50 M tokens (matching the EUK budget).
#
# Usage:
#   sbatch scripts/sae/run_collect_prok.sbatch
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

ECOLI_FASTA="${REPO}/data/reference/ecoli/ecoli_k12.fna"
OUT_DIR=/scratch/11034/atzanakak/sae_acts/generator_prok_layer2
MAX_TOKENS="${MAX_TOKENS:-50000000}"
SHARD_TOKENS="${SHARD_TOKENS:-500000}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
BATCH_SEQS="${BATCH_SEQS:-32}"
N_PASSES="${N_PASSES:-65}"

mkdir -p logs

echo "HOST=$(hostname)"
echo "MAX_TOKENS=${MAX_TOKENS}  N_PASSES=${N_PASSES}"
date
nvidia-smi || true

python sae/collect.py \
    --model        generator_prokaryote \
    --layer        2 \
    --fasta        "${ECOLI_FASTA}" \
    --out_dir      "${OUT_DIR}" \
    --max_tokens   "${MAX_TOKENS}" \
    --shard_tokens "${SHARD_TOKENS}" \
    --chunk_tokens "${CHUNK_TOKENS}" \
    --batch_seqs   "${BATCH_SEQS}" \
    --n_passes     "${N_PASSES}"

echo "Done collecting PROK activations."
date
