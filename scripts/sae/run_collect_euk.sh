#!/bin/bash
# Collect GENERator-EUK residual-stream activations for SAE training.
# Run on an interactive GPU node (idev session).
#
# Override defaults via env vars:
#   MAX_TOKENS=100000000 bash scripts/sae/run_collect_euk.sh
#   HG38_FASTA=/path/to/hg38.fa bash scripts/sae/run_collect_euk.sh

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_OFFLINE=1

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

LOG_DIR="${REPO}/logs"
mkdir -p "${LOG_DIR}"

# ── Parameters ────────────────────────────────────────────────────────────────
HG38_FASTA="${HG38_FASTA:-/scratch/11034/atzanakak/grlm/data/reference/hg38/hg38.fa}"
BED="${BED:-${REPO}/data/regions/hg38/random_262kb.bed}"
OUT_DIR="${OUT_DIR:-${REPO}/data/sae_acts/generator_euk_layer4}"
MAX_TOKENS="${MAX_TOKENS:-50000000}"
SHARD_TOKENS="${SHARD_TOKENS:-500000}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
BATCH_SEQS="${BATCH_SEQS:-32}"

# ── Logging ───────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sae-collect-euk-${TIMESTAMP}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "SAE activation collection — GENERator EUK (layer 4)"
echo "============================================================"
echo "HOST=$(hostname)"
echo "LOG=${LOG_FILE}"
echo "HG38_FASTA=${HG38_FASTA}"
echo "BED=${BED}"
echo "OUT_DIR=${OUT_DIR}"
echo "MAX_TOKENS=${MAX_TOKENS}  SHARD_TOKENS=${SHARD_TOKENS}  CHUNK_TOKENS=${CHUNK_TOKENS}"
nvidia-smi || true
date
echo "------------------------------------------------------------"

python sae/collect.py \
    --model        generator \
    --layer        4 \
    --fasta        "${HG38_FASTA}" \
    --bed          "${BED}" \
    --out_dir      "${OUT_DIR}" \
    --max_tokens   "${MAX_TOKENS}" \
    --shard_tokens "${SHARD_TOKENS}" \
    --chunk_tokens "${CHUNK_TOKENS}" \
    --batch_seqs   "${BATCH_SEQS}"

echo "------------------------------------------------------------"
echo "Done collecting EUK activations."
date
echo "Log written to: ${LOG_FILE}"
