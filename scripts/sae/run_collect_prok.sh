#!/bin/bash
# Collect GENERator-PROK residual-stream activations for SAE training.
# Run on an interactive GPU node (idev session).
#
# E. coli K-12 is ~770 K tokens per pass; 65 passes ≈ 50 M tokens.
# Override defaults via env vars:
#   N_PASSES=100 bash scripts/sae/run_collect_prok.sh

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
ECOLI_FASTA="${ECOLI_FASTA:-${REPO}/data/reference/ecoli/ecoli_k12.fna}"
OUT_DIR="${OUT_DIR:-${REPO}/data/sae_acts/generator_prok_layer2}"
MAX_TOKENS="${MAX_TOKENS:-50000000}"
SHARD_TOKENS="${SHARD_TOKENS:-500000}"
CHUNK_TOKENS="${CHUNK_TOKENS:-512}"
BATCH_SEQS="${BATCH_SEQS:-32}"
N_PASSES="${N_PASSES:-65}"

# ── Logging ───────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sae-collect-prok-${TIMESTAMP}.out"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "SAE activation collection — GENERator PROK (layer 2)"
echo "============================================================"
echo "HOST=$(hostname)"
echo "LOG=${LOG_FILE}"
echo "ECOLI_FASTA=${ECOLI_FASTA}"
echo "OUT_DIR=${OUT_DIR}"
echo "MAX_TOKENS=${MAX_TOKENS}  N_PASSES=${N_PASSES}  SHARD_TOKENS=${SHARD_TOKENS}"
nvidia-smi || true
date
echo "------------------------------------------------------------"

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

echo "------------------------------------------------------------"
echo "Done collecting PROK activations."
date
echo "Log written to: ${LOG_FILE}"
