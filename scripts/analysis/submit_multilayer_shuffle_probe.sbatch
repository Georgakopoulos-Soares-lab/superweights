#!/bin/bash
#SBATCH -J multilayer-shuffle-probe
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/multilayer-shuffle-probe.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/multilayer-shuffle-probe.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A BCS25073
#
# Tests whether shuffle sensitivity tracks with layer depth (layer-position
# hypothesis) or training corpus (corpus hypothesis).
#
# Runs 4 shuffle controls × 4 probe layers × 2 models (EUK-3B, PROK-3B)
# + actual SW layer for PROK-1B.
# n=90 real GUE sequences, 5 shuffles per sequence per shuffle type.
#
# Output:
#   results/multilayer_shuffle_probe.csv
#   results/multilayer_shuffle_probe.json
#
# Usage:
#   sbatch scripts/analysis/submit_multilayer_shuffle_probe.sbatch

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

mkdir -p logs results

echo "HOST=$(hostname)"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo ""
echo "========================================"
echo "Step 0 — PROK-1B ||U_k||_F audit (needed to detect SW)"
echo "========================================"

if [ ! -f "${REPO}/results/sw_mechanistic_generator_prokaryote_1b.json" ]; then
    python scripts/analysis/run_generator_uk_audit.py --variant prok1b
else
    echo "PROK-1B mech JSON already exists — skipping."
fi

echo ""
echo "========================================"
echo "Multi-layer shuffle probe (v2 — model-appropriate sequences)"
echo "EUK-3B (L2,L4,L6,L14): prom_300_all + EMP/H3 + splice/reconstructed (in-distribution)"
echo "PROK-3B (L2,L4,L6,L14) + PROK-1B (L1,L2): virus/species_40 + fungi/species_20"
echo "n_seqs=120, n_shuffles=5, 4 shuffle types"
echo "Hypothesis: layer-position vs corpus"
echo "========================================"

python scripts/analysis/run_multilayer_shuffle_probe.py \
    --n_seqs     120 \
    --n_shuffles 5   \
    --seq_len    300 \
    --seed       42  \
    --gue_root   /work/11034/atzanakak/GUE/GUE \
    --out_csv    results/multilayer_shuffle_probe_v2.csv \
    --out_json   results/multilayer_shuffle_probe_v2.json

echo ""
echo "Done."
date
