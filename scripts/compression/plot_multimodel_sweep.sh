#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Combine compression-sweep results for all four models into a single
# multi-model comparison figure.
#
# Run this AFTER the individual sweep jobs have completed:
#   sbatch scripts/compression/run_compression_sweep_generator_euk.sbatch
#   sbatch scripts/compression/run_compression_sweep_generator_prok.sbatch
#   sbatch scripts/compression/run_compression_sweep_evo2.sbatch
#
# The DNABERT-2 result already exists:
#   results/compression_sweep_dnabert2_prom_core_notata.json
#
# Usage (interactive, from repo root):
#   bash scripts/compression/plot_multimodel_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

source ~/miniconda3/bin/activate genomic_sw 2>/dev/null \
    || source ~/miniconda3/bin/activate base

TASK="prom_core_notata"

# Build --inputs list from whichever results files exist
INPUTS=()
declare -A PATHS=(
    ["dnabert2"]="results/compression_sweep_dnabert2_${TASK}.json"
    ["generator"]="results/compression_sweep_generator_${TASK}.json"
    ["generator_prokaryote"]="results/compression_sweep_generator_prokaryote_${TASK}.json"
    ["evo2"]="results/compression_sweep_evo2_${TASK}.json"
)

for label in dnabert2 generator generator_prokaryote evo2; do
    path="${PATHS[$label]}"
    if [ -f "$path" ]; then
        INPUTS+=("${label}:${path}")
        echo "  ✔ ${label}: ${path}"
    else
        echo "  ✘ ${label}: ${path} — NOT FOUND (skipping)"
    fi
done

if [ ${#INPUTS[@]} -eq 0 ]; then
    echo "ERROR: no sweep result files found. Run the individual sweep jobs first."
    exit 1
fi

echo ""
echo "── Accuracy overlay ─────────────────────────────────────────────────"
python scripts/analysis/plot_compression_multimodel.py \
    --inputs "${INPUTS[@]}" \
    --metric accuracy \
    --out    "results/compression_multimodel_${TASK}_accuracy.png"

echo ""
echo "── MCC overlay ──────────────────────────────────────────────────────"
python scripts/analysis/plot_compression_multimodel.py \
    --inputs "${INPUTS[@]}" \
    --metric mcc \
    --out    "results/compression_multimodel_${TASK}_mcc.png"

echo ""
echo "Done."
echo "  results/compression_multimodel_${TASK}_accuracy.png"
echo "  results/compression_multimodel_${TASK}_mcc.png"
