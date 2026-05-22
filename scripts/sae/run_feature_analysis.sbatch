#!/bin/bash
#SBATCH -J sae-analyze
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-analyze.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/sae-analyze.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A BCS25073

# ──────────────────────────────────────────────────────────────────────────────
# SAE Feature Analysis
#
# For each trained SAE (EUK + PROK):
#   1. Collect feature activations on all 4096 hexamers (11-token poly-A context)
#   2. Rank features by monosemanticity (activation entropy)
#   3. Correlate with GC content / homopolymer flag / base entropy
#   4. Cross-reference top monosemantic features vs top SW-dependent hexamers
#   5. Produce feature × motif-class heatmap
#
# Dependencies:
#   results/sae/generator_euk_layer4/sae_final.pt  (from run_train_sae.sbatch)
#   results/sae/generator_prok_layer2/sae_final.pt
#   results/sw_hexamer_causal.json                  (existing hexamer KL data)
#
# Usage:
#   sbatch scripts/sae/run_feature_analysis.sbatch
#   TOP_MONO=30 sbatch --export=ALL,TOP_MONO=30 scripts/sae/run_feature_analysis.sbatch
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

TOP_MONO="${TOP_MONO:-20}"
TOP_SW_DEP="${TOP_SW_DEP:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"

mkdir -p logs

echo "HOST=$(hostname)"
echo "top_mono=${TOP_MONO}  top_sw_dep=${TOP_SW_DEP}"
date
nvidia-smi || true

# ── EUK Feature Analysis ───────────────────────────────────────────────────────
echo "========================================"
echo "Analysing EUK SAE features (layer 4)"
echo "========================================"
python sae/analyze.py \
    --model       generator \
    --layer       4 \
    --sae_ckpt    "${REPO}/results/sae/generator_euk_layer4/sae_final.pt" \
    --hexamer_src "${REPO}/results/sw_hexamer_causal.json" \
    --out_dir     "${REPO}/results/sae/generator_euk_layer4" \
    --top_mono    "${TOP_MONO}" \
    --top_sw_dep  "${TOP_SW_DEP}" \
    --batch_size  "${BATCH_SIZE}"

echo "EUK analysis done."
date

# ── PROK Feature Analysis ──────────────────────────────────────────────────────
# Note: requires a PROK-specific hexamer causal JSON if available;
#       falls back to EUK hexamer_src if the PROK one does not exist yet.
PROK_HEXAMER="${REPO}/results/sw_hexamer_causal_generator_prok.json"
if [ ! -f "${PROK_HEXAMER}" ]; then
    echo "WARNING: PROK hexamer causal JSON not found, falling back to EUK."
    PROK_HEXAMER="${REPO}/results/sw_hexamer_causal.json"
fi

echo "========================================"
echo "Analysing PROK SAE features (layer 2)"
echo "========================================"
python sae/analyze.py \
    --model       generator_prokaryote \
    --layer       2 \
    --sae_ckpt    "${REPO}/results/sae/generator_prok_layer2/sae_final.pt" \
    --hexamer_src "${PROK_HEXAMER}" \
    --out_dir     "${REPO}/results/sae/generator_prok_layer2" \
    --top_mono    "${TOP_MONO}" \
    --top_sw_dep  "${TOP_SW_DEP}" \
    --batch_size  "${BATCH_SIZE}"

echo "PROK analysis done."
date
echo "All feature analysis complete."
