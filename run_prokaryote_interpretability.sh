#!/bin/bash
# Interpretability experiments for GENERator prokaryote 3B, SW layer=2 row=1927
# Reference: E. coli K-12 MG1655 (NC_000913.3)
#
# Usage (from repo root):
#   bash run_prokaryote_interpretability.sh

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="/work/11034/atzanakak/evo2_env"
export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export HF_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_MODE=offline

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO="/work/11034/atzanakak/glm_super_weight/genomic-super-weights"
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

FASTA="data/reference/ecoli/ecoli_k12.fna"
PROMOTERS="data/regions/ecoli/promoters_ecoli.bed"
ENHANCERS="data/regions/ecoli/terminators_ecoli.bed"
RANDOM_BED="data/regions/ecoli/random_ecoli.bed"
SW_INDEX="results/super_weight_index.json"
MODEL="generator_prokaryote"
RESULTS="results/prokaryote"
mkdir -p "${RESULTS}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="${REPO}/logs/prokaryote_interpretability_${TIMESTAMP}.out"
mkdir -p "${REPO}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================================"
echo "HOST=$(hostname)  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo N/A)"
echo "MODEL: ${MODEL}  SW: layer=2 row=1927"
date
echo "LOG: ${LOG}"
echo "============================================================"

# ── Step 1: k-mer scan ────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 1: k-mer scan"
python3 scripts/interpretability/run_sw_kmer_scan.py \
    --model      "${MODEL}" \
    --sw_index   "${SW_INDEX}" \
    --configs_dir configs \
    --top_n      50 \
    --out        "${RESULTS}/sw_kmer_scan.json" \
    --plot       "${RESULTS}/sw_kmer_scan.png"

# ── Step 2: token omission ────────────────────────────────────────────────────
echo ""
echo ">>> STEP 2: token omission"
python3 scripts/interpretability/run_sw_token_omission.py \
    --model      "${MODEL}" \
    --sw_index   "${SW_INDEX}" \
    --configs_dir configs \
    --fasta      "${FASTA}" \
    --promoters  "${PROMOTERS}" \
    --enhancers  "${ENHANCERS}" \
    --random     "${RANDOM_BED}" \
    --n_seqs     10 \
    --window_bp  384 \
    --out        "${RESULTS}/sw_token_omission.json" \
    --plot       "${RESULTS}/sw_token_omission.png"

# ── Step 3: shuffle controls ──────────────────────────────────────────────────
echo ""
echo ">>> STEP 3: shuffle controls"
python3 scripts/interpretability/run_sw_shuffle_controls.py \
    --model      "${MODEL}" \
    --sw_index   "${SW_INDEX}" \
    --configs_dir configs \
    --fasta      "${FASTA}" \
    --promoters  "${PROMOTERS}" \
    --enhancers  "${ENHANCERS}" \
    --random     "${RANDOM_BED}" \
    --n_seqs     30 \
    --window_bp  3072 \
    --out        "${RESULTS}/sw_shuffle_controls.json"

# ── Step 4: k-mer motif enrichment ───────────────────────────────────────────
echo ""
echo ">>> STEP 4: k-mer motif enrichment"
python3 scripts/interpretability/analyze_sw_kmer_motifs.py \
    --kmer_scan  "${RESULTS}/sw_kmer_scan.json" \
    --top_k      200 \
    --out        "${RESULTS}/sw_kmer_motifs.json" \
    --plot       "${RESULTS}/sw_kmer_motifs.png"

# ── Step 5: gradient attribution ─────────────────────────────────────────────
# Restrict to single GPU: backward() cannot cross device boundaries with device_map="auto"
echo ""
echo ">>> STEP 5: gradient attribution"
CUDA_VISIBLE_DEVICES=0 python3 scripts/interpretability/run_sw_gradient_attribution.py \
    --model      "${MODEL}" \
    --sw_index   "${SW_INDEX}" \
    --configs_dir configs \
    --fasta      "${FASTA}" \
    --promoters  "${PROMOTERS}" \
    --enhancers  "${ENHANCERS}" \
    --random     "${RANDOM_BED}" \
    --n_seqs     50 \
    --window_bp  3072 \
    --out        "${RESULTS}/sw_grad_attribution.json" \
    --plot       "${RESULTS}/sw_grad_attribution.png"

# ── Step 6: compare attribution methods ──────────────────────────────────────
echo ""
echo ">>> STEP 6: compare attribution methods"
python3 scripts/interpretability/compare_attribution_methods.py \
    --grad       "${RESULTS}/sw_grad_attribution.json" \
    --omit       "${RESULTS}/sw_token_omission.json" \
    --match_by   label_order \
    --out        "${RESULTS}/attribution_comparison.json" \
    --plot       "${RESULTS}/attribution_comparison.png"

echo ""
echo "============================================================"
echo "All steps complete. Results in: ${RESULTS}/"
date
