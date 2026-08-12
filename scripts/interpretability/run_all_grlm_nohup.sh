#!/bin/bash
# Run all grlm-env interpretability jobs sequentially on CUDA_VISIBLE_DEVICES=0.
# Launch with:
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/interpretability/run_all_grlm_nohup.sh \
#       > logs/nohup_grlm.log 2>&1 &
set -uo pipefail

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "$REPO"

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh
conda activate grlm

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub
export PYTHONPATH=$REPO:${PYTHONPATH:-}

mkdir -p logs results

echo "========================================="
echo " nohup_grlm start: $(date)"
echo "========================================="

# ── Broadcast impulse ─────────────────────────────────────────────────────────
# Re-run generator and prok to restore data lost when dnabert2 overwrote the
# JSON (fixed: script now merges into existing JSON)
echo "[1/8] broadcast: generator  $(date)"
python scripts/interpretability/run_sw_broadcast_impulse.py \
    --model generator --epsilon 1.0 --n_controls 5 --seed 42 \
    --device cuda --out_dir results

echo "[2/8] broadcast: generator_prokaryote  $(date)"
python scripts/interpretability/run_sw_broadcast_impulse.py \
    --model generator_prokaryote --epsilon 1.0 --n_controls 5 --seed 42 \
    --device cuda --out_dir results

echo "[3/8] broadcast: dnabert2  $(date)"
python scripts/interpretability/run_sw_broadcast_impulse.py \
    --model dnabert2 --epsilon 1.0 --n_controls 5 --seed 42 \
    --device cuda --out_dir results

echo "[4/8] broadcast: ntv3  $(date)"
python scripts/interpretability/run_sw_broadcast_impulse.py \
    --model ntv3 --epsilon 1.0 --n_controls 5 --seed 42 \
    --device cuda --out_dir results

# ── Relay heads ───────────────────────────────────────────────────────────────
echo "[5/8] relay heads: generator  $(date)"
python scripts/interpretability/run_sw_relay_heads.py \
    --model generator --top_k 10 --n_rand 10 --seed 42 --out_dir results

echo "[6/8] relay heads: generator_prokaryote  $(date)"
python scripts/interpretability/run_sw_relay_heads.py \
    --model generator_prokaryote --top_k 10 --n_rand 10 --seed 42 --out_dir results

# ── Counterfactual GC/AT swap ─────────────────────────────────────────────────
echo "[7/8] counterfactual swap: generator  $(date)"
python scripts/interpretability/run_sw_counterfactual_swap.py \
    --model generator --n_pairs 20 --gc_frac 0.70 --at_frac 0.30 \
    --n_controls 5 --seed 42 --out_dir results

echo "[8/8] counterfactual swap: generator_prokaryote  $(date)"
python scripts/interpretability/run_sw_counterfactual_swap.py \
    --model generator_prokaryote --n_pairs 20 --gc_frac 0.70 --at_frac 0.30 \
    --n_controls 5 --seed 42 --out_dir results

echo "========================================="
echo " nohup_grlm DONE: $(date)"
echo "========================================="
