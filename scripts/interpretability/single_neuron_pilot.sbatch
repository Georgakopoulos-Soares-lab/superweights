#!/bin/bash
#SBATCH -J single-neuron-pilot-dnabert2
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/single-neuron-pilot.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/single-neuron-pilot.%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A MCB26038

# Single-neuron pilot for DNABERT-2 splice/reconstructed (Apple-paper-style single-neuron
# causal analysis). Mirrors the env-activation pattern of
# scripts/interpretability/run_broadcast_impulse_dnabert2.sbatch (conda env `grlm`).
#
# If `grlm` fails to import transformers on the assigned node (a PIL/libstdc++ mismatch was
# observed on at least one node in this project — see reports/single_neuron_pilot_dnabert2.md
# for the exact error and fix), fall back to the isolated venv built for this pilot instead:
#   source /work/11034/atzanakak/venvs/dnabert2_pilot/bin/activate
# (torch==2.6.0+cu124, transformers==4.57.6 — pinned to match grlm's validated version).
#
# Usage:
#   sbatch scripts/interpretability/single_neuron_pilot.sbatch
#
# Resumable: every stage below is guarded by "does its output file already exist" and can be
# re-run independently by invoking the corresponding python command directly (see the
# per-script --out / --ckpt arguments). No stage silently substitutes defaults for a missing
# upstream output — a failed stage leaves no output file and the next stage's import/read
# will fail loudly instead of proceeding on stale or absent data.

set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub
# $HOME is quota-limited on this cluster (10GB) — keep Triton/pip caches off it.
export TRITON_CACHE_DIR=/work/11034/atzanakak/.triton_cache
export PIP_CACHE_DIR=/work/11034/atzanakak/.cache/pip

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

echo "HOST=$(hostname)"
date
nvidia-smi || true

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
GUE_ROOT=/work/11034/atzanakak/GUE/GUE
cd "$REPO"
mkdir -p logs results
export PYTHONPATH=$REPO:${PYTHONPATH:-}

# ── 0. Tests (fast, no GPU needed — fail fast before spending GPU time) ─────
python -m pytest tests/test_neuron_hooks.py tests/test_neuron_intervention.py -v

# ── 1. Checkpoints (existing, unmodified run_gue_ablation.py) ───────────────
SPLICE_CKPT="results/gue_checkpoints/dnabert2_reconstructed/model_state.pt"
if [ ! -f "$SPLICE_CKPT" ]; then
    python scripts/evaluation/run_gue_ablation.py --model dnabert2 \
        --task splice/reconstructed --gue_root "$GUE_ROOT"
else
    echo "[skip] splice checkpoint already exists: $SPLICE_CKPT"
fi

H3K_CKPT="results/gue_checkpoints/dnabert2_H3K4me3/model_state.pt"
if [ ! -f "$H3K_CKPT" ]; then
    python scripts/evaluation/run_gue_ablation.py --model dnabert2 \
        --task EMP/H3K4me3 --gue_root "$GUE_ROOT"
else
    echo "[skip] H3K4me3 checkpoint already exists: $H3K_CKPT"
fi

# ── 2. Discovery ──────────────────────────────────────────────────────────
python scripts/interpretability/run_neuron_discovery_dnabert2.py \
    --gue_root "$GUE_ROOT" --n_pos 300 --n_neg 300 --seed 42

# ── 3. Causal intervention + rerank (also reproduces existing ensemble/row603 numbers) ──
python scripts/interpretability/run_neuron_causal_intervention_dnabert2.py \
    --gue_root "$GUE_ROOT" --top_k 25 --sweep_top_n 3

# ── 4. Structural mapping (row 603 / C[k,i] decomposition) ─────────────────
python scripts/interpretability/run_neuron_structural_mapping_dnabert2.py \
    --gue_root "$GUE_ROOT" --top_n 5

# ── 5. Controls (incl. H3K4me3 task-specificity) ─────────────────────────
python scripts/interpretability/run_neuron_controls_dnabert2.py \
    --gue_root "$GUE_ROOT" --n_rand 10

# ── 6. Base vs fine-tuned ────────────────────────────────────────────────
python scripts/interpretability/run_neuron_base_vs_finetuned_dnabert2.py \
    --gue_root "$GUE_ROOT"

# ── 7. Neuron-originated broadcast trace ────────────────────────────────
python scripts/interpretability/run_neuron_broadcast_impulse_dnabert2.py

echo ""
echo "Done: $(date)"
echo "Outputs: results/neuron_pilot_dnabert2_*.{json,csv}"
