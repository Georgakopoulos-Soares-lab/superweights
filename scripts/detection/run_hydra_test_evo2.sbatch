#!/bin/bash
#SBATCH -J hydra-test-evo2
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/hydra-evo2.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/hydra-evo2.%j.err
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A BCS25073

# ──────────────────────────────────────────────────────────────────────────────
# Hydra test for Evo2
#
# Tests whether Evo2's robustness to single super-weight ablation is a
# "hydra effect": ablate the top activation outlier, let another rise,
# ablate that one too, repeat until perplexity finally degrades.
#
# Key parameters (override via --export on sbatch or edit below):
#   MAX_ROUNDS              max ablation rounds before declaring hydra-wins
#   DEGRADATION_THRESHOLD   stop (model dead) when delta_pct exceeds this %
#
# Usage examples:
#   sbatch run_hydra_test_evo2.sbatch                         # defaults
#   sbatch --export=ALL,MAX_ROUNDS=20 run_hydra_test_evo2.sbatch
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

set +u
module load tacc-apptainer
set -u

SIF="/work/11034/atzanakak/ls6/containers/evo2.sif"
REPO="/work/11034/atzanakak/glm_super_weight/genomic-super-weights"

export HF_HOME="/work/11034/atzanakak/ls6/huggingface/.hf-cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="/work/11034/atzanakak/ls6/huggingface/transformers"

# Configurable parameters (can be overridden with --export on sbatch)
MAX_ROUNDS="${MAX_ROUNDS:-10}"
DEGRADATION_THRESHOLD="${DEGRADATION_THRESHOLD:-5.0}"
PROBE="${PROBE:-actb_full}"
OUT="${OUT:-${REPO}/results/hydra_test_evo2.json}"

echo "HOST=$(hostname)"
echo "MAX_ROUNDS=${MAX_ROUNDS}"
echo "DEGRADATION_THRESHOLD=${DEGRADATION_THRESHOLD}%"
echo "PROBE=${PROBE}"
echo "OUT=${OUT}"
date
nvidia-smi || true

PYTHONPATH="" apptainer exec --nv --cleanenv \
  --bind /work,/tmp \
  --env HF_HOME="${HF_HOME}" \
  --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
  --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="${REPO}" \
  --env PYTHONNOUSERSITE=1 \
  "${SIF}" \
  python3 "${REPO}/scripts/detection/run_hydra_test_evo2.py" \
    --model evo2 \
    --probe "${PROBE}" \
    --max_rounds "${MAX_ROUNDS}" \
    --degradation_threshold "${DEGRADATION_THRESHOLD}" \
    --out "${OUT}"

echo "Done."
date
