#!/bin/bash
#SBATCH -J esm2-ablation
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/esm2-ablation.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/esm2-ablation.%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -t 04:00:00
#SBATCH -p h100
#SBATCH -A BCS25105
#
# Ablation gate test for ESM-2 15B candidate super-weight (L1 row 3245).
#
# Zeroes SW row and N_CONTROLS=20 randomly sampled rows at the same layer,
# measures MLM loss change for each.  Reports SW delta as ratio and z-score
# over the control distribution.
#
# Decision rule:  SW/control-mean > 5x  →  proceed to lifecycle/shuffle
#                 SW/control-mean < 5x  →  no functional super-weight in ESM-2
#
# Output:
#   results/esm2_ablation.json
#
# Usage (from Stampede3 login node):
#   cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights
#   sbatch scripts/analysis/submit_esm2_ablation.sbatch

set -euo pipefail

# ── environment ───────────────────────────────────────────────────────────────
set +u
source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/11034/atzanakak/envs/esm_env
set -u

export HF_HOME="${SCRATCH}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

ESM_PYTHON=/scratch/11034/atzanakak/envs/esm_env/bin/python3

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

mkdir -p logs results

echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo ""

echo "========================================"
echo "ESM-2 15B ablation gate test"
echo "SW: L1 row 3245 (frob=5.42, 17.1x)"
echo "Controls: 20 randomly sampled rows at L1"
echo "n_seqs=200, mask_frac=0.15"
echo "========================================"

"${ESM_PYTHON}" scripts/analysis/run_esm2_ablation.py \
    --flip_dir  "${SCRATCH}/flip_data"   \
    --hf_home   "${SCRATCH}/hf_cache"    \
    --n_seqs    200                       \
    --n_controls 20                       \
    --mask_frac 0.15                      \
    --seed      42                        \
    --out_json  results/esm2_ablation.json

echo ""
echo "Done."
date
