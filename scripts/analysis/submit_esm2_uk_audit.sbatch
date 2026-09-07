#!/bin/bash
#SBATCH -J esm2-uk-audit
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/esm2-uk-audit.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/esm2-uk-audit.%j.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -t 04:00:00
#SBATCH -p h100
#SBATCH -A BCS25105
#
# Compute ||U_k||_F for every MLP output row across all 48 layers of ESM-2 15B.
# Runs on CPU (no GPU needed; model loaded in float32).
# ~30–60 min total on a 4-core node.
#
# Output:
#   results/sw_mechanistic_esm2.json
#
# Usage:
#   sbatch scripts/analysis/submit_esm2_uk_audit.sbatch

set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
set +u
source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/11034/atzanakak/envs/esm_env
set -u

HF_HOME="${SCRATCH}/hf_cache"
export HF_HOME
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# ESM conda env python (has transformers + the right torch)
ESM_PYTHON=/scratch/11034/atzanakak/envs/esm_env/bin/python3

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "${REPO}"

mkdir -p logs results

echo "HOST=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
date
echo ""

echo "========================================"
echo "ESM-2 15B  ||U_k||_F audit"
echo "48 layers × 2560 output rows"
echo "CPU-only (float32), no GPU required"
echo "========================================"

"${ESM_PYTHON}" scripts/analysis/run_esm2_uk_audit.py \
    --hf_home "${HF_HOME}" \
    --out      results/sw_mechanistic_esm2.json

echo ""
echo "Done."
date
