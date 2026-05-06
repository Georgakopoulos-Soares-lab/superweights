#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_multimodule_detection_megadna.sbatch
#
# Runs run_multimodule_detection.py for MegaDNA using Python 3.9 + megadna venv.
# Scans: down_proj (FF[4]), mlp_up (FF[1]), attn_out (to_out.0)
#
# Submit:  sbatch scripts/run_multimodule_detection_megadna.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J multimodule-megadna
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/multimodule-megadna.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/multimodule-megadna.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A BCS25073

set -euo pipefail

PYTHON=/opt/apps/intel19/python3/3.9.7/bin/python3.9
VENV_SITE=/work/11034/atzanakak/ls6/venvs/megadna/lib/python3.9/site-packages
MEGADNA_SRC=/work/11034/atzanakak/ls6/megaDNA
export PYTHONNOUSERSITE=1

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
export PYTHONPATH=${MEGADNA_SRC}:${VENV_SITE}:${REPO}/genomic-super-weights/stubs:${REPO}/genomic-super-weights

OUT="${REPO}/results/multimodule_detection.json"

echo "HOST=$(hostname)"
date
nvidia-smi || true

# Dry-run first
echo "── Dry run ──────────────────────────────────────────────────────────────"
$PYTHON "${REPO}/genomic-super-weights/scripts/run_multimodule_detection.py" \
  --model megadna \
  --list_patterns

echo "── Full scan ────────────────────────────────────────────────────────────"
$PYTHON "${REPO}/genomic-super-weights/scripts/run_multimodule_detection.py" \
  --model  megadna \
  --probe  phage \
  --n_rand 10 \
  --out    "${OUT}"

echo "Done."
date
