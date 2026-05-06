#!/bin/bash
# Run multimodule detection for MegaDNA interactively on a GPU node.
# Usage:  bash scripts/run_multimodule_detection_megadna.sh

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

echo "── Dry run ──────────────────────────────────────────────────────────────"
$PYTHON "${REPO}/genomic-super-weights/scripts/run_multimodule_detection.py" \
  --model megadna --list_patterns

echo "── Full scan ────────────────────────────────────────────────────────────"
$PYTHON "${REPO}/genomic-super-weights/scripts/run_multimodule_detection.py" \
  --model  megadna \
  --probe  phage \
  --n_rand 10 \
  --out    "${OUT}"

echo "Done."
date
