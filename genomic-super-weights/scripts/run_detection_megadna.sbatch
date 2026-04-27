#!/bin/bash
#SBATCH -J superweight-detection-megadna
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/detection-megadna.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/detection-megadna.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A BCS25073

set -euo pipefail

# megaDNA uses Python 3.9 + a separate venv.
# Bypass venv activation (interpreter path may be broken) and set paths directly.
PYTHON=/opt/apps/intel19/python3/3.9.7/bin/python3.9
VENV_SITE=/work/11034/atzanakak/ls6/venvs/megadna/lib/python3.9/site-packages
MEGADNA_SRC=/work/11034/atzanakak/ls6/megaDNA
export PYTHONNOUSERSITE=1
export PYTHONPATH=${MEGADNA_SRC}:${VENV_SITE}:${PYTHONPATH:-}

echo "HOST=$(hostname)"
date
nvidia-smi || true

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd $REPO
export PYTHONPATH=$REPO:$PYTHONPATH

$PYTHON scripts/run_detection.py \
  --model megadna \
  --probe phage \
  --mode superrow \
  --threshold 0.1 \
  --max_iter 10 \
  --out results/super_weight_index.json

echo "Done."
date
