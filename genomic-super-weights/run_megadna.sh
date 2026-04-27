#!/bin/bash
# run_megadna.sh — run scripts with the megaDNA Python 3.9 environment.
# The megadna venv's interpreter path (/usr/bin/python3.9) may be broken on
# some nodes, so we bypass venv activation and call python3.9 directly with
# the venv's site-packages and megaDNA source injected via PYTHONPATH.
set -euo pipefail

PYTHON=/opt/apps/intel19/python3/3.9.7/bin/python3.9
VENV_SITE=/work/11034/atzanakak/ls6/venvs/megadna/lib/python3.9/site-packages
MEGADNA_SRC=/work/11034/atzanakak/ls6/megaDNA

export PYTHONNOUSERSITE=1
export PYTHONPATH=${MEGADNA_SRC}:${VENV_SITE}:${PYTHONPATH:-}

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd $REPO
export PYTHONPATH=$REPO:$PYTHONPATH

SCRIPT_NAME=$(basename "${1:-run}" .py)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_OUT="logs/${SCRIPT_NAME}.${TIMESTAMP}.out"
LOG_ERR="logs/${SCRIPT_NAME}.${TIMESTAMP}.err"

echo "Logging stdout → $LOG_OUT"
echo "Logging stderr → $LOG_ERR"

$PYTHON "$@" > >(tee "$LOG_OUT") 2> >(tee "$LOG_ERR" >&2)
