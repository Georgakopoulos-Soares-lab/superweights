#!/bin/bash
# scripts/analysis/submit_dnabert2_uk_audit.sh
#
# Tier-1 follow-up #2: DNABERT-2 mechanistic ||U_k||_F audit.
#
# Adapts the GENERator quadratic-amplifier audit to DNABERT-2's GLU MLP
# (BertGatedLinearUnitMLP). Tests whether the 10 detected SW rows
# concentrated at layers {3, 5, 6, 7, 9} around row 603 are also the
# top-ranking output coordinates in the weight-space amplifier metric.
#
# Predicted outcome
#   If the 10 SW rows top-rank in their respective layers' ||U_k||_F
#   distribution, the DNABERT-2 ensemble has the same quadratic-amplifier
#   mechanistic origin as GENERator (just distributed across layers).
#   If not, the ensemble works through a genuinely different mechanism
#   and the paper claim of "structurally identical phenotype" needs revision.
#
# Outputs
#   results/sw_mechanistic_dnabert2.json
#
# Estimated wall-clock: ~5 minutes on CPU. No GPU required.

set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-/home/nvidia/miniconda3/envs/generator/bin/python}"
mkdir -p logs

"$PYTHON" -u scripts/analysis/run_dnabert2_uk_audit.py \
  --model    dnabert2 \
  --sw_index results/super_weight_index.json \
  --out      results/sw_mechanistic_dnabert2.json \
  --config   configs/dnabert2.yaml \
  2>&1 | tee logs/dnabert2_uk_audit.log
