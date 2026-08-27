#!/bin/bash
# Author instruction: extend 4b to the full 22-model cohort. Runs the 12 small E11-panel
# models sequentially, cheapest (by non_embed_params) first, matching this audit's
# established staging convention. Already-run 10 models are skipped by the harness's own
# checkpoint check (outpath.exists()), so no --force needed here.
set -uo pipefail
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights
source paper-salvage/experiments/E13_full_cohort_causal_census/env_cached.sh
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/work/11034/atzanakak/ls6/huggingface/.hf-cache/datasets
export HF_DATASETS_OFFLINE=1
cd paper-salvage/experiments/E13_full_cohort_causal_census

for m in smollm2-135m eurobert-210m smollm2-360m modernbert-large qwen25-0.5b eurobert-610m generator-prok-1.2b qwen25-1.5b smollm2-1.7b eurobert-2.1b qwen25-3b generator-prok-3b; do
  echo "[queue] starting $m at $(date)"
  python3 run_singleton_census_4b.py --model "$m" \
    > /work/11034/atzanakak/glm_super_weight/genomic-super-weights/audit/round2/raw/4b_${m}.log 2>&1
  echo "[queue] finished $m at $(date), exit=$?"
done
echo "[queue] ALL 12 REMAINING 4B MODELS COMPLETE -- FULL 22-MODEL 4B SWEEP DONE"
