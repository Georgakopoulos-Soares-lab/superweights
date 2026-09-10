#!/bin/bash
set -uo pipefail
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights
source experiments/frozen/E13_full_cohort_causal_census/env_cached.sh
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/work/11034/atzanakak/ls6/huggingface/.hf-cache/datasets
export HF_DATASETS_OFFLINE=1
cd experiments/frozen/E13_full_cohort_causal_census

for m in qwen25-7b mosaicbert ntv3; do
  echo "[queue] starting $m at $(date)"
  python3 run_singleton_census_4b.py --model "$m" --force \
    > /work/11034/atzanakak/glm_super_weight/genomic-super-weights/audit/rederivations/raw/4b_${m}.log 2>&1
  echo "[queue] finished $m at $(date), exit=$?"
done
echo "[queue] ALL REMAINING 4B MODELS COMPLETE"
