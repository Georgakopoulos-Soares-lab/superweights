#!/bin/bash
# Runs the remaining 4b models SEQUENTIALLY (waits for Mistral-7B's already-running
# process to exit first, to avoid GPU collision), in priority order: OLMo-7B,
# GenomeOcean-4B, Qwen2.5-7B, MosaicBERT, NTv3. Each writes its own log under
# audit/round2/raw/. Safe to survive a disconnected terminal (nohup+disown).
set -uo pipefail
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights

echo "[queue] waiting for Mistral-7B (PID 3979747) to finish..."
tail --pid=3979747 -f /dev/null 2>/dev/null
echo "[queue] Mistral-7B done, proceeding."

source experiments/frozen/E13_full_cohort_causal_census/env_cached.sh
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/work/11034/atzanakak/ls6/huggingface/.hf-cache/datasets
export HF_DATASETS_OFFLINE=1
cd experiments/frozen/E13_full_cohort_causal_census

for m in olmo genomeocean-4b qwen25-7b mosaicbert ntv3; do
  echo "[queue] starting $m at $(date)"
  python3 run_singleton_census_4b.py --model "$m" --force \
    > /work/11034/atzanakak/glm_super_weight/genomic-super-weights/audit/round2/raw/4b_${m}.log 2>&1
  echo "[queue] finished $m at $(date), exit=$?"
done

echo "[queue] ALL REMAINING 4B MODELS COMPLETE"
