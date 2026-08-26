#!/bin/bash
# Restartable, structural-only queue.  One model at a time on the allocated GPU;
# each completed model has an atomic raw JSON checkpoint.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$HERE/env_cached.sh"
# Checkpoints are cached locally, but the frozen WikiText detector input may need
# its already-pinned dataset snapshot resolved by `datasets`; do not force an
# offline failure before that snapshot is available.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
OUT="$ROOT/results/E13_multicandidate_structural"
LOG="$OUT/logs"
mkdir -p "$LOG" "$OUT/raw"
STATUS="$OUT/status.csv"
if [ ! -e "$STATUS" ]; then echo 'timestamp,model,state,gpu,detail' > "$STATUS"; fi
MODELS=(llama mistral olmo qwen25-7b mosaicbert modernbert-base ntv3 dnabert2 generator-euk-3b genomeocean-4b qwen25-0.5b qwen25-1.5b qwen25-3b smollm2-135m smollm2-360m smollm2-1.7b generator-prok-1.2b generator-prok-3b eurobert-210m eurobert-610m eurobert-2.1b modernbert-large)
for model in "${MODELS[@]}"; do
  if [ -s "$OUT/raw/$model.json" ]; then
    echo "$(date --iso-8601=seconds),$model,completed,0,checkpoint_exists" >> "$STATUS"; continue
  fi
  echo "$(date --iso-8601=seconds),$model,running,0,started" >> "$STATUS"
  if python -u "$HERE/run_rowwise_detector.py" --model "$model" > "$LOG/$model.log" 2>&1; then
    echo "$(date --iso-8601=seconds),$model,completed,0,ok" >> "$STATUS"
  else
    rc=$?; echo "$(date --iso-8601=seconds),$model,failed,0,exit_$rc" >> "$STATUS"
  fi
done
echo "$(date --iso-8601=seconds),queue,completed,0,all_models_attempted" >> "$STATUS"
