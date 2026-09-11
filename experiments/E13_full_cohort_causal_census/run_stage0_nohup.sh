#!/bin/bash
set -euo pipefail

trap 'rc=$?; echo "[$(date --iso-8601=seconds)] STAGE0 SUPERVISOR EXIT rc=$rc"' EXIT

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$HERE/env_cached.sh"
export HF_DATASETS_CACHE=/scratch/11034/atzanakak/huggingface_cache/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for model in modernbert-base llama mistral olmo phi3 dnabert2; do
    if python "$HERE/stage0_checkpoint_complete.py" "$model"; then
        echo "[$(date --iso-8601=seconds)] SKIP completed $model"
        continue
    fi
    echo "[$(date --iso-8601=seconds)] START $model"
    python -u "$HERE/stage0_reproduction_gate.py" --model "$model"
    echo "[$(date --iso-8601=seconds)] DONE $model"
done

echo "[$(date --iso-8601=seconds)] STAGE0 MEASUREMENTS COMPLETE"
python -u "$HERE/summarize_stage0.py"
echo "[$(date --iso-8601=seconds)] STAGE0 GATE PASSED; STARTING FULL COHORT"
bash "$HERE/run_census_nohup.sh"
