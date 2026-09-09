#!/bin/bash
# Detached, restartable Stage-1 queue.  A failing model is recorded and the remaining
# cohort continues; rerunning this script skips every atomic JSON already completed.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
source "$HERE/env_cached.sh"
export HF_DATASETS_CACHE=/scratch/11034/atzanakak/huggingface_cache/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
LOG_DIR=/scratch/11034/atzanakak/genomic-super-weights/E13/logs/census
mkdir -p "$LOG_DIR"

# Small checkpoints first expose architecture/endpoint problems cheaply; large checkpoints
# follow.  Scientific panel order remains frozen in candidate_manifest.json and all compiled
# tables, independent of this operational execution order.
MODELS=(
  smollm2-135m mosaicbert modernbert-base eurobert-210m smollm2-360m
  qwen25-0.5b ntv3 dnabert2 eurobert-610m generator-prok-1.2b
  qwen25-1.5b smollm2-1.7b eurobert-2.1b modernbert-large qwen25-3b
  phi3 generator-euk-3b generator-prok-3b genomeocean-4b
  llama mistral olmo qwen25-7b
)

failures=0
for model in "${MODELS[@]}"; do
    # Per-model JSONs are written by os.replace only after every paired condition
    # finishes, so a nonempty final path is an atomic completion checkpoint.  Avoid
    # paying the multi-minute Torch/Transformers import cost merely to discover it.
    if [ -s "$ROOT/results/E13/raw/$model.json" ]; then
        echo "[$(date --iso-8601=seconds)] SKIP checkpointed $model"
        continue
    fi
    echo "[$(date --iso-8601=seconds)] START $model"
    if python -u "$HERE/run_singleton_census.py" --model "$model" \
        >"$LOG_DIR/$model.log" 2>&1; then
        echo "[$(date --iso-8601=seconds)] DONE $model"
    else
        rc=$?
        failures=$((failures + 1))
        echo "[$(date --iso-8601=seconds)] FAILED $model rc=$rc (see $LOG_DIR/$model.log)"
    fi
done

echo "[$(date --iso-8601=seconds)] QUEUE COMPLETE failures=$failures"
if [ "$failures" -ne 0 ]; then
    exit "$failures"
fi

set -e
echo "[$(date --iso-8601=seconds)] START POSTPROCESSING"
python -u "$HERE/compile_census.py"
python -u "$HERE/import_locked_tomography.py"
python -u "$HERE/build_cohort_summary.py"
python -u "$HERE/plot_census.py"
python -u "$HERE/write_final_report.py"
python -u "$HERE/validate_e13.py"
echo "[$(date --iso-8601=seconds)] E13 COMPLETE AND VALIDATED"
