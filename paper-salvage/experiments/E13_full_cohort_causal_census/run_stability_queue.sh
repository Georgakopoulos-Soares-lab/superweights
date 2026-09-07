#!/bin/bash
# Priority-2 multi-input stability sweep, all 22 non-Phi-3 panel models, one GPU,
# sequential (never split the single A100 across two jobs). Failures are logged and
# do not stop the queue. Restartable: run_candidate_input_stability.py skips any
# --model whose output already exists unless --force is passed.
set -uo pipefail

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "$REPO/paper-salvage/experiments/E13_full_cohort_causal_census"

source "$REPO/paper-salvage/experiments/E13_full_cohort_causal_census/env_cached.sh"
export HF_HOME=/scratch/11034/atzanakak/huggingface_cache
export HF_HUB_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export E13_STABILITY_N_INPUTS=24

LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/stability_queue_master.log"
STAMP() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

MODELS=(mosaicbert modernbert-base ntv3 dnabert2 generator-euk-3b genomeocean-4b \
        qwen25-0.5b qwen25-1.5b smollm2-135m smollm2-360m smollm2-1.7b \
        generator-prok-1.2b generator-prok-3b eurobert-210m eurobert-610m \
        eurobert-2.1b modernbert-large qwen25-3b llama mistral olmo qwen25-7b)

echo "[$(STAMP)] QUEUE START" | tee -a "$MASTER"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>&1 | tee -a "$MASTER"

for m in "${MODELS[@]}"; do
    logfile="$LOGDIR/stability_${m}.log"
    echo "[$(STAMP)] START $m" | tee -a "$MASTER"
    if python3 -u run_candidate_input_stability.py --model "$m" --force > "$logfile" 2>&1; then
        echo "[$(STAMP)] OK    $m -> $logfile" | tee -a "$MASTER"
    else
        echo "[$(STAMP)] FAIL  $m (exit $?) -> $logfile -- continuing queue" | tee -a "$MASTER"
    fi
done

echo "[$(STAMP)] QUEUE COMPLETE" | tee -a "$MASTER"
