#!/bin/bash
# Second-pass retry, ONLINE mode: the shared /work HF cache env_cached.sh assumes does not
# exist in this session (confirmed by direct filesystem check) -- only GENERator-EUK-3B and
# DNABERT-2 are actually present in the reachable /scratch cache. This pass allows HF Hub
# downloads (into /scratch, which has ample quota headroom, NOT /work, which is at ~90% of
# this user's file-count quota) for every model the first (offline) pass could not find
# locally, then retries the run with --force.
set -uo pipefail

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "$REPO/experiments/frozen/E13_full_cohort_causal_census"

export HF_HOME=/scratch/11034/atzanakak/huggingface_cache
export HF_HUB_CACHE="$HF_HOME"
unset PYTHONPATH PYTHONHOME
VENV=/scratch/11034/atzanakak/genomic-super-weights/.venv
export PATH="$VENV/bin:$PATH"
export SSL_CERT_FILE="$VENV/lib64/python3.12/site-packages/certifi/cacert.pem"
export CURL_CA_BUNDLE="$SSL_CERT_FILE"
unset REQUESTS_CA_BUNDLE HF_HUB_OFFLINE
export E13_STABILITY_N_INPUTS=24
export E13_LOCAL_FILES_ONLY=0

LOGDIR="$REPO/logs"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/stability_queue_online_master.log"
STAMP() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

MODELS=(mosaicbert modernbert-base ntv3 genomeocean-4b \
        qwen25-0.5b qwen25-1.5b smollm2-135m smollm2-360m smollm2-1.7b \
        generator-prok-1.2b generator-prok-3b eurobert-210m eurobert-610m \
        eurobert-2.1b modernbert-large qwen25-3b llama mistral olmo qwen25-7b)

echo "[$(STAMP)] ONLINE QUEUE START" | tee -a "$MASTER"
for m in "${MODELS[@]}"; do
    out="$REPO/results/experiments/E13_candidate_stability/${m}.json"
    if [ -f "$out" ]; then
        echo "[$(STAMP)] SKIP  $m (already succeeded)" | tee -a "$MASTER"
        continue
    fi
    logfile="$LOGDIR/stability_online_${m}.log"
    echo "[$(STAMP)] START $m" | tee -a "$MASTER"
    if python3 -u run_candidate_input_stability.py --model "$m" --force > "$logfile" 2>&1; then
        echo "[$(STAMP)] OK    $m -> $logfile" | tee -a "$MASTER"
    else
        echo "[$(STAMP)] FAIL  $m (exit $?) -> $logfile -- continuing queue" | tee -a "$MASTER"
    fi
done
echo "[$(STAMP)] ONLINE QUEUE COMPLETE" | tee -a "$MASTER"
