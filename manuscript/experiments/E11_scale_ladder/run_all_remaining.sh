#!/bin/bash
set -x
source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh
conda activate grlm
export LD_PRELOAD=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm/lib/libstdc++.so.6
export HF_HOME=/scratch/11034/atzanakak/huggingface_cache
export HF_HUB_CACHE="$HF_HOME"
unset SSL_CERT_FILE REQUESTS_CA_BUNDLE
cd /work/11034/atzanakak/glm_super_weight/genomic-super-weights

LOGDIR=results/E11/logs
mkdir -p "$LOGDIR"

MODELS=(smollm2-360m smollm2-1.7b qwen25-0.5b qwen25-1.5b qwen25-3b generator-prok-1.2b generator-prok-3b eurobert-210m eurobert-610m eurobert-2.1b modernbert-large)

for m in "${MODELS[@]}"; do
    echo "=== STARTING $m at $(date) ==="
    python3 manuscript/experiments/E11_scale_ladder/run_model.py --model "$m" > "$LOGDIR/$m.log" 2>&1
    status=$?
    echo "=== FINISHED $m at $(date) with status $status ==="
    if [ $status -ne 0 ]; then
        echo "!!! $m FAILED, see $LOGDIR/$m.log"
    fi
done
echo "=== ALL DONE ==="
