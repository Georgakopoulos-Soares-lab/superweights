#!/bin/bash
# batch_int2_gue.sh — Fine-tune DNABERT-2 on diverse GUE tasks, then run INT2 50% eval
# Uses GPUs 6 and 7.
set -euo pipefail

REPO=/home/nvidia/superweights
cd "$REPO"
export PYTHONPATH="${PYTHONPATH:-}:$REPO/scripts/evaluation:$REPO/scripts/compression"

# ── Tasks (ordered roughly by training speed, fastest first) ──
TASKS=(
    "EMP/H3K9me3"
    "EMP/H3K36me3"
    "EMP/H3K79me3"
    "EMP/H4ac"
    "EPI/K562"
    "tf/0"
    "mouse/0"
    "prom/prom_core_all"
)

# ── GPU pool ──
GPU_POOL=(6 7)

_acquire_gpu() {
    while [ ${#GPU_POOL[@]} -eq 0 ]; do sleep 5; done
    local gpu="${GPU_POOL[0]}"
    GPU_POOL=("${GPU_POOL[@]:1}")
    echo "$gpu"
}
_release_gpu() { GPU_POOL+=("$1"); }

run_one() {
    local task="$1" gpu="$2"
    local task_leaf="${task//\//_}"
    local ckpt_dir="results/gue_checkpoints/dnabert2_${task_leaf}"
    local out_name="int2_$(echo "$task" | tr '/' '_')"

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  TASK: $task  (GPU $gpu)"
    echo "╚══════════════════════════════════════════════════════════╝"

    # ── Step 1: Fine-tune (if checkpoint doesn't exist) ──
    if [ ! -f "$ckpt_dir/model_state.pt" ]; then
        echo "  [FT] Fine-tuning $task ..."
        CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluation/run_gue_ablation.py \
            --model dnabert2 \
            --task "$task" \
            --gue_root /home/nvidia/data/gue/GUE \
            --device cuda \
            --out /dev/null \
            --sweep 2>&1 | tail -5
    else
        echo "  [FT] Checkpoint exists at $ckpt_dir — skipping fine-tune"
    fi

    # ── Step 2: INT2 50% evaluation ──
    echo "  [INT2] Running INT2 50% evaluation ..."
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/compression/run_int4_multicondition_benchmark.py \
        --gpu 0 \
        --task "$task" \
        --bits 2 \
        --n_seeds 5 \
        --fracs 50.0 \
        --out "results/${out_name}.json" 2>&1 | tail -30

    echo "  [DONE] $task  → results/${out_name}.json"
    _release_gpu "$gpu"
}

# ── Launch all tasks ──
for task in "${TASKS[@]}"; do
    gpu=$(_acquire_gpu)
    run_one "$task" "$gpu" &
done

wait
echo ""
echo "=== ALL TASKS COMPLETE ==="
