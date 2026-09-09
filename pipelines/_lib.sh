#!/usr/bin/env bash
# Shared pipeline helpers: a GPU worker pool and artifact-based skipping.
#
# Every pipeline here is idempotent. A stage whose output artifact already exists is skipped,
# so an interrupted run resumes instead of recomputing, and re-running a finished pipeline
# costs nothing. Use FORCE=1 to recompute regardless.
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/data/huggingface_cache}"
export TOKENIZERS_PARALLELISM=false
PY_GENERATOR="${PY_GENERATOR:-/home/nvidia/miniconda3/envs/generator/bin/python}"
PY_DNABERT="${PY_DNABERT:-/home/nvidia/miniconda3/envs/dnabert/bin/python}"
LOGDIR="${LOGDIR:-$ROOT/results/pipeline_logs}"; mkdir -p "$LOGDIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# gpus_free -- indices of GPUs with no process resident
gpus_free() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '$2 < 500 {print $1}'
}

# stage <artifact> <name> <python> <script> [args...]
# Runs on the next free GPU; skips if <artifact> exists and FORCE is unset.
stage() {
  local art="$1" name="$2" py="$3"; shift 3
  if [[ -e "$ROOT/$art" && -z "${FORCE:-}" ]]; then
    log "SKIP  $name  (have $art)"; return 0
  fi
  local gpu; gpu="$(gpus_free | head -1)"; gpu="${gpu:-0}"
  log "RUN   $name  on GPU $gpu"
  ( cd "$ROOT" && CUDA_VISIBLE_DEVICES="$gpu" "$py" "$@" ) \
      > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then log "OK    $name"; else log "FAIL  $name (rc=$rc) -> $LOGDIR/$name.log"; fi
  return $rc
}

# parallel_stage <n_workers> -- reads "artifact|name|python|script args..." lines on stdin
parallel_stage() {
  local n="$1" pids=() line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    IFS='|' read -r art name py cmd <<< "$line"
    if [[ -e "$ROOT/$art" && -z "${FORCE:-}" ]]; then log "SKIP  $name  (have $art)"; continue; fi
    while (( $(jobs -rp | wc -l) >= n )); do sleep 5; done
    local gpu; gpu="$(gpus_free | head -1)"; gpu="${gpu:-0}"
    log "RUN   $name  on GPU $gpu"
    ( cd "$ROOT" && CUDA_VISIBLE_DEVICES="$gpu" $py $cmd > "$LOGDIR/$name.log" 2>&1 \
        && log "OK    $name" || log "FAIL  $name -> $LOGDIR/$name.log" ) &
    pids+=($!); sleep 8   # stagger so the free-GPU query sees the previous claim
  done
  wait "${pids[@]}" 2>/dev/null || true
}
