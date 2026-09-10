#!/usr/bin/env bash
# The frozen 22-model singleton causal census (Figure 2) and its structural inputs.
#
# One model per invocation of the underlying runner, which writes atomically and skips
# already-valid outputs, so this is restartable. DNABERT-2 needs the `dnabert` environment
# (transformers 4.29.x); everything else uses `generator`. Downloading the pinned checkpoints
# is the dominant cost on a cold cache.
#
# Usage:  bash pipelines/census.sh [model-slug ...]     (default: all)
source "$(dirname "$0")/_lib.sh"
CENSUS=experiments/frozen/E13_full_cohort_causal_census/run_singleton_census.py
ALL=(llama mistral olmo qwen25-7b mosaicbert modernbert-base ntv3 dnabert2 generator-euk-3b
     genomeocean-4b qwen25-0.5b qwen25-1.5b qwen25-3b smollm2-135m smollm2-360m smollm2-1.7b
     generator-prok-1.2b generator-prok-3b eurobert-210m eurobert-610m eurobert-2.1b
     modernbert-large)
MODELS=("${@:-${ALL[@]}}")
log "census: ${#MODELS[@]} model(s)"
{ for m in "${MODELS[@]}"; do
    py="$PY_GENERATOR"; [[ "$m" == dnabert2 ]] && py="$PY_DNABERT"
    echo "results/experiments/E13/raw/$m.json|census_$m|$py|$CENSUS --model $m"
  done
} | parallel_stage "${WORKERS:-4}"
stage results/experiments/E13/part2_22_model_results.csv census_compile "$PY_GENERATOR" \
      experiments/frozen/E13_full_cohort_causal_census/compile_census.py
log "done"
