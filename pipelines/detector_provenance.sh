#!/usr/bin/env bash
# EXP2 -- resolve the legacy-vs-ratio selection rule for every legacy census candidate.
#
# Runs the SmolLM2-1.7B positive control FIRST and stops if it fails: that model was selected
# by the current rule, so "frozen == ratio-argmax" holds by construction, and a failure there
# means the harness is broken rather than that a 7B model disagrees.
#
# Each model is then checked twice -- over 24 WikiText-2 windows (broader) and under the
# manuscript's own single-input discovery protocol (exact) -- and every run gates itself
# against four stored census quantities before reporting.
source "$(dirname "$0")/_lib.sh"
log "EXP2 detector provenance"

stage results/paper_closing/uniform_detector_text_smollm2-1.7b.json exp2_control \
      "$PY_GENERATOR" scripts/paper_closing/run_uniform_detector_text.py --model smollm2-1.7b \
  || { log "positive control FAILED -- harness is wrong, stopping"; exit 1; }

parallel_stage 3 <<EOF
results/paper_closing/uniform_detector_text_llama.json|exp2_llama|$PY_GENERATOR|scripts/paper_closing/run_uniform_detector_text.py --model llama
results/paper_closing/uniform_detector_text_mistral.json|exp2_mistral|$PY_GENERATOR|scripts/paper_closing/run_uniform_detector_text.py --model mistral
results/paper_closing/uniform_detector_text_olmo.json|exp2_olmo|$PY_GENERATOR|scripts/paper_closing/run_uniform_detector_text.py --model olmo
EOF

# published single-input protocol; appends all three models to one JSON, so run serially
for m in llama mistral olmo; do
  FORCE=1 stage results/paper_closing/detector_published_protocol.json "exp2_published_$m" \
        "$PY_GENERATOR" scripts/paper_closing/run_detector_published_protocol.py --model $m
done
log "done -- see results/paper_closing/EXP2_LEGACY_DETECTOR_RESOLUTION.md"
