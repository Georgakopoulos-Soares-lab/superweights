#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_multimodule_detection.sbatch
#
# Runs run_multimodule_detection.py for every model that runs in the grlm
# conda env (transformer-based + HybridNa).  Each model scans ALL module
# types (down_proj, up/gate, attn_out, attn_q/k/v) and reports Δ% PPL for
# each — a full robustness check.
#
# Models NOT included here (need separate envs):
#   - evo2        → requires apptainer with evo2 container
#   - megadna     → requires megadna venv (run_megadna.sh)
#   - genomeocean → requires apptainer with genomeocean container
#
# Submit:  sbatch scripts/run_multimodule_detection.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J multimodule-sw
#SBATCH -o /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/multimodule-sw.%j.out
#SBATCH -e /work/11034/atzanakak/glm_super_weight/genomic-super-weights/logs/multimodule-sw.%j.err
#SBATCH -p gpu-h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A BCS25073

set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export TRANSFORMERS_CACHE=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_HUB_OFFLINE=1

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH=${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}

echo "HOST=$(hostname)"
date
nvidia-smi || true

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd $REPO/genomic-super-weights
export PYTHONPATH=$REPO/genomic-super-weights/stubs:$REPO/genomic-super-weights:$PYTHONPATH

OUT=$REPO/results/multimodule_detection.json

# ── Transformer-based models (grlm env) ──────────────────────────────────────
# Models are run sequentially so GPU memory is freed between each.

for MODEL in dnabert2 ntv3 generator generator_prokaryote generator_prokaryote_1b hybridna; do
    echo ""
    echo "══════════════════════════════════════════════════════════════════════"
    echo "  Model: $MODEL"
    echo "══════════════════════════════════════════════════════════════════════"

    # First do a dry-run to show which patterns will be scanned
    python scripts/run_multimodule_detection.py \
        --model "$MODEL" \
        --list_patterns

    # Full scan
    python scripts/run_multimodule_detection.py \
        --model  "$MODEL" \
        --probe  actb_full \
        --n_rand 10 \
        --out    "$OUT" \
        || echo "[warn] $MODEL failed — continuing"

    echo ""
done

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  All grlm-env models done."
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  To run Evo2:"
echo "    sbatch scripts/run_detection_evo2.sbatch  (modify to call run_multimodule_detection.py)"
echo ""
echo "  To run MegaDNA:"
echo "    bash run_megadna.sh scripts/run_multimodule_detection.py --model megadna --out $OUT"
echo ""
date
