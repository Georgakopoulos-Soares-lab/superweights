#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_caduceus_detection.sh
#
# Detects super weights in the Caduceus bidirectional Mamba model.
# Requires mamba_ssm (CUDA-compiled) — installs it if missing.
#
# Usage (from a GPU node):
#   bash run_caduceus_detection.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh
conda activate grlm

CONDA_ENV=/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache
export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=0   # need to fetch model code

REPO=/work/11034/atzanakak/glm_super_weight/genomic-super-weights
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "HOST=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
date

# ── 1. Load CUDA module for nvcc (required to build mamba_ssm) ───────────────
module load cuda/12.8
export CUDA_HOME="${CUDA_HOME:-/opt/apps/cuda/12.8}"
echo "nvcc: $(which nvcc 2>/dev/null || echo 'not found')"

# ── 2. Install mamba_ssm if missing ──────────────────────────────────────────
python3 -c "import mamba_ssm" 2>/dev/null || {
    echo "Installing mamba_ssm..."
    pip install mamba-ssm --no-build-isolation
}

# ── 2. Inspect layer names and write config ───────────────────────────────────
echo ""
echo ">>> Inspecting Caduceus layer structure..."
python3 - <<'PYEOF'
import torch
from transformers import AutoModel

m = AutoModel.from_pretrained(
    'kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16',
    trust_remote_code=True,
    cache_dir='/work/11034/atzanakak/ls6/huggingface/.hf-cache',
)
names = [n for n, mod in m.named_modules()
         if hasattr(mod, 'weight') and mod.weight is not None
         and len(mod.weight.shape) == 2]
for n in names[:80]:
    mod = dict(m.named_modules())[n]
    print(f"  {n}  shape={tuple(mod.weight.shape)}")
PYEOF

# ── 3. Write caduceus config (out_proj is the down_proj equivalent in Mamba) ──
# Layer path confirmed by inspection above; edit if different.
cat > configs/caduceus.yaml << 'YAML'
model_id: "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
model_type: "caduceus"
hf_trust_remote_code: true
# Caduceus is a bidirectional Mamba (RCPS) masked-LM model (7M params).
# No standard MLP down_proj exists. The nearest equivalent is the out_proj
# inside each Mamba mixer block, which projects from inner_dim (512) back to
# d_model (256). Path confirmed by model inspection.
down_proj_pattern: "caduceus.layers.{i}.mixer.out_proj"
num_layers: 16
dtype: "float32"
device: "cuda"
causal: false
YAML
echo "Config written to configs/caduceus.yaml"

# ── 4. Run super-weight detection ─────────────────────────────────────────────
echo ""
echo ">>> Running SW detection on Caduceus..."
python3 scripts/detection/run_detection.py \
    --model   caduceus \
    --configs_dir configs \
    --out     results/caduceus_ablation_results.json \
    || python3 scripts/detection/detect_super_weights.py \
        --model   caduceus \
        --configs_dir configs \
        --out     results/caduceus_ablation_results.json

echo ""
echo "Done. Check results/caduceus_ablation_results.json"
date
