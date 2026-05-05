#!/bin/bash
# Run this ONCE from a login node (has internet) to pre-cache all needed assets.
# After this, the main ablation script will work fully offline on compute nodes.
#
# Usage (from login node):
#   bash scripts/prefetch_generator_assets.sh

set -euo pipefail

source /work/11034/atzanakak/work/11034/atzanakak/miniconda3/etc/profile.d/conda.sh || true
conda activate grlm

export HF_HOME=/work/11034/atzanakak/ls6/huggingface/transformers
export HF_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
# No HF_HUB_OFFLINE — we need internet here

echo "=== Pre-fetching GENERator model tokenizer + config ==="
python3 - <<'EOF'
from transformers import AutoTokenizer, AutoConfig
AutoConfig.from_pretrained("GenerTeam/GENERator-v2-eukaryote-3b-base", trust_remote_code=True)
AutoTokenizer.from_pretrained("GenerTeam/GENERator-v2-eukaryote-3b-base", trust_remote_code=True)
print("Model assets OK")
EOF

echo ""
echo "=== Pre-fetching benchmark datasets ==="
python3 - <<'EOF'
from datasets import load_dataset

TASKS = [
    ("GenerTeam/GENERator-Benchmark-EUK-EpigeneticMarks", "H3K4me3"),
    ("GenerTeam/GENERator-Benchmark-EUK-EpigeneticMarks", "H3K27me3"),
    ("GenerTeam/GENERator-Benchmark-EUK-EpigeneticMarks", "H3K36me3"),
    ("GenerTeam/GENERator-Benchmark-EUK-Splice",           None),
    ("GenerTeam/GENERator-Benchmark-EUK-GeneAnnotation",   None),
]

for dataset_name, subset in TASKS:
    label = f"{dataset_name}" + (f":{subset}" if subset else "")
    print(f"  Fetching {label} ...")
    try:
        if subset:
            ds = load_dataset(dataset_name, subset, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_name, trust_remote_code=True)
        print(f"    OK — splits: {list(ds.keys())}")
    except Exception as e:
        print(f"    FAILED: {e}")

print("Done.")
EOF

echo ""
echo "All assets pre-fetched. You can now run the ablation offline on compute nodes."
