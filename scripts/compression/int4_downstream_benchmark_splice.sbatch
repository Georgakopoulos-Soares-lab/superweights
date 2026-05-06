#!/bin/bash
#SBATCH --job-name=int4_benchmark_splice
#SBATCH --output=logs/int4_benchmark_splice_%j.log
#SBATCH --error=logs/int4_benchmark_splice_%j.log
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

echo "=== INT4 Downstream Benchmark: DNABERT-2 / splice/reconstructed ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
date

cd "$(dirname "$0")/../.."   # repo root

source ~/miniconda3/bin/activate evo

# Required: checkpoint from run_gue_ablation.py on splice/reconstructed
CKPT_DIR="results/gue_checkpoints/dnabert2_splice_reconstructed"
if [ ! -f "${CKPT_DIR}/model_state.pt" ]; then
    echo "ERROR: checkpoint not found at ${CKPT_DIR}/model_state.pt"
    echo "Run scripts/evaluation/run_gue_ablation.py first:"
    echo "  python scripts/evaluation/run_gue_ablation.py --model dnabert2 --task splice/reconstructed"
    exit 1
fi

mkdir -p logs results

python scripts/compression/run_int4_downstream_benchmark.py \
    --task         splice/reconstructed \
    --ckpt_dir     "${CKPT_DIR}" \
    --near_sw_frac 10.0 \
    --n_rand_seeds 10 \
    --bits         4 \
    --out          results/int4_downstream_benchmark_splice.json \
    --plot         results/int4_downstream_benchmark_splice.png

echo ""
echo "Done: $(date)"
