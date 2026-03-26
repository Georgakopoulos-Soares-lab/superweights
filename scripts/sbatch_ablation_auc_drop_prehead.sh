#!/usr/bin/env bash
#SBATCH -J ablate_auc_prehead
#SBATCH -o results/ablation_auc_drop/slurm.%j.out
#SBATCH -e results/ablation_auc_drop/slurm.%j.err
#SBATCH -p gpu-a100-small         # Partition
#SBATCH -N 1                 # Request 1 node
#SBATCH --ntasks-per-node=1  # CORRECT: Request 3 processes per node
#SBATCH --mem=15G           # Increased memory for a larger job
#SBATCH -t 48:00:00          # Wall time
#SBATCH -A BCS25073

set -euo pipefail

# Avoid module pollution
module purge || true
unset LD_PRELOAD || true

# Activate the known-good environment (edit if needed)
# Option A: direct venv python (recommended; simplest)
PY="/scratch/10906/arisk/envs/superweights-borzoi/bin/python"
if [ ! -x "$PY" ]; then
  echo "[ERROR] Python not found at $PY" >&2
  exit 2
fi

# Always run from repo root
cd /work/10906/arisk/ls6/SuperWeights

# Make logs directory
mkdir -p results/ablation_auc_drop

# ---- Experiment config (tuned for <=15h) ----
# Runtime control knobs:
# - batch_size: try 8 first on A100; bump to 12/16 if it fits.
# - random_draws: 15 is ~2x faster than 30.
# - topk_list: keep as-is unless you want even faster.
BATCH_SIZE="${BATCH_SIZE:-8}"
RANDOM_DRAWS="${RANDOM_DRAWS:-15}"
TOPK_LIST="${TOPK_LIST:-1,5,10,20,50}"

LAYER="${LAYER:-final_joined_convs.0.conv_layer}"
RANK_TSV="${RANK_TSV:-results/variants/cage_wholeblood_ribopure_pad4_shift4_abs_all/best_final_joined_convs_0_conv_layer_across_settings_top100.tsv}"

OUTDIR="${OUTDIR:-results/ablation_auc_drop/wholeblood_cage_abs_primary_prehead_bsz${BATCH_SIZE}_rd${RANDOM_DRAWS}_job${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"

echo "[INFO] Job: ${SLURM_JOB_ID:-no_slurm}" 
echo "[INFO] OUTDIR=$OUTDIR"
echo "[INFO] BATCH_SIZE=$BATCH_SIZE RANDOM_DRAWS=$RANDOM_DRAWS TOPK_LIST=$TOPK_LIST"
echo "[INFO] LAYER=$LAYER"
echo "[INFO] RANK_TSV=$RANK_TSV"

# Disable TF/Flax inside transformers to speed import on HPC
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1

# Run. Script writes intermediate checkpoints to:
#   $OUTDIR/ablation_results.tsv, $OUTDIR/summary.tsv, $OUTDIR/run_manifest.json
# repeatedly during the sweep, so preemption still leaves usable partials.

"$PY" -u scripts/ablation_auc_drop.py \
  --preset wholeblood_cage_abs_primary \
  --pos_vcf data/eqtl/Whole_Blood_pos.vcf.gz \
  --neg_vcf data/eqtl/Whole_Blood_neg.vcf.gz \
  --genome_fasta data/hg38.fa \
  --n_pos all --n_neg all \
  --batch_size "$BATCH_SIZE" \
  --layer "$LAYER" \
  --rank_tsv "$RANK_TSV" \
  --topk_list "$TOPK_LIST" \
  --random_draws "$RANDOM_DRAWS" \
  --out_dir "$OUTDIR" \
  2>&1 | tee "$OUTDIR/run.log"

echo "[INFO] Done. Outputs in $OUTDIR"
