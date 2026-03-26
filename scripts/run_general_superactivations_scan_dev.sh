#!/usr/bin/env bash
#SBATCH -J general_superactivations
#SBATCH -o /work/10906/arisk/ls6/SuperWeights/results/general_superactivations_dev/slurm.%j.out
#SBATCH -e /work/10906/arisk/ls6/SuperWeights/results/general_superactivations_dev/slurm.%j.err
#SBATCH -p gpu-a100-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00
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

# Make sure output directories exist.
mkdir -p /work/10906/arisk/ls6/SuperWeights/results/general_superactivations_dev || true

# Minimal runner for scripts/general_superactivations_scan.py
# Customize the variables below; then run:
#   bash scripts/run_general_superactivations_scan.sh

# You can also override variables as KEY=VAL arguments, e.g.:
#   bash scripts/run_general_superactivations_scan.sh DEVICE=cuda NUM_WINDOWS_PER_SET=20000

for arg in "$@"; do
  if [[ "$arg" == *=* ]]; then
    export "$arg"
  else
    echo "Unexpected arg (expected KEY=VAL): $arg" >&2
    exit 2
  fi
done

# Reduce noisy logs on HPC nodes.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export TRANSFORMERS_NO_FLAX="${TRANSFORMERS_NO_FLAX:-1}"

: "${PY:=/scratch/10906/arisk/envs/superweights-borzoi/bin/python}"
: "${MODEL:=johahi/borzoi-replicate-0}"
: "${GENOME_FASTA:=/work/10906/arisk/ls6/SuperWeights/data/hg38.fa}"

# Optional model wrapper config
: "${OUTPUT_KEY:=}"

# Region sets
# Preferred: set REGION_SETS as a space-separated list of name=bed specs, e.g.
#   REGION_SETS="promoters=/path/p.bed enhancers=/path/e.bed random=/path/r.bed"
# Back-compat: REGION_SET_1_* and REGION_SET_2_*.
: "${REGION_SETS:=}"
: "${REGION_SET_1_NAME:=}"
: "${REGION_SET_1_BED:=}"
: "${REGION_SET_2_NAME:=}"
: "${REGION_SET_2_BED:=}"

# If not provided, default to the repo-built hg38 region sets (if present).
if [[ -z "${REGION_SETS}" ]]; then
  default_prom="/work/10906/arisk/ls6/SuperWeights/data/regions/hg38/promoters_262kb.bed"
  default_enh="/work/10906/arisk/ls6/SuperWeights/data/regions/hg38/enhancers_ccre_262kb.bed"
  default_rand="/work/10906/arisk/ls6/SuperWeights/data/regions/hg38/random_262kb.bed"
  if [[ -f "${default_prom}" && -f "${default_enh}" && -f "${default_rand}" ]]; then
    REGION_SETS="promoters=${default_prom} enhancers=${default_enh} random=${default_rand}"
    export REGION_SETS
  fi
fi

# Core scan params
: "${NUM_WINDOWS_PER_SET:=20000}"
: "${BATCH_SIZE:=1}"
: "${SEED:=1}"
: "${DEVICE:=cuda}"

# Layers: comma-separated module names. If empty, script auto-picks 3 conv layers.
: "${LAYERS:=horizontal_conv1.conv_layer,separable0.conv_layer.1,final_joined_convs.0.conv_layer}"
: "${REDUCE:=abs_max}"
: "${QUANTILES:=0.99,0.999}"
: "${TAIL_Z:=4.0}"
: "${TOP_CHANNELS:=50}"
: "${TOP_WINDOWS_PER_CHANNEL:=200}"

# Storage/extra metrics toggles
: "${USE_MEMMAP:=1}"
: "${COMPUTE_KURTOSIS:=0}"
: "${CHECKPOINT_EVERY_BATCHES:=50}"
: "${RESUME:=0}"

# Output
: "${OUT_DIR:=/work/10906/arisk/ls6/SuperWeights/results/general_superactivations/job${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "${GENOME_FASTA}" ]]; then
  echo "[ERROR] GENOME_FASTA not found: ${GENOME_FASTA}" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

cmd=(
  "$PY" -u /work/10906/arisk/ls6/SuperWeights/scripts/general_superactivations_scan.py
  --model_name_or_path "$MODEL"
  --genome_fasta "$GENOME_FASTA"
  --num_windows_per_set "$NUM_WINDOWS_PER_SET"
  --batch_size "$BATCH_SIZE"
  --reduce "$REDUCE"
  --quantiles "$QUANTILES"
  --tail_z "$TAIL_Z"
  --top_channels "$TOP_CHANNELS"
  --top_windows_per_channel "$TOP_WINDOWS_PER_CHANNEL"
  --out_dir "$OUT_DIR"
  --seed "$SEED"
  --device "$DEVICE"
  --checkpoint_every_batches "$CHECKPOINT_EVERY_BATCHES"
)

# Region set specs
region_specs=()
if [[ -n "${REGION_SETS}" ]]; then
  # shellcheck disable=SC2206
  region_specs=( ${REGION_SETS} )
else
  if [[ -n "${REGION_SET_1_NAME}" && -n "${REGION_SET_1_BED}" ]]; then
    region_specs+=("${REGION_SET_1_NAME}=${REGION_SET_1_BED}")
  fi
  if [[ -n "${REGION_SET_2_NAME}" && -n "${REGION_SET_2_BED}" ]]; then
    region_specs+=("${REGION_SET_2_NAME}=${REGION_SET_2_BED}")
  fi
fi

if [[ ${#region_specs[@]} -lt 1 ]]; then
  echo "[ERROR] No region sets provided." >&2
  echo "Set REGION_SETS like: REGION_SETS=\"promoters=/path/p.bed enhancers=/path/e.bed random=/path/r.bed\"" >&2
  exit 2
fi

for spec in "${region_specs[@]}"; do
  bed_path="${spec#*=}"
  if [[ -z "${bed_path}" || ! -f "${bed_path}" ]]; then
    echo "[ERROR] Region BED not found for spec '${spec}': ${bed_path}" >&2
    exit 2
  fi
done

# Helpful context: why a run might be tiny.
echo "NUM_WINDOWS_PER_SET=${NUM_WINDOWS_PER_SET}"
for spec in "${region_specs[@]}"; do
  name="${spec%%=*}"
  bed_path="${spec#*=}"
  n_lines=$(grep -vc '^#' "${bed_path}" || true)
  echo "Region set '${name}': ${bed_path} lines=${n_lines}"
done

for spec in "${region_specs[@]}"; do
  cmd+=(--region_set "$spec")
done

if [[ -n "${OUTPUT_KEY}" ]]; then
  cmd+=(--output_key "$OUTPUT_KEY")
fi

if [[ "${USE_MEMMAP}" == "1" ]]; then
  cmd+=(--use_memmap)
fi

if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
fi

if [[ "${COMPUTE_KURTOSIS}" == "1" ]]; then
  cmd+=(--compute_kurtosis)
fi

# Only pass --layers if non-empty (otherwise allow auto-pick).
if [[ -n "${LAYERS}" ]]; then
  cmd+=(--layers "$LAYERS")
fi

echo "OUT_DIR=$OUT_DIR"
echo "Running: ${cmd[*]}" | tee "$OUT_DIR/run_cmd.txt"

"${cmd[@]}" 2>&1 | tee "$OUT_DIR/run.log"

echo "Done. See: $OUT_DIR"
