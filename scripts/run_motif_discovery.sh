#!/usr/bin/env bash
set -euo pipefail

# Run motif discovery + Tomtom TF matching for one channel.
#
# This script prefers STREME (pos-vs-neg discriminative motif discovery). If STREME
# is not available, it falls back to classic MEME using a Markov background model
# trained on the negative FASTA.
#
# Usage:
#   bash scripts/run_motif_discovery.sh <pos.fa> <neg.fa> <outdir>
#
# Requirements:
#   - MEME-suite installed and on PATH.
#     Preferred: streme + tomtom
#     Fallback: meme + fasta-get-markov + tomtom
#   - Provide a motif database for Tomtom (JASPAR).
#     Set: JASPAR_DB=/path/to/JASPAR.meme

POS_FASTA=${1:-}
NEG_FASTA=${2:-}
OUTDIR=${3:-}

if [[ -z "$POS_FASTA" || -z "$NEG_FASTA" || -z "$OUTDIR" ]]; then
  echo "Usage: bash scripts/run_motif_discovery.sh <pos.fa> <neg.fa> <outdir>" >&2
  exit 2
fi

if ! command -v tomtom >/dev/null 2>&1; then
  echo "ERROR: tomtom not found on PATH. Load MEME-suite." >&2
  exit 3
fi

mkdir -p "$OUTDIR"

MOTIFS=""

if command -v streme >/dev/null 2>&1; then
  echo "[1/3] STREME: $POS_FASTA vs control $NEG_FASTA"
  STREME_DIR="$OUTDIR/streme"
  streme \
    --p "$POS_FASTA" \
    --n "$NEG_FASTA" \
    --oc "$STREME_DIR" \
    --dna \
    --minw 6 --maxw 15 \
    --thresh 0.05 \
    >/dev/null
  MOTIFS="$STREME_DIR/streme.txt"
else
  if ! command -v meme >/dev/null 2>&1; then
    echo "ERROR: streme not found and meme not found on PATH. Load/install MEME-suite." >&2
    exit 3
  fi
  if ! command -v fasta-get-markov >/dev/null 2>&1; then
    echo "ERROR: streme not found and fasta-get-markov not found on PATH. Load/install MEME-suite." >&2
    exit 3
  fi

  echo "[1/3] MEME fallback: building background from $NEG_FASTA and running MEME on $POS_FASTA"
  BG="$OUTDIR/neg_background.markov"
  fasta-get-markov -dna -m 1 "$NEG_FASTA" "$BG" >/dev/null

  MEME_DIR="$OUTDIR/meme"
  meme "$POS_FASTA" \
    -dna \
    -oc "$MEME_DIR" \
    -bfile "$BG" \
    -mod zoops \
    -nmotifs 5 \
    -minw 6 -maxw 15 \
    -revcomp \
    >/dev/null
  MOTIFS="$MEME_DIR/meme.txt"
fi

JASPAR_DB=${JASPAR_DB:-}
if [[ -z "$JASPAR_DB" ]]; then
  echo "WARNING: JASPAR_DB not set; skipping Tomtom." >&2
  echo "Set JASPAR_DB=/path/to/JASPAR.meme and re-run." >&2
  exit 0
fi

if [[ ! -f "$JASPAR_DB" ]]; then
  echo "ERROR: JASPAR_DB file not found: $JASPAR_DB" >&2
  exit 4
fi

if [[ ! -f "$MOTIFS" ]]; then
  echo "ERROR: Motifs not found: $MOTIFS" >&2
  exit 5
fi

echo "[2/3] Tomtom vs JASPAR: $JASPAR_DB"
TOMTOM_DIR="$OUTDIR/tomtom"
tomtom \
  -oc "$TOMTOM_DIR" \
  -no-ssc -min-overlap 5 -dist pearson \
  -evalue \
  "$MOTIFS" "$JASPAR_DB" \
  >/dev/null

# Create a tiny summary (top match per discovered motif)
# Tomtom outputs a tab-delimited file named tomtom.tsv
TT="$TOMTOM_DIR/tomtom.tsv"
SUMMARY="$OUTDIR/summary.tsv"

echo "[3/3] Summary: $SUMMARY"
if [[ -f "$TT" ]]; then
  awk -F'\t' 'BEGIN{OFS="\t"} NR==1{next} $0!~/^#/ {
      q=$1; t=$2; e=$5;
      if (!(q in best) || e+0 < best_e[q]+0) {best[q]=t; best_e[q]=e}
    }
    END{
      print "motif","top_tf","evalue";
      for (q in best) print q,best[q],best_e[q];
    }' "$TT" | sort -k3,3g > "$SUMMARY"
else
  echo -e "motif\ttop_tf\tevalue" > "$SUMMARY"
fi

echo "Done: $OUTDIR"