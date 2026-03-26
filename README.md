# SuperWeights-Borzoi

Interpretability pipeline for the pre-trained [Borzoi](https://huggingface.co/johahi/borzoi-replicate-0) genomics model.
Given a set of causal GTEx Whole Blood eQTL SNPs, the pipeline identifies which internal CNN channels respond to those regulatory variants, localises activation sites in the genome, extracts surrounding sequences, and runs discriminative motif discovery to map channels to known transcription factor binding motifs.

---

## Repository layout

```
SuperWeights/
├── superweights/            # Thin utility library (variant parsing, FASTA, activation hooks)
├── superweights_borzoi/     # Deep-learning backend (model wrapper, encoding, statistics)
├── scripts/                 # Runnable pipeline scripts (one per stage)
├── data/                    # Input data — see "Data setup" below (large files are gitignored)
│   ├── eqtl/                # GTEx eQTL VCFs and derived parquets
│   ├── regions/hg38/        # BED files for promoter/enhancer/random regions (committed)
│   ├── hg38.fa              # Reference genome  ← gitignored, download separately
│   └── hg38.fa.fai          # FASTA index       ← gitignored, generated locally
├── results/                 # Pipeline outputs (gitignored)
├── motifs/                  # Per-channel FASTA sequences and MEME output (gitignored)
├── reports/                 # Human-readable markdown summaries (committed)
└── requirements.txt
```

---

## Environment setup

### Option A — Conda (recommended on HPC)

```bash
conda create -n superweights-borzoi python=3.11 -y
conda activate superweights-borzoi
pip install -r requirements.txt
```

On TACC LS6 the shared env is already built at:

```bash
conda activate /scratch/10906/arisk/envs/superweights-borzoi
```

### Option B — pip venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU requirement:** PyTorch ≥ 2.3 with CUDA. Tested on A100 (80 GB).
> On CPU, inference over 262 kb sequences will be extremely slow.

### Borzoi model weights

Weights download automatically from Hugging Face on first run. To pre-cache them:

```bash
PYTHONPATH=. python - <<'EOF'
from superweights_borzoi.models.borzoi_pt import BorzoiModel
BorzoiModel.from_pretrained("johahi/borzoi-replicate-0")
EOF
```

To use an offline cache, set `HF_HOME` before running:

```bash
export HF_HOME=/scratch/10906/arisk/hf_home
```

### External tool: MEME-suite (Step 5 only)

Motif discovery requires `streme`, `meme`, `fasta-get-markov`, and `tomtom`.

```bash
# Conda (easiest)
conda install -c bioconda meme -y

# Or on TACC
module load meme
```

Set `JASPAR_DB` to a MEME-format `.meme` file before running Step 5
(download from <https://jaspar.elixir.no/downloads/>).

---

## Data setup

Large files are gitignored. Download or symlink them before running the pipeline.

### 1. Reference genome (hg38, ~3 GB)

```bash
wget -P data/ https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip data/hg38.fa.gz
samtools faidx data/hg38.fa      # generates data/hg38.fa.fai
```

Or symlink an existing copy:

```bash
ln -s /path/to/hg38.fa     data/hg38.fa
ln -s /path/to/hg38.fa.fai data/hg38.fa.fai
```

### 2. GTEx Whole Blood eQTL VCFs

Download from the [GTEx portal](https://gtexportal.org/home/downloads/adult-gtex/qtl)
(GTEx v8, Whole Blood, fine-mapping credible sets with PIP ≥ 0.9).

Place the files at:

```
data/eqtl/Whole_Blood_pos.vcf.gz   # causal (positive) variants
data/eqtl/Whole_Blood_neg.vcf.gz   # non-causal (negative) controls
```

Then generate the derived parquet and variant-ID lists:

```bash
PYTHONPATH=. python scripts/derive_eqtl_parquet.py \
  --vcf_pos data/eqtl/Whole_Blood_pos.vcf.gz \
  --vcf_neg data/eqtl/Whole_Blood_neg.vcf.gz \
  --outdir  data/eqtl/derived
```

### 3. Region BED files

`data/regions/hg38/` is committed and requires no download.
It contains promoter, enhancer, and random-background BED files used for
region-stratified evaluation.

---

## Pipeline walkthrough

All scripts are run from the repo root with `PYTHONPATH=.`.

### Step 1 — Score variants & compute channel correlations

Runs Borzoi on each SNP (ref + alt allele), captures activations at 5 CNN layers,
and correlates per-channel activation deltas with GTEx effect size (`beta`) and
model-predicted expression change.

```bash
PYTHONPATH=. python scripts/run_channel_correlation.py \
  --parquet  data/eqtl/derived/Whole_Blood_pip0p9_topgene.parquet \
  --pos_ids  data/eqtl/derived/posneg/Whole_Blood_pos.variant_ids.txt \
  --neg_ids  data/eqtl/derived/posneg/Whole_Blood_neg.variant_ids.txt \
  --fasta    data/hg38.fa \
  --batch_size 8 \
  --outdir   results/full_v1
```

**Smoke test** (8 pos + 8 neg, ~5 min on A100):

```bash
PYTHONPATH=. python scripts/run_channel_correlation.py \
  --parquet  data/eqtl/derived/Whole_Blood_pip0p9_topgene.parquet \
  --pos_ids  data/eqtl/derived/posneg/Whole_Blood_pos.variant_ids.txt \
  --neg_ids  data/eqtl/derived/posneg/Whole_Blood_neg.variant_ids.txt \
  --fasta    data/hg38.fa \
  --n_pos 8 --n_neg 8 \
  --batch_size 8 \
  --outdir   results/smoke
```

Outputs: `per_variant_scores.parquet`, `channel_correlations.parquet`, `report.json`.

### Step 2 — Select a channel shortlist

Filters channels by discriminative AUC and biological alignment (Spearman ρ with
GTEx beta).

```bash
# Top-K by AUC
PYTHONPATH=. python scripts/select_channel_shortlist.py \
  --in_parquet results/full_v1/channel_correlations.parquet \
  --top_k 20 \
  --out        results/shortlist_channels.parquet \
  --report_md  reports/shortlist.md

# Threshold mode
PYTHONPATH=. python scripts/select_channel_shortlist.py \
  --in_parquet        results/full_v1/channel_correlations.parquet \
  --min_auc           0.60 \
  --min_abs_corr_beta 0.10 \
  --out               results/shortlist_channels.parquet \
  --report_md         reports/shortlist.md

# Drop "scalar-tracking" channels (high delta-expr corr but low beta corr)
PYTHONPATH=. python scripts/select_channel_shortlist.py \
  --in_parquet results/full_v1/channel_correlations.parquet \
  --top_k 50 \
  --drop_scalar_tracking \
  --out        results/shortlist_channels.parquet \
  --report_md  reports/shortlist.md
```

### Step 3 — Localise activation sites

Re-runs inference for each shortlisted (layer, channel) and extracts genomic
positions with the highest |delta-activation|.

```bash
PYTHONPATH=. python scripts/localize_channel_sites.py \
  --channels    results/shortlist_channels.parquet \
  --parquet     data/eqtl/derived/Whole_Blood_pip0p9_topgene.parquet \
  --pos_ids     data/eqtl/derived/posneg/Whole_Blood_pos.variant_ids.txt \
  --neg_ids     data/eqtl/derived/posneg/Whole_Blood_neg.variant_ids.txt \
  --fasta       data/hg38.fa \
  --k_near 5 --k_global 5 --near_window_bp 2048 \
  --batch_size  4 \
  --outdir      results/posmaps
```

Outputs one parquet per channel: `results/posmaps/<layer>_<channel>.parquet`.

### Step 4 — Extract motif FASTA sequences

```bash
PYTHONPATH=. python scripts/extract_motif_fasta.py \
  --posmaps         results/posmaps/*.parquet \
  --fasta           data/hg38.fa \
  --flank_bp        200 \
  --max_per_channel 1000 \
  --outdir          motifs
```

Outputs: `motifs/<layer>_<channel>_pos.fa` and `motifs/<layer>_<channel>_neg.fa`.

### Step 5 — Motif discovery (MEME-suite)

```bash
export JASPAR_DB=/path/to/JASPAR2024_CORE_vertebrates_non-redundant.meme

bash scripts/run_motif_discovery.sh \
  motifs/<layer>_<channel>_pos.fa \
  motifs/<layer>_<channel>_neg.fa \
  reports/motifs/<layer>_<channel>/
```

Runs STREME (discriminative: pos vs neg FASTA) then Tomtom against JASPAR.

### Step 6 — Motif ablation validation

In-silico perturbation: shuffle or substitute motif instances and measure the
effect on channel activation and predicted expression.

```bash
PYTHONPATH=. python scripts/validate_motif_ablation.py \
  --instances <motif_instances.parquet> \
  --fasta     data/hg38.fa \
  --outdir    results/ablation
```

### Optional — Task-aligned channel ranking

Rank channels by Spearman correlation with specific Borzoi output tracks
(e.g. Whole Blood CAGE tracks 260, 262, 784):

```bash
PYTHONPATH=. python scripts/variant_channel_rank.py \
  --parquet       data/eqtl/derived/Whole_Blood_pip0p9_topgene.parquet \
  --pos_ids       data/eqtl/derived/posneg/Whole_Blood_pos.variant_ids.txt \
  --neg_ids       data/eqtl/derived/posneg/Whole_Blood_neg.variant_ids.txt \
  --fasta         data/hg38.fa \
  --track_indices 260,262,784 \
  --pad_bins 4 --shift_bins -1 \
  --layers        final_joined_convs.0.conv_layer,horizontal_conv1.conv_layer \
  --outdir        results/variants/wholebloodCAGE_plus3_pad4_shiftm1
```

---

## Key design notes

- Input window: **262,144 bp** (256 kb) centred on each SNP.
- CNN layers probed: `conv_dna.conv_layer`, `res_tower.0/2.conv_layer`,
  `horizontal_conv0/1.conv_layer` — each has **768 channels**.
- Channel statistics: **Spearman ρ** (delta-activation vs beta / delta-expr) + **ROC AUC**.
- v1 supports **SNPs only**; indels are skipped automatically.
- Activation features: `mean`, `max`, `L2`, `abs_delta_max`, `abs_delta_L2`
  (all computed as allele-deltas: alt − ref).

---

## Citation / model credit

Borzoi model: [johahi/borzoi-replicate-0](https://huggingface.co/johahi/borzoi-replicate-0)  
GTEx data: [GTEx Consortium](https://gtexportal.org)
