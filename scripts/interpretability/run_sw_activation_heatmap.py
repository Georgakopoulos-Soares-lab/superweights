"""
scripts/interpretability/run_sw_activation_heatmap.py
------------------------------------------------------
For each genomic context (promoter / enhancer / random for eukaryote;
promoter / terminator / random for prokaryote), extract N sequences and record
the mean absolute post-down_proj activation of:
  - Super-weight rows identified for that model
  - 10 randomly sampled non-SW control rows (baseline)

Outputs:
  1. Heatmap: rows = SW + control rows, columns = contexts, colour = mean |act|
     annotated with mean ± std.
  2. Bar chart: mean |act| per SW row per context with error bars.
  3. JSON  results/sw_activation_heatmap_generator.json
     PNG   results/sw_activation_heatmap_generator.png
     (and  results/sw_activation_heatmap_generator_prokaryote.{json,png})

Usage
-----
    # Eukaryote
    python scripts/interpretability/run_sw_activation_heatmap.py \\
        --model generator \\
        --fasta /path/to/hg38.fa \\
        --contexts promoters_262kb.bed:promoter enhancers_ccre_262kb.bed:enhancer random_262kb.bed:random \\
        --bed_dir  data/regions/hg38 \\
        --sw_layer 4 --sw_rows 2371 1522 \\
        --n_seqs 30 --window_bp 3072 \\
        --out results/sw_activation_heatmap_generator.json \\
        --plot results/sw_activation_heatmap_generator.png

    # Prokaryote
    python scripts/interpretability/run_sw_activation_heatmap.py \\
        --model generator_prokaryote \\
        --fasta data/reference/ecoli/ecoli_k12.fna \\
        --contexts promoters_ecoli.bed:promoter terminators_ecoli.bed:terminator random_ecoli.bed:random \\
        --bed_dir  data/regions/ecoli \\
        --sw_layer 2 --sw_rows 1927 \\
        --n_seqs 30 --window_bp 3072 \\
        --out results/sw_activation_heatmap_generator_prokaryote.json \\
        --plot results/sw_activation_heatmap_generator_prokaryote.png
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── repo root on path ─────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

# ── shared FASTA / BED helpers (reuse from causal tracing) ───────────────────
def _open_fasta(path: str):
    try:
        from pyfaidx import Fasta
        return Fasta(path, as_raw=True, sequence_always_upper=True)
    except ImportError:
        raise ImportError("pyfaidx required: pip install pyfaidx")


def _fetch_window(fasta, chrom: str, center: int, window_bp: int):
    half  = window_bp // 2
    start = max(0, center - half)
    end   = start + window_bp
    key   = (chrom if chrom in fasta
             else "chr" + chrom if "chr" + chrom in fasta
             else chrom.lstrip("chr") if chrom.lstrip("chr") in fasta
             else None)
    if key is None:
        return None
    chrom_len = len(fasta[key])
    if end > chrom_len:
        end   = chrom_len
        start = max(0, end - window_bp)
    seq = str(fasta[key][start:end])
    if len(seq) < window_bp:
        seq = seq + "N" * (window_bp - len(seq))
    return seq


def _sample_bed_windows(bed_path: str, fasta, n: int, window_bp: int,
                        rng: random.Random, label: str) -> list:
    with open(bed_path) as fh:
        lines = [l.strip().split("\t") for l in fh
                 if l.strip() and not l.startswith("#")]
    pool = rng.sample(lines, min(len(lines), 10 * n))
    results = []
    for row in pool:
        if len(results) >= n:
            break
        chrom  = row[0]
        s, e   = int(row[1]), int(row[2])
        center = (s + e) // 2
        seq    = _fetch_window(fasta, chrom, center, window_bp)
        if seq is None:
            continue
        if seq.count("N") / len(seq) > 0.1:
            continue
        results.append({"label": label, "chrom": chrom,
                        "center": center, "seq": seq})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Activation recorder: records per-row mean |act| at the down_proj output
# ─────────────────────────────────────────────────────────────────────────────

class _RowMagnitudeRecorder:
    """Hooks into a single Linear layer and records absolute activation per row."""

    def __init__(self, rows: list):
        self.rows  = rows            # list of int row indices to track
        self.data  = []              # list of {row_idx: float} per forward pass
        self._handle = None

    def register(self, module):
        def _hook(_mod, _inp, out):
            y = out[0] if isinstance(out, tuple) else out   # (B, L, D)
            flat = y.reshape(-1, y.shape[-1]).abs()          # (B*L, D)
            self.data.append(
                {r: flat[:, r].mean().item() for r in self.rows}
            )
        self._handle = module.register_forward_hook(_hook)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ─────────────────────────────────────────────────────────────────────────────
# Tokenise helper (k-mer alignment for GENERator)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(tokenizer, seq: str, device: str) -> dict:
    remainder = len(seq) % 6
    if remainder:
        seq = seq[remainder:]
    return tokenizer(seq, return_tensors="pt",
                     add_special_tokens=False).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop
# ─────────────────────────────────────────────────────────────────────────────

def measure_activations(model, tokenizer, down_proj_module,
                        sequences: list, rows: list, device: str) -> dict:
    """
    Run forward passes on `sequences` and record mean |activation| per row.

    Returns {row_idx: [val_per_seq, ...]}
    """
    recorder = _RowMagnitudeRecorder(rows)
    recorder.register(down_proj_module)

    for item in sequences:
        inputs = _tokenize(tokenizer, item["seq"], device)
        with torch.no_grad():
            model(**inputs)

    recorder.remove()

    # Pivot: row → list of per-sequence means
    by_row = {r: [] for r in rows}
    for record in recorder.data:
        for r, val in record.items():
            by_row[r].append(val)

    return by_row


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(results: dict, sw_rows: list, ctrl_rows: list,
          context_labels: list, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # ── Build data matrices ───────────────────────────────────────────────────
    all_rows     = sw_rows + ctrl_rows
    row_labels   = [f"L{results['sw_layer']}r{r} (SW)" for r in sw_rows] + \
                   [f"L{results['sw_layer']}r{r} (ctrl)" for r in ctrl_rows]
    n_rows       = len(all_rows)
    n_ctx        = len(context_labels)

    mean_mat = np.zeros((n_rows, n_ctx))
    std_mat  = np.zeros((n_rows, n_ctx))

    for ci, ctx in enumerate(context_labels):
        ctx_data = results["contexts"][ctx]
        for ri, row in enumerate(all_rows):
            vals = ctx_data[str(row)]
            mean_mat[ri, ci] = np.mean(vals)
            std_mat[ri, ci]  = np.std(vals)

    fig = plt.figure(figsize=(4 + 2 * n_ctx, 2 + 0.6 * n_rows))
    gs  = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.45)

    # ── Panel 1: Heatmap ─────────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0])
    # Use a diverging colour scale anchored at the control mean
    ctrl_mean = mean_mat[len(sw_rows):, :].mean()
    vmin = 0
    vmax = max(mean_mat.max() * 1.05, ctrl_mean * 2)

    im = ax_heat.imshow(mean_mat, aspect="auto", cmap="viridis",
                        vmin=vmin, vmax=vmax)

    # Annotate cells
    for ri in range(n_rows):
        for ci in range(n_ctx):
            txt = f"{mean_mat[ri,ci]:.2f}\n±{std_mat[ri,ci]:.2f}"
            color = "white" if mean_mat[ri, ci] > (vmax * 0.5) else "black"
            ax_heat.text(ci, ri, txt, ha="center", va="center",
                         fontsize=7, color=color)

    ax_heat.set_xticks(range(n_ctx))
    ax_heat.set_xticklabels(context_labels, fontsize=9)
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(row_labels, fontsize=8)
    ax_heat.set_title("SW row post-down_proj activation magnitude\n(mean |act| across sequences)",
                      fontsize=10)

    # Draw separator line between SW and control rows
    if ctrl_rows:
        ax_heat.axhline(len(sw_rows) - 0.5, color="white", linewidth=2, linestyle="--")

    plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02,
                 label="mean |activation|")

    # ── Panel 2: Bar chart (SW rows only) ────────────────────────────────────
    ax_bar  = fig.add_subplot(gs[1])
    x       = np.arange(n_ctx)
    width   = 0.7 / max(len(sw_rows), 1)
    offsets = np.linspace(-(len(sw_rows)-1)/2, (len(sw_rows)-1)/2, len(sw_rows)) * width
    palette = plt.cm.tab10.colors

    for si, row in enumerate(sw_rows):
        ri = si    # index in mean_mat
        means = mean_mat[ri, :]
        stds  = std_mat[ri, :]
        ax_bar.bar(x + offsets[si], means, width=width * 0.9,
                   label=f"r{row} (SW)", color=palette[si % 10],
                   yerr=stds, capsize=4, error_kw={"elinewidth": 1})

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(context_labels, fontsize=9)
    ax_bar.set_ylabel("mean |activation|", fontsize=8)
    ax_bar.set_title("SW rows only — mean ± std per context", fontsize=9)
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.set_ylim(bottom=0)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SW activation heatmap across genomic contexts"
    )
    p.add_argument("--model",    required=True,
                   help="Model config name, e.g. generator or generator_prokaryote")
    p.add_argument("--fasta",    required=True,
                   help="Path to reference FASTA (indexed with samtools faidx or pyfaidx)")
    p.add_argument("--contexts", nargs="+", required=True,
                   metavar="FILE:LABEL",
                   help="BED:label pairs, e.g. promoters_262kb.bed:promoter")
    p.add_argument("--bed_dir",  default="data/regions/hg38",
                   help="Directory containing the BED files")
    p.add_argument("--sw_layer", type=int, required=True,
                   help="Layer index of the SW (0-based)")
    p.add_argument("--sw_rows",  nargs="+", type=int, required=True,
                   help="Row indices of super-weight rows")
    p.add_argument("--n_ctrl",   type=int, default=10,
                   help="Number of random control rows to sample")
    p.add_argument("--n_seqs",   type=int, default=30,
                   help="Sequences per context")
    p.add_argument("--window_bp", type=int, default=3072,
                   help="Sequence window length in bp")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out",      default="results/sw_activation_heatmap_generator.json")
    p.add_argument("--plot",     default="results/sw_activation_heatmap_generator.png")
    p.add_argument("--configs_dir", default="configs")
    args = p.parse_args()

    rng = random.Random(args.seed)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = Path(args.configs_dir) / f"{args.model}.yaml"
    config   = yaml.safe_load(cfg_path.read_text())
    print(f"Model  : {config['model_id']}")
    print(f"SW     : layer {args.sw_layer}, rows {args.sw_rows}")
    print(f"Device : {args.device}")

    # ── Load model ────────────────────────────────────────────────────────────
    from models.generator_wrapper import GeneratorWrapper
    wrapper = GeneratorWrapper(config)
    wrapper.load()
    model     = wrapper.model
    tokenizer = wrapper.tokenizer
    model.eval()

    # ── Resolve down_proj at sw_layer ─────────────────────────────────────────
    down_proj = wrapper.get_target_module(args.sw_layer)
    num_rows  = down_proj.weight.data.shape[0]
    print(f"down_proj rows: {num_rows}")

    # ── Sample control rows (avoid SW rows) ───────────────────────────────────
    exclude = set(args.sw_rows)
    ctrl_candidates = [r for r in range(num_rows) if r not in exclude]
    ctrl_rows = rng.sample(ctrl_candidates, args.n_ctrl)
    all_rows  = args.sw_rows + ctrl_rows
    print(f"Control rows: {ctrl_rows}")

    # ── Open FASTA ────────────────────────────────────────────────────────────
    fasta = _open_fasta(args.fasta)

    # ── Process each context ─────────────────────────────────────────────────
    device = args.device
    results_contexts = {}
    context_labels   = []

    for ctx_spec in args.contexts:
        bed_file, label = ctx_spec.rsplit(":", 1)
        bed_path = Path(args.bed_dir) / bed_file
        print(f"\n[{label}] Sampling {args.n_seqs} sequences from {bed_path.name} ...")
        sequences = _sample_bed_windows(str(bed_path), fasta, args.n_seqs,
                                        args.window_bp, rng, label)
        if len(sequences) < args.n_seqs:
            print(f"  [warn] Only {len(sequences)} sequences extracted (needed {args.n_seqs})")
        if not sequences:
            print(f"  [ERROR] No sequences for context {label} — skipping.")
            continue

        print(f"  Running {len(sequences)} forward passes ...")
        by_row = measure_activations(model, tokenizer, down_proj,
                                     sequences, all_rows, device)

        results_contexts[label] = {str(r): by_row[r] for r in all_rows}
        context_labels.append(label)

        for r in args.sw_rows:
            vals = by_row[r]
            print(f"  row {r:4d} (SW)   : mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
        for r in ctrl_rows:
            vals = by_row[r]
            print(f"  row {r:4d} (ctrl) : mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    # ── Serialise results ─────────────────────────────────────────────────────
    out_data = {
        "model":    config["model_id"],
        "sw_layer": args.sw_layer,
        "sw_rows":  args.sw_rows,
        "ctrl_rows": ctrl_rows,
        "n_seqs":   args.n_seqs,
        "window_bp": args.window_bp,
        "contexts": results_contexts,
        "context_order": context_labels,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"\nResults → {args.out}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if context_labels:
        _plot(out_data, args.sw_rows, ctrl_rows, context_labels, args.plot)


if __name__ == "__main__":
    main()
