"""
scripts/interpretability/run_sw_task_activation_heatmap.py
----------------------------------------------------------
Visualises super-weight row activation magnitude across GUE genomic tasks
(or equivalent prokaryote BED-based benchmark) for GENERator models.

Modes
-----
  --mode gue   Load sequences from GUE CSV files (eukaryote, default)
  --mode bed   Load sequences from BED files + reference FASTA (prokaryote)

GUE mode usage
--------------
    python scripts/interpretability/run_sw_task_activation_heatmap.py \\
        --mode gue \\
        --model generator \\
        --gue_root /work/11034/atzanakak/GUE/GUE \\
        --sw_layer 4 --sw_rows 2371 1522 \\
        --n_per_class 25 \\
        --device cuda \\
        --out  results/sw_task_activation_heatmap_generator_euk.json \\
        --plot results/sw_task_activation_heatmap_generator_euk.png

BED mode usage (prokaryote)
---------------------------
    python scripts/interpretability/run_sw_task_activation_heatmap.py \\
        --mode bed \\
        --model generator_prokaryote \\
        --fasta data/reference/ecoli/ecoli_k12.fna \\
        --bed_dir data/regions/ecoli \\
        --bed_tasks promoter:promoters_ecoli.bed terminator:terminators_ecoli.bed \\
        --bed_negative random_ecoli.bed \\
        --sw_layer 2 --sw_rows 1927 \\
        --n_per_class 25 \\
        --window_bp 512 \\
        --device cuda \\
        --out  results/sw_task_activation_heatmap_generator_prok.json \\
        --plot results/sw_task_activation_heatmap_generator_prok.png
"""

import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

# ── GUE task ordering by category ────────────────────────────────────────────
GUE_TASKS_DEFAULT = [
    "prom/prom_core_notata",
    "prom/prom_core_tata",
    "prom/prom_300_notata",
    "splice/reconstructed",
    "EMP/H3K4me1",
    "EMP/H3K4me3",
    "EMP/H3K36me3",
    "tf/0",
    "tf/1",
    "tf/2",
    "virus/covid",
]

# Category for each task (used for x-axis grouping in bar chart)
def _category(task: str) -> str:
    if task.startswith("prom"):   return "Promoter"
    if task.startswith("splice"): return "Splice"
    if task.startswith("EMP"):    return "Histone"
    if task.startswith("tf"):     return "TF binding"
    if task.startswith("virus"):  return "Virus"
    return "Other"


# Short label for plots
def _short(task: str) -> str:
    parts = task.split("/")
    return parts[-1].replace("prom_", "").replace("_notata", "_noTATA") \
                    .replace("_tata", "_TATA").replace("reconstructed", "splice")


# ─────────────────────────────────────────────────────────────────────────────
# GUE CSV sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_gue_task(csv_path: str, n_per_class: int, rng: random.Random) -> dict:
    """
    Load a GUE CSV and return balanced samples per class.
    Returns {"all": [...], "by_class": {label: [...]}}
    Each item is {"seq": str, "label": int}.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    labels = sorted(df["label"].unique().tolist())

    by_class = {}
    for lbl in labels:
        rows = df[df["label"] == lbl].sample(
            n=min(n_per_class, len(df[df["label"] == lbl])),
            random_state=rng.randint(0, 2**31)
        )
        by_class[int(lbl)] = [{"seq": r["sequence"], "label": int(lbl)}
                               for _, r in rows.iterrows()]

    all_seqs = [item for items in by_class.values() for item in items]
    return {"all": all_seqs, "by_class": by_class, "labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# BED / FASTA sampling (prokaryote)
# ─────────────────────────────────────────────────────────────────────────────

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
    n = len(fasta[key])
    if end > n:
        end = n; start = max(0, end - window_bp)
    seq = str(fasta[key][start:end])
    if len(seq) < window_bp:
        seq += "N" * (window_bp - len(seq))
    return seq


def _sample_bed(bed_path: str, fasta, n: int, window_bp: int,
                rng: random.Random, label: int) -> list:
    with open(bed_path) as fh:
        lines = [l.strip().split("\t") for l in fh
                 if l.strip() and not l.startswith("#")]
    pool = rng.sample(lines, min(len(lines), 15 * n))
    results = []
    for row in pool:
        if len(results) >= n:
            break
        chrom  = row[0]; s, e = int(row[1]), int(row[2])
        seq = _fetch_window(fasta, chrom, (s + e) // 2, window_bp)
        if seq is None or seq.count("N") / len(seq) > 0.1:
            continue
        results.append({"seq": seq, "label": label})
    return results


def _sample_bed_task(pos_bed: str, neg_bed: str, fasta, n_per_class: int,
                     window_bp: int, rng: random.Random) -> dict:
    pos = _sample_bed(pos_bed, fasta, n_per_class, window_bp, rng, label=1)
    neg = _sample_bed(neg_bed, fasta, n_per_class, window_bp, rng, label=0)
    return {"all": pos + neg, "by_class": {1: pos, 0: neg}, "labels": [0, 1]}


# ─────────────────────────────────────────────────────────────────────────────
# Activation measurement
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_seq(tokenizer, seq: str, device: str) -> torch.Tensor | None:
    """Tokenize a sequence, aligning to 6-mer boundary. Returns None if too short."""
    remainder = len(seq) % 6
    if remainder:
        seq = seq[remainder:]
    if len(seq) < 6:
        return None
    ids = tokenizer(seq, return_tensors="pt",
                    add_special_tokens=False)["input_ids"].to(device)
    return ids


def measure_task_activations(model, tokenizer, down_proj_module,
                              sequences: list, sw_rows: list,
                              device: str) -> dict:
    """
    Run forward passes; record mean |activation| per SW row per sequence.
    Returns {row: [val_per_seq, ...]}
    """
    results = {r: [] for r in sw_rows}
    captured = {}

    def _hook(_mod, _inp, out):
        captured["out"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = down_proj_module.register_forward_hook(_hook)
    try:
        for item in sequences:
            ids = _tokenize_seq(tokenizer, item["seq"], device)
            if ids is None:
                for r in sw_rows:
                    results[r].append(None)
                continue
            captured.clear()
            with torch.no_grad():
                model(input_ids=ids)
            act = captured.get("out")  # (1, L, D)
            if act is None:
                for r in sw_rows:
                    results[r].append(None)
            else:
                for r in sw_rows:
                    results[r].append(float(act[0, :, r].abs().mean().cpu()))
    finally:
        handle.remove()

    # Drop None entries
    for r in sw_rows:
        results[r] = [v for v in results[r] if v is not None]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(data: dict, sw_rows: list, task_names: list, sw_layer: int,
          out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec

    n_rows  = len(sw_rows)
    n_tasks = len(task_names)
    short   = [_short(t) for t in task_names]
    cats    = [_category(t) for t in task_names]

    # ── Build mean/std matrices ───────────────────────────────────────────────
    mean_all  = np.zeros((n_rows, n_tasks))
    std_all   = np.zeros((n_rows, n_tasks))
    mean_pos  = np.full((n_rows, n_tasks), np.nan)
    std_pos   = np.full((n_rows, n_tasks), np.nan)
    mean_neg  = np.full((n_rows, n_tasks), np.nan)
    std_neg   = np.full((n_rows, n_tasks), np.nan)

    for ti, task in enumerate(task_names):
        td = data["tasks"][task]
        for ri, row in enumerate(sw_rows):
            vals = td["all"][str(row)]
            mean_all[ri, ti] = np.mean(vals) if vals else np.nan
            std_all[ri, ti]  = np.std(vals)  if vals else np.nan
            if "pos" in td and str(row) in td["pos"]:
                pv = td["pos"][str(row)]
                if pv: mean_pos[ri, ti] = np.mean(pv); std_pos[ri, ti] = np.std(pv)
            if "neg" in td and str(row) in td["neg"]:
                nv = td["neg"][str(row)]
                if nv: mean_neg[ri, ti] = np.mean(nv); std_neg[ri, ti] = np.std(nv)

    # Global mean for diverging colormap centre
    global_mean = np.nanmean(mean_all)

    fig = plt.figure(figsize=(max(12, 1.5 * n_tasks), 14))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[2.5, 2.5, 3], hspace=0.55)

    row_labels = [f"L{sw_layer}r{r} (SW)" for r in sw_rows]

    # ── Panel 1: Overall heatmap ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    vmax = np.nanmax(np.abs(mean_all - global_mean))
    norm = mcolors.TwoSlopeNorm(vcenter=global_mean,
                                vmin=max(0.0, global_mean - vmax),
                                vmax=global_mean + vmax)
    im1 = ax1.imshow(mean_all, aspect="auto", cmap="RdBu_r", norm=norm)
    for ri in range(n_rows):
        for ti in range(n_tasks):
            v, s = mean_all[ri, ti], std_all[ri, ti]
            if not np.isnan(v):
                txt_col = "white" if abs(v - global_mean) > vmax * 0.5 else "black"
                ax1.text(ti, ri, f"{v:.2f}\n±{s:.2f}", ha="center", va="center",
                         fontsize=6.5, color=txt_col)
    _set_task_axes(ax1, short, cats, row_labels)
    ax1.set_title(f"SW row post-down_proj activation (mean |act|) — all sequences",
                  fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=ax1, fraction=0.02, pad=0.02, label="mean |act|")

    # ── Panel 2: Per-class heatmap (pos vs neg sub-columns) ──────────────────
    ax2 = fig.add_subplot(gs[1])
    # Stack pos/neg alternately: columns = task0_pos, task0_neg, task1_pos, task1_neg …
    combined_mean = np.hstack([
        np.column_stack([mean_pos[:, ti:ti+1], mean_neg[:, ti:ti+1]])
        for ti in range(n_tasks)
    ])
    combined_std = np.hstack([
        np.column_stack([std_pos[:, ti:ti+1], std_neg[:, ti:ti+1]])
        for ti in range(n_tasks)
    ])
    vmax2 = np.nanmax(np.abs(combined_mean - global_mean))
    norm2 = mcolors.TwoSlopeNorm(vcenter=global_mean,
                                 vmin=max(0.0, global_mean - vmax2),
                                 vmax=global_mean + vmax2)
    im2 = ax2.imshow(combined_mean, aspect="auto", cmap="RdBu_r", norm=norm2)
    n_cols2 = combined_mean.shape[1]
    for ri in range(n_rows):
        for ci in range(n_cols2):
            v, s = combined_mean[ri, ci], combined_std[ri, ci]
            if not np.isnan(v):
                txt_col = "white" if abs(v - global_mean) > vmax2 * 0.5 else "black"
                ax2.text(ci, ri, f"{v:.2f}\n±{s:.2f}", ha="center", va="center",
                         fontsize=5.5, color=txt_col)
    # Sub-column labels: alternating pos/neg per task
    sub_labels = []
    for t in short:
        sub_labels += [f"{t}\npos", f"{t}\nneg"]
    ax2.set_xticks(range(n_cols2))
    ax2.set_xticklabels(sub_labels, fontsize=6, rotation=45, ha="right")
    ax2.set_yticks(range(n_rows)); ax2.set_yticklabels(row_labels, fontsize=8)
    # Vertical separators between tasks
    for ti in range(n_tasks - 1):
        ax2.axvline(ti * 2 + 1.5, color="white", linewidth=1.5)
    ax2.set_title("SW activation split by class (positive vs negative sequences)\n"
                  "→ shows whether SW fires selectively on functional sequences",
                  fontsize=9, fontweight="bold")
    plt.colorbar(im2, ax=ax2, fraction=0.02, pad=0.02, label="mean |act|")

    # ── Panel 3: Bar chart ────────────────────────────────────────────────────
    ax3   = fig.add_subplot(gs[2])
    x     = np.arange(n_tasks)
    width = 0.7 / n_rows
    offsets = np.linspace(-(n_rows - 1) / 2, (n_rows - 1) / 2, n_rows) * width
    palette = plt.cm.tab10.colors

    for ri, row in enumerate(sw_rows):
        means = mean_all[ri, :]
        stds  = std_all[ri, :]
        ax3.bar(x + offsets[ri], means, width=width * 0.9,
                label=f"r{row} (SW)", color=palette[ri % 10],
                yerr=stds, capsize=3, error_kw={"elinewidth": 1})

    # Category shading
    cat_starts = {}
    for ti, cat in enumerate(cats):
        cat_starts.setdefault(cat, ti)
    for ci, (cat, start) in enumerate(cat_starts.items()):
        end = n_tasks
        for ti in range(start + 1, n_tasks):
            if cats[ti] != cat:
                end = ti; break
        if ci % 2 == 1:
            ax3.axvspan(start - 0.5, end - 0.5, color="lightgrey", alpha=0.3, zorder=0)
        ax3.text((start + end - 1) / 2, ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0 else 0.1,
                 cat, ha="center", va="bottom", fontsize=7, color="grey")

    ax3.set_xticks(x)
    ax3.set_xticklabels(short, fontsize=7, rotation=45, ha="right")
    ax3.set_ylabel("mean |activation|", fontsize=9)
    ax3.set_title("SW row activation per task (mean ± std)", fontsize=9, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_ylim(bottom=0)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(f"GENERator SW activation across tasks  (layer {sw_layer})",
                 fontsize=12, fontweight="bold", y=1.01)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def _set_task_axes(ax, short_labels, cats, row_labels):
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: positive/negative class split for multi-class tasks
# ─────────────────────────────────────────────────────────────────────────────

def _pos_neg_split(by_class: dict, labels: list) -> tuple:
    """Return (pos_seqs, neg_seqs): label 0 is negative, all others positive."""
    neg = by_class.get(0, [])
    pos = [item for lbl, items in by_class.items() if lbl != 0 for item in items]
    return pos, neg


# ─────────────────────────────────────────────────────────────────────────────
# Combined EUK + PROK panel-2 figure
# ─────────────────────────────────────────────────────────────────────────────

def _build_posneg_matrix(data: dict, sw_rows: list, task_names: list):
    """Return (combined_mean, combined_std, sub_labels) for the pos/neg panel."""
    import numpy as np
    n_rows  = len(sw_rows)
    n_tasks = len(task_names)
    mean_all = np.zeros((n_rows, n_tasks))
    mean_pos = np.full((n_rows, n_tasks), np.nan)
    std_pos  = np.full((n_rows, n_tasks), np.nan)
    mean_neg = np.full((n_rows, n_tasks), np.nan)
    std_neg  = np.full((n_rows, n_tasks), np.nan)
    for ti, task in enumerate(task_names):
        td = data["tasks"][task]
        for ri, row in enumerate(sw_rows):
            vals = td["all"][str(row)]
            mean_all[ri, ti] = np.mean(vals) if vals else np.nan
            if "pos" in td and str(row) in td["pos"]:
                pv = td["pos"][str(row)]
                if pv: mean_pos[ri, ti] = np.mean(pv); std_pos[ri, ti] = np.std(pv)
            if "neg" in td and str(row) in td["neg"]:
                nv = td["neg"][str(row)]
                if nv: mean_neg[ri, ti] = np.mean(nv); std_neg[ri, ti] = np.std(nv)
    combined_mean = np.hstack([
        np.column_stack([mean_pos[:, ti:ti+1], mean_neg[:, ti:ti+1]])
        for ti in range(n_tasks)
    ])
    combined_std = np.hstack([
        np.column_stack([std_pos[:, ti:ti+1], std_neg[:, ti:ti+1]])
        for ti in range(n_tasks)
    ])
    global_mean = float(np.nanmean(mean_all))
    short = [_short(t) for t in task_names]
    sub_labels = []
    for t in short:
        sub_labels += [f"{t}\n+", f"{t}\n−"]
    return combined_mean, combined_std, sub_labels, global_mean


def _plot_combined_posneg(
    euk_json: str, prok_json: str, out_path: str
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    euk_data  = json.load(open(euk_json))
    prok_data = json.load(open(prok_json))

    euk_tasks  = euk_data["task_order"]
    prok_tasks = prok_data["task_order"]
    euk_rows   = euk_data["sw_rows"]
    prok_rows  = prok_data["sw_rows"]
    euk_layer  = euk_data["sw_layer"]
    prok_layer = prok_data["sw_layer"]

    euk_cm,  euk_cs,  euk_sl,  euk_gm  = _build_posneg_matrix(euk_data,  euk_rows,  euk_tasks)
    prok_cm, prok_cs, prok_sl, prok_gm = _build_posneg_matrix(prok_data, prok_rows, prok_tasks)

    # Stacked layout: EUK on top row, PROK on bottom row
    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(14, 0.9 * max(len(euk_sl), len(prok_sl))), 5 * max(len(euk_rows), len(prok_rows))),
        gridspec_kw={"hspace": 0.6},
    )

    for ax, cm, cs, sl, gm, rows, layer, title in [
        (axes[0], euk_cm,  euk_cs,  euk_sl,  euk_gm,  euk_rows,  euk_layer,
         "EUK GENERator — SW activation (positive vs negative sequences)"),
        (axes[1], prok_cm, prok_cs, prok_sl, prok_gm, prok_rows, prok_layer,
         "PROK GENERator — SW activation (positive vs negative sequences)"),
    ]:
        vmax = np.nanmax(np.abs(cm - gm))
        norm = mcolors.TwoSlopeNorm(vcenter=gm,
                                    vmin=max(0.0, gm - vmax),
                                    vmax=gm + vmax)
        im = ax.imshow(cm, aspect="auto", cmap="RdBu_r", norm=norm)
        n_cols = cm.shape[1]
        n_r    = len(rows)
        row_labels = [f"L{layer}r{r}" for r in rows]
        for ri in range(n_r):
            for ci in range(n_cols):
                v, s = cm[ri, ci], cs[ri, ci]
                if not np.isnan(v):
                    txt = "white" if abs(v - gm) > vmax * 0.5 else "black"
                    ax.text(ci, ri, f"{v:.2f}\n±{s:.2f}",
                            ha="center", va="center", fontsize=5.5, color=txt)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(sl, fontsize=6.5, rotation=45, ha="right")
        ax.set_yticks(range(n_r))
        ax.set_yticklabels(row_labels, fontsize=8)
        # Vertical separators between tasks
        n_tasks = n_cols // 2
        for ti in range(n_tasks - 1):
            ax.axvline(ti * 2 + 1.5, color="white", linewidth=1.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02, label="mean |act|")

    fig.suptitle("Super-weight row activation: functional vs non-functional sequences",
                 fontsize=12, fontweight="bold")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",     choices=["gue", "bed"], default="gue")
    p.add_argument("--model",    default=None,
                   help="Config name: generator or generator_prokaryote (not needed with --replot)")
    p.add_argument("--configs_dir", default="configs")
    p.add_argument("--sw_layer", type=int, default=None)
    p.add_argument("--sw_rows",  nargs="+", type=int, default=None)
    p.add_argument("--n_per_class", type=int, default=25)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--out",   default="results/sw_task_activation_heatmap.json")
    p.add_argument("--plot",  default="results/sw_task_activation_heatmap.png")
    p.add_argument("--replot", action="store_true",
                   help="Skip model loading; re-plot from existing --out JSON")
    p.add_argument("--combined_plot", default=None,
                   metavar="EUK_JSON:PROK_JSON:OUT_PNG",
                   help="Produce a combined EUK+PROK pos/neg panel. "
                        "Format: euk_json:prok_json:out_png. No model loading needed.")

    # GUE mode
    p.add_argument("--gue_root", default="/work/11034/atzanakak/GUE/GUE")
    p.add_argument("--tasks",    nargs="+", default=GUE_TASKS_DEFAULT)

    # BED mode
    p.add_argument("--fasta",        help="Reference FASTA (BED mode)")
    p.add_argument("--bed_dir",      default="data/regions/ecoli")
    p.add_argument("--bed_tasks",    nargs="+",
                   metavar="NAME:BED_FILE",
                   help="Positive-class BED per task, e.g. promoter:promoters_ecoli.bed")
    p.add_argument("--bed_negative", help="BED file for the negative class (shared)")
    p.add_argument("--window_bp",    type=int, default=512)

    args = p.parse_args()
    rng  = random.Random(args.seed)

    # ── Combined EUK+PROK pos/neg figure ─────────────────────────────────────
    if args.combined_plot:
        parts = args.combined_plot.split(":")
        if len(parts) != 3:
            p.error("--combined_plot must be euk_json:prok_json:out_png")
        euk_j, prok_j, out_png = parts
        _plot_combined_posneg(euk_j, prok_j, out_png)
        return

    # ── Replot-only mode ──────────────────────────────────────────────────────
    if args.replot:
        out_data = json.load(open(args.out))
        _plot(out_data, out_data["sw_rows"], out_data["task_order"],
              out_data["sw_layer"], args.plot)
        return

    # Validate required args for full run
    if args.model is None or args.sw_layer is None or args.sw_rows is None:
        p.error("--model, --sw_layer, and --sw_rows are required unless --replot is set")

    # ── Load model ────────────────────────────────────────────────────────────
    cfg_path = Path(args.configs_dir) / f"{args.model}.yaml"
    config   = yaml.safe_load(cfg_path.read_text())
    print(f"Model  : {config['model_id']}")
    print(f"SW     : layer {args.sw_layer}, rows {args.sw_rows}")

    from models.generator_wrapper import GeneratorWrapper
    wrapper   = GeneratorWrapper(config)
    wrapper.load()
    model     = wrapper.model.eval()
    tokenizer = wrapper.tokenizer
    device    = args.device

    down_proj = wrapper.get_target_module(args.sw_layer)
    print(f"down_proj shape: {down_proj.weight.shape}")

    # ── Collect sequences per task ────────────────────────────────────────────
    all_task_data = {}   # task_name → {all: {row: [vals]}, pos: {row:[vals]}, neg: {row:[vals]}}

    if args.mode == "gue":
        task_names = args.tasks
        for task in task_names:
            csv_path = Path(args.gue_root) / task / "test.csv"
            if not csv_path.exists():
                print(f"  [WARN] Missing: {csv_path}  — skipping")
                continue
            print(f"\n[{task}] sampling ...")
            sampled = _sample_gue_task(str(csv_path), args.n_per_class, rng)
            pos_seqs, neg_seqs = _pos_neg_split(sampled["by_class"], sampled["labels"])

            print(f"  all={len(sampled['all'])}  pos={len(pos_seqs)}  neg={len(neg_seqs)}")

            act_all = measure_task_activations(model, tokenizer, down_proj,
                                               sampled["all"], args.sw_rows, device)
            act_pos = measure_task_activations(model, tokenizer, down_proj,
                                               pos_seqs, args.sw_rows, device) if pos_seqs else {}
            act_neg = measure_task_activations(model, tokenizer, down_proj,
                                               neg_seqs, args.sw_rows, device) if neg_seqs else {}

            for r in args.sw_rows:
                v = act_all.get(r, [])
                print(f"  row {r:4d}: mean={np.mean(v):.4f}  std={np.std(v):.4f}  n={len(v)}")

            all_task_data[task] = {
                "all": {str(r): act_all.get(r, []) for r in args.sw_rows},
                "pos": {str(r): act_pos.get(r, []) for r in args.sw_rows},
                "neg": {str(r): act_neg.get(r, []) for r in args.sw_rows},
            }

    else:  # BED mode
        assert args.fasta,        "--fasta required in BED mode"
        assert args.bed_tasks,    "--bed_tasks required in BED mode"
        assert args.bed_negative, "--bed_negative required in BED mode"

        fasta    = _open_fasta(args.fasta)
        neg_bed  = Path(args.bed_dir) / args.bed_negative
        task_names = []

        for spec in args.bed_tasks:
            name, bed_file = spec.rsplit(":", 1)
            pos_bed = Path(args.bed_dir) / bed_file
            print(f"\n[{name}] sampling {args.n_per_class} pos + {args.n_per_class} neg ...")
            sampled = _sample_bed_task(str(pos_bed), str(neg_bed), fasta,
                                       args.n_per_class, args.window_bp, rng)

            print(f"  all={len(sampled['all'])}  pos={len(sampled['by_class'].get(1,[]))}  neg={len(sampled['by_class'].get(0,[]))}")

            pos_seqs, neg_seqs = sampled["by_class"].get(1, []), sampled["by_class"].get(0, [])
            act_all = measure_task_activations(model, tokenizer, down_proj,
                                               sampled["all"], args.sw_rows, device)
            act_pos = measure_task_activations(model, tokenizer, down_proj,
                                               pos_seqs, args.sw_rows, device)
            act_neg = measure_task_activations(model, tokenizer, down_proj,
                                               neg_seqs, args.sw_rows, device)

            for r in args.sw_rows:
                v = act_all.get(r, [])
                print(f"  row {r:4d}: mean={np.mean(v):.4f}  std={np.std(v):.4f}  n={len(v)}")

            all_task_data[name] = {
                "all": {str(r): act_all.get(r, []) for r in args.sw_rows},
                "pos": {str(r): act_pos.get(r, []) for r in args.sw_rows},
                "neg": {str(r): act_neg.get(r, []) for r in args.sw_rows},
            }
            task_names.append(name)

    if not all_task_data:
        print("[ERROR] No tasks processed.")
        sys.exit(1)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_data = {
        "model":     config["model_id"],
        "sw_layer":  args.sw_layer,
        "sw_rows":   args.sw_rows,
        "task_order": task_names,
        "tasks":     all_task_data,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"\nResults → {args.out}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot(out_data, args.sw_rows, task_names, args.sw_layer, args.plot)


if __name__ == "__main__":
    main()
