"""
scripts/analysis/plot_figure3.py
---------------------------------
Figure 3 — Super-weight encoding and kingdom-asymmetric mechanism.

Panels:
  A  EUK  mean SW activation by hexamer composition class
  B  PROK same
  C  Causal hexamer scatter: SW activation (x) vs. KL divergence (y) for
     all 4,096 hexamers; left = EUK (r=+0.437), right = PROK (r=-0.710)
  D  Shuffle-control activation shifts (4 shuffle types) for EUK vs. PROK
  E  EUK  motif enrichment in top-500 SW-dependent hexamers (Fisher+BH)
  F  PROK same

Inputs (already on disk):
  results/sw_kmer_scan.json
  results/sw_hexamer_causal.json
  results/sw_shuffle_controls.json
  results/sw_kmer_motifs.json
  results/prokaryote/sw_kmer_scan.json
  results/prokaryote/sw_hexamer_causal.json
  results/prokaryote/sw_shuffle_controls.json
  results/prokaryote/sw_kmer_motifs.json

Usage:
    python scripts/analysis/plot_figure3.py
    python scripts/analysis/plot_figure3.py --out paper/media/image_fig3.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         8,
    "axes.labelsize":    8,
    "axes.titlesize":    9,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

_LABEL_KW = dict(fontsize=14, fontweight="bold", va="bottom", ha="left",
                 clip_on=False)
_REPO = Path(__file__).resolve().parent.parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Hexamer composition classification (matches Methods)
# ─────────────────────────────────────────────────────────────────────────────
def _classify(kmer: str) -> str:
    s = kmer.upper()
    if any(s.count(b) >= 5 for b in "ACGT"):
        return "homopolymer"
    at = s.count("A") + s.count("T")
    gc = s.count("G") + s.count("C")
    if at >= 5:
        return "AT-rich"
    if gc >= 5:
        return "GC-rich"
    if "CG" in s and s.count("CG") >= 2:
        return "CpG-rich"
    return "balanced"


_CLASS_ORDER = ["AT-rich", "GC-rich", "CpG-rich", "homopolymer", "balanced"]
_CLASS_COL   = {
    "AT-rich":     "#1f77b4",
    "GC-rich":     "#d62728",
    "CpG-rich":    "#9467bd",
    "homopolymer": "#ff7f0e",
    "balanced":    "#7f7f7f",
}


# ═════════════════════════════════════════════════════════════════════════════
# A & B — Composition-class activation
# ═════════════════════════════════════════════════════════════════════════════
def _draw_composition_bars(ax, scan_path, title):
    data = json.load(open(scan_path))
    rows = data["results"]
    groups = {c: [] for c in _CLASS_ORDER}
    for r in rows:
        groups[_classify(r["kmer"])].append(r["activation_mean"])

    means = np.array([np.mean(groups[c]) if groups[c] else 0.0 for c in _CLASS_ORDER])
    sems  = np.array([np.std(groups[c]) / np.sqrt(len(groups[c]))
                      if groups[c] else 0.0 for c in _CLASS_ORDER])
    ns    = [len(groups[c]) for c in _CLASS_ORDER]
    cols  = [_CLASS_COL[c] for c in _CLASS_ORDER]

    x = np.arange(len(_CLASS_ORDER))
    ax.bar(x, means, yerr=sems, color=cols, edgecolor="black", linewidth=0.5,
           capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(_CLASS_ORDER, ns)],
                       rotation=0, fontsize=6.5)
    ax.set_ylabel("Mean |SW activation|")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    # annotate GC corr
    gc = data.get("gc_activation_corr")
    if gc is not None:
        ax.text(0.02, 0.97, f"GC corr r = {gc:+.2f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))


# ═════════════════════════════════════════════════════════════════════════════
# C — Causal hexamer scatter
# ═════════════════════════════════════════════════════════════════════════════
def _draw_causal_scatter(ax, causal_path, title, color):
    data = json.load(open(causal_path))
    rows = data["results"]
    x = np.array([r["sw_activation"] for r in rows], dtype=float)
    y = np.array([r["kl_clean_ablated"] for r in rows], dtype=float)
    r = data.get("r_kl_sw_activation")

    ax.scatter(x, y, s=3, alpha=0.35, color=color, edgecolors="none",
               rasterized=True)
    # trend
    if np.std(x) > 0:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m * xs + b, color="black", lw=1.0, linestyle="--", alpha=0.7)
    ax.set_xlabel("SW activation")
    ax.set_ylabel("KL(clean ‖ ablated)")
    ax.set_title(title)
    ax.text(0.04, 0.95, f"r = {r:+.3f}\nn = {len(rows):,}",
            transform=ax.transAxes, fontsize=7.5, va="top", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="grey", lw=0.4))
    ax.grid(alpha=0.2)


# ═════════════════════════════════════════════════════════════════════════════
# D — Shuffle controls
# ═════════════════════════════════════════════════════════════════════════════
def _draw_shuffles(ax, shuffle_paths_labels):
    shuffles = ["mono", "dinuc", "trinuc", "kmer_block"]
    pretty   = {"mono": "mono", "dinuc": "dinuc", "trinuc": "trinuc",
                "kmer_block": "6-mer block"}
    n_models = len(shuffle_paths_labels)
    width = 0.36
    x = np.arange(len(shuffles))

    colors = ["#1f77b4", "#d62728"]
    for i, (path, label) in enumerate(shuffle_paths_labels):
        data = json.load(open(path))
        agg = data["aggregate"]
        means = np.array([agg[s]["mean"] for s in shuffles])
        sems  = np.array([agg[s]["std"] / np.sqrt(agg[s]["n"]) for s in shuffles])
        ps    = [agg[s]["p_two_sided"] for s in shuffles]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=sems,
                      color=colors[i], edgecolor="black", linewidth=0.5,
                      capsize=2.5, label=label)
        # significance stars
        for j, p in enumerate(ps):
            if p is None or np.isnan(p):
                continue
            star = ("***" if p < 1e-3 else "**" if p < 1e-2 else
                    "*"   if p < 5e-2 else "n.s.")
            y_top = means[j] + sems[j]
            ax.text(j + offset, y_top + 0.02 * abs(means).max() if abs(means).max() > 0 else y_top,
                    star, ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[s] for s in shuffles])
    ax.set_ylabel("Δ SW activation (shuffled − original)\n(normalised)")
    ax.set_title("Shuffle-control sensitivity")
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", alpha=0.25)


# ═════════════════════════════════════════════════════════════════════════════
# E & F — Motif enrichment
# ═════════════════════════════════════════════════════════════════════════════
_MOTIF_ORDER = [
    "splice_donor", "splice_acceptor", "TATA_box", "Kozak",
    "GC_box_Sp1", "CpG_rich", "CCAAT_box", "E_box", "AP1_TRE", "NFkB",
    "Shine_Dalgarno", "AT_rich", "homopolymer_run",
]


def _draw_motif_enrichment(ax, motif_path, title):
    data = json.load(open(motif_path))
    by_name = {m["motif"]: m for m in data["motif_results"]}
    motifs = [m for m in _MOTIF_ORDER if m in by_name]
    ors    = np.array([by_name[m]["odds_ratio"] for m in motifs])
    qs     = np.array([by_name[m]["p_adj"]     for m in motifs])

    log2or = np.log2(np.where(ors > 0, ors, np.nan))
    colors = ["#d62728" if q < 0.05 else "#999999" for q in qs]
    y = np.arange(len(motifs))
    ax.barh(y, log2or, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace("_", " ") for m in motifs], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("log2(odds ratio)")
    ax.set_title(title)
    # significance markers
    for yi, (q, OR) in enumerate(zip(qs, ors)):
        if q < 0.05:
            ax.text(np.log2(OR) + 0.05 * (1 if OR >= 1 else -1),
                    yi, f"q={q:.3f}", va="center",
                    ha="left" if OR >= 1 else "right",
                    fontsize=6, color="#d62728")
    ax.grid(axis="x", alpha=0.25)


# ═════════════════════════════════════════════════════════════════════════════
# Composite
# ═════════════════════════════════════════════════════════════════════════════
def build_figure(out_path: Path):
    fig = plt.figure(figsize=(11, 13))
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.30,
                           left=0.07, right=0.97, top=0.965, bottom=0.04)

    # Row 1 — composition bars
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    _draw_composition_bars(axA, _REPO / "results/sw_kmer_scan.json",
                           "GENERator EUK — hexamer composition")
    _draw_composition_bars(axB, _REPO / "results/prokaryote/sw_kmer_scan.json",
                           "GENERator PROK — hexamer composition")

    # Row 2 — causal scatter
    axC1 = fig.add_subplot(gs[1, 0])
    axC2 = fig.add_subplot(gs[1, 1])
    _draw_causal_scatter(axC1, _REPO / "results/sw_hexamer_causal.json",
                         "EUK — causal hexamer test (driver)", "#1f77b4")
    _draw_causal_scatter(axC2, _REPO / "results/prokaryote/sw_hexamer_causal.json",
                         "PROK — causal hexamer test (suppressive gate)",
                         "#d62728")

    # Row 3 — shuffle controls (single panel spanning both columns)
    axD = fig.add_subplot(gs[2, :])
    _draw_shuffles(axD, [
        (_REPO / "results/sw_shuffle_controls.json",            "EUK"),
        (_REPO / "results/prokaryote/sw_shuffle_controls.json", "PROK"),
    ])

    # Row 4 — motif enrichment
    axE = fig.add_subplot(gs[3, 0])
    axF = fig.add_subplot(gs[3, 1])
    _draw_motif_enrichment(axE, _REPO / "results/sw_kmer_motifs.json",
                           "EUK — motif enrichment in top-500 SW-dependent")
    _draw_motif_enrichment(axF, _REPO / "results/prokaryote/sw_kmer_motifs.json",
                           "PROK — motif enrichment in top-500 SW-dependent")

    # Panel labels
    for ax, lbl in [(axA, "A"), (axB, "B"),
                    (axC1, "C"), (axC2, ""),
                    (axD, "D"),
                    (axE, "E"), (axF, "F")]:
        if lbl:
            ax.text(-0.10, 1.05, lbl, transform=ax.transAxes, **_LABEL_KW)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"wrote {out_path.with_suffix('.pdf')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/media/image_fig3.png")
    args = parser.parse_args()
    build_figure(Path(args.out) if Path(args.out).is_absolute()
                 else _REPO / args.out)


if __name__ == "__main__":
    main()
