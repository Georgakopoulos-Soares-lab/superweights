#!/usr/bin/env python3
"""Figure 5 — within-layer activation ratio vs single-row ablation damage, two decoders.

Left  GENERator-EUK-3B  L4  (genomic decoder, frozen candidate r2371)
Right SmolLM2-1.7B      L7  (text decoder,    frozen candidate r227)

36 rows per panel, log-spaced by activation-ratio RANK inside one layer, each ablated to
alpha=0, native LM NLL. Shading marks the detector's own accept region (ratio >= 5). The
inset reports Spearman rho over all rows, excluding the frozen candidate, and within each
regime separately -- the last being the test of whether the relation is graded or gated.

Deliberately minimal: no figure title, short axis labels, no legend (marker meanings and all
protocol detail live in the manuscript caption), top/right spines removed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/paper_closing"
TH = 5.0

PANELS = [("within_model_slope.json",              "A", "GENERator-EUK-3B  L4", "#1f4e79"),
          ("within_model_slope_smollm2_1.7b.json", "B", "SmolLM2-1.7B  L7",     "#8b2500")]

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
for ax, (fn, letter, name, col) in zip(axes, PANELS):
    d = json.loads((R / fn).read_text())
    rows = d["rows"]
    x = np.array([q["activation_ratio"] for q in rows])
    y = np.array([q["rel_delta"] for q in rows])
    isc = np.array([q["is_frozen_candidate"] for q in rows])

    ax.axvspan(TH, x.max() * 4, color="0.94", lw=0, zorder=0)
    ax.axhline(0, color="0.8", lw=0.6, zorder=1)
    ax.scatter(x[~isc], y[~isc], s=22, facecolor="white", edgecolor=col, lw=1.1, zorder=3)
    ax.scatter(x[isc], y[isc], s=110, marker="*", color=col, zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xlabel("activation ratio")
    ax.set_ylabel("relative loss increase")
    ax.set_xlim(x.min() / 3, x.max() * 4)
    # clip the symlog negative arm to the data; left free it expands into empty decades
    neg = min(y.min(), -1e-5)
    ax.set_ylim(neg * 3, y.max() * 4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    m, sub, sup = ~isc, x < TH, x >= TH
    r_all = stats.spearmanr(x, y).statistic
    r_exc = stats.spearmanr(x[m], y[m]).statistic
    r_sub = stats.spearmanr(x[sub], y[sub]).statistic
    r_sup = stats.spearmanr(x[sup], y[sup]).statistic
    # labels kept plain-text so the monospace column stays aligned; the caption states
    # that these are Spearman coefficients, so the symbol is not repeated per line
    ax.text(0.03, 0.97,
            f"all     {r_all:+.2f}\n"
            f"-cand   {r_exc:+.2f}\n"
            f"<5      {r_sub:+.2f}\n"
            f">=5     {r_sup:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            family="monospace", linespacing=1.4)
    ax.set_title(f"{letter}   {name}", fontsize=9, loc="left", pad=6)

fig.tight_layout(w_pad=2.0)
for ext in ("png", "pdf"):
    fig.savefig(R / f"fig_within_model_slope_2panel.{ext}", dpi=300, bbox_inches="tight")
print("saved ->", R / "fig_within_model_slope_2panel.png")
