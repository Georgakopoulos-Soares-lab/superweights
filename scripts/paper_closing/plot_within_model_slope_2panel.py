#!/usr/bin/env python3
"""Figure 5 — within-layer activation ratio vs single-row ablation damage, two decoders.

Left  GENERator-EUK-3B  L4  (genomic decoder, frozen candidate r2371)
Right SmolLM2-1.7B      L7  (text decoder,    frozen candidate r227)

36 rows per panel, log-spaced by activation-ratio RANK inside one layer, each ablated to
alpha=0, native LM NLL. Shading marks the detector's own accept region (ratio >= 5). The
inset reports Spearman rho over all rows, excluding the frozen candidate, and within each
regime separately -- the last being the test of whether the relation is graded or gated.

Deliberately minimal: no figure title, short axis labels, top/right spines removed. A boxed
in-panel legend names both marker types AND the grey band -- without it the shading reads as
an arbitrary background rather than as the detector's own >=5 accept region.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
    # The negative arm is opened past the data on purpose: it yields an empty band at the
    # bottom of both panels for the legend, which otherwise sits on top of real points.
    neg = min(y.min(), -1e-5)
    ax.set_ylim(neg * 24, y.max() * 4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    m, sub, sup = ~isc, x < TH, x >= TH
    r_all = stats.spearmanr(x, y).statistic
    r_exc = stats.spearmanr(x[m], y[m]).statistic
    r_sub = stats.spearmanr(x[sub], y[sub]).statistic
    r_sup = stats.spearmanr(x[sup], y[sup]).statistic
    # labels kept plain-text so the monospace column stays aligned; the caption states
    # that these are Spearman coefficients, so the symbol is not repeated per line
    ax.text(0.035, 0.96,
            f"all     {r_all:+.2f}\n"
            f"-cand   {r_exc:+.2f}\n"
            f"<5      {r_sub:+.2f}\n"
            f">=5     {r_sup:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            family="monospace", linespacing=1.4,
            bbox=dict(fc="white", ec="0.75", lw=0.6, boxstyle="round,pad=0.3"))

    handles = [Line2D([], [], ls="", marker="o", mfc="white", mec=col, mew=1.1, ms=5,
                      label="swept row"),
               Line2D([], [], ls="", marker="*", color=col, ms=10,
                      label="candidate"),
               Patch(fc="0.94", ec="0.75", lw=0.5, label="ratio $\\geq$ 5")]
    # One row along the bottom: the opened negative arm leaves a band that is empty at every
    # x in both panels, so the box cannot cover a measurement. A lower-right block legend did
    # -- it hid GENERator's (531, -1.0e-4) row.
    ax.legend(handles=handles, loc="lower center", ncol=3, frameon=True, framealpha=0.95,
              edgecolor="0.75", fontsize=7, handlelength=1.1, handletextpad=0.5,
              borderpad=0.45, columnspacing=1.1, borderaxespad=0.5)
    ax.set_title(f"{letter}   {name}", fontsize=9, loc="left", pad=6)

fig.tight_layout(w_pad=2.0)

# ---- verify no legend or stats box covers a measurement ---------------------
# Four hand-placed-annotation iterations hid a real point once (GENERator's
# (531, -1.0e-4)); this makes the check mechanical instead of visual.
fig.canvas.draw()
bad = 0
for ax, (fn, letter, _n, _c) in zip(axes, PANELS):
    rows = json.loads((R / fn).read_text())["rows"]
    pts = ax.transData.transform([(q["activation_ratio"], q["rel_delta"]) for q in rows])
    boxes = {"legend": ax.get_legend().get_window_extent()}
    for t in ax.texts:
        boxes["stats"] = t.get_window_extent()
    for what, bb in boxes.items():
        hit = [rows[i]["row"] for i, (px, py) in enumerate(pts)
               if bb.x0 <= px <= bb.x1 and bb.y0 <= py <= bb.y1]
        if hit:
            print(f"  [OVERLAP] panel {letter} {what} covers row(s) {hit}"); bad += len(hit)
print("  overlap check:", "clean" if not bad else f"{bad} covered point(s)")
for ext in ("png", "pdf"):
    fig.savefig(R / f"fig_within_model_slope_2panel.{ext}", dpi=300, bbox_inches="tight")
print("saved ->", R / "fig_within_model_slope_2panel.png")
