"""
scripts/analysis/plot_fig2_generator_panels_AD.py
----------------------------------------------------
T0.1 — Fig 2 panels A-D: GENERator EUK/PROK activation lifecycle (A/B) and
per-layer ||U_k||_F bar at the step-up layer (C/D), ranking the SW row(s)
among all output coordinates.

Inputs: results/mechanism/fig2_generator_lifecycle_{euk,prok}.json
Output: paper/media/image_fig2_panels_AD.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _figstyle import apply_style, panel_label

apply_style()

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "mechanism"
OUT_PNG = ROOT / "paper" / "media" / "image_fig2_panels_AD.png"
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

CASES = [("euk", "GENERator EUK", 0), ("prok", "GENERator PROK", 1)]

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))

for tag, label, col in CASES:
    d = json.loads((RES / f"fig2_generator_lifecycle_{tag}.json").read_text())
    sw_layer = d["sw_layer"]
    sw_rows = d["sw_rows"]
    layers = d["panel_AB_activation_lifecycle"]["layers"]
    mean_sw = np.array(d["panel_AB_activation_lifecycle"]["mean_post_residual_sw"])   # (n_layers, n_rows)
    top3 = np.array(d["panel_AB_activation_lifecycle"]["mean_post_residual_top3"])     # (n_layers, 3)

    # --- panel A/B: activation lifecycle ---
    ax = axes[0, col]
    for j, r in enumerate(sw_rows):
        ax.plot(layers, mean_sw[:, j], marker="o", ms=3, lw=1.3, label=f"SW row {r}")
    ax.plot(layers, top3[:, 0], lw=1.0, ls="--", color="gray", label="global top-1 channel")
    ax.axvline(sw_layer, color="crimson", lw=0.8, ls=":", alpha=0.7)
    ax.set_yscale("symlog")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean max |post-residual h|")
    ax.legend(loc="upper left", frameon=False)
    panel_label(ax, "AB"[col])
    ax.set_title(label, fontsize=9, loc="right", color="0.4")

    # --- panel C/D: Uk bar at step-up layer ---
    ax2 = axes[1, col]
    frob = np.array(d["panel_CD_uk_bar"]["frob_norm_uk"])
    order = np.argsort(frob)[::-1]
    top_n = 40
    shown = order[:top_n]
    colors = ["crimson" if i in sw_rows else "0.75" for i in shown]
    ax2.bar(range(top_n), frob[shown], color=colors, width=0.9)
    for j, r in enumerate(sw_rows):
        rank = d["panel_CD_uk_bar"]["sw_row_ranks"][str(r)]["rank"]
        if rank <= top_n:
            ax2.annotate(f"row {r}\nrank {rank}", (rank - 1, frob[r]),
                         textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=7, color="crimson")
        else:
            ax2.text(0.98, 0.9 - 0.1 * j,
                      f"row {r}: rank {rank}/{d['panel_CD_uk_bar']['d_model']} (off-scale)",
                      transform=ax2.transAxes, ha="right", fontsize=7, color="crimson")
    ax2.set_xlabel(f"output coordinate rank (top {top_n}, layer {sw_layer})")
    ax2.set_ylabel(r"$\|U_k\|_F$")
    panel_label(ax2, "CD"[col])

fig.tight_layout()
fig.savefig(OUT_PNG)
fig.savefig(OUT_PNG.with_suffix(".pdf"))
print(f"Saved -> {OUT_PNG} (+ .pdf)")
