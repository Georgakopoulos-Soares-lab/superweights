#!/usr/bin/env python3
"""Fig. 4C replacement (round-2 Section 2): GC vs. random-direction scale c, and GC vs.
damage (native NLL), restyled to match manuscript figure conventions and combined into
one two-panel figure.

Restyles the substance already produced by audit/rederivations/scripts/section1_final_plots.py
(gc_vs_c.png / gc_vs_nll.png) using apply_style()/panel_label()/DOMAIN_COLOR instead of
the ad hoc tab:red/tab:blue/tab:green palette, drops plot titles (info moves to axis
labels/legend per _figstyle.py's stated principles), and combines the two plots into a
single side-by-side A/B figure.

Left panel (A):  GC vs. random-direction scale c, symlog x-axis (linthresh=0.01), with
                  row2371-full-ablation GC=0.3065 and untouched-baseline GC=0.4204 as
                  horizontal reference lines.
Right panel (B):  GC vs. damage (native NLL) as the organizing x-axis -- row2371's own
                  alpha-sweep, the 5 inert control rows, and the random-direction grid all
                  on shared axes, showing composition tracking damage monotonically.
                  Baseline (GC 0.4204) and the ablation point (GC 0.3065, NLL 8.754) are
                  marked explicitly, with the same "same zeroed/untouched state" overlap
                  annotations as the exploratory script (real coincident points, not
                  missing data).

Data: audit/rederivations/generator_random_direction_full.csv (already produced by
section1_final_plots.py; read directly, not recomputed).
Output:
  audit/rederivations/figures/fig4c_replacement.{png,pdf}
  audit/rederivations/figures/fig4_panel_c_source.csv
  experiments/figures/source_data/fig4_panel_c_source.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as ml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT2 = ROOT / "audit" / "round2"
FIG_DIR = AUDIT2 / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR2 = ROOT / "manuscript" / "figures" / "source_data"
SRC_DIR2.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from _figstyle import apply_style, panel_label  # noqa: E402

sys.path.insert(0, str(ROOT / "manuscript" / "figures"))
from _paper_encoding import DOMAIN_COLOR  # noqa: E402

BASELINE_GC = 0.4204
ABLATION_GC = 0.3065
ABLATION_NLL = 8.754319605827332
BASELINE_NLL = 6.38538052380085

SRC_CSV = AUDIT2 / "generator_random_direction_full.csv"


def fnum(s):
    return float(s) if s not in ("", None) else None


def main():
    apply_style()

    rows = list(csv.DictReader(open(SRC_CSV)))

    rd = [r for r in rows if r["series"] == "random_direction" and fnum(r["gc_mean"]) is not None]
    rd_c = [float(r["x_c_or_alpha"]) for r in rd]
    rd_gc = [float(r["gc_mean"]) for r in rd]
    rd_lo = [float(r["gc_ci_low"]) for r in rd]
    rd_hi = [float(r["gc_ci_high"]) for r in rd]
    rd_nll = [float(r["nll"]) for r in rd]

    r2371 = [r for r in rows if r["series"] == "row2371_own_alpha_sweep"]
    r2371.sort(key=lambda r: float(r["x_c_or_alpha"]))
    r2371_alpha = [float(r["x_c_or_alpha"]) for r in r2371]
    r2371_gc = [float(r["gc_mean"]) for r in r2371]
    r2371_nll = [float(r["nll"]) for r in r2371]

    ctrl = [r for r in rows if r["series"].startswith("control_row_")]
    ctrl_nll = [float(r["nll"]) for r in ctrl]
    ctrl_gc = [float(r["gc_mean"]) for r in ctrl]
    ctrl_labels = [r["series"] for r in ctrl]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)

    # --- Panel A: GC vs. random-direction scale c -------------------------------------
    yerr = [np.array(rd_gc) - np.array(rd_lo), np.array(rd_hi) - np.array(rd_gc)]
    axA.errorbar(rd_c, rd_gc, yerr=yerr, fmt="o-", color=DOMAIN_COLOR["genomic"],
                 markeredgecolor="black", markersize=6, lw=1.2, capsize=3, zorder=3,
                 label="random direction (matched location)")
    axA.axhline(ABLATION_GC, color="0.35", linestyle="--", linewidth=1.1,
                label=f"row 2371 full ablation (GC={ABLATION_GC:.4f})")
    axA.axhline(BASELINE_GC, color="black", linestyle=":", linewidth=1.1,
                label=f"untouched baseline (GC={BASELINE_GC:.4f})")
    axA.set_xscale("symlog", linthresh=0.01)
    axA.set_xlim(0, 9.5)  # data are all c>=0 (c=0 through c=8.0); start exactly at 0 rather
                            # than padding into negative-c space where no data exist and blank
                            # space would misleadingly read as missing data
    axA.set_ylim(0.285, 0.475)  # extra headroom above the baseline line for the legend
    axA.set_xlabel("random-direction scale $c$")
    axA.set_ylabel("generated GC fraction")
    axA.legend(fontsize=6.6, loc="upper left", frameon=True, facecolor="white",
               edgecolor="none", framealpha=0.9)
    panel_label(axA, "A")

    # --- Panel B: GC vs. damage (native NLL), all conditions on shared axes -----------
    axB.plot(r2371_nll, r2371_gc, "o-", color=DOMAIN_COLOR["text"], markeredgecolor="black",
             markersize=6, lw=1.2, alpha=0.85, zorder=3, label=r"row 2371 own $\alpha$-sweep")
    axB.plot(rd_nll, rd_gc, "s-", color=DOMAIN_COLOR["genomic"], markeredgecolor="black",
             markersize=6, lw=1.2, alpha=0.85, zorder=4, label="random direction (matched location)")

    axB.scatter([ABLATION_NLL], [ABLATION_GC], marker="*", s=130, color="0.35",
                edgecolor="black", linewidths=0.8, alpha=0.85, zorder=5)
    axB.scatter([BASELINE_NLL], [BASELINE_GC], marker="*", s=130, color="black",
                edgecolor="black", linewidths=0.8, alpha=0.85, zorder=5)
    # 5 inert control rows sit almost exactly at the untouched-baseline point (NLL~6.386,
    # GC~0.42) -- same untouched-row state as row2371's alpha=1.0 and the baseline star.
    # Drawn last, larger, unfilled, and on top so the ring is visible poking out from
    # behind the coincident star/circle markers rather than being fully occluded.
    axB.scatter(ctrl_nll, ctrl_gc, marker="^", facecolors="none", edgecolors="0.15",
                s=170, linewidths=1.3, zorder=6, label="5 inert control rows")
    axB.annotate("row2371 $\\alpha$=0 &\nrandom-dir $c$=0\n(same zeroed state)",
                 xy=(ABLATION_NLL, ABLATION_GC), xytext=(7.55, 0.255),
                 fontsize=6.3, ha="center", arrowprops=dict(arrowstyle="->", lw=0.6))
    axB.annotate("row2371 $\\alpha$=1.0 &\ncontrols (untouched)",
                 xy=(BASELINE_NLL, BASELINE_GC), xytext=(6.85, 0.365),
                 fontsize=6.3, ha="center", arrowprops=dict(arrowstyle="->", lw=0.6))

    axB.set_xlabel("NLL (damage pool)")
    axB.set_ylabel("generated GC fraction")
    axB.legend(fontsize=6.6, loc="upper right", frameon=False)
    panel_label(axB, "B")

    out = FIG_DIR / "fig4c_replacement"
    fig.savefig(out.with_suffix(".png")); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}.png / .pdf")

    # ---- source-data CSV (copy of generator_random_direction_full.csv, both locations)
    for dest in (FIG_DIR / "fig4_panel_c_source.csv", SRC_DIR2 / "fig4_panel_c_source.csv"):
        dest.write_text(SRC_CSV.read_text())
        print(f"wrote {dest} (copy of {SRC_CSV.name})")


if __name__ == "__main__":
    main()
