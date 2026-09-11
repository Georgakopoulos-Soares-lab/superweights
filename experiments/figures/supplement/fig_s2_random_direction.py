#!/usr/bin/env python3
"""
Supplement S2 -- GENERator EUK random-direction / damage-tracking control (retired from
main-text Figure 4C, 2026-09-01).

This is the SAME panel that was main-text Figure 4C before the BOS-mediation experiment
(results/experiments/E_BOS_MEDIATION/) produced a substantially stronger, position-resolved mechanistic
result for the same row (now the new main-text Fig. 4C/D). This panel is not superseded --
it answers a different question (is the row-2371 phenotype specific to its learned weight
direction, or does any equally damaging perturbation at that location reproduce it?) and
remains the evidence behind the manuscript's "does not establish full direction-independence"
qualification. Retained unchanged, only relabeled a/b and moved out of the main sequence.

  a. Generated GC fraction vs. the scale c of a fixed random unit direction substituted for
     the row-2371 weight vector (c=0 recovers ablation), against the ablation and untouched
     baseline levels.
  b. Generated GC fraction vs. native-loss damage (NLL) for the row-2371 own alpha-sweep, the
     random-direction grid, and 5 inert same-layer control rows, all on shared axes --
     composition tracks damage magnitude, not which intervention produced it.

Source: audit/rederivations/generator_random_direction_full.csv (unchanged from the original
fig4_generator.py; see that file's git history for the pre-2026-09-01 version of this code).
Output: figures/supplement/fig_s2_random_direction.{png,pdf}
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent
PAPER_SALVAGE = FIGURES.parent
REPO_ROOT = PAPER_SALVAGE.parent
AUDIT2 = REPO_ROOT / "audit" / "rederivations"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(FIGURES))
from _figstyle import apply_style, panel_label  # noqa: E402
from _paper_encoding import DOMAIN_COLOR  # noqa: E402

ABLATION_GC = 0.3065
BASELINE_GC = 0.4204
ABLATION_NLL = 8.754319605827332


def main():
    apply_style()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    gen_full = list(csv.DictReader(open(AUDIT2 / "generator_random_direction_full.csv")))
    rd = [r for r in gen_full if r["series"] == "random_direction" and r["gc_mean"]]
    rd.sort(key=lambda r: float(r["x_c_or_alpha"]))
    rd_c = [float(r["x_c_or_alpha"]) for r in rd]
    rd_gc = [float(r["gc_mean"]) for r in rd]
    rd_lo = [float(r["gc_ci_low"]) for r in rd]
    rd_hi = [float(r["gc_ci_high"]) for r in rd]
    rd_nll = [float(r["nll"]) for r in rd]

    r2371 = [r for r in gen_full if r["series"] == "row2371_own_alpha_sweep"]
    r2371.sort(key=lambda r: float(r["x_c_or_alpha"]))
    r2371_nll = [float(r["nll"]) for r in r2371]
    r2371_gc = [float(r["gc_mean"]) for r in r2371]

    ctrl = [r for r in gen_full if r["series"].startswith("control_row_")]
    ctrl_nll = [float(r["nll"]) for r in ctrl]
    ctrl_gc = [float(r["gc_mean"]) for r in ctrl]

    # --- Panel a: GC vs. random-direction scale c -------------------------------------
    yerr = [np.array(rd_gc) - np.array(rd_lo), np.array(rd_hi) - np.array(rd_gc)]
    ax_a.errorbar(rd_c, rd_gc, yerr=yerr, fmt="o-", color=DOMAIN_COLOR["genomic"],
                  markeredgecolor="black", markersize=6, lw=1.2, capsize=3, zorder=3,
                  label="random direction (matched location)")
    ax_a.axhline(ABLATION_GC, color="0.35", linestyle="--", linewidth=1.1,
                 label=f"row 2371 full ablation (GC={ABLATION_GC:.4f})")
    ax_a.axhline(BASELINE_GC, color="black", linestyle=":", linewidth=1.1,
                 label=f"untouched baseline (GC={BASELINE_GC:.4f})")
    ax_a.set_xscale("symlog", linthresh=0.01)
    ax_a.set_xlim(0, 9.5)
    ax_a.set_ylim(0.285, 0.475)
    ax_a.set_xlabel(r"random-direction scale $c$")
    ax_a.set_ylabel("generated GC fraction")
    ax_a.legend(fontsize=6.5, loc="upper left", frameon=True, facecolor="white",
               edgecolor="none", framealpha=0.9)
    panel_label(ax_a, "a")

    # --- Panel b: GC vs. damage (native NLL), all conditions on shared axes -----------
    ax_b.plot(r2371_nll, r2371_gc, "o-", color=DOMAIN_COLOR["text"], markeredgecolor="black",
              markersize=6, lw=1.2, alpha=0.85, zorder=3, label=r"row 2371 own $\alpha$-sweep")
    ax_b.plot(rd_nll, rd_gc, "s-", color=DOMAIN_COLOR["genomic"], markeredgecolor="black",
              markersize=6, lw=1.2, alpha=0.85, zorder=4, label="random direction (matched location)")
    ax_b.scatter([ABLATION_NLL], [ABLATION_GC], marker="*", s=130, color="0.35",
                edgecolor="black", linewidths=0.8, alpha=0.85, zorder=5)
    ax_b.scatter(ctrl_nll, ctrl_gc, marker="^", color="white", edgecolor="black",
                s=70, alpha=0.95, zorder=6, label="5 inert control rows")
    ax_b.annotate("row2371 $\\alpha$=0 &\nrandom-dir $c$=0\n(same zeroed state)",
                  xy=(8.754, 0.3065), xytext=(7.9, 0.26), fontsize=6.0, ha="center",
                  arrowprops=dict(arrowstyle="->", lw=0.6))
    ax_b.annotate("row2371 $\\alpha$=1.0 &\ncontrols (untouched)",
                  xy=(6.39, 0.42), xytext=(6.9, 0.365), fontsize=6.0, ha="center",
                  arrowprops=dict(arrowstyle="->", lw=0.6))
    ax_b.set_xlabel("NLL (damage pool)")
    ax_b.set_ylabel("generated GC fraction")
    ax_b.legend(fontsize=6.5, loc="upper right")
    panel_label(ax_b, "b")

    out = HERE / "fig_s2_random_direction"
    fig.savefig(f"{out}.png"); fig.savefig(f"{out}.pdf")
    print(f"saved -> {out}.png / .pdf")


if __name__ == "__main__":
    main()
