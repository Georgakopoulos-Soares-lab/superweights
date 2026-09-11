#!/usr/bin/env python3
"""
E9 Figure — DNABERT-2 F3 pairwise interaction terms (|Gamma|), ranked, with the
independently-discovered critical pair (L9/r264, L9/r294) highlighted. It was never
given special treatment during fitting; this figure is the visual form of the #1-of-45
retrospective check (H5) reported in RESULTS.md Sec.11.

Source: experiments/E9_mechanistic_tomography/fit_results_dnabert2.json (F3.gamma_pairs).
Output: figures/output/fig_e9_pair_interaction_map.{png,pdf}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E9 = HERE.parents[0] / "E9_mechanistic_tomography"
sys.path.insert(0, str(ROOT / "experiments" / "figures"))
from _figstyle import apply_style, panel_label  # noqa: E402

CRITICAL = "(9, 264)x(9, 294)"
TOP_K = 15


def main():
    apply_style()
    fit = json.loads((E9 / "fit_results_dnabert2.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))

    for ax, eps, label in zip(axes, ["0.5", "1.0"], ["A", "B"]):
        fe = fit["by_epsilon"][eps]
        gp = fe["F3"]["gamma_pairs"]
        n_pairs = fe["F3"]["n_pairs"]
        rank = fe["F3"]["critical_pair_rank_by_abs_gamma"]
        gamma_crit = fe["F3"]["critical_pair_gamma"]

        items = sorted(gp.items(), key=lambda kv: -abs(kv[1]))[:TOP_K]
        names = [k.replace("x", " x ") for k, _ in items]
        vals = [v for _, v in items]
        colors = ["#c0392b" if k == CRITICAL else "#8fa8c9" for k, _ in items]

        y = np.arange(len(items))[::-1]
        ax.barh(y, vals, color=colors, height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.3)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_xlabel(r"pair coefficient $\Gamma$ (F3, ridge)")
        ax.set_title(rf"$\epsilon$={eps}   (known pair rank {rank}/{n_pairs}, "
                     rf"$\Gamma$={gamma_crit:.3f})", fontsize=7.5, loc="left", color="0.3")
        panel_label(ax, label)

    fig.tight_layout()
    out = HERE / "output" / "fig_e9_pair_interaction_map"
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    print(f"saved -> {out}.png / .pdf")


if __name__ == "__main__":
    main()
