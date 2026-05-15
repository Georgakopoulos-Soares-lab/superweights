"""
Grouped bar chart comparing INT4 quantization cost vs SW marginal cost
for GENERator eukaryote and prokaryote models.

Two groups (EUK / PROK), each with two bars:
  - yu_all ΔPPL : all non-SW rows quantized
  - SW marginal cost : additional cost of quantizing the SW rows themselves

Log y-axis makes both visible simultaneously.

Usage:
    python scripts/analysis/plot_quantization_sw_comparison.py \
        --euk  results/whole_model_quant_generator_sw_comparison.json \
        --prok results/whole_model_quant_generator_prokaryote_sw_comparison.json \
        --out  results/quantization_sw_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


COLOR_YUALL  = "#4878CF"   # blue  — full INT4 cost
COLOR_SW_MAR = "#D65F5F"   # red   — SW marginal cost


def load(path):
    with open(path) as f:
        d = json.load(f)
    s = d["scopes"]["full"]
    return {
        "baseline":        s["baseline_ppl"],
        "yu_all_delta":    s["yu_all_delta"],
        "sw_marginal":     s["sw_marginal_cost"],
        "sw_fragility":    s["sw_fragility"]["delta_ppl"],
        "n_candidates":    s["n_candidates"],
        "n_sw":            s["n_sw_rows_in_scope"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--euk",  default="results/whole_model_quant_generator_sw_comparison.json")
    parser.add_argument("--prok", default="results/whole_model_quant_generator_prokaryote_sw_comparison.json")
    parser.add_argument("--out",  default="results/quantization_sw_comparison.png")
    args = parser.parse_args()

    euk  = load(args.euk)
    prok = load(args.prok)

    models     = ["GENERator\nEuk 3B", "GENERator\nProk 3B"]
    yu_all     = [euk["yu_all_delta"],  prok["yu_all_delta"]]
    sw_marg    = [euk["sw_marginal"],   prok["sw_marginal"]]
    baselines  = [euk["baseline"],      prok["baseline"]]
    n_cands    = [euk["n_candidates"],  prok["n_candidates"]]
    n_sws      = [euk["n_sw"],          prok["n_sw"]]

    x = np.array([0.0, 1.0])
    w = 0.28

    fig, ax = plt.subplots(figsize=(7, 5))

    bars_yu = ax.bar(x - w/2, yu_all,  w, color=COLOR_YUALL,  edgecolor="k",
                     linewidth=0.7, zorder=3, label="INT4 cost — all non-SW rows")
    bars_sw = ax.bar(x + w/2, sw_marg, w, color=COLOR_SW_MAR, edgecolor="k",
                     linewidth=0.7, zorder=3, label="SW marginal cost\n(adding SW rows to INT4 pool)")

    # Value labels
    for bar, val in zip(bars_yu, yu_all):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.6,
                f"+{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for bar, val in zip(bars_sw, sw_marg):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.6,
                f"+{val:.4f}", ha="center", va="bottom", fontsize=9,
                color=COLOR_SW_MAR, fontweight="bold")

    # Ratio annotations between pairs
    for i, (yu, sw) in enumerate(zip(yu_all, sw_marg)):
        ratio = yu / sw
        ax.annotate(
            f"×{ratio:.0f}\nlarger",
            xy=(x[i], (yu * sw) ** 0.5),   # geometric mean height — midpoint on log scale
            fontsize=8, ha="center", color="#444444",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.7),
        )

    ax.set_yscale("log")
    ax.set_ylim(1e-4, 5.0)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("ΔPPL after INT4 quantization (log scale)", fontsize=10)
    ax.set_title("SW rows are not precision-sensitive under INT4 quantization",
                 fontsize=10.5, fontweight="bold", pad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

    # Footnote with row counts
    fig.text(
        0.5, -0.04,
        f"EUK: {n_cands[0]:,} rows quantized, {n_sws[0]} SW rows  |  "
        f"PROK: {n_cands[1]:,} rows quantized, {n_sws[1]} SW rows  |  "
        "All projections (down/gate/up/q/k/v/o_proj), RTN INT4",
        ha="center", fontsize=7.5, color="#666666",
    )

    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(axis="y", which="both", linestyle=":", linewidth=0.5, color="#cccccc", zorder=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
