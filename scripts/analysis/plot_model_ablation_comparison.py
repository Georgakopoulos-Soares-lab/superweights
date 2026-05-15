"""
Dual-panel manuscript figure: super-weight detection and functional impact.

Left panel  — Scatter: peak activation (out_max, log y) vs normalised layer
              position (spike_layer / total_layers, x). Coloured by SW status.

Right panel — Horizontal bar chart: ΔPPL% after zeroing the super-row (log x).
              Random-row control shown as grey dots.

Usage:
    python scripts/analysis/plot_model_ablation_comparison.py \
        --ablation results/ablation_results.json \
        --sw_index results/super_weight_index.json \
        --out      results/model_ablation_comparison.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Model metadata: labels, layer counts, and any values not in the JSON files
# (Evo2's out_max lives in a free-text note field; MegaDNA's num_layers is None
#  in the config because it uses a custom architecture).
# ---------------------------------------------------------------------------
MODEL_META = {
    "generator":            {"label": "GENERator Euk 3B",  "num_layers": 30},
    "generator_prokaryote": {"label": "GENERator Prok 3B", "num_layers": 30},
    "dnabert2":             {"label": "DNABERT-2",         "num_layers": 12},
    "ntv3":                 {"label": "NTv3 650M",         "num_layers": 12},
    "evo2":                 {"label": "Evo2 7B",           "num_layers": 32,
                             "out_max_override": 552960, "spike_layer_override": 29},
    "hybridna":             {"label": "HybridNA 7B",       "num_layers": 32},
    "megadna":              {"label": "MegaDNA 145M",      "num_layers": 10},
}

# Exclude the 1B variant to keep the figure uncluttered
INCLUDE = ["generator", "generator_prokaryote", "dnabert2", "ntv3",
           "evo2", "hybridna", "megadna"]

# ΔPPL% threshold above which a model is considered SW-positive
SW_THRESHOLD = 5.0

COLOR_POS  = "#d62728"   # red   — SW positive
COLOR_NEG  = "#aec7e8"   # blue  — SW negative
COLOR_RAND = "#999999"   # grey  — random-row reference


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(ablation_path: str, sw_index_path: str) -> list:
    with open(ablation_path) as f:
        abl = json.load(f)
    with open(sw_index_path) as f:
        sw_idx = json.load(f)

    records = []
    for key in INCLUDE:
        if key not in abl:
            continue
        d    = abl[key]
        meta = MODEL_META[key]

        # Peak activation -------------------------------------------------------
        best_sw    = d.get("best_sw", {})
        out_max    = meta.get("out_max_override") or best_sw.get("out_max")
        spike_layer = meta.get("spike_layer_override") or best_sw.get("layer")

        # dnabert2 has no best_sw in ablation_results — use super_weight_index
        if out_max is None:
            entries = sw_idx.get(key, {}).get("results", [])
            if entries:
                best_e     = max(entries, key=lambda e: e["out_max"])
                out_max    = best_e["out_max"]
                spike_layer = best_e["layer"]

        num_layers  = meta["num_layers"]
        norm_layer  = spike_layer / num_layers if spike_layer is not None else None

        # ΔPPL ------------------------------------------------------------------
        delta_pct  = d["delta_pct"]
        rand_delta = d["delta_rand_pct"]

        records.append({
            "model":      key,
            "label":      meta["label"],
            "out_max":    out_max,
            "norm_layer": norm_layer,
            "delta_pct":  delta_pct,
            "rand_delta": rand_delta,
            "is_sw":      abs(delta_pct) > SW_THRESHOLD,
        })

    # Order: SW-positive first (by delta_pct desc), then SW-negative
    pos = sorted([r for r in records if r["is_sw"]],      key=lambda r: -r["delta_pct"])
    neg = sorted([r for r in records if not r["is_sw"]],  key=lambda r: -r["delta_pct"])
    return pos + neg


# ---------------------------------------------------------------------------
# Left panel — scatter: activation magnitude vs layer position
# ---------------------------------------------------------------------------
def _plot_scatter(ax, records):
    for r in records:
        if r["out_max"] is None or r["norm_layer"] is None:
            continue
        color = COLOR_POS if r["is_sw"] else COLOR_NEG
        ax.scatter(
            r["norm_layer"], r["out_max"],
            color=color, edgecolors="k", linewidths=0.7,
            s=130, zorder=4,
        )

        # Label offset
        ha  = "left"
        dx  = 0.03
        dy  = 1.0      # multiplicative offset on log scale
        if r["norm_layer"] > 0.80:
            ha = "right"
            dx = -0.03
        # Evo2 and GENERator Prok are close — nudge Evo2 up
        if r["model"] == "evo2":
            dy = 2.5
        ax.annotate(
            r["label"],
            xy=(r["norm_layer"], r["out_max"]),
            xytext=(r["norm_layer"] + dx, r["out_max"] * dy),
            fontsize=8, ha=ha, va="center",
        )

    # Evo2 callout arrow
    for r in records:
        if r["model"] == "evo2":
            ax.annotate(
                "Large spike,\nno effect\n(SSM arch.)",
                xy=(r["norm_layer"], r["out_max"]),
                xytext=(r["norm_layer"] - 0.30, r["out_max"] / 30),
                fontsize=7.5, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8),
            )

    ax.set_yscale("log")
    ax.set_xlim(-0.06, 1.15)
    ax.set_ylim(20, 5e6)
    ax.set_xlabel("Normalised layer position  (spike layer / total layers)", fontsize=9)
    ax.set_ylabel("Peak activation — out_max  (log scale)", fontsize=9)
    ax.set_title("A   Detection: activation magnitude", fontsize=10,
                 fontweight="bold", loc="left")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pos_patch = mpatches.Patch(color=COLOR_POS, label=f"SW-positive (ΔPPL > {SW_THRESHOLD:.0f}%)")
    neg_patch = mpatches.Patch(color=COLOR_NEG, label="SW-negative")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, frameon=False, loc="upper center")


# ---------------------------------------------------------------------------
# Right panel — horizontal bar chart: ΔPPL%
# ---------------------------------------------------------------------------
def _plot_bars(ax, records):
    labels     = [r["label"]     for r in records]
    deltas     = [r["delta_pct"] for r in records]
    rand_deltas= [r["rand_delta"] for r in records]
    colors     = [COLOR_POS if r["is_sw"] else COLOR_NEG for r in records]

    y = np.arange(len(records))

    # Clip to a tiny positive floor so log scale works; track original sign
    plot_deltas = [max(abs(d), 5e-4) for d in deltas]

    ax.barh(y, plot_deltas, color=colors, edgecolor="k", linewidth=0.6,
            height=0.6, zorder=2)

    # Random-row control dots
    ax.scatter(
        [max(abs(rd), 5e-4) for rd in rand_deltas], y,
        color=COLOR_RAND, edgecolors="k", linewidths=0.5,
        s=45, zorder=4, label="Random-row control",
    )

    # Value labels
    for i, (raw, clipped) in enumerate(zip(deltas, plot_deltas)):
        if abs(raw) < 0.01:
            txt = f"{raw:+.4f}%"
        elif abs(raw) < 100:
            txt = f"{raw:+.2f}%"
        else:
            txt = f"+{raw:,.0f}%"
        ax.text(clipped * 2.0, i, txt, va="center", ha="left", fontsize=7.5)

    # Random-control shaded region
    max_rand = max(abs(r["rand_delta"]) for r in records)
    ax.axvspan(5e-4, max(max_rand * 8, 0.5), alpha=0.07, color=COLOR_RAND, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(3e-4, 2e6)
    ax.set_xlabel("ΔPPL%  after super-row ablation  (log scale)", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_title("B   Functional impact: perplexity change", fontsize=10,
                 fontweight="bold", loc="left")
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # Separator between SW-positive and SW-negative rows
    n_pos = sum(1 for r in records if r["is_sw"])
    if 0 < n_pos < len(records):
        ax.axhline(n_pos - 0.5, color="k", linewidth=0.8, linestyle="--",
                   zorder=3, alpha=0.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", default="results/ablation_results.json")
    parser.add_argument("--sw_index", default="results/super_weight_index.json")
    parser.add_argument("--out",      default="results/model_ablation_comparison.png")
    args = parser.parse_args()

    records = load_data(args.ablation, args.sw_index)

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    fig.subplots_adjust(wspace=0.38)

    _plot_scatter(ax_left, records)
    _plot_bars(ax_right, records)

    fig.suptitle(
        "Super-weight behaviour across genomic language models",
        fontsize=11.5, fontweight="bold", y=1.02,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
