"""
scripts/analysis/plot_compression_multimodel.py
-------------------------------------------------
Multi-model overlay of compression-sweep pruning curves.

Combines results from DNABERT-2, GENERator-EUK, GENERator-PROK, and EVO2
onto a single figure so sensitivity curves can be compared across models.

All metrics are normalised to % of baseline retained (0% = full degradation,
100% = no change), allowing comparison across tasks and model families.

For each model the plot shows:
  • random criterion  (grey band: mean ± 1 std)
  • prox_far          (green solid: most "safe" rows removed first)
  • prox_near         (orange solid: SW-neighbourhood removed first)
  • l1_low            (blue solid: smallest magnitude rows first)
  • SW-only marker    (crimson star: accuracy when ONLY the SW rows are zeroed)

Usage (from repo root):
    python scripts/analysis/plot_compression_multimodel.py \\
        --inputs \\
            dnabert2:results/compression_sweep_dnabert2_prom_core_notata.json \\
            generator:results/compression_sweep_generator_prom_core_notata.json \\
            generator_prokaryote:results/compression_sweep_generator_prokaryote_prom_core_notata.json \\
            evo2:results/compression_sweep_evo2_prom_core_notata.json \\
        --out results/compression_multimodel_prom_core_notata.png \\
        --metric accuracy

Arguments
---------
--inputs    List of  label:path  pairs (colon-separated).
            `label` is used as the model display name in the legend.
--metric    accuracy | mcc  (default: accuracy)
--out       Output PNG path.
--title     Optional figure title override.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Per-model colour palette ──────────────────────────────────────────────────
_MODEL_COLORS = {
    "dnabert2":               "#2166AC",   # deep blue
    "generator":              "#1A9641",   # dark green
    "generator_prokaryote":   "#D95F02",   # burnt orange
    "generator_prokaryote_1b":"#FC8D59",   # light orange
    "evo2":                   "#7B2D8B",   # purple
    # Fallback palette for arbitrary labels
}
_FALLBACK_COLORS = ["#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
                    "#42D4F4", "#F032E6", "#BFEF45", "#469990"]

_MODEL_DISPLAY = {
    "dnabert2":               "DNABERT-2",
    "generator":              "GENERator EUK 3B",
    "generator_prokaryote":   "GENERator PROK 3B",
    "generator_prokaryote_1b":"GENERator PROK 1B",
    "evo2":                   "EVO2 7B",
}

_CRIT_STYLES = {
    "random":    dict(ls=":",  lw=1.8, alpha=0.9, label_suffix=" random"),
    "prox_far":  dict(ls="--", lw=1.8, alpha=0.9, label_suffix=" prox-far"),
    "prox_near": dict(ls="-",  lw=2.2, alpha=0.9, label_suffix=" prox-near"),
    "l1_low":    dict(ls="-.", lw=1.8, alpha=0.9, label_suffix=" L1-low"),
}

# Only these criteria are drawn by default (keeps the figure readable)
_DEFAULT_CRITERIA = ["random", "l1_low", "prox_near"]


def _pct_retained(vals, baseline):
    """Convert absolute metric values to % of baseline retained."""
    return [v / baseline * 100 for v in vals]


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_multimodel(entries, metric: str, out_png: str, title: str = None,
                    criteria=None):
    """
    Parameters
    ----------
    entries : list of (label, data_dict)
        `label` is the display label; `data_dict` is the loaded JSON.
    metric  : "accuracy" | "mcc"
    out_png : output file path
    criteria: list of criterion names to draw (default: _DEFAULT_CRITERIA)
    """
    if criteria is None:
        criteria = _DEFAULT_CRITERIA

    ylabel_map = {"accuracy": "Accuracy retained (% of baseline)",
                  "mcc":      "MCC retained (% of baseline)"}
    ylabel = ylabel_map.get(metric, f"{metric} retained (% of baseline)")

    fig, ax = plt.subplots(figsize=(11, 6))

    # Horizontal 100% line = no degradation
    ax.axhline(100, color="black", lw=1.0, ls="-", zorder=1, label="Baseline (100%)")

    fallback_idx = 0
    sw_handles = []   # for a secondary legend entry for SW-only markers

    for i, (label, data) in enumerate(entries):
        # Determine colour
        model_key = data.get("model", label)
        color = _MODEL_COLORS.get(model_key) or _MODEL_COLORS.get(label)
        if color is None:
            color = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
            fallback_idx += 1

        display = _MODEL_DISPLAY.get(model_key, label)
        baseline_val = data["baseline"][metric]
        fracs = data["fracs"]
        curves = data.get("curves", {})

        # ── Pruning curves ────────────────────────────────────────────────────
        first_line = True
        for crit in criteria:
            if crit not in curves:
                continue
            style = _CRIT_STYLES.get(crit, dict(ls="-", lw=1.5, alpha=0.8, label_suffix=""))

            if crit == "random":
                means = [p.get(f"{metric}_mean", p.get(metric, baseline_val)) for p in curves[crit]]
                stds  = [p.get(f"{metric}_std",  0.0)                         for p in curves[crit]]
                xs    = [p["frac"] for p in curves[crit]]
                pct_m = _pct_retained(means, baseline_val)
                pct_u = _pct_retained([m + s for m, s in zip(means, stds)], baseline_val)
                pct_l = _pct_retained([m - s for m, s in zip(means, stds)], baseline_val)
                lbl = display if first_line else f"_{display}"  # "_" = no legend duplicate
                line, = ax.plot(xs, pct_m, color=color, ls=style["ls"], lw=style["lw"],
                                alpha=style["alpha"], label=lbl, zorder=3)
                ax.fill_between(xs, pct_l, pct_u, color=color, alpha=0.12, zorder=2)
            else:
                xs  = [p["frac"]   for p in curves[crit]]
                ys  = [p[metric]   for p in curves[crit]]
                pct = _pct_retained(ys, baseline_val)
                lbl = display if first_line else f"_{display}"
                ax.plot(xs, pct, color=color, ls=style["ls"], lw=style["lw"],
                        alpha=style["alpha"], label=lbl, zorder=3)

            first_line = False

        # ── SW-only star marker ───────────────────────────────────────────────
        sw_only = data.get("sw_only")
        if sw_only is not None:
            sw_frac = sw_only.get("frac_pct",
                      sw_only.get("n_sw", 0) / max(data.get("n_total", 1), 1) * 100)
            sw_val  = sw_only[metric]
            sw_pct  = sw_val / baseline_val * 100
            sc = ax.scatter(
                [sw_frac], [sw_pct],
                marker="*", s=320, color=color, zorder=6,
                edgecolors="white", linewidths=0.8,
            )
            # Annotate with a short offset
            ax.annotate(
                f"{display}\nSW-only\n{sw_pct:.1f}%",
                xy=(sw_frac, sw_pct),
                xytext=(sw_frac * 2.5, sw_pct - 2),
                fontsize=7, color=color, zorder=7,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )
            sw_handles.append((display, sw_frac, sw_pct))

    # ── Criterion style legend (secondary) ────────────────────────────────────
    from matplotlib.lines import Line2D
    style_patches = [
        Line2D([0], [0], color="black", ls=_CRIT_STYLES["random"]["ls"],
               lw=1.5, label="Random (mean ± 1 std)"),
        Line2D([0], [0], color="black", ls=_CRIT_STYLES["l1_low"]["ls"],
               lw=1.5, label="L1-low (small magnitude first)"),
        Line2D([0], [0], color="black", ls=_CRIT_STYLES["prox_near"]["ls"],
               lw=2.0, label="Prox-near (SW neighbourhood first)"),
        Line2D([0], [0], color="grey", marker="*", markersize=12,
               linestyle="None", label="SW-only (superweight rows)"),
    ]

    # Primary legend: model colours
    legend1 = ax.legend(loc="lower left", fontsize=9, title="Model", title_fontsize=9,
                        framealpha=0.9)
    ax.add_artist(legend1)
    # Secondary legend: line styles
    ax.legend(handles=style_patches, loc="upper right", fontsize=8,
              title="Pruning criterion", title_fontsize=8, framealpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Rows pruned (% of non-SW candidate pool)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="both", alpha=0.25)

    # Y-axis: show a dashed line at the lowest SW-only value for reference
    if sw_handles:
        min_sw_pct = min(x[2] for x in sw_handles)
        ax.axhline(min_sw_pct, color="crimson", lw=0.8, ls=":", alpha=0.6)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    else:
        task_labels = ", ".join(d.get("task", "?") for _, d in entries)
        ax.set_title(
            f"Compression sensitivity — {metric.upper()} retained\n"
            f"Task: {task_labels}  |  ★ = superweight rows zeroed",
            fontsize=11, fontweight="bold", pad=8,
        )

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_png}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Multi-model compression-sweep overlay plot")
    parser.add_argument(
        "--inputs", nargs="+", required=True, metavar="LABEL:PATH",
        help="Model entries as 'label:path/to/sweep.json'. "
             "Label is used for display; path points to the compression sweep JSON."
    )
    parser.add_argument("--metric",   default="accuracy", choices=["accuracy", "mcc"],
                        help="Metric to plot (default: accuracy)")
    parser.add_argument("--out",      default="results/compression_multimodel.png")
    parser.add_argument("--title",    default=None)
    parser.add_argument("--criteria", nargs="+",
                        default=_DEFAULT_CRITERIA,
                        choices=list(_CRIT_STYLES.keys()),
                        help="Which pruning criteria to draw (default: random l1_low prox_near)")
    args = parser.parse_args()

    entries = []
    for item in args.inputs:
        if ":" not in item:
            raise ValueError(f"Input must be 'label:path', got: {item!r}")
        label, path = item.split(":", 1)
        data = _load(path)
        entries.append((label, data))
        print(f"Loaded {label}: {path}")

    plot_multimodel(
        entries,
        metric=args.metric,
        out_png=args.out,
        title=args.title,
        criteria=args.criteria,
    )


if __name__ == "__main__":
    main()
