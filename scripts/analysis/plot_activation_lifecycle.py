"""
scripts/analysis/plot_activation_lifecycle.py
----------------------------------------------
Visualize the "rise-plateau-fall" lifecycle of super-weight channel magnitudes
across all Transformer layers of GENERator EUK and PROK.

Reproduces the spirit of Sun et al. 2026 (arXiv 2603.05498) Figure 1 for
genomic language models.

Two-panel figure per model (or 2×2 grid for both together):
  Top    — post-residual hidden state magnitude of the SW channel
           across the 30 layers (solid line = mean over sequences,
           shaded band = ±1 std)
  Bottom — per-block delta magnitude of the SW channel (block contribution
           to the residual stream), showing which layers are "step-up" and
           "step-down" blocks

Input
-----
  results/activation_lifecycle_generator.json
  results/activation_lifecycle_generator_prokaryote.json   (optional)

Output
------
  results/activation_lifecycle.png
  results/activation_lifecycle_euk.png   (single-model fallback)
  results/activation_lifecycle_prok.png

Usage
-----
  python scripts/analysis/plot_activation_lifecycle.py
  python scripts/analysis/plot_activation_lifecycle.py --euk_only
  python scripts/analysis/plot_activation_lifecycle.py --prok_only
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_COLORS = {
    "sw0":   "#D62728",   # red — primary SW row
    "sw1":   "#FF7F0E",   # orange — secondary SW row
    "top3":  "#AAAAAA",   # grey — top-3 global channels
    "delta": "#1F77B4",   # blue — block delta
    "stepup_band":   "#FFDDC1",
    "stepdown_band": "#C1D4FF",
}

_LABELS = {
    "generator":            "GENERator EUK 3B",
    "generator_prokaryote": "GENERator PROK 3B",
}


def _load(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _plot_one(data: dict, axes, title: str):
    """
    Fill in two matplotlib Axes (top=post-residual, bottom=delta) for one model.
    """
    ax_top, ax_bot = axes
    n_layers = len(data["layers"])
    x = np.arange(n_layers)

    # ── per-seq arrays: shape (n_seqs, n_layers, n_sw_rows) ──────────────────
    per_seq_post = np.array(data["per_seq_post_residual_sw"])   # (S, L, R)
    per_seq_delt = np.array(data["per_seq_block_delta_sw"])     # (S, L, R)
    sw_rows  = data["sw_rows"]
    sw_layer = data["sw_layer"]

    top3 = np.array(data["mean_post_residual_top3"])            # (L, 3)

    # ── TOP PANEL: post-residual ──────────────────────────────────────────────
    # grey: top-3 global channels (background reference)
    ax_top.fill_between(x, top3[:, 2], top3[:, 0],
                        color=_COLORS["top3"], alpha=0.20, label="Top-3 channels (global)")
    ax_top.plot(x, top3[:, 0], color=_COLORS["top3"], lw=0.8, ls="--")

    for j, (row, ckey) in enumerate(zip(sw_rows, ["sw0", "sw1"])):
        mean_j  = per_seq_post[:, :, j].mean(axis=0)  # (L,)
        std_j   = per_seq_post[:, :, j].std(axis=0)
        ax_top.plot(x, mean_j, color=_COLORS[ckey], lw=1.8,
                    label=f"SW row {row} (d_model dim {row})")
        ax_top.fill_between(x, mean_j - std_j, mean_j + std_j,
                            color=_COLORS[ckey], alpha=0.15)

    ax_top.set_yscale("log")
    ax_top.set_ylabel("Max |h| over tokens")
    ax_top.set_title(title, fontweight="bold", pad=6)
    ax_top.axvline(sw_layer, color="grey", ls=":", lw=0.8, alpha=0.7)
    ax_top.legend(loc="upper right", frameon=False)
    ax_top.set_xticks(x[::5])

    # ── BOTTOM PANEL: block delta ─────────────────────────────────────────────
    for j, (row, ckey) in enumerate(zip(sw_rows, ["sw0", "sw1"])):
        mean_j = per_seq_delt[:, :, j].mean(axis=0)
        std_j  = per_seq_delt[:, :, j].std(axis=0)
        ax_bot.plot(x, mean_j, color=_COLORS[ckey], lw=1.8,
                    label=f"SW row {row}")
        ax_bot.fill_between(x, np.maximum(mean_j - std_j, 1e-3),
                            mean_j + std_j,
                            color=_COLORS[ckey], alpha=0.15)

    ax_bot.set_yscale("log")
    ax_bot.set_ylabel("Max |Δh| over tokens")
    ax_bot.set_xlabel("Layer index")
    ax_bot.axvline(sw_layer, color="grey", ls=":", lw=0.8, alpha=0.7,
                   label=f"SW layer ({sw_layer})")
    ax_bot.legend(loc="upper right", frameon=False)
    ax_bot.set_xticks(x[::5])

    # Annotate peak layer in top panel
    # find the layer where SW row 0 magnitude is maximal
    mean0 = per_seq_post[:, :, 0].mean(axis=0)
    peak_layer = int(np.argmax(mean0))
    ax_top.annotate(f"Layer {peak_layer}\n(peak)",
                    xy=(peak_layer, mean0[peak_layer]),
                    xytext=(peak_layer + 2, mean0[peak_layer] * 1.5),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                    fontsize=7, color="black")

    # Annotate step-up and step-down regions
    delta0 = per_seq_delt[:, :, 0].mean(axis=0)
    if delta0.max() > 0:
        # step-up: layer with largest positive delta (early)
        # step-down: layer with largest delta in latter half
        early_half = delta0[:n_layers // 2]
        late_half  = delta0[n_layers // 2:]
        stepup_layer   = int(np.argmax(early_half))
        stepdown_layer = int(np.argmax(late_half)) + n_layers // 2

        ax_bot.annotate("Step-up", xy=(stepup_layer, delta0[stepup_layer]),
                        xytext=(stepup_layer + 1, delta0[stepup_layer] * 2),
                        arrowprops=dict(arrowstyle="->", color=_COLORS["sw0"], lw=0.8),
                        fontsize=7, color=_COLORS["sw0"])
        ax_bot.annotate("Step-down", xy=(stepdown_layer, delta0[stepdown_layer]),
                        xytext=(stepdown_layer - 4, delta0[stepdown_layer] * 2),
                        arrowprops=dict(arrowstyle="->", color="#1F77B4", lw=0.8),
                        fontsize=7, color="#1F77B4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--euk_only",  action="store_true")
    parser.add_argument("--prok_only", action="store_true")
    args = parser.parse_args()

    rdir = Path(args.results_dir)

    euk_data  = _load(rdir / "activation_lifecycle_generator.json")
    prok_data = _load(rdir / "activation_lifecycle_generator_prokaryote.json")

    if args.euk_only:
        prok_data = None
    if args.prok_only:
        euk_data  = None

    models_to_plot = []
    if euk_data is not None:
        models_to_plot.append(("generator", euk_data))
    if prok_data is not None:
        models_to_plot.append(("generator_prokaryote", prok_data))

    if not models_to_plot:
        print("No data files found. Run run_activation_lifecycle.py first.")
        return

    n_models = len(models_to_plot)
    fig, axes_all = plt.subplots(
        nrows=2, ncols=n_models,
        figsize=(5 * n_models, 6.5),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.38, wspace=0.35)

    for col, (model_key, data) in enumerate(models_to_plot):
        title = _LABELS.get(model_key, model_key)
        _plot_one(data, (axes_all[0, col], axes_all[1, col]), title)

    # Add panel labels
    for col in range(n_models):
        for row, label in enumerate(["A", "B"] if n_models == 1 else
                                     [f"{'AB'[row]}{col+1}" for row in range(2)]):
            axes_all[row, col].text(-0.12, 1.05, label,
                                    transform=axes_all[row, col].transAxes,
                                    fontsize=11, fontweight="bold", va="top")

    fig.text(0.5, 0.01,
             "Top panels: SW channel magnitude in post-residual hidden state. "
             "Bottom panels: per-block contribution to that channel.\n"
             "Vertical dotted line = SW layer. Shaded band = ±1 SD across 20 sequences.",
             ha="center", va="bottom", fontsize=7, color="#555555", wrap=True)

    if n_models == 2:
        out_path = rdir / "activation_lifecycle.png"
    elif args.euk_only or (euk_data and not prok_data):
        out_path = rdir / "activation_lifecycle_euk.png"
    else:
        out_path = rdir / "activation_lifecycle_prok.png"

    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved → {out_path}")

    # Also save single-model plots if we did both
    if n_models == 2:
        for model_key, data in models_to_plot:
            fig2, axes2 = plt.subplots(2, 1, figsize=(5.5, 6.5))
            fig2.subplots_adjust(hspace=0.38)
            _plot_one(data, axes2, _LABELS.get(model_key, model_key))
            fig2.text(0.5, 0.01,
                      "Top: SW channel magnitude in post-residual hidden state. "
                      "Bottom: per-block contribution.\n"
                      "Vertical dotted line = SW layer. Shaded = ±1 SD / 20 sequences.",
                      ha="center", va="bottom", fontsize=7, color="#555555")
            suffix = "euk" if "prok" not in model_key else "prok"
            p2 = rdir / f"activation_lifecycle_{suffix}.png"
            fig2.savefig(p2, bbox_inches="tight", dpi=150)
            print(f"Saved → {p2}")
            plt.close(fig2)

    plt.close(fig)


if __name__ == "__main__":
    main()
