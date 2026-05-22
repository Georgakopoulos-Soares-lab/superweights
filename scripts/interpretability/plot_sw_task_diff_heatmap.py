"""
scripts/interpretability/plot_sw_task_diff_heatmap.py
------------------------------------------------------
Re-plots the EUK + PROK super-weight task-activation heatmaps as
**difference** maps  (mean_pos − mean_neg)  instead of side-by-side
pos/neg columns.

One column per task.  Diverging colormap: red = higher in positive
(functional) sequences; blue = higher in negative (non-functional).

Usage:
    python scripts/interpretability/plot_sw_task_diff_heatmap.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent

# ── Short task labels (copied from parent script) ─────────────────────────────
def _short(task: str) -> str:
    parts = task.split("/")
    return (parts[-1]
            .replace("prom_", "")
            .replace("_notata", "_noTATA")
            .replace("_tata", "_TATA")
            .replace("reconstructed", "splice"))


# ── Build difference matrix ───────────────────────────────────────────────────
def _diff_matrix(data: dict):
    """
    Returns
    -------
    diff      : (n_sw_rows, n_tasks)  mean_pos - mean_neg
    se        : (n_sw_rows, n_tasks)  pooled SE  sqrt(var_pos/n_pos + var_neg/n_neg)
    short_labels : list[str]
    """
    sw_rows    = data["sw_rows"]
    task_names = data["task_order"]
    n_rows     = len(sw_rows)
    n_tasks    = len(task_names)

    diff = np.full((n_rows, n_tasks), np.nan)
    se   = np.full((n_rows, n_tasks), np.nan)

    for ti, task in enumerate(task_names):
        td = data["tasks"][task]
        for ri, row in enumerate(sw_rows):
            key   = str(row)
            pv    = td.get("pos", {}).get(key, [])
            nv    = td.get("neg", {}).get(key, [])
            if pv and nv:
                mp, sp, np_ = np.mean(pv), np.std(pv, ddof=1), len(pv)
                mn, sn, nn  = np.mean(nv), np.std(nv, ddof=1), len(nv)
                diff[ri, ti] = mp - mn
                se[ri, ti]   = np.sqrt(sp**2 / np_ + sn**2 / nn)

    short_labels = [_short(t) for t in task_names]
    return diff, se, short_labels


# ── Draw one heatmap panel ────────────────────────────────────────────────────
def _draw_diff_heatmap(ax, diff, se, short_labels, row_labels, layer, title):
    n_rows, n_tasks = diff.shape

    # Symmetric diverging scale around 0
    vmax = np.nanmax(np.abs(diff))
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im   = ax.imshow(diff, aspect="auto", cmap="RdBu_r", norm=norm)

    # Cell annotations: Δ value ± SE
    for ri in range(n_rows):
        for ti in range(n_tasks):
            d, s = diff[ri, ti], se[ri, ti]
            if np.isnan(d):
                continue
            txt_col = "white" if abs(d) > vmax * 0.55 else "black"
            sign    = "+" if d >= 0 else ""
            ax.text(ti, ri, f"{sign}{d:.0f}\n±{s:.0f}",
                    ha="center", va="center", fontsize=7, color=txt_col)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(short_labels, fontsize=8, rotation=40, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Δ mean |act|  (positive − negative)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Zero-line guide on colorbar
    cbar.ax.axhline(0, color="k", lw=0.8, ls="--")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    euk_json  = _REPO / "results/sw_task_activation_heatmap_generator_euk.json"
    prok_json = _REPO / "results/sw_task_activation_heatmap_generator_prok.json"
    out_path  = _REPO / "results/sw_task_diff_heatmap.png"

    with open(euk_json)  as f:  euk_data  = json.load(f)
    with open(prok_json) as f:  prok_data = json.load(f)

    euk_diff,  euk_se,  euk_sl  = _diff_matrix(euk_data)
    prok_diff, prok_se, prok_sl = _diff_matrix(prok_data)

    euk_rows  = euk_data["sw_rows"];  euk_layer  = euk_data["sw_layer"]
    prok_rows = prok_data["sw_rows"]; prok_layer = prok_data["sw_layer"]

    euk_rlabels  = [f"L{euk_layer}r{r}"  for r in euk_rows]
    prok_rlabels = [f"L{prok_layer}r{r}" for r in prok_rows]

    # Figure: one row per model, height proportional to number of SW rows
    n_euk  = len(euk_rows)
    n_prok = len(prok_rows)
    row_h  = 2.2          # inches per SW-row row
    fig_h  = row_h * (n_euk + n_prok) + 1.5   # + title/spacing

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(10, 0.80 * max(len(euk_sl), len(prok_sl))), fig_h),
        gridspec_kw={
            "hspace":       0.55,
            "height_ratios": [n_euk, n_prok],
        },
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    _draw_diff_heatmap(
        axes[0], euk_diff, euk_se, euk_sl, euk_rlabels, euk_layer,
        "GENERator-EUK-3B — super-weight activation: functional − non-functional",
    )
    _draw_diff_heatmap(
        axes[1], prok_diff, prok_se, prok_sl, prok_rlabels, prok_layer,
        "GENERator-PROK-3B — super-weight activation: functional − non-functional",
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
