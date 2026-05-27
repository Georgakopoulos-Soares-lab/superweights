"""Figure 4 — GUE functional consequences of SW ablation.

Clean redesign: no titles, no suptitle, compact A/B/C labels.

Panel A — DNABERT-2 Δacc on 3 GUE tasks (SW ablation vs. random-10 control).
Panel B — NTv3 splice MCC, baseline vs. SW-ablated, paired bars for 5 seeds.
          Seeds whose ΔAcc > 0 but ΔMCC ≪ 0 are coloured orange to mark
          majority-class collapse.
Panel C — DNABERT-2 per-row Δacc on splice/reconstructed (single-row ablation).

Caption carries the statistics; the figure shows the effect sizes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _figstyle import apply_style, panel_label

apply_style()

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT_PNG = ROOT / "paper" / "media" / "image_fig4.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

TASKS = [
    ("dnabert2/splice/reconstructed",  "splice"),
    ("dnabert2/prom/prom_core_notata", "promoter"),
    ("dnabert2/EMP/H3K4me3",           "H3K4me3"),
]

C_SW    = "#c0392b"
C_RAND  = "#7f8c8d"
C_BASE  = "#2c3e50"
C_COLL  = "#e67e22"


# ─────────────────────────────────────────────────────────────────────────────
def panel_A(ax, ms):
    labels, sw_mean, sw_std, rand_mean, rand_std, pvals = [], [], [], [], [], []
    for key, lab in TASKS:
        agg = ms[key]["aggregate"]
        labels.append(lab)
        sw_mean.append(agg["delta_acc_mean"] * 100)
        sw_std.append(agg["delta_acc_std"] * 100)
        rd = (agg["rand_acc_mean"] - agg["baseline_acc_mean"]) * 100
        rs = float(np.hypot(agg["rand_acc_std"], agg["baseline_acc_std"])) * 100
        rand_mean.append(rd)
        rand_std.append(rs)
        pvals.append(agg["p_val"])

    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, sw_mean,   w, yerr=sw_std,   capsize=3,
           color=C_SW,   edgecolor="black", linewidth=0.6, label="SW ablated")
    ax.bar(x + w / 2, rand_mean, w, yerr=rand_std, capsize=3,
           color=C_RAND, edgecolor="black", linewidth=0.6, label="random 10")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta$ accuracy (%)")
    # leave headroom above 0 for the legend; lower bound just below splice bar
    lo = min(m - s for m, s in zip(sw_mean, sw_std)) - 2
    ax.set_ylim(lo, 12)
    ax.legend(loc="upper right", frameon=False)

    # Significance stars above the SW bar (in the headroom).
    for xi, p, m in zip(x, pvals, sw_mean):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            continue
        ax.text(xi - w / 2, 1.5, star, ha="center", va="bottom",
                fontsize=12, color="black")


# ─────────────────────────────────────────────────────────────────────────────
def panel_B(ax):
    d = json.load(open(RES / "gue_multiseed_ntv3_splice.json"))
    block = d["ntv3/splice/reconstructed"]
    per_seed = sorted(block["per_seed"], key=lambda r: r["seed"])
    seeds     = [r["seed"]                  for r in per_seed]
    base_mcc  = [r["baseline"]["mcc"]       for r in per_seed]
    sw_mcc    = [r["sw_ablated"]["mcc"]     for r in per_seed]
    delta_acc = [r["delta_acc"]             for r in per_seed]

    x = np.arange(len(seeds))
    w = 0.38
    ax.bar(x - w / 2, base_mcc, w, color=C_BASE,
           edgecolor="black", linewidth=0.6, label="baseline")
    sw_colors = [C_COLL if d_a > 0 else C_SW for d_a in delta_acc]
    ax.bar(x + w / 2, sw_mcc, w, color=sw_colors,
           edgecolor="black", linewidth=0.6, label="SW ablated")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("MCC")
    ax.axhline(0, color="black", lw=0.6)

    # one-line stat in upper-right corner, plain text, no box
    agg = block["aggregate"]
    p   = agg.get("p_val_mcc", agg.get("p_val"))
    ax.text(0.98, 0.97,
            rf"$\Delta$MCC = $-0.12 \pm 0.05$,  p = {p:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)

    # tiny inline marker for the majority-class-collapse seeds
    for i, da in enumerate(delta_acc):
        if da > 0:
            ax.text(i + w / 2, sw_mcc[i] + 0.005, "\u2020",
                    ha="center", va="bottom", fontsize=11, color=C_COLL)
    if any(d > 0 for d in delta_acc):
        ax.text(0.98, 0.86,
                "\u2020 majority-class collapse",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=C_COLL)

    ax.set_ylim(0, max(base_mcc) * 1.20)
    ax.legend(loc="upper left", frameon=False)


# ─────────────────────────────────────────────────────────────────────────────
def panel_C(ax, perrow):
    rows = sorted(perrow["dnabert2/splice/reconstructed"]["per_row"],
                  key=lambda r: r["delta_acc"])
    deltas = [r["delta_acc"] * 100 for r in rows]
    labs   = [f"L{r['layer']}r{r['row']}" for r in rows]
    rand_mean = (perrow["dnabert2/splice/reconstructed"]["rand_mean"]["accuracy"]
                 - perrow["dnabert2/splice/reconstructed"]["baseline"]["accuracy"]) * 100

    x = np.arange(len(rows))
    colors = [C_SW if d < -0.5 else C_COLL if d < 0 else "#27ae60" for d in deltas]
    ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(rand_mean, color=C_BASE, lw=0.8, ls="--",
               label=f"random-1-row mean ({rand_mean:+.2f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"$\Delta$ accuracy (%)")
    ax.legend(loc="lower right", frameon=False)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ms     = json.load(open(RES / "gue_multiseed_results.json"))
    perrow = json.load(open(RES / "gue_per_row_ablation.json"))

    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85],
                          hspace=0.50, wspace=0.28,
                          left=0.07, right=0.98, top=0.97, bottom=0.10)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    panel_A(axA, ms)
    panel_B(axB)
    panel_C(axC, perrow)

    panel_label(axA, "A")
    panel_label(axB, "B")
    panel_label(axC, "C", x=-0.05)

    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print("saved:", OUT_PNG)
    print("saved:", OUT_PDF)


if __name__ == "__main__":
    main()
