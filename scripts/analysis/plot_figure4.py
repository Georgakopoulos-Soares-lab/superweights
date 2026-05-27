"""Figure 4 — GUE functional consequences of SW ablation.

Panel A — DNABERT-2 multiseed Δacc across 3 GUE tasks (3 seeds each) vs random-10 control.
Panel B — NTv3 splice/reconstructed across 5 seeds (paired bars: baseline MCC
          vs SW-ablated MCC). MCC is reported because splice is 3-class and
          imbalanced; seeds 4 and 5 collapse to majority-class prediction
          under SW ablation (Δacc small/positive, ΔMCC strongly negative).
Panel C — DNABERT-2 per-row Δacc on splice/reconstructed (each of the 10 SW
          rows ablated individually).

Inputs:
  results/gue_multiseed_results.json         (panel A)
  results/gue_multiseed_ntv3_splice.json     (panel B — 5 seeds merged)
  results/gue_per_row_ablation.json          (panel C)

Output:
  paper/media/image_fig4.png   (+ pdf)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT_PNG = ROOT / "paper" / "media" / "image_fig4.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

TASKS = [
    ("dnabert2/splice/reconstructed", "splice/\nreconstructed"),
    ("dnabert2/prom/prom_core_notata", "prom_core_\nnotata"),
    ("dnabert2/EMP/H3K4me3", "EMP/\nH3K4me3"),
]


def load(name):
    return json.load(open(RES / name))


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
    ax.bar(x - w / 2, sw_mean, w, yerr=sw_std, capsize=4,
           color="#c0392b", label="SW rows ablated", edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, rand_mean, w, yerr=rand_std, capsize=4,
           color="#7f8c8d", label="Random 10 rows", edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Δ accuracy (%)")
    ax.set_title("A. DNABERT-2 GUE Δacc — SW vs random control\n(3 seeds, mean ± std)",
                 fontsize=10, loc="left")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    for xi, p, m in zip(x, pvals, sw_mean):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.annotate(f"p={p:.4f}\n{star}",
                    xy=(xi - w / 2, m), xytext=(0, -22 if m < 0 else 6),
                    textcoords="offset points", ha="center", fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
def panel_B(ax):
    d = json.load(open(RES / "gue_multiseed_ntv3_splice.json"))
    block = d["ntv3/splice/reconstructed"]
    per_seed = sorted(block["per_seed"], key=lambda r: r["seed"])
    seeds = [r["seed"] for r in per_seed]
    base_mcc = [r["baseline"]["mcc"]   for r in per_seed]
    sw_mcc   = [r["sw_ablated"]["mcc"] for r in per_seed]
    delta_acc = [r["delta_acc"] for r in per_seed]

    x = np.arange(len(seeds))
    w = 0.38
    ax.bar(x - w / 2, base_mcc, w, color="#2c3e50",
           edgecolor="black", linewidth=0.6, label="Baseline MCC")
    sw_colors = ["#e67e22" if d_a > 0 else "#c0392b" for d_a in delta_acc]
    ax.bar(x + w / 2, sw_mcc, w, color=sw_colors,
           edgecolor="black", linewidth=0.6, label="SW-ablated MCC")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    ax.set_ylabel("MCC (splice/reconstructed)")

    agg = block["aggregate"]
    t = agg.get("t_stat_mcc_vs_zero", agg.get("t_stat_vs_zero"))
    p = agg.get("p_val_mcc",          agg.get("p_val"))
    sign_neg = agg.get("sign_negative_mcc", 0)
    ax.set_title("B. NTv3 splice — baseline vs SW-ablated MCC (5 seeds)\n"
                 f"ΔMCC = {agg['delta_mcc_mean']:+.3f} ± {agg['delta_mcc_std']:.3f},  "
                 f"t={t:.2f}  p={p:.4f},  sign-neg {sign_neg}/5",
                 fontsize=10, loc="left")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    for xi, d_a, sw in zip(x, delta_acc, sw_mcc):
        if d_a > 0:
            ax.annotate("majority-class\ncollapse",
                        xy=(xi + w/2, sw), xytext=(0, 14),
                        textcoords="offset points", ha="center",
                        fontsize=6.5, color="#a04000",
                        arrowprops=dict(arrowstyle="-", color="#a04000", lw=0.5))


# ─────────────────────────────────────────────────────────────────────────────
def panel_C(ax, perrow):
    rows = perrow["dnabert2/splice/reconstructed"]["per_row"]
    rows = sorted(rows, key=lambda r: r["delta_acc"])
    deltas = [r["delta_acc"] * 100 for r in rows]
    labs = [f"L{r['layer']}r{r['row']}" for r in rows]
    rand_mean = (perrow["dnabert2/splice/reconstructed"]["rand_mean"]["accuracy"]
                 - perrow["dnabert2/splice/reconstructed"]["baseline"]["accuracy"]) * 100

    x = np.arange(len(rows))
    colors = ["#c0392b" if d < -0.5 else "#e67e22" if d < 0 else "#27ae60" for d in deltas]
    ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(rand_mean, color="#2c3e50", lw=0.8, ls="--",
               label=f"random-1-row mean = {rand_mean:+.3f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Δ accuracy (%)")
    ax.set_title("C. DNABERT-2 per-row Δacc on splice/reconstructed",
                 fontsize=10, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def main():
    ms = load("gue_multiseed_results.json")
    perrow = load("gue_per_row_ablation.json")

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0],
                          hspace=0.55, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    panel_A(axA, ms)
    panel_B(axB)
    panel_C(axC, perrow)

    fig.suptitle("Figure 4 — GUE Functional Consequences of SW Ablation",
                 fontsize=13, y=0.995)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved: {OUT_PNG}")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
