"""Figure 5 — Compression implications of SW structure.

Clean redesign. Three panels, no titles, no suptitle.

A — Pruning sweep on DNABERT-2 prom_core_notata: ΔMCC vs % rows pruned,
    log-x, 5 selection criteria. Headline: near-SW prunes flat,
    far-SW collapses at 20 %.
B — Per-row INT4 ΔPPL on GENERator EUK: near-SW vs random pool at
    each target fraction. Dashed line = SW-only baseline.
C — Whole-model INT4 on extended 100 k-token probe, EUK & PROK grouped:
    naive, Yu-exempt, SW-only fragility, SW marginal cost. The story is
    the height contrast (naive >> SW-only), not the precise values.

Inputs unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _figstyle import apply_style, panel_label

apply_style()

ROOT = Path(__file__).resolve().parents[2]
RES  = ROOT / "results"
OUT_PNG = ROOT / "paper" / "media" / "image_fig5.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

CRIT_STYLES = {
    "prox_near": ("near SW",       "#27ae60", "o"),
    "prox_far":  ("far from SW",   "#c0392b", "s"),
    "l1_low":    ("lowest L1",     "#3498db", "^"),
    "l1_high":   ("highest L1",    "#e67e22", "v"),
    "random":    ("random",        "#7f8c8d", "x"),
}


def safe_load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


# ─── Panel A ───────────────────────────────────────────────────────────────
def panel_A(ax, sweep):
    fracs = sweep["fracs"]
    for crit, (lab, color, mk) in CRIT_STYLES.items():
        curve = sweep["curves"][crit]
        y = [c["delta_mcc_pct"] for c in curve]
        if crit == "random":
            ys = [c.get("mcc_std", 0) * 100 / sweep["baseline"]["mcc"]
                  for c in curve]
            ax.errorbar(fracs, y, yerr=ys, label=lab, color=color, marker=mk,
                        ms=5, lw=1.0, capsize=2)
        else:
            ax.plot(fracs, y, label=lab, color=color, marker=mk, ms=5, lw=1.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("% rows pruned")
    ax.set_ylabel(r"$\Delta$ MCC (%)")
    ax.legend(loc="lower left", frameon=False, ncol=2,
              handletextpad=0.4, columnspacing=1.0)
    ax.grid(True, alpha=0.3, which="both")


# ─── Panel B ───────────────────────────────────────────────────────────────
def panel_B(ax, gen_int4):
    sw   = gen_int4["sw_fragility"]
    near = gen_int4["near_sw"]
    rand = gen_int4["random"]
    fracs    = [c["frac"]       for c in near]
    near_dp  = [c["delta_ppl"]  for c in near]
    rand_dp  = [c["delta_ppl_mean"] for c in rand]
    rand_std = [c["ppl_std"]    for c in rand]

    x = np.arange(len(fracs))
    w = 0.4
    ax.bar(x - w / 2, near_dp, w, color="#c0392b", edgecolor="black",
           linewidth=0.5, label="near SW")
    ax.bar(x + w / 2, rand_dp, w, yerr=rand_std, capsize=3, color="#7f8c8d",
           edgecolor="black", linewidth=0.5, label="random")
    ax.axhline(sw["sw_delta_ppl"], color="#2c3e50", ls="--", lw=1,
               label=f"SW-only ({sw['sw_delta_ppl']:+.3f})")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:g}%" for f in fracs])
    ax.set_xlabel("fraction of rows quantised")
    ax.set_ylabel(r"$\Delta$ PPL")
    ax.legend(loc="upper left", frameon=False)


# ─── Panel C ───────────────────────────────────────────────────────────────
def panel_C(ax, euk, prok):
    cond_labels = ["naïve INT4", "Yu-exempt", "SW-only", "SW marginal"]
    colors      = ["#c0392b",   "#27ae60",  "#8e44ad",   "#2980b9"]

    def extract(d):
        if d is None:
            return [np.nan] * 4
        f = d["scopes"]["full"] if "scopes" in d else d.get("full", d)
        return [
            f.get("yu_all_including_sw_delta", np.nan),
            f.get("yu_all_delta",              np.nan),
            f.get("sw_fragility", {}).get("delta_ppl", np.nan),
            f.get("sw_marginal_cost",          np.nan),
        ]

    eu = extract(euk)
    pr = extract(prok)

    x = np.arange(len(cond_labels))
    w = 0.38
    be = ax.bar(x - w / 2, eu, w, color=colors,
                edgecolor="black", linewidth=0.6, label="EUK")
    bp = ax.bar(x + w / 2, pr, w, color=colors, alpha=0.55, hatch="//",
                edgecolor="black", linewidth=0.6, label="PROK")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel(r"$\Delta$ PPL (vs FP16)")

    # value labels only on the bars that aren't ~0 (avoid clutter)
    for bs, vals in [(be, eu), (bp, pr)]:
        for b, v in zip(bs, vals):
            if np.isnan(v) or abs(v) < 0.01:
                continue
            ax.annotate(f"{v:+.3f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3 if b.get_height() >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=7)

    # headroom for the legend + annotation
    ax.set_ylim(-0.05, max(np.nanmax(eu), np.nanmax(pr)) * 1.35)

    # tiny inline annotation that SW-only and SW-marginal are near-zero
    ax.annotate("≈ 0 in both kingdoms",
                xy=(2.5, 0.0), xytext=(2.0, 0.18),
                fontsize=7.5, color="#2980b9", ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.5, color="#2980b9"))

    ax.legend(loc="upper center", frameon=False, ncol=2,
              handletextpad=0.4, columnspacing=1.0)


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    sweep      = safe_load("compression_sweep_dnabert2_prom_core_notata.json")
    gen_int4   = safe_load("quant_ablation_generator_int4.json")
    euk        = safe_load("whole_model_quant_generator_100k.json")
    prok       = safe_load("whole_model_quant_generator_prokaryote_100k.json")

    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 3, wspace=0.32,
                          left=0.06, right=0.98, top=0.93, bottom=0.18)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    panel_A(axA, sweep)
    panel_B(axB, gen_int4)
    panel_C(axC, euk, prok)

    panel_label(axA, "A"); panel_label(axB, "B"); panel_label(axC, "C")

    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print("saved:", OUT_PNG)
    print("saved:", OUT_PDF)


if __name__ == "__main__":
    main()
