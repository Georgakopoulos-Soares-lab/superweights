"""Figure 5 — Compression results (shadow redundancy + INT4).

3 panels:
  A) Pruning sweep on DNABERT-2 (prom_core_notata): 5 criteria × 8 fractions.
     Demonstrates near-SW prunable shadow rows (low-cost selective pruning).
  B) Per-row INT4 on GENERator-EUK: ΔPPL for SW-fragility, near-SW pool, random pool.
     Plus downstream DNABERT-2 splice INT4 sidebar (near_sw vs random ΔMCC).
  C) Whole-model INT4 on GENERator EUK & PROK at extended probe size (100k tokens):
     baseline vs naive-INT4 vs Yu-exempt-SW vs SW-only fragility. ΔPPL bars.

Inputs:
  results/compression_sweep_dnabert2_prom_core_notata.json      (panel A)
  results/quant_ablation_generator_int4.json                    (panel B left)
  results/int4_downstream_benchmark_splice.json                 (panel B right)
  results/whole_model_quant_generator_100k.json                 (panel C, EUK)
  results/whole_model_quant_generator_prokaryote_100k.json      (panel C, PROK)
        — falls back to "(awaiting run)" placeholder if missing.

Output:
  paper/media/image_fig5.png   (+ pdf)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT_PNG = ROOT / "paper" / "media" / "image_fig5.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

CRIT_STYLES = {
    "prox_near": ("Near SW (prox)", "#27ae60", "o"),
    "prox_far":  ("Far from SW (prox)", "#7f8c8d", "s"),
    "l1_low":    ("Lowest L1 norm", "#3498db", "^"),
    "l1_high":   ("Highest L1 norm", "#e67e22", "v"),
    "random":    ("Random", "#95a5a6", "x"),
}


def safe_load(name):
    p = RES / name
    if not p.exists():
        return None
    return json.load(open(p))


def panel_A(ax, sweep):
    fracs = sweep["fracs"]
    for crit, (lab, color, mk) in CRIT_STYLES.items():
        curve = sweep["curves"][crit]
        if crit == "random":
            y = [c["delta_mcc_pct"] for c in curve]
            ys = [c.get("mcc_std", 0) * 100 / sweep["baseline"]["mcc"] for c in curve]
            ax.errorbar(fracs, y, yerr=ys, label=lab, color=color, marker=mk,
                        ms=5, lw=1.2, capsize=2)
        else:
            y = [c["delta_mcc_pct"] for c in curve]
            ax.plot(fracs, y, label=lab, color=color, marker=mk, ms=5, lw=1.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("% rows pruned (log scale)")
    ax.set_ylabel("Δ MCC (%)")
    ax.set_title("A. Pruning sweep — DNABERT-2 prom_core_notata\n"
                 "Shadow rows near SW prune at near-zero cost",
                 fontsize=10, loc="left")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3, which="both")


def panel_B(axL, axR, gen_int4, splice_int4):
    # ---- LEFT: GENERator EUK per-row INT4 ΔPPL ----
    base = gen_int4["baseline_ppl"]
    sw = gen_int4["sw_fragility"]
    near = gen_int4["near_sw"]
    rand = gen_int4["random"]

    fracs = [c["frac"] for c in near]
    near_dp = [c["delta_ppl"] for c in near]
    rand_dp = [c["delta_ppl_mean"] for c in rand]
    rand_std = [c["ppl_std"] for c in rand]

    x = np.arange(len(fracs))
    w = 0.4
    axL.bar(x - w / 2, near_dp, w, color="#c0392b", label="Near SW (proximity)",
            edgecolor="black", linewidth=0.6)
    axL.bar(x + w / 2, rand_dp, w, yerr=rand_std, capsize=3,
            color="#7f8c8d", label="Random (n seeds)",
            edgecolor="black", linewidth=0.6)
    axL.axhline(sw["sw_delta_ppl"], color="#2c3e50", ls="--", lw=1,
                label=f"SW-only ΔPPL = {sw['sw_delta_ppl']:+.4f}")
    axL.axhline(0, color="black", lw=0.6)
    axL.set_xticks(x)
    axL.set_xticklabels([f"{f:g}%" for f in fracs], fontsize=8)
    axL.set_xlabel("Fraction of rows quantised")
    axL.set_ylabel("Δ PPL")
    axL.set_title(f"B1. GENERator-EUK INT4 — per-row ΔPPL\n"
                  f"(baseline PPL = {base:.3f})",
                  fontsize=10, loc="left")
    axL.legend(loc="upper left", fontsize=7, frameon=False)

    # ---- RIGHT: DNABERT-2 splice INT4 downstream ΔMCC ----
    conds = [
        ("FP16\nbaseline", 0.0, 0.0, "#34495e"),
        ("naive\nINT4", splice_int4["naive_int4"]["delta_mcc"], 0, "#c0392b"),
        ("Yu-exempt\nall_INT4", splice_int4["yu_all_int4"]["delta_mcc"], 0, "#e67e22"),
        ("near_SW\nINT4", splice_int4["near_sw_int4"]["delta_mcc"], 0, "#8e44ad"),
        ("random\nINT4 (n=10)", splice_int4["random_int4"]["delta_mcc_mean"]
            if "delta_mcc_mean" in splice_int4["random_int4"]
            else float(np.mean([s["delta_mcc"]
                                for s in splice_int4["random_int4"]["per_seed"]])),
         float(np.std([s["delta_mcc"]
                       for s in splice_int4["random_int4"]["per_seed"]])),
         "#7f8c8d"),
    ]
    xs = np.arange(len(conds))
    vals = [c[1] for c in conds]
    errs = [c[2] for c in conds]
    cols = [c[3] for c in conds]
    axR.bar(xs, vals, yerr=errs, color=cols, capsize=3,
            edgecolor="black", linewidth=0.6)
    axR.axhline(0, color="black", lw=0.6)
    axR.set_xticks(xs)
    axR.set_xticklabels([c[0] for c in conds], fontsize=8)
    axR.set_ylabel("Δ MCC (vs FP16)")
    axR.set_title("B2. DNABERT-2 splice INT4 — downstream task\n"
                  "(near-SW ≈ random; no preferential SW fragility)",
                  fontsize=10, loc="left")


def panel_C(ax, euk, prok):
    """Bar plot: ΔPPL per condition for EUK and PROK whole-model INT4."""
    cond_labels = ["Naive INT4\n(all rows incl. SW)",
                   "Yu-exempt\n(SW protected)",
                   "SW-only\nfragility",
                   "SW marginal\ncost\n(Yu_inc_SW − Yu_excl)"]
    colors = ["#c0392b", "#27ae60", "#8e44ad", "#2980b9"]

    def extract(d):
        if d is None:
            return [np.nan] * 4, None
        f = d["scopes"]["full"] if "scopes" in d else d.get("full", d)
        out = [
            f.get("yu_all_including_sw_delta", np.nan),
            f.get("yu_all_delta", np.nan),
            f.get("sw_fragility", {}).get("delta_ppl", np.nan),
            f.get("sw_marginal_cost", np.nan),
        ]
        meta = d.get("probe", {})
        return out, meta

    euk_vals, euk_meta = extract(euk)
    prok_vals, prok_meta = extract(prok)

    x = np.arange(len(cond_labels))
    w = 0.38
    bars_e = ax.bar(x - w / 2, euk_vals, w, color=colors,
                    edgecolor="black", linewidth=0.6, label="EUK (3B)")
    bars_p = ax.bar(x + w / 2, prok_vals, w, color=colors, alpha=0.55,
                    edgecolor="black", linewidth=0.6, hatch="//", label="PROK (3B)")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=8)
    ax.set_ylabel("Δ PPL (vs FP16)")
    # Annotate bar values
    for bs, vals in [(bars_e, euk_vals), (bars_p, prok_vals)]:
        for b, v in zip(bs, vals):
            if np.isnan(v):
                continue
            ax.annotate(f"{v:+.4f}", xy=(b.get_x() + b.get_width() / 2,
                                          b.get_height()),
                        xytext=(0, 3 if b.get_height() >= 0 else -10),
                        textcoords="offset points", ha="center", fontsize=6.5)
    n_tok = (euk_meta or prok_meta or {}).get("approx_tokens", "≈100k")
    title = f"C. Whole-model INT4 — extended probe ({n_tok:,} tokens)"
    if euk is None and prok is None:
        title += "\n[awaiting run — placeholders shown]"
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False)


def main():
    sweep = safe_load("compression_sweep_dnabert2_prom_core_notata.json")
    gen_int4 = safe_load("quant_ablation_generator_int4.json")
    splice_int4 = safe_load("int4_downstream_benchmark_splice.json")
    euk = safe_load("whole_model_quant_generator_100k.json")
    prok = safe_load("whole_model_quant_generator_prokaryote_100k.json")

    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          width_ratios=[1.1, 1.0],
                          hspace=0.45, wspace=0.30)
    axA = fig.add_subplot(gs[0, 0])
    axBL = fig.add_subplot(gs[0, 1])
    axBR = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])

    panel_A(axA, sweep)
    panel_B(axBL, axBR, gen_int4, splice_int4)
    panel_C(axC, euk, prok)

    fig.suptitle("Figure 5 — Compression: shadow redundancy & whole-model INT4",
                 fontsize=13, y=0.995)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved: {OUT_PNG}")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
