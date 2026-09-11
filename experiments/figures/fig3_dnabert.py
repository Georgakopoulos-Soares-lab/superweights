#!/usr/bin/env python3
"""
Figure 3 -- DNABERT-2 mechanistic organization and falsification.

Four panels, ALL loaded from E9's own from-scratch raw JSON on the pretrained-MLM
endpoint. UPDATE 2026-08-22: a separate, later reproduction attempt on the
fine-tuned splice-accuracy endpoint (C-036) and the codominance ratio sweep
(C-038) now HAS a real raw artifact -- and that artifact materially conflicts
with the previously-adopted colleague-report magnitudes (measured joint pair
effect -0.42pp vs the previously-cited -33.76pp; codominance epistasis range
-0.53->-0.04pp vs the previously-cited -33.63->-0.66pp; single seed only). C-036
and C-038 are therefore CONTESTED, not established and not simply absent -- see
FIGURE_PROVENANCE.md for the full comparison table. Per instruction, this figure
does NOT plot either the old (unsupported) magnitude or the new single-seed
value as a headline result; it stays centered on the pretrained-MLM endpoint
(panels A-B below), which is NOT in dispute and independently corroborates that
L9/r264+r294 is a real interaction, just not the magnitude/downstream claim
previously attached to it. Codominance and the disputed splice-accuracy pair
effect are cited in the caption as contested context only, never plotted:

  A. Singleton vs. pair causal effect on pretrained MLM loss: individually masking
     L9/r264 or L9/r294 (the critical pair) produces small dloss; masking BOTH
     together (a real, directly-measured 2-channel intervention, not a fitted
     prediction) produces a much larger effect than the sum of the parts.
  B. The same super-additivity (epistasis = d_AB - d_A - d_B > 0) holds at both
     tested intervention scales (epsilon=0.5, 1.0) -- the interaction is not an
     artifact of one specific ablation strength.
  C. E9 observer-family (F0-F3) held-out prediction quality (reused from
     fig_e9_observer_ladder.py's logic).
  D. F3 pairwise interaction map with the known critical pair highlighted (reused
     from fig_e9_pair_interaction_map.py's logic).

Source: experiments/E9_mechanistic_tomography/{baseline_regression_results.json,
fit_results_dnabert2.json, dnabert2_mask_responses.json}.
Output: figures/main/fig3_dnabert.{png,pdf}, figures/source_data/fig3_dnabert.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER_SALVAGE = HERE.parent
REPO_ROOT = PAPER_SALVAGE.parent
E9 = PAPER_SALVAGE / "experiments" / "E9_mechanistic_tomography"
sys.path.insert(0, str(REPO_ROOT / "experiments" / "figures"))
from _figstyle import apply_style, panel_label  # noqa: E402

N = 10
PAIRS = list(itertools.combinations(range(N), 2))
CRITICAL = "(9, 264)x(9, 294)"


def lifted(a: np.ndarray) -> np.ndarray:
    inter = np.array([a[:, i] * a[:, j] for i, j in PAIRS]).T
    return np.hstack([a, inter])


def main():
    apply_style()
    baseline = json.loads((E9 / "baseline_regression_results.json").read_text())["dnabert2"]
    fit = json.loads((E9 / "fit_results_dnabert2.json").read_text())
    resp = json.loads((E9 / "dnabert2_mask_responses.json").read_text())
    a_held = np.array([r["a"] for r in resp["responses"]["pools"]["held_out"]], dtype=float)

    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.0], hspace=0.55, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    ax_d = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

    # --- Panel A: singleton vs. pair (directly-measured), epsilon=1.0 --------------
    d = baseline["epsilon_1p0"]
    labels = ["L9/r264\nalone", "L9/r294\nalone", "both\n(joint)"]
    vals = [d["d_A"], d["d_B"], d["d_AB"]]
    colors = ["#8fa8c9", "#8fa8c9", "#c0392b"]
    ax_a.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.7)
    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"$\Delta$ pretrained MLM loss")
    ax_a.text(0.5, 0.94, rf"epistasis $= d_{{AB}}-d_A-d_B = {d['epistasis']:.3f}$",
              transform=ax_a.transAxes, ha="center", va="top", fontsize=6.6, color="0.3")
    panel_label(ax_a, "A")

    # --- Panel B: epistasis at both epsilons ----------------------------------------
    eps_keys = ["epsilon_0p5", "epsilon_1p0"]
    eps_labels = [r"$\epsilon$=0.5", r"$\epsilon$=1.0"]
    epis = [baseline[k]["epistasis"] for k in eps_keys]
    ax_b.bar(eps_labels, epis, color="#c0392b", edgecolor="black", linewidth=0.7, width=0.5)
    ax_b.axhline(0, color="0.4", lw=0.8)
    ax_b.set_ylabel("epistasis (super-additivity)")
    for xi, v in enumerate(epis):
        ax_b.text(xi, v + 0.03 * max(epis), f"{v:.3f}", ha="center", fontsize=7)
    panel_label(ax_b, "B")

    # --- Panel C: observer ladder (F0-F3 held-out prediction) -----------------------
    held = resp["responses"]["pools"]["held_out"]
    families = [("F0", "singleton additive", "s", "#9e9e9e"),
                ("F1", "scalar-calibrated", "^", "#5b8ec4"),
                ("F2", "jointly-fit additive", "D", "#e8a13a"),
                ("F3", "lifted main + pair", "o", "#c0392b")]
    for axi, eps in zip(ax_c, ["0.5", "1.0"]):
        fe = fit["by_epsilon"][eps]
        key = f"dloss_eps{eps}"
        y_held = np.array([r[key] for r in held])
        x_single = np.array(fe["singleton_effects_x_i"])
        yhat = {"F0": a_held @ x_single}
        yhat["F1"] = fe["F1"]["gain_g"] * yhat["F0"]
        yhat["F2"] = a_held @ np.array(fe["F2"]["beta"])
        gamma_vals = np.array(list(fe["F3"]["gamma_pairs"].values()))
        coef3 = np.concatenate([np.array(fe["F3"]["beta_main"]), gamma_vals])
        yhat["F3"] = lifted(a_held) @ coef3

        lims = [min(y_held.min(), min(v.min() for v in yhat.values())),
                max(y_held.max(), max(v.max() for v in yhat.values()))]
        pad = 0.06 * (lims[1] - lims[0])
        lims = [lims[0] - pad, lims[1] + pad]
        axi.plot(lims, lims, color="0.75", lw=1.0, zorder=1, ls="--")
        for fam, label, marker, color in families:
            r2 = fe[fam]["metrics"]["r2"]
            axi.scatter(y_held, yhat[fam], marker=marker, s=18, color=color, alpha=0.85,
                        linewidths=0, zorder=3, label=f"{fam}, $R^2$={r2:.2f}")
        axi.set_xlim(lims); axi.set_ylim(lims)
        axi.set_xlabel(r"actual held-out $\Delta$loss")
        axi.set_ylabel(r"predicted $\Delta$loss")
        axi.set_title(rf"$\epsilon$={eps}", fontsize=8.5, loc="right", color="0.35")
        axi.set_aspect("equal", adjustable="box")
    ax_c[0].legend(loc="upper left", frameon=False, fontsize=5.8, handletextpad=0.3, borderaxespad=0.2)
    panel_label(ax_c[0], "C")

    # --- Panel D: pair interaction map ------------------------------------------------
    TOP_K = 12
    for axi, eps in zip(ax_d, ["0.5", "1.0"]):
        fe = fit["by_epsilon"][eps]
        gp = fe["F3"]["gamma_pairs"]
        rank = fe["F3"]["critical_pair_rank_by_abs_gamma"]
        gamma_crit = fe["F3"]["critical_pair_gamma"]
        items = sorted(gp.items(), key=lambda kv: -abs(kv[1]))[:TOP_K]
        names = [k.replace("x", " x ") for k, _ in items]
        vals = [v for _, v in items]
        colors = ["#c0392b" if k == CRITICAL else "#8fa8c9" for k, _ in items]
        y = np.arange(len(items))[::-1]
        axi.barh(y, vals, color=colors, height=0.68)
        axi.set_yticks(y); axi.set_yticklabels(names, fontsize=5.6)
        axi.axvline(0, color="0.5", lw=0.8)
        axi.set_xlabel(r"pair coefficient $\Gamma$ (F3, ridge)")
        axi.set_title(rf"$\epsilon$={eps} (pair rank {rank}/45, $\Gamma$={gamma_crit:.3f})",
                      fontsize=6.6, loc="left", color="0.3")
    panel_label(ax_d[0], "D")

    out = HERE / "main" / "fig3_dnabert"
    fig.savefig(f"{out}.png"); fig.savefig(f"{out}.pdf")

    src_out = HERE / "source_data" / "fig3_dnabert.json"
    src_out.write_text(json.dumps(dict(baseline_regression=baseline,
                                        fit_results_summary={eps: {k: fit["by_epsilon"][eps][k]
                                                                    for k in ["singleton_effects_x_i", "F3"]}
                                                              for eps in ["0.5", "1.0"]}),
                                   indent=1))
    print(f"saved -> {out}.png / .pdf, {src_out}")


if __name__ == "__main__":
    main()
