#!/usr/bin/env python3
"""
E9 Figure — observer-family (F0-F3) held-out prediction quality, DNABERT-2.

Predicted-vs-actual scatter for the 20 genuinely held-out masks, one panel per
epsilon, one marker family per observer (F0 singleton-additive, F1 scalar-
calibrated, F2 jointly-fit additive, F3 lifted main+pair). Reuses the exact
prediction arithmetic from run_fit_observers.py (a_held @ beta for F0-F2,
lifted(a_held) @ coef3 for F3) applied to the already-fit, already-frozen
coefficients in fit_results_dnabert2.json -- no new fitting, no new measurement.

Source: experiments/E9_mechanistic_tomography/{fit_results_dnabert2.json,
dnabert2_mask_responses.json}. Output: figures/output/fig_e9_observer_ladder.{png,pdf}.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E9 = HERE.parents[0] / "E9_mechanistic_tomography"
sys.path.insert(0, str(ROOT / "experiments" / "figures"))
from _figstyle import apply_style, panel_label  # noqa: E402

N = 10
PAIRS = list(itertools.combinations(range(N), 2))


def lifted(a: np.ndarray) -> np.ndarray:
    inter = np.array([a[:, i] * a[:, j] for i, j in PAIRS]).T
    return np.hstack([a, inter])


def load():
    fit = json.loads((E9 / "fit_results_dnabert2.json").read_text())
    resp = json.loads((E9 / "dnabert2_mask_responses.json").read_text())
    pools = resp["responses"]["pools"]
    a_held = np.array([r["a"] for r in pools["held_out"]], dtype=float)
    return fit, a_held


def main():
    apply_style()
    fit, a_held = load()

    E9resp = json.loads((E9 / "dnabert2_mask_responses.json").read_text())
    held = E9resp["responses"]["pools"]["held_out"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=False, sharey=False)

    families = [
        ("F0", "singleton additive", "s", "#9e9e9e"),
        ("F1", "scalar-calibrated", "^", "#5b8ec4"),
        ("F2", "jointly-fit additive", "D", "#e8a13a"),
        ("F3", "lifted main + pair", "o", "#c0392b"),
    ]

    for ax, eps in zip(axes, ["0.5", "1.0"]):
        fe = fit["by_epsilon"][eps]
        key = f"dloss_eps{eps}"
        y_held = np.array([r[key] for r in held])

        x_single = np.array(fe["singleton_effects_x_i"])
        yhat = {}
        yhat["F0"] = a_held @ x_single
        yhat["F1"] = fe["F1"]["gain_g"] * yhat["F0"]
        yhat["F2"] = a_held @ np.array(fe["F2"]["beta"])
        # gamma_pairs is a JSON object written by a dict-comprehension over PAIRS in
        # run_fit_observers.py, so key order == PAIRS order; values() preserves it.
        gamma_vals = np.array(list(fe["F3"]["gamma_pairs"].values()))
        coef3 = np.concatenate([np.array(fe["F3"]["beta_main"]), gamma_vals])
        yhat["F3"] = lifted(a_held) @ coef3

        lims = [min(y_held.min(), min(v.min() for v in yhat.values())),
                max(y_held.max(), max(v.max() for v in yhat.values()))]
        pad = 0.06 * (lims[1] - lims[0])
        lims = [lims[0] - pad, lims[1] + pad]
        ax.plot(lims, lims, color="0.75", lw=1.0, zorder=1, ls="--")

        for fam, label, marker, color in families:
            r2 = fe[fam]["metrics"]["r2"]
            ax.scatter(y_held, yhat[fam], marker=marker, s=22, color=color,
                       alpha=0.85, linewidths=0, zorder=3,
                       label=f"{fam} ({label}), $R^2$={r2:.2f}")

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel(r"actual held-out $\Delta$loss")
        ax.set_ylabel(r"predicted $\Delta$loss")
        ax.set_title(rf"$\epsilon$={eps}", fontsize=9, loc="right", color="0.35")
        ax.set_aspect("equal", adjustable="box")

    axes[0].legend(loc="upper left", frameon=False, fontsize=6.6, handletextpad=0.4,
                    borderaxespad=0.2)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")

    fig.tight_layout()
    out = HERE / "output" / "fig_e9_observer_ladder"
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    print(f"saved -> {out}.png / .pdf")


if __name__ == "__main__":
    main()
