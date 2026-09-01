#!/usr/bin/env python3
"""
E9 Figure — GENERator EUK causal GC dose-response for row 2371 (primary) and row
1522 (secondary), against the random-row control band. Both rows write to output
column 2536 (layer 4); random control = 5 seeded rows, matched protocol.

Source: experiments/E9_mechanistic_tomography/baseline_regression_results.json
(generator block). Output: figures/output/fig_e9_generator_dose_response.{png,pdf}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E9 = HERE.parents[0] / "experiments" / "E9_mechanistic_tomography"
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from _figstyle import apply_style, panel_label  # noqa: E402

ALPHAS = [0.0, 0.5, 1.0]


def series(block, key):
    sub = block[key]
    return (np.array([sub[a]["gc_mean"] for a in ["0.0", "0.5", "1.0"]]),
            np.array([sub[a].get("gc_sd", np.nan) for a in ["0.0", "0.5", "1.0"]]),
            np.array([sub[a]["n"] for a in ["0.0", "0.5", "1.0"]]))


def main():
    apply_style()
    d = json.loads((E9 / "baseline_regression_results.json").read_text())["generator"]
    res = d["results"]

    fig, ax = plt.subplots(figsize=(4.6, 3.8))

    primary_mean, primary_sd, primary_n = series(res, "primary_2371")
    secondary_mean, secondary_sd, secondary_n = series(res, "secondary_1522")
    rand_mean, _, rand_n = series(res, "random_control")

    def sem(sd, n):
        return sd / np.sqrt(n)

    # Only 3 dose points exist per row (prereg-frozen, no denser grid collected) --
    # connectors are dotted "guide the eye" lines, not a fitted curve, so the plot
    # does not visually imply a validated graded/linear steering axis.
    ax.errorbar(ALPHAS, primary_mean, yerr=sem(primary_sd, primary_n),
                marker="o", ms=5, lw=1.1, ls=":", color="#c0392b", capsize=2.5,
                label=f"row 2371 (primary), span={d['gc_span_primary']:.3f}")
    ax.errorbar(ALPHAS, secondary_mean, yerr=sem(secondary_sd, secondary_n),
                marker="s", ms=5, lw=1.1, ls=":", color="#5b8ec4", capsize=2.5,
                label="row 1522 (secondary, non-monotonic)")
    ax.plot(ALPHAS, rand_mean, marker="D", ms=4, lw=1.1, color="0.55", ls=":",
            label=f"random control (5 rows), span={d['gc_span_random']:.4f}")

    ax.set_xlabel(r"row scale $\alpha$  (1.0 = intact, 0.0 = fully ablated)")
    ax.set_ylabel("generated GC fraction")
    ax.set_xticks(ALPHAS)
    ax.legend(loc="lower left", frameon=False, fontsize=6.8, handletextpad=0.5)

    fig.tight_layout()
    out = HERE / "output" / "fig_e9_generator_dose_response"
    fig.savefig(f"{out}.png")
    fig.savefig(f"{out}.pdf")
    print(f"saved -> {out}.png / .pdf")


if __name__ == "__main__":
    main()
