#!/usr/bin/env python3
"""
Figure 4 -- GENERator: functional realization of a decoder high-gain pathway.

Four panels (C replaced in round-2, audit/round2/final_check.md Section 2):
  A. Minimal structural callout -- GENERator EUK row 2371's exact q1 (near-rank-1),
     cross-referencing Figure 1B rather than restating it.
  B. BOS-centered attention/activation phenotype -- NOW REPRODUCED with a real raw
     artifact (results/mechanism/attention_sink_implicit_bias.json, produced
     2026-08-22 by this repo's own run_attention_sink.py after a hardcoded-path
     fix; see FIGURE_PROVENANCE.md). Framed strictly as association/co-occurrence
     -- the high-gain row's activation and incoming attention both concentrate at
     position 0, but this does not establish that the row CAUSES the sink.
  C. (round-2 replacement) Two sub-panels sharing one "C" label: left is GC vs.
     random-direction scale c; right is GC vs. native NLL (damage pool) with row
     2371's own alpha-sweep, the 5 inert control rows, and the random-direction grid
     on shared axes, damage as the organizing axis -- the point is that composition
     tracks damage monotonically, not which direction did the damage. Supersedes the
     old alpha-only row2371/row1522/random-band panel C entirely. Folded in from the
     standalone audit/round2/scripts/fig4c_replacement.py.

Source: results/e7_legacy_reanalysis.json, results/mechanism/attention_sink_implicit_bias.json,
experiments/E9_mechanistic_tomography/baseline_regression_results.json; panel C:
audit/round2/generator_random_direction_full.csv.
Output: figures/main/fig4_generator.{png,pdf}, figures/source_data/fig4_generator.json

Sequence-quality/specificity over the intervention range remains NOT producible --
still no raw artifact of any kind anywhere in this repository.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER_SALVAGE = HERE.parent
REPO_ROOT = PAPER_SALVAGE.parent
RESULTS = REPO_ROOT / "results"
E9 = PAPER_SALVAGE / "experiments" / "E9_mechanistic_tomography"
AUDIT2 = REPO_ROOT / "audit" / "round2"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
from _figstyle import apply_style, panel_label  # noqa: E402
from _paper_encoding import DOMAIN_COLOR  # noqa: E402

ALPHAS = [0.0, 0.5, 1.0]
ABLATION_GC = 0.3065
BASELINE_GC = 0.4204
ABLATION_NLL = 8.754319605827332


def main():
    apply_style()
    legacy = json.loads((RESULTS / "e7_legacy_reanalysis.json").read_text())["GENERator EUK"]
    sink = json.loads((RESULTS / "mechanism" / "attention_sink_implicit_bias.json").read_text())["generator"]
    gen = json.loads((E9 / "baseline_regression_results.json").read_text())["generator"]

    fig = plt.figure(figsize=(10.5, 7.6), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.15])
    top = outer[0].subgridspec(1, 2, width_ratios=[0.62, 1.0])
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(top[0, 0]); ax_b = fig.add_subplot(top[0, 1])
    ax_cL = fig.add_subplot(bot[0, 0]); ax_cR = fig.add_subplot(bot[0, 1])

    # --- Panel A: minimal structural callout -----------------------------------------
    ax_a.bar([0], [legacy["q1"]], color=DOMAIN_COLOR["genomic"], edgecolor="black",
             linewidth=0.8, width=0.5)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xticks([0]); ax_a.set_xticklabels(["GENERator EUK\n(L4/r2371)"], fontsize=7.5)
    ax_a.set_ylabel(r"$q_1$ (exact, see Fig. 1B)")
    ax_a.text(0, legacy["q1"] + 0.03, f"{legacy['q1']:.3f}", ha="center", fontsize=7.5)
    panel_label(ax_a, "A")

    # --- Panel B: BOS-centered attention/activation phenotype (reproduced) -----------
    att = sink["attention"]
    bars_x = [0, 1]
    bars_y = [att["sink_share_pos0"] * 100, att["uniform_expectation"] * 100]
    ax_b.bar(bars_x, bars_y, color=[DOMAIN_COLOR["genomic"], "0.7"], edgecolor="black",
             linewidth=0.7, width=0.55)
    ax_b.set_xticks(bars_x)
    ax_b.set_xticklabels(["attention mass\n@ position 0", "uniform\nexpectation"], fontsize=7)
    ax_b.set_ylabel("mean incoming attention (%)")
    ax_b.text(0, bars_y[0] + 1.2, f"{bars_y[0]:.1f}%\n({att['sink_over_uniform']:.1f}$\\times$ uniform)",
              ha="center", fontsize=6.8)
    ax_b.text(0.5, max(bars_y) * 0.55,
              f"{att['frac_heads_argmax_pos0']*100:.1f}% of heads\nargmax at pos. 0\n\n"
              "activation max\nalso at BOS\n(co-occurrence only)",
              ha="center", va="center", fontsize=6.2, color="0.3",
              bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", linewidth=0.6))
    ax_b.set_ylim(0, max(bars_y) * 1.35)
    panel_label(ax_b, "B")

    # --- Panel C (round-2 replacement): GC vs. c | GC vs. damage (native NLL) --------
    gen_full = list(csv.DictReader(open(AUDIT2 / "generator_random_direction_full.csv")))
    rd = [r for r in gen_full if r["series"] == "random_direction" and r["gc_mean"]]
    rd.sort(key=lambda r: float(r["x_c_or_alpha"]))
    rd_c = [float(r["x_c_or_alpha"]) for r in rd]
    rd_gc = [float(r["gc_mean"]) for r in rd]
    rd_lo = [float(r["gc_ci_low"]) for r in rd]
    rd_hi = [float(r["gc_ci_high"]) for r in rd]
    rd_nll = [float(r["nll"]) for r in rd]

    r2371 = [r for r in gen_full if r["series"] == "row2371_own_alpha_sweep"]
    r2371.sort(key=lambda r: float(r["x_c_or_alpha"]))
    r2371_nll = [float(r["nll"]) for r in r2371]
    r2371_gc = [float(r["gc_mean"]) for r in r2371]

    ctrl = [r for r in gen_full if r["series"].startswith("control_row_")]
    ctrl_nll = [float(r["nll"]) for r in ctrl]
    ctrl_gc = [float(r["gc_mean"]) for r in ctrl]

    # left: GC vs. random-direction scale c
    yerr = [np.array(rd_gc) - np.array(rd_lo), np.array(rd_hi) - np.array(rd_gc)]
    ax_cL.errorbar(rd_c, rd_gc, yerr=yerr, fmt="o-", color=DOMAIN_COLOR["genomic"],
                    markeredgecolor="black", markersize=6, lw=1.2, capsize=3, zorder=3,
                    label="random direction (matched location)")
    ax_cL.axhline(ABLATION_GC, color="0.35", linestyle="--", linewidth=1.1,
                   label=f"row 2371 full ablation (GC={ABLATION_GC:.4f})")
    ax_cL.axhline(BASELINE_GC, color="black", linestyle=":", linewidth=1.1,
                   label=f"untouched baseline (GC={BASELINE_GC:.4f})")
    ax_cL.set_xscale("symlog", linthresh=0.01)
    ax_cL.set_xlim(0, 9.5)
    ax_cL.set_ylim(0.285, 0.475)
    ax_cL.set_xlabel(r"random-direction scale $c$")
    ax_cL.set_ylabel("generated GC fraction")
    ax_cL.legend(fontsize=6.2, loc="upper left", frameon=True, facecolor="white",
                 edgecolor="none", framealpha=0.9)
    ax_cL.set_title("GC vs. random-direction scale", fontsize=8, fontweight="normal", loc="left", pad=3)

    # right: GC vs. damage (native NLL), all conditions on shared axes
    ax_cR.plot(r2371_nll, r2371_gc, "o-", color=DOMAIN_COLOR["text"], markeredgecolor="black",
               markersize=6, lw=1.2, alpha=0.85, zorder=3, label=r"row 2371 own $\alpha$-sweep")
    ax_cR.plot(rd_nll, rd_gc, "s-", color=DOMAIN_COLOR["genomic"], markeredgecolor="black",
               markersize=6, lw=1.2, alpha=0.85, zorder=4, label="random direction (matched location)")
    ax_cR.scatter([ABLATION_NLL], [ABLATION_GC], marker="*", s=130, color="0.35",
                  edgecolor="black", linewidths=0.8, alpha=0.85, zorder=5)
    ax_cR.scatter(ctrl_nll, ctrl_gc, marker="^", color="white", edgecolor="black",
                  s=70, alpha=0.95, zorder=6, label="5 inert control rows")
    ax_cR.annotate("row2371 $\\alpha$=0 &\nrandom-dir $c$=0\n(same zeroed state)",
                    xy=(8.754, 0.3065), xytext=(7.9, 0.26), fontsize=6.0, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.6))
    ax_cR.annotate("row2371 $\\alpha$=1.0 &\ncontrols (untouched)",
                    xy=(6.39, 0.42), xytext=(6.9, 0.365), fontsize=6.0, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.6))
    ax_cR.set_xlabel("NLL (damage pool)")
    ax_cR.set_ylabel("generated GC fraction")
    ax_cR.legend(fontsize=6.2, loc="upper right")
    ax_cR.set_title("GC vs. damage, all conditions", fontsize=8, fontweight="normal", loc="left", pad=3)

    panel_label(ax_cL, "C")  # single spanning label for the two C sub-panels

    out = HERE / "main" / "fig4_generator"
    fig.savefig(f"{out}.png"); fig.savefig(f"{out}.pdf")

    src_out = HERE / "source_data" / "fig4_generator.json"
    src_out.write_text(json.dumps(dict(structural=legacy, attention_sink=sink,
                                        gc_dose_response=gen), indent=1))
    print(f"saved -> {out}.png / .pdf, {src_out}")


if __name__ == "__main__":
    main()
