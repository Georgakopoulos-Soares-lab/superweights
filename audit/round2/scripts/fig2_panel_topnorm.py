#!/usr/bin/env python3
"""Fig. 2, new panel D (round-2 Section 2): candidate vs. top-norm-control causal effect,
both intervention strengths, full 22-model cohort.

Standalone deliverable -- does NOT overwrite experiments/figures/fig2_causal.py or its
output. Produces its own PNG/PDF in audit/round2/figures/.

Plots causal_topk_gap = candidate relative-NLL-change minus median top-norm-control
relative-NLL-change, per model per epsilon, on a symmetric-log y-axis (values cross zero).
Epsilon=0.5 points are filled (candidate-style, black edge); epsilon=1.0 points are open
(hollow, colored edge) at the same x position, offset slightly. The 4 sign-flip models
(union of negative-gap models across the two epsilons) are text-labeled.

Model x-order matches Fig. 1's two-panel figure (sorted by random_control_gap descending,
from audit/round2/fig1c_random_vs_topk_gaps_full22.csv) so panel position is directly
cross-referenceable between the two figures.

Data: audit/round2/structural_vs_causal_gap.csv (44 rows = 22 models x 2 epsilons).
Output:
  audit/round2/figures/fig2_topnorm_panel.{png,pdf}
  audit/round2/figures/fig2_panel_d_source.csv
  experiments/figures/source_data/fig2_panel_d_source.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as ml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT2 = ROOT / "audit" / "round2"
FIG_DIR = AUDIT2 / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR2 = ROOT / "manuscript" / "figures" / "source_data"
SRC_DIR2.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from _figstyle import apply_style, panel_label  # noqa: E402

sys.path.insert(0, str(ROOT / "manuscript" / "figures"))
from _paper_encoding import DOMAIN_COLOR, ARCH_MARKER  # noqa: E402

# same display-name / arch / domain mapping used in fig1_full22_two_panel.py
DISPLAY = {
    "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B": "Qwen2.5-3B", "HuggingFaceTB/SmolLM2-135M": "SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M": "SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B": "SmolLM2-1.7B",
    "GenerTeam/GENERator-v2-prokaryote-1.2b-base": "GEN-PROK-1.2B",
    "GenerTeam/GENERator-v2-prokaryote-3b-base": "GEN-PROK-3B",
    "EuroBERT/EuroBERT-210m": "EuroBERT-210M", "EuroBERT/EuroBERT-610m": "EuroBERT-610M",
    "EuroBERT/EuroBERT-2.1B": "EuroBERT-2.1B", "answerdotai/ModernBERT-large": "ModernBERT-large",
    "ModernBERT-base": "ModernBERT-base", "DNABERT-2": "DNABERT-2",
    "GENERator-EUK-3B": "GEN-EUK-3B", "Llama-7B": "Llama-7B", "Mistral-7B": "Mistral-7B",
    "OLMo-7B-0724-hf": "OLMo-7B", "MosaicBERT": "MosaicBERT", "GenomeOcean-4B": "GenomeOcean-4B",
    "Qwen2.5-7B": "Qwen2.5-7B", "NTv3": "NTv3",
}

SIGN_FLIP_MODELS = {"MosaicBERT", "Qwen2.5-7B", "Qwen/Qwen2.5-0.5B", "NTv3"}

EPS_STYLE = {
    "0.5": dict(fill=True, s=52, offset=-0.14),
    "1.0": dict(fill=False, s=68, offset=0.14),
}


def main():
    apply_style()

    # Fig 1's model order + architecture/domain, for a consistent cross-figure x-axis.
    fig1_summary = list(csv.DictReader(open(AUDIT2 / "fig1c_random_vs_topk_gaps_full22.csv")))
    fig1_meta = {r["model"]: r for r in fig1_summary}
    order = sorted(fig1_meta, key=lambda m: -float(fig1_meta[m]["random_control_gap"]))
    assert len(order) == 22

    causal = list(csv.DictReader(open(AUDIT2 / "structural_vs_causal_gap.csv")))
    assert len(causal) == 44, f"expected 44 rows (22 models x 2 eps), got {len(causal)}"
    by_model_eps = {(r["model"], r["epsilon"]): r for r in causal}
    causal_models = {r["model"] for r in causal}
    assert causal_models == set(order), (
        f"model-name mismatch between structural_vs_causal_gap.csv and fig1 summary: "
        f"{causal_models.symmetric_difference(order)}")

    fig, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    xs = np.arange(len(order))

    for x, m in zip(xs, order):
        arch, dom = fig1_meta[m]["architecture"], fig1_meta[m]["domain"]
        neg_eps_count = 0
        for eps in ("0.5", "1.0"):
            r = by_model_eps[(m, eps)]
            gap = float(r["causal_topk_gap"])
            st = EPS_STYLE[eps]
            kw = dict(marker=ARCH_MARKER[arch], s=st["s"], zorder=4)
            if st["fill"]:
                kw.update(color=DOMAIN_COLOR[dom], edgecolor="black", linewidths=0.9)
            else:
                kw.update(facecolors="none", edgecolors=DOMAIN_COLOR[dom], linewidths=1.2)
            ax.scatter(x + st["offset"], gap, **kw)
            if m in SIGN_FLIP_MODELS and gap < 0:
                # stagger the y-offset when a model flips sign at BOTH epsilons (NTv3) so
                # the two annotations don't overlap each other.
                dy = -13 - 22 * neg_eps_count
                ax.annotate(f"{DISPLAY[m]}, " + r"$\epsilon$=" + eps, (x + st["offset"], gap),
                            xytext=(0, dy), textcoords="offset points", ha="center", va="top",
                            fontsize=5.4, color="0.15",
                            arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5"))
                neg_eps_count += 1

    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_ylim(-1.2, 15)  # extra bottom headroom for the stacked NTv3 sign-flip labels
    ax.set_ylabel(r"causal $\Delta$NLL gap: candidate $-$ median top-norm control" "\n(relative NLL change, symlog)")
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[m] for m in order], rotation=58, ha="right", fontsize=6.4)
    ax.set_xlim(-0.7, len(order) - 0.3)

    leg = [ml.Line2D([], [], marker="o", color="black", linestyle="", markerfacecolor=".35",
                      label=r"$\epsilon$=0.5 (filled)"),
           ml.Line2D([], [], marker="o", color=".35", linestyle="", markerfacecolor="none",
                      markeredgewidth=1.2, label=r"$\epsilon$=1.0 (open)"),
           ml.Line2D([], [], marker="o", color=".35", linestyle="", label="decoder"),
           ml.Line2D([], [], marker="s", color=".35", linestyle="", label="encoder"),
           ml.Line2D([], [], marker="o", color=DOMAIN_COLOR["text"], linestyle="", label="text"),
           ml.Line2D([], [], marker="o", color=DOMAIN_COLOR["genomic"], linestyle="", label="genomic")]
    ax.legend(handles=leg, frameon=False, fontsize=6.2, ncol=3, loc="upper right")

    n_neg_05 = sum(1 for m in order if float(by_model_eps[(m, "0.5")]["causal_topk_gap"]) < 0)
    n_neg_10 = sum(1 for m in order if float(by_model_eps[(m, "1.0")]["causal_topk_gap"]) < 0)
    ax.text(0.01, 0.03, f"negative gap: {n_neg_05}/22 at eps=0.5, {n_neg_10}/22 at eps=1.0",
            transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left", color="0.25")

    panel_label(ax, "D")

    out = FIG_DIR / "fig2_topnorm_panel"
    fig.savefig(out.with_suffix(".png")); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}.png / .pdf")

    # ---- source-data CSV --------------------------------------------------------
    fieldnames = ["model", "display_name", "architecture", "domain", "epsilon",
                  "candidate_relative_loss_change", "median_topk_control_relative_loss_change",
                  "causal_topk_gap", "sign_flip_flag", "x_order_index"]
    rows_out = []
    for i, m in enumerate(order):
        for eps in ("0.5", "1.0"):
            r = by_model_eps[(m, eps)]
            gap = float(r["causal_topk_gap"])
            rows_out.append(dict(
                model=m, display_name=DISPLAY[m], architecture=fig1_meta[m]["architecture"],
                domain=fig1_meta[m]["domain"], epsilon=eps,
                candidate_relative_loss_change=r["candidate_relative_loss_change"],
                median_topk_control_relative_loss_change=r["median_topk_control_relative_loss_change"],
                causal_topk_gap=gap, sign_flip_flag=int(m in SIGN_FLIP_MODELS and gap < 0),
                x_order_index=i))

    for dest in (FIG_DIR / "fig2_panel_d_source.csv", SRC_DIR2 / "fig2_panel_d_source.csv"):
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows_out)
        print(f"wrote {dest}")

    print(f"\nnegative causal_topk_gap: {n_neg_05}/22 at eps=0.5, {n_neg_10}/22 at eps=1.0")
    neg05 = sorted(DISPLAY[m] for m in order if float(by_model_eps[(m, '0.5')]['causal_topk_gap']) < 0)
    neg10 = sorted(DISPLAY[m] for m in order if float(by_model_eps[(m, '1.0')]['causal_topk_gap']) < 0)
    print(f"  eps=0.5 negative: {neg05}")
    print(f"  eps=1.0 negative: {neg10}")
    union = sorted(set(neg05) | set(neg10))
    print(f"  union (sign-flip models, expect 4): {union}")


if __name__ == "__main__":
    main()
