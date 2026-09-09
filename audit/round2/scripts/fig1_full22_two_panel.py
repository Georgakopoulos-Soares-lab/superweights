#!/usr/bin/env python3
"""Fig. 1C, full 22-model cohort, two panels (manuscript regeneration, round-2 Section 2).

Left panel:  candidate q1 vs. 5 random same-layer controls (jittered points), all 22 models.
Right panel: candidate q1 vs. 5 top-norm-by-rank same-layer controls, same 22 models, same
             y-axis. The 7 models with a negative candidate-vs-topnorm gap are text-labeled.

Style matches manuscript/figures/fig1_structural.py panel C (filled candidate marker +
5 small hollow-ring controls per model, one x-position per model), extended from 12 to the
full 22-model cohort and split into two side-by-side panels sharing a y-axis.

Data sources (already verified consistent by the orchestrating session, all keyed by the
same raw model-name strings):
  - audit/round2/fig1c_random_vs_topk_gaps_full22.csv
        candidate_q1, mean_random_control_q1, random_control_gap, topk_by_norm_gap,
        architecture, domain, non_embed_params -- one row per model (22 rows).
  - results/E11/scale_ladder_controls.csv               -- 5 random-control q1 rows,
    audit/round2/section2_random_control_q1_batch2.csv     original-12 / batch2-10 models.
  - audit/round2/topk_norm_q1_summary.csv                -- 5 top-norm-by-rank control q1
    rows per model (coord_role rank2_by_norm..rank6_by_norm), all 22 models in one file.

Output:
  audit/round2/figures/fig1c_full22_two_panel.{png,pdf}
  audit/round2/figures/fig1_panel_c_full22_source.csv
  manuscript/figures/source_data/fig1_panel_c_full22_source.csv
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
RESULTS = ROOT / "results"
AUDIT2 = ROOT / "audit" / "round2"
FIG_DIR = AUDIT2 / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR2 = ROOT / "manuscript" / "figures" / "source_data"
SRC_DIR2.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from _figstyle import apply_style, panel_label  # noqa: E402

sys.path.insert(0, str(ROOT / "manuscript" / "figures"))
from _paper_encoding import DOMAIN_COLOR, ARCH_MARKER  # noqa: E402

# ---- display-name mapping, extended to the full 22-model cohort --------------------
# Raw model-name -> short display label. Reuses the conventions already used in
# fig1_structural.py's key()/display() helpers, extended to cover all 22 raw names that
# appear in full22_topk_gap_confounds.csv.
DISPLAY = {
    "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
    "HuggingFaceTB/SmolLM2-135M": "SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M": "SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B": "SmolLM2-1.7B",
    "GenerTeam/GENERator-v2-prokaryote-1.2b-base": "GEN-PROK-1.2B",
    "GenerTeam/GENERator-v2-prokaryote-3b-base": "GEN-PROK-3B",
    "EuroBERT/EuroBERT-210m": "EuroBERT-210M",
    "EuroBERT/EuroBERT-610m": "EuroBERT-610M",
    "EuroBERT/EuroBERT-2.1B": "EuroBERT-2.1B",
    "answerdotai/ModernBERT-large": "ModernBERT-large",
    "ModernBERT-base": "ModernBERT-base",
    "DNABERT-2": "DNABERT-2",
    "GENERator-EUK-3B": "GEN-EUK-3B",
    "Llama-7B": "Llama-7B",
    "Mistral-7B": "Mistral-7B",
    "OLMo-7B-0724-hf": "OLMo-7B",
    "MosaicBERT": "MosaicBERT",
    "GenomeOcean-4B": "GenomeOcean-4B",
    "Qwen2.5-7B": "Qwen2.5-7B",
    "NTv3": "NTv3",
}

NEGATIVE_TOPNORM_MODELS = {
    "HuggingFaceTB/SmolLM2-360M", "EuroBERT/EuroBERT-210m", "EuroBERT/EuroBERT-2.1B",
    "answerdotai/ModernBERT-large", "DNABERT-2", "GENERator-EUK-3B", "GenomeOcean-4B",
}


def main():
    apply_style()

    summary = list(csv.DictReader(open(AUDIT2 / "fig1c_random_vs_topk_gaps_full22.csv")))
    assert len(summary) == 22, f"expected 22 models, got {len(summary)}"
    meta = {r["model"]: r for r in summary}

    # 5 random controls per model (original 12 + batch2 10)
    random_ctrl = defaultdict(list)
    for r in csv.DictReader(open(RESULTS / "E11" / "scale_ladder_controls.csv")):
        random_ctrl[r["model"]].append(float(r["q1"]))
    for r in csv.DictReader(open(AUDIT2 / "section2_random_control_q1_batch2.csv")):
        random_ctrl[r["model"]].append(float(r["q1"]))
    for m in meta:
        assert len(random_ctrl[m]) == 5, f"{m}: expected 5 random controls, got {len(random_ctrl[m])}"

    # 5 top-norm-by-rank controls per model (rank2..rank6, all 22 models in one file)
    topnorm_ctrl = defaultdict(list)
    for r in csv.DictReader(open(AUDIT2 / "topk_norm_q1_summary.csv")):
        if r["coord_role"] != "candidate":
            topnorm_ctrl[r["model"]].append(float(r["q1"]))
    for m in meta:
        assert len(topnorm_ctrl[m]) == 5, f"{m}: expected 5 top-norm controls, got {len(topnorm_ctrl[m])}"

    # order: by random_control_gap descending (same order fig1c_full22.py uses, and the
    # order Fig 2's new panel D also uses, so the two figures cross-reference by position)
    order = sorted(meta, key=lambda m: -float(meta[m]["random_control_gap"]))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True, constrained_layout=True)

    xs = np.arange(len(order))
    for x, m in zip(xs, order):
        r = meta[m]
        arch, dom = r["architecture"], r["domain"]
        cand_q1 = float(r["candidate_q1"])
        axL.scatter(np.full(5, x) + np.linspace(-0.16, 0.16, 5), random_ctrl[m],
                    s=22, facecolors="none", edgecolors=DOMAIN_COLOR[dom], linewidths=0.8, zorder=3)
        axL.scatter(x, cand_q1, s=58, marker=ARCH_MARKER[arch], color=DOMAIN_COLOR[dom],
                    edgecolor="black", zorder=4)

        axR.scatter(np.full(5, x) + np.linspace(-0.16, 0.16, 5), topnorm_ctrl[m],
                    s=22, facecolors="none", edgecolors=DOMAIN_COLOR[dom], linewidths=0.8, zorder=3)
        axR.scatter(x, cand_q1, s=58, marker=ARCH_MARKER[arch], color=DOMAIN_COLOR[dom],
                    edgecolor="black", zorder=4)
        if m in NEGATIVE_TOPNORM_MODELS:
            axR.annotate(DISPLAY[m], (x, cand_q1), xytext=(0, 8), textcoords="offset points",
                         ha="center", va="bottom", fontsize=5.6, color="0.2",
                         arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))

    labels = [DISPLAY[m] for m in order]
    subtitles = {axL: "random same-layer controls", axR: "top-norm same-layer controls"}
    for ax, sub in ((axL, "random"), (axR, "top-norm")):
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=58, ha="right", fontsize=6.4)
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_title(subtitles[ax], fontsize=8, fontweight="normal", loc="left", pad=3)
    axL.set_ylim(0, 1.18)
    axL.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axL.set_ylabel(r"$q_1$ (candidate and same-layer controls)")

    med_rc = np.median([float(meta[m]["random_control_gap"]) for m in order])
    med_tk = np.median([float(meta[m]["topk_by_norm_gap"]) for m in order])
    axL.text(0.02, 0.99, f"median gap = {med_rc:.3f}\nn = 22", transform=axL.transAxes,
              fontsize=7, va="top", ha="left")
    axR.text(0.02, 0.99, f"median gap = {med_tk:.3f}\nn = 22, 7 negative", transform=axR.transAxes,
              fontsize=7, va="top", ha="left")

    axL.legend(handles=[ml.Line2D([], [], marker="o", color="black", linestyle="", label="candidate", markerfacecolor=".35"),
                         ml.Line2D([], [], marker="o", color=".35", linestyle="", label="5 controls", markerfacecolor="none")],
               frameon=False, fontsize=6.5, loc="lower left")
    arch_dom_handles = [ml.Line2D([], [], marker="o", color=".35", linestyle="", label="decoder"),
                         ml.Line2D([], [], marker="s", color=".35", linestyle="", label="encoder"),
                         ml.Line2D([], [], marker="o", color=DOMAIN_COLOR["text"], linestyle="", label="text"),
                         ml.Line2D([], [], marker="o", color=DOMAIN_COLOR["genomic"], linestyle="", label="genomic")]
    axR.legend(handles=arch_dom_handles, frameon=False, fontsize=6, ncol=2, loc="lower left")

    # Single spanning "C" label (one letter per panel *position*, not per sub-panel),
    # matching journal convention; the two halves are distinguished by their subtitles
    # above instead of separate C1/C2 labels.
    panel_label(axL, "C")

    out = FIG_DIR / "fig1c_full22_two_panel"
    fig.savefig(out.with_suffix(".png")); fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}.png / .pdf")

    # ---- source-data CSV --------------------------------------------------------
    fieldnames = ["model", "display_name", "architecture", "domain", "candidate_q1",
                  "random_control_q1_1", "random_control_q1_2", "random_control_q1_3",
                  "random_control_q1_4", "random_control_q1_5", "random_control_gap",
                  "topnorm_control_q1_1", "topnorm_control_q1_2", "topnorm_control_q1_3",
                  "topnorm_control_q1_4", "topnorm_control_q1_5", "topnorm_control_gap",
                  "negative_topnorm_gap_flag"]
    rows_out = []
    for m in order:
        r = meta[m]
        row = dict(model=m, display_name=DISPLAY[m], architecture=r["architecture"], domain=r["domain"],
                   candidate_q1=r["candidate_q1"], random_control_gap=r["random_control_gap"],
                   topnorm_control_gap=r["topk_by_norm_gap"],
                   negative_topnorm_gap_flag=int(m in NEGATIVE_TOPNORM_MODELS))
        for i, v in enumerate(random_ctrl[m], 1):
            row[f"random_control_q1_{i}"] = v
        for i, v in enumerate(topnorm_ctrl[m], 1):
            row[f"topnorm_control_q1_{i}"] = v
        rows_out.append(row)

    for dest in (FIG_DIR / "fig1_panel_c_full22_source.csv", SRC_DIR2 / "fig1_panel_c_full22_source.csv"):
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows_out)
        print(f"wrote {dest}")

    print(f"\nmedian random_control_gap = {med_rc:.4f}, median topnorm gap = {med_tk:.4f}")
    print(f"negative topnorm-gap models ({len(NEGATIVE_TOPNORM_MODELS)}): "
          f"{sorted(DISPLAY[m] for m in NEGATIVE_TOPNORM_MODELS)}")


if __name__ == "__main__":
    main()
