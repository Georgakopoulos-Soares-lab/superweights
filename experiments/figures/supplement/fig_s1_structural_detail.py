#!/usr/bin/env python3
"""
Supplement S1 -- full structural / control-row detail behind Figures 1 and 2.

  a. Phi-3's 6 individual published rows' exact q1 (Fig 1 only shows the model-level
     median) -- shows the within-model spread that the aggregate hides.
  b. All 5 decoders' same-layer control-row causal effects (relative %NLL change),
     the individual points behind Figure 2A's "control median" dotted line.
  c. Local spectral exceptionalness (candidate q1 minus mean same-layer-control q1)
     against non-embedding parameter count, 12-model common-control panel. Round-3:
     demoted here from the main Fig. 1 (formerly panel D) -- Fig. 1B already tests q1
     against scale/architecture on the full 23-model panel and Fig. 1C's top-norm
     comparison is the stronger structural result, so this panel was mostly redundant
     with B while adding its own one-off term ("local spectral exceptionalness"). Not
     deleted, only moved: same data, same script logic as the old panel D.

Source: results/experiments/E7/e7_phi3_spectral.json, results/experiments/E10/e10_decoder_concentration.json,
        experiments/figures/source_data/fig1_structural_panel_bcd.csv (written by
        experiments/figures/fig1_structural.py).
Output: figures/supplement/fig_s1_structural_detail.{png,pdf}
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent
PAPER_SALVAGE = FIGURES.parent
REPO_ROOT = PAPER_SALVAGE.parent
RESULTS = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "experiments" / "figures"))
sys.path.insert(0, str(FIGURES))
from _figstyle import apply_style, panel_label  # noqa: E402
from _paper_encoding import DOMAIN_COLOR, ARCH_MARKER  # noqa: E402

ARCH = {m: 'decoder' for m in ['Qwen2.5-0.5B', 'Qwen2.5-1.5B', 'Qwen2.5-3B', 'SmolLM2-135M',
        'SmolLM2-360M', 'SmolLM2-1.7B', 'GENERator-PROK-1.2B', 'GENERator-PROK-3B']}
ARCH.update({m: 'encoder' for m in ['EuroBERT-210M', 'EuroBERT-610M', 'EuroBERT-2.1B', 'ModernBERT-large']})

DECODERS = ["llama", "mistral", "olmo", "phi3", "qwen25"]
DECODER_LABEL = {"llama": "Llama", "mistral": "Mistral", "olmo": "OLMo",
                  "phi3": "Phi-3", "qwen25": "Qwen2.5"}


def main():
    apply_style()
    phi3 = json.loads((RESULTS / "e7_phi3_spectral.json").read_text())
    conc = json.loads((RESULTS / "e10_decoder_concentration.json").read_text())

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(10.6, 3.6))

    # --- Panel a: Phi-3 per-row q1 ---------------------------------------------------
    rows = phi3["rows"]
    labels = [f"L{r['layer']}/r{r['row']}" for r in rows]
    q1s = [r["q1"] for r in rows]
    y = np.arange(len(rows))
    ax_a.barh(y, q1s, color=DOMAIN_COLOR["text"], edgecolor="black", linewidth=0.6)
    ax_a.axvline(phi3["model_level"]["q1_median"], color="0.3", ls="--", lw=0.9)
    ax_a.text(phi3["model_level"]["q1_median"] + 0.01, -0.7, "model-level median\n(used in Fig. 1B)",
              fontsize=6, color="0.3")
    ax_a.set_yticks(y); ax_a.set_yticklabels(labels, fontsize=7)
    ax_a.set_xlabel(r"$q_1$")
    ax_a.set_xlim(0, 1.05)
    panel_label(ax_a, "a")

    # --- Panel b: decoder control-row causal effects ----------------------------------
    xpos = 0
    xticks, xticklabels = [], []
    for m in DECODERS:
        model = conc["models"][m]
        pcts = [abs(c["relative_pct_change"]) for c in model["controls"]]
        jitter = np.random.default_rng(1).uniform(-0.15, 0.15, size=len(pcts))
        ax_b.scatter(np.full(len(pcts), xpos) + jitter, pcts, facecolors="none",
                     edgecolor=DOMAIN_COLOR["text"], s=30, linewidths=0.8, zorder=2)
        xticks.append(xpos); xticklabels.append(DECODER_LABEL[m])
        xpos += 1
    ax_b.set_yscale("log")
    ax_b.set_xticks(xticks); ax_b.set_xticklabels(xticklabels, rotation=0, fontsize=7.5)
    ax_b.set_ylabel("|relative NLL change| (%, log)\nsame-layer control rows")
    panel_label(ax_b, "b")

    # --- Panel c: local spectral exceptionalness vs. scale (formerly main Fig. 1D) ----
    bcd_path = FIGURES / "source_data" / "fig1_structural_panel_bcd.csv"
    bcd_rows = list(csv.DictReader(bcd_path.open()))
    bcd_rows.sort(key=lambda r: float(r["non_embedding_params"]))
    offsets = [(4, 4), (4, -13), (4, 13), (4, -22)]
    for i, r in enumerate(bcd_rows):
        model, params, gap = r["model"], float(r["non_embedding_params"]), float(r["candidate_control_gap"])
        arch, dom = ARCH.get(model, r["architecture"]), r["domain"]
        ax_c.scatter(params, gap, s=42, marker=ARCH_MARKER[arch], color=DOMAIN_COLOR[dom], edgecolor="black", zorder=3)
        ax_c.annotate(model.replace("GENERator-", "GEN-"), (params, gap), xytext=offsets[i % len(offsets)],
                      textcoords="offset points", fontsize=5.2, ha="left", va="center")
    ax_c.set_xscale("log")
    ax_c.set_xlabel("non-embedding parameters")
    ax_c.set_ylabel(r"candidate $q_1$ $-$ mean control $q_1$" + "\n(12-model common-control panel)")
    ax_c.set_xlim(6e7, 1.2e10)
    ax_c.set_ylim(0, 1.05)
    panel_label(ax_c, "c")

    fig.tight_layout()
    out = HERE / "fig_s1_structural_detail"
    fig.savefig(f"{out}.png"); fig.savefig(f"{out}.pdf")

    src_out = FIGURES / "source_data" / "fig_s1_structural_detail.json"
    src_out.write_text(json.dumps(dict(phi3_rows=phi3, decoder_controls={
        m: conc["models"][m]["controls"] for m in DECODERS},
        panel_c_note="see fig1_structural_panel_bcd.csv for panel c's raw data"), indent=1))
    print(f"saved -> {out}.png / .pdf, {src_out}")


if __name__ == "__main__":
    main()
