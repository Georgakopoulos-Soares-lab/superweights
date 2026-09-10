#!/usr/bin/env python3
"""
Figure 2 -- Cross-model causal organization.

Three panels, all loaded directly from raw E10 JSON:
  A. Decoder singleton causal spectrum (relative %change in causal-LM NLL vs. control
     median), all 5 decoders (Llama/Mistral/Qwen K=1; OLMo K=4; Phi-3 K=6).
  B. OLMo structural-vs-causal rank dissociation: structurally-largest row (L24/r269)
     is causally weakest; causally-dominant row (L1/r269) ranks 3rd structurally.
  C. F2->F3 held-out MAE improvement, both epsilon scales, with bootstrap 95% CI, for
     MosaicBERT + ModernBERT (encoders -- replicates DNABERT-2's, Fig 3C,
     interactional phenotype) AND Phi-3 (E10b, the one decoder E10 flagged
     MULTI_COMPONENT_CANDIDATE and therefore eligible for tomography). Phi-3's own
     result SPLITS by epsilon: pair terms required at eps=0.5, but at eps=1.0 the
     pairwise model is reliably WORSE than additive (negative bar) -- driven by a
     specific three-way redundancy break among its three layer-2 rows that no
     second-order model in this ladder can represent. Reported honestly as split,
     not forced into either class.

Source: results/E10/e10_decoder_concentration.json, results/E10/e10_exact_uknorm_olmo.json,
results/E10/e10_encoder_fit_results.json, results/E10/e10b_phi3_fit_results.json
(repo-root results/). See FIGURE_PROVENANCE.md.
Output: figures/main/fig2_causal.{png,pdf}, figures/source_data/fig2_causal.json

E10b (Phi-3 pair tomography) is now frozen (as of this update) and included in panel C.

NOTE (round-2 correction): this is NOT the manuscript's Figure 2 -- the 22-model
functional-criticality census (build_part2_evidence_packet.py's fig2_part2_functional_
criticality.png) is. This decoder-spectrum/OLMo-dissociation/tomography-adequacy figure
predates that census and is retained per PART2_EVIDENCE_PACKET.md's own note ("retain
tomography only as mechanistic case-study material") -- i.e. as supplementary/case-study
material, not the main-text Figure 2. Do not add the round-2 panel D (candidate vs.
top-norm-control gap) here; it belongs on the real Figure 2 instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

HERE = Path(__file__).resolve().parent
PAPER_SALVAGE = HERE.parent
REPO_ROOT = PAPER_SALVAGE.parent
RESULTS = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
from _figstyle import apply_style, panel_label  # noqa: E402
from _paper_encoding import DOMAIN_COLOR  # noqa: E402

DECODERS = ["llama", "mistral", "olmo", "phi3", "qwen25"]
DECODER_LABEL = {"llama": "Llama", "mistral": "Mistral", "olmo": "OLMo",
                  "phi3": "Phi-3", "qwen25": "Qwen2.5"}
DECISION_COLOR = {"SINGLE_COMPONENT_DOMINANT": "#4C72B0", "MULTI_COMPONENT_CANDIDATE": "#C44E52"}


def load_all():
    conc = json.loads((RESULTS / "e10_decoder_concentration.json").read_text())
    olmo_struct = json.loads((RESULTS / "e10_exact_uknorm_olmo.json").read_text())
    enc = json.loads((RESULTS / "e10_encoder_fit_results.json").read_text())
    phi3b = json.loads((RESULTS / "e10b_phi3_fit_results.json").read_text())
    return conc, olmo_struct, enc, phi3b


def main():
    apply_style()
    conc, olmo_struct, enc, phi3b = load_all()

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # --- Panel A: decoder singleton causal spectrum --------------------------------
    xpos = 0
    xticks, xticklabels = [], []
    for m in DECODERS:
        model = conc["models"][m]
        decision = model["decision"]
        color = DECISION_COLOR[decision]
        n = len(model["target_rows"])
        offsets = np.linspace(-0.25, 0.25, n) if n > 1 else [0.0]
        for off, t in zip(offsets, model["target_rows"]):
            pct = abs(t["relative_pct_change"])
            ax_a.scatter(xpos + off, pct, color=color, edgecolor="black", linewidths=0.7,
                         s=55, zorder=3)
        ctrl_pct = model["control_median_abs_delta"] / model["baseline_nll"] * 100
        ax_a.hlines(ctrl_pct, xpos - 0.35, xpos + 0.35, color="0.6", lw=1.0, ls=":", zorder=2)
        xticks.append(xpos)
        xticklabels.append(f"{DECODER_LABEL[m]}\n(K={model['K']})")
        xpos += 1
    ax_a.set_yscale("log")
    ax_a.set_ylabel("|relative NLL change| at full ablation (%, log)")
    ax_a.set_xticks(xticks); ax_a.set_xticklabels(xticklabels, fontsize=7.2)
    ax_a.set_xlim(-0.6, xpos - 0.4)
    leg = [mlines.Line2D([], [], marker="o", color="none", markerfacecolor=DECISION_COLOR[k],
                          markeredgecolor="black", markersize=7, label=k.replace("_", " ").title())
           for k in DECISION_COLOR]
    leg.append(mlines.Line2D([], [], color="0.6", lw=1.2, ls=":", label="control median"))
    ax_a.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3,
                frameon=False, fontsize=6.6, handletextpad=0.4, columnspacing=1.2)
    panel_label(ax_a, "A")

    # --- Panel B: OLMo structural-vs-causal dissociation -----------------------------
    rows = conc["models"]["olmo"]["target_rows"]
    row_layers = {t["layer"]: t for t in rows}
    struct = {int(L): v["layer_max_uk"] for L, v in olmo_struct["layers"].items()}
    layers_sorted_struct = sorted(struct, key=lambda L: -struct[L])
    layers_sorted_causal = sorted(row_layers, key=lambda L: -row_layers[L]["relative_pct_change"])

    y_struct = {L: i for i, L in enumerate(layers_sorted_struct)}
    y_causal = {L: i for i, L in enumerate(layers_sorted_causal)}
    for L in row_layers:
        ax_b.plot([0, 1], [y_struct[L], y_causal[L]], color="0.75", lw=1.3, zorder=1,
                  marker="o", mfc=DOMAIN_COLOR["text"], mec="black", mew=0.7, ms=7)
        ax_b.text(-0.05, y_struct[L], f"L{L}", ha="right", va="center", fontsize=7)
        ax_b.text(1.05, y_causal[L], f"L{L}", ha="left", va="center", fontsize=7)
    ax_b.set_xlim(-0.3, 1.3)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["structural rank\n(exact $\\|U_k\\|_F$)", "causal rank\n(|%NLL change|)"],
                          fontsize=7)
    ax_b.set_ylim(-0.6, 3.6)
    ax_b.invert_yaxis()
    ax_b.set_yticks([])
    ax_b.set_ylabel("rank (top = 0)")
    panel_label(ax_b, "B")

    # --- Panel C: F2->F3 improvement, encoders + the one eligible decoder (Phi-3/E10b) --
    model_specs = [("mosaicbert", "MosaicBERT\n(encoder)", enc["models"]["mosaicbert"]["by_epsilon"]),
                   ("modernbert", "ModernBERT\n(encoder)", enc["models"]["modernbert"]["by_epsilon"]),
                   ("phi3", "Phi-3\n(decoder, E10b)", phi3b["by_epsilon"])]
    epsilons = ["0.5", "1.0"]
    width = 0.35
    xs = np.arange(len(model_specs))
    for j, eps in enumerate(epsilons):
        vals, err_lo, err_hi = [], [], []
        for _, _, by_eps in model_specs:
            cell = by_eps[eps]
            mae_f2 = cell["F2"]["metrics"]["mae"]
            imp = cell["F2_vs_F3_relative_MAE_improvement"] * 100
            boot = cell["F2_vs_F3_bootstrap"]
            ci_lo_pct = boot["ci_2.5"] / mae_f2 * 100
            ci_hi_pct = boot["ci_97.5"] / mae_f2 * 100
            vals.append(imp)
            err_lo.append(max(imp - ci_lo_pct, 0))
            err_hi.append(max(ci_hi_pct - imp, 0))
        offs = (j - 0.5) * width
        bars = ax_c.bar(xs + offs, vals, width=width * 0.9, color=DOMAIN_COLOR["text"],
                        alpha=0.55 + 0.35 * j, edgecolor="black", linewidth=0.7,
                        yerr=[err_lo, err_hi], capsize=3, ecolor="0.3", error_kw=dict(lw=0.9),
                        label=rf"$\epsilon$={eps}")
        bars[2].set_hatch("///")  # flag Phi-3 as the decoder/E10b bar, same color scheme
    ax_c.set_xticks(xs); ax_c.set_xticklabels([s[1] for s in model_specs], fontsize=6.8)
    ax_c.set_ylabel(r"F2$\to$F3 held-out MAE improvement (%)")
    ax_c.axhline(0, color="0.3", lw=0.8)
    ax_c.axhline(10, color="0.6", lw=0.8, ls="--")
    ax_c.text(2.45, 12, "prereg adequacy\nbar (10%)", fontsize=5.4, color="0.45", ha="right", va="bottom")
    ax_c.text(2.0, 63, "Phi-3 splits: pair terms\nhelp at $\\epsilon$=0.5,\nreliably hurt at $\\epsilon$=1.0",
              fontsize=5.8, color="#C44E52", ha="center", va="bottom", style="italic")
    ax_c.legend(loc="upper left", frameon=False, fontsize=6.6)
    panel_label(ax_c, "C")

    fig.tight_layout()
    out = HERE / "main" / "fig2_causal"
    fig.savefig(f"{out}.png"); fig.savefig(f"{out}.pdf")

    src_out = HERE / "source_data" / "fig2_causal.json"
    src_out.write_text(json.dumps(dict(decoder_concentration=conc, olmo_structural=olmo_struct,
                                        encoder_fit=enc, phi3_e10b_fit=phi3b), indent=1))
    print(f"saved -> {out}.png / .pdf, {src_out}")


if __name__ == "__main__":
    main()
