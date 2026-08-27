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
  D. (round-2 addition, audit/round2/final_check.md Section 2) Candidate vs. top-norm-
     control causal effect at both epsilons, full 22-model cohort, symlog y-axis. The 4
     sign-flip models (MosaicBERT, Qwen2.5-7B, Qwen2.5-0.5B, NTv3) are labeled. Model
     x-order matches Fig. 1's panel C so position is cross-referenceable between figures.
     Folded in from the standalone audit/round2/scripts/fig2_panel_topnorm.py.

Source: results/e10_decoder_concentration.json, results/e10_exact_uknorm_olmo.json,
results/e10_encoder_fit_results.json, results/e10b_phi3_fit_results.json
(repo-root results/); panel D: audit/round2/structural_vs_causal_gap.csv.
See FIGURE_PROVENANCE.md.
Output: figures/main/fig2_causal.{png,pdf}, figures/source_data/fig2_causal.json

E10b (Phi-3 pair tomography) is now frozen (as of this update) and included in panel C.
"""
from __future__ import annotations

import csv
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
from _paper_encoding import DOMAIN_COLOR, ARCH_MARKER  # noqa: E402

DECODERS = ["llama", "mistral", "olmo", "phi3", "qwen25"]
DECODER_LABEL = {"llama": "Llama", "mistral": "Mistral", "olmo": "OLMo",
                  "phi3": "Phi-3", "qwen25": "Qwen2.5"}
DECISION_COLOR = {"SINGLE_COMPONENT_DOMINANT": "#4C72B0", "MULTI_COMPONENT_CANDIDATE": "#C44E52"}

AUDIT2 = REPO_ROOT / "audit" / "round2"

# panel D (round-2): same display-name map used in fig1_structural.py's panel C / the
# standalone audit/round2/scripts/fig2_panel_topnorm.py.
DISPLAY_D = {
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
SIGN_FLIP_MODELS_D = {"MosaicBERT", "Qwen2.5-7B", "Qwen/Qwen2.5-0.5B", "NTv3"}
EPS_STYLE_D = {
    "0.5": dict(fill=True, s=52, offset=-0.14),
    "1.0": dict(fill=False, s=68, offset=0.14),
}


def load_all():
    conc = json.loads((RESULTS / "e10_decoder_concentration.json").read_text())
    olmo_struct = json.loads((RESULTS / "e10_exact_uknorm_olmo.json").read_text())
    enc = json.loads((RESULTS / "e10_encoder_fit_results.json").read_text())
    phi3b = json.loads((RESULTS / "e10b_phi3_fit_results.json").read_text())
    return conc, olmo_struct, enc, phi3b


def main():
    apply_style()
    conc, olmo_struct, enc, phi3b = load_all()

    fig = plt.figure(figsize=(11.5, 10.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1.35])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, :])

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
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.05, 0.98), frameon=False, fontsize=6.6)
    panel_label(ax_c, "C")

    # --- Panel D (round-2): candidate vs. top-norm-control causal gap, both epsilons ----
    fig1_summary = list(csv.DictReader(open(AUDIT2 / "fig1c_random_vs_topk_gaps_full22.csv")))
    fig1_meta = {r["model"]: r for r in fig1_summary}
    order_d = sorted(fig1_meta, key=lambda m: -float(fig1_meta[m]["random_control_gap"]))
    assert len(order_d) == 22

    causal_d = list(csv.DictReader(open(AUDIT2 / "structural_vs_causal_gap.csv")))
    assert len(causal_d) == 44
    by_model_eps_d = {(r["model"], r["epsilon"]): r for r in causal_d}

    xs_d = np.arange(len(order_d))
    neg_offsets_d = [-13, -35, -57]
    for x, m in zip(xs_d, order_d):
        arch, dom = fig1_meta[m]["architecture"], fig1_meta[m]["domain"]
        neg_eps_count = 0
        for eps in ("0.5", "1.0"):
            r = by_model_eps_d[(m, eps)]
            gap = float(r["causal_topk_gap"])
            st = EPS_STYLE_D[eps]
            kw = dict(marker=ARCH_MARKER[arch], s=st["s"], zorder=4)
            if st["fill"]:
                kw.update(color=DOMAIN_COLOR[dom], edgecolor="black", linewidths=0.9)
            else:
                kw.update(facecolors="none", edgecolors=DOMAIN_COLOR[dom], linewidths=1.2)
            ax_d.scatter(x + st["offset"], gap, **kw)
            if m in SIGN_FLIP_MODELS_D and gap < 0:
                dy = neg_offsets_d[neg_eps_count % len(neg_offsets_d)]
                ax_d.annotate(f"{DISPLAY_D[m]}, " + r"$\epsilon$=" + eps, (x + st["offset"], gap),
                              xytext=(0, dy), textcoords="offset points", ha="center", va="top",
                              fontsize=5.4, color="0.15",
                              arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5"))
                neg_eps_count += 1

    ax_d.axhline(0, color="black", lw=0.8, zorder=2)
    ax_d.set_yscale("symlog", linthresh=1e-4)
    ax_d.set_ylim(-1.2, 15)
    ax_d.set_ylabel(r"causal $\Delta$NLL gap: candidate $-$ median top-norm control" "\n(relative NLL change, symlog)")
    ax_d.set_xticks(xs_d)
    ax_d.set_xticklabels([DISPLAY_D[m] for m in order_d], rotation=58, ha="right", fontsize=6.4)
    ax_d.set_xlim(-0.7, len(order_d) - 0.3)

    leg_d = [mlines.Line2D([], [], marker="o", color="black", linestyle="", markerfacecolor=".35",
                            label=r"$\epsilon$=0.5 (filled)"),
             mlines.Line2D([], [], marker="o", color=".35", linestyle="", markerfacecolor="none",
                            markeredgewidth=1.2, label=r"$\epsilon$=1.0 (open)"),
             mlines.Line2D([], [], marker="o", color=".35", linestyle="", label="decoder"),
             mlines.Line2D([], [], marker="s", color=".35", linestyle="", label="encoder"),
             mlines.Line2D([], [], marker="o", color=DOMAIN_COLOR["text"], linestyle="", label="text"),
             mlines.Line2D([], [], marker="o", color=DOMAIN_COLOR["genomic"], linestyle="", label="genomic")]
    ax_d.legend(handles=leg_d, frameon=False, fontsize=6.2, ncol=3, loc="upper right")

    n_neg_05 = sum(1 for m in order_d if float(by_model_eps_d[(m, "0.5")]["causal_topk_gap"]) < 0)
    n_neg_10 = sum(1 for m in order_d if float(by_model_eps_d[(m, "1.0")]["causal_topk_gap"]) < 0)
    ax_d.text(0.01, 0.03, f"negative gap: {n_neg_05}/22 at eps=0.5, {n_neg_10}/22 at eps=1.0",
              transform=ax_d.transAxes, fontsize=6.5, va="bottom", ha="left", color="0.25")
    panel_label(ax_d, "D")

    out = HERE / "main" / "fig2_causal"
    fig.savefig(f"{out}.png"); fig.savefig(f"{out}.pdf")

    src_out = HERE / "source_data" / "fig2_causal.json"
    src_out.write_text(json.dumps(dict(decoder_concentration=conc, olmo_structural=olmo_struct,
                                        encoder_fit=enc, phi3_e10b_fit=phi3b), indent=1))
    print(f"saved -> {out}.png / .pdf, {src_out}")


if __name__ == "__main__":
    main()
