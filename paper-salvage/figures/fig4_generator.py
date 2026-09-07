#!/usr/bin/env python3
"""
Figure 4 -- GENERator: a BOS-localized causal mechanism at a decoder high-gain pathway.

Four panels:
  A. Minimal structural callout -- GENERator EUK row 2371's exact q1 (near-rank-1),
     cross-referencing Figure 1B rather than restating it. UNCHANGED.
  B. BOS-centered attention/activation phenotype (results/mechanism/attention_sink_
     implicit_bias.json). Framed strictly as association/co-occurrence -- the high-gain
     row's activation and incoming attention both concentrate at position 0, but this does
     not establish that the row CAUSES the sink. UNCHANGED.
  C. (NEW, 2026-09-01, replaces the old random-direction panel -- see Supplement S2) Native
     NLL under the six BOS-mediation conditions from results/E_BOS_MEDIATION/
     bos_mediation_results.json: intact, full row-2371 ablation, BOS-only ablation,
     preserve-BOS-only, restore-at-BOS, restore-at-matched-non-BOS. Loss of the row's BOS
     contribution alone reproduces essentially the full ablation damage; preserving or
     restoring only that BOS contribution rescues essentially all of it; restoring the same
     magnitude at a matched non-BOS position does not.
  D. (NEW) Generated GC fraction under the same six conditions -- the secondary phenotype
     mostly tracks the same BOS-specific pattern, with one honestly-reported anomaly: BOS-
     only ablation lowers GC slightly *below* the full-ablation level rather than sitting at
     it (not explained here, plotted as measured).

Provenance gate and 6 smoke/invariance tests behind panels C/D both passed; see
results/E_BOS_MEDIATION/{provenance_check,smoke_tests}.json and
results/E13/PRIORITY1_2_3_INTEGRATED_REPORT.md.

The pre-2026-09-01 panel C (random-direction / damage-tracking control) is NOT deleted -- it
answers a different question (direction-specificity vs. location-sensitivity) and is retained
essentially unchanged as candidate Supplementary Figure S2
(figures/supplement/fig_s2_random_direction.py).

Source: results/e7_legacy_reanalysis.json, results/mechanism/attention_sink_implicit_bias.json,
results/E_BOS_MEDIATION/bos_mediation_results.json.
Output: figures/main/fig4_generator.{png,pdf}, figures/source_data/fig4_generator.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PAPER_SALVAGE = HERE.parent
REPO_ROOT = PAPER_SALVAGE.parent
RESULTS = REPO_ROOT / "results"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
from _figstyle import apply_style, panel_label  # noqa: E402
from _paper_encoding import DOMAIN_COLOR  # noqa: E402

BOS_CONDITIONS = [
    ("A_intact", "Intact"),
    ("B_full_ablation_weight", "Full\nablation"),
    ("C_ablate_bos_only", "BOS-only\nablation"),
    ("D_ablate_all_except_bos", "Preserve\nBOS"),
    ("E_restore_at_bos", "Restore\nBOS"),
    ("F_restore_at_matched_nonbos", "Restore\nnon-BOS"),
]

# spec values from the okstillnotlast.md follow-up request -- checked against the artifact
# at load time (see the assert loop in main()), not used for plotting directly.
EXPECTED_NLL = {
    "A_intact": 6.385, "B_full_ablation_weight": 8.754, "C_ablate_bos_only": 8.722,
    "D_ablate_all_except_bos": 6.386, "E_restore_at_bos": 6.386,
    "F_restore_at_matched_nonbos": 8.754,
}
EXPECTED_GC = {
    "A_intact": 0.419, "B_full_ablation_weight": 0.309, "C_ablate_bos_only": 0.292,
    "D_ablate_all_except_bos": 0.412, "E_restore_at_bos": 0.412,
    "F_restore_at_matched_nonbos": 0.303,
}


def main():
    apply_style()
    legacy = json.loads((RESULTS / "e7_legacy_reanalysis.json").read_text())["GENERator EUK"]
    sink = json.loads((RESULTS / "mechanism" / "attention_sink_implicit_bias.json").read_text())["generator"]
    bos = json.loads((RESULTS / "E_BOS_MEDIATION" / "bos_mediation_results.json").read_text())

    # verify every plotted number directly against the artifact before plotting anything
    for key, _ in BOS_CONDITIONS:
        measured_nll = bos["nll_pooled_mean_by_condition"][key]
        measured_gc = bos["gc_mean_by_condition"][key]
        assert abs(measured_nll - EXPECTED_NLL[key]) < 0.01, (key, measured_nll, EXPECTED_NLL[key])
        assert abs(measured_gc - EXPECTED_GC[key]) < 0.002, (key, measured_gc, EXPECTED_GC[key])
    assert bos["provenance_check"]["intact_ok"] and bos["provenance_check"]["ablated_ok"]
    assert all(v["pass"] for v in bos["smoke_tests"].values())

    fig = plt.figure(figsize=(10.5, 7.6), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.15])
    top = outer[0].subgridspec(1, 2, width_ratios=[0.62, 1.0])
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(top[0, 0]); ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bot[0, 0]); ax_d = fig.add_subplot(bot[0, 1])

    # --- Panel A: minimal structural callout (unchanged) ------------------------------
    ax_a.bar([0], [legacy["q1"]], color=DOMAIN_COLOR["genomic"], edgecolor="black",
             linewidth=0.8, width=0.5)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xticks([0]); ax_a.set_xticklabels(["GENERator EUK\n(L4/r2371)"], fontsize=7.5)
    ax_a.set_ylabel(r"$q_1$ (exact, see Fig. 1B)")
    ax_a.text(0, legacy["q1"] + 0.03, f"{legacy['q1']:.3f}", ha="center", fontsize=7.5)
    panel_label(ax_a, "A")

    # --- Panel B: BOS-centered attention/activation phenotype (unchanged) -------------
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

    # --- Panel C (NEW): native NLL across the six BOS-mediation conditions ------------
    xs = list(range(len(BOS_CONDITIONS)))
    nll_vals = [bos["nll_pooled_mean_by_condition"][k] for k, _ in BOS_CONDITIONS]
    labels = [lab for _, lab in BOS_CONDITIONS]
    intact_nll = bos["nll_pooled_mean_by_condition"]["A_intact"]
    ablated_nll = bos["nll_pooled_mean_by_condition"]["B_full_ablation_weight"]

    ax_c.axhline(intact_nll, color="black", linestyle=":", linewidth=1.1, zorder=1,
                 label=f"intact (NLL={intact_nll:.3f})")
    ax_c.axhline(ablated_nll, color="0.35", linestyle="--", linewidth=1.1, zorder=1,
                 label=f"full ablation (NLL={ablated_nll:.3f})")
    ax_c.bar(xs, nll_vals, color=DOMAIN_COLOR["genomic"], edgecolor="black",
             linewidth=0.8, width=0.6, zorder=3)
    ax_c.set_xticks(xs); ax_c.set_xticklabels(labels, fontsize=7)
    ax_c.set_ylabel("native NLL")
    ax_c.set_ylim(0, max(nll_vals) * 1.28)
    ax_c.legend(fontsize=6.2, loc="upper center", frameon=True, facecolor="white",
               edgecolor="none", framealpha=0.9, ncol=2)

    rescue_e = bos["nll_rescue_fraction"]["E_restore_at_bos"]
    rescue_f = bos["nll_rescue_fraction"]["F_restore_at_matched_nonbos"]
    ax_c.annotate(f"~{rescue_e['point']*100:.0f}% rescue\n[{rescue_e['ci95'][0]*100:.1f}, {rescue_e['ci95'][1]*100:.1f}]",
                  xy=(4, nll_vals[4]), xytext=(4, nll_vals[4] + max(nll_vals) * 0.12),
                  ha="center", fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax_c.annotate(f"~{rescue_f['point']*100:.0f}% rescue\n[{rescue_f['ci95'][0]*100:.1f}, {rescue_f['ci95'][1]*100:.1f}]",
                  xy=(5, nll_vals[5]), xytext=(5, nll_vals[5] + max(nll_vals) * 0.12),
                  ha="center", fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6))
    panel_label(ax_c, "C")

    # --- Panel D (NEW): generated GC fraction across the same six conditions ----------
    gc_vals = [bos["gc_mean_by_condition"][k] for k, _ in BOS_CONDITIONS]
    intact_gc = bos["gc_mean_by_condition"]["A_intact"]
    ablated_gc = bos["gc_mean_by_condition"]["B_full_ablation_weight"]

    ax_d.axhline(intact_gc, color="black", linestyle=":", linewidth=1.1, zorder=1,
                 label=f"intact (GC={intact_gc:.3f})")
    ax_d.axhline(ablated_gc, color="0.35", linestyle="--", linewidth=1.1, zorder=1,
                 label=f"full ablation (GC={ablated_gc:.3f})")
    ax_d.bar(xs, gc_vals, color=DOMAIN_COLOR["genomic"], edgecolor="black",
             linewidth=0.8, width=0.6, zorder=3)
    ax_d.set_xticks(xs); ax_d.set_xticklabels(labels, fontsize=7)
    ax_d.set_ylabel("generated GC fraction")
    ax_d.set_ylim(0, max(gc_vals) * 1.28)
    ax_d.legend(fontsize=6.2, loc="upper center", frameon=True, facecolor="white",
               edgecolor="none", framealpha=0.9, ncol=2)
    panel_label(ax_d, "D")

    out = HERE / "main" / "fig4_generator"
    fig.savefig(f"{out}.png"); fig.savefig(f"{out}.pdf")

    src_out = HERE / "source_data" / "fig4_generator.json"
    src_out.write_text(json.dumps(dict(structural=legacy, attention_sink=sink,
                                        bos_mediation=bos), indent=1))
    print(f"saved -> {out}.png / .pdf, {src_out}")


if __name__ == "__main__":
    main()
