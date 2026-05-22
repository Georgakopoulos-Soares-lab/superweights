"""
scripts/analysis/plot_figure2.py
---------------------------------
Composite manuscript figure (8.5 × 11 in portrait, 300 dpi).

Panel A  — EUK GENERator: ΔPPL after SW vs random-row ablation, by region type
Panel B  — PROK GENERator: same
Panel C  — EUK GENERator: mean SW activation per hexamer category (correlation)
Panel D  — PROK GENERator: same
Panel E  — EUK GENERator: hexamer-motif enrichment (odds ratio)
Panel F  — PROK GENERator: same
Panel G  — EUK GENERator: SW activation difference (functional − non-functional)
Panel H  — PROK GENERator: same

Usage:
    python scripts/analysis/plot_figure2.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         8,
    "axes.labelsize":    8,
    "axes.titlesize":    9,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

_LABEL_KW = dict(fontsize=14, fontweight="bold", va="bottom", ha="left",
                 clip_on=False)

_REPO = Path(__file__).resolve().parent.parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# PANELS A & B — Causal-tracing ΔPPL bar charts
# ═════════════════════════════════════════════════════════════════════════════

_C_SW   = "#4878CF"
_C_RAND = "#CCCCCC"


def _draw_causal_bars(ax, json_path, label_map=None):
    data     = json.load(open(json_path))
    results  = data["results"]

    sw_by_ctx   = defaultdict(list)
    rand_by_ctx = defaultdict(list)
    for r in results:
        ctx = r["label"]
        if label_map:
            ctx = label_map.get(ctx, ctx)
        sw_by_ctx[ctx].append(r["delta_sw"])
        rand_by_ctx[ctx].append(r["delta_rand_mean"])

    contexts   = sorted(sw_by_ctx.keys())
    x          = np.arange(len(contexts))
    w          = 0.35
    sw_means   = [np.mean(sw_by_ctx[c])   for c in contexts]
    rand_means = [np.mean(rand_by_ctx[c]) for c in contexts]
    sw_sems    = [np.std(sw_by_ctx[c])    for c in contexts]
    rand_sems  = [np.std(rand_by_ctx[c])  for c in contexts]

    ax.bar(x - w/2, sw_means,   w, color=_C_SW,   label="SW row",
           yerr=sw_sems,   capsize=3, error_kw={"linewidth": 1})
    ax.bar(x + w/2, rand_means, w, color=_C_RAND, label="Random row",
           yerr=rand_sems, capsize=3, error_kw={"linewidth": 1})
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(contexts, fontsize=8)
    ax.set_ylabel("ΔPPL  (ablated − clean)  ↑ = more important")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)


# ═════════════════════════════════════════════════════════════════════════════
# PANELS C & D — Hexamer-category SW activation (correlation bars)
# ═════════════════════════════════════════════════════════════════════════════

_CAT_ORDER  = ["splice_donor", "splice_acceptor", "AT_rich", "other"]
_CAT_COLORS = {
    "splice_donor":   "#D62728",
    "splice_acceptor":"#FF7F0E",
    "AT_rich":        "#4878CF",
    "other":          "#999999",
}
_CAT_LABELS = {
    "splice_donor":   "Splice\ndonor (GT)",
    "splice_acceptor":"Splice\nacceptor",
    "AT_rich":        "AT-rich\n(≥5 AT/6)",
    "other":          "Other",
}


def _kmer_cat(km: str) -> str:
    if "GT" in km:
        return "splice_donor"
    if km.endswith("AG") and sum(c in "CT" for c in km) >= 3:
        return "splice_acceptor"
    if sum(c in "AT" for c in km) >= 5:
        return "AT_rich"
    return "other"


_CAT_LABELS_PROK = {
    "splice_donor":   "GT-start\nhexamers",
    "splice_acceptor":"AG-end\nhexamers",
    "AT_rich":        "AT-rich\n(≥5 AT/6)",
    "other":          "Other",
}


def _draw_category_bars(ax, json_path, cat_labels=None):
    if cat_labels is None:
        cat_labels = _CAT_LABELS
    data    = json.load(open(json_path))
    records = data["results"]

    by_cat = defaultdict(list)
    for r in records:
        cat = _kmer_cat(r["kmer"])
        by_cat[cat].append(abs(r["sw_activation"]))   # magnitude

    cats   = _CAT_ORDER
    means  = [np.mean(by_cat[c]) if by_cat[c] else 0 for c in cats]
    sems   = [np.std(by_cat[c]) / np.sqrt(max(len(by_cat[c]), 1)) for c in cats]
    colors = [_CAT_COLORS[c] for c in cats]
    xlabs  = [cat_labels[c] for c in cats]
    ns     = [len(by_cat[c]) for c in cats]

    bars = ax.bar(range(len(cats)), means, color=colors, width=0.6,
                  yerr=sems, capsize=3, error_kw={"linewidth": 1})
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(xlabs, fontsize=7)
    ax.set_ylabel("|SW activation|  (mean ± SEM)")
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((-2, 4))
    ax.grid(True, axis="y", alpha=0.25)

    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + sems[i] * 1.2, f"n={n}", ha="center",
                va="bottom", fontsize=6, color="#444")

    # Pearson r (category ordinal encoding vs activation)
    cat_idx = [_CAT_ORDER.index(_kmer_cat(r["kmer"])) for r in records]
    acts    = [abs(r["sw_activation"]) for r in records]
    r_val, p_val = pearsonr(cat_idx, acts)
    ax.text(0.98, 0.96, f"r={r_val:.3f}  p={p_val:.1e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color="#333")


# ═════════════════════════════════════════════════════════════════════════════
# PANELS E & F — Hexamer motif enrichment (odds ratio)
# ═════════════════════════════════════════════════════════════════════════════

def _draw_enrichment(ax, json_path, show_legend=False):
    data       = json.load(open(json_path))
    enrichment = data["motif_enrichment"]
    motifs     = [e["motif"]                     for e in enrichment]
    ors        = [min(e["odds_ratio"], 10)        for e in enrichment]
    sigs       = [e["significant"]               for e in enrichment]
    bcolors    = ["#e74c3c" if s else "#95a5a6"  for s in sigs]

    ax.barh(range(len(motifs)), ors, color=bcolors, height=0.65)
    ax.set_yticks(range(len(motifs)))
    ax.set_yticklabels(motifs, fontsize=7.5)
    ax.axvline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Odds ratio  (capped at 10)")
    ax.grid(True, axis="x", alpha=0.2)

    for i, e in enumerate(enrichment):
        q = e["q_value"]
        q_str = f"q={q:.1e}" if e["significant"] else f"q={q:.2f}"
        ax.text(max(ors[i], 0.05) + 0.08, i, q_str,
                va="center", fontsize=5.5, color="#333")

    if show_legend:
        sig_p = mpatches.Patch(color="#e74c3c", label="Significant (q<0.05)")
        ns_p  = mpatches.Patch(color="#95a5a6", label="Not significant")
        ax.legend(handles=[sig_p, ns_p], fontsize=6, frameon=False, loc="upper right")


# ═════════════════════════════════════════════════════════════════════════════
# PANELS G & H — SW activation difference heatmap (functional − non-functional)
# ═════════════════════════════════════════════════════════════════════════════

def _short_task(task: str) -> str:
    parts = task.split("/")
    return (parts[-1]
            .replace("prom_", "")
            .replace("_notata", "_noTATA")
            .replace("_tata", "_TATA")
            .replace("reconstructed", "splice"))


def _diff_matrix(data: dict):
    sw_rows    = data["sw_rows"]
    task_names = data["task_order"]
    n_rows     = len(sw_rows)
    n_tasks    = len(task_names)
    diff = np.full((n_rows, n_tasks), np.nan)
    se   = np.full((n_rows, n_tasks), np.nan)
    for ti, task in enumerate(task_names):
        td = data["tasks"][task]
        for ri, row in enumerate(sw_rows):
            key  = str(row)
            pv   = td.get("pos", {}).get(key, [])
            nv   = td.get("neg", {}).get(key, [])
            if pv and nv:
                mp, sp, np_ = np.mean(pv), np.std(pv, ddof=1), len(pv)
                mn, sn, nn  = np.mean(nv), np.std(nv, ddof=1), len(nv)
                diff[ri, ti] = mp - mn
                se[ri, ti]   = np.sqrt(sp**2/np_ + sn**2/nn)
    short_labels = [_short_task(t) for t in task_names]
    return diff, se, short_labels


def _draw_diff_heatmap(ax, json_path):
    import matplotlib.colors as mcolors
    data = json.load(open(json_path))
    diff, se, short_labels = _diff_matrix(data)
    sw_rows    = data["sw_rows"]
    sw_layer   = data["sw_layer"]
    row_labels = [f"L{sw_layer}r{r}" for r in sw_rows]

    n_rows, n_tasks = diff.shape
    vmax = np.nanmax(np.abs(diff))
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im   = ax.imshow(diff, aspect="auto", cmap="RdBu_r", norm=norm)

    for ri in range(n_rows):
        for ti in range(n_tasks):
            d, s = diff[ri, ti], se[ri, ti]
            if np.isnan(d):
                continue
            txt_col = "white" if abs(d) > vmax * 0.55 else "black"
            sign    = "+" if d >= 0 else ""
            ax.text(ti, ri, f"{sign}{d:.0f}\n±{s:.0f}",
                    ha="center", va="center", fontsize=6.5, color=txt_col)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(short_labels, fontsize=7, rotation=38, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δ mean |act|  (pos − neg)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.ax.axhline(0, color="k", lw=0.8, ls="--")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    fig = plt.figure(figsize=(8.5, 11))

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[2.8, 4.2, 4.0],
        hspace=0.52,
        left=0.12, right=0.97,
        top=0.97, bottom=0.04,
    )

    # ── Row 0: Causal tracing bars (A, B) ─────────────────────────────────
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.40)
    ax_A = fig.add_subplot(gs0[0])
    ax_B = fig.add_subplot(gs0[1])
    _draw_causal_bars(ax_A, _REPO / "results/sw_causal_tracing.json")
    _draw_causal_bars(ax_B, _REPO / "results/prokaryote/sw_causal_tracing.json",
                      label_map={"enhancer": "terminator"})
    ax_A.text(0.0, 1.04, "A", transform=ax_A.transAxes, **_LABEL_KW)
    ax_B.text(0.0, 1.04, "B", transform=ax_B.transAxes, **_LABEL_KW)

    # Model labels (italic, above each bar panel)
    for ax, name in [(ax_A, "GENERator-EUK-3B"), (ax_B, "GENERator-PROK-3B")]:
        ax.text(0.5, 1.04, name, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8, fontstyle="italic")

    # ── Row 1: Hexamer (C–F) ────────────────────────────────────────────────
    gs1 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[1], hspace=0.55, wspace=0.42,
    )
    ax_C = fig.add_subplot(gs1[0, 0])
    ax_D = fig.add_subplot(gs1[0, 1])
    ax_E = fig.add_subplot(gs1[1, 0])
    ax_F = fig.add_subplot(gs1[1, 1])

    _draw_category_bars(ax_C, _REPO / "results/sw_hexamer_causal.json")
    _draw_category_bars(ax_D, _REPO / "results/sw_hexamer_causal_generator_prok.json",
                        cat_labels=_CAT_LABELS_PROK)
    _draw_enrichment(ax_E, _REPO / "results/sw_hexamer_motifs_top500_generator_euk.json")
    _draw_enrichment(ax_F, _REPO / "results/sw_hexamer_motifs_top500_generator_prok.json",
                     show_legend=True)

    ax_C.text(0.0, 1.06, "C", transform=ax_C.transAxes, **_LABEL_KW)
    ax_D.text(0.0, 1.06, "D", transform=ax_D.transAxes, **_LABEL_KW)
    ax_E.text(0.0, 1.06, "E", transform=ax_E.transAxes, **_LABEL_KW)
    ax_F.text(0.0, 1.06, "F", transform=ax_F.transAxes, **_LABEL_KW)

    # ── Row 2: Diff heatmaps (G, H) ─────────────────────────────────────────
    # width proportional to number of tasks
    euk_hm  = json.load(open(_REPO / "results/sw_task_activation_heatmap_generator_euk.json"))
    prok_hm = json.load(open(_REPO / "results/sw_task_activation_heatmap_generator_prok.json"))
    n_euk   = len(euk_hm["task_order"])
    n_prok  = len(prok_hm["task_order"])

    gs2 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2],
        width_ratios=[n_euk, n_prok], wspace=0.38,
    )
    ax_G = fig.add_subplot(gs2[0])
    ax_H = fig.add_subplot(gs2[1])
    _draw_diff_heatmap(ax_G, _REPO / "results/sw_task_activation_heatmap_generator_euk.json")
    _draw_diff_heatmap(ax_H, _REPO / "results/sw_task_activation_heatmap_generator_prok.json")
    ax_G.text(0.0, 1.06, "G", transform=ax_G.transAxes, **_LABEL_KW)
    ax_H.text(0.0, 1.06, "H", transform=ax_H.transAxes, **_LABEL_KW)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = _REPO / "results/figure2.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
