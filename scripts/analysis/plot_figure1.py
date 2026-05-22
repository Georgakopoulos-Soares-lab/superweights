"""
scripts/analysis/plot_figure1.py
---------------------------------
Unified composite manuscript figure (11 × 8.5 in, 300 dpi).

Panel A — Super-weight detection: activation scatter + ΔPPL bar chart
Panel B — Architecture schematic: GENERator EUK and PROK transformer stacks
Panel C — Activation lifecycle: SW channel magnitude across layers

All panels rendered from scratch via their original drawing logic.

Usage:
    python scripts/analysis/plot_figure1.py
    python scripts/analysis/plot_figure1.py --out results/figure1.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.size":          8,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "axes.titlesize":     8.5,
})

PANEL_LABEL_KW = dict(fontsize=14, fontweight="bold", va="bottom", ha="left",
                      fontfamily="DejaVu Sans", clip_on=False)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL A — Ablation
# ═════════════════════════════════════════════════════════════════════════════

_MODEL_META = {
    "generator":            {"label": "GENERator Euk 3B",  "num_layers": 30},
    "generator_prokaryote": {"label": "GENERator Prok 3B", "num_layers": 30},
    "dnabert2":             {"label": "DNABERT-2",          "num_layers": 12},
    "ntv3":                 {"label": "NTv3 650M",          "num_layers": 12},
    "evo2":                 {"label": "Evo2 7B",            "num_layers": 32,
                             "out_max_override": 552960, "spike_layer_override": 29},
    "hybridna":             {"label": "HybridNA 7B",        "num_layers": 32},
    "megadna":              {"label": "MegaDNA 145M",       "num_layers": 10},
}
_INCLUDE      = ["generator", "generator_prokaryote", "dnabert2", "ntv3",
                 "evo2", "hybridna", "megadna"]
_SW_THRESHOLD = 5.0
_C_POS  = "#d62728"
_C_NEG  = "#aec7e8"
_C_RAND = "#999999"


def _ablation_records(ablation_path, sw_index_path):
    with open(ablation_path) as f:  abl    = json.load(f)
    with open(sw_index_path)  as f:  sw_idx = json.load(f)

    records = []
    for key in _INCLUDE:
        if key not in abl:
            continue
        d    = abl[key]
        meta = _MODEL_META[key]
        best = d.get("best_sw", {})
        out_max     = meta.get("out_max_override") or best.get("out_max")
        spike_layer = meta.get("spike_layer_override") or best.get("layer")
        if out_max is None:
            entries = sw_idx.get(key, {}).get("results", [])
            if entries:
                be          = max(entries, key=lambda e: e["out_max"])
                out_max     = be["out_max"]
                spike_layer = be["layer"]
        nl = spike_layer / meta["num_layers"] if spike_layer is not None else None
        records.append({
            "model":      key,
            "label":      meta["label"],
            "out_max":    out_max,
            "norm_layer": nl,
            "delta_pct":  d["delta_pct"],
            "rand_delta": d["delta_rand_pct"],
            "is_sw":      abs(d["delta_pct"]) > _SW_THRESHOLD,
        })

    pos = sorted([r for r in records if r["is_sw"]],     key=lambda r: -r["delta_pct"])
    neg = sorted([r for r in records if not r["is_sw"]], key=lambda r: -r["delta_pct"])
    return pos + neg


def _draw_ablation_scatter(ax, records):
    for r in records:
        if r["out_max"] is None or r["norm_layer"] is None:
            continue
        color = _C_POS if r["is_sw"] else _C_NEG
        ax.scatter(r["norm_layer"], r["out_max"],
                   color=color, edgecolors="k", linewidths=0.6, s=90, zorder=4)
        ha, dx, dy = "left", 0.03, 1.0
        if r["norm_layer"] > 0.80:
            ha, dx = "right", -0.03
        if r["model"] == "evo2":
            dy = 2.5
        ax.annotate(r["label"],
                    xy=(r["norm_layer"], r["out_max"]),
                    xytext=(r["norm_layer"] + dx, r["out_max"] * dy),
                    fontsize=7, ha=ha, va="center")

    for r in records:
        if r["model"] == "evo2":
            ax.annotate("Large spike,\nno effect\n(SSM arch.)",
                        xy=(r["norm_layer"], r["out_max"]),
                        xytext=(r["norm_layer"] - 0.30, r["out_max"] / 30),
                        fontsize=6.5, color="#555555",
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7))

    ax.set_yscale("log")
    ax.set_xlim(-0.06, 1.15)
    ax.set_ylim(20, 5e6)
    ax.set_xlabel("Normalised layer position  (spike layer / total layers)")
    ax.set_ylabel("Peak activation — out_max  (log scale)")
    ax.tick_params(labelsize=7)

    pos_p = mpatches.Patch(color=_C_POS, label=f"SW-positive (ΔPPL > {_SW_THRESHOLD:.0f}%)")
    neg_p = mpatches.Patch(color=_C_NEG, label="SW-negative")
    ax.legend(handles=[pos_p, neg_p], fontsize=7, frameon=False, loc="upper center")


def _draw_ablation_bars(ax, records):
    labels      = [r["label"]      for r in records]
    deltas      = [r["delta_pct"]  for r in records]
    rand_deltas = [r["rand_delta"] for r in records]
    colors      = [_C_POS if r["is_sw"] else _C_NEG for r in records]
    y           = np.arange(len(records))
    plot_d      = [max(abs(d), 5e-4) for d in deltas]

    ax.barh(y, plot_d, color=colors, edgecolor="k", linewidth=0.5, height=0.6, zorder=2)
    ax.scatter([max(abs(rd), 5e-4) for rd in rand_deltas], y,
               color=_C_RAND, edgecolors="k", linewidths=0.4, s=35, zorder=4,
               label="Random-row control")

    for i, (raw, clipped) in enumerate(zip(deltas, plot_d)):
        txt = (f"{raw:+.4f}%" if abs(raw) < 0.01
               else f"{raw:+.2f}%" if abs(raw) < 100
               else f"+{raw:,.0f}%")
        ax.text(clipped * 2.0, i, txt, va="center", ha="left", fontsize=6.5)

    max_rand = max(abs(r["rand_delta"]) for r in records)
    ax.axvspan(5e-4, max(max_rand * 8, 0.5), alpha=0.07, color=_C_RAND, zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(3e-4, 2e6)
    ax.set_xlabel("ΔPPL%  after super-row ablation  (log scale)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc="upper right")

    n_pos = sum(1 for r in records if r["is_sw"])
    if 0 < n_pos < len(records):
        ax.axhline(n_pos - 0.5, color="k", lw=0.8, ls="--", zorder=3, alpha=0.5)


def draw_panel_A(ax_scatter, ax_bars, ablation_path, sw_index_path):
    records = _ablation_records(ablation_path, sw_index_path)
    _draw_ablation_scatter(ax_scatter, records)
    _draw_ablation_bars(ax_bars, records)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL B — Architecture schematic
# ═════════════════════════════════════════════════════════════════════════════

_ARCH_MODELS = [
    {"sw_layer": 4, "sw_rows": [1522, 2371], "model_label": "GENERator-EUK-3B",
     "sub_label": "eukaryote"},
    {"sw_layer": 2, "sw_rows": [1927],       "model_label": "GENERator-PROK-3B",
     "sub_label": "prokaryote"},
]
_NUM_LAYERS = 30
_D_MODEL    = 3072

_CB_LAYER  = "#DDE4EF"; _CB_LEDGE = "#9AAAC8"
_CB_SW     = "#4878CF"; _CB_SEDGE = "#2A54A8"
_CB_EMBED  = "#B8D8B8"; _CB_EEDGE = "#6FA86F"
_CB_SWROW  = "#E8541A"
_CB_NORM   = "#D8DEE8"; _CB_STRIP = "#808080"


def _arch_rect(ax, x, y, w, h, fc, ec, lw=1.0, zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0",
                                fc=fc, ec=ec, lw=lw, zorder=zorder,
                                transform=ax.transData))


def draw_panel_B_one(ax, model):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("auto"); ax.axis("off")

    sw_layer = model["sw_layer"]
    sw_rows  = model["sw_rows"]

    # Geometry
    sx, sw   = 0.10, 0.38
    top_y    = 0.96
    bot_y    = 0.14
    embed_h  = 0.030
    lmh_h    = 0.030
    gap      = 0.004

    lmh_y       = bot_y
    layer_top_y = top_y - embed_h - gap
    layer_bot_y = lmh_y + lmh_h + gap
    layer_h     = (layer_top_y - layer_bot_y) / _NUM_LAYERS

    # Embedding
    _arch_rect(ax, sx, top_y - embed_h, sw, embed_h,
               fc=_CB_EMBED, ec=_CB_EEDGE, lw=0.8)
    ax.text(sx + sw / 2, top_y - embed_h / 2, "Embedding",
            ha="center", va="center", fontsize=6.5, color="#2A5A2A")

    # Layers
    layer_tops = []
    for i in range(_NUM_LAYERS):
        ly  = layer_top_y - i * layer_h
        lyb = ly - layer_h
        layer_tops.append(ly)
        is_sw = (i == sw_layer)
        _arch_rect(ax, sx, lyb, sw, layer_h,
                   fc=_CB_SW if is_sw else _CB_LAYER,
                   ec=_CB_SEDGE if is_sw else _CB_LEDGE,
                   lw=1.3 if is_sw else 0.4,
                   zorder=3 if is_sw else 2)
        if i % 5 == 0 or is_sw:
            ax.text(sx + sw / 2, lyb + layer_h / 2, f"L{i}",
                    ha="center", va="center",
                    fontsize=5.5 if not is_sw else 6.5,
                    color="white" if is_sw else "#444444",
                    fontweight="bold" if is_sw else "normal", zorder=4)

    # LM head
    _arch_rect(ax, sx, lmh_y, sw, lmh_h, fc=_CB_EMBED, ec=_CB_EEDGE, lw=0.8)
    ax.text(sx + sw / 2, lmh_y + lmh_h / 2, "LM head",
            ha="center", va="center", fontsize=6.5, color="#2A5A2A")

    # down_proj strip
    strx  = 0.62; strw = 0.13
    strt  = layer_top_y
    strb  = layer_bot_y - gap
    strh  = strt - strb

    ax.add_patch(mpatches.Rectangle((strx, strb), strw, strh,
                                    fc=_CB_NORM, ec=_CB_STRIP, lw=0.7, zorder=2))

    row_h = max(strh / _D_MODEL * 6, 0.010)
    for row in sw_rows:
        ry = strb + (_D_MODEL - 1 - row) / _D_MODEL * strh
        ax.add_patch(mpatches.Rectangle((strx, ry), strw, row_h,
                                        fc=_CB_SWROW, ec="none", zorder=3))
        ax.annotate(f"  row {row}",
                    xy=(strx + strw, ry + row_h / 2),
                    xytext=(strx + strw + 0.01, ry + row_h / 2),
                    ha="left", va="center", fontsize=6.5, color=_CB_SWROW,
                    fontweight="bold", annotation_clip=False)

    ax.text(strx + strw / 2, strt + 0.013, "down_proj",
            ha="center", va="bottom", fontsize=6.5, fontstyle="italic", color="#333333")
    ax.text(strx + strw / 2, strt + 0.034, f"({_D_MODEL} rows)",
            ha="center", va="bottom", fontsize=6, color="#666666")
    ax.text(strx - 0.02, strt, "row 0",
            ha="right", va="center", fontsize=5.5, color="#666666")
    ax.text(strx - 0.02, strb, f"row {_D_MODEL-1}",
            ha="right", va="center", fontsize=5.5, color="#666666")
    ax.plot([strx-0.01, strx], [strt, strt], color="#888", lw=0.7)
    ax.plot([strx-0.01, strx], [strb, strb], color="#888", lw=0.7)

    # Arrow SW layer → strip
    sw_top = layer_tops[sw_layer]
    sw_ctr = sw_top - layer_h / 2
    ax.annotate("", xy=(strx, (strt + strb) / 2),
                xytext=(sx + sw, sw_ctr),
                arrowprops=dict(arrowstyle="-|>", color="#505050", lw=0.8,
                                connectionstyle="arc3,rad=-0.10"), zorder=5)

    # Legend
    sl_p = mpatches.Patch(color=_CB_SW,   label="SW layer")
    sr_p = mpatches.Patch(color=_CB_SWROW, label="SW row")
    nr_p = mpatches.Patch(color=_CB_NORM, ec=_CB_STRIP, lw=0.5, label="Other rows")
    ax.legend(handles=[sl_p, sr_p, nr_p], loc="lower left",
              fontsize=6, framealpha=0.9, edgecolor="#CCC",
              handlelength=1.2, handleheight=0.9, borderpad=0.5,
              bbox_to_anchor=(0.02, 0.01))


def draw_panel_B(ax_euk, ax_prok):
    for ax, m in zip([ax_euk, ax_prok], _ARCH_MODELS):
        draw_panel_B_one(ax, m)


# ═════════════════════════════════════════════════════════════════════════════
# PANEL C — Activation lifecycle
# ═════════════════════════════════════════════════════════════════════════════

_LC_COLORS = {
    "sw0":  "#D62728",
    "sw1":  "#FF7F0E",
    "top3": "#AAAAAA",
}


def _draw_lifecycle_col(ax_top, ax_bot, data):
    n_layers = len(data["layers"])
    x        = np.arange(n_layers)
    sw_layer = data["sw_layer"]
    sw_rows  = data["sw_rows"]

    per_seq_post = np.array(data["per_seq_post_residual_sw"])   # (S, L, R)
    per_seq_delt = np.array(data["per_seq_block_delta_sw"])     # (S, L, R)
    top3         = np.array(data["mean_post_residual_top3"])    # (L, 3)

    # ── Top: post-residual magnitude ─────────────────────────────────────────
    ax_top.fill_between(x, top3[:, 2], top3[:, 0],
                        color=_LC_COLORS["top3"], alpha=0.20,
                        label="Top-3 global channels")
    ax_top.plot(x, top3[:, 0], color=_LC_COLORS["top3"], lw=0.7, ls="--")

    ckeys = ["sw0", "sw1"]
    for j, row in enumerate(sw_rows):
        if j >= len(ckeys):
            break
        m = per_seq_post[:, :, j].mean(0)
        s = per_seq_post[:, :, j].std(0)
        ax_top.plot(x, m, color=_LC_COLORS[ckeys[j]], lw=1.6,
                    label=f"SW row {row}")
        ax_top.fill_between(x, m - s, m + s, color=_LC_COLORS[ckeys[j]], alpha=0.15)

    ax_top.set_yscale("log")
    ax_top.set_ylabel("max |h| over tokens")
    ax_top.axvline(sw_layer, color="grey", ls=":", lw=0.7, alpha=0.7)
    ax_top.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax_top.set_xticks(x[::5])
    ax_top.tick_params(labelsize=7)

    # Peak annotation
    m0 = per_seq_post[:, :, 0].mean(0)
    pk = int(np.argmax(m0))
    ax_top.annotate(f"L{pk} peak",
                    xy=(pk, m0[pk]),
                    xytext=(pk + 1.5, m0[pk] * 1.6),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="k"),
                    fontsize=6.5, color="k")

    # ── Bottom: per-block delta ───────────────────────────────────────────────
    for j, row in enumerate(sw_rows):
        if j >= len(ckeys):
            break
        m = per_seq_delt[:, :, j].mean(0)
        s = per_seq_delt[:, :, j].std(0)
        ax_bot.plot(x, m, color=_LC_COLORS[ckeys[j]], lw=1.6, label=f"SW row {row}")
        ax_bot.fill_between(x, np.maximum(m - s, 1e-3), m + s,
                            color=_LC_COLORS[ckeys[j]], alpha=0.15)

    ax_bot.set_yscale("log")
    ax_bot.set_ylabel("max |Δh| over tokens")
    ax_bot.set_xlabel("Layer index")
    ax_bot.axvline(sw_layer, color="grey", ls=":", lw=0.7, alpha=0.7,
                   label=f"SW layer (L{sw_layer})")
    ax_bot.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax_bot.set_xticks(x[::5])
    ax_bot.tick_params(labelsize=7)

    # Step-up / step-down annotations
    d0 = per_seq_delt[:, :, 0].mean(0)
    if d0.max() > 0:
        half = n_layers // 2
        su = int(np.argmax(d0[:half]))
        sd = int(np.argmax(d0[half:])) + half
        ax_bot.annotate("Step-up",
                        xy=(su, d0[su]), xytext=(su + 1, d0[su] * 2.0),
                        arrowprops=dict(arrowstyle="->", color=_LC_COLORS["sw0"], lw=0.7),
                        fontsize=6.5, color=_LC_COLORS["sw0"])
        ax_bot.annotate("Step-down",
                        xy=(sd, d0[sd]), xytext=(sd - 5, d0[sd] * 2.0),
                        arrowprops=dict(arrowstyle="->", color="#1F77B4", lw=0.7),
                        fontsize=6.5, color="#1F77B4")


def draw_panel_C(axes_2x2, euk_path, prok_path):
    """axes_2x2 = [[ax_euk_top, ax_prok_top], [ax_euk_bot, ax_prok_bot]]"""
    with open(euk_path)  as f:  euk_data  = json.load(f)
    with open(prok_path) as f:  prok_data = json.load(f)

    _draw_lifecycle_col(axes_2x2[0][0], axes_2x2[1][0], euk_data)
    _draw_lifecycle_col(axes_2x2[0][1], axes_2x2[1][1], prok_data)

    # Column headers (model names only, no decorative title)
    for ax, name in zip(axes_2x2[0], ["GENERator EUK 3B", "GENERator PROK 3B"]):
        ax.text(0.5, 1.06, name, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8, fontstyle="italic")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation",   default="results/ablation_results.json")
    parser.add_argument("--sw_index",   default="results/super_weight_index.json")
    parser.add_argument("--euk_lc",     default="results/activation_lifecycle_generator.json")
    parser.add_argument("--prok_lc",    default="results/activation_lifecycle_generator_prokaryote.json")
    parser.add_argument("--out",        default="results/figure1.png")
    args = parser.parse_args()

    fig = plt.figure(figsize=(8.5, 11))

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[3.5, 4.5, 4.0],
        hspace=0.52,
        left=0.08, right=0.97,
        top=0.97,  bottom=0.05,
    )

    # ── Panel A ───────────────────────────────────────────────────────────────
    gs_A = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[1.15, 1], wspace=0.44,
    )
    ax_A1 = fig.add_subplot(gs_A[0])
    ax_A2 = fig.add_subplot(gs_A[1])
    draw_panel_A(ax_A1, ax_A2, args.ablation, args.sw_index)
    # A and B: place at top-left corner of each subplot axes (x=0 avoids y-axis overlap)
    ax_A1.text(0.0, 1.03, "A", transform=ax_A1.transAxes, **PANEL_LABEL_KW)
    ax_A2.text(0.0, 1.03, "B", transform=ax_A2.transAxes, **PANEL_LABEL_KW)

    # ── Panel C–D ─────────────────────────────────────────────────────────────
    gs_B = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1], wspace=0.06,
    )
    ax_B1 = fig.add_subplot(gs_B[0])
    ax_B2 = fig.add_subplot(gs_B[1])
    draw_panel_B(ax_B1, ax_B2)
    ax_B1.text(0.0, 1.03, "C", transform=ax_B1.transAxes, **PANEL_LABEL_KW)
    ax_B2.text(0.0, 1.03, "D", transform=ax_B2.transAxes, **PANEL_LABEL_KW)

    # ── Panel E–F ─────────────────────────────────────────────────────────────
    gs_C = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[2],
        hspace=0.52, wspace=0.38,
    )
    axes_C = [[fig.add_subplot(gs_C[r, c]) for c in range(2)] for r in range(2)]
    draw_panel_C(axes_C, args.euk_lc, args.prok_lc)
    # E and F label the two model columns (top axes only; bottom axes are part of same panel)
    axes_C[0][0].text(0.0, 1.14, "E", transform=axes_C[0][0].transAxes, **PANEL_LABEL_KW)
    axes_C[0][1].text(0.0, 1.14, "F", transform=axes_C[0][1].transAxes, **PANEL_LABEL_KW)

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
