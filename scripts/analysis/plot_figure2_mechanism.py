"""Figure 2 — mechanism figure, full 6-panel (GENERator + cross-architecture).

NOTE: the FIGURES_README.md says to rename plot_figure2_panels_EF.py → plot_figure2.py,
but that name is already taken by an older hexamer-correlation figure. This script
is the intended successor; coordinate with collaborator to retire or renumber the old one.

Layout (2 rows × 3 columns):
  A  B  E
  C  D  F

Panel A: GENERator-EUK activation lifecycle — |residual| at SW row 2371 vs layer,
         showing the explosive write at layer 4 then DC freeze through layer 23.
Panel B: GENERator-PROK activation lifecycle — |residual| at SW row 1927 vs layer,
         explosive write at layer 2.
Panel C: GENERator-EUK ||U_k||_F distribution at layer 4 — sorted bar, SW rows
         2371 (rank 1) and 1522 (rank 2) highlighted.
Panel D: GENERator-PROK ||U_k||_F distribution at layer 2 — SW row 1927 (rank 1)
         highlighted; concentration ratio 17.9×.
Panel E: Cross-architecture ||U_k||_F percentile scatter (DNABERT-2, NTv3,
         Evo1 null, GENERator-EUK, GENERator-PROK).
Panel F: Dual residual attribution — DNABERT-2 row 603 (top) + Evo1 row 3776 (bot).

Data:
  results/activation_lifecycle_generator{,_prokaryote}.json
  results/sw_mechanistic_generator{,_prokaryote}.json
  results/sw_mechanistic_{dnabert2,ntv3,evo1}.json
  results/sw_residual_attribution_dnabert2.json
  results/sw_residual_attribution_evo1_fp32.json

Output:
  paper/media/image_fig2_mechanism.png (.pdf)

Usage (from repo root):
  python scripts/analysis/plot_figure2_mechanism.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

from _figstyle import apply_style, panel_label

apply_style()

ROOT    = Path(__file__).resolve().parents[2]
RES     = ROOT / "results"
OUT_PNG = ROOT / "paper" / "media" / "image_fig2_mechanism.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

SW_COLOR_EUK  = "#c0392b"
SW_COLOR_PROK = "#8e44ad"


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _rows_from_mechanistic(p: Path) -> list[dict]:
    d = _load(p)
    n_layers = max(int(k) for k in d["frob_norm_uk_by_layer"]) + 1
    out = []
    for li_str, ranks in d.get("sw_row_ranks", {}).items():
        li = int(li_str)
        dm = len(d["frob_norm_uk_by_layer"][li_str])
        for r_str, rank in ranks.items():
            out.append(dict(layer=li, n_layers=n_layers,
                            sw_row=int(r_str), rank=rank, d_model=dm,
                            pct=100.0 * (1 - (rank - 1) / dm)))
    return out


def _rows_evo1(p: Path) -> list[dict]:
    d = _load(p)
    n_blocks = d.get("n_blocks", 32)
    out = []
    for li_str, ranks in d.get("detected_row_ranks", {}).items():
        li = int(li_str)
        dm = len(d["frob_norm_uk_by_layer"][li_str])
        for r_str, rank in ranks.items():
            out.append(dict(layer=li, n_layers=n_blocks,
                            sw_row=int(r_str), rank=rank, d_model=dm,
                            pct=100.0 * (1 - (rank - 1) / dm)))
    return out


# ─── PANELS A / B — lifecycle ───────────────────────────────────────────────

def _lifecycle_panel(ax, json_path: Path, sw_layer: int, sw_row: int,
                     color: str, xlabel: str) -> None:
    d  = _load(json_path)
    pl = d["per_layer"]
    layers = [r["layer"] for r in pl]
    rout   = [abs(r.get("max_abs_at_sw_pos", r.get("residual_out", 0.0))) for r in pl]
    mlp    = [abs(r.get("mlp_out_at_sw_pos",  r.get("mlp_out",    0.0))) for r in pl]

    ax.plot(layers, rout, "o-", color="#1f618d", lw=1.4, ms=4,
            label=r"$\|h_{\mathrm{out}}\|$ at SW pos")
    ax.plot(layers, mlp,  "s--", color=color, lw=1.0, ms=3,
            label="MLP write")

    # symlog so the pre-write region is visible
    linthresh = max(rout[0] if rout[0] > 0 else 1.0, 1.0)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$|$value at row " + str(sw_row) + r"$|$")

    if sw_layer in layers:
        idx = layers.index(sw_layer)
        ax.annotate(f"layer {sw_layer}\n(write {mlp[idx]:.0f})",
                    xy=(sw_layer, rout[idx]),
                    xytext=(sw_layer + (3 if sw_layer < 20 else -8),
                            rout[idx] * 0.25),
                    fontsize=6.5, color=color, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color=color))

    ax.legend(loc="lower right", frameon=False, fontsize=7, handletextpad=0.4)


def panel_A(ax):
    _lifecycle_panel(ax, RES / "activation_residual_generator.json",
                     sw_layer=4, sw_row=2371,
                     color=SW_COLOR_EUK, xlabel="GENERator-EUK layer")


def panel_B(ax):
    _lifecycle_panel(ax, RES / "activation_residual_generator_prokaryote.json",
                     sw_layer=2, sw_row=1927,
                     color=SW_COLOR_PROK, xlabel="GENERator-PROK layer")


# ─── PANELS C / D — ||U_k||_F bar at the step-up layer ─────────────────────

def _frob_bar_panel(ax, json_path: Path, sw_layer: int,
                    sw_rows: list[int], color: str, xlabel: str) -> None:
    d    = _load(json_path)
    frob = np.array(d["frob_norm_uk_by_layer"][str(sw_layer)])
    dm   = len(frob)

    order    = np.argsort(-frob)
    sorted_f = frob[order]
    sw_set   = set(sw_rows)
    colors   = [color if int(order[i]) in sw_set else "#bdc3c7" for i in range(dm)]

    ax.bar(np.arange(dm), sorted_f, color=colors, width=1.0, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\|U_k\|_F$")

    for row in sw_rows:
        rank = int(np.where(order == row)[0][0]) + 1
        val  = frob[row]
        ax.annotate(f"row {row}\n(rank {rank})",
                    xy=(rank - 1, val),
                    xytext=(rank + dm * 0.04, val * 0.82),
                    fontsize=6.5, color=color, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color=color))

    median = float(np.median(frob))
    ax.axhline(median, color="#7f8c8d", lw=0.7, ls=":",
               label=f"median = {median:.1f}")
    ax.legend(loc="upper right", frameon=False, fontsize=7)


def panel_C(ax):
    _frob_bar_panel(ax, RES / "sw_mechanistic_generator.json",
                    sw_layer=4, sw_rows=[2371, 1522],
                    color=SW_COLOR_EUK, xlabel="GENERator-EUK row rank (layer 4)")


def panel_D(ax):
    _frob_bar_panel(ax, RES / "sw_mechanistic_generator_prokaryote.json",
                    sw_layer=2, sw_rows=[1927],
                    color=SW_COLOR_PROK, xlabel="GENERator-PROK row rank (layer 2)")


# ─── PANEL E — cross-architecture percentile scatter ────────────────────────

def panel_E(ax):
    db       = _rows_from_mechanistic(RES / "sw_mechanistic_dnabert2.json")
    nt       = _rows_from_mechanistic(RES / "sw_mechanistic_ntv3.json")
    ev       = _rows_evo1(RES / "sw_mechanistic_evo1.json")
    gen_euk  = _rows_from_mechanistic(RES / "sw_mechanistic_generator.json") \
               if (RES / "sw_mechanistic_generator.json").exists() else []
    gen_prok = _rows_from_mechanistic(RES / "sw_mechanistic_generator_prokaryote.json") \
               if (RES / "sw_mechanistic_generator_prokaryote.json").exists() else []

    def scatter(rows, marker, color, label, *, filled=True, size=85):
        if not rows:
            return
        xs = [r["layer"] / max(r["n_layers"] - 1, 1) for r in rows]
        ys = [r["pct"] for r in rows]
        if filled:
            ax.scatter(xs, ys, marker=marker, color=color, s=size,
                       edgecolor="black", linewidth=0.9, label=label, zorder=3)
        else:
            ax.scatter(xs, ys, marker=marker, facecolors="none",
                       edgecolor=color, linewidth=1.4, s=size,
                       label=label, zorder=3)

    scatter(db,       "s", "#2c3e50", "DNABERT-2",          filled=True,  size=70)
    scatter(nt,       "D", "#16a085", "NTv3",               filled=True,  size=95)
    scatter(ev,       "^", "#444444", "Evo1 (null)",        filled=False, size=85)
    scatter(gen_euk,  "o", SW_COLOR_EUK,  "GENERator-EUK",  filled=True,  size=90)
    scatter(gen_prok, "o", SW_COLOR_PROK, "GENERator-PROK", filled=True,  size=90)

    out = [r for r in db if r["layer"] == 7 and r["sw_row"] == 603]
    if out:
        r = out[0]
        x = r["layer"] / max(r["n_layers"] - 1, 1)
        ax.annotate("L7 r603\n(residual-carried)",
                    xy=(x, r["pct"]), xytext=(x + 0.05, r["pct"] + 18),
                    fontsize=7, color="#a04000", ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="#a04000"))

    ax.axhline(99, color="#aaa", lw=0.5, ls=":")
    ax.text(0.02, 99.4, "99th pct", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6.5, color="#777")
    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-3, 108)
    ax.set_xlabel("relative layer position")
    ax.set_ylabel(r"$\|U_k\|_F$ percentile within layer")
    ax.legend(loc="lower left", frameon=False, ncol=2,
              handletextpad=0.4, columnspacing=1.2, fontsize=8)


# ─── PANEL F — residual attribution (DNABERT-2 + Evo1) ─────────────────────

def panel_F1(ax):
    d    = _load(RES / "sw_residual_attribution_dnabert2.json")
    keep = [r for r in d["per_layer"] if 2 <= r["layer"] <= 11]
    layers = [r["layer"] for r in keep]
    res_in = [abs(r["residual_in"]) for r in keep]
    mlp    = [abs(r["mlp_out"])     for r in keep]

    for li, color in [*((li, "#fff3e0") for li in {3, 5, 6}),
                      *((li, "#e7f0fa") for li in {7, 8})]:
        if li in layers:
            xi = layers.index(li)
            ax.axvspan(xi - 0.5, xi + 0.5, color=color, zorder=0)

    x = np.arange(len(layers)); w = 0.40
    ax.bar(x - w/2, res_in, w, color="#7f8c8d", edgecolor="black",
           linewidth=0.5, label=r"$|h_{\mathrm{in}}^{(603)}|$")
    ax.bar(x + w/2, mlp,    w, color="#c0392b", edgecolor="black",
           linewidth=0.5, label=r"$|\mathrm{MLP}^{(603)}|$")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("DNABERT-2 layer")
    ax.set_ylabel(r"$|$value at row 603$|$")

    src_patch  = mpatches.Patch(facecolor="#fff3e0", edgecolor="none",
                                label="source")
    prop_patch = mpatches.Patch(facecolor="#e7f0fa", edgecolor="none",
                                label="propagator")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles + [src_patch, prop_patch],
              labels_ + ["source", "propagator"],
              loc="upper right", frameon=False, ncol=2,
              handletextpad=0.4, columnspacing=1.2, fontsize=7)


def panel_F2(ax):
    p = RES / "sw_residual_attribution_evo1_fp32.json"
    if not p.exists():
        p = RES / "sw_residual_attribution_evo1.json"
    d = _load(p)
    pl   = d["per_layer"]
    keep = [r for r in pl
            if np.isfinite(r.get("block_residual_in",  float("nan")))
            and np.isfinite(r.get("block_residual_out", float("nan")))]
    layers = [r["layer"] for r in keep]
    mlp    = np.abs([r.get("mlp_out_at_row") or 0.0 for r in keep])
    delta  = np.abs([r["block_delta"]               for r in keep])
    rout   = np.abs([r["block_residual_out"]        for r in keep])

    ax.plot(layers, mlp,   "o-", color="#c0392b", lw=1.2, ms=4, label="MLP write")
    ax.plot(layers, delta, "s-", color="#e67e22", lw=1.0, ms=4, label="block Δ")
    ax.plot(layers, rout,  "^-", color="#1f618d", lw=1.0, ms=4, label="cum. residual")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Evo1 block")
    ax.set_ylabel(r"$|$value at row 3776$|$ (log)")
    ax.legend(loc="lower right", frameon=False, ncol=3,
              handletextpad=0.4, columnspacing=1.0, fontsize=7)
    ax.annotate("MLP writes blow up", xy=(11, rout[layers.index(11)]),
                xytext=(1.5, 3e6), fontsize=7.5, color="#a04000",
                arrowprops=dict(arrowstyle="->", lw=0.5, color="#a04000"))
    if 25 in layers:
        ax.annotate("frozen \u22483\u00d710\u2077\n(DC offset)",
                    xy=(25, rout[layers.index(25)]),
                    xytext=(18, 3e2), fontsize=7.5, color="#1f618d",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="#1f618d"))


# ─── MAIN ───────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(8.5, 11))

    # 3 rows × 2 cols layout: A B / C D / E F
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[1, 1, 1],
        hspace=0.50, wspace=0.38,
        left=0.09, right=0.98, top=0.97, bottom=0.06,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0])

    gsF  = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 1], hspace=0.50)
    axF1 = fig.add_subplot(gsF[0])
    axF2 = fig.add_subplot(gsF[1])

    panel_A(axA);  panel_label(axA,  "A")
    panel_B(axB);  panel_label(axB,  "B")
    panel_C(axC);  panel_label(axC,  "C")
    panel_D(axD);  panel_label(axD,  "D")
    panel_E(axE);  panel_label(axE,  "E")
    panel_F1(axF1); panel_label(axF1, "F", x=-0.08)
    panel_F2(axF2)

    fig.savefig(OUT_PNG, dpi=150)
    fig.savefig(OUT_PDF)
    print("saved:", OUT_PNG)
    print("saved:", OUT_PDF)


if __name__ == "__main__":
    main()
