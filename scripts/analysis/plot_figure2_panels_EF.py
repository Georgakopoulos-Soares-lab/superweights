"""Figure 2, panels E & F — clean redesign (no titles, compact labels).

Panel E: cross-architecture ‖U_k‖_F percentile of detected SW rows.
         Filled marker = ablation-positive; open marker = ablation-null.
         The y-axis says it all; no extra annotation needed.

Panel F: residual-stream attribution.
         F1 (top)  — DNABERT-2 row 603 across layers 2–11.
                     bars: residual_in (carry-in) vs mlp_out (this-layer MLP).
                     Source layers shaded gold, propagator layers shaded blue.
         F2 (bot)  — Evo1 row 3776 across all 32 blocks (bf16 trace).
                     three lines: per-block MLP write, full block contribution,
                     cumulative residual; symlog y-axis.

Inputs as before; falls back to fp16 trace if fp32-clean file is missing.
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

ROOT = Path(__file__).resolve().parents[2]
RES  = ROOT / "results"
OUT_PNG = ROOT / "paper" / "media" / "image_fig2_panels_EF.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)


# ─── PANEL E helpers ────────────────────────────────────────────────────────
def _rows_dnabert2(p: Path):
    d = json.loads(p.read_text())
    out = []
    n_layers = max(int(k) for k in d["frob_norm_uk_by_layer"].keys()) + 1
    for li_str, ranks in d.get("sw_row_ranks", {}).items():
        li = int(li_str)
        dm = len(d["frob_norm_uk_by_layer"][li_str])
        for r_str, rank in ranks.items():
            out.append(dict(layer=li, n_layers=n_layers,
                            sw_row=int(r_str), rank=rank, d_model=dm,
                            pct=100.0 * (1 - (rank - 1) / dm)))
    return out


def _rows_ntv3(p: Path):
    return _rows_dnabert2(p)


def _rows_evo1(p: Path):
    d = json.loads(p.read_text())
    out = []
    n_blocks = d.get("n_blocks", 32)
    for li_str, ranks in d.get("detected_row_ranks", {}).items():
        li = int(li_str)
        dm = len(d["frob_norm_uk_by_layer"][li_str])
        for r_str, rank in ranks.items():
            out.append(dict(layer=li, n_layers=n_blocks,
                            sw_row=int(r_str), rank=rank, d_model=dm,
                            pct=100.0 * (1 - (rank - 1) / dm)))
    return out


def _rows_generator(p: Path):
    """Same schema as _rows_dnabert2 but uses 'sw_row_ranks' key."""
    return _rows_dnabert2(p)


def panel_E(ax):
    db = _rows_dnabert2(RES / "sw_mechanistic_dnabert2.json")
    nt = _rows_ntv3   (RES / "sw_mechanistic_ntv3.json")
    ev = _rows_evo1   (RES / "sw_mechanistic_evo1.json")

    gen_euk_path  = RES / "sw_mechanistic_generator.json"
    gen_prok_path = RES / "sw_mechanistic_generator_prokaryote.json"
    gen_euk  = _rows_generator(gen_euk_path)  if gen_euk_path.exists()  else []
    gen_prok = _rows_generator(gen_prok_path) if gen_prok_path.exists() else []
    gen_rows = gen_euk + gen_prok

    def scatter(rows, marker, color, label, *, filled=True, size=85):
        if not rows:
            return
        xs = [r["layer"] / max(r["n_layers"] - 1, 1) for r in rows]
        ys = [r["pct"] for r in rows]
        if filled:
            ax.scatter(xs, ys, marker=marker, color=color, s=size,
                       edgecolor="black", linewidth=0.9, label=label, zorder=3)
        else:
            ax.scatter(xs, ys, marker=marker, facecolors="none", edgecolor=color,
                       linewidth=1.4, s=size, label=label, zorder=3)

    scatter(db,       "s", "#2c3e50", "DNABERT-2",    filled=True,  size=70)
    scatter(nt,       "D", "#16a085", "NTv3",          filled=True,  size=95)
    scatter(ev,       "^", "#444444", "Evo1 (null)",   filled=False, size=85)
    scatter(gen_rows, "o", "#c0392b", "GENERator",     filled=True,  size=90)

    if not gen_rows:
        # GENERator data not yet generated — show placeholder in legend
        gen_patch = mpatches.Patch(facecolor="#c0392b", edgecolor="black",
                                   label="GENERator (pending)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [gen_patch], labels + ["GENERator (pending)"],
                  loc="lower left", frameon=False, ncol=2,
                  handletextpad=0.4, columnspacing=1.2)
    else:
        ax.legend(loc="lower left", frameon=False, ncol=2,
                  handletextpad=0.4, columnspacing=1.2)

    # mark the DNABERT-2 L7 r603 outlier (low-percentile, residual-carried)
    out = [r for r in db if r["layer"] == 7 and r["sw_row"] == 603]
    if out:
        r = out[0]
        x = r["layer"] / max(r["n_layers"] - 1, 1)
        ax.annotate("L7 r603 (residual-carried)",
                    xy=(x, r["pct"]), xytext=(x + 0.05, r["pct"] + 20),
                    fontsize=7.5, color="#a04000", ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="#a04000"))

    ax.axhline(99, color="#aaa", lw=0.5, ls=":")
    ax.text(0.02, 99.4, "99th pct", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6.5, color="#777")
    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-3, 108)
    ax.set_xlabel("relative layer position")
    ax.set_ylabel(r"$\|U_k\|_F$ percentile within layer")


# ─── PANEL F1 (DNABERT-2) ───────────────────────────────────────────────────
def panel_F1(ax):
    d = json.loads((RES / "sw_residual_attribution_dnabert2.json").read_text())
    keep = [r for r in d["per_layer"] if 2 <= r["layer"] <= 11]
    layers = [r["layer"] for r in keep]
    res_in = [abs(r["residual_in"]) for r in keep]
    mlp    = [abs(r["mlp_out"])     for r in keep]

    SOURCE     = {3, 5, 6}
    PROPAGATOR = {7, 8}

    # background shading
    for li, kind, color in [
        *((li, "src",  "#fff3e0") for li in SOURCE),
        *((li, "prop", "#e7f0fa") for li in PROPAGATOR),
    ]:
        xi = layers.index(li)
        ax.axvspan(xi - 0.5, xi + 0.5, color=color, zorder=0)

    x = np.arange(len(layers))
    w = 0.40
    ax.bar(x - w/2, res_in, w, color="#7f8c8d", edgecolor="black",
           linewidth=0.5, label=r"$|h_{\mathrm{in}}^{(603)}|$ (carry-in)")
    ax.bar(x + w/2, mlp,    w, color="#c0392b", edgecolor="black",
           linewidth=0.5, label=r"$|\mathrm{MLP}^{(603)}|$ (this layer)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("DNABERT-2 layer")
    ax.set_ylabel(r"$|$value at row 603$|$")

    # combined legend: bar labels + regime swatches in one box
    src_patch  = mpatches.Patch(facecolor="#fff3e0", edgecolor="none",
                                label="source (MLP-driven)")
    prop_patch = mpatches.Patch(facecolor="#e7f0fa", edgecolor="none",
                                label="propagator (residual-carried)")
    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles + [src_patch, prop_patch],
              labels_ + ["source (MLP-driven)", "propagator (residual-carried)"],
              loc="upper right", frameon=False, ncol=2,
              handletextpad=0.4, columnspacing=1.2)


# ─── PANEL F2 (Evo1) ────────────────────────────────────────────────────────
def panel_F2(ax):
    p = RES / "sw_residual_attribution_evo1_fp32.json"
    if not p.exists():
        p = RES / "sw_residual_attribution_evo1.json"
    d = json.loads(p.read_text())

    pl = d["per_layer"]
    keep = [r for r in pl
            if np.isfinite(r.get("block_residual_in",  float("nan")))
            and np.isfinite(r.get("block_residual_out", float("nan")))]
    layers = [r["layer"] for r in keep]
    mlp    = np.abs([r.get("mlp_out_at_row") or 0.0 for r in keep])
    delta  = np.abs([r["block_delta"]              for r in keep])
    rout   = np.abs([r["block_residual_out"]       for r in keep])

    ax.plot(layers, mlp,   "o-", color="#c0392b", lw=1.2, ms=4, label="MLP write")
    ax.plot(layers, delta, "s-", color="#e67e22", lw=1.0, ms=4, label="block Δ")
    ax.plot(layers, rout,  "^-", color="#1f618d", lw=1.0, ms=4, label="cum. residual")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Evo1 block")
    ax.set_ylabel(r"$|$value at row 3776$|$ (log)")
    ax.legend(loc="lower right", frameon=False, ncol=3, handletextpad=0.4,
              columnspacing=1.0)

    # Two compact inline labels
    ax.annotate("MLP writes blow up", xy=(11, rout[layers.index(11)]),
                xytext=(1.5, 3e6), fontsize=7.5, color="#a04000",
                arrowprops=dict(arrowstyle="->", lw=0.5, color="#a04000"))
    if 25 in layers:
        ax.annotate("frozen \u22483\u00b710\u2077\n(DC offset, normalised\naway at unembed)",
                    xy=(25, rout[layers.index(25)]),
                    xytext=(18, 3e2), fontsize=7.5, color="#1f618d",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="#1f618d"))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.35],
                           hspace=0.32,
                           left=0.07, right=0.98, top=0.97, bottom=0.06)
    axE = fig.add_subplot(gs[0])
    panel_E(axE)
    panel_label(axE, "E")

    gsF = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.40)
    axF1 = fig.add_subplot(gsF[0])
    axF2 = fig.add_subplot(gsF[1])
    panel_F1(axF1)
    panel_F2(axF2)
    panel_label(axF1, "F", x=-0.06)
    # no second label — F1/F2 are stacked sub-panels of F

    fig.savefig(OUT_PNG)
    fig.savefig(OUT_PDF)
    print("saved:", OUT_PNG)
    print("saved:", OUT_PDF)


if __name__ == "__main__":
    main()
