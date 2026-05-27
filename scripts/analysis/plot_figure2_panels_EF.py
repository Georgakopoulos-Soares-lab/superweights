"""
scripts/analysis/plot_figure2_panels_EF.py
------------------------------------------
Renders the two NEW panels of Figure 2 in the v4 manuscript:

  Panel E — Cross-architecture ||U_k||_F percentile of the empirically
            detected SW row(s), one marker per known SW row.
              ● GENERator EUK / PROK  (filled red circles)    — ablation-positive
              ■ DNABERT-2              (filled blue squares)   — ablation-positive
              ◆ NTv3                   (filled green diamond)  — ablation-positive
              △ Evo1                   (open  black triangles) — ablation-NULL

            X-axis: layer index (normalized to [0,1] for cross-model display).
            Y-axis: ||U_k||_F percentile of the SW row within its layer
                   (100 = top of the layer; 0 = bottom).
            The filled / open marker key communicates the
            "necessary-but-not-sufficient" finding visually.

  Panel F — Residual-stream attribution.
            Top sub-panel    : DNABERT-2 row 603 across layers 2–11.
                                  bars: |residual_in[603]| (carry-in)
                                        |mlp_out[603]|     (this layer's MLP)
                                  source vs propagator regimes annotated.
            Bottom sub-panel : Evo1 row 3776 across all 32 blocks (bf16-stable
                                trace). Three traces:
                                  mlp_out[3776], block_delta[3776],
                                  residual_out[3776].
                                Annotated: the row enters the residual
                                stream essentially unchanged, accumulates
                                across the StripedHyena conv blocks,
                                saturates after L12 (post-L12 MLP writes
                                ~0), and is normalized away at the unembed
                                — hence ΔPPL = 0%.

Inputs (must all exist):
  results/sw_mechanistic_dnabert2.json
  results/sw_mechanistic_ntv3.json
  results/sw_mechanistic_evo1.json
  results/sw_residual_attribution_dnabert2.json
  results/sw_residual_attribution_evo1_fp32.json   (preferred, bf16-clean)
  OR results/sw_residual_attribution_evo1.json     (fp16, layers >10 NaN)
  results/super_weight_index.json                  (for layer counts)

GENERator EUK / PROK markers in panel E are read from
  results/sw_mechanistic_generator.json
  results/sw_mechanistic_generator_prokaryote.json
If those JSONs are not present, panel E annotates "GENERator data missing
— request from collaborator with EUK/PROK environment" and renders the
DNABERT-2 / NTv3 / Evo1 markers only.

Output:
  paper/media/image_fig2_panels_EF.png  (+ pdf)

Used as a drop-in to the full Figure 2 composite — combine with the EUK /
PROK activation-lifecycle (panels A–B) and GENERator U_k step-up bar
charts (panels C–D) once those are generated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[2]
RES  = ROOT / "results"
OUT_PNG = ROOT / "paper" / "media" / "image_fig2_panels_EF.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PANEL E — Cross-architecture ||U_k||_F percentile scatter
# ─────────────────────────────────────────────────────────────────────────────
def _percentile_for_sw(uk_path: Path, model_name: str) -> list[dict]:
    """Return list of dicts with layer_idx, n_layers, sw_row, rank, d_model, percentile."""
    if not uk_path.exists():
        return []
    d = json.loads(uk_path.read_text())
    rows = []
    sw_rows_by_layer = d.get("sw_rows_by_layer") or {}
    sw_row_ranks    = d.get("sw_row_ranks")    or {}
    d_model = d.get("d_model")
    # n_layers from frob_norm_uk_by_layer or detected
    layer_keys = sorted({int(k) for k in d.get("frob_norm_uk_by_layer", {}).keys()})
    n_layers = max(layer_keys) + 1 if layer_keys else 12
    if "n_blocks" in d:
        n_layers = d["n_blocks"]
    # if we have explicit sw_row_ranks
    for li_str, ranks in sw_row_ranks.items():
        li = int(li_str)
        for r_str, rank in ranks.items():
            # Need d_model: read frob array length if possible
            frob_list = d.get("frob_norm_uk_by_layer", {}).get(str(li), [])
            dm = len(frob_list) if frob_list else (d_model or 1)
            pct = 100.0 * (1.0 - (rank - 1) / dm) if dm > 0 else None
            rows.append({"model": model_name, "layer": li, "n_layers": n_layers,
                         "sw_row": int(r_str), "rank": rank, "d_model": dm,
                         "percentile": pct, "ablation_positive": True})
    return rows


def _evo1_rows(uk_path: Path) -> list[dict]:
    """Evo1 returns 'detected_row_ranks' rather than 'sw_row_ranks'."""
    if not uk_path.exists():
        return []
    d = json.loads(uk_path.read_text())
    rows = []
    detected = d.get("detected_row_ranks", {})
    n_blocks = d.get("n_blocks", 32)
    for li_str, ranks in detected.items():
        li = int(li_str)
        for r_str, rank in ranks.items():
            frob_list = d.get("frob_norm_uk_by_layer", {}).get(str(li), [])
            dm = len(frob_list) if frob_list else 4096
            pct = 100.0 * (1.0 - (rank - 1) / dm)
            rows.append({"model": "Evo1", "layer": li, "n_layers": n_blocks,
                         "sw_row": int(r_str), "rank": rank, "d_model": dm,
                         "percentile": pct, "ablation_positive": False})
    return rows


def _generator_rows(uk_path: Path, label: str) -> list[dict]:
    """GENERator-format ||U_k||_F audit (if present)."""
    if not uk_path.exists():
        return []
    d = json.loads(uk_path.read_text())
    rows = []
    # Try multiple plausible schemas
    sw_row_ranks = d.get("sw_row_ranks") or {}
    n_layers = d.get("n_layers", 30)
    if sw_row_ranks:
        for li_str, ranks in sw_row_ranks.items():
            li = int(li_str)
            for r_str, rank in ranks.items():
                frob_list = d.get("frob_norm_uk_by_layer", {}).get(str(li), [])
                dm = len(frob_list) if frob_list else d.get("d_model", 3072)
                pct = 100.0 * (1.0 - (rank - 1) / dm)
                rows.append({"model": label, "layer": li, "n_layers": n_layers,
                             "sw_row": int(r_str), "rank": rank, "d_model": dm,
                             "percentile": pct, "ablation_positive": True})
        return rows
    # Sun-et-al-style single-layer schema (sw_layer, sw_row, frob_rank)
    if "sw_layer" in d and "sw_row" in d and "frob_rank" in d:
        li = int(d["sw_layer"])
        rows.append({"model": label, "layer": li, "n_layers": n_layers,
                     "sw_row": int(d["sw_row"]), "rank": int(d["frob_rank"]),
                     "d_model": int(d.get("d_model", 3072)),
                     "percentile": 100.0 * (1.0 - (d["frob_rank"] - 1) / d.get("d_model", 3072)),
                     "ablation_positive": True})
    return rows


def panel_E(ax):
    dnabert = _percentile_for_sw(RES / "sw_mechanistic_dnabert2.json", "DNABERT-2")
    ntv3    = _percentile_for_sw(RES / "sw_mechanistic_ntv3.json",     "NTv3")
    evo1    = _evo1_rows(         RES / "sw_mechanistic_evo1.json")
    euk     = _generator_rows(    RES / "sw_mechanistic_generator.json",            "GENERator EUK")
    prok    = _generator_rows(    RES / "sw_mechanistic_generator_prokaryote.json", "GENERator PROK")

    def _plot(rows, marker, color, label, fill="full", size=85, edge_lw=1.0):
        if not rows:
            return False
        xs = [r["layer"] / max(r["n_layers"] - 1, 1) for r in rows]
        ys = [r["percentile"] for r in rows]
        if fill == "full":
            ax.scatter(xs, ys, marker=marker, color=color, s=size,
                       edgecolor="black", linewidth=edge_lw, label=label,
                       zorder=3)
        else:
            ax.scatter(xs, ys, marker=marker, facecolors="none",
                       edgecolor=color, linewidth=edge_lw + 0.3, s=size,
                       label=label, zorder=3)
        return True

    plotted_any = False
    plotted_any |= _plot(euk,     "o", "#c0392b", "GENERator EUK (ablation+)", "full", 85)
    plotted_any |= _plot(prok,    "o", "#7d0c0c", "GENERator PROK (ablation+)", "full", 70)
    plotted_any |= _plot(dnabert, "s", "#2c3e50", "DNABERT-2 (ablation+)",       "full", 60)
    plotted_any |= _plot(ntv3,    "D", "#16a085", "NTv3 (ablation+)",            "full", 95)
    plotted_any |= _plot(evo1,    "^", "#404040", "Evo1 (ablation null)",         "open", 75)

    # Annotate the DNABERT-2 layer-7 outlier explicitly (rank 706/768)
    if dnabert:
        outlier = [r for r in dnabert if r["layer"] == 7 and r["sw_row"] == 603]
        if outlier:
            r = outlier[0]
            x = r["layer"] / max(r["n_layers"] - 1, 1)
            # Plot the L7 outlier emphatically so it's not lost behind annotations
            ax.scatter([x], [r["percentile"]], marker="s", color="#2c3e50",
                       edgecolor="#d35400", linewidth=2.0, s=140, zorder=5)
            ax.annotate("DNABERT-2 L7 r603\n(residual-carried,\nU_k blind: pct ≈ 8)",
                        xy=(x, r["percentile"]),
                        xytext=(x - 0.27, r["percentile"] + 28),
                        fontsize=8, color="#a04000",
                        arrowprops=dict(arrowstyle="->", lw=0.8, color="#a04000"))

    ax.axhline(99, color="#999", lw=0.6, ls=":", zorder=1)
    ax.text(0.99, 99.5, "99th percentile", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6.5, color="#666")
    ax.set_xlim(-0.02, 1.06)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Normalised layer position (0 = first, 1 = last)")
    ax.set_ylabel("||U_k||_F percentile of SW row within its layer")
    ax.set_title("E. Cross-architecture ||U_k||_F percentile of detected SW rows\n"
                 "Filled = ablation-positive, Open = ablation-null",
                 fontsize=10, loc="left")
    ax.legend(loc="lower left", fontsize=7, frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)

    if not euk and not prok:
        ax.text(0.5, 0.06,
                "GENERator EUK / PROK ||U_k||_F missing\n"
                "— see scripts/analysis/FIGURES_README.md for the data file\n"
                "  expected from the collaborator with GENERator access",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, color="#666",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#fff7e6", edgecolor="#d68000", lw=0.5))


# ─────────────────────────────────────────────────────────────────────────────
# PANEL F — Residual-stream attribution (DNABERT-2 + Evo1)
# ─────────────────────────────────────────────────────────────────────────────
def _draw_dnabert2_residual(ax):
    d = json.loads((RES / "sw_residual_attribution_dnabert2.json").read_text())
    pl = d["per_layer"]
    # Skip layers 0,1 (row 603 not yet present); plot 2..11
    keep = [r for r in pl if 2 <= r["layer"] <= 11]
    layers = [r["layer"] for r in keep]
    res_in  = [abs(r["residual_in"])  for r in keep]
    mlp_out = [abs(r["mlp_out"])      for r in keep]

    x = np.arange(len(layers))
    w = 0.40
    ax.bar(x - w/2, res_in,  w, color="#7f8c8d", edgecolor="black",
           linewidth=0.5, label="|residual_in[603]|  (carry-in)")
    ax.bar(x + w/2, mlp_out, w, color="#c0392b", edgecolor="black",
           linewidth=0.5, label="|mlp_out[603]|  (this layer's MLP)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_xlabel("DNABERT-2 layer", fontsize=8)
    ax.set_ylabel("|value at SW row 603|", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.set_title("F (top). DNABERT-2 row 603 — source vs propagator layers",
                 fontsize=9.5, loc="left")
    # Source / propagator annotations
    for li, ri, mi in zip(layers, res_in, mlp_out):
        xi = layers.index(li)
        if li in (3, 5, 6):
            ax.annotate("source\n(MLP-dominant)", xy=(xi + w/2, mi),
                        xytext=(xi, mi + 1.5), ha="center", fontsize=6,
                        color="#a04000",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#a04000"))
        elif li in (7, 8):
            ax.annotate("propagator\n(residual-carried)", xy=(xi - w/2, ri),
                        xytext=(xi, ri + 1.5), ha="center", fontsize=6,
                        color="#1f618d",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#1f618d"))
    ax.grid(True, axis="y", alpha=0.25)


def _draw_evo1_residual(ax):
    fp32_path = RES / "sw_residual_attribution_evo1_fp32.json"
    fp16_path = RES / "sw_residual_attribution_evo1.json"
    path = fp32_path if fp32_path.exists() else fp16_path
    d = json.loads(path.read_text())
    dtype_note = d.get("dtype", "float16")
    pl = d["per_layer"]
    keep = [r for r in pl if r and "block_residual_in" in r
            and np.isfinite(r.get("block_residual_in", float("nan")))
            and np.isfinite(r.get("block_residual_out", float("nan")))]
    layers = [r["layer"] for r in keep]
    mlp    = [r.get("mlp_out_at_row") or 0.0 for r in keep]
    delta  = [r["block_delta"] for r in keep]
    rout   = [r["block_residual_out"] for r in keep]

    ax.plot(layers, np.abs(mlp),   "o-", color="#c0392b",
            label="|mlp_out[3776]|  (per-block write)", lw=1.2, ms=4)
    ax.plot(layers, np.abs(delta), "s-", color="#e67e22",
            label="|block_delta[3776]|  (full block contribution)", lw=1.0, ms=4)
    ax.plot(layers, np.abs(rout),  "^-", color="#1f618d",
            label="|residual_out[3776]|  (cumulative residual)", lw=1.0, ms=4)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Evo1 block index", fontsize=8)
    ax.set_ylabel("magnitude at row 3776 (log)", fontsize=8)
    ax.set_title(f"F (bottom). Evo1 row 3776 — residual trajectory across all 32 blocks "
                 f"(dtype={dtype_note})",
                 fontsize=9.5, loc="left")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    # Annotate L10/L11 spike & post-L12 freeze
    if 10 in layers:
        ax.annotate("L10/11: gated MLP\nwrites huge values\ninto row 3776",
                    xy=(11, np.abs(rout[layers.index(11)] if 11 in layers else rout[layers.index(10)])),
                    xytext=(2, np.abs(rout[layers.index(10)]) * 0.05),
                    fontsize=6.5, ha="left", color="#a04000",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="#a04000"))
    if 20 in layers:
        ax.annotate("Post-L12: MLP@row ≈ 0,\nresidual frozen — channel\nbecomes a DC offset\n(normalised away at unembed)",
                    xy=(28, np.abs(rout[layers.index(28)])),
                    xytext=(15, 5.0),
                    fontsize=6.5, ha="left", color="#1f618d",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="#1f618d"))
    ax.grid(True, axis="y", alpha=0.25, which="both")


def panel_F(top_ax, bot_ax):
    _draw_dnabert2_residual(top_ax)
    _draw_evo1_residual(bot_ax)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(13, 11.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.05, 1.45],
                           hspace=0.40, top=0.92, bottom=0.05,
                           left=0.07, right=0.97)

    # Panel E
    axE = fig.add_subplot(gs[0])
    panel_E(axE)

    # Panel F (two stacked sub-axes)
    gs_F = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1], hspace=0.40)
    axF_top = fig.add_subplot(gs_F[0])
    axF_bot = fig.add_subplot(gs_F[1])
    panel_F(axF_top, axF_bot)

    fig.suptitle("Figure 2 (panels E & F) — Cross-architecture ||U_k||_F percentile + Residual-stream attribution",
                 fontsize=13, y=0.995)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved: {OUT_PNG}")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
