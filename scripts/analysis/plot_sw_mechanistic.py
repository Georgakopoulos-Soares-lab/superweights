"""
scripts/analysis/plot_sw_mechanistic.py
-----------------------------------------
Two-panel figure per model showing mechanistic origin of super-weight rows.

Panel A — W_gate/W_up collinearity scatter
    x: cosine similarity between W_gate[i,:] and W_up[i,:] for each intermediate dim i
    y: W_down column norm  |W_down[:,i]|_2
    SW-driving dimensions highlighted (those with highest combined score)

Panel B — Frobenius norm ||U_k||_F across output dimensions k
    Bar/line plot of frob_norm_uk[k] for all k
    SW output rows highlighted with vertical lines

Reads: results/sw_mechanistic_{model}.json
Saves: results/sw_mechanistic_{model}.png
       results/sw_mechanistic_combined.png   (EUK + PROK side by side)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"

_LABEL = {
    "generator":            "GENERator EUK 3B",
    "generator_prokaryote": "GENERator PROK 3B",
}

_SW_COLOR  = "#d62728"   # red — SW rows
_BG_COLOR  = "#aec7e8"   # light blue — background dims
_HI_COLOR  = "#1f77b4"   # blue — top collinear dims


def load(model_key: str):
    p = RESULTS / f"sw_mechanistic_{model_key}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def plot_model(data: dict, fig, gs_row, title: str):
    cos_sim  = np.array(data["cos_sim"])
    wdn_norm = np.array(data["wdown_col_norm"])
    frob_uk  = np.array(data["frob_norm_uk"])
    sw_rows  = data["sw_rows"]
    d_ffn    = data["d_ffn"]
    d_model  = data["d_model"]

    ax_a = fig.add_subplot(gs_row[0])
    ax_b = fig.add_subplot(gs_row[1])

    # ── Panel A: collinearity scatter ─────────────────────────────────────────
    # colour by combined rank (cos_sim rank + wdn_norm rank)
    rank_cs  = np.argsort(np.argsort(cos_sim))          # ascending rank
    rank_wdn = np.argsort(np.argsort(wdn_norm))
    combined = rank_cs + rank_wdn                        # higher = more extreme

    # subsample background for visibility
    rng  = np.random.default_rng(0)
    keep = rng.choice(d_ffn, size=min(3000, d_ffn), replace=False)

    ax_a.scatter(cos_sim[keep], wdn_norm[keep],
                 c=combined[keep], cmap="Blues", s=4, alpha=0.5,
                 rasterized=True, label="Intermediate dims")

    # top-100 by combined rank highlighted
    top100 = np.argsort(combined)[-100:]
    ax_a.scatter(cos_sim[top100], wdn_norm[top100],
                 color=_HI_COLOR, s=18, alpha=0.85, zorder=3,
                 label="Top-100 by combined rank")

    ax_a.set_xlabel("cos(W_gate[i], W_up[i])", fontsize=9)
    ax_a.set_ylabel(r"$\|W_{\mathrm{down}}[:,i]\|_2$", fontsize=9)
    ax_a.set_title(f"{title}\nA  W_gate / W_up collinearity", fontsize=9, fontweight="bold")
    ax_a.legend(fontsize=7, loc="upper left")

    # ── Panel B: Frobenius norm ||U_k||_F ─────────────────────────────────────
    x = np.arange(d_model)
    ax_b.plot(x, frob_uk, color=_BG_COLOR, lw=0.6, alpha=0.8, label=r"$\|U_k\|_F$")

    # highlight SW rows
    for r in sw_rows:
        ax_b.axvline(r, color=_SW_COLOR, lw=1.4, ls="--", zorder=4)
        ax_b.scatter([r], [frob_uk[r]], color=_SW_COLOR, s=40, zorder=5,
                     label=f"SW row {r}  (rank {sorted(frob_uk, reverse=True).index(frob_uk[r])+1})")

    ax_b.set_xlabel("Output dimension k", fontsize=9)
    ax_b.set_ylabel(r"$\|U_k\|_F$", fontsize=9)
    ax_b.set_title(f"B  Frobenius norm $\\|U_k\\|_F$ (layer {data['sw_layer']})", fontsize=9, fontweight="bold")
    ax_b.legend(fontsize=7, loc="upper right")

    return ax_a, ax_b


def main():
    models = ["generator", "generator_prokaryote"]
    datasets = {m: load(m) for m in models}
    available = [m for m in models if datasets[m] is not None]

    if not available:
        print("No sw_mechanistic_*.json files found in results/. Run run_sw_mechanistic.py first.")
        return

    # ── individual figures ─────────────────────────────────────────────────────
    for m in available:
        data = datasets[m]
        fig = plt.figure(figsize=(12, 4.5))
        gs2 = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
        _plot_pair(data, fig, gs2, _LABEL.get(m, m))
        fig.suptitle(f"Mechanistic origin of super-weight rows — {_LABEL.get(m, m)}",
                     fontsize=10, y=1.01)
        out = RESULTS / f"sw_mechanistic_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {out}")

    # ── combined figure (EUK top row, PROK bottom row) ──────────────────────
    n = len(available)
    fig2 = plt.figure(figsize=(12, 4.5 * n))
    outer = gridspec.GridSpec(n, 1, figure=fig2, hspace=0.55)
    for row_idx, m in enumerate(available):
        data = datasets[m]
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row_idx], wspace=0.35)
        _plot_pair(data, fig2, inner, _LABEL.get(m, m))
    fig2.suptitle("Mechanistic origin of super-weight rows", fontsize=11, y=1.01)
    out2 = RESULTS / "sw_mechanistic_combined.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {out2}")


def _plot_pair(data: dict, fig, gs, title: str):
    cos_sim  = np.array(data["cos_sim"])
    wdn_norm = np.array(data["wdown_col_norm"])
    frob_uk  = np.array(data["frob_norm_uk"])
    sw_rows  = data["sw_rows"]
    d_ffn    = data["d_ffn"]
    d_model  = data["d_model"]

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])

    # Panel A
    rank_cs  = np.argsort(np.argsort(cos_sim))
    rank_wdn = np.argsort(np.argsort(wdn_norm))
    combined = rank_cs + rank_wdn

    rng  = np.random.default_rng(0)
    keep = rng.choice(d_ffn, size=min(3000, d_ffn), replace=False)

    sc = ax_a.scatter(cos_sim[keep], wdn_norm[keep],
                      c=combined[keep], cmap="Blues", s=4, alpha=0.5,
                      rasterized=True)

    top100 = np.argsort(combined)[-100:]
    ax_a.scatter(cos_sim[top100], wdn_norm[top100],
                 color=_HI_COLOR, s=18, alpha=0.9, zorder=3,
                 label="Top-100 (combined rank)")
    ax_a.set_xlabel("cos(W\u2091\u2090\u209c\u2091[i], W\u1d64\u209a[i])", fontsize=9)
    ax_a.set_ylabel(r"$\|W_{\mathrm{down}}[:,i]\|_2$", fontsize=9)
    ax_a.set_title(f"{title}\nA  Gate/Up collinearity vs W_down magnitude", fontsize=8.5, fontweight="bold")
    ax_a.legend(fontsize=7, loc="upper left")
    plt.colorbar(sc, ax=ax_a, label="Combined rank", fraction=0.046, pad=0.04)

    # Panel B
    x = np.arange(d_model)
    ax_b.plot(x, frob_uk, color=_BG_COLOR, lw=0.6, alpha=0.9)

    for r in sw_rows:
        rank_r = sorted(frob_uk, reverse=True).index(frob_uk[r]) + 1
        ax_b.axvline(r, color=_SW_COLOR, lw=1.4, ls="--", zorder=4)
        ax_b.scatter([r], [frob_uk[r]], color=_SW_COLOR, s=50, zorder=5,
                     label=f"SW row {r}  (rank {rank_r}/{d_model})")

    ax_b.set_xlabel("Output dimension k", fontsize=9)
    ax_b.set_ylabel(r"$\|U_k\|_F$ (proxy)", fontsize=9)
    ax_b.set_title(f"B  Quadratic amplifier strength $\\|U_k\\|_F$  (layer {data['sw_layer']})",
                   fontsize=8.5, fontweight="bold")
    ax_b.legend(fontsize=7, loc="upper right")


if __name__ == "__main__":
    main()
