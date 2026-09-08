#!/usr/bin/env python3
"""Two-panel figure: within-model ratio->damage relation in a genomic and a text decoder.

Left  GENERator-EUK-3B  L4/r2371  (genomic decoder)
Right SmolLM2-1.7B      L7/r227   (text decoder)

Both panels use the same design (36 rows log-spaced by activation-ratio RANK inside one
layer, each ablated to alpha=0, native LM NLL) so the only difference is the model. The
shaded band marks the detector's own accept region (ratio >= 5); the inset reports the
Spearman rho over all rows, over rows excluding the frozen candidate, and over the
sub-threshold rows alone AND the supra-threshold rows alone.

The two-regime decomposition (rho below vs above the detector's ratio>=5 accept rule) was
added POST HOC, after the pre-specified statistics -- rho(all), rho(excl. candidate),
rho(ratio<5) -- were computed and plotted. The split point is the detector's own
pre-existing threshold and was not fitted to these data, but the decision to decompose came
after seeing the scatter, and is labelled exploratory everywhere it is reported.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/paper_closing"
TH = 5.0

PANELS = [("within_model_slope.json", "GENERator-EUK-3B", "genomic decoder, L4 / r2371", "#1f4e79"),
          ("within_model_slope_smollm2_1.7b.json", "SmolLM2-1.7B", "text decoder, L7 / r227", "#8b2500")]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
for ax, (fn, name, sub, col) in zip(axes, PANELS):
    d = json.loads((R / fn).read_text())
    rows = d["rows"]
    x = np.array([q["activation_ratio"] for q in rows])
    y = np.array([q["rel_delta"] for q in rows])
    isc = np.array([q["is_frozen_candidate"] for q in rows])
    # plot relative damage on a symlog axis: values span 1e-5 .. 7
    ax.axhline(0, color="0.75", lw=0.8, zorder=1)
    ax.axvspan(TH, x.max() * 3, color="0.92", zorder=0)
    ax.scatter(x[~isc], y[~isc], s=34, facecolor="white", edgecolor=col, lw=1.3,
               zorder=3, label="swept rows (rank log-spaced)")
    ax.scatter(x[isc], y[isc], s=145, marker="*", color=col, zorder=4,
               label="frozen census candidate")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xlabel(r"activation ratio  $\max_t|a_{l,r,t}|\ /\ \mathrm{median}_{r'}$")
    ax.set_ylabel("relative NLL increase when row ablated")
    ax.set_title(f"{name}\n{sub}", fontsize=10.5)
    m, s, u = ~isc, x < TH, x >= TH
    rho, p = stats.spearmanr(x, y)
    rx, px = stats.spearmanr(x[m], y[m])
    rs, ps = stats.spearmanr(x[s], y[s])
    ru, pu = stats.spearmanr(x[u], y[u])
    txt = (f"all rows        $\\rho$={rho:+.3f} (p={p:.1e}, n={len(x)})\n"
           f"excl. candidate $\\rho$={rx:+.3f} (p={px:.1e}, n={m.sum()})\n"
           f"ratio<5  only   $\\rho$={rs:+.3f} (p={ps:.1e}, n={s.sum()})\n"
           f"ratio$\\geq$5 only   $\\rho$={ru:+.3f} (p={pu:.1e}, n={u.sum()})")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            family="monospace",
            bbox=dict(fc="white", ec="0.7", lw=0.7, boxstyle="round,pad=0.35"))
    ax.text(0.985, 0.06, "detector accept\nregion (ratio$\\geq$5)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="0.35")
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.95,
              bbox_to_anchor=(1.0, 0.20))
    ax.set_xlim(x.min() / 2.5, x.max() * 3)

fig.suptitle("Activation ratio is a GATED severity predictor: uninformative among ordinary "
             "rows, strongly graded once the detector accepts\n"
             r"(sub- vs supra-threshold split is post hoc; $\rho$(ratio$\geq$5) "
             r"= +0.64 genomic, +0.98 text)", fontsize=10.5, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(R / f"fig_within_model_slope_2panel.{ext}", dpi=200, bbox_inches="tight")
print("saved ->", R / "fig_within_model_slope_2panel.png")
