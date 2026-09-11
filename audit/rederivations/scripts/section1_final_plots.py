#!/usr/bin/env python
"""Round-2 Section 1 final analysis: GC vs c and GC vs NLL, combined with row2371's own
alpha-sweep and the five inert control rows. Produces audit/rederivations/generator_random_direction_full.csv
and two plots in audit/rederivations/figures/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "audit/rederivations"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_NLL = 6.38538052380085
TARGET_NLL = 8.754319605827332

# ---- random-direction grid: NLL from damage_matching.csv, GC from dose_response_extended + new extension
dm_rows = list(csv.DictReader(open(ROOT / "results/experiments/E12/damage_matching.csv")))
rd = [r for r in dm_rows if r["label"] == "random_direction"][0]
grid_nll = {g["scale"]: g["damage"] for g in json.loads(rd["grid_json"])}

dose_rows = list(csv.DictReader(open(ROOT / "results/experiments/E12/dose_response_extended.csv")))
gc_existing = {}
for r in dose_rows:
    if r["label"] == "matched_random_direction":
        gc_existing.setdefault(float(r["alpha_or_c"]), []).append(float(r["gc"]))

ext_recs = [json.loads(l) for l in open(OUT / "raw/gc_extension_records.jsonl")]
gc_new = {}
for r in ext_recs:
    if not np.isnan(r["gc"]):
        gc_new.setdefault(r["c"], []).append(r["gc"])

rd_c, rd_nll, rd_gc, rd_gc_lo, rd_gc_hi = [], [], [], [], []
for c in sorted(grid_nll):
    nll = grid_nll[c]
    gcs = gc_existing.get(c) or gc_new.get(c)
    rd_c.append(c)
    rd_nll.append(nll)
    if gcs:
        rd_gc.append(float(np.mean(gcs)))
        rng = np.random.default_rng(42)
        arr = np.array(gcs)
        boot = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(2000)])
        rd_gc_lo.append(float(np.percentile(boot, 2.5)))
        rd_gc_hi.append(float(np.percentile(boot, 97.5)))
    else:
        rd_gc.append(None)
        rd_gc_lo.append(None)
        rd_gc_hi.append(None)

# ---- row2371's own alpha-sweep (NLL from damage_evals.jsonl, GC from dose_response_extended)
damage_evals = [json.loads(l) for l in open(ROOT / "results/experiments/E12/raw/damage_evals.jsonl")]
row2371_nll = {r["value"]: r["damage"] for r in damage_evals if r["label"] == "row2371"}
row2371_gc = {}
for r in dose_rows:
    if r["label"] == "row2371_grid":
        row2371_gc.setdefault(float(r["alpha_or_c"]), []).append(float(r["gc"]))
row2371_alpha = sorted(row2371_nll)
row2371_nll_y = [row2371_nll[a] for a in row2371_alpha]
row2371_gc_y = [float(np.mean(row2371_gc[a])) for a in row2371_alpha]

# ---- 5 control rows (matched point only, NLL + GC)
control_rows = [2621, 456, 102, 3039, 1126]
ctrl_nll, ctrl_gc = [], []
for cr in control_rows:
    label = f"control_row_{cr}"
    matched_label = f"matched_control_row_{cr}"
    nll = [g for g in dm_rows if g["label"] == label][0]
    matched_scale = float(nll["matched_scale"])
    grid = json.loads(nll["grid_json"])
    nll_val = [g["damage"] for g in grid if g["scale"] == matched_scale][0]
    gcs = [float(r["gc"]) for r in dose_rows if r["label"] == matched_label]
    ctrl_nll.append(nll_val)
    ctrl_gc.append(float(np.mean(gcs)))

# ---- write combined CSV
with open(OUT / "generator_random_direction_full.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["series", "x_c_or_alpha", "nll", "gc_mean", "gc_ci_low", "gc_ci_high", "n_prompts"])
    for c, nll, gc, lo, hi in zip(rd_c, rd_nll, rd_gc, rd_gc_lo, rd_gc_hi):
        n = len(gc_existing.get(c) or gc_new.get(c) or [])
        w.writerow(["random_direction", c, nll, gc, lo, hi, n])
    for a, nll, gc in zip(row2371_alpha, row2371_nll_y, row2371_gc_y):
        w.writerow(["row2371_own_alpha_sweep", a, nll, gc, "", "", 288])
    for cr, nll, gc in zip(control_rows, ctrl_nll, ctrl_gc):
        w.writerow([f"control_row_{cr}", "matched", nll, gc, "", "", 288])
print(f"wrote {OUT/'generator_random_direction_full.csv'}")

# ---- Plot 1: GC vs c (random direction) with reference lines
fig, ax = plt.subplots(figsize=(6.5, 4.5))
xs = [c for c, g in zip(rd_c, rd_gc) if g is not None]
ys = [g for g in rd_gc if g is not None]
los = [g for g in rd_gc_lo if g is not None]
his = [g for g in rd_gc_hi if g is not None]
ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
            fmt="o-", color="tab:red", label="random direction (matched location)", capsize=3)
ax.axhline(0.3065, color="gray", linestyle="--", linewidth=1, label="row2371 full ablation (α=0)")
ax.axhline(0.4204, color="black", linestyle=":", linewidth=1, label="untouched baseline (α=1.0)")
ax.set_xscale("symlog", linthresh=0.01)
ax.set_xlabel("random-direction scale c")
ax.set_ylabel("GC fraction (mean, 96 prompts x 3 seeds)")
ax.set_title("GENERator row 2371: GC vs. random-direction scale")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "gc_vs_c.png", dpi=150)
fig.savefig(FIG_DIR / "gc_vs_c.pdf")
print(f"wrote {FIG_DIR/'gc_vs_c.png'}")

# ---- Plot 2: GC vs NLL, combined (note: row2371's own alpha=0 point sits almost exactly
# under the random-direction c=0 point -- same underlying zeroed-row state -- and alpha=1.0
# sits near the control-row cluster -- same untouched-row state. This is expected overlap,
# not missing data; markers are made semi-transparent with distinct edges so both remain
# visible rather than one fully occluding the other.)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(row2371_nll_y, row2371_gc_y, "o-", color="tab:blue", label="row2371 own α-sweep",
        alpha=0.75, markeredgecolor="navy", markersize=8, zorder=3)
xs_nll = [n for n, g in zip(rd_nll, rd_gc) if g is not None]
ys_nll = [g for g in rd_gc if g is not None]
ax.plot(xs_nll, ys_nll, "s-", color="tab:red", label="random direction (matched location)",
        alpha=0.75, markeredgecolor="darkred", markersize=8, zorder=4)
ax.scatter(ctrl_nll, ctrl_gc, marker="^", color="tab:green", edgecolor="darkgreen",
           s=70, alpha=0.85, label="5 inert control rows", zorder=5)
ax.annotate("row2371 α=0 &\nrandom-dir c=0\n(same zeroed state)",
            xy=(8.754, 0.3065), xytext=(7.9, 0.26),
            fontsize=6.5, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.6))
ax.annotate("row2371 α=1.0 &\ncontrols (untouched)",
            xy=(6.39, 0.42), xytext=(6.9, 0.365),
            fontsize=6.5, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xlabel("NLL (damage pool)")
ax.set_ylabel("GC fraction")
ax.set_title("GENERator row 2371: GC vs. damage (NLL), all conditions")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "gc_vs_nll.png", dpi=150)
fig.savefig(FIG_DIR / "gc_vs_nll.pdf")
print(f"wrote {FIG_DIR/'gc_vs_nll.png'}")

print("\nRandom-direction grid summary (c, NLL, GC):")
for c, nll, gc in zip(rd_c, rd_nll, rd_gc):
    frac_dmg = (nll - BASELINE_NLL) / (TARGET_NLL - BASELINE_NLL)
    gc_str = f"{gc:.4f}" if gc is not None else "n/a"
    print(f"  c={c:<8} nll={nll:.4f} (frac_damage={frac_dmg:.3f})  gc={gc_str}")
