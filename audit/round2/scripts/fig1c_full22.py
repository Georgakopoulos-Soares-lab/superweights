#!/usr/bin/env python
"""Task 2: full 22-model Fig1C comparison. Merges the original 12-model random-control
gap data (results/E11/scale_ladder_controls.csv, via fig1c_random_vs_topk_gaps.csv) with
the newly-computed 10-model batch-2 random-control q1s
(audit/round2/section2_random_control_q1_batch2.csv, same SeedSequence(42) convention)
and the full-22 topK-by-norm gaps (audit/round2/full22_topk_gap_confounds.csv).

Outputs:
  audit/round2/fig1c_random_vs_topk_gaps_full22.csv -- both gaps per model + cohort medians
  audit/round2/figures/fig1c_two_panel.png -- candidate-vs-random-controls (left),
    candidate-vs-top5-by-norm (right), same y-axis scale, same model order.
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT_CSV = ROOT / "audit/round2/fig1c_random_vs_topk_gaps_full22.csv"
OUT_FIG_DIR = ROOT / "audit/round2/figures"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# 12-model original panel: already has per-control-row q1 in scale_ladder_controls.csv
ladder = list(csv.DictReader(open(ROOT / "results/E11/scale_ladder.csv")))
controls = list(csv.DictReader(open(ROOT / "results/E11/scale_ladder_controls.csv")))
by_model_ctrl = defaultdict(list)
for r in controls:
    by_model_ctrl[r["model"]].append(float(r["q1"]))
orig12 = {r["model"]: float(np.mean(by_model_ctrl[r["model"]]))
          for r in ladder if r["source_type"] == "measured"}

# 10 batch-2 models: newly computed random control q1s (this session)
batch2_ctrl = list(csv.DictReader(open(ROOT / "audit/round2/section2_random_control_q1_batch2.csv")))
by_model_ctrl2 = defaultdict(list)
for r in batch2_ctrl:
    by_model_ctrl2[r["model"]].append(float(r["q1"]))
batch2_mean = {m: float(np.mean(v)) for m, v in by_model_ctrl2.items()}

# full-22 candidate q1 + topK gap + confound columns
full22 = list(csv.DictReader(open(ROOT / "audit/round2/full22_topk_gap_confounds.csv")))

rows_out = []
for r in full22:
    model = r["model"]
    cand_q1 = float(r["candidate_q1"])
    if model in orig12:
        mean_ctrl_q1 = orig12[model]
        ctrl_source = "original_12panel_scale_ladder_controls"
    elif model in batch2_mean:
        mean_ctrl_q1 = batch2_mean[model]
        ctrl_source = "batch2_this_session_seedsequence42"
    else:
        raise KeyError(f"no random-control data for {model}")
    random_control_gap = cand_q1 - mean_ctrl_q1
    topk_gap = float(r["topk_gap"])
    rows_out.append(dict(
        model=model, candidate_q1=cand_q1, mean_random_control_q1=mean_ctrl_q1,
        random_control_gap=random_control_gap, topk_by_norm_gap=topk_gap,
        architecture=r["architecture"], domain=r["domain"],
        non_embed_params=r["non_embed_params"], control_source=ctrl_source,
    ))

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
print(f"wrote {OUT_CSV} ({len(rows_out)} rows)")

rc = [r["random_control_gap"] for r in rows_out]
tk = [r["topk_by_norm_gap"] for r in rows_out]
med_rc, med_tk = np.median(rc), np.median(tk)
print(f"\nFull 22-model cohort:")
print(f"  random-control gap:  median={med_rc:.4f}  range=[{min(rc):.4f}, {max(rc):.4f}]")
print(f"  top5-by-norm gap:    median={med_tk:.4f}  range=[{min(tk):.4f}, {max(tk):.4f}]")
print(f"  collapse factor (median random-control / median topK): {med_rc/med_tk:.1f}x")
print("\nper-model:")
for r in sorted(rows_out, key=lambda x: -x["random_control_gap"]):
    print(f"  {r['model']:45s} random={r['random_control_gap']:>+8.4f}  "
          f"topK={r['topk_by_norm_gap']:>+8.4f}  ({r['control_source']})")

# Two-panel figure, same axes, same model order (sorted by random-control gap desc)
order = [r["model"] for r in sorted(rows_out, key=lambda x: -x["random_control_gap"])]
by_model = {r["model"]: r for r in rows_out}
y = np.arange(len(order))
rc_vals = [by_model[m]["random_control_gap"] for m in order]
tk_vals = [by_model[m]["topk_by_norm_gap"] for m in order]
short = [m.split("/")[-1] for m in order]

fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharex=True, sharey=True)
axes[0].barh(y, rc_vals, color="#4C72B0")
axes[0].set_yticks(y)
axes[0].set_yticklabels(short, fontsize=8)
axes[0].set_xlabel("candidate q1 - mean(5 random same-layer controls)")
axes[0].set_title("Candidate vs. random controls\n(existing manuscript claim, n=22)")
axes[0].axvline(0, color="black", linewidth=0.8)

axes[1].barh(y, tk_vals, color="#C44E52")
axes[1].set_xlabel("candidate q1 - max(top-5-by-norm)")
axes[1].set_title("Candidate vs. top-5-by-norm\n(audit's stricter comparison, n=22)")
axes[1].axvline(0, color="black", linewidth=0.8)

fig.suptitle("Figure 1C, honest version -- full 22-model cohort", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_FIG_DIR / "fig1c_two_panel.png", dpi=150)
print(f"\nwrote {OUT_FIG_DIR / 'fig1c_two_panel.png'}")
