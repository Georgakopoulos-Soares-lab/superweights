#!/usr/bin/env python
"""Emit audit/rederivations/fig1c_random_vs_topk_gaps.csv: for each of the 12 E11-panel models,
the Fig-1C random-control q1 gap (candidate_q1 - mean(5 random same-layer control q1s))
side by side with the top-5-by-norm q1 gap (candidate_q1 - max(top5-by-norm q1s)) computed
in Section 4a. Plus cohort median/range for each.
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "audit/rederivations/fig1c_random_vs_topk_gaps.csv"

ladder = list(csv.DictReader(open(ROOT / "results/experiments/E11/scale_ladder.csv")))
controls = list(csv.DictReader(open(ROOT / "results/experiments/E11/scale_ladder_controls.csv")))
by_model_ctrl = defaultdict(list)
for r in controls:
    by_model_ctrl[r["model"]].append(float(r["q1"]))

cohort = [r for r in ladder if r["source_type"] == "measured"]

topk = list(csv.DictReader(open(ROOT / "audit/rederivations/topk_norm_q1_summary.csv")))
by_model_topk = defaultdict(dict)
for r in topk:
    by_model_topk[r["model"]][r["coord_role"]] = float(r["q1"])

rows_out = []
for r in cohort:
    model = r["model"]
    cand_q1 = float(r["q1"])
    ctrl_q1s = by_model_ctrl[model]
    random_control_gap = cand_q1 - float(np.mean(ctrl_q1s))
    topk_q1s = [v for k, v in by_model_topk[model].items() if k != "candidate"]
    topk_gap = cand_q1 - max(topk_q1s)
    rows_out.append(dict(model=model, candidate_q1=cand_q1,
                          random_control_gap_fig1c=random_control_gap,
                          topk_by_norm_gap=topk_gap,
                          non_embed_params=r["non_embed_params"], architecture=r["architecture"]))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)
print(f"wrote {OUT} ({len(rows_out)} rows)")

rc_gaps = [r["random_control_gap_fig1c"] for r in rows_out]
tk_gaps = [r["topk_by_norm_gap"] for r in rows_out]
print(f"\nFig1C random-control gap:  median={np.median(rc_gaps):.4f}  range=[{min(rc_gaps):.4f}, {max(rc_gaps):.4f}]")
print(f"Top-5-by-norm gap:         median={np.median(tk_gaps):.4f}  range=[{min(tk_gaps):.4f}, {max(tk_gaps):.4f}]")
print(f"Collapse factor (median random-control gap / median topK gap): "
      f"{np.median(rc_gaps)/np.median(tk_gaps):.1f}x")

print("\nper-model:")
for r in rows_out:
    print(f"  {r['model']:45s} random_ctrl_gap={r['random_control_gap_fig1c']:>+8.4f}  "
          f"topK_gap={r['topk_by_norm_gap']:>+8.4f}  arch={r['architecture']}")

# architecture split
print("\n=== Confound 1: architecture split ===")
for arch in ("decoder", "encoder"):
    vals = [r["topk_by_norm_gap"] for r in rows_out if r["architecture"] == arch]
    print(f"  {arch}: n={len(vals)}  median topK gap={np.median(vals):.4f}  range=[{min(vals):.4f},{max(vals):.4f}]")

# scale confound
print("\n=== Confound 2: scale (log10 non-embed params) ===")
from scipy.stats import spearmanr
log_params = [np.log10(float(r["non_embed_params"])) for r in rows_out]
rho, p = spearmanr(log_params, tk_gaps)
print(f"  Spearman rho(log10 non_embed_params, topK gap) = {rho:.4f}  p={p:.4f}  n={len(rows_out)}")
for r in sorted(rows_out, key=lambda r: float(r["non_embed_params"])):
    print(f"  {r['model']:45s} non_embed_params={float(r['non_embed_params']):.3e}  topK_gap={r['topk_by_norm_gap']:>+8.4f}")
