#!/usr/bin/env python
"""Section 2 audit recomputation: 22-model causal census verification.

Reads results/experiments/E13/part2_22_model_results.csv (STORED census master) and
recomputes every cohort-level manuscript claim independently of
PART2_EVIDENCE_PACKET.md's narrative, using only the formulas stated in that
packet's Sections 3 and 6 (G = R_candidate - median(R_control_1..5); model-level
percentile bootstrap on the median of G, 5000 draws, seeds 47/52).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "results/experiments/E13/part2_22_model_results.csv"
OUT = ROOT / "audit"


def load():
    with open(CENSUS) as f:
        return list(csv.DictReader(f))


def g_values(rows, eps_suffix):
    return {r["model"]: float(r[f"candidate_minus_median_control_relative_{eps_suffix}"])
            for r in rows}


def bootstrap_median_ci(values, seed, n_boot=5000):
    rng = np.random.default_rng(seed)
    vals = np.array(values)
    n = len(vals)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = np.median(vals[idx])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    rows = load()
    print(f"n models: {len(rows)}")

    for eps_suffix, eps_label, seed in (("eps0p5", "0.5", 47), ("eps1p0", "1.0", 52)):
        g = g_values(rows, eps_suffix)
        n_pos = sum(1 for v in g.values() if v > 0)
        med_g = float(np.median(list(g.values())))
        ci = bootstrap_median_ci(list(g.values()), seed)
        cand_key = f"candidate_relative_loss_change_{eps_suffix}"
        med_cand = float(np.median([float(r[cand_key]) for r in rows]))
        ctrl_key = f"median_control_relative_loss_change_{eps_suffix}"
        med_ctrl = float(np.median([float(r[ctrl_key]) for r in rows]))
        print(f"\n=== epsilon={eps_label} ===")
        print(f"  G>0 count: {n_pos}/{len(g)}")
        print(f"  median G: {med_g*100:.4f}%   (bootstrap 95% CI: [{ci[0]*100:.4f}%, {ci[1]*100:.4f}%], seed={seed})")
        print(f"  median candidate R: {med_cand*100:.4f}%   median-control R median: {med_ctrl*100:.4f}%")
        # named extremes
        sorted_models = sorted(rows, key=lambda r: float(r[cand_key]))
        print("  weakest 3:", [(r["display_model"], f"{float(r[cand_key])*100:.4f}%") for r in sorted_models[:3]])
        print("  strongest 3:", [(r["display_model"], f"{float(r[cand_key])*100:.4f}%") for r in sorted_models[-3:]])

    # Spearman rho: q1 vs signed full-ablation (eps1.0) candidate R
    q1_vals, r_vals, models = [], [], []
    for r in rows:
        q1 = r.get("q1")
        if q1 not in (None, "", "nan", "NaN"):
            q1_vals.append(float(q1))
            r_vals.append(float(r["candidate_relative_loss_change_eps1p0"]))
            models.append(r["model"])
    rho, p = spearmanr(q1_vals, r_vals)
    print(f"\nSpearman q1 vs eps1.0 candidate R: rho={rho:.6f} p={p:.6f} n={len(q1_vals)}")
    rng = np.random.default_rng(43)
    boot = []
    n = len(q1_vals)
    q1a, ra = np.array(q1_vals), np.array(r_vals)
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        rr = spearmanr(q1a[idx], ra[idx])[0]
        if np.isfinite(rr):
            boot.append(rr)
    print(f"  model-bootstrap CI (seed=43, 5000 draws): [{np.percentile(boot,2.5):.6f}, {np.percentile(boot,97.5):.6f}]")

    # subgroup medians at eps1.0
    print("\nSubgroup medians (eps1.0 signed candidate R):")
    groups = {}
    for r in rows:
        key = (r["domain"], r["architecture"])
        groups.setdefault(key, []).append(float(r["candidate_relative_loss_change_eps1p0"]))
    for key, vals in sorted(groups.items()):
        print(f"  {key}: n={len(vals)} median={np.median(vals)*100:.4f}%")

    # the 111.78% coincidence check
    olmo = [r for r in rows if "OLMo" in r["display_model"]][0]
    olmo_eps1 = float(olmo["candidate_relative_loss_change_eps1p0"])
    g_eps1 = g_values(rows, "eps1p0")
    ci_eps1 = bootstrap_median_ci(list(g_eps1.values()), 52)
    print(f"\n111.78% coincidence check:")
    print(f"  OLMo-7B eps1.0 candidate R = {olmo_eps1*100:.6f}%")
    print(f"  cohort median-G bootstrap CI upper bound (seed 52) = {ci_eps1[1]*100:.6f}%")
    print(f"  exactly equal to full precision? {abs(olmo_eps1 - ci_eps1[1]) < 1e-12}")


if __name__ == "__main__":
    main()
