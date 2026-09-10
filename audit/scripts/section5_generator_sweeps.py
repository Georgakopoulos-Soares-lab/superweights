#!/usr/bin/env python
"""Build audit/generator_alpha_sweeps.csv from E12's raw artifacts.

Columns: row_id, alpha, nll, gc_fraction, gc_ci_low, gc_ci_high, n_prompts, n_seeds
for row 2371, the five control rows, and the random-direction condition.

nll/damage comes from results/experiments/E12/raw/damage_evals.jsonl (100-window damage pool,
matches the manuscript's "damage" endpoint). gc_fraction/CI comes from aggregating
results/experiments/E12/dose_response_extended.csv's per-record gc column (96-window generation
pool) over prompts and seeds at matching alpha values, with a percentile bootstrap CI
(5000 draws, seed 42 -- matches E12's own bootstrap_gc_diff convention in e12_lib.py).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DAMAGE = ROOT / "results/experiments/E12/raw/damage_evals.jsonl"
DOSE = ROOT / "results/experiments/E12/dose_response_extended.csv"
OUT = ROOT / "audit/generator_alpha_sweeps.csv"

LABEL_TO_ROW = {"row2371": 2371, "control_row_2621": 2621, "control_row_456": 456,
                "control_row_102": 102, "control_row_3039": 3039, "control_row_1126": 1126,
                "random_direction": 2371}


def bootstrap_ci(vals, seed=42, n_boot=5000):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    n = len(arr)
    draws = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    damage_recs = [json.loads(l) for l in open(DAMAGE)]
    damage_by_label_alpha = {(r["label"], r["value"]): r["damage"] for r in damage_recs}

    dose_rows = list(csv.DictReader(open(DOSE)))
    gc_groups = defaultdict(list)
    for r in dose_rows:
        key = (r["label"], float(r["alpha_or_c"]))
        gc_groups[key].append(float(r["gc"]))

    out_rows = []
    labels_matching = set(l for l, a in damage_by_label_alpha) | set(l for l, a in gc_groups)
    for (label, alpha), damage in sorted(damage_by_label_alpha.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row_id = LABEL_TO_ROW.get(label)
        gcs = gc_groups.get((label, alpha), [])
        seeds = set()
        for r in dose_rows:
            if r["label"] == label and float(r["alpha_or_c"]) == alpha:
                seeds.add(r["seed"])
        if gcs:
            gc_mean = float(np.mean(gcs))
            lo, hi = bootstrap_ci(gcs)
        else:
            gc_mean, lo, hi = "", "", ""
        out_rows.append({
            "row_id": row_id, "condition_label": label, "alpha": alpha, "nll": damage,
            "gc_fraction": gc_mean, "gc_ci_low": lo, "gc_ci_high": hi,
            "n_prompts": len(gcs), "n_seeds": len(seeds),
        })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "condition_label", "alpha", "nll",
                                          "gc_fraction", "gc_ci_low", "gc_ci_high",
                                          "n_prompts", "n_seeds"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT} ({len(out_rows)} rows)")

    # sanity printout: row2371's own sweep and each control's max achievable damage
    print("\nrow2371 damage sweep:")
    for r in out_rows:
        if r["condition_label"] == "row2371":
            print(f"  alpha={r['alpha']}: nll={r['nll']:.4f}")

    print("\nControl / random-direction max achievable damage vs target 8.754320:")
    by_label = defaultdict(list)
    for r in out_rows:
        by_label[r["condition_label"]].append(r)
    for label, rows in by_label.items():
        if label == "row2371":
            continue
        max_damage = max(r["nll"] for r in rows)
        alpha_at_max = [r["alpha"] for r in rows if r["nll"] == max_damage][0]
        min_damage = min(r["nll"] for r in rows)
        print(f"  {label}: damage range [{min_damage:.6f}, {max_damage:.6f}] (max at alpha={alpha_at_max}), "
              f"gap to target = {8.754319605827332 - max_damage:.6f}")


if __name__ == "__main__":
    main()
