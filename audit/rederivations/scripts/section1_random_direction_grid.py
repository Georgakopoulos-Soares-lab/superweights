#!/usr/bin/env python
"""Round-2 Section 1: GENERator random-direction c-grid re-analysis.

No new compute -- reads only stored artifacts. Emits
audit/rederivations/generator_random_direction_grid.csv with, for every c on the grid:
c, nll, gc_fraction, gc_ci_low, gc_ci_high, n_prompts, n_seeds, obtained_via,
frac_of_full_ablation_damage.

Baseline NLL (untouched row) = 6.38538052380085, confirmed identical (byte-for-byte)
across all six alpha=1.0 / value=1.0 entries in damage_evals.jsonl (row2371 and all
five control rows) -- this is the fixed reference point, not part of the c-grid itself
(the c-grid never contains an "untouched" state; c=0 means the row is replaced by the
zero vector, i.e. full ablation, not "no perturbation").
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DAMAGE_MATCHING = ROOT / "results/experiments/E12/damage_matching.csv"
DAMAGE_EVALS = ROOT / "results/experiments/E12/raw/damage_evals.jsonl"
DOSE = ROOT / "results/experiments/E12/dose_response_extended.csv"
OUT = ROOT / "audit/rederivations/generator_random_direction_grid.csv"

BASELINE_NLL = 6.38538052380085  # verified identical across all 6 alpha=1.0 entries


def bootstrap_ci(vals, seed=42, n_boot=5000):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    n = len(arr)
    draws = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main():
    # damage/NLL grid (14 points, all measured)
    rows = list(csv.DictReader(open(DAMAGE_MATCHING)))
    rd_row = [r for r in rows if r["label"] == "random_direction"][0]
    grid = json.loads(rd_row["grid_json"])

    # GC records: label is "matched_random_direction" in dose_response_extended.csv,
    # NOT "random_direction" (that label is only used in the damage-only artifacts).
    dose_rows = list(csv.DictReader(open(DOSE)))
    gc_by_c = {}
    seeds_by_c = {}
    for r in dose_rows:
        if r["label"] != "matched_random_direction":
            continue
        c = float(r["alpha_or_c"])
        gc_by_c.setdefault(c, []).append(float(r["gc"]))
        seeds_by_c.setdefault(c, set()).add(r["seed"])

    nll_c0 = [g["damage"] for g in grid if g["scale"] == 0.0][0]

    out_rows = []
    for g in sorted(grid, key=lambda g: g["scale"]):
        c, nll = g["scale"], g["damage"]
        gcs = gc_by_c.get(c, [])
        if gcs:
            gc_mean = float(np.mean(gcs))
            lo, hi = bootstrap_ci(gcs)
            n_prompts = len(gcs)
            n_seeds = len(seeds_by_c[c])
            obtained_via = "STORED"
        else:
            gc_mean, lo, hi, n_prompts, n_seeds = "", "", "", 0, 0
            obtained_via = "NOT FOUND (NLL only -- no generation run at this c)"
        frac = (nll - BASELINE_NLL) / (nll_c0 - BASELINE_NLL)
        out_rows.append({
            "c": c, "nll": nll, "gc_fraction": gc_mean, "gc_ci_low": lo, "gc_ci_high": hi,
            "n_prompts": n_prompts, "n_seeds": n_seeds, "obtained_via": obtained_via,
            "frac_of_full_ablation_damage": frac,
        })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["c", "nll", "gc_fraction", "gc_ci_low", "gc_ci_high",
                                          "n_prompts", "n_seeds", "obtained_via",
                                          "frac_of_full_ablation_damage"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT} ({len(out_rows)} rows)")
    print(f"\nbaseline NLL (untouched row, verified identical across 6 conditions): {BASELINE_NLL}")
    print(f"nll at c=0 (full ablation target): {nll_c0}")
    print()
    for r in out_rows:
        print(f"c={r['c']:<8} nll={r['nll']:.6f}  frac_damage={r['frac_of_full_ablation_damage']:.4f}  "
              f"gc={r['gc_fraction'] if r['gc_fraction']!='' else 'NOT MEASURED':>20}  "
              f"n_prompts={r['n_prompts']}")


if __name__ == "__main__":
    main()
