#!/usr/bin/env python
"""Round-2 Section 2: tomography condition-level uncertainty via repeated-split analysis.

Feasibility check (reported, not assumed): the fit/calibration/held_out pool sizes per
density bucket (rho_0.25=34, rho_0.5=50, rho_0.75=34) exactly equal the number of
already-measured non-singleton conditions at that density (34/50/34) -- confirmed by
direct count against masks_dnabert2.json. This means there is exactly enough measured
data to cover one partition and no slack to draw additional, never-measured subsets
without new model runs. A "fresh partition" here is therefore a re-permutation of ROLE
labels (fit/calibration/held_out) among the fixed set of 118 already-measured non-singleton
conditions per density, not a resample from the wider C(10,k) combinatorial space. This
still gives genuine variation in which subsets land in held-out across resamples (the
scientific question asked), it just cannot introduce subsets that were never measured --
reported explicitly rather than silently assumed.

Reuses run_fit_observers.fit_epsilon() unmodified -- only the pool dict's fit/calibration/
held_out membership is resampled; the function itself doesn't know or care how pools were
constructed.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
E9 = ROOT / "paper-salvage/experiments/E9_mechanistic_tomography"
OUT = ROOT / "audit/round2"
sys.path.insert(0, str(E9))
import run_fit_observers as rfo  # noqa: E402

# The per-split nested 5000-draw bootstrap (bootstrap_mae_diff) is not needed for this
# analysis -- we want the point R2/MAE distribution ACROSS 100 outer splits, which already
# captures split-driven variance; a 5000-draw inner CI per split x per epsilon x per split
# (200 calls) is redundant here and was the actual cost driver (pure-Python nested loops).
# Stubbed to a cheap no-op; fit_epsilon still records mechanical_decision etc. correctly
# since it only *uses* the bootstrap dict for the pair_improves check, not for R2/MAE.
_ORIG_BOOTSTRAP = rfo.bootstrap_mae_diff
def _fast_stub_bootstrap(*a, **kw):
    return {"mean_MAE_F2_minus_F3": float("nan"), "ci_2.5": float("nan"),
            "ci_97.5": float("nan"), "excludes_zero": False}
rfo.bootstrap_mae_diff = _fast_stub_bootstrap

N = 10
DENSITIES = {"rho_0.25": [2, 3], "rho_0.5": [5], "rho_0.75": [7, 8]}
POOL_COUNTS = {
    "fit": {"rho_0.25": 22, "rho_0.5": 34, "rho_0.75": 22},
    "calibration": {"rho_0.25": 6, "rho_0.5": 8, "rho_0.75": 6},
    "held_out": {"rho_0.25": 6, "rho_0.5": 8, "rho_0.75": 6},
}
N_SPLITS = 100
SPLIT_SEED = 900001


def density_of(vec):
    k = sum(vec)
    for name, ks in DENSITIES.items():
        if k in ks:
            return name
    raise ValueError(f"unexpected density for vec with k={k}: {vec}")


def main():
    resp = json.loads((E9 / "dnabert2_mask_responses.json").read_text())
    pools_orig = resp["responses"]["pools"]
    baseline = resp["baseline_mlm_loss"]
    baseline_per_batch = resp["baseline_per_batch"]
    epsilons = resp["epsilons"]

    # pool ALL non-singleton measured records by density
    all_records = pools_orig["fit"] + pools_orig["calibration"] + pools_orig["held_out"]
    by_density = {d: [] for d in DENSITIES}
    for r in all_records:
        by_density[density_of(r["a"])].append(r)

    print("Feasibility check:")
    total_needed = 0
    for d in DENSITIES:
        needed = sum(POOL_COUNTS[p][d] for p in POOL_COUNTS)
        have = len(by_density[d])
        total_needed += needed
        print(f"  {d}: measured={have}  needed_per_full_partition={needed}  "
              f"{'EXACT MATCH -- no slack for new subsets' if have == needed else 'SLACK AVAILABLE'}")
    print(f"  TOTAL non-singleton measured: {len(all_records)}  needed: {total_needed}")
    print("  -> Repeated-split analysis proceeds as a role-permutation resampling of the "
          "existing 118 measured conditions (no new model runs). Falling back to "
          "leave-one-condition-out jackknife was NOT needed.\n")

    rng = np.random.default_rng(SPLIT_SEED)
    results = {str(eps): {"F0_r2": [], "F1_r2": [], "F2_r2": [], "F3_r2": [],
                           "F2_mae": [], "F3_mae": [], "rel_improve": [],
                           "f3_beats_f2": [], "lam2": [], "lam3": []}
               for eps in epsilons}

    for split_i in range(N_SPLITS):
        fit_recs, cal_recs, held_recs = [], [], []
        for d in DENSITIES:
            pool = list(by_density[d])
            rng.shuffle(pool)
            nf, nc, nh = (POOL_COUNTS["fit"][d], POOL_COUNTS["calibration"][d],
                          POOL_COUNTS["held_out"][d])
            fit_recs += pool[:nf]
            cal_recs += pool[nf:nf + nc]
            held_recs += pool[nf + nc:nf + nc + nh]
        assert len(fit_recs) == 78 and len(cal_recs) == 20 and len(held_recs) == 20

        pools_resampled = {"singletons": pools_orig["singletons"],
                            "fit": fit_recs, "calibration": cal_recs, "held_out": held_recs}

        for eps in epsilons:
            r = rfo.fit_epsilon(pools_resampled, baseline, baseline_per_batch, eps)
            key = str(eps)
            results[key]["F0_r2"].append(r["F0"]["metrics"]["r2"])
            results[key]["F1_r2"].append(r["F1"]["metrics"]["r2"])
            results[key]["F2_r2"].append(r["F2"]["metrics"]["r2"])
            results[key]["F3_r2"].append(r["F3"]["metrics"]["r2"])
            results[key]["F2_mae"].append(r["F2"]["metrics"]["mae"])
            results[key]["F3_mae"].append(r["F3"]["metrics"]["mae"])
            results[key]["rel_improve"].append(r["F2_vs_F3_relative_MAE_improvement"])
            results[key]["f3_beats_f2"].append(r["F3"]["metrics"]["mae"] < r["F2"]["metrics"]["mae"])
            results[key]["lam2"].append(r["F2"]["lambda"])
            results[key]["lam3"].append(r["F3"]["lambda"])

    # ---- report + CSV ----
    import csv
    rows = []
    for eps in epsilons:
        key = str(eps)
        d = results[key]
        print(f"=== epsilon={eps}, n_splits={N_SPLITS} ===")
        for fam in ("F0", "F1", "F2", "F3"):
            vals = np.array(d[f"{fam}_r2"])
            med, lo, hi = np.median(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)
            print(f"  held-out R2 {fam}: median={med:.4f}  [{lo:.4f}, {hi:.4f}]")
            rows.append({"epsilon": eps, "metric": f"{fam}_held_out_r2", "median": med,
                         "p2.5": lo, "p97.5": hi, "n_splits": N_SPLITS})
        frac_f3_wins = np.mean(d["f3_beats_f2"])
        ri = np.array(d["rel_improve"])
        print(f"  F2->F3 relative MAE improvement: median={np.median(ri)*100:.1f}%  "
              f"[{np.percentile(ri,2.5)*100:.1f}%, {np.percentile(ri,97.5)*100:.1f}%]")
        print(f"  fraction of splits where F3 beats F2 on held-out MAE: {frac_f3_wins*100:.1f}%")
        rows.append({"epsilon": eps, "metric": "F2_to_F3_relative_MAE_improvement",
                     "median": np.median(ri), "p2.5": np.percentile(ri, 2.5),
                     "p97.5": np.percentile(ri, 97.5), "n_splits": N_SPLITS})
        rows.append({"epsilon": eps, "metric": "fraction_F3_beats_F2_on_held_out_MAE",
                     "median": frac_f3_wins, "p2.5": "", "p97.5": "", "n_splits": N_SPLITS})

        lam2_counts = {v: d["lam2"].count(v) for v in sorted(set(d["lam2"]))}
        lam3_counts = {v: d["lam3"].count(v) for v in sorted(set(d["lam3"]))}
        print(f"  F2 selected-lambda distribution across splits: {lam2_counts}")
        print(f"  F3 selected-lambda distribution across splits: {lam3_counts}")
        rows.append({"epsilon": eps, "metric": "F2_lambda_distribution",
                     "median": json.dumps(lam2_counts), "p2.5": "", "p97.5": "", "n_splits": N_SPLITS})
        rows.append({"epsilon": eps, "metric": "F3_lambda_distribution",
                     "median": json.dumps(lam3_counts), "p2.5": "", "p97.5": "", "n_splits": N_SPLITS})
        print()

    out_csv = OUT / "tomography_split_stability.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epsilon", "metric", "median", "p2.5", "p97.5", "n_splits"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
