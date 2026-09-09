#!/usr/bin/env python
"""Section 1 audit recomputation: DNABERT-2 tomography split/design/coefficient checks.

Reads only stored artifacts under manuscript/experiments/E9_mechanistic_tomography/.
Writes audit/tomography_splits.csv and audit/tomography_pair_coeffs.csv.
Deterministic, no GPU, no re-execution of the measurement pipeline.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E9 = ROOT / "manuscript/experiments/E9_mechanistic_tomography"
OUT = ROOT / "audit"

BASIS = [(5, 603), (3, 86), (3, 399), (9, 264), (9, 294),
         (3, 603), (3, 641), (7, 603), (6, 603), (5, 86)]
N = 10
PAIRS = list(itertools.combinations(range(N), 2))

DENSITY_OF_K = {}
for name, ks in {"rho_0.25": [2, 3], "rho_0.5": [5], "rho_0.75": [7, 8]}.items():
    for k in ks:
        DENSITY_OF_K[k] = name


def rows_ablated_str(vec):
    return ";".join(f"L{BASIS[i][0]}r{BASIS[i][1]}" for i, x in enumerate(vec) if x)


def main():
    masks = json.loads((E9 / "masks_dnabert2.json").read_text())
    responses = json.loads((E9 / "dnabert2_mask_responses.json").read_text())
    fit_results = json.loads((E9 / "fit_results_dnabert2.json").read_text())

    pools = masks["pools"]
    resp_pools = responses["responses"]["pools"]

    # --- tomography_splits.csv ---
    rows_out = []
    for split in ("singletons", "fit", "calibration", "held_out"):
        vecs = pools[split]
        n_masks = len(vecs)
        for idx, vec in enumerate(vecs):
            k = sum(vec)
            density = DENSITY_OF_K.get(k, f"k={k}")
            for eps in (0.5, 1.0):
                rows_out.append({
                    "split": split,
                    "condition_id": f"{split}_{idx:03d}",
                    "rows_ablated": rows_ablated_str(vec),
                    "epsilon": eps,
                    "mask_pool_id": density,
                    "n_masks": n_masks,
                })
    with open(OUT / "tomography_splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "condition_id", "rows_ablated",
                                          "epsilon", "mask_pool_id", "n_masks"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {OUT/'tomography_splits.csv'} ({len(rows_out)} rows)")

    # --- overlap check (recomputed) ---
    def subsets(vecs):
        return [frozenset(i for i, x in enumerate(v) if x) for v in vecs]
    fit_s, cal_s, held_s, single_s = (subsets(pools[p]) for p in
                                       ("fit", "calibration", "held_out", "singletons"))
    overlap = {
        "fit_and_held_out": sorted(set(fit_s) & set(held_s)),
        "calibration_and_held_out": sorted(set(cal_s) & set(held_s)),
        "fit_and_calibration": sorted(set(fit_s) & set(cal_s)),
        "singletons_and_held_out": sorted(set(single_s) & set(held_s)),
    }
    print("RECOMPUTED overlap (row-subset intersection) across splits:")
    for k, v in overlap.items():
        print(f"  {k}: {len(v)} shared subsets -> {v}")

    # --- RECOMPUTED rank/cond of stored fit design (additive + lifted) ---
    fit_arr = np.array(pools["fit"], dtype=float)
    def lifted(a):
        inter = np.array([a[:, i] * a[:, j] for i, j in PAIRS]).T
        return np.hstack([a, inter])
    X_add, X_lift = fit_arr, lifted(fit_arr)
    print(f"RECOMPUTED fit additive design {X_add.shape}: rank={np.linalg.matrix_rank(X_add)}, "
          f"cond(X)={np.linalg.cond(X_add):.4f}, cond(XtX)={np.linalg.cond(X_add.T@X_add):.4f}")
    print(f"RECOMPUTED fit lifted design {X_lift.shape}: rank={np.linalg.matrix_rank(X_lift)}, "
          f"cond(X)={np.linalg.cond(X_lift):.4f}, cond(XtX)={np.linalg.cond(X_lift.T@X_lift):.4f}")

    # --- tomography_pair_coeffs.csv: full sorted |gamma| table, both epsilons ---
    coeff_rows = []
    for eps_key in ("0.5", "1.0"):
        d = fit_results["by_epsilon"][eps_key]["F3"]
        gamma_pairs = d["gamma_pairs"]  # dict "(l1,r1)x(l2,r2)" -> gamma
        items = sorted(gamma_pairs.items(), key=lambda kv: -abs(kv[1]))
        for rank, (pair_label, gamma) in enumerate(items, start=1):
            coeff_rows.append({
                "epsilon": eps_key,
                "rank_by_abs_gamma": rank,
                "pair": pair_label,
                "gamma": gamma,
                "abs_gamma": abs(gamma),
            })
    with open(OUT / "tomography_pair_coeffs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epsilon", "rank_by_abs_gamma", "pair", "gamma", "abs_gamma"])
        w.writeheader()
        w.writerows(coeff_rows)
    print(f"wrote {OUT/'tomography_pair_coeffs.csv'} ({len(coeff_rows)} rows)")

    # sanity: confirm L9r264 x L9r294 ranks 1st at both epsilons
    for eps_key in ("0.5", "1.0"):
        top = [r for r in coeff_rows if r["epsilon"] == eps_key and r["rank_by_abs_gamma"] == 1][0]
        print(f"epsilon={eps_key}: top pair by |gamma| = {top['pair']} (gamma={top['gamma']:.6f})")


if __name__ == "__main__":
    main()
