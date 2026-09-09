#!/usr/bin/env python
"""
E9 Phase 3 — designed masks for DNABERT-2 tomography (n_D = 10).

GENERator receives no mask design (BASIS_FREEZE.md: 1D dose-response only).

Produces, with a fixed pre-lock RNG seed, four disjoint mask pools over the 10-row basis
(index order fixed by BASIS_FREEZE.md, NOT re-sorted here):
  - singletons  : all 10 one-hot masks (exhaustive; n=10 makes this trivial per Phase 3)
  - fit         : ridge/OLS training rows for F2/F3
  - calibration : F1 scalar-gain fit + F2/F3 hyperparameter (ridge lambda) selection
  - held_out    : final evaluation only -- NEVER touched by any fitting or selection step

Sampled at three densities rho in {0.25, 0.5, 0.75} (k = round(rho*10) = 2 or 3, 5, 7 or 8
active components -- 0.25*10=2.5 rounds to both 2 and 3 across draws for coverage), drawn
without replacement within each (density, pool) cell using a single deterministic
random.Random(SEED) stream consumed in a fixed order so the split is reproducible from the
seed alone.

Also runs the required pre-fitting design checks (Phase 3): rank/condition number/column
correlation of the additive (10-column) design on the fit pool, and of the pair-lifted
(55-column) design on the fit pool, including a check that the held-out pool's co-occurrence
pattern actually differs from the fit pool's (needed to detect main/pair aliasing).
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np

SEED = 20260822  # pre-lock seed, frozen with this file's first commit
N = 10
OUT_DIR = Path(__file__).resolve().parent
DENSITIES = {
    "rho_0.25": [2, 3],
    "rho_0.5": [5],
    "rho_0.75": [7, 8],
}
# (pool, n_masks_per_density_bucket) -- fit gets the most support, held_out and
# calibration get enough to estimate F1's single scalar and to evaluate F0-F3 with
# reasonable power at each density.
POOL_COUNTS = {
    # fit must give the lifted (55-col: 10 main + 45 pair) design full column rank --
    # 40 rows can never reach rank 55 regardless of which masks are chosen; bumped until
    # the design check below passes full rank (see generate_masks.py run log).
    "fit": {"rho_0.25": 22, "rho_0.5": 34, "rho_0.75": 22},
    "calibration": {"rho_0.25": 6, "rho_0.5": 8, "rho_0.75": 6},
    "held_out": {"rho_0.25": 6, "rho_0.5": 8, "rho_0.75": 6},
}


def all_masks_of_size(k: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(N), k))


def sample_disjoint(rng: random.Random, pool_by_k: dict[int, list], k_choices: list[int],
                    n_needed: int, used: set) -> list[tuple[int, ...]]:
    chosen = []
    attempts = 0
    while len(chosen) < n_needed:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError("could not sample enough disjoint masks -- widen the pool")
        k = rng.choice(k_choices)
        candidates = pool_by_k[k]
        idx = rng.randrange(len(candidates))
        m = candidates[idx]
        if m in used:
            continue
        used.add(m)
        chosen.append(m)
    return chosen


def mask_to_vec(m: tuple[int, ...]) -> list[int]:
    v = [0] * N
    for i in m:
        v[i] = 1
    return v


def design_matrix_additive(masks: list[list[int]]) -> np.ndarray:
    return np.array(masks, dtype=float)


def design_matrix_lifted(masks: list[list[int]]) -> np.ndarray:
    rows = []
    pairs = list(itertools.combinations(range(N), 2))
    for a in masks:
        main = list(a)
        inter = [a[i] * a[j] for i, j in pairs]
        rows.append(main + inter)
    return np.array(rows, dtype=float)


def check_design(X: np.ndarray, label: str) -> dict:
    rank = int(np.linalg.matrix_rank(X))
    # condition number of X^T X (what ridge actually regularizes)
    XtX = X.T @ X
    try:
        cond = float(np.linalg.cond(XtX))
    except np.linalg.LinAlgError:
        cond = float("inf")
    # pairwise column correlations (off-diagonal only), flag near-duplicates
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    max_abs_corr = float(np.max(np.abs(off_diag))) if off_diag.size else 0.0
    n_near_dup = int(np.sum(np.abs(off_diag) > 0.98)) // 2
    return {
        "label": label, "n_rows": X.shape[0], "n_cols": X.shape[1],
        "rank": rank, "full_rank": rank == X.shape[1],
        "condition_number_XtX": cond,
        "max_abs_column_correlation": max_abs_corr,
        "n_near_duplicate_column_pairs": n_near_dup,
    }


def main():
    rng = random.Random(SEED)

    pool_by_k = {k: all_masks_of_size(k) for ks in DENSITIES.values() for k in ks}
    for k in pool_by_k:
        rng.shuffle(pool_by_k[k])  # shuffle once per k so later .index draws are order-free

    used: set = set()
    result = {"seed": SEED, "n": N, "densities": DENSITIES, "pools": {}}

    singleton_masks = [tuple([i]) for i in range(N)]
    for m in singleton_masks:
        used.add(m)
    result["pools"]["singletons"] = [mask_to_vec(m) for m in singleton_masks]

    for pool_name in ("fit", "calibration", "held_out"):
        pool_masks = []
        for density_name, k_choices in DENSITIES.items():
            n_needed = POOL_COUNTS[pool_name][density_name]
            chosen = sample_disjoint(rng, pool_by_k, k_choices, n_needed, used)
            pool_masks.extend(chosen)
        result["pools"][pool_name] = [mask_to_vec(m) for m in pool_masks]

    # design checks on the FIT pool (additive + lifted)
    fit_vecs = result["pools"]["fit"]
    checks = []
    checks.append(check_design(design_matrix_additive(fit_vecs), "fit_additive_10col"))
    checks.append(check_design(design_matrix_lifted(fit_vecs), "fit_lifted_55col"))

    # co-occurrence pattern comparison: fit vs held_out (needed to detect aliasing --
    # if held-out never exercises a pair combination fit never saw either, that's fine;
    # if held-out ONLY repeats fit's exact co-occurrence pattern, pair terms are
    # unidentifiable on the evaluation set even if identifiable on the fit set)
    def pair_cooccurrence_set(vecs):
        s = set()
        for v in vecs:
            on = [i for i, x in enumerate(v) if x]
            for i, j in itertools.combinations(on, 2):
                s.add((i, j))
        return s

    fit_pairs = pair_cooccurrence_set(fit_vecs)
    held_pairs = pair_cooccurrence_set(result["pools"]["held_out"])
    all_pairs = set(itertools.combinations(range(N), 2))
    result["cooccurrence_check"] = {
        "n_pairs_total": len(all_pairs),
        "n_pairs_seen_in_fit": len(fit_pairs),
        "n_pairs_seen_in_held_out": len(held_pairs),
        "n_pairs_seen_in_both": len(fit_pairs & held_pairs),
        "n_pairs_in_held_out_not_in_fit": len(held_pairs - fit_pairs),
    }
    result["design_checks"] = checks

    out_path = OUT_DIR / "masks_dnabert2.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved -> {out_path}")
    print(f"pool sizes: singletons={len(singleton_masks)}, "
          f"fit={len(result['pools']['fit'])}, "
          f"calibration={len(result['pools']['calibration'])}, "
          f"held_out={len(result['pools']['held_out'])}")
    for c in checks:
        print(f"  [{c['label']}] rank={c['rank']}/{c['n_cols']} "
              f"full_rank={c['full_rank']} cond(XtX)={c['condition_number_XtX']:.3e} "
              f"max|corr|={c['max_abs_column_correlation']:.3f} "
              f"near_dup_pairs={c['n_near_duplicate_column_pairs']}")
    print(f"  cooccurrence: {result['cooccurrence_check']}")

    if not checks[1]["full_rank"]:
        print("\n*** LIFTED DESIGN IS RANK-DEFICIENT ON THE FIT POOL -- "
              "STOP per Phase 3, redesign masks before measuring responses ***")


if __name__ == "__main__":
    main()
