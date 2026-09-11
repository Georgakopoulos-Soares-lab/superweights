#!/usr/bin/env python
"""
E10b Phase 5-7 -- Phi-3 observer ladder F0-F3, held-out evaluation, bootstrap, and the
mechanical PAIR_TERMS_REQUIRED / ADDITIVE_SUFFICIENT / MIXED_OR_UNRESOLVED decision.

E9/E10's estimator, metric functions, bootstrap and adequacy thresholds are reused by DIRECT
IMPORT from E10's fit_encoder_observers.py (which itself imports from E9's
run_fit_observers.py) -- re-parameterized for N=6 pairs (15) instead of N=10 (45). Nothing is
re-derived. Governing lock: PREREG_E10b_phi3_tomography.md (sha256 098fd52c...).

Pure numpy -- no GPU.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E9 = HERE.parent / "E9_mechanistic_tomography"
sys.path.insert(0, str(E9))

from run_fit_observers import LAMBDA_GRID, ridge_fit, metrics, bootstrap_mae_diff  # noqa: E402

N = 6
PAIRS = list(itertools.combinations(range(N), 2))
R2_ADEQUATE = 0.90
NORM_MAE_ADEQUATE = 0.10
REL_IMPROVE_THRESHOLD = 0.10


def lifted(a: np.ndarray) -> np.ndarray:
    inter = np.array([a[:, i] * a[:, j] for i, j in PAIRS]).T
    return np.hstack([a, inter])


def fit_epsilon(pools, baseline_per_batch, eps):
    key, pbkey = f"dloss_eps{eps}", f"per_batch_eps{eps}"

    x_singleton = np.zeros(N)
    for row in pools["singletons"]:
        x_singleton[row["a"].index(1)] = row[key]

    av = lambda p: np.array([r["a"] for r in pools[p]], dtype=float)   # noqa: E731
    yv = lambda p: np.array([r[key] for r in pools[p]], dtype=float)   # noqa: E731
    a_fit, y_fit = av("fit"), yv("fit")
    a_cal, y_cal = av("calibration"), yv("calibration")
    a_held, y_held = av("held_out"), yv("held_out")
    n_batches = len(pools["held_out"][0][pbkey])

    yhat0_cal, yhat0_held = a_cal @ x_singleton, a_held @ x_singleton
    m0 = metrics(y_held, yhat0_held, a_held)

    g = (float(np.dot(y_cal, yhat0_cal) / np.dot(yhat0_cal, yhat0_cal))
         if np.dot(yhat0_cal, yhat0_cal) > 0 else 0.0)
    m1 = metrics(y_held, g * yhat0_held, a_held)

    best_lam2 = min(LAMBDA_GRID,
                    key=lambda lam: np.mean((y_cal - a_cal @ ridge_fit(a_fit, y_fit, lam)) ** 2))
    beta2 = ridge_fit(a_fit, y_fit, best_lam2)
    yhat2_held = a_held @ beta2
    m2 = metrics(y_held, yhat2_held, a_held)

    Xf, Xc, Xh = lifted(a_fit), lifted(a_cal), lifted(a_held)
    best_lam3 = min(LAMBDA_GRID,
                    key=lambda lam: np.mean((y_cal - Xc @ ridge_fit(Xf, y_fit, lam)) ** 2))
    coef3 = ridge_fit(Xf, y_fit, best_lam3)
    yhat3_held = Xh @ coef3
    m3 = metrics(y_held, yhat3_held, a_held)

    per_batch_held = [r[pbkey] for r in pools["held_out"]]
    boot = bootstrap_mae_diff(baseline_per_batch, per_batch_held,
                              yhat2_held, yhat3_held, n_batches)
    rel_improve = (m2["mae"] - m3["mae"]) / m2["mae"] if m2["mae"] > 0 else float("nan")
    abs_improve = m2["mae"] - m3["mae"]

    additive_adequate = (m2["r2"] >= R2_ADEQUATE) and (m2["normalized_mae"] <= NORM_MAE_ADEQUATE)
    pair_required = (rel_improve >= REL_IMPROVE_THRESHOLD) and boot["excludes_zero"]
    if pair_required:
        decision = "PAIR_TERMS_REQUIRED"
    elif additive_adequate:
        decision = "ADDITIVE_SUFFICIENT"
    else:
        decision = "MIXED_OR_UNRESOLVED"

    return {
        "epsilon": eps,
        "singleton_effects_x_i": x_singleton.tolist(),
        "F0": {"metrics": m0},
        "F1": {"gain_g": g, "metrics": m1},
        "F2": {"lambda": best_lam2, "beta": beta2.tolist(), "metrics": m2},
        "F3": {"lambda": best_lam3, "beta_main": coef3[:N].tolist(),
               "gamma": coef3[N:].tolist(), "metrics": m3},
        "F2_vs_F3_absolute_MAE_improvement": abs_improve,
        "F2_vs_F3_relative_MAE_improvement": rel_improve,
        "F2_vs_F3_bootstrap": boot,
        "mechanical_decision": decision,
    }


def retrospective_pairs(basis, gamma, top_k=15):
    gamma = np.asarray(gamma)
    order = np.argsort(-np.abs(gamma))
    top = []
    for k in order[:top_k]:
        i, j = PAIRS[k]
        li, ri = basis[i]
        lj, rj = basis[j]
        top.append({"pair_basis_idx": [int(i), int(j)],
                    "structural_ranks": [i + 1, j + 1],
                    "rows": [f"L{li}/r{ri}", f"L{lj}/r{rj}"],
                    "gamma": float(gamma[k]),
                    "same_layer": bool(li == lj),
                    "involves_top2_singleton_rows": bool({i, j} == {0, 1})})
    top1_top2_pair_idx = PAIRS.index((0, 1))
    top1_top2_rank = int(np.where(order == top1_top2_pair_idx)[0][0]) + 1
    all_same_layer = np.mean([basis[i][0] == basis[j][0] for i, j in PAIRS])
    top_same_layer = np.mean([t["same_layer"] for t in top])
    return {
        "all_pairs_by_abs_gamma": top,
        "top1_top2_singleton_pair_rank_by_abs_gamma": top1_top2_rank,
        "n_pairs": len(PAIRS),
        "frac_same_layer_top5": float(np.mean([t["same_layer"] for t in top[:5]])),
        "frac_same_layer_all_pairs": float(all_same_layer),
    }


def main():
    resp = json.loads((ROOT / "results" / "e10b_phi3_tomography_responses.json").read_text())
    basis = [tuple(b) for b in resp["basis"]]
    pools = resp["responses"]["pools"]

    out = {"prereg": "PREREG_E10b_phi3_tomography.md", "model": "phi3",
           "basis": [list(b) for b in basis], "baseline_nll": resp["baseline_nll"],
           "thresholds": {"r2_adequate": R2_ADEQUATE,
                          "normalized_mae_adequate": NORM_MAE_ADEQUATE,
                          "F2_to_F3_relative_MAE_improvement": REL_IMPROVE_THRESHOLD},
           "by_epsilon": {}}

    print(f"{'='*66}\nphi3 tomography  (baseline NLL {resp['baseline_nll']:.4f})\n{'='*66}")
    for eps in resp["epsilons"]:
        fr = fit_epsilon(pools, resp["baseline_per_batch"], eps)
        fr["retrospective_pair_structure"] = retrospective_pairs(basis, fr["F3"]["gamma"])
        out["by_epsilon"][str(eps)] = fr
        print(f"\nepsilon={eps}")
        for fam in ("F0", "F1", "F2", "F3"):
            mm = fr[fam]["metrics"]
            print(f"  {fam}: R2={mm['r2']:>8.4f} MAE={mm['mae']:.6f} normMAE={mm['normalized_mae']:.4f}")
        b = fr["F2_vs_F3_bootstrap"]
        print(f"  F2->F3: abs improvement={fr['F2_vs_F3_absolute_MAE_improvement']:+.6f}  "
              f"rel={fr['F2_vs_F3_relative_MAE_improvement']*100:+.1f}%  "
              f"CI=({b['ci_2.5']:+.6f},{b['ci_97.5']:+.6f}) excludes_zero={b['excludes_zero']}")
        print(f"  DECISION: {fr['mechanical_decision']}")
        rp = fr["retrospective_pair_structure"]
        print(f"  retrospective: (rank1,rank2) pair rank by |Gamma| = "
              f"{rp['top1_top2_singleton_pair_rank_by_abs_gamma']}/{rp['n_pairs']}")

    p = ROOT / "results" / "e10b_phi3_fit_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
