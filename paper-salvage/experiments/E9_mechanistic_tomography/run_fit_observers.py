#!/usr/bin/env python
"""
E9 — DNABERT-2 observer-family fitting (F0-F3), held-out evaluation, bootstrap, and the
Phase 6 mechanical adequacy decision.

Pure numpy/scipy -- no GPU, no torch needed. Reads masks_dnabert2.json (design) and
dnabert2_mask_responses.json (measured responses, from run_dnabert2_measurements.py).

Fits and evaluates each family SEPARATELY per epsilon (0.5 and 1.0), per the prereg. All
thresholds below are copied verbatim from the locked prereg
(docs/prereg/PREREG_mechanistic_tomography_E9.md) -- not re-derived here.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N = 10
LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
N_BOOT = 5000
RNG = np.random.default_rng(20260822)

PAIRS = list(itertools.combinations(range(N), 2))
BASIS = [
    (5, 603), (3, 86), (3, 399), (9, 264), (9, 294),
    (3, 603), (3, 641), (7, 603), (6, 603), (5, 86),
]
CRITICAL_PAIR_IDX = (3, 4)  # (9,264), (9,294) -- indices into BASIS, NOT special-cased
                            # during fitting; only used for the retrospective H5 check.


def lifted(a: np.ndarray) -> np.ndarray:
    inter = np.array([a[:, i] * a[:, j] for i, j in PAIRS]).T
    return np.hstack([a, inter])


def ridge_fit(X, y, lam):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def metrics(y_true, y_pred, a_mat):
    resid = y_true - y_pred
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    span = float(np.max(y_true) - np.min(y_true)) if len(y_true) else float("nan")
    norm_mae = float(mae / span) if span > 0 else float("nan")
    density = a_mat.sum(axis=1)
    resid_vs_density = float(np.corrcoef(resid, density)[0, 1]) if len(resid) > 1 else float("nan")
    resid_vs_mag = float(np.corrcoef(resid, np.abs(y_pred))[0, 1]) if len(resid) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "normalized_mae": norm_mae,
            "signed_residual_mean": float(np.mean(resid)),
            "residual_vs_density_corr": resid_vs_density,
            "residual_vs_predicted_magnitude_corr": resid_vs_mag}


def bootstrap_mae_diff(baseline_per_batch, per_batch_held_eps, y_pred_f2, y_pred_f3, n_batches):
    """Resample held-out batches (and the baseline, with the SAME draw, so the
    resample stays paired) with replacement; recompute each held-out mask's dloss
    under the resample; recompute MAE_F2 - MAE_F3.

    y_pred_f2/f3 are on the dloss scale (fit against dloss targets), so the resampled
    raw aggregate loss must have the correspondingly-resampled baseline subtracted
    before comparison -- omitting this silently compares dloss (~0.01-0.5) against raw
    loss (~4.7) and produces a meaningless, near-constant result that looks like a
    tight CI for the wrong reason. Caught by comparing against the unresampled point
    estimate before trusting this function.
    """
    diffs = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n_batches, size=n_batches)
        base_s = sum(baseline_per_batch[i][0] for i in idx)
        base_n = sum(baseline_per_batch[i][1] for i in idx)
        base_boot = base_s / max(base_n, 1)
        y_boot = []
        for pb in per_batch_held_eps:
            s = sum(pb[i][0] for i in idx)
            n = sum(pb[i][1] for i in idx)
            y_boot.append(s / max(n, 1) - base_boot)
        y_boot = np.array(y_boot)
        mae2 = np.mean(np.abs(y_boot - y_pred_f2))
        mae3 = np.mean(np.abs(y_boot - y_pred_f3))
        diffs.append(mae2 - mae3)
    diffs = np.array(diffs)
    return {
        "mean_MAE_F2_minus_F3": float(np.mean(diffs)),
        "ci_2.5": float(np.percentile(diffs, 2.5)),
        "ci_97.5": float(np.percentile(diffs, 97.5)),
        "excludes_zero": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0),
    }


def fit_epsilon(pools, baseline, baseline_per_batch, eps):
    key = f"dloss_eps{eps}"
    pbkey = f"per_batch_eps{eps}"

    singles = pools["singletons"]
    x_singleton = np.zeros(N)
    for row in singles:
        i = row["a"].index(1)
        x_singleton[i] = row[key]

    def av(pool_name):
        return np.array([r["a"] for r in pools[pool_name]], dtype=float)

    def yv(pool_name):
        return np.array([r[key] for r in pools[pool_name]], dtype=float)

    a_fit, y_fit = av("fit"), yv("fit")
    a_cal, y_cal = av("calibration"), yv("calibration")
    a_held, y_held = av("held_out"), yv("held_out")
    n_batches = len(pools["held_out"][0][pbkey])

    # F0
    yhat0_fit = a_fit @ x_singleton
    yhat0_cal = a_cal @ x_singleton
    yhat0_held = a_held @ x_singleton
    m0 = metrics(y_held, yhat0_held, a_held)

    # F1: g fit on calibration only, no intercept
    g = float(np.dot(y_cal, yhat0_cal) / np.dot(yhat0_cal, yhat0_cal)) if np.dot(yhat0_cal, yhat0_cal) > 0 else 0.0
    yhat1_held = g * yhat0_held
    m1 = metrics(y_held, yhat1_held, a_held)

    # F2: ridge, additive, lambda chosen on calibration
    best_lam2, best_mse2 = None, float("inf")
    for lam in LAMBDA_GRID:
        beta = ridge_fit(a_fit, y_fit, lam)
        mse = float(np.mean((y_cal - a_cal @ beta) ** 2))
        if mse < best_mse2:
            best_mse2, best_lam2 = mse, lam
    beta2 = ridge_fit(a_fit, y_fit, best_lam2)
    yhat2_held = a_held @ beta2
    m2 = metrics(y_held, yhat2_held, a_held)

    # F3: ridge, lifted, lambda chosen on calibration
    Xf, Xc, Xh = lifted(a_fit), lifted(a_cal), lifted(a_held)
    best_lam3, best_mse3 = None, float("inf")
    for lam in LAMBDA_GRID:
        coef = ridge_fit(Xf, y_fit, lam)
        mse = float(np.mean((y_cal - Xc @ coef) ** 2))
        if mse < best_mse3:
            best_mse3, best_lam3 = mse, lam
    coef3 = ridge_fit(Xf, y_fit, best_lam3)
    yhat3_held = Xh @ coef3
    m3 = metrics(y_held, yhat3_held, a_held)

    per_batch_held = [r[pbkey] for r in pools["held_out"]]
    boot = bootstrap_mae_diff(baseline_per_batch, per_batch_held, yhat2_held, yhat3_held, n_batches)
    rel_improve = (m2["mae"] - m3["mae"]) / m2["mae"] if m2["mae"] > 0 else float("nan")

    beta_main3 = coef3[:N]
    gamma = coef3[N:]
    pair_rank = sorted(range(len(PAIRS)), key=lambda k: -abs(gamma[k]))
    critical_pair_flat_idx = PAIRS.index(CRITICAL_PAIR_IDX)
    critical_pair_rank = pair_rank.index(critical_pair_flat_idx) + 1  # 1-indexed

    additive_adequate = (m2["r2"] >= 0.90) and (m2["normalized_mae"] <= 0.10)
    f1_adequate = (m1["r2"] >= 0.90) and (m1["normalized_mae"] <= 0.10)
    pair_improves = (rel_improve >= 0.10) and boot["excludes_zero"]
    if f1_adequate and not pair_improves:
        decision = "CALIBRATION_SUFFICIENT"
    elif additive_adequate and not pair_improves:
        decision = "ADDITIVE_ADEQUATE"
    elif pair_improves:
        decision = "PAIR_TERMS_REQUIRED"
    else:
        decision = "INTERMEDIATE_NO_CLEAN_DECISION"

    return {
        "epsilon": eps,
        "singleton_effects_x_i": x_singleton.tolist(),
        "F0": {"metrics": m0},
        "F1": {"gain_g": g, "metrics": m1},
        "F2": {"lambda": best_lam2, "beta": beta2.tolist(), "metrics": m2},
        "F3": {"lambda": best_lam3, "beta_main": beta_main3.tolist(),
              "gamma_pairs": {f"{BASIS[i]}x{BASIS[j]}": float(gamma[k])
                             for k, (i, j) in enumerate(PAIRS)},
              "metrics": m3,
              "critical_pair_gamma": float(gamma[critical_pair_flat_idx]),
              "critical_pair_rank_by_abs_gamma": critical_pair_rank,
              "n_pairs": len(PAIRS)},
        "F2_vs_F3_relative_MAE_improvement": rel_improve,
        "F2_vs_F3_bootstrap": boot,
        "mechanical_decision": decision,
    }


def h6_norm_covariation(resp):
    """Phase 7 H6: does layer-9 residual-channel norm drop covary with functional
    damage over held-out interventions? Reported, never called mediation."""
    channels = resp["layer9_norm_channels"]
    base_norms = np.array(resp["layer9_norm_baseline"])
    held = resp["responses"]["pools"]["held_out"]
    basis_channels = [r for _l, r in BASIS]
    out = {}
    for eps in resp["epsilons"]:
        norm_drop, damage = [], []
        for row in held:
            a = row["a"]
            norms = np.array(row[f"layer9_channel_norm_eps{eps}"])
            drop = float(np.sum(base_norms - norms))
            norm_drop.append(drop)
            damage.append(abs(row[f"dloss_eps{eps}"]))
        norm_drop, damage = np.array(norm_drop), np.array(damage)
        r = float(np.corrcoef(norm_drop, damage)[0, 1]) if len(norm_drop) > 1 else float("nan")
        out[str(eps)] = {"pearson_r_norm_drop_vs_abs_damage": r,
                         "n_held_out": len(norm_drop)}
    return out


def main():
    resp = json.loads((HERE / "dnabert2_mask_responses.json").read_text())
    pools = resp["responses"]["pools"]
    baseline = resp["baseline_mlm_loss"]
    baseline_per_batch = resp["baseline_per_batch"]

    h6 = h6_norm_covariation(resp)
    print("H6 (layer-9 norm drop vs. functional damage, held-out):")
    for eps, v in h6.items():
        print(f"  epsilon={eps}: r={v['pearson_r_norm_drop_vs_abs_damage']:.4f} "
              f"(n={v['n_held_out']})")

    out = {"baseline_mlm_loss": baseline, "H6_norm_covariation": h6, "by_epsilon": {}}
    for eps in resp["epsilons"]:
        print(f"\n{'='*60}\nepsilon = {eps}\n{'='*60}")
        r = fit_epsilon(pools, baseline, baseline_per_batch, eps)
        out["by_epsilon"][str(eps)] = r
        for fam in ("F0", "F1", "F2", "F3"):
            m = r[fam]["metrics"]
            print(f"  {fam}: R2={m['r2']:.4f} MAE={m['mae']:.6f} "
                  f"normMAE={m['normalized_mae']:.4f}")
        print(f"  F2->F3 relative MAE improvement: "
              f"{r['F2_vs_F3_relative_MAE_improvement']*100:.1f}%  "
              f"bootstrap CI=({r['F2_vs_F3_bootstrap']['ci_2.5']:.6f}, "
              f"{r['F2_vs_F3_bootstrap']['ci_97.5']:.6f}) "
              f"excludes_zero={r['F2_vs_F3_bootstrap']['excludes_zero']}")
        print(f"  critical pair (L9r264,L9r294) rank by |Gamma|: "
              f"{r['F3']['critical_pair_rank_by_abs_gamma']} / {r['F3']['n_pairs']}")
        print(f"  MECHANICAL DECISION: {r['mechanical_decision']}")

    out_path = HERE / "fit_results_dnabert2.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
