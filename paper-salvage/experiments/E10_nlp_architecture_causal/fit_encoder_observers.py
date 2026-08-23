#!/usr/bin/env python
"""
E10 ARM B -- encoder observer ladder F0-F3, held-out evaluation, bootstrap, and the
mechanical pair-required decision.

E9's estimator, metric functions, bootstrap and adequacy thresholds are reused by DIRECT
IMPORT from run_fit_observers.py (not copied, not re-derived) -- the v2 prereg specifies
"Encoder (unchanged from E9's own adequacy rule, Phase 6)".

  F0: y = sum_i a_i x_i           (x from measured singletons)
  F1: y = g * sum_i a_i x_i       (g fit on calibration only)
  F2: y = sum_i beta_i a_i        (ridge, lambda on calibration)
  F3: y = sum_i beta_i a_i + sum_{i<j} Gamma_ij a_i a_j   (ridge, lambda on calibration)

  PAIR_TERMS_REQUIRED iff F3 improves held-out MAE by >= 10% AND the paired bootstrap CI
  excludes zero (E9's rule).

The retrospective pair-structure inspection runs only AFTER the primary decision is
computed, and never feeds back into the basis or the fit. Per e10_prompt.md: there is no
independently known critical pair for either text encoder, and none is manufactured.

Pure numpy -- no GPU.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E9 = HERE.parent / "E9_mechanistic_tomography"
ROOT = HERE.parents[3] if (HERE.parents[3] / "results").exists() else HERE.parents[2].parent
RES = ROOT / "results"
sys.path.insert(0, str(E9))

# E9's machinery, imported unmodified.
from run_fit_observers import (  # noqa: E402
    N, PAIRS, LAMBDA_GRID, lifted, ridge_fit, metrics, bootstrap_mae_diff,
)

MODELS = ["mosaicbert", "modernbert"]
R2_ADEQUATE = 0.90          # E9 thresholds, unchanged
NORM_MAE_ADEQUATE = 0.10
REL_IMPROVE_THRESHOLD = 0.10


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

    additive_adequate = (m2["r2"] >= R2_ADEQUATE) and (m2["normalized_mae"] <= NORM_MAE_ADEQUATE)
    f1_adequate = (m1["r2"] >= R2_ADEQUATE) and (m1["normalized_mae"] <= NORM_MAE_ADEQUATE)
    pair_improves = (rel_improve >= REL_IMPROVE_THRESHOLD) and boot["excludes_zero"]
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
        "F3": {"lambda": best_lam3, "beta_main": coef3[:N].tolist(),
               "gamma": coef3[N:].tolist(), "metrics": m3},
        "F2_vs_F3_relative_MAE_improvement": rel_improve,
        "F2_vs_F3_bootstrap": boot,
        "mechanical_decision": decision,
    }


def retrospective_pairs(basis, gamma, canonical_idx, top_k=10):
    """Runs only after the primary decision is frozen. Descriptive; never refits."""
    gamma = np.asarray(gamma)
    order = np.argsort(-np.abs(gamma))
    top = []
    for k in order[:top_k]:
        i, j = PAIRS[k]
        li, ri = basis[i]
        lj, rj = basis[j]
        top.append({"pair_basis_idx": [int(i), int(j)],
                    "rows": [f"L{li}/r{ri}", f"L{lj}/r{rj}"],
                    "gamma": float(gamma[k]),
                    "same_layer": bool(li == lj),
                    "same_output_row": bool(ri == rj),
                    "involves_canonical": bool(canonical_idx in (i, j)),
                    "mean_structural_rank": float((i + j) / 2 + 1)})
    all_same_row = np.mean([basis[i][1] == basis[j][1] for i, j in PAIRS])
    top_same_row = np.mean([t["same_output_row"] for t in top])
    all_same_layer = np.mean([basis[i][0] == basis[j][0] for i, j in PAIRS])
    top_same_layer = np.mean([t["same_layer"] for t in top])
    canon_ranks = [int(np.where(order == k)[0][0]) + 1
                   for k, (i, j) in enumerate(PAIRS) if canonical_idx in (i, j)]
    return {
        "top_pairs_by_abs_gamma": top,
        "frac_same_output_row_top": float(top_same_row),
        "frac_same_output_row_all_pairs": float(all_same_row),
        "frac_same_layer_top": float(top_same_layer),
        "frac_same_layer_all_pairs": float(all_same_layer),
        "mean_structural_rank_top": float(np.mean([t["mean_structural_rank"] for t in top])),
        "mean_structural_rank_all_pairs": float(np.mean([(i + j) / 2 + 1 for i, j in PAIRS])),
        "canonical_row_pair_ranks_by_abs_gamma": sorted(canon_ranks),
        "best_canonical_pair_rank": int(min(canon_ranks)) if canon_ranks else None,
        "n_pairs": len(PAIRS),
    }


def main():
    out = {"prereg": "PREREG_E10_nlp_architecture_causal_v2.md",
           "thresholds": {"r2_adequate": R2_ADEQUATE,
                          "normalized_mae_adequate": NORM_MAE_ADEQUATE,
                          "F2_to_F3_relative_MAE_improvement": REL_IMPROVE_THRESHOLD},
           "models": {}}
    for m in MODELS:
        f = RES / f"e10_encoder_responses_{m}.json"
        if not f.exists():
            print(f"[skip] {m}: no responses yet")
            continue
        resp = json.loads(f.read_text())
        basis = [tuple(b) for b in resp["basis"]]
        canonical_idx = resp["canonical_row_basis_index"]
        pools = resp["responses"]["pools"]
        r = {"model": m, "basis": [list(b) for b in basis],
             "canonical_row_basis_index": canonical_idx,
             "baseline_mlm_loss": resp["baseline_mlm_loss"], "by_epsilon": {}}

        print(f"\n{'='*66}\n{m}  (baseline MLM loss {resp['baseline_mlm_loss']:.4f})\n{'='*66}")
        for eps in resp["epsilons"]:
            fr = fit_epsilon(pools, resp["baseline_per_batch"], eps)
            # retrospective inspection AFTER the decision is computed
            fr["retrospective_pair_structure"] = retrospective_pairs(
                basis, fr["F3"]["gamma"], canonical_idx)
            r["by_epsilon"][str(eps)] = fr
            print(f"  epsilon={eps}")
            for fam in ("F0", "F1", "F2", "F3"):
                mm = fr[fam]["metrics"]
                print(f"    {fam}: R2={mm['r2']:>8.4f} MAE={mm['mae']:.6f} "
                      f"normMAE={mm['normalized_mae']:.4f}")
            b = fr["F2_vs_F3_bootstrap"]
            print(f"    F2->F3 rel MAE improvement: "
                  f"{fr['F2_vs_F3_relative_MAE_improvement']*100:+.1f}%  "
                  f"CI=({b['ci_2.5']:+.6f},{b['ci_97.5']:+.6f}) "
                  f"excludes_zero={b['excludes_zero']}")
            print(f"    DECISION: {fr['mechanical_decision']}")
        out["models"][m] = r

    p = RES / "e10_encoder_fit_results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
