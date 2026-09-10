#!/usr/bin/env python
"""Round-2 Section 5: persist round-1's Frobenius/q1 structure-function re-analysis
(audit/scripts/section3_frobenius.py) as a reproducible, panel-generalized artifact.

Emits audit/rederivations/structure_function_correlations.csv: for every combination of
{predictor: q1, layer_relative_frobenius} x {epsilon: 0.5, 1.0} x
{panel: all-22, text-decoders-only, text-encoders-only, genomic-only}:
rho, p, ci_low, ci_high, n, seed, loo_rho_min, loo_rho_max, underpowered (n<8).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
CENSUS = ROOT / "results/experiments/E13/part2_22_model_results.csv"
OUT = ROOT / "audit/rederivations/structure_function_correlations.csv"

PANELS = {
    "all-22": lambda r: True,
    "text-decoders-only": lambda r: r["domain"] == "text" and r["architecture"] == "decoder",
    "text-encoders-only": lambda r: r["domain"] == "text" and r["architecture"] == "encoder",
    "genomic-only": lambda r: r["domain"] == "genomic",
}
PREDICTOR_COL = {"q1": "q1", "layer_relative_frobenius": "frob_ratio_to_layer_median"}
PREDICTOR_SEED = {"q1": 43, "layer_relative_frobenius": 44}
N_BOOT = 5000


def spearman_boot(x, y, seed, n_boot=N_BOOT):
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    xa, ya = np.array(x), np.array(y)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rr = spearmanr(xa[idx], ya[idx])[0]
        if np.isfinite(rr):
            boot.append(rr)
    if not boot:
        return rho, p, float("nan"), float("nan")
    return rho, p, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def loo_range(x, y):
    if len(x) < 4:
        return None, None
    rhos = []
    for i in range(len(x)):
        xf = x[:i] + x[i + 1:]
        yf = y[:i] + y[i + 1:]
        rr, _ = spearmanr(xf, yf)
        if np.isfinite(rr):
            rhos.append(rr)
    if not rhos:
        return None, None
    return min(rhos), max(rhos)


def main():
    rows_all = list(csv.DictReader(open(CENSUS)))
    out_rows = []
    for panel_name, panel_filter in PANELS.items():
        panel_rows = [r for r in rows_all if panel_filter(r)]
        n = len(panel_rows)
        for eps_suffix, eps_label in (("eps0p5", "0.5"), ("eps1p0", "1.0")):
            r_key = f"candidate_relative_loss_change_{eps_suffix}"
            y = [float(r[r_key]) for r in panel_rows]
            for pred_name, col in PREDICTOR_COL.items():
                x = [float(r[col]) for r in panel_rows]
                seed = PREDICTOR_SEED[pred_name]
                if n < 3:
                    rho = p = lo = hi = loo_lo = loo_hi = float("nan")
                else:
                    rho, p, lo, hi = spearman_boot(x, y, seed)
                    loo_lo, loo_hi = loo_range(x, y)
                out_rows.append(dict(
                    predictor=pred_name, epsilon=eps_label, panel=panel_name, n=n,
                    rho=rho, p=p, ci_low=lo, ci_high=hi, seed=seed,
                    loo_rho_min=loo_lo if loo_lo is not None else "",
                    loo_rho_max=loo_hi if loo_hi is not None else "",
                    underpowered_n_lt_8=(n < 8),
                ))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT} ({len(out_rows)} rows)")

    for r in out_rows:
        flag = " [UNDERPOWERED n<8]" if r["underpowered_n_lt_8"] else ""
        loo = f"  LOO=[{r['loo_rho_min']:.4f},{r['loo_rho_max']:.4f}]" if r["loo_rho_min"] != "" else ""
        print(f"  {r['predictor']:25s} eps={r['epsilon']:4s} {r['panel']:20s} n={r['n']:3d}  "
              f"rho={r['rho']:.4f} p={r['p']:.4f}  CI=[{r['ci_low']:.4f},{r['ci_high']:.4f}]{loo}{flag}")


if __name__ == "__main__":
    main()
