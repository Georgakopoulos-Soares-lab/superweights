#!/usr/bin/env python
"""Section 3 audit recomputation: layer-relative Frobenius magnitude vs causal effect."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "results/experiments/E13/part2_22_model_results.csv"


def load():
    return list(csv.DictReader(open(CENSUS)))


def spearman_boot(x, y, seed, n_boot=5000):
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
    return rho, p, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    rows = load()
    for eps_suffix, eps_label in (("eps0p5", "0.5"), ("eps1p0", "1.0")):
        r_key = f"candidate_relative_loss_change_{eps_suffix}"
        q1 = [float(r["q1"]) for r in rows]
        frob = [float(r["frob_ratio_to_layer_median"]) for r in rows]
        r_vals = [float(r[r_key]) for r in rows]
        print(f"\n=== epsilon={eps_label}, n={len(rows)} ===")
        rho, p, lo, hi = spearman_boot(frob, r_vals, seed=44)
        print(f"  Frobenius-ratio vs R:  rho={rho:.6f} p={p:.6f}  bootstrap CI(seed44)=[{lo:.6f},{hi:.6f}]")
        rho, p, lo, hi = spearman_boot(q1, r_vals, seed=43)
        print(f"  q1 vs R:               rho={rho:.6f} p={p:.6f}  bootstrap CI(seed43)=[{lo:.6f},{hi:.6f}]")

        # leave-one-model-out for Frobenius-ratio vs R
        loo_rhos = []
        for i in range(len(rows)):
            xf = frob[:i] + frob[i+1:]
            yf = r_vals[:i] + r_vals[i+1:]
            rr, _ = spearmanr(xf, yf)
            loo_rhos.append((rows[i]["model"], rr))
        vals = [v for _, v in loo_rhos]
        print(f"  LOO rho (Frobenius): min={min(vals):.6f} ({[m for m,v in loo_rhos if v==min(vals)][0]}) "
              f"max={max(vals):.6f} ({[m for m,v in loo_rhos if v==max(vals)][0]})")

        # text decoders only
        td_idx = [i for i, r in enumerate(rows) if r["architecture"] == "decoder" and r["domain"] == "text"]
        if len(td_idx) >= 3:
            xf_td = [frob[i] for i in td_idx]
            xq_td = [q1[i] for i in td_idx]
            y_td = [r_vals[i] for i in td_idx]
            rho_f, p_f = spearmanr(xf_td, y_td)
            rho_q, p_q = spearmanr(xq_td, y_td)
            print(f"  text-decoders-only (n={len(td_idx)}): Frobenius rho={rho_f:.6f} p={p_f:.6f}; "
                  f"q1 rho={rho_q:.6f} p={p_q:.6f}")

        # q1 binary split at 0.95
        hi_idx = [i for i, v in enumerate(q1) if v >= 0.95]
        lo_idx = [i for i, v in enumerate(q1) if v < 0.95]
        hi_vals = [r_vals[i] for i in hi_idx]
        lo_vals = [r_vals[i] for i in lo_idx]
        print(f"  q1>=0.95 split: n_high={len(hi_idx)} median_R={np.median(hi_vals)*100:.4f}%  "
              f"n_low={len(lo_idx)} median_R={np.median(lo_vals)*100:.4f}%")

    # q1 distribution (22-model causal panel, not the 23-model structural panel)
    q1_all = [float(r["q1"]) for r in rows]
    print(f"\nq1 distribution across the 22-model CAUSAL panel:")
    print(f"  >0.95: {sum(1 for v in q1_all if v > 0.95)}/22")
    print(f"  >0.97: {sum(1 for v in q1_all if v > 0.97)}/22")
    print(f"  <0.8:  {sum(1 for v in q1_all if v < 0.8)}/22")

    # coverage of layer_relative_frobenius
    n_have = sum(1 for r in rows if r["frob_ratio_to_layer_median"] not in (None, "", "nan"))
    print(f"\nlayer_relative_frobenius (frob_ratio_to_layer_median) coverage: {n_have}/22")


if __name__ == "__main__":
    main()
