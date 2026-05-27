"""
scripts/analysis/run_evo1_uk_audit.py
-------------------------------------
Mechanistic ||U_k||_F audit for Evo1 (StripedHyena), mirroring
run_sw_mechanistic.py (GENERator) and run_dnabert2_uk_audit.py (DNABERT-2).

Evo1's ParallelGatedMLP has the same GLU topology used by GENERator/DNABERT-2:

    out = l3( SiLU( l1(x) ) * l2(x) )

So the closed-form rank-1 Frobenius bound applies directly:

    W_gate = l1.weight  : (d_ffn, d_model)
    W_up   = l2.weight  : (d_ffn, d_model)
    W_down = l3.weight  : (d_model, d_ffn)

    ||U_k||_F = sqrt( sum_i  W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2 )

Detection (earlier this session) identified an activation hotspot at layer 11
spanning many rows (3582, 156, 3292, 682, 616, 411, 843, ...). Ablation on
the top-ranked row showed Δppl ≈ 0 % vs random controls. The question this
audit closes: does the U_k predictor *also* fail to concentrate at the row
level on Evo1, completing the "mechanism requires gated FFN + downstream
load-bearing role" story?

Output
------
results/sw_mechanistic_evo1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/sw_mechanistic_evo1.json")
    p.add_argument("--sw_index", default="results/super_weight_index.json")
    # Layers detected as activation hotspots (from earlier sweep). Default to
    # layer 11 (the layer the detection iterator returned).
    p.add_argument("--detected_layer", type=int, default=11)
    args = p.parse_args()

    # Lazy-import torch in the evo env so this can be importable everywhere
    import torch
    from evo import Evo

    print("[uk_audit_evo1] loading Evo1 …")
    evo = Evo("evo-1-8k-base", device="cpu")
    model = evo.model
    n_blocks = len(model.blocks)
    print(f"[uk_audit_evo1] n_blocks = {n_blocks}")

    # Collect the detected SW rows from the index
    sw_idx_path = ROOT / args.sw_index
    detected_rows: list[int] = []
    if sw_idx_path.exists():
        full = json.loads(sw_idx_path.read_text())
        entry = full.get("evo1", {})
        results = entry.get("results", []) if isinstance(entry, dict) else []
        detected_rows = sorted({int(r["row"]) for r in results
                                if int(r["layer"]) == args.detected_layer})
    print(f"[uk_audit_evo1] detected rows @ layer {args.detected_layer}: "
          f"{detected_rows[:10]}{' …' if len(detected_rows) > 10 else ''}")

    out: dict = {
        "model": "evo1",
        "n_blocks": n_blocks,
        "detected_layer": args.detected_layer,
        "detected_rows":  detected_rows,
        "frob_norm_uk_by_layer": {},
        "median_frob_by_layer": {},
        "max_frob_by_layer":    {},
        "concentration_ratio_by_layer": {},  # max / median — gate vs encoder
        "detected_row_ranks":   {},
        "detected_row_frob":    {},
        "top1_frob_by_layer":   {},
    }

    for li in range(n_blocks):
        mlp = model.blocks[li].mlp
        Wg = mlp.l1.weight.detach().float().cpu().numpy()
        Wu = mlp.l2.weight.detach().float().cpu().numpy()
        Wd = mlp.l3.weight.detach().float().cpu().numpy()

        d_ffn, d_model = Wg.shape
        assert Wu.shape == (d_ffn, d_model)
        assert Wd.shape == (d_model, d_ffn)

        norm_g2 = (Wg ** 2).sum(axis=1)
        norm_u2 = (Wu ** 2).sum(axis=1)
        weight  = norm_g2 * norm_u2
        frob    = np.sqrt((Wd ** 2 * weight[None, :]).sum(axis=1))   # (d_model,)

        order = np.argsort(-frob)
        rank_of = {int(r): int(np.where(order == r)[0][0] + 1)
                   for r in range(d_model)}

        median = float(np.median(frob))
        mx     = float(frob.max())
        out["frob_norm_uk_by_layer"][str(li)] = frob.tolist()
        out["median_frob_by_layer"][str(li)]  = median
        out["max_frob_by_layer"][str(li)]     = mx
        out["concentration_ratio_by_layer"][str(li)] = mx / median if median > 0 else float("nan")
        out["top1_frob_by_layer"][str(li)]    = int(order[0])

        if li == args.detected_layer and detected_rows:
            ranks = {int(r): rank_of[r] for r in detected_rows}
            fvals = {int(r): float(frob[r]) for r in detected_rows}
            out["detected_row_ranks"][str(li)] = ranks
            out["detected_row_frob"][str(li)]  = fvals
            print(f"  layer {li:2d}  d_model={d_model}  d_ffn={d_ffn}  "
                  f"median={median:.3f}  max={mx:.3f}  ratio={mx/median:.2f}")
            for r in detected_rows[:10]:
                print(f"    detected row {r:5d}  ||U_k||_F = {frob[r]:8.3f}  "
                      f"rank {rank_of[r]:5d}/{d_model}")
        elif li in (0, n_blocks // 2, n_blocks - 1):
            print(f"  layer {li:2d}  median={median:.3f}  max={mx:.3f}  ratio={mx/median:.2f}")

    # Cross-architecture concentration comparison summary
    ratios = list(out["concentration_ratio_by_layer"].values())
    print(f"\n[uk_audit_evo1] concentration ratio (max/median ||U_k||_F):")
    print(f"  min   = {min(ratios):.2f}")
    print(f"  med   = {float(np.median(ratios)):.2f}")
    print(f"  max   = {max(ratios):.2f}")
    print(f"  (DNABERT-2 SW layers ratio is ~3-7x; gated transformer with concentrated SW)")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[uk_audit_evo1] wrote {out_path}")


if __name__ == "__main__":
    main()
