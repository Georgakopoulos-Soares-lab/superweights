#!/usr/bin/env python
"""
TASK 3a/3b — Corrected scale preservation: the GROUP, not just the max element.

Correction to the prior claim
-----------------------------
We previously reported "per-row RTN preserves the super weight bit-exactly (0.000e+00 error
at INT8/4/3/2)". That is true, but ONLY of the single scale-defining element. Per-row
symmetric RTN sets s = max|w_row| / qmax, so the max element maps to round(qmax) = qmax and
dequantises to exactly max|w|. Every OTHER element -- including the other large outliers
that a "super weight" often denotes in the literature -- is quantised normally and can be
destroyed (we measured up to 98% mean relative error at INT2).

So the honest statement depends on the definition of "super weight":
  * super weight = one scalar   -> per-row RTN protects it for free (exemption is a no-op)
  * super weight = a GROUP of outlier weights -> per-row RTN protects only the largest;
    the rest need explicit protection.

This script quantifies that distinction, and adds the granularity people actually deploy.

3a — Group preservation under PER-ROW RTN
  For each SW row, define the SW group as the top-M elements by |w| (M = 1, 4, 16).
  Report per-element quantisation error by rank within the group, at INT8/4/3/2.
  Expectation: rank-1 -> exactly 0; ranks 2..M -> increasing error.

3b — GROUP-WISE RTN (g = 64, 128 contiguous blocks, as production kernels use)
  Scale is per contiguous block of g weights. A SW element is protected iff it is the max
  |w| WITHIN ITS OWN BLOCK. Report the fraction of SW elements that are block-max (hence
  exact) and the error distribution for those that are not.

Output: results/mechanism/scale_preservation_grouped.{json,csv}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from run_gue_multiseed import _load_model  # noqa: E402
from run_gue_ablation import _resolve_module  # noqa: E402


def rtn_per_row(W, bits):
    qmax = 2 ** (bits - 1) - 1
    s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
    return torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s


def rtn_per_tensor(W, bits):
    qmax = 2 ** (bits - 1) - 1
    s = (W.abs().max() / qmax).clamp(min=1e-12)
    return torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s


def rtn_group(W, bits, g):
    """Group-wise RTN over contiguous blocks of g along the input dim."""
    qmax = 2 ** (bits - 1) - 1
    R, C = W.shape
    pad = (-C) % g
    Wp = torch.nn.functional.pad(W, (0, pad))
    Wb = Wp.view(R, -1, g)
    s = Wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / qmax
    Qb = torch.clamp(torch.round(Wb / s), -qmax - 1, qmax) * s
    return Qb.view(R, -1)[:, :C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--ckpt", default="results/gue_checkpoints_multiseed/"
                                      "dnabert2_reconstructed/seed_0/model_state.pt")
    ap.add_argument("--sw_index", default="results/negative_results/super_weight_index.json")
    ap.add_argument("--bits", nargs="+", type=int, default=[8, 4, 3, 2])
    ap.add_argument("--M", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--groups", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--out", default="results/mechanism/scale_preservation_grouped.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    pattern = cfg["down_proj_pattern"]
    model, _ = _load_model(cfg["model_id"], 3, None, args.device)
    ck = ROOT / args.ckpt
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location="cpu"), strict=False)
    model.eval()

    sw = json.loads((ROOT / args.sw_index).read_text())[args.model]["results"]
    coords = sorted({(int(e["layer"]), int(e["row"])) for e in sw})

    rows_out, groupwise_out = [], []
    for (l, r) in coords:
        W = _resolve_module(model, pattern, l).weight.data.float()
        w = W[r]
        order = torch.argsort(w.abs(), descending=True)
        Mmax = max(args.M)
        grp_idx = order[:Mmax]

        # ---- 3a: per-row RTN, error by rank within the SW group ----
        for bits in args.bits:
            Wq = rtn_per_row(W, bits)
            err = (Wq[r] - w).abs()
            rel = err / w.abs().clamp(min=1e-12)
            rec = {"layer": l, "row": r, "bits": bits, "granularity": "per_row",
                   "row_max_abs": float(w.abs().max()),
                   "tensor_max_abs": float(W.abs().max()),
                   "is_tensor_max": bool(abs(float(w.abs().max()) - float(W.abs().max())) < 1e-9)}
            for k, gi in enumerate(grp_idx.tolist(), start=1):
                rec[f"rank{k}_abs_err"] = float(err[gi])
                rec[f"rank{k}_rel_err"] = float(rel[gi])
            for M in args.M:
                sel = grp_idx[:M]
                rec[f"group_M{M}_mean_abs_err"] = float(err[sel].mean())
                rec[f"group_M{M}_mean_rel_err"] = float(rel[sel].mean())
                rec[f"group_M{M}_max_rel_err"] = float(rel[sel].max())
                rec[f"group_M{M}_n_exact"] = int((err[sel] == 0).sum())
            rows_out.append(rec)

        # ---- 3b: group-wise RTN ----
        for g in args.groups:
            block_of = (grp_idx // g)
            for bits in args.bits:
                Wq = rtn_group(W, bits, g)
                err = (Wq[r] - w).abs()
                rel = err / w.abs().clamp(min=1e-12)
                # is each SW-group element the max within its own block?
                is_block_max = []
                for gi in grp_idx.tolist():
                    b0 = (gi // g) * g
                    blk = w[b0:b0 + g]
                    is_block_max.append(bool(w[gi].abs() >= blk.abs().max() - 1e-12))
                rec = {"layer": l, "row": r, "bits": bits, "granularity": f"group_{g}",
                       "group_size": g}
                for M in args.M:
                    sel = grp_idx[:M]
                    rec[f"group_M{M}_frac_block_max"] = float(np.mean(is_block_max[:M]))
                    rec[f"group_M{M}_n_exact"] = int((err[sel] == 0).sum())
                    rec[f"group_M{M}_mean_rel_err"] = float(rel[sel].mean())
                    rec[f"group_M{M}_max_rel_err"] = float(rel[sel].max())
                rec["rank1_abs_err"] = float(err[grp_idx[0]])
                groupwise_out.append(rec)

    # ---- summary tables ----
    def agg(recs, gran, bits, field):
        v = [x[field] for x in recs if x["granularity"] == gran and x["bits"] == bits
             and field in x]
        return float(np.mean(v)) if v else None

    summary = {"per_row": {}, "group": {}}
    for bits in args.bits:
        summary["per_row"][f"int{bits}"] = {
            f"M{M}": {"mean_rel_err": agg(rows_out, "per_row", bits, f"group_M{M}_mean_rel_err"),
                      "max_rel_err": agg(rows_out, "per_row", bits, f"group_M{M}_max_rel_err"),
                      "mean_n_exact": agg(rows_out, "per_row", bits, f"group_M{M}_n_exact")}
            for M in args.M}
        summary["per_row"][f"int{bits}"]["rank_err"] = {
            f"rank{k}": agg(rows_out, "per_row", bits, f"rank{k}_rel_err")
            for k in range(1, min(max(args.M), 8) + 1)}
    for g in args.groups:
        summary["group"][f"g{g}"] = {}
        for bits in args.bits:
            summary["group"][f"g{g}"][f"int{bits}"] = {
                f"M{M}": {"frac_block_max": agg(groupwise_out, f"group_{g}", bits, f"group_M{M}_frac_block_max"),
                          "mean_rel_err": agg(groupwise_out, f"group_{g}", bits, f"group_M{M}_mean_rel_err"),
                          "mean_n_exact": agg(groupwise_out, f"group_{g}", bits, f"group_M{M}_n_exact")}
                for M in args.M}

    import transformers as _tf
    out = {"task": "T3ab_group_scale_preservation", "model": args.model,
           "model_id": cfg["model_id"], "ckpt": str(args.ckpt),
           "provenance": {"torch": torch.__version__, "transformers": _tf.__version__,
                          "bits": args.bits, "M": args.M, "groups": args.groups},
           "sw_coords": [list(c) for c in coords],
           "per_row_records": rows_out, "group_records": groupwise_out,
           "summary": summary}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))

    csv = str(op).replace(".json", ".csv")
    with open(csv, "w") as f:
        f.write("granularity,bits,layer,row,M,metric,value\n")
        for rec in rows_out + groupwise_out:
            for M in args.M:
                for m in ("mean_rel_err", "max_rel_err", "n_exact", "frac_block_max"):
                    k = f"group_M{M}_{m}"
                    if k in rec:
                        f.write(f"{rec['granularity']},{rec['bits']},{rec['layer']},"
                                f"{rec['row']},{M},{m},{rec[k]}\n")

    # ---- console report ----
    print("\n=== 3a: PER-ROW RTN — error by rank within the SW group ===")
    print(f"{'bits':>5}" + "".join(f"{'rank'+str(k):>12}" for k in range(1, 6)))
    for bits in args.bits:
        r = summary["per_row"][f"int{bits}"]["rank_err"]
        print(f"{bits:>5}" + "".join(f"{r.get('rank'+str(k)) or 0:>12.4f}" for k in range(1, 6)))
    print("  (mean relative error; rank1 = the scale-defining max)")

    print("\n=== 3a: group mean relative error by M ===")
    print(f"{'bits':>5}" + "".join(f"{'M='+str(M):>12}" for M in args.M))
    for bits in args.bits:
        s = summary["per_row"][f"int{bits}"]
        print(f"{bits:>5}" + "".join(f"{s[f'M{M}']['mean_rel_err']:>12.4f}" for M in args.M))

    print("\n=== 3b: GROUP-WISE RTN — fraction of SW-group elements that are block-max ===")
    for g in args.groups:
        print(f" g={g}:")
        print(f"{'bits':>7}" + "".join(f"{'M='+str(M):>16}" for M in args.M))
        for bits in args.bits:
            s = summary["group"][f"g{g}"][f"int{bits}"]
            print(f"{bits:>7}" + "".join(
                f"{s[f'M{M}']['frac_block_max']:>8.2f}/{s[f'M{M}']['mean_rel_err']:>7.4f}"
                for M in args.M))
        print("        (frac_block_max / mean_rel_err)")
    print(f"\nsaved → {op}\n        {csv}")


if __name__ == "__main__":
    main()
