#!/usr/bin/env python
"""
Does exempting super-weight rows from quantisation ever help? Per-row vs per-tensor.

Why this experiment exists
--------------------------
We proved that per-row symmetric RTN preserves a super weight BIT-EXACTLY: the scale
is s = max|w_row| / qmax, and the SW *is* the row's max element, so it dequantises to
itself (measured: 0.000e+00 error at INT8/4/3/2). Every "retain SW in full precision"
experiment at per-row granularity therefore tests a no-op.

Per-tensor is the granularity where the question becomes real -- the scale comes from
the global matrix max, so a SW is only protected if it happens to BE that global max.
Measured on DNABERT-2 down_proj:

    SW is the global tensor max (error stays 0):  L5r603, L9r294, L3r603, L7r603, L6r603
    SW is NOT the global max (genuinely perturbed): L3r86 (.83), L3r399 (.79),
                                                    L9r264 (.77), L3r641 (.75), L5r86 (.35)

The splice critical pair is split: r294 is the tensor max, r264 is not. So per-tensor
quantisation degrades exactly one member of a redundant pair -- a direct test of whether
the pair structure buys real compression headroom.

Conditions (per granularity x bit-width)
----------------------------------------
  quant_all        quantise every down_proj row, SW included
  exempt_sw        quantise all rows EXCEPT the SW rows (the Yu et al. heuristic)
  exempt_random    quantise all rows except N RANDOM rows, N matched to the SW count
                   (isolates "SW-specific exemption" from "exempting any N rows")
  exempt_pair_a/b  exempt only one member of the critical pair (pair-redundancy test)

The headline number is (exempt_sw - quant_all): the value of the Yu heuristic. At per-row
it must be ~0 by construction; at per-tensor it is an open empirical question.

Usage
-----
python scripts/negative_results/run_per_tensor_sw_exemption.py \
    --task splice/reconstructed --gue_root /data/.../GUE \
    --ckpt_root results/gue_checkpoints_multiseed/dnabert2_reconstructed \
    --bits 4 3 2 --out results/compression/per_tensor_exemption_splice.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "evaluation"))

from run_gue_multiseed import _load_model  # noqa: E402
from run_gue_ablation import (  # noqa: E402
    GUEDataset, _resolve_module, evaluate, _task_key, _MAX_LEN,
)


def _quantize_matrix(W, bits, granularity, exempt_rows=None):
    """Return a quantised copy of W. exempt_rows are left in full precision.

    granularity: 'per_row' | 'per_tensor' | 'group_<g>' (contiguous blocks of g along the
    input dim -- the scheme production INT4/INT8 kernels actually use).
    """
    qmax = 2 ** (bits - 1) - 1
    if granularity.startswith("group_"):
        g = int(granularity.split("_")[1])
        R, C = W.shape
        pad = (-C) % g
        Wp = torch.nn.functional.pad(W, (0, pad))
        Wb = Wp.view(R, -1, g)
        s = Wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / qmax
        Qb = torch.clamp(torch.round(Wb / s), -qmax - 1, qmax) * s
        Wq = Qb.view(R, -1)[:, :C].contiguous()
        if exempt_rows:
            for r in exempt_rows:
                Wq[r] = W[r]
        return Wq
    if granularity == "per_row":
        amax = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        s = amax / qmax
    else:  # per_tensor
        s = (W.abs().max() / qmax).clamp(min=1e-12)
    Wq = torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s
    if exempt_rows:
        for r in exempt_rows:
            Wq[r] = W[r]
    return Wq


def _eval_quantized(model, pattern, num_layers, bits, granularity,
                    exempt_by_layer, test_ds, device):
    saved = {}
    for li in range(num_layers):
        m = _resolve_module(model, pattern, li)
        saved[li] = m.weight.data.clone()
        m.weight.data = _quantize_matrix(
            m.weight.data, bits, granularity, exempt_by_layer.get(li, []))
    out = evaluate(model, test_ds, device=device)
    for li in range(num_layers):
        _resolve_module(model, pattern, li).weight.data = saved[li]
    return out


def _by_layer(coords):
    d = {}
    for l, r in coords:
        d.setdefault(l, []).append(r)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--task", required=True)
    ap.add_argument("--gue_root", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--sw_index", default="results/negative_results/super_weight_index.json")
    ap.add_argument("--bits", nargs="+", type=int, default=[4, 3, 2])
    ap.add_argument("--granularities", nargs="+",
                    default=["per_row", "group_64", "group_128", "per_tensor"])
    ap.add_argument("--pair", nargs=2, default=["9,264", "9,294"])
    ap.add_argument("--n_rand", type=int, default=5)
    ap.add_argument("--max_seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pair = [tuple(int(x) for x in v.split(",")) for v in args.pair]
    cfg = yaml.safe_load(open(f"configs/{args.model}.yaml"))
    pattern, num_layers = cfg["down_proj_pattern"], cfg["num_layers"]
    max_length = _MAX_LEN.get(_task_key(args.task), 512)

    _probe, tokenizer = _load_model(cfg["model_id"], 2, None, "cpu")
    tokenizer.model_max_length = max_length
    del _probe
    test_ds = GUEDataset(f"{args.gue_root}/{args.task}/test.csv", tokenizer, max_length)

    sw = json.loads(Path(args.sw_index).read_text())[args.model]["results"]
    sw_coords = sorted({(s["layer"], s["row"]) for s in sw})
    n_sw = len(sw_coords)

    seed_dirs = sorted([d for d in Path(args.ckpt_root).iterdir()
                        if d.is_dir() and (d / "model_state.pt").exists()])[:args.max_seeds]
    print(f"seeds={[d.name for d in seed_dirs]}  bits={args.bits}  n_sw={n_sw}")

    per_seed = {}
    for sd in seed_dirs:
        model, _ = _load_model(cfg["model_id"], test_ds.num_labels, None, args.device)
        model.load_state_dict(torch.load(sd / "model_state.pt", map_location="cpu"),
                              strict=False)
        model = model.to(args.device).eval()
        nrows = _resolve_module(model, pattern, 0).weight.data.shape[0]
        base = evaluate(model, test_ds, device=args.device)["accuracy"]
        print(f"\n{sd.name}: baseline acc={base:.4f}")

        # record whether each SW is its tensor max (explains any null)
        struct = []
        for l, r in sw_coords:
            W = _resolve_module(model, pattern, l).weight.data
            struct.append({"layer": l, "row": r,
                           "row_max": float(W[r].abs().max()),
                           "tensor_max": float(W.abs().max()),
                           "is_tensor_max": bool(
                               abs(W[r].abs().max() - W.abs().max()) < 1e-9)})

        rec = {"baseline": base, "sw_structure": struct, "runs": {}}
        for gran in args.granularities:
            for bits in args.bits:
                key = f"{gran}_int{bits}"
                conds = {
                    "quant_all":    {},
                    "exempt_sw":    _by_layer(sw_coords),
                    "exempt_pair_a": _by_layer([pair[0]]),
                    "exempt_pair_b": _by_layer([pair[1]]),
                }
                res = {}
                for cname, ex in conds.items():
                    a = _eval_quantized(model, pattern, num_layers, bits, gran,
                                        ex, test_ds, args.device)["accuracy"]
                    res[cname] = {"accuracy": a, "delta_acc_pp": (a - base) * 100}
                # random-exemption control, matched count
                pool = [(l, r) for l in range(num_layers) for r in range(nrows)
                        if (l, r) not in set(sw_coords)]
                accs = []
                for s in range(args.n_rand):
                    ex = _by_layer(random.Random(900 + s).sample(pool, n_sw))
                    accs.append(_eval_quantized(model, pattern, num_layers, bits, gran,
                                                ex, test_ds, args.device)["accuracy"])
                res["exempt_random"] = {"accuracy_mean": float(np.mean(accs)),
                                        "delta_acc_pp_mean": float((np.mean(accs) - base) * 100),
                                        "sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0}
                # the headline: value of the Yu heuristic, and its SW-specific part
                res["exemption_value_pp"] = (res["exempt_sw"]["accuracy"]
                                             - res["quant_all"]["accuracy"]) * 100
                res["exemption_value_vs_random_pp"] = (res["exempt_sw"]["accuracy"]
                                                       - res["exempt_random"]["accuracy_mean"]) * 100
                rec["runs"][key] = res
                print(f"  {key:18s} all={res['quant_all']['delta_acc_pp']:+7.2f}  "
                      f"exSW={res['exempt_sw']['delta_acc_pp']:+7.2f}  "
                      f"exRand={res['exempt_random']['delta_acc_pp_mean']:+7.2f}  "
                      f"VALUE={res['exemption_value_pp']:+6.2f} "
                      f"(vs rand {res['exemption_value_vs_random_pp']:+6.2f})")
        per_seed[sd.name] = rec
        del model
        torch.cuda.empty_cache()

    agg = {}
    keys = list(next(iter(per_seed.values()))["runs"].keys())
    for k in keys:
        agg[k] = {
            "quant_all_pp": float(np.mean([per_seed[s]["runs"][k]["quant_all"]["delta_acc_pp"] for s in per_seed])),
            "exempt_sw_pp": float(np.mean([per_seed[s]["runs"][k]["exempt_sw"]["delta_acc_pp"] for s in per_seed])),
            "exempt_random_pp": float(np.mean([per_seed[s]["runs"][k]["exempt_random"]["delta_acc_pp_mean"] for s in per_seed])),
            "exemption_value_pp": float(np.mean([per_seed[s]["runs"][k]["exemption_value_pp"] for s in per_seed])),
            "exemption_value_vs_random_pp": float(np.mean([per_seed[s]["runs"][k]["exemption_value_vs_random_pp"] for s in per_seed])),
            "exempt_pair_a_pp": float(np.mean([per_seed[s]["runs"][k]["exempt_pair_a"]["delta_acc_pp"] for s in per_seed])),
            "exempt_pair_b_pp": float(np.mean([per_seed[s]["runs"][k]["exempt_pair_b"]["delta_acc_pp"] for s in per_seed])),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "task": args.task, "sw_coords": sw_coords,
         "pair": pair, "per_seed": per_seed, "aggregate": agg}, indent=2))
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
