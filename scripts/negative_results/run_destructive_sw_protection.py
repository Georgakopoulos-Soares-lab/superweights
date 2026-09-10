#!/usr/bin/env python
"""
T2.3 — Honest compression: a DESTRUCTIVE regime, and Yu's actual mechanism.

Why
---
Our prior no-op result tested weight-row exemption in regimes where quantisation barely
moved the baseline -- "no effect" measured where nothing has an effect is weak evidence.
And Yu's real claim is not about exempting weight rows: it is about preserving the super
ACTIVATION and about how far you can push RTN block size. This tests both properly.

Design
------
  1. DESTRUCTIVE REGIME. Sweep until the baseline demonstrably degrades (per-tensor, and
     deep bits). Only cells whose quant_all damage exceeds --min_damage are treated as
     informative; everything else is reported but excluded, so we never claim a null from
     a regime with no headroom.
  2. GROUP protection (not row exemption). Protect the top-M outlier ELEMENTS of each SW
     row (M = 1, 4, 16) in full precision, versus protecting the same number of elements
     in count-matched RANDOM rows. Decisive statistic: protect_sw - protect_random.
  3. BLOCK-SIZE axis (Yu's actual mechanism). For group-wise RTN at g in {32,64,128,256},
     measure quality with and without SW-group protection. If protecting the SW group lets
     you use a LARGER block at fixed quality, that is Yu's benefit and it replicates.

Kill condition (stated first)
-----------------------------
If SW-group protection still confers no benefit in a regime where quantisation genuinely
destroys the model, and does not extend the usable block size, then the no-op conclusion is
bulletproof. If it DOES help here, our earlier claim was scoped too broadly and we report
the crossover point.
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
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "evaluation"))
ROOT = _HERE.parents[1]
sys.path.insert(0, str(ROOT))

from run_gue_multiseed import _load_model  # noqa: E402
from run_gue_ablation import (  # noqa: E402
    GUEDataset, _resolve_module, evaluate, _task_key, _MAX_LEN,
)

SEED = 42


def quantize(W, bits, gran, protect=None):
    """protect: dict row -> LongTensor of column indices kept in full precision."""
    qmax = 2 ** (bits - 1) - 1
    if gran.startswith("group_"):
        g = int(gran.split("_")[1])
        R, C = W.shape
        pad = (-C) % g
        Wp = torch.nn.functional.pad(W, (0, pad))
        Wb = Wp.view(R, -1, g)
        s = Wb.abs().amax(dim=2, keepdim=True).clamp(min=1e-12) / qmax
        Q = (torch.clamp(torch.round(Wb / s), -qmax - 1, qmax) * s).view(R, -1)[:, :C].contiguous()
    elif gran == "per_row":
        s = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
        Q = torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s
    else:
        s = (W.abs().max() / qmax).clamp(min=1e-12)
        Q = torch.clamp(torch.round(W / s), -qmax - 1, qmax) * s
    if protect:
        for r, cols in protect.items():
            Q[r, cols] = W[r, cols]
    return Q


def eval_with(model, pattern, nl, bits, gran, protect_by_layer, ds, dev):
    saved = {}
    for li in range(nl):
        m = _resolve_module(model, pattern, li)
        saved[li] = m.weight.data.clone()
        m.weight.data = quantize(m.weight.data, bits, gran, protect_by_layer.get(li))
    a = evaluate(model, ds, device=dev)["accuracy"]
    for li in range(nl):
        _resolve_module(model, pattern, li).weight.data = saved[li]
    return a


def topM_cols(W, row, M):
    return torch.argsort(W[row].abs(), descending=True)[:M]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--task", default="splice/reconstructed")
    ap.add_argument("--gue_root", default="/data/nvidia/data/gue/GUE")
    ap.add_argument("--ckpt", default="results/gue_checkpoints_multiseed/"
                                      "dnabert2_reconstructed/seed_0/model_state.pt")
    ap.add_argument("--sw_index", default="results/super_weight_index.json")
    ap.add_argument("--M", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--destructive", nargs="+",
                    default=["per_tensor:4", "per_tensor:3", "per_tensor:2", "per_row:2"])
    ap.add_argument("--blocks", nargs="+", type=int, default=[32, 64, 128, 256])
    ap.add_argument("--block_bits", type=int, default=2)
    ap.add_argument("--n_rand", type=int, default=5)
    ap.add_argument("--min_damage", type=float, default=2.0)
    ap.add_argument("--out", default="results/mechanism/compression_destructive_and_activation.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    pattern, nl = cfg["down_proj_pattern"], cfg["num_layers"]
    maxlen = _MAX_LEN.get(_task_key(args.task), 512)
    model, tok = _load_model(cfg["model_id"], 3, None, args.device)
    tok.model_max_length = maxlen
    ds = GUEDataset(f"{args.gue_root}/{args.task}/test.csv", tok, maxlen)
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location="cpu"), strict=False)
    model.eval()
    base = evaluate(model, ds, device=args.device)["accuracy"]
    print(f"[t2.3] baseline acc={base:.4f}")

    sw = json.loads((ROOT / args.sw_index).read_text())[args.model]["results"]
    sw_coords = sorted({(int(e["layer"]), int(e["row"])) for e in sw})
    nrows = _resolve_module(model, pattern, 0).weight.data.shape[0]
    rng = random.Random(SEED)

    def sw_protect(M):
        d = {}
        for l, r in sw_coords:
            W = _resolve_module(model, pattern, l).weight.data
            d.setdefault(l, {})[r] = topM_cols(W, r, M)
        return d

    def rand_protect(M, seed):
        rr = random.Random(seed)
        d = {}
        for l, r in sw_coords:
            W = _resolve_module(model, pattern, l).weight.data
            r2 = rr.randrange(nrows)
            d.setdefault(l, {})[r2] = topM_cols(W, r2, M)
        return d

    out = {"task": "T2.3", "model": args.model, "baseline": base,
           "sw_coords": [list(c) for c in sw_coords], "destructive": {}, "block_axis": {}}

    print("\n=== 1-2. DESTRUCTIVE REGIME + GROUP PROTECTION ===")
    print(f"{'regime':16s}{'quant_all':>11s}{'damage':>9s}" +
          "".join(f"{'M='+str(M):>20s}" for M in args.M))
    for spec in args.destructive:
        gran, bits = spec.split(":")
        bits = int(bits)
        a_all = eval_with(model, pattern, nl, bits, gran, {}, ds, args.device)
        dmg = (base - a_all) * 100
        row = {"quant_all_acc": a_all, "damage_pp": dmg,
               "informative": bool(dmg >= args.min_damage), "by_M": {}}
        cells = ""
        for M in args.M:
            a_sw = eval_with(model, pattern, nl, bits, gran, sw_protect(M), ds, args.device)
            a_rd = [eval_with(model, pattern, nl, bits, gran, rand_protect(M, 700 + s),
                              ds, args.device) for s in range(args.n_rand)]
            gain = (a_sw - float(np.mean(a_rd))) * 100
            row["by_M"][str(M)] = {"protect_sw_acc": a_sw,
                                   "protect_random_acc_mean": float(np.mean(a_rd)),
                                   "sw_gain_vs_random_pp": gain}
            cells += f"{gain:>+20.2f}"
        out["destructive"][spec] = row
        flag = "" if row["informative"] else "  (not destructive - excluded)"
        print(f"{spec:16s}{a_all:>11.4f}{dmg:>+9.2f}{cells}{flag}")

    print("\n=== 3. BLOCK-SIZE AXIS (Yu's actual mechanism) ===")
    print(f"{'block g':>9s}{'no protect':>13s}{'protect SW M=16':>18s}{'gain pp':>10s}")
    for g in args.blocks:
        gran = f"group_{g}"
        a0 = eval_with(model, pattern, nl, args.block_bits, gran, {}, ds, args.device)
        a1 = eval_with(model, pattern, nl, args.block_bits, gran, sw_protect(16), ds, args.device)
        out["block_axis"][str(g)] = {"bits": args.block_bits, "no_protect_acc": a0,
                                     "protect_sw_acc": a1,
                                     "gain_pp": (a1 - a0) * 100}
        print(f"{g:>9d}{a0:>13.4f}{a1:>18.4f}{(a1-a0)*100:>+10.2f}")

    import transformers as _tf
    out["provenance"] = {"seed": SEED, "torch": torch.__version__,
                         "transformers": _tf.__version__, "min_damage": args.min_damage}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    csvp = str(op).replace(".json", ".csv")
    with open(csvp, "w") as f:
        f.write("section,regime,M,metric,value\n")
        for spec, r in out["destructive"].items():
            f.write(f"destructive,{spec},,damage_pp,{r['damage_pp']:.4f}\n")
            for M, v in r["by_M"].items():
                f.write(f"destructive,{spec},{M},sw_gain_vs_random_pp,"
                        f"{v['sw_gain_vs_random_pp']:.4f}\n")
        for g, v in out["block_axis"].items():
            f.write(f"block_axis,group_{g},16,gain_pp,{v['gain_pp']:.4f}\n")
    print(f"\nsaved → {op}\n        {csvp}")


if __name__ == "__main__":
    main()
