#!/usr/bin/env python
"""
Pair-aware compression: can you compress ONE member of a redundant SW pair?

Motivation
----------
We established that DNABERT-2's super-weight rows are not independent: rows that
share a layer, or share a row index across layers, are mutually redundant.
Ablating one is nearly free; ablating both collapses the task (splice seed 0:
-0.13 pp separately vs -33.76 pp jointly).

That predicts a concrete, testable compression rule that no existing heuristic
captures. The Yu et al. rule says "keep SW rows in full precision". If SW rows
are pairwise redundant, that is over-conservative: you should be able to degrade
ONE member of a critical pair for free, and the binding constraint is that you
must not degrade BOTH.

Conditions (per seed, per task)
-------------------------------
  baseline                 no modification
  pair_a_only              degrade member A of the critical pair
  pair_b_only              degrade member B
  pair_both                degrade A and B
  sw_singletons            degrade all SW rows NOT in the critical pair
  random_pair (n seeds)    degrade a random non-SW pair, matched count
  all_sw                   degrade every SW row (the Yu-style worst case)

Degradation modes
-----------------
  --mode zero   zero the row (matches the ablation convention used throughout)
  --mode int4   per-row round-to-nearest INT4 (realistic compression)
  --mode int8   per-row RTN INT8

INT4/INT8 use per-row symmetric RTN: q = clamp(round(w/s), -2^(b-1), 2^(b-1)-1)
with s = max|w| / (2^(b-1)-1), then dequantise. This matches
run_quantization_ablation.py's convention.

Usage
-----
python scripts/negative_results/run_pair_aware_compression.py \
    --task splice_reconstructed --gue_root /data/.../GUE \
    --ckpt_root results/gue_checkpoints_multiseed/dnabert2_reconstructed \
    --pair 9,264 9,294 --mode int4 --out results/compression/pair_aware_splice_int4.json
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
import transformers

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "evaluation"))

from run_gue_multiseed import _load_model  # noqa: E402
from run_gue_ablation import (  # noqa: E402
    GUEDataset, _resolve_module, _save_row, _restore_row,
    evaluate, _task_key, _MAX_LEN,
)


def _quantize_row_(model, pattern, layer, row, bits):
    """In-place per-row symmetric RTN quantise-dequantise."""
    m = _resolve_module(model, pattern, layer)
    w = m.weight.data[row]
    qmax = 2 ** (bits - 1) - 1
    s = w.abs().max() / qmax
    if s == 0:
        return
    m.weight.data[row] = torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s


def _degrade_(model, pattern, rows, mode):
    for l, r in rows:
        if mode == "zero":
            _resolve_module(model, pattern, l).weight.data[r].zero_()
        else:
            _quantize_row_(model, pattern, l, r, int(mode[3:]))


def _eval_rows(model, pattern, rows, test_ds, device, mode):
    saves = [(l, r, _save_row(model, pattern, l, r)) for l, r in rows]
    _degrade_(model, pattern, rows, mode)
    m = evaluate(model, test_ds, device=device)
    for l, r, s in saves:
        _restore_row(model, pattern, l, r, s)
    return m


def _parse_pair(vals):
    out = []
    for v in vals:
        l, r = v.split(",")
        out.append((int(l), int(r)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--task", required=True)
    ap.add_argument("--gue_root", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--sw_index", default="results/negative_results/super_weight_index.json")
    ap.add_argument("--pair", nargs=2, required=True,
                    help="two 'layer,row' coords, e.g. 9,264 9,294")
    ap.add_argument("--mode", choices=["zero", "int2", "int3", "int4", "int8"], default="int4")
    ap.add_argument("--n_rand", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pair = _parse_pair(args.pair)
    cfg = yaml.safe_load(open(f"configs/{args.model}.yaml"))
    pattern, num_layers = cfg["down_proj_pattern"], cfg["num_layers"]
    max_length = _MAX_LEN.get(_task_key(args.task), 512)

    # NOTE: use the same loader that produced the n=5 epistasis results.
    # Do NOT pin the DNABERT-2 revision -- the pinned commit's bert_layers.py
    # requires config.pad_token_id, which transformers 5.5.0 no longer provides.
    _probe, tokenizer = _load_model(cfg["model_id"], 2, None, "cpu")
    tokenizer.model_max_length = max_length
    del _probe
    test_ds = GUEDataset(f"{args.gue_root}/{args.task}/test.csv", tokenizer, max_length)

    sw_list = json.loads(Path(args.sw_index).read_text())[args.model]["results"]
    sw_coords = sorted({(s["layer"], s["row"]) for s in sw_list})
    singletons = [c for c in sw_coords if c not in pair]

    seed_dirs = sorted([d for d in Path(args.ckpt_root).iterdir()
                        if d.is_dir() and (d / "model_state.pt").exists()])
    print(f"seeds: {[d.name for d in seed_dirs]}  mode={args.mode}  pair={pair}")

    per_seed = {}
    for sd in seed_dirs:
        model, _ = _load_model(cfg["model_id"], test_ds.num_labels, None, args.device)
        model.load_state_dict(torch.load(sd / "model_state.pt", map_location="cpu"),
                              strict=False)
        model = model.to(args.device).eval()
        num_rows = _resolve_module(model, pattern, 0).weight.data.shape[0]

        base = evaluate(model, test_ds, device=args.device)
        b = base["accuracy"]
        rec = {"baseline": base}

        conds = {
            "pair_a_only":  [pair[0]],
            "pair_b_only":  [pair[1]],
            "pair_both":    list(pair),
            "sw_singletons": singletons,
            "all_sw":       sw_coords,
        }
        for name, rows in conds.items():
            m = _eval_rows(model, pattern, rows, test_ds, args.device, args.mode)
            rec[name] = {"accuracy": m["accuracy"], "mcc": m["mcc"],
                         "delta_acc_pp": (m["accuracy"] - b) * 100, "n_rows": len(rows)}
            print(f"  {sd.name} {name:15s} acc={m['accuracy']:.4f} "
                  f"dpp={(m['accuracy']-b)*100:+.2f}")

        # random non-SW pairs, matched count = 2
        pool = [(l, r) for l in range(num_layers) for r in range(num_rows)
                if (l, r) not in set(sw_coords)]
        pps = []
        for s in range(args.n_rand):
            rows = random.Random(500 + s).sample(pool, 2)
            m = _eval_rows(model, pattern, rows, test_ds, args.device, args.mode)
            pps.append((m["accuracy"] - b) * 100)
        rec["random_pair"] = {"delta_acc_pp_mean": float(np.mean(pps)),
                              "delta_acc_pp_sd": float(np.std(pps, ddof=1)),
                              "n": args.n_rand}
        print(f"  {sd.name} random_pair     dpp={np.mean(pps):+.3f} +-{np.std(pps,ddof=1):.3f}")

        per_seed[sd.name] = rec
        del model
        torch.cuda.empty_cache()

    # aggregate
    agg = {}
    for k in ("pair_a_only", "pair_b_only", "pair_both", "sw_singletons", "all_sw"):
        v = [per_seed[s][k]["delta_acc_pp"] for s in per_seed]
        agg[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                  "per_seed": v}
    rv = [per_seed[s]["random_pair"]["delta_acc_pp_mean"] for s in per_seed]
    agg["random_pair"] = {"mean": float(np.mean(rv)), "per_seed": rv}
    # the key quantity: superadditivity of the pair under this compression mode
    agg["pair_epistasis_pp"] = {
        "mean": agg["pair_both"]["mean"] - (agg["pair_a_only"]["mean"] + agg["pair_b_only"]["mean"]),
        "per_seed": [per_seed[s]["pair_both"]["delta_acc_pp"]
                     - (per_seed[s]["pair_a_only"]["delta_acc_pp"]
                        + per_seed[s]["pair_b_only"]["delta_acc_pp"]) for s in per_seed],
    }

    out = {"model": args.model, "task": args.task, "mode": args.mode,
           "pair": pair, "sw_coords": sw_coords, "per_seed": per_seed, "aggregate": agg}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {args.out}")
    print(f"PAIR EPISTASIS ({args.mode}): {agg['pair_epistasis_pp']['mean']:+.2f} pp")


if __name__ == "__main__":
    main()
