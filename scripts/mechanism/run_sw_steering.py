#!/usr/bin/env python
"""
T2.2 — Does scaling the super-weight row causally steer generated composition?

If the SW encodes nucleotide composition, scaling its write should shift the GC content of
generated sequence monotonically, and matched random rows should not.

Design
------
  conditions : SW row scaled by {0.0, 0.5, 1.0, 2.0, 5.0}
  control    : the SAME scalings applied to 5 matched random rows (seed 42)
  readout    : GC fraction and dinucleotide composition of generated continuations
  prompts    : real hg38 windows (seed 42), greedy-free sampling with fixed seed so the
               only thing that varies between conditions is the scaling

Kill condition (stated first)
-----------------------------
If scaling the SW row does NOT move generated GC monotonically -- or moves it no more than
random rows do -- then the SW does not causally control composition during generation, the
steering application is dropped, and the composition claim stays correlational.

NOTE on the prior result: we found the SW activation is BOS-anchored (peaks at token 0 in
100% of windows, 45,585x other positions). If the SW is an implicit bias rather than a
content feature, the PREDICTION is that steering will NOT move composition. This experiment
is therefore a direct test between the two accounts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "mechanism"))
from run_ensemble_encoding import read_windows, gc_frac, dinuc_freqs  # noqa: E402

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="generator")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--row", type=int, default=2371)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--n_prompts", type=int, default=24)
    ap.add_argument("--prompt_bp", type=int, default=120)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--n_random_rows", type=int, default=5)
    ap.add_argument("--out", default="results/mechanism/sw_steering_generation.json")
    args = ap.parse_args()

    from models import WRAPPER_MAP
    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    w = WRAPPER_MAP[args.model](cfg)
    w.load()
    model, tok = w.model, w.tokenizer
    dev = next(model.parameters()).device
    mods = dict(model.named_modules())
    key = cfg["down_proj_pattern"].format(i=args.layer)
    mod = mods[key]
    nrows = mod.weight.data.shape[0]

    rng = random.Random(SEED)
    wins = read_windows("/data/nvidia/data/hg38/hg38.fa",
                        str(ROOT / "data/regions/hg38/random_262kb.bed"),
                        args.n_prompts, args.prompt_bp + 50, rng)
    prompts = [s[:args.prompt_bp] for _c, _s, s in wins]
    print(f"[steer] {len(prompts)} prompts, scales={args.scales}")

    rand_rows = random.Random(SEED).sample(
        [r for r in range(nrows) if r != args.row], args.n_random_rows)

    def generate(scale_row, scale):
        saved = mod.weight.data[scale_row].clone()
        mod.weight.data[scale_row] = saved * scale
        gcs, seqs = [], []
        try:
            for i, p in enumerate(prompts):
                enc = tok(p, return_tensors="pt")
                ids = enc["input_ids"].to(dev)
                torch.manual_seed(SEED + i)
                with torch.no_grad():
                    o = model.generate(input_ids=ids, max_new_tokens=args.max_new,
                                       do_sample=True, top_k=50, temperature=1.0,
                                       pad_token_id=getattr(tok, "pad_token_id", None) or 0)
                new = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
                new = "".join(c for c in new.upper() if c in "ACGT")
                if len(new) >= 10:
                    gcs.append(gc_frac(new))
                    seqs.append(new)
        finally:
            mod.weight.data[scale_row] = saved
        return gcs, seqs

    results = {"sw": {}, "random": {}}
    print("\n  SW row:")
    for s in args.scales:
        gcs, seqs = generate(args.row, s)
        dn = {}
        for sq in seqs:
            for k, v in dinuc_freqs(sq).items():
                dn.setdefault(k, []).append(v)
        results["sw"][str(s)] = {"n": len(gcs), "gc_mean": float(np.mean(gcs)) if gcs else None,
                                 "gc_sd": float(np.std(gcs)) if gcs else None,
                                 "dinuc_mean": {k: float(np.mean(v)) for k, v in dn.items()},
                                 "example": seqs[0][:60] if seqs else ""}
        print(f"    scale {s:>4}: n={len(gcs):3d}  GC={np.mean(gcs) if gcs else float('nan'):.4f} "
              f"+-{np.std(gcs) if gcs else 0:.4f}")

    print("\n  random control rows:")
    for s in args.scales:
        allgc = []
        for rr in rand_rows:
            gcs, _ = generate(rr, s)
            allgc.extend(gcs)
        results["random"][str(s)] = {"n": len(allgc),
                                     "gc_mean": float(np.mean(allgc)) if allgc else None,
                                     "gc_sd": float(np.std(allgc)) if allgc else None}
        print(f"    scale {s:>4}: n={len(allgc):3d}  GC={np.mean(allgc) if allgc else float('nan'):.4f}")

    sw_gc = [results["sw"][str(s)]["gc_mean"] for s in args.scales]
    rd_gc = [results["random"][str(s)]["gc_mean"] for s in args.scales]
    ok = [g is not None for g in sw_gc]
    span_sw = (max(g for g, k in zip(sw_gc, ok) if k) - min(g for g, k in zip(sw_gc, ok) if k)) if any(ok) else 0
    span_rd = (max(g for g in rd_gc if g is not None) - min(g for g in rd_gc if g is not None)) if any(g is not None for g in rd_gc) else 0
    print(f"\n  GC span across scales: SW={span_sw:.4f}   random={span_rd:.4f}   "
          f"ratio={span_sw/max(span_rd,1e-9):.2f}x")

    import transformers as _tf
    out = {"task": "T2.2_sw_steering_generation", "model": args.model,
           "layer": args.layer, "row": args.row, "scales": args.scales,
           "random_rows": rand_rows,
           "provenance": {"seed": SEED, "torch": torch.__version__,
                          "transformers": _tf.__version__,
                          "n_prompts": len(prompts), "max_new": args.max_new},
           "results": results,
           "gc_span_sw": span_sw, "gc_span_random": span_rd,
           "span_ratio": span_sw / max(span_rd, 1e-9)}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    csvp = str(op).replace(".json", ".csv")
    with open(csvp, "w") as f:
        f.write("condition,scale,n,gc_mean,gc_sd\n")
        for cond in ("sw", "random"):
            for s in args.scales:
                r = results[cond][str(s)]
                f.write(f"{cond},{s},{r['n']},{r['gc_mean']},{r['gc_sd']}\n")
    print(f"saved → {op}\n        {csvp}")


if __name__ == "__main__":
    main()
