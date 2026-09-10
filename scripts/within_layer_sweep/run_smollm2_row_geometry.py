#!/usr/bin/env python3
"""
Geometry of the three critical rows in SmolLM2-1.7B layer 7 (Results; Table S6).

Data-free: reads down_proj weights only, no forward pass. Produces the cosine and norm
figures quoted in the manuscript, each referenced against an explicit null -- all pairs among
the layer's 200 highest-norm rows -- so "most aligned in the layer" is a measured percentile
rather than an impression.

  r227  the frozen census candidate      (activation ratio 3181.7, rank 1 of 2048)
  r161  the second individually catastrophic row (ratio 63.5, rank 4)
  r749  higher ratio than r161 yet near-inert   (ratio 358.6, rank 2)

The pair whose ablations cancel (r161, r749) turns out to be the most POSITIVELY aligned pair
in the layer, which is what rules out a simple mutual-cancellation account; that is the reason
this measurement exists. No mechanism is claimed from it.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/within_layer_sweep"
RATIO = {227: 3181.70, 161: 63.54, 749: 358.56}
RANK = {227: 1, 161: 4, 749: 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--revision", default="effd688a12921b4cc83e3312b6feb579f70f9c71")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--rows", type=int, nargs="+", default=[161, 749, 227])
    ap.add_argument("--n_ref", type=int, default=200,
                    help="reference pool: the N highest-norm rows of the layer")
    ap.add_argument("--out", default="results/within_layer_sweep/smollm2_161_749_geometry.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.repo, revision=args.revision,
                                             dtype=torch.float32)
    W = m.model.layers[args.layer].mlp.down_proj.weight.data.float()

    def cos(a, b):
        x, y = W[a], W[b]
        return float((x @ y) / (x.norm() * y.norm()))

    top = torch.topk(W.norm(dim=1), args.n_ref).indices.tolist()
    ref = np.array([cos(a, b) for a, b in itertools.combinations(top, 2)])
    mean, sd = float(ref.mean()), float(ref.std())
    print(f"reference: {len(ref)} pairs among the {args.n_ref} highest-norm rows of layer "
          f"{args.layer}; mean {mean:+.4f}  sd {sd:.4f}  min {ref.min():+.4f}  max {ref.max():+.4f}")

    norms = {str(r): float(W[r].norm()) for r in args.rows}
    for r, n in norms.items():
        print(f"  ||row {r}|| = {n:.4f}   (ratio {RATIO.get(int(r), float('nan')):.2f}, "
              f"rank {RANK.get(int(r), '?')})")
    cosines = {}
    for a, b in itertools.combinations(args.rows, 2):
        c = cos(a, b)
        cosines[f"{a}_{b}"] = c
        print(f"  cos(r{a}, r{b}) = {c:+.4f}   z = {(c - mean) / sd:+.2f}   "
              f"percentile {100 * float((ref < c).mean()):.2f}")

    import transformers as _tf
    rec = dict(experiment="SMOLLM2_L7_ROW_GEOMETRY", repo=args.repo, revision=args.revision,
               layer=args.layer, rows=args.rows,
               row_norms=norms, cosines=cosines,
               reference=dict(n_pairs=int(ref.size), mean=mean, sd=sd,
                              min=float(ref.min()), max=float(ref.max()),
                              pool=f"{args.n_ref} highest-norm rows of layer {args.layer} down_proj"),
               z_scores={k: (v - mean) / sd for k, v in cosines.items()},
               percentiles={k: 100 * float((ref < v).mean()) for k, v in cosines.items()},
               provenance=dict(torch=torch.__version__, transformers=_tf.__version__,
                               data_free=True))
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(rec, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
