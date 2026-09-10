#!/usr/bin/env python
"""
Does "SW proximity" predict pruning fragility, or is it a layer-depth confound?

The manuscript's shadow-redundancy claim rests on prox-far rows being far more
fragile than prox-near rows. But proximity is Euclidean distance in normalised
(layer, row) space, and DNABERT-2's SW layers (3,5,6,7,9 of 12) sit in the
middle -- so prox-far preferentially selects the OUTER layers. Measured at 20%:

    prox_far  per-layer: [768, 14, 0, 0, 0, 0, 121, 0, 72, 75, 354, 439]
                          ^^^ ALL of layer 0
    prox_near per-layer: [  0,  0, 179, 479, 176, 294, 194, 147, 120, 177, 77, 0]
                          ^^^ layers 0/10/11 never touched

Pruning all of layer 0 is catastrophic for reasons that have nothing to do with
super weights. This script adds the controls that separate the two accounts:

  prox_far            reproduce the manuscript criterion
  prox_near           reproduce
  random              uniform random (matched count)
  layer_matched_random   <-- KEY: random rows drawn to match prox_far's EXACT
                             per-layer counts at each fraction. If this equals
                             prox_far, SW proximity adds nothing beyond depth.
  sham_prox_far          <-- proximity to SHAM reference points: same layers as
                             the real SWs but non-SW row indices. If this equals
                             prox_far, the SW identity is irrelevant.
  depth_extreme          <-- pure outer-layers-first ordering, no SW anywhere.

Reports ABSOLUTE percentage-point deltas alongside the harness's relative
delta_acc_pct, because the manuscript reports relative values labelled "pp".

Usage
-----
python scripts/negative_results/run_proximity_confound_control.py \
    --model dnabert2 --task prom_core_notata --gue_root /data/.../GUE \
    --ckpt_dir results/gue_checkpoints_multiseed/dnabert2_prom_core_notata/seed_0 \
    --out results/compression/proximity_confound_seed0.json
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
import transformers

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "evaluation"))

from run_gue_multiseed import _load_model  # noqa: E402
from run_gue_ablation import (  # noqa: E402
    GUEDataset, _resolve_module, _save_row, _zero_row, _restore_row,
    evaluate, _task_key, _MAX_LEN,
)


# ── candidate pool ────────────────────────────────────────────────────────────
def _all_candidates(model, pattern, num_layers, sw_coords):
    cands = []
    for li in range(num_layers):
        nrows = _resolve_module(model, pattern, li).weight.data.shape[0]
        for ri in range(nrows):
            if (li, ri) not in sw_coords:
                cands.append((li, ri))
    return cands


def _rank_proximity(candidates, ref_pts, num_layers, num_rows, ascending=True):
    """Rank by normalised Euclidean distance to nearest reference point.

    ascending=True  -> FURTHEST first (prox_far: largest distance pruned first)
    ascending=False -> CLOSEST  first (prox_near)

    Mirrors run_compression_sweep._rank_proximity's normalisation exactly.
    """
    pts = np.array([[l / max(num_layers - 1, 1), r / max(num_rows - 1, 1)]
                    for l, r in candidates], dtype=np.float64)
    ref = np.array([[l / max(num_layers - 1, 1), r / max(num_rows - 1, 1)]
                    for l, r in ref_pts], dtype=np.float64)
    d = np.sqrt(((pts[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min(1)
    order = np.argsort(-d) if ascending else np.argsort(d)
    return [candidates[i] for i in order]


def _rank_depth_extreme(candidates, num_layers):
    """Outer layers first (0, 11, 1, 10, ...), random within layer. No SW used."""
    mid = (num_layers - 1) / 2.0
    rng = random.Random(1234)
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    return sorted(shuffled, key=lambda lr: -abs(lr[0] - mid))


def _layer_matched_random(candidates, template_rows, seed):
    """Random rows matching template_rows' per-layer counts exactly."""
    want = Counter(l for l, _ in template_rows)
    by_layer = {}
    for l, r in candidates:
        by_layer.setdefault(l, []).append((l, r))
    rng = random.Random(seed)
    out = []
    for l, n in want.items():
        pool = by_layer.get(l, [])
        n = min(n, len(pool))
        out.extend(rng.sample(pool, n))
    return out


def _eval_rows(model, pattern, rows, test_ds, device):
    saves = [(l, r, _save_row(model, pattern, l, r)) for l, r in rows]
    for l, r in rows:
        _zero_row(model, pattern, l, r)
    m = evaluate(model, test_ds, device=device)
    for l, r, s in saves:
        _restore_row(model, pattern, l, r, s)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--task", required=True)
    ap.add_argument("--gue_root", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--sw_index", default="results/super_weight_index.json")
    ap.add_argument("--fracs", nargs="+", type=float, default=[5.0, 10.0, 20.0, 30.0])
    ap.add_argument("--n_rand_seeds", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(f"configs/{args.model}.yaml"))
    pattern, num_layers = cfg["down_proj_pattern"], cfg["num_layers"]
    tkey = _task_key(args.task)
    max_length = _MAX_LEN.get(tkey, 512)

    # NOTE: use the same loader that produced the n=5 epistasis results.
    # Do NOT pin the DNABERT-2 revision -- the pinned commit's bert_layers.py
    # requires config.pad_token_id, which transformers 5.5.0 no longer provides.
    _probe, tokenizer = _load_model(cfg["model_id"], 2, None, "cpu")
    tokenizer.model_max_length = max_length
    del _probe
    test_ds = GUEDataset(f"{args.gue_root}/{args.task}/test.csv", tokenizer, max_length)

    model, _ = _load_model(cfg["model_id"], test_ds.num_labels, None, args.device)
    ckpt = Path(args.ckpt_dir) / "model_state.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        print(f"[ok] loaded {ckpt}")
    else:
        raise SystemExit(f"[fatal] no checkpoint at {ckpt}")
    model = model.to(args.device).eval()

    num_rows = _resolve_module(model, pattern, 0).weight.data.shape[0]
    sw_list = json.loads(Path(args.sw_index).read_text())[args.model]["results"]
    sw_coords = {(s["layer"], s["row"]) for s in sw_list}
    sw_pts = sorted(sw_coords)
    cands = _all_candidates(model, pattern, num_layers, sw_coords)
    n_total = len(cands)

    base = evaluate(model, test_ds, device=args.device)
    print(f"baseline acc={base['accuracy']:.4f} mcc={base['mcc']:.4f}  pool={n_total}")

    # SHAM reference: same layers as real SWs, but row indices shifted well away
    rng = random.Random(7)
    sham_pts = []
    real_rows = {r for _, r in sw_pts}
    for l, _ in sw_pts:
        while True:
            rr = rng.randrange(num_rows)
            if all(abs(rr - x) > 50 for x in real_rows):
                sham_pts.append((l, rr)); break

    ranked = {
        "prox_far":   _rank_proximity(cands, sw_pts, num_layers, num_rows, True),
        "prox_near":  _rank_proximity(cands, sw_pts, num_layers, num_rows, False),
        "sham_prox_far": _rank_proximity(cands, sham_pts, num_layers, num_rows, True),
        "depth_extreme": _rank_depth_extreme(cands, num_layers),
    }

    out = {"model": args.model, "task": args.task, "ckpt_dir": args.ckpt_dir,
           "baseline": base, "n_total": n_total, "fracs": args.fracs,
           "sw_coords": sw_pts, "sham_coords": sham_pts, "curves": {}}

    def _rec(m, n):
        return {"n_pruned": n, "accuracy": m["accuracy"], "mcc": m["mcc"],
                "delta_acc_pp": (m["accuracy"] - base["accuracy"]) * 100,
                "delta_acc_rel_pct": (m["accuracy"] - base["accuracy"]) / base["accuracy"] * 100}

    for crit, order in ranked.items():
        print(f"\n[{crit}]")
        curve = []
        for f in args.fracs:
            n = max(1, int(round(n_total * f / 100)))
            r = _rec(_eval_rows(model, pattern, order[:n], test_ds, args.device), n)
            r["frac"] = f
            # per-layer composition of this selection (the confound, made explicit)
            r["layer_counts"] = Counter(l for l, _ in order[:n])
            r["layer_counts"] = [r["layer_counts"].get(i, 0) for i in range(num_layers)]
            print(f"  {f:5.1f}% n={n:6d} acc={r['accuracy']:.4f} "
                  f"dpp={r['delta_acc_pp']:+.2f} layers={r['layer_counts']}")
            curve.append(r)
        out["curves"][crit] = curve

    # stochastic criteria
    for crit in ("random", "layer_matched_random"):
        print(f"\n[{crit}] ({args.n_rand_seeds} seeds)")
        curve = []
        for f in args.fracs:
            n = max(1, int(round(n_total * f / 100)))
            accs, pps = [], []
            for s in range(args.n_rand_seeds):
                if crit == "random":
                    rows = random.Random(1000 + s).sample(cands, n)
                else:
                    rows = _layer_matched_random(cands, ranked["prox_far"][:n], 2000 + s)
                m = _eval_rows(model, pattern, rows, test_ds, args.device)
                accs.append(m["accuracy"])
                pps.append((m["accuracy"] - base["accuracy"]) * 100)
            curve.append({"frac": f, "n_pruned": n,
                          "accuracy_mean": float(np.mean(accs)),
                          "delta_acc_pp_mean": float(np.mean(pps)),
                          "delta_acc_pp_sd": float(np.std(pps, ddof=1)) if len(pps) > 1 else 0.0})
            print(f"  {f:5.1f}% n={n:6d} dpp={np.mean(pps):+.2f} +-{np.std(pps, ddof=1):.2f}")
        out["curves"][crit] = curve

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
