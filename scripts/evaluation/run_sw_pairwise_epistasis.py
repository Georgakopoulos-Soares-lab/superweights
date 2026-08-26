"""
scripts/evaluation/run_sw_pairwise_epistasis.py
---------------------------------------------------
Which super-weight rows INTERACT?

The k-of-N curve (run_sw_superadditivity.py) showed DNABERT-2 splice collapses
by ~20 pp between k=4 and k=5 -- i.e. when L9/r294 joins a set that already
contains L9/r264. Individually those rows cost -0.13 and -0.15 pp. That
predicts a specific pairwise interaction rather than diffuse superadditivity.

This measures the full C(N,2) interaction matrix:

    epistasis(a,b) = delta_acc(ablate a AND b) - [delta_acc(a) + delta_acc(b)]

Strongly negative epistasis = the pair is jointly far more damaging than the
sum of its parts. If the cliff is carried by one specific pair, that pair will
stand out and the rest of the matrix will be ~0. If superadditivity is diffuse,
many pairs will show it.

Control: matched random pairs, same count, to confirm any epistasis is
SW-specific rather than a generic consequence of zeroing two rows.

Usage:
  python scripts/evaluation/run_sw_pairwise_epistasis.py --model dnabert2 --task splice/reconstructed

Output: results/mechanism/pairwise_epistasis_{model}_{taskleaf}.json
"""
from __future__ import annotations

import argparse
import csv as _csv
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from run_gue_ablation import (  # noqa: E402
    GUEDataset, collate_fn, _save_row, _zero_row, _restore_row,
    _MAX_LEN, _task_key, _resolve_module,
)
from run_gue_multiseed import _load_model, _set_seed  # noqa: E402

import sklearn.metrics  # noqa: E402

SEED = 42
# Original hardcoded path (/data/nvidia/data/gue/GUE) is absent on this filesystem;
# repointed to the GUE data actually present here. No science changed.
GUE_ROOT = Path("/work/11034/atzanakak/GUE/GUE")


def evaluate(model, ds, device, batch=64):
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, collate_fn=collate_fn)
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            preds.extend(model(**b).logits.argmax(-1).cpu().numpy())
            labels.extend(b["labels"].cpu().numpy())
    return float(sklearn.metrics.accuracy_score(np.array(labels), np.array(preds)))


def with_ablation(model, pattern, coords, fn):
    saves = [(l, r, _save_row(model, pattern, l, r)) for l, r in coords]
    for l, r in coords:
        _zero_row(model, pattern, l, r)
    try:
        return fn()
    finally:
        for l, r, s in saves:
            _restore_row(model, pattern, l, r, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt_root", default="results/gue_checkpoints_multiseed")
    ap.add_argument("--sw_index", default="results/super_weight_index.json",
                    help="SW index JSON. Use results/mechanism/super_weight_index_ntv3_deep.json "
                         "for the unblocked NTv3 detection (30 rows vs the frozen index's 1).")
    ap.add_argument("--max_length", type=int, default=None,
                    help="Override _MAX_LEN. REQUIRED for NTv3: _MAX_LEN is tuned for "
                         "DNABERT-2 BPE (~5bp/token); NTv3 is nucleotide-level, so 80 "
                         "tokens truncates a 400bp splice window to 20%.")
    ap.add_argument("--top_n", type=int, default=0,
                    help="Use only the top-N rows by out_max (0 = all). Set 10 to match the "
                         "DNABERT-2 protocol exactly.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_rand_pairs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    task_leaf = args.task.split("/")[-1]
    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    pattern = cfg["down_proj_pattern"]
    max_length = args.max_length or _MAX_LEN.get(_task_key(args.task), 512)

    _swp = Path(args.sw_index)
    if not _swp.is_absolute():
        _swp = ROOT / _swp
    sw = json.loads(_swp.read_text())[args.model]["results"]
    sw = sorted(sw, key=lambda e: -(e.get("out_max") or 0))
    coords = [(int(e["layer"]), int(e["row"])) for e in sw]
    if args.top_n and args.top_n < len(coords):
        coords = coords[:args.top_n]
    N = len(coords)

    ck_dir = ROOT / args.ckpt_root / f"{args.model}_{task_leaf}"
    seeds = sorted(int(m.group(1)) for d in ck_dir.glob("seed_*")
                    if (m := re.match(r"seed_(\d+)$", d.name)) and (d / "model_state.pt").exists())
    print(f"[epistasis] {args.model} {args.task} N={N} seeds={seeds} max_len={max_length}")

    test_csv = GUE_ROOT / args.task / "test.csv"
    with open(test_csv) as f:
        num_labels = len({int(r[-1]) for r in list(_csv.reader(f))[1:]})
    _m, tokenizer = _load_model(cfg["model_id"], num_labels, None, args.device)
    del _m
    torch.cuda.empty_cache()
    ds = GUEDataset(str(test_csv), tokenizer, max_length)

    per_seed = {}
    for seed in seeds:
        print(f"\n[epistasis] === seed {seed} ===")
        _set_seed(seed)
        model, _ = _load_model(cfg["model_id"], ds.num_labels, None, args.device)
        model.load_state_dict(torch.load(ck_dir / f"seed_{seed}" / "model_state.pt",
                                          map_location="cpu"), strict=False)

        base = evaluate(model, ds, args.device, args.batch)
        singles = {}
        for i, c in enumerate(coords):
            singles[i] = with_ablation(model, pattern, [c],
                                        lambda: evaluate(model, ds, args.device, args.batch)) - base
        print(f"  baseline={base:.4f}  singles(pp)="
              f"{[round(singles[i]*100,2) for i in range(N)]}")

        pairs = {}
        for i, j in itertools.combinations(range(N), 2):
            d_ij = with_ablation(model, pattern, [coords[i], coords[j]],
                                  lambda: evaluate(model, ds, args.device, args.batch)) - base
            eps = d_ij - (singles[i] + singles[j])
            pairs[f"{i}-{j}"] = {
                "row_a": list(coords[i]), "row_b": list(coords[j]),
                "delta_pair": d_ij, "delta_a": singles[i], "delta_b": singles[j],
                "epistasis": eps}
            if eps < -0.02:
                print(f"    STRONG  L{coords[i][0]}r{coords[i][1]} + L{coords[j][0]}r{coords[j][1]}"
                      f"  pair={d_ij*100:+.2f}  parts={(singles[i]+singles[j])*100:+.2f}"
                      f"  epistasis={eps*100:+.2f} pp")

        # random-pair control
        n_out = _resolve_module(model, pattern, coords[0][0]).weight.shape[0]
        sw_rows = {r for _, r in coords}
        rng = np.random.RandomState(SEED + seed)
        rand_eps = []
        for _ in range(args.n_rand_pairs):
            pick = []
            while len(pick) < 2:
                l = coords[rng.randint(N)][0]
                r = int(rng.randint(n_out))
                if r not in sw_rows:
                    pick.append((l, r))
            da = with_ablation(model, pattern, [pick[0]], lambda: evaluate(model, ds, args.device, args.batch)) - base
            db = with_ablation(model, pattern, [pick[1]], lambda: evaluate(model, ds, args.device, args.batch)) - base
            dab = with_ablation(model, pattern, pick, lambda: evaluate(model, ds, args.device, args.batch)) - base
            rand_eps.append(dab - (da + db))
        rand_eps = np.array(rand_eps)
        print(f"  random-pair epistasis: {rand_eps.mean()*100:+.3f} +- {rand_eps.std()*100:.3f} pp")

        per_seed[seed] = {"baseline": base, "singles": {str(k): v for k, v in singles.items()},
                          "pairs": pairs,
                          "random_pair_epistasis_mean": float(rand_eps.mean()),
                          "random_pair_epistasis_sd": float(rand_eps.std())}
        del model
        torch.cuda.empty_cache()

    # aggregate epistasis across seeds
    keys = list(per_seed[seeds[0]]["pairs"])
    agg = []
    for k in keys:
        vals = np.array([per_seed[s]["pairs"][k]["epistasis"] for s in seeds]) * 100
        p0 = per_seed[seeds[0]]["pairs"][k]
        agg.append({"pair": k, "row_a": p0["row_a"], "row_b": p0["row_b"],
                    "epistasis_pp_mean": float(vals.mean()),
                    "epistasis_pp_sd": float(vals.std())})
    agg.sort(key=lambda x: x["epistasis_pp_mean"])
    rand_all = np.array([per_seed[s]["random_pair_epistasis_mean"] for s in seeds]) * 100

    print(f"\n[epistasis] strongest interactions (mean over {len(seeds)} seeds):")
    for a in agg[:6]:
        print(f"   L{a['row_a'][0]}r{a['row_a'][1]} + L{a['row_b'][0]}r{a['row_b'][1]}: "
              f"{a['epistasis_pp_mean']:+.2f} +- {a['epistasis_pp_sd']:.2f} pp")
    print(f"[epistasis] random-pair control: {rand_all.mean():+.3f} pp")

    out = {"model": args.model, "task": args.task, "seeds": seeds, "n_sw": N,
           "sw_coords": [list(c) for c in coords], "max_length": max_length, "seed": SEED,
           "definition": "epistasis(a,b) = delta(a AND b) - [delta(a) + delta(b)], in accuracy pp",
           "per_seed": per_seed, "aggregate_sorted": agg,
           "random_pair_epistasis_pp_mean": float(rand_all.mean())}
    out["sw_index_used"] = args.sw_index
    p = (Path(args.out) if args.out else
         ROOT / "results/mechanism" / f"pairwise_epistasis_{args.model}_{task_leaf}.json")
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"[epistasis] wrote {p}")


if __name__ == "__main__":
    main()
