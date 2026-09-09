"""
scripts/evaluation/run_sw_superadditivity.py
------------------------------------------------
Is the super-weight effect an ENSEMBLE property?

Single-seed data in results/gue_per_row_ablation.json showed something the
manuscript never used: on DNABERT-2 splice, ablating each of the 10 SW rows
individually costs -1.97 pp SUMMED, but ablating all 10 together costs
-33.3 pp -- a ~17x superadditivity. On promoter it is ~187x (sum of parts
-0.17 pp; all-together -31.7 pp).

This script tests that properly:
  1. MULTISEED  -- repeat across every fine-tuned seed checkpoint available,
     since the original observation was single-seed.
  2. k-of-N CURVE -- ablate the top-k SW rows for k = 1..N (ordered by the
     frozen index's out_max, descending) to locate where the collapse
     happens. A linear curve means additive; a cliff means the ensemble is
     the functional unit.
  3. RANDOM-k CONTROL -- at each k, ablate k random non-SW rows (n_rand
     draws) so the curve is read against the right null.
  4. SUM-OF-PARTS -- each row ablated alone, summed, as the additive
     prediction.

Ablation = zeroing the down-proj weight row, matching run_gue_multiseed.py
and run_gue_per_row_ablation.py so numbers are directly comparable.

Usage:
  python scripts/evaluation/run_sw_superadditivity.py \
      --model dnabert2 --task splice/reconstructed

Output: results/mechanism/superadditivity_{model}_{taskleaf}.json
"""
from __future__ import annotations

import argparse
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
    _MAX_LEN, _task_key,
)
from run_gue_multiseed import _load_model, _set_seed  # noqa: E402

import sklearn.metrics  # noqa: E402

SEED = 42
GUE_ROOT = Path("/data/nvidia/data/gue/GUE")


def evaluate(model, ds, device, batch=64) -> dict:
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, collate_fn=collate_fn)
    preds, labels = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            preds.extend(model(**b).logits.argmax(-1).cpu().numpy())
            labels.extend(b["labels"].cpu().numpy())
    preds, labels = np.array(preds), np.array(labels)
    return {"accuracy": float(sklearn.metrics.accuracy_score(labels, preds)),
            "mcc": float(sklearn.metrics.matthews_corrcoef(labels, preds))}


def ablate_set(model, pattern, coords):
    """Zero a set of (layer,row); returns saved rows for restoration."""
    saves = [(l, r, _save_row(model, pattern, l, r)) for l, r in coords]
    for l, r in coords:
        _zero_row(model, pattern, l, r)
    return saves


def restore(model, pattern, saves):
    for l, r, s in saves:
        _restore_row(model, pattern, l, r, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2", choices=["dnabert2", "ntv3"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt_root", default="results/gue_checkpoints_multiseed")
    ap.add_argument("--n_rand", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    task_leaf = args.task.split("/")[-1]
    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    pattern = cfg["down_proj_pattern"]
    max_length = _MAX_LEN.get(_task_key(args.task), 512)

    sw_list = json.loads((ROOT / "results/super_weight_index.json").read_text())[args.model]["results"]
    # order by detection magnitude, descending -- the order a practitioner
    # would peel them off in
    sw_list = sorted(sw_list, key=lambda e: -(e.get("out_max") or 0))
    coords = [(int(e["layer"]), int(e["row"])) for e in sw_list]
    N = len(coords)

    # auto-discover seed checkpoints
    ck_dir = ROOT / args.ckpt_root / f"{args.model}_{task_leaf}"
    seeds = sorted(int(m.group(1)) for d in ck_dir.glob("seed_*")
                    if (m := re.match(r"seed_(\d+)$", d.name)) and (d / "model_state.pt").exists())
    if not seeds:
        print(f"[superadd] no seed checkpoints under {ck_dir} -- abort")
        return
    print(f"[superadd] model={args.model} task={args.task} max_len={max_length} "
          f"N_sw={N} seeds={seeds}")

    test_csv = GUE_ROOT / args.task / "test.csv"
    # num_labels straight from the CSV, so no throwaway model load is needed
    import csv as _csv
    with open(test_csv) as _f:
        num_labels = len({int(r[-1]) for r in list(_csv.reader(_f))[1:]})

    # the tokenizer is seed-independent: build the dataset once and reuse it
    _m, tokenizer = _load_model(cfg["model_id"], num_labels, None, args.device)
    del _m
    torch.cuda.empty_cache()
    ds = GUEDataset(str(test_csv), tokenizer, max_length)
    print(f"[superadd] test n={len(ds)}  num_labels={ds.num_labels}")

    per_seed = {}
    for seed in seeds:
        print(f"\n[superadd] === seed {seed} ===")
        _set_seed(seed)
        model, _ = _load_model(cfg["model_id"], ds.num_labels, None, args.device)
        model.load_state_dict(
            torch.load(ck_dir / f"seed_{seed}" / "model_state.pt", map_location="cpu"),
            strict=False)

        base = evaluate(model, ds, args.device, args.batch)
        print(f"  baseline acc={base['accuracy']:.4f} mcc={base['mcc']:.4f}")

        # --- individual rows (k=1 each) -> sum of parts ---
        individual = []
        for (l, r) in coords:
            s = ablate_set(model, pattern, [(l, r)])
            m = evaluate(model, ds, args.device, args.batch)
            restore(model, pattern, s)
            individual.append({"layer": l, "row": r,
                               "accuracy": m["accuracy"], "mcc": m["mcc"],
                               "delta_acc": m["accuracy"] - base["accuracy"]})
        sum_parts = sum(x["delta_acc"] for x in individual)
        print(f"  sum-of-parts (10 x k=1) = {sum_parts*100:+.2f} pp")

        # --- cumulative k = 1..N ---
        cumulative = []
        for k in range(1, N + 1):
            s = ablate_set(model, pattern, coords[:k])
            m = evaluate(model, ds, args.device, args.batch)
            restore(model, pattern, s)
            d = m["accuracy"] - base["accuracy"]
            cumulative.append({"k": k, "accuracy": m["accuracy"], "mcc": m["mcc"],
                               "delta_acc": d})
            print(f"    k={k:2d}  acc={m['accuracy']:.4f}  delta={d*100:+7.2f} pp")

        # --- random-k control ---
        # infer down-proj output dim from the weight matrix itself
        from run_gue_ablation import _resolve_module
        n_out = _resolve_module(model, pattern, coords[0][0]).weight.shape[0]
        sw_rows_set = {r for _, r in coords}
        rand_curve = []
        for k in range(1, N + 1):
            accs = []
            for j in range(args.n_rand):
                rng = np.random.RandomState(SEED + 1000 * k + j)
                pick = []
                while len(pick) < k:
                    l = coords[len(pick) % N][0]
                    r = int(rng.randint(n_out))
                    if r not in sw_rows_set:
                        pick.append((l, r))
                s = ablate_set(model, pattern, pick)
                accs.append(evaluate(model, ds, args.device, args.batch)["accuracy"])
                restore(model, pattern, s)
            accs = np.array(accs)
            rand_curve.append({"k": k, "mean_acc": float(accs.mean()),
                               "sd_acc": float(accs.std()),
                               "delta_acc": float(accs.mean() - base["accuracy"])})
        print(f"    random k={N}: delta={rand_curve[-1]['delta_acc']*100:+.2f} pp")

        obs = cumulative[-1]["delta_acc"]
        per_seed[seed] = {
            "baseline": base, "individual": individual,
            "sum_of_parts_delta_acc": sum_parts,
            "cumulative": cumulative, "random_curve": rand_curve,
            "superadditivity_ratio": (obs / sum_parts) if sum_parts != 0 else None,
        }
        print(f"  ALL {N}: {obs*100:+.2f} pp   sum-of-parts {sum_parts*100:+.2f} pp   "
              f"ratio={per_seed[seed]['superadditivity_ratio']}")
        del model
        torch.cuda.empty_cache()

    # ---- aggregate ----
    obs = np.array([per_seed[s]["cumulative"][-1]["delta_acc"] for s in seeds]) * 100
    sop = np.array([per_seed[s]["sum_of_parts_delta_acc"] for s in seeds]) * 100
    ratios = [per_seed[s]["superadditivity_ratio"] for s in seeds]
    curve = np.array([[c["delta_acc"] for c in per_seed[s]["cumulative"]] for s in seeds]) * 100

    out = {
        "model": args.model, "task": args.task, "max_length": max_length,
        "n_sw_rows": N, "sw_coords": [[l, r] for l, r in coords],
        "seeds": seeds, "seed": SEED, "n_rand": args.n_rand,
        "ablation_method": "zero down-proj weight row (matches run_gue_multiseed.py)",
        "per_seed": per_seed,
        "aggregate": {
            "observed_all_rows_pp_mean": float(obs.mean()), "observed_all_rows_pp_sd": float(obs.std()),
            "sum_of_parts_pp_mean": float(sop.mean()), "sum_of_parts_pp_sd": float(sop.std()),
            "superadditivity_ratio_mean": float(np.mean(ratios)),
            "superadditivity_ratio_per_seed": ratios,
            "cumulative_curve_pp_mean": curve.mean(axis=0).tolist(),
            "cumulative_curve_pp_sd": curve.std(axis=0).tolist(),
        },
    }
    outp = ROOT / "results/mechanism" / f"superadditivity_{args.model}_{task_leaf}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\n[superadd] observed {obs.mean():+.2f}+-{obs.std():.2f} pp  vs "
          f"sum-of-parts {sop.mean():+.2f}+-{sop.std():.2f} pp")
    print(f"[superadd] curve (k=1..N): {np.round(curve.mean(axis=0),2).tolist()}")
    print(f"[superadd] wrote {outp}")


if __name__ == "__main__":
    main()
