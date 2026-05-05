"""
scripts/run_gue_multiseed.py
----------------------------
Multi-seed GUE fine-tuning for robust error-bar estimation.

Addresses reviewer concern: "The GUE downstream numbers have no error bars or
seed variance. Re-run with at least 3 seeds for fine-tuning and report mean ± std."

This script wraps run_gue_ablation.py's fine_tune / evaluate / ablation logic,
iterates over N random seeds, and aggregates per-condition statistics:

  - baseline       : evaluate fine-tuned model normally
  - pruned_sw      : zero all SW rows, re-evaluate
  - pruned_rand    : zero N_random_repeats matched-count random row sets, re-evaluate

Outputs merged into --out JSON with mean, std, and per-seed records.

Usage
-----
    python scripts/run_gue_multiseed.py \\
        --model  dnabert2 \\
        --task   prom/prom_core_notata \\
        --gue_root /home/nvidia/data/gue/GUE \\
        --seeds  0 1 2 \\
        --device cuda:2

    python scripts/run_gue_multiseed.py \\
        --model dnabert2 --task EMP/H3K4me3 \\
        --gue_root /home/nvidia/data/gue/GUE --seeds 0 1 2 --device cuda:2

    python scripts/run_gue_multiseed.py \\
        --model dnabert2 --task splice/reconstructed \\
        --gue_root /home/nvidia/data/gue/GUE --seeds 0 1 2 --device cuda:2
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Import shared helpers from run_gue_ablation ──────────────────────────────
from run_gue_ablation import (
    GUEDataset,
    collate_fn,
    evaluate,
    fine_tune,
    _MAX_LEN,
    _EPOCHS,
    _task_key,
    _resolve_module,
    _save_row,
    _zero_row,
    _restore_row,
)


# ─────────────────────────────────────────────────────────────────────────────
# Seed helper
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (mirrors run_gue_ablation.py, supports dnabert2 / ntv3)
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_id: str, num_labels: int, hf_token: str | None, device: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        trust_remote_code=True,
        token=hf_token,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# One seed: fine-tune + evaluate all conditions
# ─────────────────────────────────────────────────────────────────────────────

def run_one_seed(
    seed: int,
    model_id: str,
    sw_list: list,
    pattern: str,
    train_ds,
    val_ds,
    test_ds,
    epochs: int,
    lr: float,
    batch: int,
    n_rand_repeats: int,
    device: str,
    hf_token: str | None,
    ckpt_dir: str,
) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Seed {seed}")
    print(f"{'─'*60}")

    _set_seed(seed)
    model, _ = _load_model(model_id, train_ds.num_labels, hf_token, device)

    # Fine‑tune
    seed_ckpt = os.path.join(ckpt_dir, f"seed_{seed}")
    if os.path.exists(os.path.join(seed_ckpt, "model_state.pt")):
        print(f"  Loading existing checkpoint: {seed_ckpt}")
        model.load_state_dict(
            torch.load(os.path.join(seed_ckpt, "model_state.pt"), map_location="cpu"),
            strict=False,
        )
    else:
        fine_tune(model, train_ds, val_ds, epochs, lr, batch, device, seed_ckpt)

    # Baseline evaluation
    baseline = evaluate(model, test_ds, device=device)
    print(f"  Baseline  acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")

    # SW-ablated evaluation: zero SW rows across all layers via weight modification
    sw_coords = [(int(e["layer"]), int(e["row"])) for e in sw_list]
    saves = [(l, r, _save_row(model, pattern, l, r)) for l, r in sw_coords]
    for l, r in sw_coords:
        _zero_row(model, pattern, l, r)
    sw_res = evaluate(model, test_ds, device=device)
    for l, r, saved in saves:
        _restore_row(model, pattern, l, r, saved)
    print(f"  SW-ablated acc={sw_res['accuracy']:.4f}  mcc={sw_res['mcc']:.4f}"
          f"  Δacc={sw_res['accuracy']-baseline['accuracy']:+.4f}")

    # Derive total rows per layer from actual weight shape
    num_rows_per_layer = _resolve_module(model, pattern, 0).weight.data.shape[0]

    # Random controls: match the exact (layer, row-count-per-layer) distribution
    rand_results = []
    rng = np.random.default_rng(seed + 1000)   # offset from fine-tune seed
    for _ in range(n_rand_repeats):
        # Sample same number of (layer, row) pairs as SW, uniform random
        rand_rows = [
            (int(e["layer"]),
             int(rng.integers(0, num_rows_per_layer)))
            for e in sw_list
        ]
        # Avoid accidentally hitting a real SW row
        rand_rows = [(l, r) for l, r in rand_rows if (l, r) not in sw_coords]
        if not rand_rows:
            continue
        r_saves = [(l, r, _save_row(model, pattern, l, r)) for l, r in rand_rows]
        for l, r in rand_rows:
            _zero_row(model, pattern, l, r)
        r_res = evaluate(model, test_ds, device=device)
        for l, r, saved in r_saves:
            _restore_row(model, pattern, l, r, saved)
        rand_results.append(r_res)

    rand_acc_mean = float(np.mean([r["accuracy"] for r in rand_results]))
    rand_mcc_mean = float(np.mean([r["mcc"] for r in rand_results]))
    rand_acc_std  = float(np.std( [r["accuracy"] for r in rand_results]))
    print(f"  Random ctrl acc={rand_acc_mean:.4f}±{rand_acc_std:.4f}")

    del model
    torch.cuda.empty_cache()

    return {
        "seed": seed,
        "baseline":       baseline,
        "sw_ablated":     sw_res,
        "delta_acc":      sw_res["accuracy"] - baseline["accuracy"],
        "delta_mcc":      sw_res["mcc"]      - baseline["mcc"],
        "rand_acc_mean":  rand_acc_mean,
        "rand_acc_std":   rand_acc_std,
        "rand_mcc_mean":  rand_mcc_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-seed GUE ablation with error bars")
    parser.add_argument("--model",        required=True, choices=["dnabert2", "ntv3"])
    parser.add_argument("--task",         required=True,
                        help="GUE task path, e.g. prom/prom_core_notata")
    parser.add_argument("--gue_root",     required=True)
    parser.add_argument("--seeds",        nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs",       type=int, default=None)
    parser.add_argument("--lr",           type=float, default=3e-5)
    parser.add_argument("--batch",        type=int, default=8)
    parser.add_argument("--max_length",   type=int, default=None)
    parser.add_argument("--n_rand",       type=int, default=10,
                        help="Random control repeats per seed (default 10)")
    parser.add_argument("--ckpt_dir",     default=None)
    parser.add_argument("--sw_index",     default="results/super_weight_index.json")
    parser.add_argument("--configs_dir",  default="configs")
    parser.add_argument("--out",          default="results/gue_multiseed_results.json")
    parser.add_argument("--hf_token",     default=None)
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    cfg      = yaml.safe_load(open(f"{args.configs_dir}/{args.model}.yaml"))
    sw_data  = json.load(open(args.sw_index))
    sw_entry = sw_data.get(args.model, {})
    sw_list  = sw_entry.get("results", []) if isinstance(sw_entry, dict) else sw_entry

    tkey       = _task_key(args.task)
    task_leaf  = args.task.split("/")[-1]
    max_length = args.max_length or _MAX_LEN.get(tkey, 512)
    epochs     = args.epochs     or _EPOCHS.get(tkey, 3)
    ckpt_dir   = args.ckpt_dir   or f"results/gue_checkpoints_multiseed/{args.model}_{task_leaf}"

    model_id   = cfg["model_id"]
    pattern    = cfg["down_proj_pattern"]

    # SW list: full (layer, row) entries for accurate per-layer zeroing
    if not sw_list:
        print(f"[error] No SW found for {args.model} in {args.sw_index}")
        sys.exit(1)
    print(f"  SW entries ({len(sw_list)}): "
          + ", ".join(f"L{e['layer']}r{e['row']}" for e in sw_list))

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # ── Build datasets (shared across seeds) ─────────────────────────────────
    task_path = os.path.join(args.gue_root, args.task)
    print(f"  Loading data from {task_path} ...")

    # Load tokenizer once for dataset construction
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token
    )

    train_ds  = GUEDataset(os.path.join(task_path, "train.csv"), tokenizer, max_length)
    val_ds    = GUEDataset(os.path.join(task_path, "dev.csv"),   tokenizer, max_length)
    test_path = os.path.join(task_path, "test.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join(task_path, "dev.csv")
    test_ds   = GUEDataset(test_path, tokenizer, max_length)

    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"  num_labels={train_ds.num_labels}  max_length={max_length}  epochs={epochs}")

    # ── Run each seed ─────────────────────────────────────────────────────────
    per_seed = []
    for seed in args.seeds:
        result = run_one_seed(
            seed       = seed,
            model_id   = model_id,
            sw_list    = sw_list,
            pattern    = pattern,
            train_ds   = train_ds,
            val_ds     = val_ds,
            test_ds    = test_ds,
            epochs     = epochs,
            lr         = args.lr,
            batch      = args.batch,
            n_rand_repeats = args.n_rand,
            device     = args.device,
            hf_token   = hf_token,
            ckpt_dir   = ckpt_dir,
        )
        per_seed.append(result)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    bsl_acc   = np.array([r["baseline"]["accuracy"]   for r in per_seed])
    bsl_mcc   = np.array([r["baseline"]["mcc"]        for r in per_seed])
    sw_acc    = np.array([r["sw_ablated"]["accuracy"]  for r in per_seed])
    sw_mcc    = np.array([r["sw_ablated"]["mcc"]       for r in per_seed])
    rand_acc  = np.array([r["rand_acc_mean"]           for r in per_seed])
    delta_acc = np.array([r["delta_acc"]               for r in per_seed])

    print(f"\n{'='*60}")
    print(f"  SUMMARY  {args.model} / {args.task}  seeds={args.seeds}")
    print(f"{'='*60}")
    print(f"  Baseline   acc = {bsl_acc.mean():.4f} ± {bsl_acc.std():.4f}  "
          f"  mcc = {bsl_mcc.mean():.4f} ± {bsl_mcc.std():.4f}")
    print(f"  SW-ablated acc = {sw_acc.mean():.4f} ± {sw_acc.std():.4f}  "
          f"  Δacc = {delta_acc.mean():+.4f} ± {delta_acc.std():.4f}")
    print(f"  Random ctrl acc= {rand_acc.mean():.4f} ± {rand_acc.std():.4f}")

    # Statistical test: is mean Δacc significantly < 0?
    from scipy import stats as sp
    t_stat, p_val = sp.ttest_1samp(delta_acc, 0)
    print(f"  One-sample t-test Δacc vs 0: t={t_stat:.3f}  p={p_val:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if Path(args.out).exists():
        existing = json.load(open(args.out))

    key = f"{args.model}/{args.task}"
    existing[key] = {
        "model": args.model,
        "task":  args.task,
        "seeds": args.seeds,
        "n_sw_rows": len(sw_list),
        "aggregate": {
            "baseline_acc_mean":   float(bsl_acc.mean()),
            "baseline_acc_std":    float(bsl_acc.std()),
            "baseline_mcc_mean":   float(bsl_mcc.mean()),
            "baseline_mcc_std":    float(bsl_mcc.std()),
            "sw_acc_mean":         float(sw_acc.mean()),
            "sw_acc_std":          float(sw_acc.std()),
            "sw_mcc_mean":         float(sw_mcc.mean()),
            "sw_mcc_std":          float(sw_mcc.std()),
            "delta_acc_mean":      float(delta_acc.mean()),
            "delta_acc_std":       float(delta_acc.std()),
            "rand_acc_mean":       float(rand_acc.mean()),
            "rand_acc_std":        float(rand_acc.std()),
            "t_stat_vs_zero":      float(t_stat),
            "p_val":               float(p_val),
        },
        "per_seed": per_seed,
    }

    with open(args.out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n  Results saved → {args.out}")


if __name__ == "__main__":
    main()
