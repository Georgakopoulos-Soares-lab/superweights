"""
scripts/negative_results/run_int4_multicondition_benchmark.py
----------------------------------------------------------
Comprehensive INT4 downstream benchmark across ALL conditions:
  near-SW, far-SW, random, U_k-low, U_k-high, yu_all, naive, sw_only, fp16 baseline

Fractions: 10%, 20%, 30%  |  Seeds: 5  |  Tasks: splice, prom_core, H3K4me3

Uses proven model loading from run_quantization_ablation.

Usage:
    python scripts/negative_results/run_int4_multicondition_benchmark.py \\
        --gpu 6 --task splice/reconstructed --bits 4 --n_seeds 5
    python scripts/negative_results/run_int4_multicondition_benchmark.py \\
        --gpu 7 --task prom_core_notata --bits 4 --n_seeds 5
"""
from __future__ import annotations

import argparse, csv as _csv, json, random as _random, sys, time
from pathlib import Path
import numpy as np, torch, yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "scripts" / "evaluation"))

from run_gue_ablation import (
    GUEDataset, evaluate, _task_key, _MAX_LEN, collate_fn,
)
from run_compression_sweep import _all_candidates, _rank_proximity
from run_quantization_ablation import (
    _quantize_row_rtn, _apply_quantization, _restore_quantization,
)
import run_gue_ablation as _rga
import run_compression_sweep as _rcs

def _resolve_module_patched(model, pattern, layer_idx):
    path = pattern.replace("{i}", str(layer_idx))
    obj = model.backbone if hasattr(model, "backbone") else model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj
_rga._resolve_module = _resolve_module_patched
_rcs._resolve_module = _resolve_module_patched


def _load_uk_values(uk_path: Path) -> dict:
    data = json.load(open(uk_path))
    return {int(k): v for k, v in data.get("frob_norm_uk_by_layer", {}).items()}


def _rank_by_uk(uk_by_layer, candidates, num_layers, num_rows, ascending=True):
    uk_flat = []
    for li in range(num_layers):
        layer_uk = uk_by_layer.get(li, [0.0] * num_rows)
        for ri in range(num_rows):
            uk_flat.append(layer_uk[ri] if ri < len(layer_uk) else 0.0)
    scored = []
    for li, ri in candidates:
        val = uk_flat[li * num_rows + ri] if li * num_rows + ri < len(uk_flat) else 0.0
        scored.append(((li, ri), val))
    scored.sort(key=lambda x: x[1], reverse=not ascending)
    return [item[0] for item in scored]


def run_extended_sweep(model, sw_list, test_ds, pattern, num_layers, num_rows,
                       fracs, n_seeds, device, bits=4, uk_by_layer=None):
    sw_coords = {(sw["layer"], sw["row"]) for sw in sw_list}
    sw_rows_list = [(sw["layer"], sw["row"]) for sw in sw_list]
    candidates = _all_candidates(model, pattern, num_layers, sw_coords)
    n_candidates = len(candidates)
    all_rows = [(li, ri) for li in range(num_layers) for ri in range(num_rows)]

    # Baseline
    baseline = evaluate(model, test_ds, device=device)
    print(f"  Baseline: acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")

    # yu_all
    print("  yu_all ...", flush=True)
    saves = _apply_quantization(model, pattern, candidates, bits=bits)
    yu_all = evaluate(model, test_ds, device=device)
    yu_all["delta_acc"] = yu_all["accuracy"] - baseline["accuracy"]
    yu_all["delta_mcc"] = yu_all["mcc"] - baseline["mcc"]
    _restore_quantization(model, pattern, saves)
    print(f"    yu_all: acc={yu_all['accuracy']:.4f}  Δacc={yu_all['delta_acc']:+.4f}")

    # naive (ALL rows)
    print("  naive_int4 ...", flush=True)
    saves = _apply_quantization(model, pattern, all_rows, bits=bits)
    naive = evaluate(model, test_ds, device=device)
    naive["delta_acc"] = naive["accuracy"] - baseline["accuracy"]
    naive["delta_mcc"] = naive["mcc"] - baseline["mcc"]
    _restore_quantization(model, pattern, saves)
    print(f"    naive:  acc={naive['accuracy']:.4f}  Δacc={naive['delta_acc']:+.4f}")

    # SW-only
    print(f"  sw_only ({len(sw_rows_list)} rows) ...", flush=True)
    saves = _apply_quantization(model, pattern, sw_rows_list, bits=bits)
    sw_only = evaluate(model, test_ds, device=device)
    sw_only["delta_acc"] = sw_only["accuracy"] - baseline["accuracy"]
    sw_only["delta_mcc"] = sw_only["mcc"] - baseline["mcc"]
    _restore_quantization(model, pattern, saves)
    print(f"    sw_only: acc={sw_only['accuracy']:.4f}  Δacc={sw_only['delta_acc']:+.4f}")

    # Rankings
    print("  Computing rankings ...", flush=True)
    ranked_near = _rank_proximity(model, pattern, candidates, sw_list,
                                  num_layers, num_rows, ascending=False)
    ranked_far = _rank_proximity(model, pattern, candidates, sw_list,
                                 num_layers, num_rows, ascending=True)
    ranked_uk_low, ranked_uk_high = None, None
    if uk_by_layer is not None:
        ranked_uk_low = _rank_by_uk(uk_by_layer, candidates, num_layers, num_rows, True)
        ranked_uk_high = _rank_by_uk(uk_by_layer, candidates, num_layers, num_rows, False)
        print("    near-SW, far-SW, U_k-low, U_k-high ready")
    else:
        print("    near-SW, far-SW ready (U_k unavailable)")

    results = {
        "model": "dnabert2", "bits": bits, "fracs": fracs, "n_seeds": n_seeds,
        "n_sw": len(sw_rows_list), "n_total": len(all_rows), "n_candidates": n_candidates,
        "baseline": baseline, "yu_all_int4": yu_all, "naive_int4": naive,
        "sw_only_int4": sw_only, "per_frac": [],
    }

    for frac in fracs:
        n_select = max(1, int(round(n_candidates * frac / 100.0)))
        print(f"\n  ── frac={frac:.0f}%  n={n_select} ──")
        fr = {"frac": frac, "n_rows": n_select}

        # near_sw
        chosen = ranked_near[:n_select]
        saves = _apply_quantization(model, pattern, chosen, bits=bits)
        m = evaluate(model, test_ds, device=device)
        m["delta_acc"] = m["accuracy"] - baseline["accuracy"]
        m["delta_mcc"] = m["mcc"] - baseline["mcc"]
        _restore_quantization(model, pattern, saves)
        fr["near_sw"] = m
        print(f"    near_sw:    acc={m['accuracy']:.4f}  Δacc={m['delta_acc']:+.4f}")

        # far_sw
        chosen = ranked_far[:n_select]
        saves = _apply_quantization(model, pattern, chosen, bits=bits)
        m = evaluate(model, test_ds, device=device)
        m["delta_acc"] = m["accuracy"] - baseline["accuracy"]
        m["delta_mcc"] = m["mcc"] - baseline["mcc"]
        _restore_quantization(model, pattern, saves)
        fr["far_sw"] = m
        print(f"    far_sw:     acc={m['accuracy']:.4f}  Δacc={m['delta_acc']:+.4f}")

        # random (multi-seed)
        rand_accs, rand_mccs = [], []
        for seed in range(n_seeds):
            rng = _random.Random(seed)
            chosen = rng.sample(candidates, n_select)
            saves = _apply_quantization(model, pattern, chosen, bits=bits)
            m = evaluate(model, test_ds, device=device)
            _restore_quantization(model, pattern, saves)
            rand_accs.append(m["accuracy"]); rand_mccs.append(m["mcc"])
        fr["random"] = {
            "n_seeds": n_seeds,
            "accuracy_mean": float(np.mean(rand_accs)), "accuracy_std": float(np.std(rand_accs)),
            "mcc_mean": float(np.mean(rand_mccs)), "mcc_std": float(np.std(rand_mccs)),
            "delta_acc_mean": float(np.mean(rand_accs)) - baseline["accuracy"],
            "delta_mcc_mean": float(np.mean(rand_mccs)) - baseline["mcc"],
        }
        print(f"    random:     acc={fr['random']['accuracy_mean']:.4f}"
              f"±{fr['random']['accuracy_std']:.4f}")

        # U_k-low
        if ranked_uk_low is not None:
            uk_accs, uk_mccs = [], []
            for seed in range(n_seeds):
                chosen = ranked_uk_low[:n_select]
                saves = _apply_quantization(model, pattern, chosen, bits=bits)
                m = evaluate(model, test_ds, device=device)
                _restore_quantization(model, pattern, saves)
                uk_accs.append(m["accuracy"]); uk_mccs.append(m["mcc"])
            fr["uk_low"] = {
                "n_seeds": n_seeds,
                "accuracy_mean": float(np.mean(uk_accs)), "accuracy_std": float(np.std(uk_accs)),
                "mcc_mean": float(np.mean(uk_mccs)), "mcc_std": float(np.std(uk_mccs)),
                "delta_acc_mean": float(np.mean(uk_accs)) - baseline["accuracy"],
                "delta_mcc_mean": float(np.mean(uk_mccs)) - baseline["mcc"],
            }
            print(f"    uk_low:     acc={fr['uk_low']['accuracy_mean']:.4f}"
                  f"±{fr['uk_low']['accuracy_std']:.4f}")

        # U_k-high
        if ranked_uk_high is not None:
            ukh_accs, ukh_mccs = [], []
            for seed in range(n_seeds):
                chosen = ranked_uk_high[:n_select]
                saves = _apply_quantization(model, pattern, chosen, bits=bits)
                m = evaluate(model, test_ds, device=device)
                _restore_quantization(model, pattern, saves)
                ukh_accs.append(m["accuracy"]); ukh_mccs.append(m["mcc"])
            fr["uk_high"] = {
                "n_seeds": n_seeds,
                "accuracy_mean": float(np.mean(ukh_accs)), "accuracy_std": float(np.std(ukh_accs)),
                "mcc_mean": float(np.mean(ukh_mccs)), "mcc_std": float(np.std(ukh_mccs)),
                "delta_acc_mean": float(np.mean(ukh_accs)) - baseline["accuracy"],
                "delta_mcc_mean": float(np.mean(ukh_mccs)) - baseline["mcc"],
            }
            print(f"    uk_high:    acc={fr['uk_high']['accuracy_mean']:.4f}"
                  f"±{fr['uk_high']['accuracy_std']:.4f}")

        results["per_frac"].append(fr)

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-condition INT4 downstream benchmark")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--task", type=str, required=True,
                        help="GUE task path, e.g. EMP/H3K9me3, prom/prom_core_all")
    parser.add_argument("--gue_root", type=str, default="/home/nvidia/data/gue/GUE")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--configs_dir", type=str, default=str(_ROOT / "configs"))
    parser.add_argument("--sw_index", type=str, default=str(_ROOT / "results/super_weight_index.json"))
    parser.add_argument("--uk_path", type=str, default=str(_ROOT / "results/sw_mechanistic_dnabert2.json"))
    parser.add_argument("--fracs", type=float, nargs="+", default=[10.0, 20.0, 30.0])
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    task = args.task

    cfg_path = Path(args.configs_dir) / "dnabert2.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    with open(args.sw_index) as f:
        sw_list = json.load(f)["dnabert2"]["results"]
    print(f"Loaded {len(sw_list)} SW rows")

    if args.ckpt_dir is None:
        task_leaf = task.split("/")[-1]  # same leaf convention as run_gue_ablation.py
        ckpt_dir = f"results/gue_checkpoints/dnabert2_{task_leaf}"
    else:
        ckpt_dir = args.ckpt_dir
    ckpt_path = Path(_ROOT) / ckpt_dir / "model_state.pt"
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}"); sys.exit(1)

    train_csv = Path(args.gue_root) / task / "train.csv"
    with open(train_csv) as f:
        rows = list(_csv.reader(f))[1:]
    num_labels = len({r[-1] for r in rows})

    import transformers as _tf
    _tf.logging.set_verbosity_error()
    DNABERT2_REV = "7bce263b15377fc15361f52cfab88f8b586abda0"
    tokenizer = _tf.AutoTokenizer.from_pretrained(cfg["model_id"], revision=DNABERT2_REV, trust_remote_code=True)
    backbone = _tf.AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"], revision=DNABERT2_REV, trust_remote_code=True,
        num_labels=num_labels, device_map={"": "cpu"})
    pattern = cfg["down_proj_pattern"]
    num_layers = cfg["num_layers"]
    num_rows = backbone.config.hidden_size
    model = backbone.to(device)

    task_key = _task_key(task)
    max_len = _MAX_LEN.get(task_key, 128)
    test_csv = Path(args.gue_root) / task / "test.csv"
    if not test_csv.exists():
        print(f"ERROR: test CSV not found at {test_csv}"); sys.exit(1)
    test_ds = GUEDataset(str(test_csv), tokenizer, max_len)
    print(f"Test size: {len(test_ds)}")

    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print("Checkpoint loaded.")

    uk_by_layer = _load_uk_values(Path(args.uk_path)) if Path(args.uk_path).exists() else None
    if uk_by_layer:
        print(f"U_k values loaded for {len(uk_by_layer)} layers")

    out_path = args.out or f"results/int4_multicondition_{task.replace('/', '_')}.json"
    out_full = Path(_ROOT) / out_path
    out_full.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"INT4 Multi-Condition Benchmark: {task} | GPU {args.gpu}")
    print(f"{'='*60}\n")

    results = run_extended_sweep(
        model=model, sw_list=sw_list, test_ds=test_ds, pattern=pattern,
        num_layers=num_layers, num_rows=num_rows, fracs=args.fracs,
        n_seeds=args.n_seeds, device=device, bits=args.bits, uk_by_layer=uk_by_layer)
    results["task"] = task

    json.dump(results, open(out_full, "w"), indent=2)
    print(f"\nSaved: {out_full}")


if __name__ == "__main__":
    main()
