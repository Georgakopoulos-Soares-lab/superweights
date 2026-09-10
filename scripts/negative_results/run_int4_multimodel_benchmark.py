"""
scripts/negative_results/run_int4_multimodel_benchmark.py
------------------------------------------------------
Unified INT2 downstream benchmark across DNABERT-2, NTv3, and GENERator.

Conditions: near-SW, far-SW, random, U_k-low, U_k-high, Yu-all, naive, SW-only
Fractions: 50% (configurable)
Seeds: 5 for stochastic conditions

Usage:
    # DNABERT-2 on GUE tasks
    python scripts/negative_results/run_int4_multimodel_benchmark.py \\
        --model dnabert2 --task EMP/H3K4me3 --bits 2 --fracs 50.0

    # NTv3 on splice (uses multi-seed checkpoints)
    python scripts/negative_results/run_int4_multimodel_benchmark.py \\
        --model ntv3 --task splice/reconstructed --bits 2 --fracs 50.0

    # GENERator EUK perplexity
    python scripts/negative_results/run_int4_multimodel_benchmark.py \\
        --model generator --bits 2 --fracs 50.0
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import random as _random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(_ROOT / "scripts" / "compression"))

# Shared helpers
from run_gue_ablation import GUEDataset, evaluate, _task_key, _MAX_LEN, collate_fn
from run_compression_sweep import _all_candidates, _rank_proximity
from run_quantization_ablation import (
    _quantize_row_rtn, _apply_quantization, _restore_quantization,
    _resolve_module,
    _generator_perplexity, PROBE_SEQS,
)

import run_gue_ablation as _rga
import run_compression_sweep as _rcs

# Patch resolve_module for classifier wrappers
def _resolve_patched(model, pattern, layer_idx):
    path = pattern.replace("{i}", str(layer_idx))
    obj = model.backbone if hasattr(model, "backbone") else model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj
_rga._resolve_module = _resolve_patched
_rcs._resolve_module = _resolve_patched


# ═══════════════════════════════════════════════════════════════════
# U_k ranking
# ═══════════════════════════════════════════════════════════════════

def _load_uk_values(model_name: str) -> dict | None:
    """Load U_k Frobenius norms from pre-computed data file."""
    fname_map = {
        "dnabert2": "results/sw_mechanistic_dnabert2.json",
        "ntv3": "results/sw_mechanistic_ntv3.json",
    }
    fname = fname_map.get(model_name)
    if fname is None:
        return None
    path = _ROOT / fname
    if not path.exists():
        return None
    data = json.load(open(path))
    return {int(k): v for k, v in data.get("frob_norm_uk_by_layer", {}).items()}


def _compute_generator_uk(model, pattern: str, sw_list: list,
                          num_layers: int, num_rows: int) -> dict:
    """Compute U_k for all rows in GENERator from weights.
    
    U_k = sqrt(sum_i W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2)
    For GENERator SwiGLU: gate=down_proj, gate_proj, up_proj
    """
    uk_by_layer = {}
    for li in range(num_layers):
        down_m = _resolve_patched(model, pattern, li)
        # GENERator: gate_proj, up_proj are in the same module
        block = model.model.layers[li].mlp
        try:
            gate_w = block.gate_proj.weight.data.float()
            up_w = block.up_proj.weight.data.float()
        except AttributeError:
            uk_by_layer[str(li)] = [0.0] * num_rows
            continue
        
        down_w = down_m.weight.data.float()
        # Compute gate/up norms per hidden unit
        gate_norms = torch.norm(gate_w, dim=1)  # (d_ffn,)
        up_norms = torch.norm(up_w, dim=1)      # (d_ffn,)
        
        # U_k for each output row k
        uk = []
        for k in range(num_rows):
            term = (down_w[k, :] ** 2) * (gate_norms ** 2) * (up_norms ** 2)
            uk.append(torch.sqrt(term.sum()).item())
        uk_by_layer[str(li)] = uk
    return uk_by_layer


def _rank_by_uk(uk_by_layer: dict, candidates: list[tuple],
                num_layers: int, num_rows: int,
                ascending: bool = True) -> list[tuple]:
    """Rank (layer, row) candidates by U_k.
    ascending=True  → lowest U_k first (quantize low-U_k rows)
    ascending=False → highest U_k first (quantize high-U_k rows)
    """
    uk_flat = []
    for li in range(num_layers):
        layer_uk = uk_by_layer.get(str(li), uk_by_layer.get(li, [0.0] * num_rows))
        for ri in range(num_rows):
            uk_flat.append(layer_uk[ri] if ri < len(layer_uk) else 0.0)
    scored = []
    for li, ri in candidates:
        val = uk_flat[li * num_rows + ri] if li * num_rows + ri < len(uk_flat) else 0.0
        scored.append(((li, ri), val))
    scored.sort(key=lambda x: x[1], reverse=not ascending)
    return [item[0] for item in scored]


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

def _load_model(model_name: str, task: str, device: str, gue_root: str):
    """Load model and return (model, test_ds, tokenizer, config info)."""
    cfg_path = _ROOT / "configs" / f"{model_name}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    pattern = cfg["down_proj_pattern"]
    num_layers = cfg["num_layers"]
    
    import transformers as _tf
    _tf.logging.set_verbosity_error()
    
    if model_name == "dnabert2":
        DNABERT2_REV = "7bce263b15377fc15361f52cfab88f8b586abda0"
        tokenizer = _tf.AutoTokenizer.from_pretrained(
            cfg["model_id"], revision=DNABERT2_REV, trust_remote_code=True)
        backbone = _tf.AutoModelForSequenceClassification.from_pretrained(
            cfg["model_id"], revision=DNABERT2_REV, trust_remote_code=True,
            num_labels=2, device_map={"": "cpu"})
        num_rows = backbone.config.hidden_size
        model = backbone.to(device)
        
        # Load checkpoint and test dataset
        task_leaf = task.split("/")[-1]
        ckpt_dir = _ROOT / "results" / "gue_checkpoints" / f"dnabert2_{task_leaf}"
        ckpt_path = ckpt_dir / "model_state.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        train_csv = Path(gue_root) / task / "train.csv"
        with open(train_csv) as f:
            rows = list(_csv.reader(f))[1:]
        num_labels = len({r[-1] for r in rows})
        
        # Reload with correct num_labels
        model = _tf.AutoModelForSequenceClassification.from_pretrained(
            cfg["model_id"], revision=DNABERT2_REV, trust_remote_code=True,
            num_labels=num_labels, device_map={"": "cpu"})
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        model = model.to(device).eval()
        
        task_key = _task_key(task)
        max_len = _MAX_LEN.get(task_key, 128)
        test_csv = Path(gue_root) / task / "test.csv"
        test_ds = GUEDataset(str(test_csv), tokenizer, max_len)
        return model, test_ds, tokenizer, pattern, num_layers, num_rows
    
    elif model_name == "ntv3":
        tokenizer = _tf.AutoTokenizer.from_pretrained(
            cfg["model_id"], trust_remote_code=True)
        backbone = _tf.AutoModelForMaskedLM.from_pretrained(
            cfg["model_id"], trust_remote_code=True).to(device)
        num_rows = backbone.config.embed_dim  # NTv3 uses embed_dim
        
        # Load NTv3 classifier and checkpoint
        task_leaf = task.split("/")[-1]
        ckpt_path = _ROOT / "results" / "gue_checkpoints_multiseed" / f"ntv3_{task_leaf}" / "seed_0" / "model_state.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        # NTv3 splice has 3 classes
        ntv3_labels = 3 if "splice" in task else 2
        from run_quantization_ablation import _NTv3Classifier
        model = _NTv3Classifier(backbone, num_rows, ntv3_labels)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()
        
        task_key = _task_key(task)
        max_len = _MAX_LEN.get(task_key, 128)
        test_csv = Path(gue_root) / task / "test.csv"
        test_ds = GUEDataset(str(test_csv), tokenizer, max_len)
        return model, test_ds, tokenizer, pattern, num_layers, num_rows
    
    elif model_name in ("generator", "generator_prokaryote"):
        sys.path.insert(0, str(_ROOT))
        from models.generator_wrapper import GeneratorWrapper
        wrapper = GeneratorWrapper(cfg)
        wrapper.load()
        model = wrapper.model.eval()
        tokenizer = wrapper.tokenizer
        num_rows = model.config.hidden_size
        return model, None, tokenizer, pattern, num_layers, num_rows
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def _eval_gue(model, test_ds, device: str) -> dict:
    """Evaluate on GUE downstream task."""
    return evaluate(model, test_ds, device=device)


def _eval_ppl(model, tokenizer, sequences: list[str]) -> float:
    """Evaluate perplexity for GENERator."""
    return _generator_perplexity(model, tokenizer, sequences)


# ═══════════════════════════════════════════════════════════════════
# Condition runner
# ═══════════════════════════════════════════════════════════════════

def run_condition_gue(label: str, model, rows_to_quant: list, pattern: str,
                      test_ds, device: str, bits: int = 4,
                      baseline: dict | None = None) -> dict:
    if rows_to_quant:
        saves = _apply_quantization(model, pattern, rows_to_quant, bits=bits)
    metrics = _eval_gue(model, test_ds, device=device)
    if rows_to_quant:
        _restore_quantization(model, pattern, saves)
    result = {"label": label, "n_quantized_rows": len(rows_to_quant), **metrics}
    if baseline is not None:
        result["delta_acc"] = round(metrics["accuracy"] - baseline["accuracy"], 6)
        result["delta_mcc"] = round(metrics["mcc"] - baseline["mcc"], 6)
    print(f"  [{label:<22}]  acc={metrics['accuracy']:.4f}  mcc={metrics['mcc']:.4f}"
          f"  Δ={result.get('delta_acc', 0):+.4f}")
    return result


def run_condition_ppl(label: str, model, tokenizer, rows_to_quant: list,
                      pattern: str, sequences: list[str], device: str,
                      bits: int = 4, baseline_ppl: float | None = None) -> dict:
    if rows_to_quant:
        saves = _apply_quantization(model, pattern, rows_to_quant, bits=bits)
    ppl = _eval_ppl(model, tokenizer, sequences)
    if rows_to_quant:
        _restore_quantization(model, pattern, saves)
    result = {"label": label, "n_quantized_rows": len(rows_to_quant), "ppl": round(ppl, 6)}
    if baseline_ppl is not None:
        result["delta_ppl"] = round(ppl - baseline_ppl, 6)
    print(f"  [{label:<22}]  ppl={ppl:.4f}  Δ={result.get('delta_ppl', 0):+.4f}")
    return result


# ═══════════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Unified INT2 multi-model benchmark")
    parser.add_argument("--model", type=str, required=True,
                        choices=["dnabert2", "ntv3", "generator", "generator_prokaryote"])
    parser.add_argument("--task", type=str, default=None,
                        help="GUE task (required for dnabert2/ntv3)")
    parser.add_argument("--gue_root", type=str, default="/home/nvidia/data/gue/GUE")
    parser.add_argument("--configs_dir", type=str, default=str(_ROOT / "configs"))
    parser.add_argument("--sw_index", type=str, default=str(_ROOT / "results/super_weight_index.json"))
    parser.add_argument("--fracs", type=float, nargs="+", default=[50.0])
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    model_name = args.model
    is_generator = model_name in ("generator", "generator_prokaryote")
    
    # ── Load model ──
    print(f"\nLoading {model_name} ...")
    model, test_ds, tokenizer, pattern, num_layers, num_rows = _load_model(
        model_name, args.task, device, args.gue_root)
    print(f"  {num_layers} layers, {num_rows} rows per layer")
    
    # ── Load SW index ──
    with open(args.sw_index) as f:
        sw_data = json.load(f)
    entry = sw_data.get(model_name, sw_data.get("ntv3" if model_name == "ntv3" else None))
    if isinstance(entry, dict):
        sw_list = entry["results"]
    elif isinstance(entry, list):
        sw_list = entry
    else:
        sw_list = []
    print(f"  SW rows: {len(sw_list)}")
    
    sw_coords = {(s["layer"], s["row"]) for s in sw_list}
    sw_rows_list = [(s["layer"], s["row"]) for s in sw_list]
    
    all_rows = [(li, ri) for li in range(num_layers) for ri in range(num_rows)]
    candidates = _all_candidates(model, pattern, num_layers, sw_coords)
    n_candidates = len(candidates)
    print(f"  Total rows: {len(all_rows)}  Candidates: {n_candidates}")
    
    # ── Load U_k ──
    uk_by_layer = _load_uk_values(model_name)
    if uk_by_layer is None and is_generator:
        print("  Computing U_k from GENERator weights ...")
        uk_by_layer = _compute_generator_uk(model, pattern, sw_list, num_layers, num_rows)
        print(f"  U_k computed for {len(uk_by_layer)} layers")
    elif uk_by_layer is None:
        print("  WARNING: U_k data not available — skipping U_k-guided conditions")
    
    # ── Rankings ──
    print("  Computing rankings ...")
    ranked_near = _rank_proximity(model, pattern, candidates, sw_list,
                                  num_layers, num_rows, ascending=False)
    ranked_far = _rank_proximity(model, pattern, candidates, sw_list,
                                 num_layers, num_rows, ascending=True)
    ranked_uk_low = None
    ranked_uk_high = None
    if uk_by_layer is not None:
        ranked_uk_low = _rank_by_uk(uk_by_layer, candidates, num_layers, num_rows, True)
        ranked_uk_high = _rank_by_uk(uk_by_layer, candidates, num_layers, num_rows, False)
        print("  near-SW, far-SW, U_k-low, U_k-high ready")
    else:
        print("  near-SW, far-SW ready")
    
    # ── Evaluation function ──
    if is_generator:
        run_cond = lambda label, model, rows, pattern, ds, dev, bits, bl: \
            run_condition_ppl(label, model, tokenizer, rows, pattern, PROBE_SEQS, dev, bits, bl)
        eval_baseline = lambda: _eval_ppl(model, tokenizer, PROBE_SEQS)
    else:
        run_cond = lambda label, model, rows, pattern, ds, dev, bits, bl: \
            run_condition_gue(label, model, rows, pattern, ds, dev, bits, bl)
        eval_baseline = lambda: _eval_gue(model, test_ds, device)
    
    # ── Baseline ──
    print("\n─── Baseline ───")
    baseline = eval_baseline()
    if isinstance(baseline, dict):
        bl_val = baseline.get("accuracy", baseline.get("mcc", 0))
        print(f"  Baseline: acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")
        bl_ppl = None
    else:
        bl_val = baseline
        bl_ppl = baseline
        print(f"  Baseline PPL: {baseline:.4f}")
    
    # ── Yu-all and naive ──
    print("\n─── Full conditions ───")
    yu_all = run_cond("yu_all_int4", model, candidates, pattern, test_ds, device, args.bits,
                      bl_ppl if is_generator else baseline)
    naive = run_cond("naive_int4", model, all_rows, pattern, test_ds, device, args.bits,
                     bl_ppl if is_generator else baseline)
    sw_only = run_cond("sw_only_int4", model, sw_rows_list, pattern, test_ds, device, args.bits,
                       bl_ppl if is_generator else baseline)
    
    # ── Per-fraction sweep ──
    results = {
        "model": model_name,
        "bits": args.bits,
        "fracs": args.fracs,
        "n_seeds": args.n_seeds,
        "n_sw": len(sw_rows_list),
        "n_total": len(all_rows),
        "n_candidates": n_candidates,
        "baseline": baseline if isinstance(baseline, dict) else {"ppl": baseline},
        "yu_all_int4": yu_all,
        "naive_int4": naive,
        "sw_only_int4": sw_only,
        "per_frac": [],
    }
    if args.task:
        results["task"] = args.task
    
    for frac in args.fracs:
        n_select = max(1, int(round(n_candidates * frac / 100.0)))
        print(f"\n─── frac={frac:.0f}%  n={n_select} ───")
        fr = {"frac": frac, "n_rows": n_select}
        
        # near_sw
        chosen = ranked_near[:n_select]
        fr["near_sw"] = run_cond(f"near_sw_{frac:.0f}pct", model, chosen, pattern,
                                 test_ds, device, args.bits, bl_ppl if is_generator else baseline)
        
        # far_sw
        chosen = ranked_far[:n_select]
        fr["far_sw"] = run_cond(f"far_sw_{frac:.0f}pct", model, chosen, pattern,
                                test_ds, device, args.bits, bl_ppl if is_generator else baseline)
        
        # random
        rand_results = []
        for seed in range(args.n_seeds):
            rng = _random.Random(seed)
            chosen = rng.sample(candidates, n_select)
            r = run_cond(f"random_seed{seed}", model, chosen, pattern,
                         test_ds, device, args.bits, bl_ppl if is_generator else baseline)
            rand_results.append(r)
        
        if is_generator:
            ppls = [r["ppl"] for r in rand_results]
            fr["random"] = {
                "n_seeds": args.n_seeds,
                "ppl_mean": float(np.mean(ppls)),
                "ppl_std": float(np.std(ppls)),
                "delta_ppl_mean": float(np.mean(ppls)) - bl_ppl,
            }
        else:
            accs = [r["accuracy"] for r in rand_results]
            mccs = [r["mcc"] for r in rand_results]
            fr["random"] = {
                "n_seeds": args.n_seeds,
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "mcc_mean": float(np.mean(mccs)),
                "mcc_std": float(np.std(mccs)),
                "delta_acc_mean": float(np.mean(accs)) - baseline["accuracy"],
                "delta_mcc_mean": float(np.mean(mccs)) - baseline["mcc"],
            }
        print(f"  [random (n={args.n_seeds})]  "
              f"{'ppl' if is_generator else 'acc'}={fr['random']['ppl_mean' if is_generator else 'accuracy_mean']:.4f}"
              f"±{fr['random']['ppl_std' if is_generator else 'accuracy_std']:.4f}")
        
        # U_k-low
        if ranked_uk_low is not None:
            uk_results = []
            for seed in range(args.n_seeds):
                chosen = ranked_uk_low[:n_select]
                r = run_cond(f"uk_low_seed{seed}", model, chosen, pattern,
                             test_ds, device, args.bits, bl_ppl if is_generator else baseline)
                uk_results.append(r)
            if is_generator:
                ppls = [r["ppl"] for r in uk_results]
                fr["uk_low"] = {
                    "n_seeds": args.n_seeds,
                    "ppl_mean": float(np.mean(ppls)),
                    "ppl_std": float(np.std(ppls)),
                    "delta_ppl_mean": float(np.mean(ppls)) - bl_ppl,
                }
            else:
                accs = [r["accuracy"] for r in uk_results]
                mccs = [r["mcc"] for r in uk_results]
                fr["uk_low"] = {
                    "n_seeds": args.n_seeds,
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs)),
                    "mcc_mean": float(np.mean(mccs)),
                    "mcc_std": float(np.std(mccs)),
                    "delta_acc_mean": float(np.mean(accs)) - baseline["accuracy"],
                    "delta_mcc_mean": float(np.mean(mccs)) - baseline["mcc"],
                }
            print(f"  [uk_low (n={args.n_seeds})]  "
                  f"{'ppl' if is_generator else 'acc'}={fr['uk_low']['ppl_mean' if is_generator else 'accuracy_mean']:.4f}")
        
        # U_k-high
        if ranked_uk_high is not None:
            ukh_results = []
            for seed in range(args.n_seeds):
                chosen = ranked_uk_high[:n_select]
                r = run_cond(f"uk_high_seed{seed}", model, chosen, pattern,
                             test_ds, device, args.bits, bl_ppl if is_generator else baseline)
                ukh_results.append(r)
            if is_generator:
                ppls = [r["ppl"] for r in ukh_results]
                fr["uk_high"] = {
                    "n_seeds": args.n_seeds,
                    "ppl_mean": float(np.mean(ppls)),
                    "ppl_std": float(np.std(ppls)),
                    "delta_ppl_mean": float(np.mean(ppls)) - bl_ppl,
                }
            else:
                accs = [r["accuracy"] for r in ukh_results]
                mccs = [r["mcc"] for r in ukh_results]
                fr["uk_high"] = {
                    "n_seeds": args.n_seeds,
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs)),
                    "mcc_mean": float(np.mean(mccs)),
                    "mcc_std": float(np.std(mccs)),
                    "delta_acc_mean": float(np.mean(accs)) - baseline["accuracy"],
                    "delta_mcc_mean": float(np.mean(mccs)) - baseline["mcc"],
                }
            print(f"  [uk_high (n={args.n_seeds})]  "
                  f"{'ppl' if is_generator else 'acc'}={fr['uk_high']['ppl_mean' if is_generator else 'accuracy_mean']:.4f}")
        
        results["per_frac"].append(fr)
    
    # ── Save ──
    out_path = args.out or f"results/int2_multimodel_{model_name}"
    if args.task:
        out_path += f"_{args.task.replace('/', '_')}"
    out_path += ".json"
    out_full = _ROOT / out_path
    out_full.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_full, "w"), indent=2)
    print(f"\nSaved: {out_full}")


if __name__ == "__main__":
    main()
