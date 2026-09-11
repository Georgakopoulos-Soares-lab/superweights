"""
scripts/run_quantization_ablation.py
--------------------------------------
RTN INT8 quantization ablation on `mlp.wo` rows for DNABERT-2 and
GENERator/NTv3 models.

Overview
--------
We simulate INT8 weight quantization using round-to-nearest (RTN):
    quantize(row) = round(row / scale) * scale
    scale = max(|row|) / 127

This precision-loss simulation is applied selectively to different subsets of
`mlp.wo` rows to test three hypotheses from Yu et al. (2024) adapted to genomic LMs:

  1. baseline     : no quantization.
  2. yu_all       : quantize ALL rows EXCEPT the detected super rows → SW-exempt
                    quantization as proposed by Yu et al. for LLMs.
  3. sw_fragility : quantize ONLY the SW rows themselves vs. an equal count of
                    random non-SW rows. Tests whether SWs are uniquely sensitive
                    to precision loss (Option 2 experiment).
  4. near_sw      : quantize rows CLOSEST to SW coordinates (most redundant per
                    compression sweep) — swept over increasing fracs.
  5. random       : quantize same count of random non-SW rows (10 seeds, ±std).

Use --bits 8 (default, INT8) or --bits 4 (INT4) to control quantization strength.

For DNABERT-2: evaluate on GUE downstream tasks (uses fine-tuned checkpoint).
For GENERator: evaluate via masked-token entropy / perplexity (no GUE ckpt needed).

Usage (DNABERT-2, GUE task):
    python scripts/run_quantization_ablation.py \\
        --model dnabert2 \\
        --task prom/prom_core_notata \\
        --gue_root /home/nvidia/data/gue/GUE \\
        --ckpt_dir results/gue_checkpoints/dnabert2_prom_core_notata \\
        --out results/quant_ablation_dnabert2_prom_core_notata.json

Usage (GENERator, perplexity):
    python scripts/run_quantization_ablation.py \\
        --model generator \\
        --out results/quant_ablation_generator.json

Optional:
    --fracs 1 5 10 20 30   # quantization fractions for near_sw / random (pct of non-SW rows)
    --n_rand_seeds 10       # seeds for random condition
    --sw_index results/negative_results/super_weight_index.json
    --device cuda
"""

import argparse
import json
import random as _random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Re-use helpers from run_gue_ablation / run_compression_sweep ─────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
# run_gue_ablation lives in scripts/evaluation/, not beside this file
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from run_gue_ablation import (
    GUEDataset,
    _NTv3Classifier,
    _HybriDNAClassifier,
    evaluate,
    _task_key,
    _MAX_LEN,
    collate_fn,
)
import run_gue_ablation as _rga
import run_compression_sweep as _rcs
from run_compression_sweep import (
    _all_candidates,
    _rank_proximity,
)

# Patch _resolve_module so both imported modules handle any wrapper that stores
# the backbone as self.backbone (including our local _DNABERT2Classifier).
def _resolve_module(model, pattern: str, layer_idx: int):
    path = pattern.replace("{i}", str(layer_idx))
    obj  = model.backbone if hasattr(model, "backbone") else model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj

# Apply patch into both imported modules so their functions pick it up
_rga._resolve_module = _resolve_module
_rcs._resolve_module = _resolve_module

# ─────────────────────────────────────────────────────────────────────────────
# RTN INT8 quantization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quantize_row_rtn(row: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Simulate per-row round-to-nearest quantization (INT8 or INT4).

    Returns a new tensor with the same shape and dtype as `row` but with values
    rounded to the symmetric integer grid:  scale = max(|row|) / maxval,
    where maxval = 2^(bits-1) - 1  (127 for INT8, 7 for INT4).
    """
    maxval = 2 ** (bits - 1) - 1
    amax = row.abs().max().item()
    if amax == 0.0:
        return row.clone()
    scale = amax / maxval
    q = torch.round(row.float() / scale).clamp(-maxval, maxval)
    return (q * scale).to(row.dtype)


def _apply_quantization(model, pattern: str, rows_to_quantize: list,
                        bits: int = 8) -> list:
    """Quantize selected (layer, row) pairs in-place.

    Returns list of (layer, row, original_tensor) for restoration.
    Uses the local _resolve_module so all classifier wrappers are handled.
    """
    saves = []
    with torch.no_grad():
        for li, ri in rows_to_quantize:
            m = _resolve_module(model, pattern, li)
            orig = m.weight.data[ri, :].clone()
            saves.append((li, ri, orig))
            m.weight.data[ri, :] = _quantize_row_rtn(orig, bits=bits)
    return saves


def _restore_quantization(model, pattern: str, saves: list):
    with torch.no_grad():
        for li, ri, orig in saves:
            m = _resolve_module(model, pattern, li)
            m.weight.data[ri, :] = orig


# ─────────────────────────────────────────────────────────────────────────────
# Proximity ranking (re-use from run_compression_sweep via import above, but
# also need sw_coords exclusion that _rank_proximity already does via candidates)
# ─────────────────────────────────────────────────────────────────────────────

def _rank_prox_near(model, pattern, candidates, sw_list, num_layers, num_rows):
    """Return candidates sorted closest-to-SW-first (ascending=False)."""
    return _rank_proximity(
        model, pattern, candidates, sw_list,
        num_layers, num_rows, ascending=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GUE quantization sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_gue_quant_sweep(
    model,
    sw_list: list,
    test_ds,
    pattern: str,
    num_layers: int,
    num_rows: int,
    fracs: list[float],
    n_rand_seeds: int,
    device: str,
    bits: int = 8,
) -> dict:
    """Run quantization conditions over increasing fractions of rows, plus SW fragility.

    Conditions: baseline, yu_all, sw_fragility, near_sw, random (fracs sweep).
    Returns nested dict with per-fraction results.
    """
    sw_coords  = {(sw["layer"], sw["row"]) for sw in sw_list}
    candidates = _all_candidates(model, pattern, num_layers, sw_coords)
    n_total    = len(candidates)

    baseline = evaluate(model, test_ds, device=device)
    print(f"  Baseline: acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")
    print(f"  Candidate pool: {n_total} rows  (total {num_layers*num_rows} − {len(sw_coords)} SW)")
    print(f"  Quantization: INT{bits}")

    # yu_all: quantize ALL candidates (everything except SW)
    print("  Running yu_all (quantize all non-SW rows) ...", flush=True)
    saves_all = _apply_quantization(model, pattern, candidates, bits=bits)
    yu_all_m  = evaluate(model, test_ds, device=device)
    print(f"    yu_all: acc={yu_all_m['accuracy']:.4f}  mcc={yu_all_m['mcc']:.4f}")
    _restore_quantization(model, pattern, saves_all)

    # ── sw_fragility: quantize ONLY the SW rows themselves ──────────────────
    print(f"  Running sw_fragility (quantize {len(sw_list)} SW rows) ...", flush=True)
    sw_rows = [(sw["layer"], sw["row"]) for sw in sw_list]
    saves_sw = _apply_quantization(model, pattern, sw_rows, bits=bits)
    m_sw_frag = evaluate(model, test_ds, device=device)
    _restore_quantization(model, pattern, saves_sw)
    delta_sw = m_sw_frag["accuracy"] - baseline["accuracy"]
    print(f"    sw_only: acc={m_sw_frag['accuracy']:.4f}  Δacc={delta_sw:+.4f}")
    rand_accs_sw, rand_mccs_sw = [], []
    for seed in range(n_rand_seeds):
        chosen_sw_rand = _random.Random(seed).sample(candidates, len(sw_rows))
        saves = _apply_quantization(model, pattern, chosen_sw_rand, bits=bits)
        m_r = evaluate(model, test_ds, device=device)
        _restore_quantization(model, pattern, saves)
        rand_accs_sw.append(m_r["accuracy"])
        rand_mccs_sw.append(m_r["mcc"])
    mean_sw_rand = float(np.mean(rand_accs_sw))
    std_sw_rand  = float(np.std(rand_accs_sw))
    print(f"    random({len(sw_rows)} rows): acc={mean_sw_rand:.4f}±{std_sw_rand:.4f}  "
          f"Δacc={mean_sw_rand - baseline['accuracy']:+.4f}")

    # Pre-rank near_sw order once
    print("  Ranking by proximity to super rows ...", flush=True)
    ranked_near = _rank_prox_near(model, pattern, candidates, sw_list,
                                  num_layers, num_rows)

    results = {
        "baseline": baseline,
        "yu_all":   yu_all_m,
        "sw_fragility": {
            "n_sw":                  len(sw_rows),
            "sw_acc":                m_sw_frag["accuracy"],
            "sw_mcc":                m_sw_frag["mcc"],
            "sw_delta_acc":          delta_sw,
            "sw_delta_mcc":          m_sw_frag["mcc"] - baseline["mcc"],
            "random_acc_mean":        mean_sw_rand,
            "random_acc_std":         std_sw_rand,
            "random_delta_acc_mean":  mean_sw_rand - baseline["accuracy"],
        },
        "near_sw":  [],
        "random":   [],
    }

    rng = _random.Random(0)

    for frac in fracs:
        n_prune = max(1, int(round(n_total * frac / 100.0)))
        print(f"\n  --- frac={frac}%  n_prune={n_prune} ---")

        # ── near_sw ────────────────────────────────────────────────────────
        chosen_near = ranked_near[:n_prune]
        saves = _apply_quantization(model, pattern, chosen_near, bits=bits)
        m_near = evaluate(model, test_ds, device=device)
        _restore_quantization(model, pattern, saves)
        delta_near = m_near["accuracy"] - baseline["accuracy"]
        print(f"    near_sw:  acc={m_near['accuracy']:.4f}  Δacc={delta_near:+.4f}")
        results["near_sw"].append({
            "frac": frac, "n_rows": n_prune,
            **{k: v for k, v in m_near.items()},
            "delta_acc": delta_near,
            "delta_mcc": m_near["mcc"] - baseline["mcc"],
        })

        # ── random seeds ───────────────────────────────────────────────────
        rand_accs, rand_mccs = [], []
        for seed in range(n_rand_seeds):
            rng = _random.Random(seed)
            chosen_rand = rng.sample(candidates, n_prune)
            saves = _apply_quantization(model, pattern, chosen_rand, bits=bits)
            m_rand = evaluate(model, test_ds, device=device)
            _restore_quantization(model, pattern, saves)
            rand_accs.append(m_rand["accuracy"])
            rand_mccs.append(m_rand["mcc"])
        mean_acc = float(np.mean(rand_accs))
        std_acc  = float(np.std(rand_accs))
        print(f"    random:   acc={mean_acc:.4f}±{std_acc:.4f}  "
              f"Δacc={mean_acc - baseline['accuracy']:+.4f}")
        results["random"].append({
            "frac": frac, "n_rows": n_prune,
            "accuracy_mean": mean_acc, "accuracy_std": std_acc,
            "mcc_mean": float(np.mean(rand_mccs)), "mcc_std": float(np.std(rand_mccs)),
            "delta_acc_mean": mean_acc - baseline["accuracy"],
            "delta_mcc_mean": float(np.mean(rand_mccs)) - baseline["mcc"],
            "seeds": n_rand_seeds,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# GENERator perplexity-based quantization ablation
# ─────────────────────────────────────────────────────────────────────────────

PROBE_SEQS = [
    "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC",
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG",
    "AAAATTTTCCCCGGGGAAAATTTTCCCCGGGGAAAATTTTCCCCGGGG",
    "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    "GTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC",
    "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT",
]


def _generator_perplexity(model, tokenizer, sequences: list[str]) -> float:
    """Mean per-token cross-entropy loss over sequences."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for seq in sequences:
            remainder = len(seq) % 6
            if remainder:
                seq = seq[remainder:]
            if not seq:
                continue
            bos = tokenizer.bos_token or ""
            seq = bos + seq
            inputs = tokenizer(seq, return_tensors="pt",
                               add_special_tokens=False).to(next(model.parameters()).device)
            ids = inputs["input_ids"]
            if ids.shape[1] < 2:
                continue
            out = model(input_ids=ids, labels=ids)
            total_loss   += out.loss.item() * (ids.shape[1] - 1)
            total_tokens += ids.shape[1] - 1
    return total_loss / max(total_tokens, 1)


def run_generator_quant_ablation(
    model,
    tokenizer,
    sw_list: list,
    pattern: str,
    num_layers: int,
    num_rows: int,
    fracs: list[float],
    n_rand_seeds: int,
    sequences: list[str],
    bits: int = 8,
) -> dict:
    """Quantization ablation using perplexity for causal LM (GENERator).

    Conditions: baseline, yu_all, sw_fragility, near_sw (fracs sweep), random (fracs sweep).
    """
    device = next(model.parameters()).device
    sw_coords  = {(sw["layer"], sw["row"]) for sw in sw_list}

    # Build candidates (all rows except SW)
    candidates = []
    for li in range(num_layers):
        m = _resolve_module(model, pattern, li)
        nrows = m.weight.data.shape[0]
        for ri in range(nrows):
            if (li, ri) not in sw_coords:
                candidates.append((li, ri))
    n_total = len(candidates)

    baseline_ppl = _generator_perplexity(model, tokenizer, sequences)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")
    print(f"  Candidate pool: {n_total} rows")
    print(f"  Quantization: INT{bits}")

    # yu_all
    print("  Running yu_all (quantize all non-SW rows) ...", flush=True)
    saves_all = _apply_quantization(model, pattern, candidates, bits=bits)
    yu_all_ppl = _generator_perplexity(model, tokenizer, sequences)
    print(f"    yu_all PPL: {yu_all_ppl:.4f}  Δ={yu_all_ppl - baseline_ppl:+.4f}")
    _restore_quantization(model, pattern, saves_all)

    # sw_fragility: quantize ONLY the SW rows themselves
    print(f"  Running sw_fragility (quantize {len(sw_list)} SW rows) ...", flush=True)
    sw_rows = [(sw["layer"], sw["row"]) for sw in sw_list]
    saves_sw = _apply_quantization(model, pattern, sw_rows, bits=bits)
    ppl_sw_frag = _generator_perplexity(model, tokenizer, sequences)
    _restore_quantization(model, pattern, saves_sw)
    delta_sw_frag = ppl_sw_frag - baseline_ppl
    print(f"    sw_only PPL: {ppl_sw_frag:.4f}  Δ={delta_sw_frag:+.4f}")
    ppl_sw_rand = []
    for seed in range(n_rand_seeds):
        chosen = _random.Random(seed).sample(candidates, len(sw_rows))
        saves = _apply_quantization(model, pattern, chosen, bits=bits)
        ppl_r = _generator_perplexity(model, tokenizer, sequences)
        _restore_quantization(model, pattern, saves)
        ppl_sw_rand.append(ppl_r)
    mean_sw_rand = float(np.mean(ppl_sw_rand))
    std_sw_rand  = float(np.std(ppl_sw_rand))
    print(f"    random({len(sw_rows)} rows) PPL: {mean_sw_rand:.4f}±{std_sw_rand:.4f}  "
          f"Δ={mean_sw_rand - baseline_ppl:+.4f}")

    # Pre-rank proximity
    print("  Ranking by proximity ...", flush=True)
    ranked_near = _rank_prox_near(model, pattern, candidates, sw_list,
                                  num_layers, num_rows)

    results = {
        "baseline_ppl": baseline_ppl,
        "yu_all_ppl":   yu_all_ppl,
        "yu_all_delta": yu_all_ppl - baseline_ppl,
        "sw_fragility": {
            "n_sw":                   len(sw_rows),
            "sw_ppl":                 ppl_sw_frag,
            "sw_delta_ppl":           delta_sw_frag,
            "random_ppl_mean":        mean_sw_rand,
            "random_ppl_std":         std_sw_rand,
            "random_delta_ppl_mean":  mean_sw_rand - baseline_ppl,
        },
        "near_sw":      [],
        "random":       [],
    }

    for frac in fracs:
        n_prune = max(1, int(round(n_total * frac / 100.0)))
        print(f"\n  --- frac={frac}%  n_prune={n_prune} ---")

        # near_sw
        chosen_near = ranked_near[:n_prune]
        saves = _apply_quantization(model, pattern, chosen_near, bits=bits)
        ppl_near = _generator_perplexity(model, tokenizer, sequences)
        _restore_quantization(model, pattern, saves)
        print(f"    near_sw PPL: {ppl_near:.4f}  Δ={ppl_near - baseline_ppl:+.4f}")
        results["near_sw"].append({
            "frac": frac, "n_rows": n_prune,
            "ppl": ppl_near, "delta_ppl": ppl_near - baseline_ppl,
        })

        # random seeds
        ppl_randoms = []
        for seed in range(n_rand_seeds):
            rng = _random.Random(seed)
            chosen_rand = rng.sample(candidates, n_prune)
            saves = _apply_quantization(model, pattern, chosen_rand, bits=bits)
            ppl_rand = _generator_perplexity(model, tokenizer, sequences)
            _restore_quantization(model, pattern, saves)
            ppl_randoms.append(ppl_rand)
        mean_ppl = float(np.mean(ppl_randoms))
        std_ppl  = float(np.std(ppl_randoms))
        print(f"    random PPL: {mean_ppl:.4f}±{std_ppl:.4f}  Δ={mean_ppl - baseline_ppl:+.4f}")
        results["random"].append({
            "frac": frac, "n_rows": n_prune,
            "ppl_mean": mean_ppl, "ppl_std": std_ppl,
            "delta_ppl_mean": mean_ppl - baseline_ppl,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model + checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_and_tokenizer(model_name: str, config: dict, device: str):
    """Load backbone model. Returns (model, tokenizer, pattern, num_layers, num_rows)."""
    import transformers
    transformers.logging.set_verbosity_error()

    if model_name == "dnabert2":
        _DNABERT2_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config["model_id"],
            revision=_DNABERT2_REVISION,
            trust_remote_code=True,
        )
        # NOTE: load AutoModelForSequenceClassification so the internal structure
        # has self.bert = BertModel — matching the pattern "bert.encoder.layer.{i}.mlp.wo"
        # and matching the checkpoint format from run_gue_ablation.py.
        # num_labels will be updated when we load the checkpoint.
        backbone = transformers.AutoModelForSequenceClassification.from_pretrained(
            config["model_id"],
            revision=_DNABERT2_REVISION,
            trust_remote_code=True,
            num_labels=2,          # placeholder; overwritten by checkpoint load
            device_map={"": "cpu"},
        )
        pattern    = config["down_proj_pattern"]   # "bert.encoder.layer.{i}.mlp.wo"
        num_layers = config["num_layers"]
        num_rows   = backbone.config.hidden_size   # 768
        return backbone.to(device), tokenizer, pattern, num_layers, num_rows

    elif model_name == "ntv3":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config["model_id"], trust_remote_code=True
        )
        backbone  = transformers.AutoModelForMaskedLM.from_pretrained(
            config["model_id"], trust_remote_code=True
        ).to(device)
        pattern    = config["down_proj_pattern"]
        num_layers = config["num_layers"]
        num_rows   = backbone.config.hidden_size
        return backbone, tokenizer, pattern, num_layers, num_rows

    elif model_name in ("generator", "generator_prokaryote", "generator_prokaryote_1b"):
        # Use the project's own wrapper to avoid protobuf/tokenizer issues
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from models.generator_wrapper import GeneratorWrapper
        wrapper   = GeneratorWrapper(config)
        wrapper.load()
        model     = wrapper.model
        tokenizer = wrapper.tokenizer
        model.eval()
        pattern    = config["down_proj_pattern"]
        num_layers = config["num_layers"]
        num_rows   = model.config.hidden_size
        return model, tokenizer, pattern, num_layers, num_rows

    else:
        raise ValueError(f"Unknown model: {model_name}")


def _build_classifier(model_name: str, backbone, tokenizer, num_labels: int, device: str):
    """Wrap backbone in a classification head (only needed for NTv3)."""
    if model_name == "dnabert2":
        # backbone is already AutoModelForSequenceClassification — return as-is
        return backbone
    elif model_name == "ntv3":
        return _NTv3Classifier(backbone, backbone.config.hidden_size,
                               num_labels).to(device)
    else:
        raise ValueError(f"No classifier wrapper for {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot_results(results: dict, model_name: str, task: str, out_path: str,
                  bits: int = 8):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    baseline_acc = results["baseline"]["accuracy"]
    baseline_mcc = results["baseline"]["mcc"]

    near_fracs  = [r["frac"]       for r in results["near_sw"]]
    near_dA     = [r["delta_acc"]  for r in results["near_sw"]]
    near_dM     = [r["delta_mcc"]  for r in results["near_sw"]]

    rand_fracs  = [r["frac"]            for r in results["random"]]
    rand_dA_m   = [r["delta_acc_mean"]  for r in results["random"]]
    rand_dA_s   = [r["accuracy_std"]    for r in results["random"]]
    rand_dM_m   = [r["delta_mcc_mean"]  for r in results["random"]]

    yu_all_dA = results["yu_all"]["accuracy"] - baseline_acc
    yu_all_dM = results["yu_all"]["mcc"] - baseline_mcc
    sw_frag   = results.get("sw_fragility")

    for ax, near_d, rand_d_m, rand_d_s, yu_d, ylabel, title, \
            sw_d_key, sw_rand_d_key, sw_rand_std_key in [
        (ax1, near_dA, rand_dA_m, rand_dA_s, yu_all_dA, "Δ Accuracy", "Accuracy Change",
         "sw_delta_acc", "random_delta_acc_mean", "random_acc_std"),
        (ax2, near_dM, rand_dM_m, [0]*len(rand_dM_m), yu_all_dM, "Δ MCC", "MCC Change",
         "sw_delta_mcc", None, None),
    ]:
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.plot(near_fracs, near_d,   "o-", color="steelblue",
                label=f"near_sw (INT{bits})")
        ax.errorbar(rand_fracs, rand_d_m, yerr=rand_d_s,
                    fmt="s--", color="gray",
                    label=f"random (INT{bits}, mean±std)", capsize=3)
        ax.axhline(yu_d, color="darkorange", linestyle=":", linewidth=1.5,
                   label=f"yu_all INT{bits} ({yu_d:+.3f})")
        if sw_frag and sw_d_key in sw_frag:
            sw_d = sw_frag[sw_d_key]
            ax.axhline(sw_d, color="crimson", linestyle="-.", linewidth=1.8,
                       label=f"SW rows only INT{bits} n={sw_frag['n_sw']} ({sw_d:+.3f})")
            if sw_rand_d_key and sw_rand_d_key in sw_frag:
                rand_sw_d   = sw_frag[sw_rand_d_key]
                rand_sw_std = sw_frag.get(sw_rand_std_key, 0)
                ax.axhline(rand_sw_d, color="lightcoral", linestyle="--", linewidth=1.2,
                           label=f"random n={sw_frag['n_sw']} ({rand_sw_d:+.3f})")
                if rand_sw_std:
                    ax.axhspan(rand_sw_d - rand_sw_std, rand_sw_d + rand_sw_std,
                               alpha=0.15, color="lightcoral")
        ax.set_xlabel(f"% of non-SW rows quantized to INT{bits}")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name} / {task.split('/')[-1]}\n{title}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="dnabert2",
                   choices=["dnabert2", "ntv3", "generator",
                             "generator_prokaryote", "generator_prokaryote_1b"])
    p.add_argument("--task",      default="prom/prom_core_notata",
                   help="GUE task path (e.g. prom/prom_core_notata). Ignored for generator.")
    p.add_argument("--gue_root",  default="/home/nvidia/data/gue/GUE")
    p.add_argument("--ckpt_dir",  default=None,
                   help="Directory with model_state.pt from fine-tuning. "
                        "If omitted, script expects --task checkpoint auto-path.")
    p.add_argument("--sw_index",  default="results/negative_results/super_weight_index.json")
    p.add_argument("--configs_dir", default="configs")
    p.add_argument("--fracs",    nargs="+", type=float,
                   default=[1.0, 5.0, 10.0, 20.0, 30.0],
                   help="Fractions (pct of non-SW rows) for near_sw and random sweeps.")
    p.add_argument("--n_rand_seeds", type=int, default=10)
    p.add_argument("--bits",    type=int, default=8, choices=[4, 8],
                   help="Quantization bit width: 8=INT8 (default), 4=INT4.")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--out",      default=None)
    p.add_argument("--plot",     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    # ── Load config ──────────────────────────────────────────────────────────
    cfg_path = Path(args.configs_dir) / f"{args.model}.yaml"
    if not cfg_path.exists():
        # try stripping variant suffix  generator_prokaryote → generator_prokaryote.yaml
        cfg_path = Path(args.configs_dir) / "generator_prokaryote.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ── Load SW index ─────────────────────────────────────────────────────────
    model_key = args.model  # "dnabert2", "generator", etc.
    sw_index_path = Path(args.sw_index)
    sw_list = []
    if sw_index_path.exists():
        with open(sw_index_path) as f:
            sw_data = json.load(f)
        if model_key in sw_data:
            sw_list = sw_data[model_key]["results"]
            print(f"  Loaded {len(sw_list)} super rows for {model_key}")
        else:
            print(f"  Warning: {model_key} not in SW index — SW list empty")
    else:
        print(f"  Warning: SW index not found at {args.sw_index}")

    # ── GENERator: perplexity mode ────────────────────────────────────────────
    if args.model in ("generator", "generator_prokaryote", "generator_prokaryote_1b"):
        print(f"\n=== GENERator quantization ablation (perplexity mode) ===")
        model, tokenizer, pattern, num_layers, num_rows = \
            _load_model_and_tokenizer(args.model, cfg, device)

        results = run_generator_quant_ablation(
            model, tokenizer,
            sw_list    = sw_list,
            pattern    = pattern,
            num_layers = num_layers,
            num_rows   = num_rows,
            fracs      = args.fracs,
            n_rand_seeds = args.n_rand_seeds,
            sequences  = PROBE_SEQS,
            bits       = args.bits,
        )
        out_path = args.out or f"results/quant_ablation_{args.model}.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {out_path}")
        return

    # ── DNABERT-2 / NTv3: GUE classification mode ────────────────────────────
    task     = args.task
    task_key = _task_key(task)

    # Auto-detect ckpt directory if not supplied
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        task_leaf = task.replace("/", "_")
        ckpt_dir  = f"results/gue_checkpoints/{args.model}_{task_leaf}"
    ckpt_path = Path(ckpt_dir) / "model_state.pt"
    if not ckpt_path.exists():
        print(f"  ERROR: checkpoint not found at {ckpt_path}")
        print(f"  Run run_gue_ablation.py first to fine-tune and save checkpoint.")
        sys.exit(1)
    print(f"  Using checkpoint: {ckpt_path}")

    print(f"\n=== {args.model} / {task} quantization ablation ===")

    # Count labels from train CSV to set num_labels before loading
    import csv as _csv
    train_csv = Path(args.gue_root) / task / "train.csv"
    with open(train_csv) as f:
        rows = list(_csv.reader(f))[1:]
    num_labels = len({r[-1] for r in rows})
    print(f"  Task: {task}  num_labels={num_labels}")

    # Load backbone (with correct num_labels for DNABERT-2 seq-classification)
    if args.model == "dnabert2":
        import transformers as _tf
        _tf.logging.set_verbosity_error()
        _DNABERT2_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
        tokenizer = _tf.AutoTokenizer.from_pretrained(
            cfg["model_id"], revision=_DNABERT2_REVISION, trust_remote_code=True,
        )
        backbone = _tf.AutoModelForSequenceClassification.from_pretrained(
            cfg["model_id"],
            revision=_DNABERT2_REVISION,
            trust_remote_code=True,
            num_labels=num_labels,
            device_map={"": "cpu"},
        )
        pattern    = cfg["down_proj_pattern"]
        num_layers = cfg["num_layers"]
        num_rows   = backbone.config.hidden_size
        model      = backbone.to(args.device)
    else:
        backbone, tokenizer, pattern, num_layers, num_rows = \
            _load_model_and_tokenizer(args.model, cfg, args.device)
        model = _build_classifier(args.model, backbone, tokenizer, num_labels, args.device)

    # Load test dataset
    test_csv = Path(args.gue_root) / task / "test.csv"
    if not test_csv.exists():
        print(f"  ERROR: test CSV not found at {test_csv}")
        sys.exit(1)
    max_len = _MAX_LEN.get(task_key, 128)
    test_ds = GUEDataset(str(test_csv), tokenizer, max_len)
    print(f"  test_size={len(test_ds)}")

    # Load fine-tuned checkpoint
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(args.device).eval()
    print("  Checkpoint loaded.")

    # ── Run sweep ─────────────────────────────────────────────────────────────
    results = run_gue_quant_sweep(
        model      = model,
        sw_list    = sw_list,
        test_ds    = test_ds,
        pattern    = pattern,
        num_layers = num_layers,
        num_rows   = num_rows,
        fracs      = args.fracs,
        n_rand_seeds = args.n_rand_seeds,
        device     = device,
        bits       = args.bits,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    task_slug = task.replace("/", "_")
    out_path  = args.out or f"results/quant_ablation_{args.model}_{task_slug}.json"
    plot_path = args.plot or out_path.replace(".json", ".png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    results_with_meta = {
        "model": args.model, "task": task,
        "fracs": args.fracs, "n_rand_seeds": args.n_rand_seeds,
        "bits": args.bits,
        "sw_count": len(sw_list),
        **results,
    }
    with open(out_path, "w") as f:
        json.dump(results_with_meta, f, indent=2)
    print(f"\nResults saved → {out_path}")

    _plot_results(results, args.model, task, plot_path, bits=args.bits)


if __name__ == "__main__":
    main()
