"""
scripts/negative_results/run_int4_downstream_benchmark.py
-----------------------------------------------------
Practical INT4 downstream benchmark: does SW-aware quantization preserve
task performance better than naive INT4?

Four conditions at a fixed compression target (--near_sw_frac % of non-SW rows):

  1. fp16_baseline  : no quantization (reference)
  2. naive_int4     : INT4 ALL rows, SW rows included  ← worst-case baseline
  3. yu_all_int4    : INT4 all non-SW rows, SW rows stay FP16  (Yu et al. strategy)
  4. near_sw_int4   : INT4 only near-SW rows (shadow-redundant subset), all else FP16
  5. random_int4    : INT4 same count as near_sw, random non-SW rows (--n_rand_seeds seeds)

Supported models and metrics:
  --model dnabert2   : GUE downstream accuracy / MCC on a fine-tuned checkpoint
  --model generator  : Perplexity (PPL) on probe sequences — no GUE fine-tuning needed

Additional metrics per condition:
  - Peak GPU memory (MB) during inference
  - Inference wall-clock time (ms / sample)
  - Theoretical weight memory saved (MB): FP16→INT4 on quantized rows
    (RTN simulation keeps dtype FP16; actual memory needs hardware INT4 kernels)

Usage (DNABERT-2, splice):
    python scripts/negative_results/run_int4_downstream_benchmark.py \\
        --model dnabert2 \\
        --task splice/reconstructed \\
        --ckpt_dir results/gue_checkpoints/dnabert2_reconstructed \\
        --near_sw_frac 10.0 \\
        --out results/int4_downstream_benchmark_splice.json

Usage (GENERator, PPL):
    python scripts/negative_results/run_int4_downstream_benchmark.py \\
        --model generator \\
        --near_sw_frac 30.0 \\
        --out results/int4_downstream_benchmark_generator.json

    --near_sw_frac   % of non-SW candidate rows to INT4 in the near_sw condition.
                     Use 30.0 for GENERator (where the near-SW effect was clearest).
                     Use 10.0 for DNABERT-2. Default: 10.0
    --n_rand_seeds   Number of random seeds for condition 5. Default: 10.
"""

import argparse
import csv
import json
import random as _random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Shared helpers ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))              # scripts/negative_results/
sys.path.insert(0, str(_HERE.parent / "evaluation"))  # scripts/evaluation/
sys.path.insert(0, str(_ROOT))              # repo root (models/, detection/, etc.)

from run_gue_ablation import (
    GUEDataset,
    evaluate,
    _task_key,
    _MAX_LEN,
    collate_fn,
)
import run_compression_sweep as _rcs
from run_compression_sweep import _all_candidates, _rank_proximity
from run_quantization_ablation import (
    _quantize_row_rtn,
    _apply_quantization,
    _restore_quantization,
    _resolve_module,
)

# ─────────────────────────────────────────────────────────────────────────────
# Profiled evaluation: measure accuracy + peak mem + inference time together
# ─────────────────────────────────────────────────────────────────────────────

def profiled_evaluate(model, dataset, device: str = "cuda", batch_size: int = 64) -> dict:
    """Run evaluate() while measuring peak GPU memory and wall-clock time.

    Memory note: RTN simulation leaves weights in FP16, so peak memory here
    reflects activation/KV memory during inference, not weight compression.
    Theoretical memory savings from quantization are computed separately.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn
    )

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().numpy())
    t1 = time.perf_counter()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_mem_mb = None

    elapsed_ms = (t1 - t0) * 1000.0
    n = len(dataset)
    ms_per_sample = elapsed_ms / max(n, 1)

    import sklearn.metrics
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    return {
        "accuracy":      float(sklearn.metrics.accuracy_score(labels, preds)),
        "mcc":           float(sklearn.metrics.matthews_corrcoef(labels, preds)),
        "f1":            float(sklearn.metrics.f1_score(labels, preds, average="macro", zero_division=0)),
        "peak_mem_mb":   round(peak_mem_mb, 1) if peak_mem_mb is not None else None,
        "ms_per_sample": round(ms_per_sample, 3),
        "n_samples":     n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Theoretical weight memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _weight_mb(model, pattern: str, rows: list, bits_per_element: int = 16) -> float:
    """Compute weight tensor size (MB) for a list of (layer, row) pairs."""
    if not rows:
        return 0.0
    # All rows have the same column dimension; sample from first
    li0, _ = rows[0]
    m = _resolve_module(model, pattern, li0)
    row_cols = m.weight.data.shape[1]
    n_params = len(rows) * row_cols
    return n_params * bits_per_element / 8 / 1024 ** 2


def _theoretical_savings_mb(model, pattern: str, rows: list, from_bits: int = 16,
                             to_bits: int = 4) -> float:
    """MB saved by compressing `rows` from `from_bits` to `to_bits` per element."""
    before = _weight_mb(model, pattern, rows, from_bits)
    after  = _weight_mb(model, pattern, rows, to_bits)
    return round(before - after, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Condition runner
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(label: str, model, rows_to_quant: list[tuple],
                  pattern: str, test_ds, device: str,
                  bits: int = 4, baseline: dict | None = None) -> dict:
    """Apply RTN quantization to rows_to_quant, evaluate, restore."""
    if rows_to_quant:
        saves = _apply_quantization(model, pattern, rows_to_quant, bits=bits)

    metrics = profiled_evaluate(model, test_ds, device=device)

    if rows_to_quant:
        _restore_quantization(model, pattern, saves)

    result = {"label": label, "n_quantized_rows": len(rows_to_quant), **metrics}
    if baseline is not None:
        result["delta_acc"] = round(metrics["accuracy"] - baseline["accuracy"], 6)
        result["delta_mcc"] = round(metrics["mcc"] - baseline["mcc"], 6)
        result["delta_f1"]  = round(metrics["f1"]  - baseline["f1"],  6)

    pct = 100.0 * metrics["accuracy"]
    delta_str = f"  Δacc={result.get('delta_acc', 0):+.4f}" if baseline else ""
    print(f"  [{label:<18}]  acc={pct:.2f}%  mcc={metrics['mcc']:.4f}"
          f"  mem={metrics['peak_mem_mb']} MB  {metrics['ms_per_sample']:.2f} ms/sample"
          f"{delta_str}")
    return result


def _ppl_condition(label: str, model, tokenizer, rows_to_quant: list,
                   pattern: str, sequences: list, device: str,
                   bits: int = 4, baseline_ppl: float | None = None) -> dict:
    """Apply RTN quantization, measure PPL + peak GPU mem + wall-clock, restore.

    Handles both single-GPU (device="cuda") and multi-GPU (device_map="auto") models.
    For multi-GPU models the peak_mem_mb is the sum across all CUDA devices.
    """
    from run_quantization_ablation import _generator_perplexity

    if rows_to_quant:
        saves = _apply_quantization(model, pattern, rows_to_quant, bits=bits)

    # Reset peak-memory counters on all available CUDA devices
    if torch.cuda.is_available():
        for _dev in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(_dev)
        torch.cuda.synchronize()

    t0  = time.perf_counter()
    ppl = _generator_perplexity(model, tokenizer, sequences)
    t1  = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem_mb = round(
            sum(torch.cuda.max_memory_allocated(_dev)
                for _dev in range(torch.cuda.device_count())) / 1024 ** 2, 1)
    else:
        peak_mem_mb = None

    if rows_to_quant:
        _restore_quantization(model, pattern, saves)

    elapsed_ms = (t1 - t0) * 1000.0
    ms_per_seq = round(elapsed_ms / max(len(sequences), 1), 3)
    delta_ppl  = round(ppl - baseline_ppl, 6) if baseline_ppl is not None else None

    delta_str = f"  ΔPPL={delta_ppl:+.4f}" if delta_ppl is not None else ""
    print(f"  [{label:<18}]  ppl={ppl:.4f}  mem={peak_mem_mb} MB"
          f"  {ms_per_seq:.1f} ms/seq{delta_str}")

    return {
        "label":            label,
        "n_quantized_rows": len(rows_to_quant),
        "ppl":              round(ppl, 6),
        "delta_ppl":        delta_ppl,
        "peak_mem_mb":      peak_mem_mb,
        "ms_per_seq":       ms_per_seq,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    model,
    sw_list: list,
    test_ds,
    pattern: str,
    num_layers: int,
    num_rows: int,
    near_sw_frac: float,
    n_rand_seeds: int,
    device: str,
    bits: int = 4,
) -> dict:
    sw_coords  = {(sw["layer"], sw["row"]) for sw in sw_list}
    sw_rows    = [(sw["layer"], sw["row"]) for sw in sw_list]

    # All (layer, row) pairs in the down-proj weight matrices
    all_rows = [(li, ri)
                for li in range(num_layers)
                for ri in range(num_rows)]

    # Non-SW candidate pool
    candidates = _all_candidates(model, pattern, num_layers, sw_coords)
    n_total    = len(candidates)

    # Near-SW ranking (closest rows first)
    print("  Ranking rows by proximity to SW coordinates ...", flush=True)
    ranked_near = _rank_proximity(model, pattern, candidates, sw_list,
                                  num_layers, num_rows, ascending=False)

    # Fixed count for near_sw and random conditions
    n_near = max(1, int(round(n_total * near_sw_frac / 100.0)))
    near_rows   = ranked_near[:n_near]
    n_sw        = len(sw_rows)

    print(f"\n  Compression target: near_sw_frac={near_sw_frac}%  "
          f"n_near={n_near}  n_total_candidates={n_total}  n_sw={n_sw}")
    print(f"  Quantization: INT{bits} (RTN simulation)\n")

    # ── Condition 1: fp16_baseline ───────────────────────────────────────────
    baseline = run_condition("fp16_baseline", model, [], pattern, test_ds, device)

    # ── Condition 2: naive_int4 ──────────────────────────────────────────────
    # INT4 ALL rows including SW rows — no exemptions
    naive = run_condition("naive_int4", model, all_rows, pattern, test_ds, device,
                          bits=bits, baseline=baseline)
    naive["theoretical_savings_mb"] = _theoretical_savings_mb(
        model, pattern, all_rows, from_bits=16, to_bits=bits)

    # ── Condition 3: yu_all ──────────────────────────────────────────────────
    # INT4 all non-SW rows, SW rows remain FP16
    yu_all = run_condition("yu_all_int4", model, candidates, pattern, test_ds, device,
                           bits=bits, baseline=baseline)
    yu_all["theoretical_savings_mb"] = _theoretical_savings_mb(
        model, pattern, candidates, from_bits=16, to_bits=bits)

    # ── Condition 4: near_sw_int4 ────────────────────────────────────────────
    # INT4 only near-SW rows (shadow-redundant subset), all else FP16
    near = run_condition("near_sw_int4", model, near_rows, pattern, test_ds, device,
                         bits=bits, baseline=baseline)
    near["theoretical_savings_mb"] = _theoretical_savings_mb(
        model, pattern, near_rows, from_bits=16, to_bits=bits)

    # ── Condition 5: random_int4 (n_rand_seeds seeds) ────────────────────────
    rand_results = []
    for seed in range(n_rand_seeds):
        rand_rows = _random.Random(seed).sample(candidates, n_near)
        r = run_condition(f"random_int4_seed{seed}", model, rand_rows,
                          pattern, test_ds, device, bits=bits, baseline=baseline)
        r["theoretical_savings_mb"] = _theoretical_savings_mb(
            model, pattern, rand_rows, from_bits=16, to_bits=bits)
        rand_results.append(r)

    rand_accs  = [r["accuracy"]  for r in rand_results]
    rand_mccs  = [r["mcc"]       for r in rand_results]
    rand_dAs   = [r["delta_acc"] for r in rand_results]
    rand_times = [r["ms_per_sample"] for r in rand_results]
    rand_mems  = [r["peak_mem_mb"] for r in rand_results if r["peak_mem_mb"] is not None]
    print(f"  [random_int4 (n={n_rand_seeds})]  "
          f"acc={np.mean(rand_accs):.4f}±{np.std(rand_accs):.4f}  "
          f"mcc={np.mean(rand_mccs):.4f}±{np.std(rand_mccs):.4f}  "
          f"Δacc={np.mean(rand_dAs):+.4f}±{np.std(rand_dAs):.4f}")

    random_summary = {
        "label":                "random_int4",
        "n_quantized_rows":     n_near,
        "n_seeds":              n_rand_seeds,
        "accuracy_mean":        float(np.mean(rand_accs)),
        "accuracy_std":         float(np.std(rand_accs)),
        "mcc_mean":             float(np.mean(rand_mccs)),
        "mcc_std":              float(np.std(rand_mccs)),
        "delta_acc_mean":       float(np.mean(rand_dAs)),
        "delta_acc_std":        float(np.std(rand_dAs)),
        "ms_per_sample_mean":   float(np.mean(rand_times)),
        "peak_mem_mb_mean":     float(np.mean(rand_mems)) if rand_mems else None,
        "theoretical_savings_mb": rand_results[0]["theoretical_savings_mb"],
        "per_seed":             rand_results,
    }

    return {
        "fp16_baseline": baseline,
        "naive_int4":    naive,
        "yu_all_int4":   yu_all,
        "near_sw_int4":  near,
        "random_int4":   random_summary,
        "config": {
            "near_sw_frac": near_sw_frac,
            "n_near_rows":  n_near,
            "n_sw_rows":    n_sw,
            "n_candidate_rows": n_total,
            "bits": bits,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def _plot(results: dict, task: str, out_path: str, bits: int = 4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    baseline = results["fp16_baseline"]
    conditions = [
        ("FP16 baseline",  0.0, 0.0, 0.0, None, "black"),
        ("Naive INT4\n(all rows)", results["naive_int4"]["delta_acc"],
         results["naive_int4"]["delta_mcc"], None, None, "crimson"),
        (f"Yu et al.\n(non-SW INT4)", results["yu_all_int4"]["delta_acc"],
         results["yu_all_int4"]["delta_mcc"], None, None, "darkorange"),
        (f"Near-SW INT4\n({results['config']['near_sw_frac']}% near-SW rows)",
         results["near_sw_int4"]["delta_acc"], results["near_sw_int4"]["delta_mcc"],
         None, None, "steelblue"),
        ("Random INT4\n(matched count)",
         results["random_int4"]["delta_acc_mean"], results["random_int4"]["mcc_mean"] - baseline["mcc"],
         results["random_int4"]["delta_acc_std"], results["random_int4"]["mcc_std"],
         "gray"),
    ]

    labels   = [c[0] for c in conditions]
    delta_acc = [c[1] for c in conditions]
    delta_mcc = [c[2] for c in conditions]
    err_acc   = [c[3] for c in conditions]
    colors    = [c[5] for c in conditions]
    x = range(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"INT{bits} Downstream Benchmark: DNABERT-2 / {task.split('/')[-1]}\n"
                 f"FP16 baseline: acc={baseline['accuracy']*100:.2f}%  "
                 f"mcc={baseline['mcc']:.4f}", fontsize=11)

    for ax, deltas, errs, ylabel, title in [
        (ax1, delta_acc, err_acc, "Δ Accuracy vs FP16", "Accuracy Impact"),
        (ax2, delta_mcc, [None]*len(labels), "Δ MCC vs FP16", "MCC Impact"),
    ]:
        bars = ax.bar(x, deltas, color=colors, alpha=0.8, edgecolor="black", linewidth=0.6)
        for i, (bar, err) in enumerate(zip(bars, errs)):
            if err is not None:
                ax.errorbar(i, deltas[i], yerr=err, fmt="none",
                            ecolor="black", capsize=4, linewidth=1.2)
            v = deltas[i]
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.003 if v >= 0 else -0.01),
                    f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    # Annotate near_sw vs random comparison
    near_dA  = results["near_sw_int4"]["delta_acc"]
    rand_dA  = results["random_int4"]["delta_acc_mean"]
    rand_std = results["random_int4"]["delta_acc_std"]
    better   = near_dA > rand_dA
    ax1.annotate(
        f"near_sw {'better ✓' if better else 'worse ✗'} than random\n"
        f"(Δ={near_dA - rand_dA:+.4f} vs random mean)",
        xy=(3, near_dA), xytext=(3.2, near_dA + 0.02),
        fontsize=7, color="steelblue",
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


def _print_summary_table(results: dict, bits: int = 4):
    b  = results["fp16_baseline"]
    cfg = results["config"]
    print("\n" + "="*80)
    print(f"INT{bits} DOWNSTREAM BENCHMARK SUMMARY")
    print(f"near_sw_frac={cfg['near_sw_frac']}%  "
          f"n_near={cfg['n_near_rows']}  n_sw={cfg['n_sw_rows']}  "
          f"n_total_candidates={cfg['n_candidate_rows']}")
    print("="*80)
    header = f"{'Condition':<22} {'Acc':>7} {'MCC':>7} {'Δacc':>8} {'ΔMCC':>8} "
    header += f"{'ΔWt(MB)':>9}  {'mem(MB)':>8} {'ms/smp':>7}"
    print(header)
    print("-" * len(header))

    def _row(label, cond, is_rand=False):
        acc  = cond.get("accuracy_mean", cond.get("accuracy"))
        mcc  = cond.get("mcc_mean",      cond.get("mcc"))
        dA   = cond.get("delta_acc_mean", cond.get("delta_acc", 0.0))
        dM   = cond.get("delta_mcc_mean", cond.get("delta_mcc", 0.0))
        sav  = cond.get("theoretical_savings_mb", 0.0)
        mem  = cond.get("peak_mem_mb_mean", cond.get("peak_mem_mb"))
        ms   = cond.get("ms_per_sample_mean", cond.get("ms_per_sample"))
        mem_s = f"{mem:.1f}" if mem else "—"
        std_s = f"±{cond.get('accuracy_std',0)*100:.2f}" if is_rand else ""
        return (f"  {label:<20} {acc*100:>6.2f}%{std_s:>5}  {mcc:>6.4f}  "
                f"{dA:>+7.4f}  {dM:>+7.4f}  {sav:>8.1f}  {mem_s:>8}  {ms:>6.2f}")

    print(_row("fp16_baseline", b))
    print(_row("naive_int4",    results["naive_int4"]))
    print(_row("yu_all_int4",   results["yu_all_int4"]))
    print(_row("near_sw_int4",  results["near_sw_int4"]))
    print(_row("random_int4",   results["random_int4"], is_rand=True))
    print("="*80)
    near_dA  = results["near_sw_int4"]["delta_acc"]
    rand_dA  = results["random_int4"]["delta_acc_mean"]
    rand_std = results["random_int4"]["delta_acc_std"]
    print(f"\n  near_sw vs random: Δacc = {near_dA:+.4f} vs {rand_dA:+.4f}±{rand_std:.4f}")
    print(f"  near_sw is {'BETTER' if near_dA > rand_dA else 'NOT BETTER'} than random "
          f"(diff={near_dA - rand_dA:+.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# GENERator PPL benchmark
# ─────────────────────────────────────────────────────────────────────────────

PROBE_SEQS = [
    "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC",
    "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG",
    "AAAATTTTCCCCGGGGAAAATTTTCCCCGGGGAAAATTTTCCCCGGGG",
    "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG",
    "GTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC",
    "CGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGAT",
]


def run_generator_benchmark(
    model, tokenizer,
    sw_list: list,
    pattern: str,
    num_layers: int,
    near_sw_frac: float,
    n_rand_seeds: int,
    device: str,
    bits: int = 4,
    sequences: list | None = None,
) -> dict:
    """Four-condition INT4 benchmark for GENERator (perplexity metric).

    Unlike the DNABERT-2 benchmark, num_rows varies per layer so all_rows
    and candidates are built by inspecting each layer's weight shape.
    """
    if sequences is None:
        sequences = PROBE_SEQS

    sw_coords = {(sw["layer"], sw["row"]) for sw in sw_list}
    sw_rows   = [(sw["layer"], sw["row"]) for sw in sw_list]

    # Build all (layer, row) pairs dynamically — GENERator num_rows varies
    all_rows = []
    for li in range(num_layers):
        m = _resolve_module(model, pattern, li)
        nrows = m.weight.data.shape[0]
        for ri in range(nrows):
            all_rows.append((li, ri))

    # Non-SW candidate pool
    candidates = [(li, ri) for li, ri in all_rows if (li, ri) not in sw_coords]
    n_total    = len(candidates)

    # Near-SW ranking (highest shadow similarity first → ascending=False)
    print("  Ranking rows by proximity to SW coordinates …", flush=True)
    # For GENERator the "num_rows" sentinel passed to _rank_proximity can be
    # anything; the function uses the actual weight shape internally.
    sentinel_num_rows = _resolve_module(model, pattern, 0).weight.data.shape[0]
    ranked_near = _rank_proximity(model, pattern, candidates, sw_list,
                                  num_layers, sentinel_num_rows, ascending=False)
    n_near    = max(1, int(round(n_total * near_sw_frac / 100.0)))
    near_rows = ranked_near[:n_near]

    print(f"\n  Compression target: near_sw_frac={near_sw_frac}%  "
          f"n_near={n_near}  n_total_candidates={n_total}  n_sw={len(sw_rows)}")
    print(f"  Quantization: INT{bits} (RTN simulation)\n")

    # ── Condition 1: fp16_baseline ────────────────────────────────────────────
    base         = _ppl_condition("fp16_baseline", model, tokenizer, [],
                                  pattern, sequences, device)
    baseline_ppl = base["ppl"]

    # ── Condition 2: naive_int4  (ALL rows including SW) ─────────────────────
    naive = _ppl_condition("naive_int4", model, tokenizer, all_rows,
                           pattern, sequences, device, bits=bits,
                           baseline_ppl=baseline_ppl)
    naive["theoretical_savings_mb"] = _theoretical_savings_mb(
        model, pattern, all_rows, from_bits=16, to_bits=bits)

    # ── Condition 3: yu_all_int4  (non-SW rows only) ──────────────────────────
    yu = _ppl_condition("yu_all_int4", model, tokenizer, candidates,
                        pattern, sequences, device, bits=bits,
                        baseline_ppl=baseline_ppl)
    yu["theoretical_savings_mb"] = _theoretical_savings_mb(
        model, pattern, candidates, from_bits=16, to_bits=bits)

    # ── Condition 4: near_sw_int4 ─────────────────────────────────────────────
    near = _ppl_condition("near_sw_int4", model, tokenizer, near_rows,
                          pattern, sequences, device, bits=bits,
                          baseline_ppl=baseline_ppl)
    near["theoretical_savings_mb"] = _theoretical_savings_mb(
        model, pattern, near_rows, from_bits=16, to_bits=bits)

    # ── Condition 5: random_int4  (n_rand_seeds seeds) ────────────────────────
    rand_results = []
    for seed in range(n_rand_seeds):
        rand_rows = _random.Random(seed).sample(candidates, n_near)
        r = _ppl_condition(f"random_int4_seed{seed}", model, tokenizer, rand_rows,
                           pattern, sequences, device, bits=bits,
                           baseline_ppl=baseline_ppl)
        r["theoretical_savings_mb"] = _theoretical_savings_mb(
            model, pattern, rand_rows, from_bits=16, to_bits=bits)
        rand_results.append(r)

    rand_ppls  = [r["ppl"]       for r in rand_results]
    rand_dppls = [r["delta_ppl"] for r in rand_results]
    rand_mems  = [r["peak_mem_mb"] for r in rand_results if r["peak_mem_mb"] is not None]
    rand_times = [r["ms_per_seq"] for r in rand_results]
    print(f"  [random_int4 (n={n_rand_seeds})]  "
          f"ppl={np.mean(rand_ppls):.4f}±{np.std(rand_ppls):.4f}  "
          f"ΔPPL={np.mean(rand_dppls):+.4f}±{np.std(rand_dppls):.4f}")

    random_summary = {
        "label":             "random_int4",
        "n_quantized_rows":  n_near,
        "n_seeds":           n_rand_seeds,
        "ppl_mean":          float(np.mean(rand_ppls)),
        "ppl_std":           float(np.std(rand_ppls)),
        "delta_ppl_mean":    float(np.mean(rand_dppls)),
        "delta_ppl_std":     float(np.std(rand_dppls)),
        "ms_per_seq_mean":   float(np.mean(rand_times)),
        "peak_mem_mb_mean":  float(np.mean(rand_mems)) if rand_mems else None,
        "theoretical_savings_mb": rand_results[0]["theoretical_savings_mb"],
        "per_seed":          rand_results,
    }

    return {
        "fp16_baseline": base,
        "naive_int4":    naive,
        "yu_all_int4":   yu,
        "near_sw_int4":  near,
        "random_int4":   random_summary,
        "config": {
            "near_sw_frac":     near_sw_frac,
            "n_near_rows":      n_near,
            "n_sw_rows":        len(sw_rows),
            "n_candidate_rows": n_total,
            "bits":             bits,
        },
    }


def _print_ppl_summary_table(results: dict, bits: int = 4):
    cfg = results["config"]
    print("\n" + "="*80)
    print(f"INT{bits} PPL BENCHMARK SUMMARY  (GENERator)")
    print(f"near_sw_frac={cfg['near_sw_frac']}%  "
          f"n_near={cfg['n_near_rows']}  n_sw={cfg['n_sw_rows']}  "
          f"n_total_candidates={cfg['n_candidate_rows']}")
    print("="*80)
    header = f"{'Condition':<22} {'PPL':>8} {'ΔPPL':>10} {'ΔWt(MB)':>9}  {'mem(MB)':>8} {'ms/seq':>7}"
    print(header)
    print("-" * len(header))

    def _prow(label, cond, is_rand=False):
        ppl  = cond.get("ppl_mean",        cond.get("ppl",       float("nan")))
        dppl = cond.get("delta_ppl_mean",   cond.get("delta_ppl", 0.0)) or 0.0
        sav  = cond.get("theoretical_savings_mb", 0.0)
        mem  = cond.get("peak_mem_mb_mean", cond.get("peak_mem_mb"))
        ms   = cond.get("ms_per_seq_mean",  cond.get("ms_per_seq"))
        mem_s = f"{mem:.1f}" if mem is not None else "—"
        ms_s  = f"{ms:.1f}"  if ms  is not None else "—"
        std_s = f"±{cond.get('ppl_std', 0):.4f}" if is_rand else ""
        return (f"  {label:<20} {ppl:>8.4f}{std_s:<10}  {dppl:>+8.4f}  "
                f"{sav:>8.1f}  {mem_s:>8} {ms_s:>7}")

    print(_prow("fp16_baseline", results["fp16_baseline"]))
    print(_prow("naive_int4",    results["naive_int4"]))
    print(_prow("yu_all_int4",   results["yu_all_int4"]))
    print(_prow("near_sw_int4",  results["near_sw_int4"]))
    print(_prow("random_int4",   results["random_int4"], is_rand=True))
    print("="*80)
    near_d = results["near_sw_int4"]["delta_ppl"]
    rand_d = results["random_int4"]["delta_ppl_mean"]
    rand_s = results["random_int4"]["ppl_std"]
    print(f"\n  near_sw vs random: ΔPPL = {near_d:+.4f} vs {rand_d:+.4f}±{rand_s:.4f}")
    print(f"  near_sw is {'BETTER (lower PPL)' if near_d < rand_d else 'NOT BETTER'} than random "
          f"(diff={near_d - rand_d:+.4f})")


def _plot_ppl(results: dict, model_name: str, out_path: str, bits: int = 4):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    b      = results["fp16_baseline"]
    cfg    = results["config"]
    labels = ["FP16\nbaseline", "Naive\nINT4", "Yu et al.\n(non-SW)",
              f"Near-SW\n({cfg['near_sw_frac']}%)", "Random\nINT4"]
    dppls  = [
        0.0,
        results["naive_int4"]["delta_ppl"],
        results["yu_all_int4"]["delta_ppl"],
        results["near_sw_int4"]["delta_ppl"],
        results["random_int4"]["delta_ppl_mean"],
    ]
    errs   = [None, None, None, None, results["random_int4"]["delta_ppl_std"]]
    colors = ["black", "crimson", "darkorange", "steelblue", "gray"]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"INT{bits} Benchmark: {model_name} (Perplexity)\n"
                 f"FP16 baseline PPL={b['ppl']:.4f}", fontsize=11)
    x    = range(len(labels))
    bars = ax.bar(x, dppls, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    for i, (bar, err) in enumerate(zip(bars, errs)):
        if err is not None:
            ax.errorbar(i, dppls[i], yerr=err, fmt="none",
                        ecolor="black", capsize=4, linewidth=1.2)
        v = dppls[i]
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.002 if v >= 0 else -0.008),
                f"{v:+.4f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ΔPPL vs FP16 baseline")
    ax.set_title("Perplexity impact of INT4 quantization per condition")
    ax.grid(True, axis="y", alpha=0.3)

    near_d = results["near_sw_int4"]["delta_ppl"]
    rand_d = results["random_int4"]["delta_ppl_mean"]
    better = near_d < rand_d
    ax.annotate(
        f"near_sw {'better ✓' if better else 'worse ✗'}\n(Δ={near_d - rand_d:+.4f})",
        xy=(3, near_d), xytext=(3.3, near_d + 0.03),
        fontsize=7, color="steelblue",
        arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8),
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Practical INT4 downstream benchmark (DNABERT-2 GUE or GENERator PPL)."
    )
    p.add_argument("--model",    default="dnabert2",
                   choices=["dnabert2", "generator", "generator_prokaryote",
                             "generator_prokaryote_1b"],
                   help="Model to benchmark. dnabert2=GUE accuracy; generator*=PPL.")
    p.add_argument("--task",     default="splice/reconstructed",
                   help="GUE task path (dnabert2 only). Use splice/reconstructed.")
    p.add_argument("--gue_root", default="/home/nvidia/data/gue/GUE")
    p.add_argument("--ckpt_dir", default=None,
                   help="Dir with model_state.pt (dnabert2 only).")
    p.add_argument("--sw_index", default="results/negative_results/super_weight_index.json")
    p.add_argument("--configs_dir", default="configs")
    p.add_argument("--near_sw_frac", type=float, default=10.0,
                   help="Pct of non-SW rows to INT4 in near_sw + random conditions. "
                        "Use 10.0 for dnabert2, 30.0 for generator. Default: 10.0")
    p.add_argument("--n_rand_seeds", type=int, default=10)
    p.add_argument("--bits",  type=int, default=4, choices=[4, 8])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out",   default=None)
    p.add_argument("--plot",  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.model != "dnabert2":
        return _main_generator(args)
    return _main_dnabert2(args)


def _main_generator(args):
    """GENERator PPL-mode benchmark."""
    cfg_path = Path(args.configs_dir) / f"{args.model}.yaml"
    if not cfg_path.exists():
        cfg_path = _ROOT / "configs" / f"{args.model}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    sw_index_path = Path(args.sw_index)
    if not sw_index_path.exists():
        sw_index_path = _ROOT / args.sw_index
    with open(sw_index_path) as f:
        sw_data = json.load(f)
    sw_list = sw_data.get(args.model, {}).get("results", [])
    print(f"  Loaded {len(sw_list)} super rows for {args.model}")

    print(f"\n  Loading {args.model} …", flush=True)
    from models.generator_wrapper import GeneratorWrapper
    wrapper   = GeneratorWrapper(cfg)
    wrapper.load()
    # GeneratorWrapper uses device_map="auto" — do NOT call .to(device) after load.
    model     = wrapper.model.eval()
    tokenizer = wrapper.tokenizer
    pattern    = cfg["down_proj_pattern"]
    num_layers = cfg["num_layers"]
    print(f"  Model loaded.  layers={num_layers}")

    print(f"\n=== INT{args.bits} PPL Benchmark: {args.model} ===")
    results = run_generator_benchmark(
        model        = model,
        tokenizer    = tokenizer,
        sw_list      = sw_list,
        pattern      = pattern,
        num_layers   = num_layers,
        near_sw_frac = args.near_sw_frac,
        n_rand_seeds = args.n_rand_seeds,
        device       = args.device,
        bits         = args.bits,
    )

    _print_ppl_summary_table(results, bits=args.bits)

    task_slug = args.model
    out_path  = args.out  or f"results/int4_downstream_benchmark_{task_slug}.json"
    plot_path = args.plot or out_path.replace(".json", ".png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model":         args.model,
        "metric":        "perplexity",
        "bits":          args.bits,
        "near_sw_frac":  args.near_sw_frac,
        "n_rand_seeds":  args.n_rand_seeds,
        **results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")
    _plot_ppl(results, args.model, plot_path, bits=args.bits)


def _main_dnabert2(args):
    """DNABERT-2 GUE accuracy benchmark (original behaviour)."""
    cfg_path = Path(args.configs_dir) / "dnabert2.yaml"
    if not cfg_path.exists():
        cfg_path = _ROOT / "configs" / "dnabert2.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ── Load SW index ─────────────────────────────────────────────────────────
    sw_index_path = Path(args.sw_index)
    if not sw_index_path.exists():
        sw_index_path = _ROOT / args.sw_index
    with open(sw_index_path) as f:
        sw_data = json.load(f)
    sw_list = sw_data.get("dnabert2", {}).get("results", [])
    print(f"  Loaded {len(sw_list)} super rows for dnabert2")

    # ── Checkpoint path ───────────────────────────────────────────────────────
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        task_leaf = args.task.replace("/", "_")
        ckpt_dir  = f"results/gue_checkpoints/dnabert2_{task_leaf}"
    ckpt_path = Path(ckpt_dir) / "model_state.pt"
    if not ckpt_path.exists():
        ckpt_path = _ROOT / ckpt_dir / "model_state.pt"
    if not ckpt_path.exists():
        print(f"  ERROR: checkpoint not found at {ckpt_path}")
        print("  Run scripts/evaluation/run_gue_ablation.py first to produce the checkpoint.")
        sys.exit(1)

    # ── Count labels ──────────────────────────────────────────────────────────
    train_csv = Path(args.gue_root) / args.task / "train.csv"
    with open(train_csv) as f:
        num_labels = len({row[-1] for row in list(csv.reader(f))[1:]})

    # ── Load model ────────────────────────────────────────────────────────────
    import transformers
    transformers.logging.set_verbosity_error()
    _DNABERT2_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"

    print("\n  Loading DNABERT-2 ...", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg["model_id"], revision=_DNABERT2_REVISION, trust_remote_code=True,
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"],
        revision=_DNABERT2_REVISION,
        trust_remote_code=True,
        num_labels=num_labels,
        device_map={"": "cpu"},
    )
    pattern    = cfg["down_proj_pattern"]
    num_layers = cfg["num_layers"]
    num_rows   = model.config.hidden_size

    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(args.device).eval()
    print(f"  Checkpoint loaded: {ckpt_path}")

    # ── Load test dataset ─────────────────────────────────────────────────────
    task_key = _task_key(args.task)
    max_len  = _MAX_LEN.get(task_key, 128)
    test_csv = Path(args.gue_root) / args.task / "test.csv"
    test_ds  = GUEDataset(str(test_csv), tokenizer, max_len)
    print(f"  Test set: {len(test_ds)} samples  max_len={max_len}\n")

    print(f"=== INT{args.bits} Downstream Benchmark: dnabert2 / {args.task} ===")

    # ── Run benchmark ─────────────────────────────────────────────────────────
    results = run_benchmark(
        model        = model,
        sw_list      = sw_list,
        test_ds      = test_ds,
        pattern      = pattern,
        num_layers   = num_layers,
        num_rows     = num_rows,
        near_sw_frac = args.near_sw_frac,
        n_rand_seeds = args.n_rand_seeds,
        device       = args.device,
        bits         = args.bits,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary_table(results, bits=args.bits)

    # ── Save ──────────────────────────────────────────────────────────────────
    task_slug = args.task.replace("/", "_")
    out_path  = args.out  or f"results/int4_downstream_benchmark_{task_slug}.json"
    plot_path = args.plot or out_path.replace(".json", ".png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model": "dnabert2",
        "task":  args.task,
        "bits":  args.bits,
        "near_sw_frac": args.near_sw_frac,
        "n_rand_seeds": args.n_rand_seeds,
        **results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    _plot(results, args.task, plot_path, bits=args.bits)


if __name__ == "__main__":
    main()
