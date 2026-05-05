"""
scripts/interpretability/run_cross_kingdom_transfer.py
-------------------------------------------------------
Cross-kingdom domain-transfer test for genomic super-weights.

Tests whether a super-weight learned on one domain (eukaryote or prokaryote)
still fires at elevated levels when fed sequences from the opposite domain.

Two experiments:
  1. Eukaryote model (GENERator-v2-eukaryote-3b) on E. coli sequences
     → In-domain: hg38 promoters/random
     → Out-of-domain: E. coli promoters/random

  2. Prokaryote model (GENERator-v2-prokaryote-3b) on hg38 sequences
     → In-domain: E. coli promoters/random
     → Out-of-domain: hg38 promoters/random

Metric: mean absolute SW row activation (down_proj output at SW row),
        averaged across all sequence positions and reported as:
          - mean ± std per label group
          - fold-change vs. random baseline
          - Mann-Whitney U p-value (in-domain vs. out-of-domain)

Usage:
    python scripts/interpretability/run_cross_kingdom_transfer.py \\
        --sw_index   results/super_weight_index.json \\
        --configs_dir configs \\
        --hg38_fasta  /scratch/11034/atzanakak/grlm/data/reference/hg38/hg38.fa \\
        --hg38_promoters data/regions/hg38/promoters_262kb.bed \\
        --hg38_random    data/regions/hg38/random_262kb.bed \\
        --ecoli_fasta    data/reference/ecoli/ecoli_k12.fna \\
        --ecoli_promoters data/regions/ecoli/promoters_ecoli.bed \\
        --ecoli_random    data/regions/ecoli/random_ecoli.bed \\
        --n_seqs   40 \\
        --window_bp 384 \\
        --out   results/cross_kingdom_transfer.json \\
        --plot  results/cross_kingdom_transfer.png
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# FASTA / BED helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_fasta(path: str):
    try:
        from pyfaidx import Fasta
        return Fasta(path, as_raw=True, sequence_always_upper=True)
    except ImportError:
        raise ImportError("pyfaidx required: pip install pyfaidx")


def _fetch_window(fasta, chrom: str, center: int, window_bp: int):
    half  = window_bp // 2
    start = max(0, center - half)
    end   = start + window_bp
    key   = (chrom if chrom in fasta
             else "chr" + chrom if "chr" + chrom in fasta
             else chrom.lstrip("chr") if chrom.lstrip("chr") in fasta
             else None)
    if key is None:
        return None
    n = len(fasta[key])
    if end > n:
        end = n
        start = max(0, end - window_bp)
    seq = str(fasta[key][start:end])
    if len(seq) < window_bp:
        seq += "N" * (window_bp - len(seq))
    return seq


def _sample_bed_windows(bed_path: str, fasta, n: int, window_bp: int,
                         rng: random.Random, label: str) -> list:
    with open(bed_path) as fh:
        lines = [l.strip().split("\t") for l in fh
                 if l.strip() and not l.startswith("#")]
    pool = rng.sample(lines, min(len(lines), 10 * n))
    results = []
    for row in pool:
        if len(results) >= n:
            break
        chrom  = row[0]
        s, e   = int(row[1]), int(row[2])
        center = (s + e) // 2
        seq    = _fetch_window(fasta, chrom, center, window_bp)
        if seq is None:
            continue
        if seq.count("N") / len(seq) > 0.1:
            continue
        results.append({"label": label, "chrom": chrom, "center": center, "seq": seq})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_key: str, configs_dir: str):
    cfg_path = Path(configs_dir) / f"{model_key}.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    from models.generator_wrapper import GeneratorWrapper
    wrapper = GeneratorWrapper(cfg)
    wrapper.load()
    model     = wrapper.model.eval()
    tokenizer = wrapper.tokenizer
    device    = next(model.parameters()).device
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters())//1_000_000}M params,"
          f" primary device={device})")
    return model, tokenizer, cfg, device


# ─────────────────────────────────────────────────────────────────────────────
# SW activation measurement
# ─────────────────────────────────────────────────────────────────────────────

def _measure_sw_activation(model, tokenizer, seq: str,
                            sw_layer: int, sw_row: int,
                            device: torch.device) -> float | None:
    """Return mean abs activation of SW row across all token positions."""
    try:
        inputs = tokenizer(seq, return_tensors="pt",
                           add_special_tokens=False, truncation=True,
                           max_length=2048)
        input_ids = inputs["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            return None

        captured = {}

        def _hook(_m, _inp, _out):
            captured["act"] = _out.detach()

        target = model.model.layers[sw_layer].mlp.down_proj
        handle = target.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            handle.remove()

        act = captured.get("act")
        if act is None:
            return None
        # [1, L, hidden] → mean abs at SW row
        return float(act[0, :, sw_row].abs().mean().cpu())
    except Exception as e:
        return None


def _run_experiment(model_key: str, configs_dir: str,
                    sw_index: dict,
                    in_domain_seqs: list,
                    out_domain_seqs: list,
                    in_domain_label: str,
                    out_domain_label: str) -> dict:
    """Load model, measure SW activation on in-domain and out-of-domain seqs."""
    print(f"\n  Loading {model_key} ...")
    model, tokenizer, cfg, device = _load_model(model_key, configs_dir)

    entry    = sw_index[model_key]
    results  = entry.get("results", [{}])
    best     = results[0]
    sw_layer = best["layer"]
    sw_row   = best["row"]
    out_max  = best.get("out_max", "?")
    print(f"  SW: layer={sw_layer}  row={sw_row}  out_max={out_max:.0f}" if isinstance(out_max, float) else f"  SW: layer={sw_layer}  row={sw_row}")

    def _measure_list(seqs, label):
        vals = []
        for i, rec in enumerate(seqs):
            v = _measure_sw_activation(model, tokenizer, rec["seq"],
                                       sw_layer, sw_row, device)
            tag = f"[{i+1}/{len(seqs)}] {label} {rec['chrom']}:{rec['center']}"
            if v is None:
                print(f"    {tag}  SKIP")
            else:
                print(f"    {tag}  act={v:.2f}")
                vals.append(v)
        return vals

    print(f"\n  [IN-DOMAIN: {in_domain_label}]")
    in_vals  = _measure_list(in_domain_seqs, in_domain_label)
    print(f"\n  [OUT-OF-DOMAIN: {out_domain_label}]")
    out_vals = _measure_list(out_domain_seqs, out_domain_label)

    # Stats
    from scipy.stats import mannwhitneyu
    stat, pval = (None, None)
    if len(in_vals) >= 3 and len(out_vals) >= 3:
        stat, pval = mannwhitneyu(in_vals, out_vals, alternative="two-sided")

    in_mean  = float(np.mean(in_vals))  if in_vals  else None
    out_mean = float(np.mean(out_vals)) if out_vals else None
    fold     = (out_mean / in_mean) if (in_mean and out_mean and in_mean > 0) else None

    print(f"\n  {in_domain_label:>20}: mean={in_mean:.3f}  n={len(in_vals)}")
    print(f"  {out_domain_label:>20}: mean={out_mean:.3f}  n={len(out_vals)}")
    if pval is not None:
        print(f"  Mann-Whitney p={pval:.4f}  fold={fold:.3f}")

    # Free GPU memory before next model
    del model
    torch.cuda.empty_cache()

    return {
        "model": model_key,
        "sw_layer": sw_layer,
        "sw_row": sw_row,
        "in_domain": in_domain_label,
        "out_domain": out_domain_label,
        "in_domain_vals": in_vals,
        "out_domain_vals": out_vals,
        "in_domain_mean": in_mean,
        "in_domain_std": float(np.std(in_vals)) if in_vals else None,
        "out_domain_mean": out_mean,
        "out_domain_std": float(np.std(out_vals)) if out_vals else None,
        "fold_change": fold,
        "mannwhitney_stat": float(stat) if stat is not None else None,
        "mannwhitney_p": float(pval) if pval is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(results: list, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5),
                              squeeze=False)
    for ax, res in zip(axes[0], results):
        labels = [res["in_domain"], res["out_domain"]]
        vals   = [res["in_domain_vals"], res["out_domain_vals"]]
        means  = [res["in_domain_mean"], res["out_domain_mean"]]
        stds   = [res["in_domain_std"], res["out_domain_std"]]
        colors = ["#2196F3", "#FF5722"]

        for i, (lbl, v, m, s, c) in enumerate(zip(labels, vals, means, stds, colors)):
            ax.scatter([i] * len(v), v, color=c, alpha=0.5, s=20, zorder=3)
            ax.errorbar(i, m, yerr=s, fmt="o", color=c, markersize=10,
                        capsize=6, linewidth=2, zorder=4)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Mean SW row activation (abs)", fontsize=10)
        p = res.get("mannwhitney_p")
        pstr = f"p={p:.3f}" if p is not None else "p=n/a"
        fold = res.get("fold_change")
        fstr = f"fold={fold:.2f}" if fold is not None else ""
        ax.set_title(f"{res['model']}\nlayer={res['sw_layer']} row={res['sw_row']}\n"
                     f"{pstr}  {fstr}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Cross-kingdom transfer: SW activation on foreign sequences",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sw_index",       default="results/super_weight_index.json")
    p.add_argument("--configs_dir",    default="configs")
    p.add_argument("--hg38_fasta",     default="/scratch/11034/atzanakak/grlm/data/reference/hg38/hg38.fa")
    p.add_argument("--hg38_promoters", default="data/regions/hg38/promoters_262kb.bed")
    p.add_argument("--hg38_random",    default="data/regions/hg38/random_262kb.bed")
    p.add_argument("--ecoli_fasta",    default="data/reference/ecoli/ecoli_k12.fna")
    p.add_argument("--ecoli_promoters",default="data/regions/ecoli/promoters_ecoli.bed")
    p.add_argument("--ecoli_random",   default="data/regions/ecoli/random_ecoli.bed")
    p.add_argument("--n_seqs",    type=int, default=40,
                   help="Sequences per group per domain")
    p.add_argument("--window_bp", type=int, default=384)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out",  default="results/cross_kingdom_transfer.json")
    p.add_argument("--plot", default="results/cross_kingdom_transfer.png")
    args = p.parse_args()

    rng = random.Random(args.seed)

    with open(args.sw_index) as f:
        sw_index = json.load(f)

    # ── Load both FASTAs ──────────────────────────────────────────────────────
    print("Loading FASTAs ...")
    hg38_fasta  = _open_fasta(args.hg38_fasta)
    ecoli_fasta = _open_fasta(args.ecoli_fasta)

    # ── Sample sequences from each domain ────────────────────────────────────
    print(f"Sampling {args.n_seqs} sequences per group (window={args.window_bp}bp) ...")

    hg38_promoter_seqs = _sample_bed_windows(
        args.hg38_promoters, hg38_fasta, args.n_seqs, args.window_bp, rng, "hg38_promoter")
    hg38_random_seqs   = _sample_bed_windows(
        args.hg38_random,    hg38_fasta, args.n_seqs, args.window_bp, rng, "hg38_random")
    ecoli_promoter_seqs = _sample_bed_windows(
        args.ecoli_promoters, ecoli_fasta, args.n_seqs, args.window_bp, rng, "ecoli_promoter")
    ecoli_random_seqs   = _sample_bed_windows(
        args.ecoli_random,    ecoli_fasta, args.n_seqs, args.window_bp, rng, "ecoli_random")

    hg38_seqs  = hg38_promoter_seqs  + hg38_random_seqs
    ecoli_seqs = ecoli_promoter_seqs + ecoli_random_seqs

    print(f"  hg38 : {len(hg38_seqs)} sequences")
    print(f"  ecoli: {len(ecoli_seqs)} sequences")

    all_results = []

    # ── Experiment 1: Eukaryote model on hg38 (in) vs E. coli (out) ──────────
    print("\n" + "="*70)
    print("EXPERIMENT 1: GENERator-eukaryote  |  in=hg38  out=E.coli")
    print("="*70)
    res1 = _run_experiment(
        model_key="generator",
        configs_dir=args.configs_dir,
        sw_index=sw_index,
        in_domain_seqs=hg38_seqs,
        out_domain_seqs=ecoli_seqs,
        in_domain_label="hg38",
        out_domain_label="E. coli",
    )
    all_results.append(res1)

    # ── Experiment 2: Prokaryote model on E. coli (in) vs hg38 (out) ─────────
    print("\n" + "="*70)
    print("EXPERIMENT 2: GENERator-prokaryote  |  in=E.coli  out=hg38")
    print("="*70)
    res2 = _run_experiment(
        model_key="generator_prokaryote",
        configs_dir=args.configs_dir,
        sw_index=sw_index,
        in_domain_seqs=ecoli_seqs,
        out_domain_seqs=hg38_seqs,
        in_domain_label="E. coli",
        out_domain_label="hg38",
    )
    all_results.append(res2)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {args.out}")

    _plot(all_results, args.plot)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for res in all_results:
        p = res.get("mannwhitney_p")
        fold = res.get("fold_change")
        sig = "**SIGNIFICANT**" if (p is not None and p < 0.05) else "not significant"
        print(f"  {res['model']:30s}  in={res['in_domain_mean']:.3f}"
              f"  out={res['out_domain_mean']:.3f}"
              f"  fold={fold:.3f}  p={p:.4f}  {sig}"
              if (p is not None and fold is not None)
              else f"  {res['model']}  insufficient data")


if __name__ == "__main__":
    main()
