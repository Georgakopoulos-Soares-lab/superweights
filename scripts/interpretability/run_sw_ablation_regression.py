"""
scripts/run_sw_ablation_regression.py
---------------------------------------
Regression analysis: what sequence properties predict the SW ablation cost
(ΔPPL when SW rows are zeroed)?

Loads per-sequence results from run_sw_causal_tracing.py (sw_causal_tracing.json),
re-fetches the sequences from hg38 via the stored chromosome + center coordinates,
and computes per-sequence features:

  gc_frac       — fraction of G/C bases
  cpg_density   — observed CpG / expected CpG  (CG dinuc count * len) / (C_count * G_count)
  kmer_entropy  — Shannon entropy over 6-mer token distribution  H = -Σ p log p
  repeat_frac   — fraction of sequence covered by tandem runs
                  (token that equals its predecessor)
  complexity    — linguistic complexity: observed_kmers / total_kmers
  mean_gc_run   — mean length of longest consecutive GC run

Then fits ordinary-least-squares regression of delta_sw ~ features and reports
correlations, coefficients, and a figure.

No GPU needed — pure feature analysis on sequences.

Usage:
    python scripts/run_sw_ablation_regression.py \
        --trace results/sw_causal_tracing.json \
        --fasta /home/nvidia/data/hg38/hg38.fa \
        --out   results/sw_ablation_regression.json \
        --plot  results/sw_ablation_regression.png
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# FASTA helper (reuse same pattern)
# ─────────────────────────────────────────────────────────────────────────────

def _open_fasta(path: str):
    try:
        from pyfaidx import Fasta
        return Fasta(path, as_raw=True, sequence_always_upper=True)
    except ImportError:
        raise ImportError("pyfaidx required: pip install pyfaidx")


def _fetch_window(fasta, chrom: str, center: int, window_bp: int) -> str | None:
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
        end   = n
        start = max(0, end - window_bp)
    seq = str(fasta[key][start:end])
    if len(seq) < window_bp:
        seq += "N" * (window_bp - len(seq))
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# Sequence features
# ─────────────────────────────────────────────────────────────────────────────

def _gc_frac(seq: str) -> float:
    valid = sum(1 for b in seq if b in "ACGT")
    if valid == 0:
        return 0.0
    return sum(1 for b in seq if b in "GC") / valid


def _cpg_density(seq: str) -> float:
    """Observed/expected CpG ratio."""
    seq  = seq.upper()
    L    = len(seq)
    if L < 2:
        return 0.0
    cg  = sum(1 for i in range(L - 1) if seq[i]=="C" and seq[i+1]=="G")
    c   = seq.count("C")
    g   = seq.count("G")
    exp = (c * g) / L if L > 0 else 1e-9
    return cg / exp if exp > 0 else 0.0


def _kmer_entropy(seq: str, k: int = 6) -> float:
    """Shannon entropy of k-mer distribution."""
    tokens = [seq[i:i+k] for i in range(0, len(seq) - k + 1, k)]
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total  = sum(counts.values())
    return -sum((v / total) * math.log2(v / total) for v in counts.values())


def _repeat_frac(seq: str, k: int = 6) -> float:
    """Fraction of non-overlapping k-mer positions where token == previous token."""
    tokens = [seq[i:i+k] for i in range(0, len(seq) - k + 1, k)]
    if len(tokens) < 2:
        return 0.0
    runs = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
    return runs / (len(tokens) - 1)


def _complexity(seq: str, k: int = 6) -> float:
    """Linguistic complexity: observed distinct k-mers / total k-mer positions."""
    tokens = [seq[i:i+k] for i in range(0, len(seq) - k + 1, k)]
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _extract_features(seq: str) -> dict:
    seq = seq.upper()
    return {
        "gc_frac":     _gc_frac(seq),
        "cpg_density": _cpg_density(seq),
        "kmer_entropy": _kmer_entropy(seq, k=6),
        "repeat_frac": _repeat_frac(seq, k=6),
        "complexity":  _complexity(seq, k=6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# OLS regression
# ─────────────────────────────────────────────────────────────────────────────

def _ols(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Fit OLS y ~ X (with intercept) and return coefficients + stats."""
    n, p = X.shape
    Xb   = np.column_stack([np.ones(n), X])   # add intercept
    # Normal equations: β = (XᵀX)⁻¹ Xᵀy
    try:
        coef = np.linalg.lstsq(Xb, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {}
    y_hat  = Xb @ coef
    resid  = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Per-feature correlations
    corrs  = {feat: float(np.corrcoef(X[:, i], y)[0, 1])
              for i, feat in enumerate(feature_names)}
    return {
        "intercept": float(coef[0]),
        "coefs":     {feat: float(coef[i+1]) for i, feat in enumerate(feature_names)},
        "r2":        r2,
        "pearson_r": corrs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(records: list, feature_names: list[str], reg: dict, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    delta_sw = np.array([r["delta_sw"]        for r in records])
    labels   =          [r["label"]            for r in records]
    ctx_map  = {"promoter": "steelblue", "enhancer": "darkorange", "random": "gray"}

    n_feat = len(feature_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for fi, feat in enumerate(feature_names):
        ax   = axes[fi]
        vals = np.array([r["features"][feat] for r in records])
        cols = [ctx_map.get(lbl, "purple") for lbl in labels]
        ax.scatter(vals, delta_sw, c=cols, s=20, alpha=0.7)
        r = float(np.corrcoef(vals, delta_sw)[0, 1])
        # regression line
        x_ = np.linspace(vals.min(), vals.max(), 100)
        coef_feat = reg.get("coefs", {}).get(feat, 0.0)
        intercept = reg.get("intercept", 0.0)
        ax.plot(x_, intercept + coef_feat * x_, "k--", linewidth=1)
        ax.set_xlabel(feat)
        ax.set_ylabel("ΔPPL (SW ablation cost)")
        ax.set_title(f"{feat}\nr = {r:.3f}")
        ax.grid(True, alpha=0.3)

    # Last panel: R² summary bar
    ax = axes[n_feat]
    feat_rs = [reg.get("pearson_r", {}).get(f, 0.0) for f in feature_names]
    cols = ["#d73027" if r < 0 else "#4575b4" for r in feat_rs]
    ax.barh(feature_names, feat_rs, color=cols)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson r with ΔPPL")
    ax.set_title(f"Feature correlations  (OLS R²={reg.get('r2', 0):.3f})")
    ax.grid(True, alpha=0.3, axis="x")

    # hide any unused axes
    for ax in axes[n_feat+1:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace",  default="results/sw_causal_tracing.json")
    p.add_argument("--fasta",  default="/home/nvidia/data/hg38/hg38.fa")
    p.add_argument("--window_bp", type=int, default=3072)
    p.add_argument("--out",    default="results/sw_ablation_regression.json")
    p.add_argument("--plot",   default="results/sw_ablation_regression.png")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load causal tracing results ───────────────────────────────────────────
    trace = json.load(open(args.trace))
    seqs_meta = trace["results"]
    print(f"  Loaded {len(seqs_meta)} sequences from {args.trace}")

    # ── Re-fetch sequences from hg38 ─────────────────────────────────────────
    print("  Loading FASTA ...", flush=True)
    fasta = _open_fasta(args.fasta)
    window_bp = (args.window_bp // 6) * 6

    records = []
    for rec in seqs_meta:
        seq = _fetch_window(fasta, rec["chrom"], rec["center"], window_bp)
        if seq is None:
            print(f"  [skip] {rec['chrom']}:{rec['center']} not found in FASTA")
            continue
        feats = _extract_features(seq)
        records.append({
            "label":    rec["label"],
            "chrom":    rec["chrom"],
            "center":   rec["center"],
            "delta_sw": rec["delta_sw"],
            "clean_ppl": rec["clean_ppl"],
            "features": feats,
        })

    print(f"  {len(records)} sequences with features computed")

    # ── Regression ────────────────────────────────────────────────────────────
    feature_names = ["gc_frac", "cpg_density", "kmer_entropy", "repeat_frac", "complexity"]
    X = np.array([[r["features"][f] for f in feature_names] for r in records])
    y = np.array([r["delta_sw"]                              for r in records])

    # Standardise features for comparable coefficients
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    reg      = _ols(X_std, y, feature_names)
    reg_raw  = _ols(X,     y, feature_names)   # raw (unstandardised) for reporting

    print(f"\n  === Regression summary (standardised X) ===")
    print(f"  R² = {reg['r2']:.4f}")
    print(f"  Pearson r per feature:")
    for feat, r in sorted(reg["pearson_r"].items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:18s}  r = {r:+.4f}")

    # Per-context breakdown
    from collections import defaultdict
    ctx_delta = defaultdict(list)
    for r in records:
        ctx_delta[r["label"]].append(r["delta_sw"])

    print(f"\n  === ΔPPL by context ===")
    for ctx, vals in sorted(ctx_delta.items()):
        print(f"  {ctx:10s}: mean={np.mean(vals):+.3f}  std={np.std(vals):.3f}  n={len(vals)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_seqs":       len(records),
        "features":     feature_names,
        "regression_standardised": reg,
        "regression_raw":          reg_raw,
        "records":      records,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved → {args.out}")

    _plot(records, feature_names, reg_raw, args.plot)


if __name__ == "__main__":
    main()
