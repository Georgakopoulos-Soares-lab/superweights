"""
scripts/compare_attribution_methods.py
-----------------------------------------
Validate the two token-level attribution methods — gradient saliency and token
omission — by measuring their per-sequence positional agreement.

Background
----------
Both run_sw_gradient_attribution.py and run_sw_token_omission.py produce a
per-token score for each genomic sequence that aims to answer the same question:
"which positions in this sequence drive the SW row activation?".  They differ
fundamentally in mechanics:

  Gradient saliency — backpropagates through the network; captures linear
  sensitivity at the operating point but suffers from gradient saturation and
  noise in highly nonlinear networks.

  Token omission — perturbation-based; measures the *functional* impact of
  masking each token; no backprop assumptions, no saturation, but can miss
  long-range synergistic effects.

If both methods converge on the same positional ranking, we can trust the
signal. Divergence highlights where gradient saturation or non-local effects
are misleading one of the methods.

Metrics computed
----------------
For each matched sequence (identified by chrom + center):
  - Spearman ρ between the two attribution vectors
  - Kendall τ (more robust to rank ties)
  - Pearson r after interpolating to a common 100-position fractional grid

Aggregate statistics:
  - Mean / median Spearman ρ per genomic context (promoter, enhancer, random)
  - One-sample t-test: H₀ ρ = 0 (random alignment); H₁ ρ > 0 (consistent)
  - Distribution of ρ values plotted as violin + strip

Usage
-----
    python scripts/compare_attribution_methods.py \\
        --grad   results/sw_grad_attribution.json \\
        --omit   results/sw_token_omission.json \\
        --out    results/attribution_comparison.json \\
        --plot   results/attribution_comparison.png

No GPU required.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Rank-based correlation utilities (no scipy dependency required)
# ─────────────────────────────────────────────────────────────────────────────

def _rank(arr: np.ndarray) -> np.ndarray:
    """Return rank array (1-indexed, averaged for ties)."""
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        # find run of equal values
        while j + 1 < n and arr[order[j + 1]] == arr[order[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1  # 1-indexed average
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = _rank(a)
    rb = _rank(b)
    d2 = np.sum((ra - rb) ** 2)
    n  = len(a)
    return float(1 - 6 * d2 / (n * (n * n - 1)))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall τ-b (handles ties)."""
    n = len(a)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    ties_a = ties_b = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                pass
            elif da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) // 2
    denom = math.sqrt((total - ties_a) * (total - ties_b))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def _t_test_one_sample(values: list, mu0: float = 0.0) -> tuple:
    """One-sample t-test (H₁: mean > mu0). Returns (t, p)."""
    vals = [v for v in values if not math.isnan(v)]
    n    = len(vals)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(vals) / n
    var  = sum((v - mean) ** 2 for v in vals) / (n - 1)
    if var == 0:
        return float("inf"), 0.0
    t = (mean - mu0) / math.sqrt(var / n)
    # approximate p via regularised incomplete beta (two-sided then halve)
    # Use a simple t-distribution CDF approximation for large n
    try:
        from scipy.stats import t as t_dist
        p = float(t_dist.sf(t, df=n - 1))
    except ImportError:
        # Normal approximation (valid for n > 30)
        def _norm_sf(z):
            return 0.5 * math.erfc(z / math.sqrt(2))
        p = _norm_sf(t)
    return float(t), float(p)


# ─────────────────────────────────────────────────────────────────────────────
# Interpolate attribution vector to a common fractional grid
# ─────────────────────────────────────────────────────────────────────────────

def _to_frac_grid(vec: np.ndarray, n_grid: int = 100) -> np.ndarray:
    """Linearly interpolate vec of arbitrary length to n_grid points."""
    if len(vec) < 2:
        return np.full(n_grid, float("nan"))
    xs = np.linspace(0, 1, len(vec))
    xq = np.linspace(0, 1, n_grid)
    return np.interp(xq, xs, vec)


# ─────────────────────────────────────────────────────────────────────────────
# Load and align records from the two result files
# ─────────────────────────────────────────────────────────────────────────────

def _load_grad(path: str) -> dict:
    """Returns {(chrom, center): {label, saliency}}."""
    data = json.load(open(path))
    if isinstance(data, dict):
        records = data.get("results", [])
    else:
        records = data
    out = {}
    for r in records:
        key = (r["chrom"], int(r["center"]))
        out[key] = {"label": r["label"], "saliency": np.array(r["saliency"])}
    return out


def _load_omit(path: str) -> dict:
    """Returns {(chrom, center): {label, omission_effect}}."""
    data = json.load(open(path))
    if isinstance(data, dict):
        records = data.get("results", [])
    else:
        records = data
    out = {}
    for r in records:
        key = (r["chrom"], int(r["center"]))
        # omission_effect can be [L] or [L, n_rows]; reduce to 1D
        eff = np.array(r["omission_effect"])
        if eff.ndim > 1:
            eff = eff.mean(axis=-1)
        out[key] = {"label": r["label"], "omission_effect": eff}
    return out


def _load_grad_ordered(path: str) -> list:
    """Returns list of {chrom, center, label, saliency} in file order."""
    data = json.load(open(path))
    records = data.get("results", []) if isinstance(data, dict) else data
    return [{"chrom": r["chrom"], "center": int(r["center"]),
             "label": r["label"], "saliency": np.array(r["saliency"])}
            for r in records]


def _load_omit_ordered(path: str) -> list:
    """Returns list of {chrom, center, label, omission_effect} in file order."""
    data = json.load(open(path))
    records = data.get("results", []) if isinstance(data, dict) else data
    out = []
    for r in records:
        eff = np.array(r["omission_effect"])
        if eff.ndim > 1:
            eff = eff.mean(axis=-1)
        out.append({"chrom": r["chrom"], "center": int(r["center"]),
                    "label": r["label"], "omission_effect": eff})
    return out


def _compare_by_label_order(grad_list: list, omit_list: list, n_grid: int = 100) -> list:
    """
    Pair sequences by label + sequential position within that label.
    Used when chrom+center matching fails (different BED samples).
    """
    from collections import defaultdict
    grad_by_label = defaultdict(list)
    omit_by_label = defaultdict(list)
    for r in grad_list:
        grad_by_label[r["label"]].append(r)
    for r in omit_list:
        omit_by_label[r["label"]].append(r)

    common_labels = set(grad_by_label) & set(omit_by_label)
    total_pairs = sum(min(len(grad_by_label[l]), len(omit_by_label[l]))
                      for l in common_labels)
    print(f"  Matching by label+order across {len(common_labels)} labels, "
          f"{total_pairs} pairs total.")

    results = []
    for label in sorted(common_labels):
        g_recs = grad_by_label[label]
        o_recs = omit_by_label[label]
        for g_r, o_r in zip(g_recs, o_recs):
            g = _to_frac_grid(g_r["saliency"], n_grid)
            o = _to_frac_grid(o_r["omission_effect"], n_grid)
            mask = np.isfinite(g) & np.isfinite(o)
            g_, o_ = g[mask], o[mask]
            if len(g_) < 5:
                continue
            rho = _spearman(g_, o_)
            tau = _kendall_tau(g_[:50], o_[:50])
            r   = _pearson(g_, o_)
            results.append({
                "chrom":          g_r["chrom"],
                "center":         g_r["center"],
                "chrom_omit":     o_r["chrom"],
                "center_omit":    o_r["center"],
                "label":          label,
                "spearman":       round(rho, 5),
                "kendall":        round(tau, 5),
                "pearson":        round(r,   5),
                "n_tokens_grad":  len(g_r["saliency"]),
                "n_tokens_omit":  len(o_r["omission_effect"]),
                "grid_g":         g.tolist(),
                "grid_o":         o.tolist(),
            })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def _compare(grad_data: dict, omit_data: dict, n_grid: int = 100) -> list:
    """
    For each sequence present in both datasets, compute Spearman ρ, Kendall τ,
    and Pearson r between the two attribution vectors (interpolated to n_grid).
    """
    common_keys = set(grad_data.keys()) & set(omit_data.keys())
    print(f"  Gradient records : {len(grad_data)}")
    print(f"  Omission records : {len(omit_data)}")
    print(f"  Matched sequences: {len(common_keys)}")

    if not common_keys:
        print("\n  No sequences match between the two files.")
        print("  Ensure both were run on the same BED regions / FASTA.")
        return []

    results = []
    for key in sorted(common_keys):
        chrom, center = key
        grad_vec = grad_data[key]["saliency"]
        omit_vec = omit_data[key]["omission_effect"]
        label    = grad_data[key]["label"]

        # Interpolate both to a common grid
        g = _to_frac_grid(grad_vec, n_grid)
        o = _to_frac_grid(omit_vec, n_grid)

        # Remove positions with NaN
        mask = np.isfinite(g) & np.isfinite(o)
        g_   = g[mask]
        o_   = o[mask]

        if len(g_) < 5:
            continue

        rho  = _spearman(g_, o_)
        tau  = _kendall_tau(g_[:50], o_[:50])   # O(n²) limit to 50 pts
        r    = _pearson(g_, o_)

        results.append({
            "chrom":    chrom,
            "center":   center,
            "label":    label,
            "spearman": round(rho, 5),
            "kendall":  round(tau, 5),
            "pearson":  round(r,   5),
            "n_tokens_grad":  len(grad_vec),
            "n_tokens_omit":  len(omit_vec),
            "grid_g":   g.tolist(),
            "grid_o":   o.tolist(),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(results: list) -> dict:
    from collections import defaultdict
    ctx_rho = defaultdict(list)
    for r in results:
        ctx_rho[r["label"]].append(r["spearman"])
    ctx_rho["all"] = [r["spearman"] for r in results]

    agg = {}
    for ctx, rhos in ctx_rho.items():
        valid = [v for v in rhos if not math.isnan(v)]
        if not valid:
            continue
        n    = len(valid)
        mean = float(np.mean(valid))
        med  = float(np.median(valid))
        std  = float(np.std(valid))
        t, p = _t_test_one_sample(valid, mu0=0.0)
        agg[ctx] = {
            "n":      n,
            "mean":   round(mean, 5),
            "median": round(med,  5),
            "std":    round(std,  5),
            "t":      round(t,    4),
            "p_gt0":  round(p,    6),
        }
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(results: list, agg: dict, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    from collections import defaultdict
    ctx_rho = defaultdict(list)
    for r in results:
        ctx_rho[r["label"]].append(r["spearman"])

    contexts = sorted(ctx_rho.keys())
    colours  = {"promoter": "steelblue", "enhancer": "darkorange", "random": "gray"}

    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

    ax_viol  = fig.add_subplot(gs[0, :2])
    ax_stat  = fig.add_subplot(gs[0, 2])
    ax_ovl1  = fig.add_subplot(gs[1, 0])
    ax_ovl2  = fig.add_subplot(gs[1, 1])
    ax_ovl3  = fig.add_subplot(gs[1, 2])
    overlay_axes = [ax_ovl1, ax_ovl2, ax_ovl3]

    # ── Violin / strip plot of Spearman ρ per context ─────────────────────────
    FRAC = np.linspace(0, 1, 100)
    all_rho_flat = [r["spearman"] for r in results if not math.isnan(r["spearman"])]
    pos_x  = np.arange(len(contexts))

    for xi, ctx in enumerate(contexts):
        rhos = [v for v in ctx_rho[ctx] if not math.isnan(v)]
        if not rhos:
            continue
        col = colours.get(ctx, "purple")
        # violin using kde-like approach (scatter with jitter)
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(rhos))
        ax_viol.scatter(xi + jitter, rhos, s=25, alpha=0.6, color=col, zorder=3)
        ax_viol.plot([xi - 0.25, xi + 0.25], [np.mean(rhos)] * 2,
                     color="black", linewidth=2.5, zorder=4)
        ax_viol.plot([xi - 0.2, xi + 0.2], [np.median(rhos)] * 2,
                     color=col, linewidth=1.5, linestyle="--", zorder=4)

    ax_viol.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_viol.set_xticks(pos_x)
    ax_viol.set_xticklabels([f"{ctx}\n(n={len(ctx_rho[ctx])})" for ctx in contexts])
    ax_viol.set_ylabel("Spearman ρ  (gradient vs omission)")
    ax_viol.set_title("Per-sequence attribution agreement\n(black bar = mean, dashed = median)")
    ax_viol.set_ylim(-1.05, 1.05)
    ax_viol.grid(True, alpha=0.3, axis="y")

    # ── Stats table panel ─────────────────────────────────────────────────────
    ax_stat.axis("off")
    rows = [["Context", "n", "mean ρ", "median ρ", "p (ρ>0)"]]
    for ctx in list(contexts) + ["all"]:
        if ctx not in agg:
            continue
        a = agg[ctx]
        p_str = f"{a['p_gt0']:.4f}" if a["p_gt0"] >= 1e-4 else "<0.0001"
        rows.append([ctx, str(a["n"]), f"{a['mean']:.3f}",
                     f"{a['median']:.3f}", p_str])
    table = ax_stat.table(cellText=rows[1:], colLabels=rows[0],
                          loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)
    ax_stat.set_title("Aggregate statistics", pad=12)

    # ── Overlaid mean profile per context (top 3 contexts) ───────────────────
    FRAC_AXIS = np.linspace(0, 1, 100)
    ctx_for_overlay = contexts[:3]
    for ax_ov, ctx in zip(overlay_axes, ctx_for_overlay + [None] * (3 - len(ctx_for_overlay))):
        if ctx is None:
            ax_ov.set_visible(False)
            continue
        col = colours.get(ctx, "purple")
        ctx_recs = [r for r in results if r["label"] == ctx
                    and not math.isnan(r["spearman"])]
        if not ctx_recs:
            ax_ov.set_visible(False)
            continue

        ctx_recs_s = sorted(ctx_recs, key=lambda x: x["spearman"], reverse=True)

        def _norm(v):
            span = v.max() - v.min()
            return (v - v.min()) / span if span > 0 else v

        for rec in ctx_recs_s:
            g = np.array(rec["grid_g"])
            o = np.array(rec["grid_o"])
            if not (np.isfinite(g).all() and np.isfinite(o).all()):
                continue
            g_n = _norm(g)
            o_n = _norm(o)
            rng = max(r["spearman"] for r in ctx_recs_s) - min(r["spearman"] for r in ctx_recs_s)
            alpha = 0.25 + 0.5 * (rec["spearman"] - min(r["spearman"] for r in ctx_recs_s)) / max(1e-6, rng)
            ax_ov.plot(FRAC_AXIS, g_n, color=col, alpha=float(alpha)*0.6, linewidth=0.8)
            ax_ov.plot(FRAC_AXIS, o_n, color="black", alpha=float(alpha)*0.4, linewidth=0.8, linestyle="--")

        g_mat = np.array([_norm(np.array(r["grid_g"])) for r in ctx_recs_s
                          if np.isfinite(np.array(r["grid_g"])).all()])
        o_mat = np.array([_norm(np.array(r["grid_o"])) for r in ctx_recs_s
                          if np.isfinite(np.array(r["grid_o"])).all()])
        if len(g_mat):
            ax_ov.plot(FRAC_AXIS, g_mat.mean(0), color=col, linewidth=2, label="gradient (mean)")
        if len(o_mat):
            ax_ov.plot(FRAC_AXIS, o_mat.mean(0), color="black", linewidth=2,
                       linestyle="--", label="omission (mean)")

        mean_rho = float(np.mean([r["spearman"] for r in ctx_recs_s
                                  if not math.isnan(r["spearman"])]))
        ax_ov.set_title(f"{ctx}  (mean \u03c1 = {mean_rho:.3f})", fontsize=9)
        ax_ov.set_xlabel("Fractional position")
        ax_ov.set_ylabel("Normalised attribution")
        ax_ov.legend(fontsize=7)
        ax_ov.grid(True, alpha=0.3)

    plt.suptitle("Gradient saliency vs token omission \u2014 attribution agreement", fontsize=13)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved \u2192 {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare gradient saliency vs token omission attributions"
    )
    p.add_argument("--grad",   default="results/sw_grad_attribution.json",
                   help="Output of run_sw_gradient_attribution.py")
    p.add_argument("--omit",   default="results/sw_token_omission.json",
                   help="Output of run_sw_token_omission.py")
    p.add_argument("--match_by", choices=["chrom_center", "label_order"],
                   default="chrom_center",
                   help="How to pair sequences. 'chrom_center' requires matching loci; "
                        "'label_order' pairs i-th sequence per label when BED samples differ.")
    p.add_argument("--n_grid", type=int, default=100,
                   help="Grid points for fractional-position interpolation")
    p.add_argument("--out",    default="results/attribution_comparison.json")
    p.add_argument("--plot",   default="results/attribution_comparison.png")
    return p.parse_args()


def main():
    args = parse_args()

    for fpath, label in [(args.grad, "gradient saliency"), (args.omit, "token omission")]:
        if not Path(fpath).exists():
            print(f"  {label} file not found: {fpath}")
            if label == "token omission":
                print("  Run scripts/run_sw_token_omission.py first.")
            sys.exit(1)

    print("  Loading gradient saliency results ...", flush=True)
    grad_data = _load_grad(args.grad)

    print("  Loading token omission results ...", flush=True)
    omit_data = _load_omit(args.omit)

    print("\n  Computing per-sequence correlations ...", flush=True)
    if args.match_by == "label_order":
        print("  Using label+order matching (--match_by label_order) ...", flush=True)
        grad_list = _load_grad_ordered(args.grad)
        omit_list = _load_omit_ordered(args.omit)
        results = _compare_by_label_order(grad_list, omit_list, n_grid=args.n_grid)
    else:
        results = _compare(grad_data, omit_data, n_grid=args.n_grid)

    if not results:
        if args.match_by != "label_order":
            print("  No matched sequences \u2014 cannot compute correlations.")
            print("  Tip: re-run with --match_by label_order to pair by label+rank instead.")
        else:
            print("  No results after label+order matching \u2014 cannot compute correlations.")
        sys.exit(1)

    agg = _aggregate(results)

    print("\n  === Attribution agreement summary ===")
    for ctx in sorted(agg.keys()):
        a = agg[ctx]
        sig = ("***" if a["p_gt0"] < 0.001 else
               "**"  if a["p_gt0"] < 0.01  else
               "*"   if a["p_gt0"] < 0.05  else "n.s.")
        print(f"  {ctx:12s}  n={a['n']:3d}  mean \u03c1={a['mean']:+.3f}  "
              f"median \u03c1={a['median']:+.3f}  p={a['p_gt0']:.4f}  {sig}")

    all_rho = [r["spearman"] for r in results if not math.isnan(r["spearman"])]
    print(f"\n  Overall Spearman \u03c1 distribution:")
    print(f"    mean   = {float(np.mean(all_rho)):+.4f}")
    print(f"    median = {float(np.median(all_rho)):+.4f}")
    print(f"    std    = {float(np.std(all_rho)):.4f}")
    print(f"    frac \u03c1 > 0.3 : {sum(v > 0.3 for v in all_rho) / len(all_rho):.1%}")
    print(f"    frac \u03c1 > 0.5 : {sum(v > 0.5 for v in all_rho) / len(all_rho):.1%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    slim_results = [{k: v for k, v in r.items() if k not in ("grid_g", "grid_o")}
                    for r in results]
    payload = {
        "n_matched":    len(results),
        "match_by":     args.match_by,
        "aggregate":    agg,
        "per_sequence": slim_results,
    }
    with open(args.out, "w") as f:
        import json as _json
        _json.dump(payload, f, indent=2)
    print(f"\n  Results saved \u2192 {args.out}")

    _plot(results, agg, args.plot)


if __name__ == "__main__":
    main()
