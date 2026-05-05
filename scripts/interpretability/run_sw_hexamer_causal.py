"""
scripts/run_sw_hexamer_causal.py
---------------------------------
Causal hexamer test: does SW ablation specifically degrade the model's
next-token predictions for CC/CT-rich hexamers more than for CpG hexamers?

This closes the interpretability loop left open by the k-mer vocabulary scan.
The scan shows *which* hexamers activate the SW row most strongly.  This
script asks: when the SW rows are zeroed mid-forward-pass, does the model's
predictive distribution specifically deteriorate for those high-activating
hexamers?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method
------
For each of the 4,096 6-mer tokens in GENERator's vocabulary, we construct
the same poly-A context used in the k-mer vocabulary scan:

    [A×5]  [target k-mer]  [A×5]   →  11 tokens  (66 bp)

We then:

  1. Run a clean forward pass, extract logits at the target position
     → p_clean   (next-token distribution after seeing the target k-mer)

  2. Repeat with the SW row(s) zeroed via a forward hook
     → p_ablated

  3. Compute KL(p_clean ‖ p_ablated) at the target position
     This measures how much the model's prediction changes when the SW is gone.

  4. Rank all 4,096 k-mers by their KL divergence.

Hypothesis
----------
  - High-SW-activating k-mers (CC/CT-rich) should show the HIGHEST KL
    divergence → the SW row is causally encoding the predictive signal for
    these tokens.
  - Low-SW-activating k-mers (CpG, poly-A) should show LOW KL divergence
    → the model's predictions for these tokens are independent of the SW.

If this pattern holds, we have a complete causal chain:
  "SW fires on CC/CT k-mers"  +  "SW ablation disrupts CC/CT predictions"
  → SW is causally responsible for CC/CT next-token modelling.

Outputs
-------
  JSON : results/sw_hexamer_causal.json  — per-k-mer KL and SW activation
  PNG  : results/sw_hexamer_causal.png   — scatter plot + ranked bar chart

Usage
-----
    python scripts/run_sw_hexamer_causal.py \\
        --kmer_scan results/sw_kmer_scan.json \\
        --out       results/sw_hexamer_causal.json \\
        --plot      results/sw_hexamer_causal.png

GPU required.
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POLY_A_TOKEN = 32   # AAAAAA in GENERator vocab
TARGET_POS   = 5    # index of target k-mer in 11-token context
CTX_HALF     = 5


# ─────────────────────────────────────────────────────────────────────────────
# Build input batch (identical to run_sw_kmer_scan.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_batch(kmer_ids: list) -> torch.Tensor:
    prefix = [POLY_A_TOKEN] * CTX_HALF
    suffix = [POLY_A_TOKEN] * CTX_HALF
    rows   = [prefix + [kid] + suffix for kid in kmer_ids]
    return torch.tensor(rows, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# KL divergence  (numerically stable)
# ─────────────────────────────────────────────────────────────────────────────

def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p ‖ q) = sum(p * log(p / q)), both distributions over vocabulary."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ─────────────────────────────────────────────────────────────────────────────
# Run clean + ablated forward pass in mini-batches
# ─────────────────────────────────────────────────────────────────────────────

def _scan(
    model,
    kmer_ids: list,
    sw_layer: int,
    sw_rows: list,
    device: str,
    batch_size: int = 128,
) -> tuple:
    """
    Returns:
        logits_clean   [N, vocab]  — clean logits at TARGET_POS
        logits_ablated [N, vocab]  — SW-ablated logits at TARGET_POS
        sw_acts        [N]         — mean SW row activation (clean), for correlation
    """
    target_module = model.model.layers[sw_layer].mlp.down_proj
    N = len(kmer_ids)

    logits_clean_list   = []
    logits_ablated_list = []
    sw_acts_list        = []

    for start in range(0, N, batch_size):
        chunk = kmer_ids[start: start + batch_size]
        ids   = _build_batch(chunk).to(device)

        # ── Clean pass ──────────────────────────────────────────────────────
        _store = {}
        def _hook_clean(_mod, _inp, _out, _s=_store):
            _s["act"] = _out[:, TARGET_POS, :]   # [B, hidden]

        h = target_module.register_forward_hook(_hook_clean)
        with torch.no_grad():
            out_clean = model(input_ids=ids)
        h.remove()

        lc = out_clean.logits[:, TARGET_POS, :].cpu().float().numpy()   # [B, V]
        sw_act = _store["act"][:, sw_rows].mean(dim=1).cpu().float().numpy()  # [B]
        logits_clean_list.append(lc)
        sw_acts_list.append(sw_act)

        # ── Ablated pass: zero SW rows in down_proj output ──────────────────
        def _hook_ablate(_mod, _inp, _out):
            out = _out.clone()
            out[:, :, sw_rows] = 0.0
            return out

        h2 = target_module.register_forward_hook(_hook_ablate)
        with torch.no_grad():
            out_ablated = model(input_ids=ids)
        h2.remove()

        la = out_ablated.logits[:, TARGET_POS, :].cpu().float().numpy()  # [B, V]
        logits_ablated_list.append(la)

        print(f"    batch {start // batch_size + 1}/{(N - 1) // batch_size + 1}"
              f"  ({start}–{min(start + batch_size, N) - 1})", flush=True)

    logits_clean   = np.concatenate(logits_clean_list,   axis=0)
    logits_ablated = np.concatenate(logits_ablated_list, axis=0)
    sw_acts        = np.concatenate(sw_acts_list,        axis=0)
    return logits_clean, logits_ablated, sw_acts


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(kmers, kl_divs, sw_acts, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib unavailable")
        return

    rank = np.argsort(kl_divs)[::-1]
    top_n = 30

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter KL vs SW activation
    ax = axes[0]
    ax.scatter(sw_acts, kl_divs, s=8, alpha=0.3, color="steelblue")
    # Highlight top-20 and bottom-20 SW activators
    sw_rank = np.argsort(sw_acts)
    for i in sw_rank[-20:]:
        ax.scatter(sw_acts[i], kl_divs[i], s=30, color="red",   zorder=5)
    for i in sw_rank[:20]:
        ax.scatter(sw_acts[i], kl_divs[i], s=30, color="green", zorder=5)

    r = float(np.corrcoef(sw_acts, kl_divs)[0, 1])
    ax.set_xlabel("SW row activation (clean)")
    ax.set_ylabel("KL(clean ‖ ablated) at target position")
    ax.set_title(f"SW activation vs ablation cost\nr={r:.3f}  (red=top-20 activators, green=bottom-20)")

    # Right: top-30 k-mers by KL divergence
    ax2 = axes[1]
    top_kmers  = [kmers[i] for i in rank[:top_n]]
    top_kl     = kl_divs[rank[:top_n]]
    colors = ["tomato" if (km.count("C") + km.count("G")) / 6 >= 0.5 else "steelblue"
              for km in top_kmers]
    ax2.barh(range(top_n)[::-1], top_kl, color=colors)
    ax2.set_yticks(range(top_n)[::-1])
    ax2.set_yticklabels(top_kmers, fontsize=8)
    ax2.set_xlabel("KL(clean ‖ ablated)")
    ax2.set_title(f"Top-{top_n} k-mers by ablation cost\n(red=GC≥50%, blue=GC<50%)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Softmax helper
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Causal hexamer test: KL(clean‖ablated)")
    p.add_argument("--model",       default="generator",
                   choices=["generator", "generator_prokaryote",
                            "generator_prokaryote_1b"])
    p.add_argument("--kmer_scan",   default="results/sw_kmer_scan.json",
                   help="Path to existing k-mer scan results (for correlation)")
    p.add_argument("--sw_index",    default="results/super_weight_index.json")
    p.add_argument("--configs_dir", default="configs")
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--out",         default="results/sw_hexamer_causal.json")
    p.add_argument("--plot",        default="results/sw_hexamer_causal.png")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load config + SW index ─────────────────────────────────────────────
    cfg      = yaml.safe_load(open(Path(args.configs_dir) / f"{args.model}.yaml"))
    sw_data  = json.load(open(args.sw_index))
    sw_entry = sw_data.get(args.model, {})
    sw_list  = sw_entry.get("results", []) if isinstance(sw_entry, dict) else sw_entry
    if not sw_list:
        print(f"No SW found for {args.model} in {args.sw_index}.")
        sys.exit(1)

    top_sw   = max(sw_list, key=lambda x: x["out_max"])
    sw_layer = int(top_sw["layer"])
    sw_rows  = sorted({int(s["row"]) for s in sw_list if s["layer"] == sw_layer})
    print(f"  SW: layer={sw_layer}  rows={sw_rows}")

    # ── Load model + tokenizer ─────────────────────────────────────────────
    print(f"  Loading {args.model} ...", flush=True)
    from models.generator_wrapper import GeneratorWrapper
    wrapper   = GeneratorWrapper(cfg)
    wrapper.load()
    model     = wrapper.model
    tokenizer = wrapper.tokenizer
    model.eval()

    # ── Build k-mer vocabulary (same method as run_sw_kmer_scan.py) ────────
    print("  Building 6-mer vocabulary ...", flush=True)
    bases = "ACGT"
    all_kmers = ["".join(c) for c in product(bases, repeat=6)]

    kmer_ids   = []
    valid_kmers = []
    for km in all_kmers:
        ids = tokenizer.encode(km, add_special_tokens=False)
        if len(ids) == 1:
            kmer_ids.append(ids[0])
            valid_kmers.append(km)

    print(f"  {len(valid_kmers)} / 4096 k-mers tokenize to a single token")

    # ── Run clean + ablated scans ──────────────────────────────────────────
    print(f"  Running clean + ablated forward passes ...", flush=True)
    logits_clean, logits_ablated, sw_acts = _scan(
        model, kmer_ids, sw_layer, sw_rows, args.device, args.batch_size
    )

    # Convert logits → probabilities
    p_clean   = _softmax(logits_clean)    # [N, vocab]
    p_ablated = _softmax(logits_ablated)  # [N, vocab]

    # ── Per-k-mer KL divergence ────────────────────────────────────────────
    print("  Computing KL divergences ...", flush=True)
    kl_divs = np.array([_kl(p_clean[i], p_ablated[i]) for i in range(len(valid_kmers))])

    # ── Correlation with SW activation ────────────────────────────────────
    r_kl_sw = float(np.corrcoef(sw_acts, kl_divs)[0, 1])
    print(f"\n  Pearson r(SW activation, KL divergence) = {r_kl_sw:.4f}")

    # ── Load kmer scan results for cross-reference ─────────────────────────
    scan_acts = {}
    if Path(args.kmer_scan).exists():
        ks = json.load(open(args.kmer_scan))
        for row in ks.get("results", []):
            scan_acts[row["kmer"]] = row["activation_mean"]

    # ── Rank and report ────────────────────────────────────────────────────
    rank = np.argsort(kl_divs)[::-1]

    print(f"\n  === Top 20 k-mers by ablation cost (KL) ===")
    for i in rank[:20]:
        km  = valid_kmers[i]
        gc  = (km.count("G") + km.count("C")) / 6
        act = scan_acts.get(km, float("nan"))
        print(f"  {km}  KL={kl_divs[i]:.4f}  SW_act={act:>10.1f}  GC={gc:.0%}")

    print(f"\n  === Bottom 10 k-mers by ablation cost ===")
    for i in rank[-10:]:
        km  = valid_kmers[i]
        gc  = (km.count("G") + km.count("C")) / 6
        act = scan_acts.get(km, float("nan"))
        print(f"  {km}  KL={kl_divs[i]:.4f}  SW_act={act:>10.1f}  GC={gc:.0%}")

    # ── Group statistics: compare top-SW-activating vs bottom ─────────────
    # Use scan activations to group k-mers into top/bottom quartiles
    if scan_acts:
        scan_scores = np.array([scan_acts.get(km, 0.0) for km in valid_kmers])
        q75 = np.percentile(scan_scores, 75)
        q25 = np.percentile(scan_scores, 25)
        top_mask = scan_scores >= q75
        bot_mask = scan_scores <= q25
        kl_top = kl_divs[top_mask]
        kl_bot = kl_divs[bot_mask]

        from scipy import stats as scipy_stats
        t, p = scipy_stats.ttest_ind(kl_top, kl_bot)
        print(f"\n  KL(top-SW quartile) = {kl_top.mean():.4f} ± {kl_top.std():.4f}")
        print(f"  KL(bot-SW quartile) = {kl_bot.mean():.4f} ± {kl_bot.std():.4f}")
        print(f"  t={t:.3f}  p={p:.4e}  (Welch's t-test, two-sided)")
    else:
        t, p = float("nan"), float("nan")
        kl_top = kl_bot = np.array([])

    # ── Save ──────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "sw_layer": sw_layer,
        "sw_rows": sw_rows,
        "n_kmers": len(valid_kmers),
        "r_kl_sw_activation": r_kl_sw,
        "group_test": {
            "kl_top_sw_quartile_mean": float(kl_top.mean()) if len(kl_top) else None,
            "kl_top_sw_quartile_std":  float(kl_top.std())  if len(kl_top) else None,
            "kl_bot_sw_quartile_mean": float(kl_bot.mean()) if len(kl_bot) else None,
            "kl_bot_sw_quartile_std":  float(kl_bot.std())  if len(kl_bot) else None,
            "t": float(t),
            "p_two_sided": float(p),
        },
        "results": sorted(
            [
                {
                    "kmer":           valid_kmers[i],
                    "token_id":       int(kmer_ids[i]),
                    "kl_clean_ablated": float(kl_divs[i]),
                    "sw_activation":  float(sw_acts[i]),
                    "scan_sw_activation": float(scan_acts.get(valid_kmers[i], float("nan"))),
                    "gc_frac":        (valid_kmers[i].count("G") + valid_kmers[i].count("C")) / 6,
                    "rank_by_kl":     int(np.where(rank == i)[0][0]) + 1,
                }
                for i in range(len(valid_kmers))
            ],
            key=lambda x: x["rank_by_kl"],
        ),
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved → {args.out}")

    _plot(valid_kmers, kl_divs, sw_acts, args.plot)


if __name__ == "__main__":
    main()
