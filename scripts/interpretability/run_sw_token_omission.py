"""
scripts/run_sw_token_omission.py
-----------------------------------
Token omission (leave-one-out) attribution for GENERator SW rows.

For each genomic sequence, creates a batch of L copies where copy i has token i
replaced by the poly-A token (AAAAAA, id=32).  A single batched forward pass
gives the SW-row activation for all L omissions simultaneously, avoiding L
separate forward passes.

Attribution score at position t:
    omission_effect[t] = baseline_act[t] - omitted_act[t]

  Positive value → token t *contributes* to the SW activation at that position
  Negative value → token t *suppresses* the SW activation

Then averages omission profiles across sequences by genomic context,
interpolated to a common 100-position fractional axis.

This is cleaner than gradient saliency (no backprop assumptions, no saturation)
and directly measures functional impact rather than linear sensitivity.

Window: 384 bp (64 tokens) — short enough for efficient batching.

Usage:
    python scripts/run_sw_token_omission.py \
        --n_seqs 10 \
        --window_bp 384 \
        --promoters ../data/regions/hg38/promoters_262kb.bed \
        --enhancers ../data/regions/hg38/enhancers_ccre_262kb.bed \
        --random    ../data/regions/hg38/random_262kb.bed \
        --out  results/sw_token_omission.json \
        --plot results/sw_token_omission.png
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POLY_A_TOKEN = 32   # AAAAAA token id in GENERator vocab


# ─────────────────────────────────────────────────────────────────────────────
# FASTA + BED helpers
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
        end   = n
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
# Tokenize
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(tokenizer, seq: str, device: str) -> torch.Tensor:
    remainder = len(seq) % 6
    if remainder:
        seq = seq[remainder:]
    return tokenizer(seq, return_tensors="pt",
                     add_special_tokens=False)["input_ids"].to(device)   # [1, L]


# ─────────────────────────────────────────────────────────────────────────────
# Single-pass omission attribution
# ─────────────────────────────────────────────────────────────────────────────

def _omission_attribution(model, tokenizer, seq: str,
                           sw_layer: int, sw_rows: list,
                           device: str) -> dict | None:
    """
    Returns:
        baseline_act  : [L, n_sw_rows]  — activation with original sequence
        omission_act  : [L, L, n_sw_rows] — omission_act[t] = activation when token t is omitted
        omission_effect : [L, n_sw_rows] — baseline_act - mean(omission_act diag)
    """
    ids = _tokenize(tokenizer, seq, device)   # [1, L]
    L   = ids.shape[1]
    if L < 2:
        return None

    target = model.model.layers[sw_layer].mlp.down_proj
    n_rows = len(sw_rows)

    # ── Baseline forward pass ─────────────────────────────────────────────────
    _store = {}
    def _hook_base(_mod, _inp, _out, _s=_store):
        _s["act"] = _out.detach().cpu().float()   # [1, L, hidden]
    h = target.register_forward_hook(_hook_base)
    with torch.no_grad():
        model(input_ids=ids)
    h.remove()
    baseline_full = _store["act"][0, :, :][:, sw_rows]   # [L, n_rows]

    # ── Omission batch: L copies, each with one token replaced ────────────────
    omit_ids = ids.repeat(L, 1)                   # [L, L]
    for t in range(L):
        omit_ids[t, t] = POLY_A_TOKEN             # omit token t

    _acts_list = []
    # Process in sub-batches to avoid memory issues with large L
    sub_bs = 64
    for start in range(0, L, sub_bs):
        chunk = omit_ids[start : start + sub_bs].to(device)
        _s2   = {}
        def _hook_omit(_mod, _inp, _out, _s=_s2):
            _s["act"] = _out.detach().cpu().float()
        h2 = target.register_forward_hook(_hook_omit)
        with torch.no_grad():
            model(input_ids=chunk)
        h2.remove()
        _acts_list.append(_s2["act"])   # [sub_bs, L, hidden]

    omit_acts_full = torch.cat(_acts_list, dim=0)   # [L, L, hidden]
    # omit_acts_full[t, pos, row] = activation at position pos when token t is omitted

    # We want per-position omission effect: for each position t, how much does
    # omitting token t change the activation AT POSITION t?
    # diagonal: omit_acts_full[t, t, :]
    diag_acts = omit_acts_full[torch.arange(L), torch.arange(L), :]  # [L, hidden]
    diag_sw   = diag_acts[:, sw_rows]                                 # [L, n_rows]

    omission_effect = (baseline_full - diag_sw).numpy()   # [L, n_rows] positive = token contributes

    return {
        "baseline_act":    baseline_full.numpy(),
        "omission_effect": omission_effect,
        "L":               L,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(results: list, out_path: str, sw_info: str, sw_rows: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    from collections import defaultdict
    FRAC_AXIS = np.linspace(0, 1, 100)

    # Interpolate each sequence's omission profile to the common fractional axis
    ctx_profiles = defaultdict(list)
    for r in results:
        eff = np.array(r["omission_effect"])   # [L, n_rows]
        L   = eff.shape[0]
        if L < 2:
            continue
        score = eff.mean(axis=1)               # [L] — mean over SW rows
        frac  = np.linspace(0, 1, L)
        try:
            interp = interp1d(frac, score, kind="linear", fill_value="extrapolate")
            ctx_profiles[r["label"]].append(interp(FRAC_AXIS))
        except Exception:
            pass

    contexts = sorted(ctx_profiles.keys())
    ctx_colours = {"promoter": "steelblue", "enhancer": "darkorange", "random": "gray"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean omission effect profiles per context
    ax = axes[0]
    for ctx in contexts:
        mats  = np.array(ctx_profiles[ctx])    # [n_seqs, 100]
        mean  = mats.mean(axis=0)
        std   = mats.std(axis=0)
        x     = FRAC_AXIS
        col   = ctx_colours.get(ctx, "purple")
        ax.plot(x, mean, label=ctx, color=col, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=col)
    ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Fractional position in sequence")
    ax.set_ylabel("Omission effect  (↑ = token contributes to SW activation)")
    ax.set_title(f"Token omission attribution\n{sw_info}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: heatmap of top-10 sequences by max omission effect
    top_recs = sorted(results, key=lambda r: np.max(r["omission_effect"]), reverse=True)[:10]
    hm_data  = []
    hm_ylabels = []
    for r in top_recs:
        eff  = np.array(r["omission_effect"]).mean(axis=1)   # [L]
        L    = len(eff)
        frac = np.linspace(0, 1, L)
        try:
            from scipy.interpolate import interp1d
            interp = interp1d(frac, eff, kind="linear", fill_value="extrapolate")
            hm_data.append(interp(FRAC_AXIS))
        except Exception:
            hm_data.append(np.zeros(100))
        hm_ylabels.append(f"{r['label'][:3]} {r['chrom']}:{r['center']//1000}k")
    hm_arr = np.array(hm_data)
    im = axes[1].imshow(hm_arr, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    axes[1].set_yticks(range(len(hm_ylabels)))
    axes[1].set_yticklabels(hm_ylabels, fontsize=7)
    axes[1].set_xlabel("Fractional position")
    axes[1].set_title("Top-10 sequences by peak omission effect")
    plt.colorbar(im, ax=axes[1], label="omission effect")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta",       default="/home/nvidia/data/hg38/hg38.fa")
    p.add_argument("--promoters",   default="data/regions/hg38/promoters_262kb.bed")
    p.add_argument("--enhancers",   default="data/regions/hg38/enhancers_ccre_262kb.bed")
    p.add_argument("--random",      default="data/regions/hg38/random_262kb.bed")
    p.add_argument("--n_seqs",      type=int, default=10,
                   help="Sequences per context (batching is expensive; default=10)")
    p.add_argument("--window_bp",   type=int, default=384,
                   help="Window size in bp; shorter = faster batching")
    p.add_argument("--model",       default="generator",
                   choices=["generator", "generator_prokaryote",
                            "generator_prokaryote_1b"])
    p.add_argument("--sw_index",    default="results/super_weight_index.json")
    p.add_argument("--configs_dir", default="configs")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default="results/sw_token_omission.json")
    p.add_argument("--plot",        default="results/sw_token_omission.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)
    args.window_bp = (args.window_bp // 6) * 6

    # ── Load config + SW ──────────────────────────────────────────────────────
    cfg      = yaml.safe_load(open(Path(args.configs_dir) / f"{args.model}.yaml"))
    sw_data  = json.load(open(args.sw_index))
    sw_entry = sw_data.get(args.model, {})
    sw_list  = sw_entry.get("results", []) if isinstance(sw_entry, dict) else sw_entry
    if not sw_list:
        print(f"No SW found for {args.model}.")
        sys.exit(1)

    top_sw   = max(sw_list, key=lambda x: x["out_max"])
    sw_layer = int(top_sw["layer"])
    sw_rows  = sorted({int(s["row"]) for s in sw_list if s["layer"] == sw_layer})
    print(f"  SW: layer={sw_layer}  rows={sw_rows}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"  Loading {args.model} ...", flush=True)
    from models.generator_wrapper import GeneratorWrapper
    wrapper   = GeneratorWrapper(cfg)
    wrapper.load()
    model     = wrapper.model
    tokenizer = wrapper.tokenizer
    model.eval()

    # ── Load sequences ────────────────────────────────────────────────────────
    print("  Loading FASTA ...", flush=True)
    fasta = _open_fasta(args.fasta)

    bed_files = [
        ("promoter", args.promoters),
        ("enhancer", args.enhancers),
        ("random",   args.random),
    ]
    sequences = []
    for label, bed_path in bed_files:
        if not Path(bed_path).exists():
            print(f"  [skip] BED not found: {bed_path}")
            continue
        seqs = _sample_bed_windows(bed_path, fasta, args.n_seqs, args.window_bp, rng, label)
        print(f"  {label:10s}: {len(seqs)} sequences  ({args.window_bp} bp)")
        sequences.extend(seqs)

    if not sequences:
        print("No sequences loaded. Check BED paths.")
        sys.exit(1)

    # ── Run omission attribution ──────────────────────────────────────────────
    print(f"\n  Running token omission on {len(sequences)} sequences ...", flush=True)
    results = []
    for i, rec in enumerate(sequences):
        out = _omission_attribution(model, tokenizer, rec["seq"],
                                    sw_layer, sw_rows, args.device)
        if out is None:
            continue
        eff    = out["omission_effect"]            # [L, n_rows]
        L      = out["L"]
        score  = eff.mean(axis=1)                  # [L]
        top_t  = int(np.argmax(score))
        # decode top token
        ids    = _tokenize(tokenizer, rec["seq"], "cpu")[0]
        top_kmer = tokenizer.decode([int(ids[top_t])]) if top_t < len(ids) else "?"

        print(f"  [{i+1}/{len(sequences)}] {rec['label']:10s} {rec['chrom']}:{rec['center']}"
              f"  L={L}  max_effect={score.max():+.1f}  mean_effect={score.mean():+.3f}"
              f"  top_pos={top_t}/{L} ({top_kmer})")

        results.append({
            "label":           rec["label"],
            "chrom":           rec["chrom"],
            "center":          rec["center"],
            "L":               L,
            "omission_effect": eff.tolist(),        # [L, n_rows]
            "effect_score":    score.tolist(),      # [L]  mean over rows
            "top_token_pos":   top_t,
            "top_token_kmer":  top_kmer,
            "sw_rows":         sw_rows,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        all_max  = [np.max(r["effect_score"])  for r in results]
        all_mean = [np.mean(r["effect_score"]) for r in results]
        print(f"\n  === Summary ===")
        print(f"  Max omission effect:  {np.mean(all_max):.2f} ± {np.std(all_max):.2f}")
        print(f"  Mean omission effect: {np.mean(all_mean):.4f} ± {np.std(all_mean):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model":    args.model,
            "sw_layer": sw_layer,
            "sw_rows":  sw_rows,
            "window_bp": args.window_bp,
            "n_seqs":   len(results),
            "results":  results,
        }, f, indent=2)
    print(f"  Results saved → {args.out}")

    sw_info = f"layer={sw_layer}  rows={sw_rows}"
    _plot(results, args.plot, sw_info, sw_rows)


if __name__ == "__main__":
    main()
