"""
scripts/run_sw_gradient_attribution.py
---------------------------------------
Gradient-based saliency attribution for super-weight rows in GENERator.

For each genomic sequence (drawn from promoter / enhancer / random region BED files),
we compute:

    saliency[t] = || d( sum |SW_activation| ) / d( embedding[t] ) ||_2

i.e. the L2 norm of the gradient of the SW row's total activation magnitude with
respect to each token's embedding vector.  This tells us which token positions in
the input sequence causally drive the super-weight spike.

What this answers:
  "Which k-mer positions in a genomic sequence are most responsible for
   triggering the super-weight activation in GENERator?"

Biological interpretation signal:
  - Peaks near the TSS / TATA box → SW encodes transcription-initiation context
  - High saliency at exon–intron boundaries → SW encodes splicing context
  - Uniform saliency → SW is a generic sequence-length or GC-content detector

Usage:
    python scripts/run_sw_gradient_attribution.py \\
        --fasta /home/nvidia/data/hg38/hg38.fa \\
        --promoters data/regions/hg38/promoters_262kb.bed \\
        --enhancers data/regions/hg38/enhancers_ccre_262kb.bed \\
        --random    data/regions/hg38/random_262kb.bed \\
        --n_seqs 50 \\
        --window_bp 3072 \\
        --out results/sw_grad_attribution.json \\
        --plot results/sw_grad_attribution.png

Outputs:
    JSON : per-sequence saliency vectors + metadata
    PNG  : mean saliency ± std per position grouped by genomic context
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

# ─────────────────────────────────────────────────────────────────────────────
# FASTA / BED helpers  (adapted from borzoi.genome, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _open_fasta(path: str):
    try:
        from pyfaidx import Fasta
        return Fasta(path, as_raw=True, sequence_always_upper=True)
    except ImportError:
        raise ImportError("pyfaidx is required: conda run -n evo pip install pyfaidx")


def _fetch_window(fasta, chrom: str, center: int, window_bp: int) -> str:
    """Fetch window_bp sequence centered on `center` (0-based)."""
    half = window_bp // 2
    start = max(0, center - half)
    end   = start + window_bp
    key   = chrom if chrom in fasta else ("chr" + chrom if "chr" + chrom in fasta
                                          else chrom.lstrip("chr") if chrom.lstrip("chr") in fasta
                                          else None)
    if key is None:
        return None
    chrom_len = len(fasta[key])
    if end > chrom_len:
        end   = chrom_len
        start = max(0, end - window_bp)
    seq = str(fasta[key][start:end])
    # Pad to exact length with N if near chromosome edge
    if len(seq) < window_bp:
        seq = seq + "N" * (window_bp - len(seq))
    return seq


def _sample_bed_windows(bed_path: str, fasta, n: int, window_bp: int,
                        rng: random.Random, label: str) -> list:
    """Sample n regions from a BED file, return list of (label, chrom, center, seq)."""
    with open(bed_path) as f:
        lines = [l.strip().split("\t") for l in f if l.strip() and not l.startswith("#")]
    if len(lines) > 10 * n:
        lines = rng.sample(lines, 10 * n)
    results = []
    rng.shuffle(lines)
    for row in lines:
        if len(results) >= n:
            break
        chrom = row[0]
        s, e  = int(row[1]), int(row[2])
        center = (s + e) // 2
        seq = _fetch_window(fasta, chrom, center, window_bp)
        if seq is None:
            continue
        # Skip if > 10% N
        n_frac = seq.count("N") / len(seq)
        if n_frac > 0.1:
            continue
        results.append({"label": label, "chrom": chrom, "center": center, "seq": seq})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tokenize for GENERator (6-mer, no special tokens)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_generator(tokenizer, seq: str, device: str) -> torch.Tensor:
    """Return token IDs [1, L] for a DNA sequence with GENERator's 6-mer tokenizer."""
    # Length must be divisible by 6
    remainder = len(seq) % 6
    if remainder:
        seq = seq[remainder:]
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
    return inputs["input_ids"].to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient saliency computation
# ─────────────────────────────────────────────────────────────────────────────

def _sw_saliency(model, tokenizer, seq: str, sw_layer: int, sw_row: int,
                 device: str) -> np.ndarray:
    """Compute per-token saliency = ||d(SW_activation_magnitude)/d(embed[t])||_2.

    Returns np.ndarray of shape [n_tokens].
    """
    # Tokenize
    input_ids = _tokenize_generator(tokenizer, seq, device)  # [1, L]
    if input_ids.shape[1] < 2:
        return None

    # Hook into embed_tokens output to capture embeddings with gradient.
    # Using inputs_embeds directly breaks device_map="auto" (multi-GPU), so
    # we hook into embed_tokens and call retain_grad() on its output instead.
    _captured = {}

    def _embed_hook(_module, _inp, _out):
        _out.retain_grad()
        _captured["embed"] = _out

    def _sw_hook(_module, _inp, _out):
        _captured["act"] = _out  # [1, L, hidden_size]

    target_module = model.model.layers[sw_layer].mlp.down_proj
    h_embed = model.model.embed_tokens.register_forward_hook(_embed_hook)
    h_sw    = target_module.register_forward_hook(_sw_hook)

    try:
        with torch.enable_grad():
            _ = model(input_ids=input_ids)
    finally:
        h_embed.remove()
        h_sw.remove()

    embed_out = _captured.get("embed")
    sw_act    = _captured.get("act")
    if embed_out is None or sw_act is None:
        return None

    # Target: total abs activation at the SW row across all positions
    scalar = sw_act[..., sw_row].abs().sum()
    scalar.backward()

    if embed_out.grad is None:
        return None

    # Saliency: L2 norm over embedding dimension
    saliency = embed_out.grad.norm(dim=-1).squeeze(0)  # [L]
    return saliency.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(results: list, out_path: str, window_bp: int, sw_info: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    from collections import defaultdict
    by_label = defaultdict(list)
    for r in results:
        sal = np.array(r["saliency"])
        # Normalize per sequence (so magnitude differences don't dominate)
        span = sal.max() - sal.min()
        if span > 0:
            sal = (sal - sal.min()) / span
        by_label[r["label"]].append(sal)

    colors = {"promoter": "steelblue", "enhancer": "darkorange", "random": "gray"}
    fig, ax = plt.subplots(figsize=(12, 4))

    for label, saliencies in by_label.items():
        # Interpolate to common length (100 fractional positions)
        resampled = []
        for s in saliencies:
            xs = np.linspace(0, 1, len(s))
            xq = np.linspace(0, 1, 100)
            resampled.append(np.interp(xq, xs, s))
        mat  = np.array(resampled)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        xpos = np.linspace(-window_bp // 2, window_bp // 2, 100)
        ax.plot(xpos, mean, label=f"{label} (n={len(saliencies)})",
                color=colors.get(label, "black"), linewidth=1.5)
        ax.fill_between(xpos, mean - std, mean + std,
                        alpha=0.15, color=colors.get(label, "black"))

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
               label="region center")
    ax.set_xlabel(f"Position relative to region center (bp)")
    ax.set_ylabel("Normalised saliency (mean ± std)")
    ax.set_title(f"GENERator SW row gradient saliency\n{sw_info}", fontsize=11)
    ax.legend(fontsize=9)
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
    p.add_argument("--fasta",     default="/home/nvidia/data/hg38/hg38.fa")
    p.add_argument("--promoters", default="data/regions/hg38/promoters_262kb.bed")
    p.add_argument("--enhancers", default="data/regions/hg38/enhancers_ccre_262kb.bed")
    p.add_argument("--random",    default="data/regions/hg38/random_262kb.bed")
    p.add_argument("--n_seqs",    type=int, default=50,
                   help="Number of sequences per genomic context (promoter/enhancer/random)")
    p.add_argument("--window_bp", type=int, default=3072,
                   help="Sequence window size in bp (must be divisible by 6 for GENERator)")
    p.add_argument("--model",     default="generator",
                   choices=["generator", "generator_prokaryote", "generator_prokaryote_1b"])
    p.add_argument("--sw_index",  default="results/super_weight_index.json")
    p.add_argument("--configs_dir", default="configs")
    p.add_argument("--device",    default="cuda")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out",       default="results/sw_grad_attribution.json")
    p.add_argument("--plot",      default="results/sw_grad_attribution.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    # Ensure window_bp is divisible by 6
    args.window_bp = (args.window_bp // 6) * 6

    # ── Load config and SW index ──────────────────────────────────────────────
    cfg_path = Path(args.configs_dir) / f"{args.model}.yaml"
    cfg      = yaml.safe_load(open(cfg_path))

    sw_data  = json.load(open(args.sw_index))
    sw_entry = sw_data.get(args.model, {})
    sw_list  = sw_entry.get("results", []) if isinstance(sw_entry, dict) else sw_entry
    if not sw_list:
        print(f"No SW found for {args.model} in {args.sw_index}. Run run_detection.py first.")
        sys.exit(1)

    # Use the top SW (highest out_max)
    top_sw  = max(sw_list, key=lambda x: x["out_max"])
    sw_layer = int(top_sw["layer"])
    sw_row   = int(top_sw["row"])
    print(f"  Primary SW: layer={sw_layer}  row={sw_row}  out_max={top_sw['out_max']:.0f}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"  Loading {args.model} ...", flush=True)
    from models.generator_wrapper import GeneratorWrapper
    wrapper   = GeneratorWrapper(cfg)
    wrapper.load()
    model     = wrapper.model.to(args.device)
    tokenizer = wrapper.tokenizer
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters())//1e6:.0f}M params)")

    # ── Load genomic regions ──────────────────────────────────────────────────
    print(f"  Loading FASTA from {args.fasta} ...", flush=True)
    fasta = _open_fasta(args.fasta)

    contexts = [
        ("promoter", args.promoters),
        ("enhancer", args.enhancers),
        ("random",   args.random),
    ]
    sequences = []
    for label, bed_path in contexts:
        if not Path(bed_path).exists():
            print(f"  [skip] BED not found: {bed_path}")
            continue
        seqs = _sample_bed_windows(bed_path, fasta, args.n_seqs, args.window_bp, rng, label)
        print(f"  {label:10s}: {len(seqs)} sequences sampled (window={args.window_bp} bp)")
        sequences.extend(seqs)

    if not sequences:
        print("No sequences loaded. Check BED paths.")
        sys.exit(1)

    # ── Compute saliency ─────────────────────────────────────────────────────
    print(f"\n  Computing gradient saliency for {len(sequences)} sequences ...")
    results = []
    for i, rec in enumerate(sequences):
        try:
            sal = _sw_saliency(model, tokenizer, rec["seq"], sw_layer, sw_row, args.device)
        except Exception as e:
            print(f"    [{i+1}/{len(sequences)}] {rec['label']} {rec['chrom']}:{rec['center']}  ERROR: {e}")
            continue
        if sal is None:
            continue
        n_toks = len(sal)
        print(f"    [{i+1}/{len(sequences)}] {rec['label']:10s} {rec['chrom']}:{rec['center']}  "
              f"tokens={n_toks}  sal_max={sal.max():.4f}  sal_mean={sal.mean():.4f}")
        results.append({
            "label":    rec["label"],
            "chrom":    rec["chrom"],
            "center":   rec["center"],
            "n_tokens": n_toks,
            "saliency": sal.tolist(),
        })

    # ── Save JSON ─────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model":    args.model,
        "sw_layer": sw_layer,
        "sw_row":   sw_row,
        "window_bp": args.window_bp,
        "n_seqs": len(results),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Results saved → {args.out}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    sw_info = f"layer={sw_layer}  row={sw_row}"
    _plot(results, args.plot, args.window_bp, sw_info)


if __name__ == "__main__":
    main()
