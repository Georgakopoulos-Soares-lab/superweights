"""
scripts/run_sw_shuffle_controls.py
-------------------------------------
Stronger shuffle controls for the SW activation-distribution shift experiment
(Experiment B in run_sw_causal_tracing.py).

Background
-----------
The original causal tracing script compared SW row activation on real genomic
sequences vs. their *dinucleotide-shuffled* counterparts and found no significant
shift (mean shift = +0.0002, p = 0.15, n=90).  The null was attributed to
GENERator's 6-mer tokenizer: each token spans exactly one 6-mer, so
dinucleotide shuffling (which preserves 2-mer frequencies) already preserves
most of the local 6-mer composition that the model sees.

This script tests three *progressively stronger* shuffle types that destroy
increasing levels of sequence context:

  1. dinuc_shuffle  — dinucleotide-preserving (baseline, reproduced here for
                      direct comparison with the original result)
  2. mono_shuffle   — mononucleotide (i.i.d.) shuffle: destroys all k>1 context
                      while preserving per-base GC composition.  Since 6-mer
                      frequencies are no longer preserved, this should reveal
                      whether the SW is sensitive to any above-monomer context.
  3. kmer_block_shuffle — block-shuffle at the token level: the sequence of
                      6-mer tokens is randomly permuted (token order destroyed,
                      but each individual 6-mer is present exactly once).  This
                      preserves the per-token identity distribution perfectly
                      while destroying all positional/context information.

Prediction
----------
  - If mono_shuffle produces a significant shift: the SW is sensitive to local
    k-mer composition above the 2-mer level.
  - If kmer_block_shuffle produces a significant shift but mono does not: the SW
    is sensitive to *token identity* but not to order within the 6-mer.
  - If neither: the SW fires identically regardless of sequence context — it is
    a generic sequence-length / token-count detector.

Metrics
-------
For each sequence and each shuffle type:
    shift = (mean_real_activation - mean_shuffled_activation) /
            (std_real + std_shuffled + ε)

Two-sided t-test across sequences: H₀ shift = 0.

Usage
-----
    python scripts/run_sw_shuffle_controls.py \\
        --fasta     /home/nvidia/data/hg38/hg38.fa \\
        --promoters data/regions/hg38/promoters_262kb.bed \\
        --enhancers data/regions/hg38/enhancers_ccre_262kb.bed \\
        --random    data/regions/hg38/random_262kb.bed \\
        --n_seqs    30 \\
        --window_bp 3072 \\
        --n_shuffles 5 \\
        --out  results/sw_shuffle_controls.json \\
        --plot results/sw_shuffle_controls.png

GPU required.
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
# FASTA + BED helpers  (same pattern as run_sw_causal_tracing.py)
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
# Shuffle implementations
# ─────────────────────────────────────────────────────────────────────────────

def _mono_shuffle(seq: str, rng: random.Random) -> str:
    """Independent (i.i.d.) per-base shuffle — destroys all k>1 context."""
    bases = list(seq.upper())
    rng.shuffle(bases)
    return "".join(bases)


def _dinuc_shuffle(seq: str, rng: random.Random) -> str:
    """
    Dinucleotide-preserving shuffle (Altschul–Erickson).
    Reproduced from run_sw_causal_tracing.py for direct comparison.
    """
    VALID = set("ACGT")
    bases = [b if b in VALID else "N" for b in seq.upper()]
    if len(bases) < 4:
        rng.shuffle(bases)
        return "".join(bases)
    from collections import defaultdict
    transitions = defaultdict(list)
    for i in range(len(bases) - 1):
        transitions[bases[i]].append(bases[i + 1])
    for k in transitions:
        rng.shuffle(transitions[k])
    result  = [bases[0]]
    current = bases[0]
    for _ in range(len(bases) - 1):
        nexts = transitions.get(current)
        if not nexts:
            remaining = [b for v in transitions.values() for b in v]
            if not remaining:
                break
            nxt = rng.choice(remaining)
            for k in transitions:
                if nxt in transitions[k]:
                    transitions[k].remove(nxt)
                    break
        else:
            nxt = nexts.pop(0)
        result.append(nxt)
        current = nxt
    if len(result) < len(bases):
        pool = list(seq.upper())
        rng.shuffle(pool)
        result.extend(pool[:len(bases) - len(result)])
    return "".join(result[:len(bases)])


def _kmer_block_shuffle(seq: str, k: int, rng: random.Random) -> str:
    """
    Permute the sequence of non-overlapping k-mer tokens uniformly at random.
    Each token is kept intact; only the token *order* is randomised.
    Preserves: per-token identity distribution (same bag of k-mers).
    Destroys: positional order and all inter-token context.
    """
    # Trim to exact multiple of k
    trim = len(seq) - (len(seq) % k)
    seq_t = seq.upper()[:trim]
    tokens = [seq_t[i:i+k] for i in range(0, trim, k)]
    rng.shuffle(tokens)
    return "".join(tokens)


def _trinuc_shuffle(seq: str, rng: random.Random) -> str:
    """
    Trinucleotide-preserving shuffle via random Eulerian path in De Bruijn graph.

    Formalisation
    -------------
    Build a directed multigraph:
      nodes  = distinct dinucleotides (2-mers) appearing in the sequence
      edges  = one edge per overlapping trinucleotide (XY → YZ for each XYZ)

    The original sequence IS an Eulerian path through this graph.  Any other
    Eulerian path through the same multigraph is a valid shuffle with identical
    trinucleotide frequencies (Altschul–Erickson, CABIOS 1985, generalised to
    k=3).  We randomise the path by shuffling each node's out-edge list before
    running Hierholzer's algorithm.

    Preserves: exact per-trinucleotide (3-mer) counts.
    Destroys : positional order of trinucleotides and all higher-order context.

    Why this matters for the paper
    ------------------------------
    GENERator uses non-overlapping 6-mer tokens.  A dinucleotide shuffle already
    preserves ≥90 % of hexamer composition (because each hexamer overlaps many
    2-mers).  Trinucleotide preservation is a strictly stronger control: if SW
    activation is still flat on trinucleotide-shuffled sequences (p > 0.05),
    we can rule out any residual 3-mer composition effect and conclude the SW
    fires purely on token identity — not on context above the 3-mer level.
    """
    from collections import defaultdict

    VALID = set("ACGT")
    bases = [b if b in VALID else "N" for b in seq.upper()]
    n = len(bases)
    if n < 3:
        rng.shuffle(bases)
        return "".join(bases)

    # Build adjacency list: dinucleotide node → [next character, ...]
    adj: dict[str, list] = defaultdict(list)
    for i in range(n - 2):
        di = bases[i] + bases[i + 1]
        adj[di].append(bases[i + 2])

    # Shuffle each node's edge list → randomises which Eulerian path we take
    for node in adj:
        rng.shuffle(adj[node])

    # Index-tracked copies for Hierholzer's algorithm
    adj_lst = {node: list(edges) for node, edges in adj.items()}
    adj_idx = {node: 0 for node in adj_lst}

    # Hierholzer's algorithm: find a random Eulerian PATH starting at seq[0:2]
    start = bases[0] + bases[1]
    stack: list[str] = [start]
    path_nodes: list[str] = []

    while stack:
        u = stack[-1]
        if adj_idx.get(u, 0) < len(adj_lst.get(u, [])):
            c = adj_lst[u][adj_idx[u]]
            adj_idx[u] += 1
            stack.append(u[1] + c)       # next dinucleotide node
        else:
            path_nodes.append(stack.pop())

    path_nodes = path_nodes[::-1]         # Hierholzer returns path in reverse

    # Reconstruct sequence: first dinucleotide + last char of each subsequent node
    if not path_nodes:
        return "".join(bases)

    result = list(path_nodes[0])          # 2 characters
    for node in path_nodes[1:]:
        result.append(node[-1])

    # Safety pad (should never trigger on valid DNA)
    if len(result) < n:
        pool = bases[:]
        rng.shuffle(pool)
        result.extend(pool[: n - len(result)])

    return "".join(result[:n])


SHUFFLE_TYPES = {
    "dinuc":      _dinuc_shuffle,
    "mono":       _mono_shuffle,
    "kmer_block": _kmer_block_shuffle,
    "trinuc":     _trinuc_shuffle,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tokenize
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(tokenizer, seq: str, device: str) -> torch.Tensor:
    remainder = len(seq) % 6
    if remainder:
        seq = seq[remainder:]
    return tokenizer(seq, return_tensors="pt",
                     add_special_tokens=False)["input_ids"].to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Capture SW activation
# ─────────────────────────────────────────────────────────────────────────────

def _capture_sw_activation(model, input_ids: torch.Tensor,
                            sw_layer: int, sw_rows: list) -> np.ndarray:
    """Return SW row activations as 1-D numpy array [n_tokens * n_sw_rows]."""
    _store = {}
    def _hook(_mod, _inp, _out):
        _store["act"] = _out[..., sw_rows].detach().cpu().float().numpy()
    h = model.model.layers[sw_layer].mlp.down_proj.register_forward_hook(_hook)
    with torch.no_grad():
        model(input_ids=input_ids)
    h.remove()
    return _store["act"].ravel()


def _activation_shift(real_acts: np.ndarray, shuf_acts: np.ndarray) -> float:
    """Standardised mean difference: (μ_real − μ_shuf) / (σ_real + σ_shuf + ε)."""
    mu_r = float(real_acts.mean())
    mu_s = float(shuf_acts.mean())
    sig  = float(real_acts.std()) + float(shuf_acts.std()) + 1e-9
    return (mu_r - mu_s) / sig


# ─────────────────────────────────────────────────────────────────────────────
# T-test (two-sided, H₀: mean shift = 0)
# ─────────────────────────────────────────────────────────────────────────────

def _t_test_shifts(shifts: list) -> tuple:
    import math
    vals = [v for v in shifts if not math.isnan(v)]
    n    = len(vals)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(vals) / n
    var  = sum((v - mean) ** 2 for v in vals) / (n - 1)
    if var == 0:
        return float("inf"), 0.0
    t = mean / (var / n) ** 0.5
    try:
        from scipy.stats import t as tdist
        p = float(tdist.sf(abs(t), df=n - 1) * 2)
    except ImportError:
        def _norm_sf(z):
            return 0.5 * math.erfc(z / math.sqrt(2))
        p = _norm_sf(abs(t)) * 2
    return float(t), float(p)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(seq_results: list, out_path: str, sw_info: str, n_shuffles: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    shuffle_types = list(SHUFFLE_TYPES.keys())
    ctx_colours = {"promoter": "steelblue", "enhancer": "darkorange", "random": "gray"}
    contexts    = sorted({r["label"] for r in seq_results})
    n_types     = len(shuffle_types)

    fig = plt.figure(figsize=(5 * n_types + 2, 9))
    gs  = GridSpec(2, n_types, figure=fig, hspace=0.5, wspace=0.35)

    label_to_col = {ctx: ctx_colours.get(ctx, "purple") for ctx in contexts}

    # ── Per-shuffle-type violin strip ─────────────────────────────────────────
    for ti, stype in enumerate(shuffle_types):
        ax = fig.add_subplot(gs[0, ti])
        pos_x = np.arange(len(contexts))
        for xi, ctx in enumerate(contexts):
            shifts = [r["shifts"][stype] for r in seq_results
                      if r["label"] == ctx and stype in r["shifts"]]
            if not shifts:
                continue
            col = label_to_col[ctx]
            jitter = np.random.RandomState(xi).uniform(-0.15, 0.15, len(shifts))
            ax.scatter(xi + jitter, shifts, s=25, alpha=0.6, color=col, zorder=3)
            ax.plot([xi - 0.25, xi + 0.25], [float(np.mean(shifts))] * 2,
                    color="black", linewidth=2.5, zorder=4)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(pos_x)
        ax.set_xticklabels(contexts, fontsize=9)
        ax.set_ylabel("Activation shift (std. units)")
        ax.set_title(f"{stype} shuffle\n(n_shuffles={n_shuffles})", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    # ── Summary bar: mean shift per shuffle type across all contexts ──────────
    ax_bar = fig.add_subplot(gs[1, :])
    type_means = []
    type_sems  = []
    type_ps    = []
    for stype in shuffle_types:
        all_shifts = [r["shifts"][stype] for r in seq_results if stype in r["shifts"]]
        type_means.append(float(np.mean(all_shifts)) if all_shifts else 0.0)
        type_sems.append(float(np.std(all_shifts) / max(len(all_shifts) ** 0.5, 1)))
        _, p = _t_test_shifts(all_shifts)
        type_ps.append(p)

    colours_bar = ["#d73027" if p < 0.05 else "#4575b4" for p in type_ps]
    bars = ax_bar.bar(range(n_types), type_means, color=colours_bar,
                      yerr=type_sems, capsize=6)

    for xi, (m, p) in enumerate(zip(type_means, type_ps)):
        sig = ("***" if p < 0.001 else "**" if p < 0.01
               else "*" if p < 0.05 else "n.s.")
        ax_bar.text(xi, m + (type_sems[xi] if m >= 0 else -type_sems[xi]) + 0.005,
                    sig, ha="center", va="bottom" if m >= 0 else "top", fontsize=11)

    ax_bar.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_bar.set_xticks(range(n_types))
    ax_bar.set_xticklabels([f"{t}\np={p:.4f}" for t, p in zip(shuffle_types, type_ps)],
                            fontsize=10)
    ax_bar.set_ylabel("Mean activation shift (all contexts, std. units)")
    ax_bar.set_title(f"SW activation shift: real vs shuffled — comparison of shuffle types\n{sw_info}",
                     fontsize=11)
    ax_bar.grid(True, alpha=0.3, axis="y")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stronger shuffle controls for SW activation shift experiment"
    )
    p.add_argument("--fasta",       default="/home/nvidia/data/hg38/hg38.fa")
    p.add_argument("--promoters",   default="data/regions/hg38/promoters_262kb.bed")
    p.add_argument("--enhancers",   default="data/regions/hg38/enhancers_ccre_262kb.bed")
    p.add_argument("--random",      default="data/regions/hg38/random_262kb.bed")
    p.add_argument("--n_seqs",      type=int, default=30,
                   help="Sequences per genomic context")
    p.add_argument("--window_bp",   type=int, default=3072)
    p.add_argument("--n_shuffles",  type=int, default=5,
                   help="Shuffles per sequence (mean is taken)")
    p.add_argument("--model",       default="generator",
                   choices=["generator", "generator_prokaryote",
                            "generator_prokaryote_1b"])
    p.add_argument("--shuffle_types", nargs="+",
                   default=["dinuc", "mono", "kmer_block", "trinuc"],
                   choices=list(SHUFFLE_TYPES.keys()),
                   help="Which shuffle types to run")
    p.add_argument("--sw_index",    default="results/super_weight_index.json")
    p.add_argument("--configs_dir", default="configs")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default="results/sw_shuffle_controls.json")
    p.add_argument("--plot",        default="results/sw_shuffle_controls.png")
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
    sw_info  = f"{args.model}  layer={sw_layer}  rows={sw_rows}"
    print(f"  SW: {sw_info}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"  Loading {args.model} ...", flush=True)
    from models.generator_wrapper import GeneratorWrapper
    wrapper   = GeneratorWrapper(cfg)
    wrapper.load()
    model     = wrapper.model
    tokenizer = wrapper.tokenizer
    model.eval()
    # Derive input device from the embedding layer (works with both single-GPU
    # and multi-GPU accelerate dispatch — do NOT call model.to() after load)
    args.device = str(next(model.parameters()).device)

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
        seqs = _sample_bed_windows(bed_path, fasta, args.n_seqs,
                                   args.window_bp, rng, label)
        print(f"  {label:10s}: {len(seqs)} sequences  ({args.window_bp} bp)")
        sequences.extend(seqs)

    if not sequences:
        print("No sequences loaded. Check BED paths.")
        sys.exit(1)

    # ── Run shuffle controls ──────────────────────────────────────────────────
    print(f"\n  Running {len(args.shuffle_types)} shuffle type(s) × "
          f"{args.n_shuffles} shuffles × {len(sequences)} sequences ...", flush=True)

    seq_results = []
    for i, rec in enumerate(sequences):
        seq = rec["seq"]
        ids_real = _tokenize(tokenizer, seq, args.device)
        real_acts = _capture_sw_activation(model, ids_real, sw_layer, sw_rows)

        shifts = {}
        for stype in args.shuffle_types:
            shuffle_fn = SHUFFLE_TYPES[stype]
            all_shifts = []
            for _ in range(args.n_shuffles):
                if stype == "kmer_block":
                    shuf_seq = shuffle_fn(seq, k=6, rng=rng)
                else:
                    shuf_seq = shuffle_fn(seq, rng)
                ids_shuf  = _tokenize(tokenizer, shuf_seq, args.device)
                shuf_acts = _capture_sw_activation(model, ids_shuf, sw_layer, sw_rows)
                all_shifts.append(_activation_shift(real_acts, shuf_acts))
            shifts[stype] = float(np.mean(all_shifts))

        seq_results.append({
            "label":  rec["label"],
            "chrom":  rec["chrom"],
            "center": rec["center"],
            "shifts": shifts,
        })

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(sequences)}", flush=True)

    # ── Aggregate and print ───────────────────────────────────────────────────
    print("\n  === Shuffle control results ===")
    print(f"  {'Shuffle type':15s}  {'n':>4}  {'mean shift':>12}  {'std':>8}  {'t':>8}  {'p (two-sided)':>14}  sig")

    agg = {}
    for stype in args.shuffle_types:
        all_s = [r["shifts"][stype] for r in seq_results if stype in r["shifts"]]
        t, p  = _t_test_shifts(all_s)
        mean  = float(np.mean(all_s))
        std   = float(np.std(all_s))
        sig   = ("***" if p < 0.001 else "**" if p < 0.01
                 else "*" if p < 0.05 else "n.s.")
        print(f"  {stype:15s}  {len(all_s):4d}  {mean:+12.6f}  {std:8.6f}"
              f"  {t:8.3f}  {p:14.6f}  {sig}")
        agg[stype] = {
            "n": len(all_s), "mean": round(mean, 6), "std": round(std, 6),
            "t": round(t, 4), "p_two_sided": round(p, 6),
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":         args.model,
        "sw_layer":      sw_layer,
        "sw_rows":       sw_rows,
        "n_seqs":        len(sequences),
        "n_shuffles":    args.n_shuffles,
        "window_bp":     args.window_bp,
        "shuffle_types": args.shuffle_types,
        "aggregate":     agg,
        "per_sequence":  seq_results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved → {args.out}")

    _plot(seq_results, args.plot, sw_info, args.n_shuffles)


if __name__ == "__main__":
    main()
