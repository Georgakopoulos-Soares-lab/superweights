"""
scripts/run_sw_causal_tracing.py
----------------------------------
Causal importance of super-weight rows in GENERator via two complementary
mechanistic experiments.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Experiment A: SW Row Ablation (primary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each clean genomic sequence:

  1. Measure clean perplexity.
  2. Zero the SW row(s) in the MLP down_proj output at sw_layer → ablated PPL.
  3. Zero a matched-count random row set (10 seeds) → rand_ablated PPL.

  delta_sw   = ablated_ppl   - clean_ppl   (cost of losing SW rows)
  delta_rand = rand_ablated_ppl - clean_ppl (cost of losing random rows)

  If delta_sw >> delta_rand → SW rows are causally necessary for prediction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Experiment B: Activation Distribution Shift (secondary, via --also_patch)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare the SW row activation distributions between:
  - Promoter sequences (should have higher / more structured activations)
  - Shuffled versions of the same sequences

  shift = (mean_real - mean_shuffled) / (std_real + std_shuffled)

  Large positive shift → SW row encodes real promoter/enhancer context.

Together these answer:
  "Are the super-weight rows in layer 4 of GENERator causally necessary for
   sequence prediction, and do they differentiate real vs. shuffled genomic
   context?"

Usage:
    python scripts/run_sw_causal_tracing.py \\
        --fasta    /home/nvidia/data/hg38/hg38.fa \\
        --promoters ../data/regions/hg38/promoters_262kb.bed \\
        --enhancers ../data/regions/hg38/enhancers_ccre_262kb.bed \\
        --random    ../data/regions/hg38/random_262kb.bed \\
        --n_seqs 30 --window_bp 3072 \\
        --out  results/sw_causal_tracing.json \\
        --plot results/sw_causal_tracing.png
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
# FASTA + BED helpers (shared with run_sw_gradient_attribution.py)
# ─────────────────────────────────────────────────────────────────────────────

def _open_fasta(path: str):
    try:
        from pyfaidx import Fasta
        return Fasta(path, as_raw=True, sequence_always_upper=True)
    except ImportError:
        raise ImportError("pyfaidx required: conda run -n evo pip install pyfaidx")


def _fetch_window(fasta, chrom: str, center: int, window_bp: int) -> str | None:
    half = window_bp // 2
    start = max(0, center - half)
    end   = start + window_bp
    key   = (chrom if chrom in fasta
             else "chr" + chrom if "chr" + chrom in fasta
             else chrom.lstrip("chr") if chrom.lstrip("chr") in fasta
             else None)
    if key is None:
        return None
    chrom_len = len(fasta[key])
    if end > chrom_len:
        end   = chrom_len
        start = max(0, end - window_bp)
    seq = str(fasta[key][start:end])
    if len(seq) < window_bp:
        seq = seq + "N" * (window_bp - len(seq))
    return seq


def _sample_bed_windows(bed_path: str, fasta, n: int, window_bp: int,
                        rng: random.Random, label: str) -> list:
    with open(bed_path) as fh:
        lines = [l.strip().split("\t") for l in fh
                 if l.strip() and not l.startswith("#")]
    candidate_pool = rng.sample(lines, min(len(lines), 10 * n))
    results = []
    for row in candidate_pool:
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
# Dinucleotide shuffle (Altschul–Erickson, simplified)
# ─────────────────────────────────────────────────────────────────────────────

def _dinuc_shuffle(seq: str, rng: random.Random) -> str:
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


# ─────────────────────────────────────────────────────────────────────────────
# Tokenize
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(tokenizer, seq: str, device: str) -> torch.Tensor:
    # GENERator k-mer tokenizer requires sequence length divisible by k (6)
    remainder = len(seq) % 6
    if remainder:
        seq = seq[remainder:]
    return tokenizer(seq, return_tensors="pt",
                     add_special_tokens=False)["input_ids"].to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity (cross-entropy loss)
# ─────────────────────────────────────────────────────────────────────────────

def _ppl(model, input_ids: torch.Tensor) -> float:
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
        return out.loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A: ablation
# ─────────────────────────────────────────────────────────────────────────────

def _run_ablation(model, input_ids: torch.Tensor,
                  sw_layer: int, rows_to_zero: list) -> float:
    """Return CE loss after zeroing `rows_to_zero` in layer sw_layer's MLP."""
    target = model.model.layers[sw_layer].mlp.down_proj
    def _hook(_mod, _inp, _out):
        patched = _out.clone()
        patched[..., rows_to_zero] = 0.0
        return patched
    h = target.register_forward_hook(_hook)
    try:
        loss = _ppl(model, input_ids)
    finally:
        h.remove()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: activation distribution shift
# ─────────────────────────────────────────────────────────────────────────────

def _capture_sw_activation(model, input_ids: torch.Tensor,
                            sw_layer: int, sw_rows: list) -> np.ndarray:
    """Return SW row activations as flat numpy array [n_tokens * n_sw_rows]."""
    _store = {}
    def _hook(_mod, _inp, _out):
        _store["act"] = _out[..., sw_rows].detach().cpu().float().numpy()
    h = model.model.layers[sw_layer].mlp.down_proj.register_forward_hook(_hook)
    with torch.no_grad():
        model(input_ids=input_ids)
    h.remove()
    return _store["act"].ravel()  # flatten [1, L, n_rows] → 1-D


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(results: list, out_path: str, sw_info: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    from collections import defaultdict
    sw_by_ctx   = defaultdict(list)
    rand_by_ctx = defaultdict(list)
    shift_by_ctx = defaultdict(list)
    for r in results:
        ctx = r["label"]
        sw_by_ctx[ctx].append(r["delta_sw"])
        rand_by_ctx[ctx].append(r["delta_rand_mean"])
        shift_by_ctx[ctx].append(r.get("act_shift", 0.0))

    contexts = sorted(sw_by_ctx.keys())
    x = np.arange(len(contexts))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: ablation cost (delta PPL)
    sw_means   = [np.mean(sw_by_ctx[c])   for c in contexts]
    rand_means = [np.mean(rand_by_ctx[c]) for c in contexts]
    sw_sems    = [np.std(sw_by_ctx[c])    for c in contexts]
    rand_sems  = [np.std(rand_by_ctx[c])  for c in contexts]

    bars1 = ax1.bar(x - w/2, sw_means,   w, label="SW rows",      color="steelblue",
                    yerr=sw_sems,   capsize=4)
    bars2 = ax1.bar(x + w/2, rand_means, w, label="Random rows",   color="lightgray",
                    yerr=rand_sems, capsize=4)
    ax1.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(contexts)
    ax1.set_ylabel("ΔPPL  (ablated − clean)  ↑ = more important")
    ax1.set_title(f"SW row ablation cost by genomic context\n{sw_info}")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: activation distribution shift (real vs shuffled)
    all_shifts  = [r.get("act_shift", 0.0) for r in results]
    all_labels  = [r["label"]              for r in results]
    ctx_colours = {"promoter": "steelblue", "enhancer": "darkorange", "random": "gray"}
    cols = [ctx_colours.get(lbl, "purple") for lbl in all_labels]
    ax2.scatter(range(len(all_shifts)), all_shifts, c=cols, s=25, alpha=0.7)
    ax2.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Sequence index")
    ax2.set_ylabel("Activation shift  (real − shuffled) / pooled-std")
    ax2.set_title("SW row activation: real vs dinucleotide-shuffled")
    for ctx, col in ctx_colours.items():
        ax2.scatter([], [], c=col, label=ctx)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

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
    p.add_argument("--n_seqs",      type=int, default=30)
    p.add_argument("--window_bp",   type=int, default=3072)
    p.add_argument("--model",       default="generator",
                   choices=["generator", "generator_prokaryote",
                            "generator_prokaryote_1b"])
    p.add_argument("--sw_index",    default="results/super_weight_index.json")
    p.add_argument("--configs_dir", default="configs")
    p.add_argument("--n_rand_seeds", type=int, default=10)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default="results/sw_causal_tracing.json")
    p.add_argument("--plot",        default="results/sw_causal_tracing.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)
    args.window_bp = (args.window_bp // 6) * 6   # divisible by k-mer size

    # ── Load config + SW index ────────────────────────────────────────────────
    cfg      = yaml.safe_load(open(Path(args.configs_dir) / f"{args.model}.yaml"))
    sw_data  = json.load(open(args.sw_index))
    sw_entry = sw_data.get(args.model, {})
    sw_list  = sw_entry.get("results", []) if isinstance(sw_entry, dict) else sw_entry
    if not sw_list:
        print(f"No SW found for {args.model}. Run run_detection.py first.")
        sys.exit(1)

    top_sw   = max(sw_list, key=lambda x: x["out_max"])
    sw_layer = int(top_sw["layer"])
    sw_rows  = sorted({int(s["row"]) for s in sw_list if s["layer"] == sw_layer})
    print(f"  SW: layer={sw_layer}  rows={sw_rows}  ({len(sw_rows)} rows)")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"  Loading {args.model} ...", flush=True)
    from models.generator_wrapper import GeneratorWrapper
    wrapper   = GeneratorWrapper(cfg)
    wrapper.load()
    model     = wrapper.model.to(args.device)
    tokenizer = wrapper.tokenizer
    model.eval()

    n_total_rows = model.model.layers[sw_layer].mlp.down_proj.weight.shape[0]
    non_sw_rows  = [r for r in range(n_total_rows) if r not in set(sw_rows)]
    print(f"  Total MLP rows at layer {sw_layer}: {n_total_rows}")

    # ── Genomic sequences ─────────────────────────────────────────────────────
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
        print(f"  {label:10s}: {len(seqs)} sequences")
        sequences.extend(seqs)

    if not sequences:
        print("No sequences loaded. Check BED paths.")
        sys.exit(1)

    # ── Run experiments ───────────────────────────────────────────────────────
    print(f"\n  Running ablation + activation-shift on {len(sequences)} sequences ...")
    results = []

    for i, rec in enumerate(sequences):
        seq   = rec["seq"]
        ids   = _tokenize(tokenizer, seq, args.device)
        if ids.shape[1] < 2:
            continue

        # ── A: ablation ───────────────────────────────────────────────────────
        clean_ppl  = _ppl(model, ids)
        sw_abl_ppl = _run_ablation(model, ids, sw_layer, sw_rows)
        delta_sw   = sw_abl_ppl - clean_ppl

        # Random ablation (same count as SW rows, multiple seeds)
        rand_deltas = []
        for rseed in range(args.n_rand_seeds):
            rand_rows = random.Random(rseed).sample(non_sw_rows, len(sw_rows))
            rand_ppl  = _run_ablation(model, ids, sw_layer, rand_rows)
            rand_deltas.append(rand_ppl - clean_ppl)
        delta_rand_mean = float(np.mean(rand_deltas))
        delta_rand_std  = float(np.std(rand_deltas))

        # ── B: activation distribution shift (real vs shuffled) ──────────────
        shuffled_seq = _dinuc_shuffle(seq, rng)
        shuf_ids     = _tokenize(tokenizer, shuffled_seq, args.device)
        L = min(ids.shape[1], shuf_ids.shape[1])
        act_real = _capture_sw_activation(model, ids[:, :L],   sw_layer, sw_rows)
        act_shuf = _capture_sw_activation(model, shuf_ids[:, :L], sw_layer, sw_rows)
        act_shift = float(
            (act_real.mean() - act_shuf.mean()) /
            (act_real.std() + act_shuf.std() + 1e-8)
        )

        print(f"  [{i+1}/{len(sequences)}]"
              f" {rec['label']:10s} {rec['chrom']}:{rec['center']}"
              f"  clean={clean_ppl:.3f}"
              f"  Δsw={delta_sw:+.3f}"
              f"  Δrand={delta_rand_mean:+.3f}±{delta_rand_std:.3f}"
              f"  act_shift={act_shift:+.3f}")

        results.append({
            "label":          rec["label"],
            "chrom":          rec["chrom"],
            "center":         rec["center"],
            "n_tokens":       L,
            "clean_ppl":      clean_ppl,
            "sw_ablated_ppl": sw_abl_ppl,
            "delta_sw":       delta_sw,
            "delta_rand_mean": delta_rand_mean,
            "delta_rand_std":  delta_rand_std,
            "act_shift":      act_shift,
            "sw_rows":        sw_rows,
        })

    # ── Statistical summary ───────────────────────────────────────────────────
    if results:
        ds = [r["delta_sw"]        for r in results]
        dr = [r["delta_rand_mean"] for r in results]
        sh = [r["act_shift"]       for r in results]
        print(f"\n  === A: Ablation cost ===")
        print(f"  SW row ΔPPL:     {np.mean(ds):+.4f} ± {np.std(ds):.4f}")
        print(f"  Random row ΔPPL: {np.mean(dr):+.4f} ± {np.std(dr):.4f}")
        from scipy import stats as _stats
        t_abl, p_abl = _stats.ttest_rel(ds, dr)
        print(f"  Paired t (SW vs rand): t={t_abl:.3f}  p={p_abl:.4f}")
        print(f"\n  === B: Activation shift (real vs shuffled) ===")
        print(f"  Mean shift: {np.mean(sh):+.4f} ± {np.std(sh):.4f}")
        t_sh, p_sh = _stats.ttest_1samp(sh, 0.0)
        print(f"  t-test vs 0: t={t_sh:.3f}  p={p_sh:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":     args.model,
        "sw_layer":  sw_layer,
        "sw_rows":   sw_rows,
        "window_bp": args.window_bp,
        "n_seqs":    len(results),
        "results":   results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved → {args.out}")

    sw_info = f"layer={sw_layer}  rows={sw_rows}"
    _plot(results, args.plot, sw_info)


if __name__ == "__main__":
    main()
