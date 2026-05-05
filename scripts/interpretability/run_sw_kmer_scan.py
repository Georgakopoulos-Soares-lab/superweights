"""
scripts/run_sw_kmer_scan.py
-----------------------------
Enumerate all 4^6 = 4 096 possible DNA 6-mers and rank them by the activation
they induce in the super-weight rows of GENERator.

Each 6-mer maps to a single token in GENERator's vocabulary.  To measure the
activation cleanly, each target token is embedded in a fixed poly-A context:

    [A×5]  [target k-mer]  [A×5]   →  11 tokens  (66 bp)

We capture the SW row activation at the target position (index 5).  All 4096
k-mers are batched together for a single GPU call.

Outputs
-------
  JSON : results/sw_kmer_scan.json   — (kmer, token_id, activation) sorted desc
  PNG  : results/sw_kmer_scan.png    — top-50 bar chart + nucleotide-composition
                                       heatmaps + GC% scatter

Usage
-----
    python scripts/run_sw_kmer_scan.py \
        --out  results/sw_kmer_scan.json \
        --plot results/sw_kmer_scan.png
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

POLY_A_TOKEN = 32   # AAAAAA token id — confirmed for GENERator vocab
TARGET_POS   = 5    # position of target k-mer in 11-token context
CTX_HALF     = 5    # poly-A tokens on each side


# ─────────────────────────────────────────────────────────────────────────────
# Build input batch
# ─────────────────────────────────────────────────────────────────────────────

def _build_batch(kmer_ids: list[int]) -> torch.Tensor:
    """Return [N, 11] input_ids tensor: poly-A context with target at pos 5."""
    N      = len(kmer_ids)
    prefix = [POLY_A_TOKEN] * CTX_HALF
    suffix = [POLY_A_TOKEN] * CTX_HALF
    rows   = [prefix + [kid] + suffix for kid in kmer_ids]
    return torch.tensor(rows, dtype=torch.long)   # [N, 11]


# ─────────────────────────────────────────────────────────────────────────────
# Run scan in mini-batches
# ─────────────────────────────────────────────────────────────────────────────

def _scan(model, kmer_ids: list[int], sw_layer: int, sw_rows: list[int],
          device: str, batch_size: int = 256) -> np.ndarray:
    """Return activation array [N, n_sw_rows] at TARGET_POS for every kmer."""
    target_module = model.model.layers[sw_layer].mlp.down_proj
    N             = len(kmer_ids)
    all_acts      = []

    for start in range(0, N, batch_size):
        chunk   = kmer_ids[start : start + batch_size]
        ids     = _build_batch(chunk).to(device)

        _store  = {}
        def _hook(_mod, _inp, _out, _s=_store):
            _s["act"] = _out.detach().cpu().float()  # [B, 11, hidden]
        h = target_module.register_forward_hook(_hook)
        with torch.no_grad():
            model(input_ids=ids)
        h.remove()

        # activation at target position for each SW row
        acts = _store["act"][:, TARGET_POS, :][:, sw_rows]   # [B, n_sw_rows]
        all_acts.append(acts.numpy())
        print(f"    batch {start//batch_size + 1}/{(N-1)//batch_size + 1}"
              f"  ({start}–{min(start+batch_size, N)-1})", flush=True)

    return np.concatenate(all_acts, axis=0)   # [N, n_sw_rows]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot(kmers: list[str], activations: np.ndarray, out_path: str,
          sw_rows: list[int], top_n: int = 50):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    # Use mean activation across SW rows as ranking score
    score = activations.mean(axis=1)         # [N]
    rank  = np.argsort(score)[::-1]          # descending

    top_kmers = [kmers[i]  for i in rank[:top_n]]
    top_scores = score[rank[:top_n]]

    gc_all  = np.array([( k.count("G") + k.count("C") ) / 6 for k in kmers])
    gc_top  = np.array([( k.count("G") + k.count("C") ) / 6 for k in top_kmers])

    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_bar  = fig.add_subplot(gs[0, :2])
    ax_gc   = fig.add_subplot(gs[0, 2])
    ax_pos  = fig.add_subplot(gs[1, :])

    # ── Top-N bar chart ───────────────────────────────────────────────────────
    colours = ["#d73027" if gc >= 0.5 else "#4575b4" for gc in gc_top]
    ax_bar.barh(range(top_n), top_scores[::-1], color=colours[::-1])
    ax_bar.set_yticks(range(top_n))
    ax_bar.set_yticklabels(top_kmers[::-1], fontsize=6)
    ax_bar.set_xlabel("Mean SW row activation at k-mer position")
    ax_bar.set_title(f"Top {top_n} 6-mers by SW activation  (SW rows={sw_rows})")
    from matplotlib.patches import Patch
    ax_bar.legend(handles=[Patch(color="#d73027", label="GC ≥ 50%"),
                            Patch(color="#4575b4", label="GC < 50%")],
                  fontsize=8, loc="lower right")

    # ── GC% vs activation scatter ─────────────────────────────────────────────
    ax_gc.scatter(gc_all, score, s=3, alpha=0.3, color="steelblue")
    # highlight top-50
    ax_gc.scatter(gc_top, top_scores, s=10, alpha=0.8, color="#d73027", zorder=3)
    ax_gc.set_xlabel("GC fraction")
    ax_gc.set_ylabel("Mean SW activation")
    ax_gc.set_title("GC% vs SW activation\n(red = top 50)")
    ax_gc.grid(True, alpha=0.3)
    # correlation
    r = float(np.corrcoef(gc_all, score)[0, 1])
    ax_gc.text(0.05, 0.92, f"r = {r:.3f}", transform=ax_gc.transAxes, fontsize=9)

    # ── Per-position nucleotide enrichment in top-N ────────────────────────────
    nt_order = list("ACGT")
    pos_enrich = np.zeros((4, 6))
    background = {n: 0.25 for n in nt_order}
    for km in top_kmers:
        for pos, nt in enumerate(km):
            idx = nt_order.index(nt)
            pos_enrich[idx, pos] += 1
    pos_enrich /= top_n   # fraction
    im = ax_pos.imshow(pos_enrich, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=0.5)
    ax_pos.set_yticks(range(4))
    ax_pos.set_yticklabels(nt_order, fontsize=10)
    ax_pos.set_xticks(range(6))
    ax_pos.set_xticklabels([f"pos {i+1}" for i in range(6)])
    ax_pos.set_title(f"Nucleotide frequency in top-{top_n} k-mers (green=enriched, red=depleted)")
    plt.colorbar(im, ax=ax_pos, label="fraction in top k-mers")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="generator",
                   choices=["generator", "generator_prokaryote",
                            "generator_prokaryote_1b"])
    p.add_argument("--sw_index",    default="results/super_weight_index.json")
    p.add_argument("--configs_dir", default="configs")
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--top_n",       type=int, default=50)
    p.add_argument("--out",         default="results/sw_kmer_scan.json")
    p.add_argument("--plot",        default="results/sw_kmer_scan.png")
    return p.parse_args()


def main():
    args = parse_args()

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

    # ── Load model + tokenizer ────────────────────────────────────────────────
    print(f"  Loading {args.model} ...", flush=True)
    from models.generator_wrapper import GeneratorWrapper
    wrapper   = GeneratorWrapper(cfg)
    wrapper.load()
    model     = wrapper.model
    tokenizer = wrapper.tokenizer
    model.eval()

    # ── Build 4096 k-mer vocabulary ───────────────────────────────────────────
    print("  Building 6-mer vocabulary ...", flush=True)
    bases  = "ACGT"
    kmers  = ["".join(c) for c in product(bases, repeat=6)]  # 4096 k-mers
    kmer_ids = []
    valid_kmers = []
    for km in kmers:
        ids = tokenizer.encode(km, add_special_tokens=False)
        if len(ids) == 1:
            kmer_ids.append(ids[0])
            valid_kmers.append(km)
        # skip k-mers that don't tokenize to a single token (e.g. contain N)

    print(f"  {len(valid_kmers)} / 4096 k-mers tokenize to a single token")

    # ── Run scan ──────────────────────────────────────────────────────────────
    print(f"  Scanning SW activation for {len(valid_kmers)} k-mers ...", flush=True)
    activations = _scan(model, kmer_ids, sw_layer, sw_rows, args.device, args.batch_size)
    # activations: [N, n_sw_rows]

    score = activations.mean(axis=1)
    rank  = np.argsort(score)[::-1]

    print(f"\n  === Top 20 k-mers by SW activation ===")
    for i in rank[:20]:
        gc = (valid_kmers[i].count("G") + valid_kmers[i].count("C")) / 6
        print(f"  {valid_kmers[i]}  score={score[i]:+.1f}  GC={gc:.0%}")

    print(f"\n  === Bottom 10 k-mers ===")
    for i in rank[-10:]:
        gc = (valid_kmers[i].count("G") + valid_kmers[i].count("C")) / 6
        print(f"  {valid_kmers[i]}  score={score[i]:+.1f}  GC={gc:.0%}")

    # GC correlation
    gc_frac = np.array([(km.count("G") + km.count("C")) / 6 for km in valid_kmers])
    r = float(np.corrcoef(gc_frac, score)[0, 1])
    print(f"\n  GC% vs SW activation: r = {r:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":    args.model,
        "sw_layer": sw_layer,
        "sw_rows":  sw_rows,
        "gc_activation_corr": r,
        "results": [
            {
                "kmer":      valid_kmers[i],
                "token_id":  int(kmer_ids[i]),
                "activation_mean": float(score[i]),
                "activations_per_row": activations[i].tolist(),
                "gc_frac":   float(gc_frac[i]),
                "rank":      int(np.where(rank == i)[0][0]) + 1,
            }
            for i in range(len(valid_kmers))
        ],
    }
    # sort by rank
    payload["results"].sort(key=lambda x: x["rank"])

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved → {args.out}")

    _plot(valid_kmers, activations, args.plot, sw_rows, top_n=args.top_n)


if __name__ == "__main__":
    main()
