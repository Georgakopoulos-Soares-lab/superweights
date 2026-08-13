# sae/analyze.py
"""
SAE Feature Analysis: characterise learned features against genomic hexamers.

Steps
-----
1. Load the trained SAE + the frozen GENERator model.
2. Collect SAE feature activations for all 4,096 hexamers using the same
   11-token poly-A context used in the k-mer scan (5 flanking, target at pos 5).
3. Per feature: compute mean activation per hexamer → [n_features, 4096].
4. Correlate each feature with:
     - GC content (per hexamer)
     - Homopolymer flag (any base repeated ≥ 4 times)
     - Shannon entropy of the hexamer's base composition
5. Identify the top-20 monosemantic features: those with the lowest entropy
   over their hexamer activation distribution (high activation concentrated on
   a narrow, coherent set of hexamers).
6. Cross-reference those top-20 features against the top-500 SW-dependent
   hexamers (ranked by KL divergence from sw_hexamer_causal.json).
7. Produce a heatmap: top-20 monosemantic features (rows) × hexamer motif
   class (columns): homopolymer, AT-rich, GC-rich, splice-donor,
   Shine-Dalgarno, other.

Usage
-----
  python sae/analyze.py \\
      --model       generator                                \\
      --layer       4                                        \\
      --sae_ckpt    results/sae/generator_euk_layer4/sae_final.pt  \\
      --hexamer_src results/sw_hexamer_causal.json           \\
      --out_dir     results/sae/generator_euk_layer4         \\
      --top_mono    20                                       \\
      --top_sw_dep  500

For PROK:
  --model generator_prokaryote --layer 2 \\
  --hexamer_src results/sw_hexamer_causal_generator_prok.json
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from models import WRAPPER_MAP
from sae.model import BatchTopKSAE

# Matches run_sw_kmer_scan.py — confirmed for GENERator vocab
POLY_A_TOKEN = 32
TARGET_POS   = 5
CTX_HALF     = 5


# ──────────────────────────────────────────────────────────────────────────────
# Hexamer utilities
# ──────────────────────────────────────────────────────────────────────────────

def _all_hexamers():
    """Return list of all 4096 DNA hexamer strings."""
    return ["".join(b) for b in product("ACGT", repeat=6)]


def _gc_frac(kmer: str) -> float:
    return (kmer.count("G") + kmer.count("C")) / len(kmer)


def _homopolymer_flag(kmer: str, min_run: int = 4) -> bool:
    """True if any single base runs for ≥ min_run consecutive positions."""
    for base in "ACGT":
        if base * min_run in kmer:
            return True
    return False


def _base_entropy(kmer: str) -> float:
    """Shannon entropy of the base composition (max = log2(4) ≈ 2)."""
    counts = np.array([kmer.count(b) for b in "ACGT"], dtype=float)
    counts /= counts.sum()
    counts = counts[counts > 0]
    return float(-np.sum(counts * np.log2(counts)))


def _motif_class(kmer: str) -> str:
    """
    Assign a motif class label to a hexamer.
    Priority order: homopolymer > GC-rich > AT-rich > splice-donor >
                    Shine-Dalgarno > other.
    """
    if _homopolymer_flag(kmer):
        return "homopolymer"
    gc = _gc_frac(kmer)
    if gc >= 0.75:
        return "GC-rich"
    if gc <= 0.25:
        return "AT-rich"
    # Splice donor: starts with GT (GT→AG rule; hexamers are coding-strand DNA)
    if kmer.startswith("GT"):
        return "splice_donor"
    # Shine-Dalgarno: contains core AAGG / GAGG / AGGA (partial SD)
    if any(sd in kmer for sd in ("AAGG", "GAGG", "AGGA", "AGGAG"[:len(kmer)])):
        return "shine_dalgarno"
    return "other"


MOTIF_CLASSES = ["homopolymer", "AT-rich", "GC-rich", "splice_donor",
                 "shine_dalgarno", "other"]


# ──────────────────────────────────────────────────────────────────────────────
# Hexamer → token ID mapping
# ──────────────────────────────────────────────────────────────────────────────

def _build_hexamer_token_ids(tokenizer, hexamers: list[str]) -> list[int]:
    """Return token IDs for each hexamer (single-token encoding expected)."""
    token_ids = []
    for kmer in hexamers:
        enc = tokenizer(kmer, add_special_tokens=False)["input_ids"]
        if len(enc) == 1:
            token_ids.append(enc[0])
        else:
            # Hexamer tokenises to more than one token — use first token as proxy
            token_ids.append(enc[0] if enc else 0)
    return token_ids


# ──────────────────────────────────────────────────────────────────────────────
# Collect SAE feature activations on hexamers
# ──────────────────────────────────────────────────────────────────────────────

def collect_hexamer_sae_acts(
    wrapper,
    sae: BatchTopKSAE,
    layer_idx: int,
    hexamers: list[str],
    token_ids: list[int],
    batch_size: int = 256,
) -> np.ndarray:
    """
    For each hexamer, run the 11-token context forward pass and encode
    the residual-stream activation at TARGET_POS through the SAE.

    Returns
    -------
    acts_matrix : float32 np.ndarray  [n_hexamers, n_features]
    """
    model        = wrapper.model
    device       = next(model.parameters()).device
    layer_module = model.model.layers[layer_idx]
    sae_device   = next(sae.parameters()).device

    n_hexamers   = len(hexamers)
    n_features   = sae.n_features
    acts_matrix  = np.zeros((n_hexamers, n_features), dtype=np.float32)

    for b_start in range(0, n_hexamers, batch_size):
        b_end    = min(b_start + batch_size, n_hexamers)
        b_ids    = token_ids[b_start : b_end]
        B        = len(b_ids)

        # Build 11-token input: [PA, PA, PA, PA, PA, target, PA, PA, PA, PA, PA]
        prefix = [POLY_A_TOKEN] * CTX_HALF
        suffix = [POLY_A_TOKEN] * CTX_HALF
        rows   = [prefix + [kid] + suffix for kid in b_ids]
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)   # [B, 11]

        store = {}

        def _hook(_mod, _inp, output, _s=store):
            h = output[0] if isinstance(output, tuple) else output
            _s["h"] = h.detach()   # [B, 11, D]

        handle = layer_module.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            handle.remove()

        h_target = store["h"][:, TARGET_POS, :]                # [B, D]
        h_target = h_target.to(sae_device).float()

        with torch.no_grad():
            # Match training-time preprocessing. Without this the SAE receives
            # inputs whose SW channel is ~6900x the median sd and the dictionary
            # degenerates to single-hexamer detectors.
            _sc = getattr(sae, "data_scale", None)
            if _sc is not None:
                h_target = h_target / _sc
            enc = sae.encode(h_target)
            acts = enc["acts"].cpu().numpy()                    # [B, n_features]

        acts_matrix[b_start:b_end] = acts

        print(
            f"  hexamer batch {b_start//batch_size + 1}/"
            f"{(n_hexamers - 1)//batch_size + 1}  "
            f"({b_start}–{b_end - 1})",
            flush=True,
        )

    return acts_matrix   # [4096, n_features]


# ──────────────────────────────────────────────────────────────────────────────
# Feature characterisation
# ──────────────────────────────────────────────────────────────────────────────

def activation_entropy(feat_acts: np.ndarray, eps: float = 1e-10) -> float:
    """
    Shannon entropy of the hexamer activation distribution for one feature.
    feat_acts: [n_hexamers]  (non-negative)
    Normalised to a probability distribution before computing entropy.
    Lower entropy = more monosemantic (fires on narrow set of hexamers).
    Dead features (max activation == 0) return inf so they sort last.
    """
    p = feat_acts - feat_acts.min()
    total = p.sum()
    if total < eps:
        return float("inf")   # dead feature — exclude from monosemantic ranking
    p = p / total
    p = p[p > eps]
    return float(-np.sum(p * np.log2(p)))


def correlate_features(
    acts_matrix: np.ndarray,   # [n_hexamers, n_features]
    gc_arr:   np.ndarray,      # [n_hexamers]
    homo_arr: np.ndarray,      # [n_hexamers]  bool → float
    ent_arr:  np.ndarray,      # [n_hexamers]
) -> dict:
    """
    Pearson correlation of each feature's mean activation per hexamer with
    gc_arr, homo_arr, ent_arr.  Returns dict of arrays [n_features].
    """
    results = {}
    for name, arr in [("gc", gc_arr), ("homopolymer", homo_arr), ("entropy", ent_arr)]:
        corrs = np.zeros(acts_matrix.shape[1])
        for f in range(acts_matrix.shape[1]):
            col = acts_matrix[:, f]
            if col.std() < 1e-8:
                corrs[f] = 0.0
            else:
                corrs[f] = float(np.corrcoef(col, arr)[0, 1])
        results[name] = corrs
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse SAE features against hexamer motifs."
    )
    parser.add_argument("--model",       required=True)
    parser.add_argument("--layer",       type=int, required=True)
    parser.add_argument("--sae_ckpt",    required=True,
                        help="Path to trained SAE checkpoint (sae_final.pt)")
    parser.add_argument("--hexamer_src", default="results/sw_hexamer_causal.json",
                        help="JSON with hexamer KL divergence data")
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--top_mono",    type=int, default=20,
                        help="Number of top monosemantic features to report (default: 20)")
    parser.add_argument("--top_sw_dep",  type=int, default=500,
                        help="Number of SW-dependent hexamers to cross-reference (default: 500)")
    parser.add_argument("--batch_size",  type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load models ────────────────────────────────────────────────────────────
    print("[analyze] Loading GENERator...", flush=True)
    config  = yaml.safe_load(open(_REPO / f"configs/{args.model}.yaml"))
    wrapper = WRAPPER_MAP[args.model](config)
    wrapper.load()
    wrapper.model.eval()

    print("[analyze] Loading SAE...", flush=True)
    sae = BatchTopKSAE.load(args.sae_ckpt, device=str(device))
    if getattr(sae, "data_scale", None) is not None:
        print(f"[analyze] checkpoint is STANDARDIZED — applying saved per-channel scale "
              f"(max/median = {(sae.data_scale.max()/sae.data_scale.median()).item():.0f}x)")
    else:
        print("[analyze] checkpoint has no data_scale (raw-activation training)")
    sae.eval()

    # ── Hexamers ───────────────────────────────────────────────────────────────
    hexamers  = _all_hexamers()           # 4096 strings
    token_ids = _build_hexamer_token_ids(wrapper.tokenizer, hexamers)

    gc_arr   = np.array([_gc_frac(k)         for k in hexamers])
    homo_arr = np.array([_homopolymer_flag(k) for k in hexamers], dtype=float)
    ent_arr  = np.array([_base_entropy(k)     for k in hexamers])
    class_arr = [_motif_class(k) for k in hexamers]

    # ── Collect SAE activations on hexamers ────────────────────────────────────
    print(f"\n[analyze] Collecting SAE activations for {len(hexamers)} hexamers...", flush=True)
    acts_matrix = collect_hexamer_sae_acts(
        wrapper, sae, args.layer, hexamers, token_ids, args.batch_size
    )
    # acts_matrix: [4096, n_features]

    # ── Correlations ──────────────────────────────────────────────────────────
    print("\n[analyze] Computing feature correlations...", flush=True)
    corrs = correlate_features(acts_matrix, gc_arr, homo_arr, ent_arr)

    # ── Monosemantic features ─────────────────────────────────────────────────
    print("[analyze] Ranking features by monosemanticity...", flush=True)
    feat_entropy = np.array([activation_entropy(acts_matrix[:, f])
                              for f in range(acts_matrix.shape[1])])
    # Sort ascending: lowest entropy = most monosemantic.
    # Dead features return inf (excluded from top).
    mono_rank    = np.argsort(feat_entropy)
    top_mono_idx = mono_rank[:args.top_mono].tolist()

    # ── SW-dependent hexamers cross-reference ─────────────────────────────────
    print(f"\n[analyze] Loading SW-dependent hexamers from {args.hexamer_src}...", flush=True)
    hexamer_data = json.loads(Path(args.hexamer_src).read_text())
    results_list = hexamer_data.get("results", [])

    # Build a KL-score vector aligned to the hexamers list [4096].
    # kl_arr[i] = kl_clean_ablated for hexamer i; 0 if not present.
    kl_lookup = {r["kmer"]: r.get("kl_clean_ablated", 0.0) for r in results_list}
    kl_arr    = np.array([kl_lookup.get(k, 0.0) for k in hexamers], dtype=np.float64)

    # Sort by KL descending → top SW-dependent hexamers set
    sw_dep_sorted = sorted(results_list, key=lambda r: r.get("kl_clean_ablated", 0), reverse=True)
    sw_dep_top    = {r["kmer"] for r in sw_dep_sorted[:args.top_sw_dep]}

    # For each top-mono feature compute:
    #   (a) Pearson r with SW KL scores — the primary cross-reference
    #   (b) Top hexamers where activation > 0 only (avoid tie-breaking noise)
    overlap_results = []
    for fi in top_mono_idx:
        col     = acts_matrix[:, fi].astype(np.float64)
        nnz_idx = np.where(col > 0)[0]

        # Top active hexamers (activation > 0, sorted descending)
        top_active = [hexamers[i] for i in nnz_idx[np.argsort(col[nnz_idx])[::-1]][:10]]

        # Pearson r between feature activation and SW KL scores
        if col.std() > 1e-8 and kl_arr.std() > 1e-8:
            sw_pearson_r = float(np.corrcoef(col, kl_arr)[0, 1])
        else:
            sw_pearson_r = 0.0

        # Fraction of the feature's active hexamers that are SW-dependent
        active_in_sw = [hexamers[i] for i in nnz_idx if hexamers[i] in sw_dep_top]
        sw_active_frac = len(active_in_sw) / max(len(nnz_idx), 1)

        overlap_results.append({
            "feature_idx":        fi,
            "activation_entropy": float(feat_entropy[fi]),
            "n_active_hexamers":  int(len(nnz_idx)),
            "gc_corr":            float(corrs["gc"][fi]),
            "homo_corr":          float(corrs["homopolymer"][fi]),
            "entropy_corr":       float(corrs["entropy"][fi]),
            "sw_pearson_r":       sw_pearson_r,
            "sw_active_frac":     sw_active_frac,
            "top_active_hexamers": top_active,
            "active_in_sw_dep":   active_in_sw[:20],
        })

    print(f"\n  Top-{args.top_mono} monosemantic features:")
    for rank, r in enumerate(overlap_results):
        fi = r["feature_idx"]
        print(f"    [{rank+1:2d}] feature {fi:5d}  entropy={feat_entropy[fi]:.4f}"
              f"  n_active={r['n_active_hexamers']}"
              f"  GC-corr={r['gc_corr']:+.3f}"
              f"  SW-r={r['sw_pearson_r']:+.4f}"
              f"  top_active={r['top_active_hexamers'][:5]}")

    # Also rank by SW Pearson r to find the features most correlated with SW sensitivity
    sw_corr_per_feat = np.zeros(acts_matrix.shape[1])
    if kl_arr.std() > 1e-8:
        for f in range(acts_matrix.shape[1]):
            col = acts_matrix[:, f].astype(np.float64)
            if col.std() > 1e-8:
                sw_corr_per_feat[f] = float(np.corrcoef(col, kl_arr)[0, 1])
    top_sw_feat_idx = np.argsort(np.abs(sw_corr_per_feat))[::-1][:10].tolist()
    print(f"\n  Top-10 features by |Pearson r with SW KL scores|:")
    for fi in top_sw_feat_idx:
        col     = acts_matrix[:, fi]
        nnz_idx = np.where(col > 0)[0]
        top_act = [hexamers[i] for i in nnz_idx[np.argsort(col[nnz_idx])[::-1]][:5]]
        print(f"    feature {fi:5d}  SW-r={sw_corr_per_feat[fi]:+.4f}"
              f"  GC-corr={corrs['gc'][fi]:+.3f}"
              f"  n_active={len(nnz_idx)}"
              f"  top_active={top_act}")

    # ── Heatmap: top-mono features × motif class ───────────────────────────────
    print("\n[analyze] Building heatmap...", flush=True)
    # Mean activation per (feature, motif_class) cell
    class_to_mask = {c: np.array([cl == c for cl in class_arr]) for c in MOTIF_CLASSES}
    heatmap = np.zeros((args.top_mono, len(MOTIF_CLASSES)))
    for row_i, fi in enumerate(top_mono_idx):
        for col_j, mc in enumerate(MOTIF_CLASSES):
            mask = class_to_mask[mc]
            if mask.sum() > 0:
                heatmap[row_i, col_j] = acts_matrix[mask, fi].mean()

    _plot_heatmap(
        heatmap, top_mono_idx, feat_entropy, MOTIF_CLASSES,
        out_path=str(out_dir / "sae_feature_heatmap.png"),
    )

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "model":           args.model,
        "layer":           args.layer,
        "sae_ckpt":        args.sae_ckpt,
        "n_features":      int(sae.n_features),
        "k":               int(sae.k),
        "top_mono_features": overlap_results,
        "top_sw_corr_features": [
            {
                "feature_idx": int(fi),
                "sw_pearson_r": float(sw_corr_per_feat[fi]),
                "gc_corr": float(corrs["gc"][fi]),
                "n_active_hexamers": int(np.sum(acts_matrix[:, fi] > 0)),
                "top_active_hexamers": [
                    hexamers[i]
                    for i in np.where(acts_matrix[:, fi] > 0)[0][
                        np.argsort(acts_matrix[acts_matrix[:, fi] > 0, fi])[::-1][:10]
                    ]
                ] if np.sum(acts_matrix[:, fi] > 0) > 0 else [],
            }
            for fi in top_sw_feat_idx
        ],
        "correlations_summary": {
            name: {
                "mean":  float(np.mean(corrs[name])),
                "std":   float(np.std(corrs[name])),
                "max":   float(np.max(np.abs(corrs[name]))),
                "top5_features": np.argsort(np.abs(corrs[name]))[::-1][:5].tolist(),
            }
            for name in corrs
        },
        "sw_dep_top_n":       args.top_sw_dep,
        "heatmap_classes":    MOTIF_CLASSES,
    }

    out_json = out_dir / "sae_feature_analysis.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[analyze] Results → {out_json}")

    # Save feature activation matrix (float16 for space)
    np.save(str(out_dir / "hexamer_sae_acts.npy"), acts_matrix.astype(np.float16))
    print(f"[analyze] Hexamer activation matrix → {out_dir}/hexamer_sae_acts.npy")

    print("\n[analyze] Done.")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _plot_heatmap(
    heatmap:     np.ndarray,   # [n_features, n_classes]
    feature_ids: list[int],
    feat_entropy: np.ndarray,
    class_names:  list[str],
    out_path:    str,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available — skipping heatmap.")
        return

    n_feat, n_class = heatmap.shape

    # Normalise each row to [0, 1] for visual clarity
    row_max = heatmap.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    heatmap_norm = heatmap / row_max

    fig, ax = plt.subplots(figsize=(max(8, n_class * 1.4), max(5, n_feat * 0.5)))

    im = ax.imshow(heatmap_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Mean activation (row-normalised)")

    ax.set_xticks(range(n_class))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)

    ytick_labels = [
        f"F{fid}  H={feat_entropy[fid]:.2f}" for fid in feature_ids
    ]
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(ytick_labels, fontsize=7)

    ax.set_xlabel("Hexamer motif class")
    ax.set_ylabel("SAE feature  (H = activation entropy)")
    ax.set_title(
        "Top monosemantic SAE features × hexamer motif class\n"
        "(row-normalised mean activation;  lower H = more monosemantic)"
    )

    # Annotate cells with raw mean values
    for i in range(n_feat):
        for j in range(n_class):
            val = heatmap[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=5, color="black" if heatmap_norm[i, j] < 0.6 else "white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analyze] Heatmap → {out_path}")


if __name__ == "__main__":
    main()
