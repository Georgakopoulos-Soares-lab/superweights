#!/usr/bin/env python3
"""
run_sw_hexamer_extended.py

Three extensions to the hexamer causal test:
  A: Motif enrichment for EUK GENERator top-500 hexamers by SW activation
  B: Prok GENERator hexamer motif enrichment (requires prok_json to exist)
  C: DNABERT-2 splice checkpoint hexamer causal test

Usage (after running run_sw_hexamer_causal.py for both EUK and PROK):

    python scripts/interpretability/run_sw_hexamer_extended.py \\
        --euk_json  results/sw_hexamer_causal.json \\
        --prok_json results/sw_hexamer_causal_generator_prok.json \\
        --dnabert2_ckpt results/gue_checkpoints/dnabert2_reconstructed \\
        --sw_index  results/super_weight_index.json \\
        --out_euk_motifs  results/sw_hexamer_motifs_top500_generator_euk.json \\
        --out_prok_motifs results/sw_hexamer_motifs_top500_generator_prok.json \\
        --out_dnabert2    results/sw_hexamer_causal_dnabert2_splice.json \\
        --plot results/sw_hexamer_extended.png \\
        --device cuda
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import fisher_exact, pearsonr
from statsmodels.stats.multitest import multipletests


# ─────────────────────────────────────────────────────────────────────────────
# Motif classifiers  (all operate on a 6-character uppercase DNA string)
# ─────────────────────────────────────────────────────────────────────────────

def _is_splice_donor(km: str) -> bool:
    """Contains the GT dinucleotide (canonical splice-donor GT|AAGT core)."""
    return "GT" in km


def _is_splice_acceptor(km: str) -> bool:
    """Pyrimidine-rich (≥3 C/T bases) and ends in AG (canonical -AG acceptor)."""
    return km.endswith("AG") and sum(c in "CT" for c in km) >= 3


def _is_tata_box(km: str) -> bool:
    """Contains TATA tetranucleotide (TATA-box core element)."""
    return "TATA" in km


def _is_gc_box(km: str) -> bool:
    """Contains the Sp1/GC-box hexamer GGGCGG."""
    return "GGGCGG" in km


def _is_kozak(km: str) -> bool:
    """Exact canonical Kozak hexamers only (GCCACC, GCCGCC)."""
    return km in {"GCCACC", "GCCGCC"}


def _is_cpg_rich(km: str) -> bool:
    """Contains ≥2 CpG dinucleotides."""
    return sum(km[i : i + 2] == "CG" for i in range(len(km) - 1)) >= 2


def _is_at_rich(km: str) -> bool:
    """≥5 out of 6 bases are A or T."""
    return sum(c in "AT" for c in km) >= 5


def _is_homopolymer(km: str) -> bool:
    """Any single base repeated ≥4 times consecutively."""
    return any(base * 4 in km for base in "ACGT")


# Prokaryote-specific motifs
def _is_shine_dalgarno(km: str) -> bool:
    """Ribosome-binding Shine–Dalgarno sequence variants."""
    return any(m in km for m in ("AGGAGG", "GAGGAG", "AAGGAG"))


def _is_minus10_box(km: str) -> bool:
    """Bacterial −10 Pribnow box variants."""
    return any(m in km for m in ("TATAAT", "TACAAT", "TATACT"))


def _is_minus35_box(km: str) -> bool:
    """Bacterial −35 box variants."""
    return any(m in km for m in ("TTGACA", "TTGATA", "TTGAAA"))


# ── Motif dictionaries ────────────────────────────────────────────────────────

EUK_MOTIFS: Dict[str, object] = {
    "splice_donor":   _is_splice_donor,
    "splice_acceptor": _is_splice_acceptor,
    "TATA_box":       _is_tata_box,
    "GC_box_Sp1":     _is_gc_box,
    "Kozak":          _is_kozak,
    "CpG_rich":       _is_cpg_rich,
    "AT_rich":        _is_at_rich,
    "homopolymer":    _is_homopolymer,
}

PROK_MOTIFS: Dict[str, object] = {
    **EUK_MOTIFS,
    "Shine_Dalgarno": _is_shine_dalgarno,
    "minus10_box":    _is_minus10_box,
    "minus35_box":    _is_minus35_box,
}


# ─────────────────────────────────────────────────────────────────────────────
# KL divergence / softmax helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p ‖ q) in nats — numerically stable."""
    p = p + eps;  q = q + eps
    p = p / p.sum();  q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fisher's exact test + BH correction
# ─────────────────────────────────────────────────────────────────────────────

def _motif_enrichment(
    kmers: List[str],
    kl_divs: np.ndarray,
    motif_fns: Dict,
    top_k: int = 500,
) -> List[Dict]:
    """
    For each motif class, 2×2 Fisher exact test:
        rows = top_k vs rest (by KL divergence — most SW-dependent sequences)
        cols = motif+ vs motif-
    Returns list of dicts, BH FDR-corrected.
    """
    N = len(kmers)
    rank = np.argsort(kl_divs)[::-1]
    top_set = set(rank[:top_k].tolist())

    # Classify every k-mer
    kmer_motifs: Dict[str, set] = {name: set() for name in motif_fns}
    for idx, km in enumerate(kmers):
        for name, fn in motif_fns.items():
            if fn(km):
                kmer_motifs[name].add(idx)

    raw_results = []
    for name, motif_set in kmer_motifs.items():
        fg_pos = len(top_set & motif_set)
        fg_neg = top_k - fg_pos
        bg_pos = len(motif_set) - fg_pos
        bg_neg = (N - top_k) - bg_pos
        table = [[fg_pos, fg_neg], [bg_pos, bg_neg]]
        _, p = fisher_exact(table, alternative="greater")
        denom = max(1, fg_neg * bg_pos)
        raw_results.append({
            "motif":         name,
            "top_k":         top_k,
            "fg_pos":        fg_pos,
            "fg_neg":        fg_neg,
            "bg_pos":        bg_pos,
            "bg_neg":        bg_neg,
            "n_motif_total": len(motif_set),
            "p_value":       float(p),
            "odds_ratio":    float(fg_pos * bg_neg) / denom,
        })

    # Benjamini-Hochberg FDR correction
    p_vals = np.array([r["p_value"] for r in raw_results])
    reject, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")
    for r, q, rej in zip(raw_results, q_vals, reject):
        r["q_value"]    = float(q)
        r["significant"] = bool(rej)

    return sorted(raw_results, key=lambda x: x["p_value"])


# ─────────────────────────────────────────────────────────────────────────────
# Extension A — EUK GENERator motif enrichment
# ─────────────────────────────────────────────────────────────────────────────

def run_extension_a(euk_json: str, top_k: int = 500) -> Dict:
    print("\n[Extension A] EUK GENERator motif enrichment (top-k by KL divergence) ...")
    d = json.load(open(euk_json))
    records = d["results"]

    kmers   = [r["kmer"]          for r in records]
    sw_acts = np.array([r["sw_activation"]   for r in records])
    kl_divs = np.array([r["kl_clean_ablated"] for r in records])

    enrichment = _motif_enrichment(kmers, kl_divs, EUK_MOTIFS, top_k)

    r_kl, p_kl = pearsonr(sw_acts, kl_divs)
    print(f"  n_kmers={len(kmers)}, r(SW_act, KL)={r_kl:.3f}, p={p_kl:.2e}")
    for e in enrichment:
        sig = "***" if e["significant"] else ""
        print(
            f"  {e['motif']:20s}  fg={e['fg_pos']}/{top_k}"
            f"  OR={e['odds_ratio']:.2f}  q={e['q_value']:.2e}  {sig}"
        )

    return {
        "source":               euk_json,
        "n_kmers":              len(kmers),
        "top_k":                top_k,
        "ranked_by":            "kl_divergence",
        "r_kl_sw_activation":   float(r_kl),
        "p_kl_sw_activation":   float(p_kl),
        "motif_enrichment":     enrichment,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extension B — PROK GENERator motif enrichment
# ─────────────────────────────────────────────────────────────────────────────

def run_extension_b(prok_json: str, top_k: int = 500) -> Dict:
    print("\n[Extension B] PROK GENERator motif enrichment (top-k by KL divergence) ...")
    d = json.load(open(prok_json))
    records = d["results"]

    kmers   = [r["kmer"]          for r in records]
    sw_acts = np.array([r["sw_activation"]   for r in records])
    kl_divs = np.array([r["kl_clean_ablated"] for r in records])

    enrichment = _motif_enrichment(kmers, kl_divs, PROK_MOTIFS, top_k)

    r_kl, p_kl = pearsonr(sw_acts, kl_divs)
    print(f"  n_kmers={len(kmers)}, r(SW_act, KL)={r_kl:.3f}, p={p_kl:.2e}")
    for e in enrichment:
        sig = "***" if e["significant"] else ""
        print(
            f"  {e['motif']:20s}  fg={e['fg_pos']}/{top_k}"
            f"  OR={e['odds_ratio']:.2f}  q={e['q_value']:.2e}  {sig}"
        )

    return {
        "source":               prok_json,
        "n_kmers":              len(kmers),
        "top_k":                top_k,
        "ranked_by":            "kl_divergence",
        "r_kl_sw_activation":   float(r_kl),
        "p_kl_sw_activation":   float(p_kl),
        "motif_enrichment":     enrichment,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extension C — DNABERT-2 splice checkpoint hexamer causal test
# ─────────────────────────────────────────────────────────────────────────────

def _get_dnabert2_down_projs(model, sw_by_layer: Dict[int, List[int]]) -> Dict:
    """Return {layer_idx: (module, rows_list)} for DNABERT-2's mlp.wo."""
    return {
        layer: (model.bert.encoder.layer[layer].mlp.wo, rows)
        for layer, rows in sw_by_layer.items()
    }


def run_extension_c(
    ckpt_dir: str,
    sw_index: str,
    device: str = "cuda",
    hf_home: str = "/work/11034/atzanakak/ls6/huggingface/.hf-cache",
) -> Dict:
    print("\n[Extension C] DNABERT-2 hexamer causal test ...")
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    model_id = "zhihan1996/DNABERT-2-117M"

    print(f"  Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=hf_home
    )

    print(f"  Loading model from {model_id} ...")
    model = AutoModelForMaskedLM.from_pretrained(
        model_id, trust_remote_code=True,
        cache_dir=hf_home, torch_dtype=torch.float32,
    )

    # Load fine-tuned backbone weights (ignore classifier head)
    ckpt_path = Path(ckpt_dir) / "model_state.pt"
    if ckpt_path.exists():
        state_dict = torch.load(ckpt_path, map_location="cpu")
        backbone_sd = {k: v for k, v in state_dict.items()
                       if not k.startswith("classifier")}
        missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
        print(f"  Loaded {len(backbone_sd)} backbone keys; "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"  WARNING: checkpoint not found at {ckpt_path}; using base weights")

    model = model.to(device).eval()

    # Build SW row index per layer
    sw_data = json.load(open(sw_index))
    sw_entries = sw_data["dnabert2"]["results"]
    sw_by_layer: Dict[int, List[int]] = {}
    for e in sw_entries:
        layer = int(e["layer"])
        row   = int(e["row"])
        sw_by_layer.setdefault(layer, []).append(row)

    down_projs = _get_dnabert2_down_projs(model, sw_by_layer)

    # All 4096 hexamers
    bases     = "ACGT"
    all_kmers = ["".join(c) for c in product(bases, repeat=6)]
    print(f"  Processing {len(all_kmers)} 6-mers ...")

    results: List[Dict] = []

    for idx, km in enumerate(all_kmers):
        if idx % 512 == 0:
            print(f"    {idx}/{len(all_kmers)}", flush=True)

        enc          = tokenizer(km, add_special_tokens=True, return_tensors="pt")
        input_ids    = enc["input_ids"].to(device)        # [1, seq_len]
        attn_mask    = enc["attention_mask"].to(device)
        seq_len      = input_ids.shape[1]

        # Positions of 6-mer tokens (exclude CLS at 0 and SEP at -1)
        kmer_positions = list(range(1, seq_len - 1))
        if not kmer_positions:
            continue   # Shouldn't happen for a non-empty 6-mer

        # ── Clean (unmasked) forward pass → SW activation ─────────────────
        layer_acts: Dict[int, float] = {}
        clean_hooks = []
        for layer_idx, (module, rows) in down_projs.items():
            layer_store: Dict = {}

            def _make_clean_hook(store: Dict, row_list: List[int]):
                def _hook(mod, inp, out):
                    # out may be [batch, seq_len, hidden] or [seq_len, hidden]
                    if out.dim() == 3:
                        act = out[0, kmer_positions, :][:, row_list]
                    else:
                        act = out[kmer_positions, :][:, row_list]
                    store["act"] = act.abs().max().item()
                return _hook

            h = module.register_forward_hook(
                _make_clean_hook(layer_store, rows)
            )
            clean_hooks.append((h, layer_idx, layer_store))

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask)

        for h, li, store in clean_hooks:
            h.remove()
            layer_acts[li] = store.get("act", 0.0)

        sw_activation = max(layer_acts.values()) if layer_acts else 0.0

        # ── Clean (unmasked) MLM logits at kmer positions ─────────────────
        # Using unmasked input is critical: [CLS, kmer_tok(s), SEP] differs
        # per k-mer, so logits are k-mer-specific.  Masking would produce an
        # identical [CLS, MASK, SEP] for every k-mer → constant KL.
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attn_mask)
        logits_clean = (
            out_clean.logits[0, kmer_positions, :].cpu().float().numpy()
        )   # [k, vocab]

        # ── Ablated (unmasked) forward pass ───────────────────────────────
        ablate_hooks = []
        for layer_idx, (module, rows) in down_projs.items():
            def _make_ablate_hook(row_list: List[int]):
                def _hook(mod, inp, out):
                    out_c = out.clone()
                    if out_c.dim() == 3:
                        out_c[:, :, row_list] = 0.0
                    else:
                        out_c[:, row_list] = 0.0
                    return out_c
                return _hook
            h = module.register_forward_hook(_make_ablate_hook(rows))
            ablate_hooks.append(h)

        with torch.no_grad():
            out_ablated = model(input_ids=input_ids, attention_mask=attn_mask)

        for h in ablate_hooks:
            h.remove()

        logits_ablated = (
            out_ablated.logits[0, kmer_positions, :].cpu().float().numpy()
        )   # [k, vocab]

        # ── KL divergence: sum across positions, normalised by n_tokens ────
        p_c = _softmax(logits_clean)    # [k, vocab]
        p_a = _softmax(logits_ablated)  # [k, vocab]
        n   = len(kmer_positions)
        kl  = sum(_kl(p_c[j], p_a[j]) for j in range(n)) / max(1, n)

        results.append({
            "kmer":             km,
            "sw_activation":    float(sw_activation),
            "kl_clean_ablated": float(kl),
            "n_tokens":         n,
        })

    # Correlation
    sw_acts = np.array([r["sw_activation"]    for r in results])
    kl_divs = np.array([r["kl_clean_ablated"] for r in results])
    r_kl, p_kl = pearsonr(sw_acts, kl_divs)
    print(f"  r(SW activation, KL) = {r_kl:.3f}  p = {p_kl:.2e}")

    # GT-containing k-mers in high-SW / high-KL quadrant (top 25th percentile)
    sw_thresh = float(np.percentile(sw_acts, 75))
    kl_thresh = float(np.percentile(kl_divs, 75))
    top_q     = [r for r in results
                 if r["sw_activation"] >= sw_thresh
                 and r["kl_clean_ablated"] >= kl_thresh]
    n_gt = sum("GT" in r["kmer"] for r in top_q)
    print(
        f"  GT-containing in top quadrant: {n_gt}/{len(top_q)}"
        f" = {n_gt / max(1, len(top_q)):.1%}"
    )

    # Motif enrichment on top-500 by SW activation
    top_k = 500
    enrichment = _motif_enrichment(
        [r["kmer"] for r in results], sw_acts, EUK_MOTIFS, top_k
    )
    print("  Motif enrichment (DNABERT-2):")
    for e in enrichment:
        sig = "***" if e["significant"] else ""
        print(
            f"    {e['motif']:20s}  fg={e['fg_pos']}/{top_k}"
            f"  OR={e['odds_ratio']:.2f}  q={e['q_value']:.2e}  {sig}"
        )

    return {
        "model":                  "dnabert2_splice",
        "ckpt_dir":               str(ckpt_dir),
        "n_kmers":                len(results),
        "r_kl_sw_activation":     float(r_kl),
        "p_kl_sw_activation":     float(p_kl),
        "top_quadrant_sw_thresh": sw_thresh,
        "top_quadrant_kl_thresh": kl_thresh,
        "top_quadrant_n":         len(top_q),
        "top_quadrant_gt":        n_gt,
        "motif_enrichment":       enrichment,
        "results":                results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined figure (3-row × 2-panel layout)
# ─────────────────────────────────────────────────────────────────────────────

def _kmer_color(km: str) -> str:
    if "GT" in km:
        return "red"
    if km.endswith("AG") and sum(c in "CT" for c in km) >= 3:
        return "orange"
    if sum(c in "AT" for c in km) >= 5:
        return "steelblue"
    return "grey"


def _plot_combined(
    euk_records:     List[Dict],
    prok_records:    Optional[List[Dict]],
    euk_enrichment:  List[Dict],
    prok_enrichment: Optional[List[Dict]],
    out_path: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("  [skip plot] matplotlib unavailable")
        return

    # Determine layout — EUK and PROK only (DNABERT-2 excluded from plot)
    scatter_panels = [("EUK GENERator",   euk_records)]
    if prok_records:
        scatter_panels.append(("PROK GENERator", prok_records))

    enrich_panels = [("EUK enrichment", euk_enrichment)]
    if prok_enrichment:
        enrich_panels.append(("PROK enrichment", prok_enrichment))

    n_cols = max(len(scatter_panels), len(enrich_panels))
    fig    = plt.figure(figsize=(6 * n_cols, 10))
    gs     = GridSpec(2, n_cols, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0 — scatter plots
    for col, (title, records) in enumerate(scatter_panels):
        ax      = fig.add_subplot(gs[0, col])
        sw_acts = np.array([r["sw_activation"]    for r in records])
        kl_divs = np.array([r["kl_clean_ablated"] for r in records])
        colors  = [_kmer_color(r["kmer"]) for r in records]
        ax.scatter(sw_acts, kl_divs, s=4, alpha=0.25, c=colors, rasterized=True)
        r, p = pearsonr(sw_acts, kl_divs)
        ax.set_title(f"{title}\nr={r:.3f}  p={p:.1e}", fontsize=9)
        ax.set_xlabel("SW activation (clean)", fontsize=8)
        ax.set_ylabel("KL(clean ‖ ablated)", fontsize=8)
        for label, color in [
            ("splice_donor (GT)", "red"),
            ("splice_acceptor",   "orange"),
            ("AT-rich",           "steelblue"),
            ("other",             "grey"),
        ]:
            ax.scatter([], [], s=15, color=color, label=label)
        ax.legend(fontsize=6, loc="upper left")

    # Row 1 — enrichment bar charts
    for col, (title, enrichment) in enumerate(enrich_panels):
        ax      = fig.add_subplot(gs[1, col])
        motifs  = [e["motif"]      for e in enrichment]
        ors     = [min(e["odds_ratio"], 10) for e in enrichment]
        sigs    = [e["significant"] for e in enrichment]
        bcolors = ["#e74c3c" if s else "#95a5a6" for s in sigs]
        ax.barh(range(len(motifs)), ors, color=bcolors)
        ax.set_yticks(range(len(motifs)))
        ax.set_yticklabels(motifs, fontsize=8)
        ax.axvline(x=1.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Odds ratio (capped at 10)", fontsize=8)
        ax.set_title(title, fontsize=9)
        for i, e in enumerate(enrichment):
            q_str = f"q={e['q_value']:.1e}" if e["significant"] else f"q={e['q_value']:.2f}"
            ax.text(max(ors[i], 0.05) + 0.1, i, q_str, va="center", fontsize=5.5)

    fig.suptitle(
        "Super-weight hexamer causal analysis — motif enrichment",
        fontsize=11, y=1.01,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Combined plot saved → {out_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hexamer causal extensions: motif enrichment + DNABERT-2"
    )
    p.add_argument("--euk_json",        default="results/sw_hexamer_causal.json",
                   help="Existing EUK GENERator hexamer causal JSON")
    p.add_argument("--prok_json",       default=None,
                   help="PROK GENERator hexamer causal JSON (run Extension B analysis if present)")
    p.add_argument("--dnabert2_ckpt",   default="results/gue_checkpoints/dnabert2_reconstructed",
                   help="Directory containing DNABERT-2 splice fine-tuned model_state.pt")
    p.add_argument("--sw_index",        default="results/super_weight_index.json")
    p.add_argument("--top_k",           type=int, default=500,
                   help="Number of top hexamers by SW activation for enrichment")
    p.add_argument("--out_euk_motifs",  default="results/sw_hexamer_motifs_top500_generator_euk.json")
    p.add_argument("--out_prok_motifs", default="results/sw_hexamer_motifs_top500_generator_prok.json")
    p.add_argument("--out_dnabert2",    default="results/sw_hexamer_causal_dnabert2_splice.json")
    p.add_argument("--plot",            default="results/sw_hexamer_extended.png")
    p.add_argument("--device",          default="cuda")
    p.add_argument("--hf_home",         default="/work/11034/atzanakak/ls6/huggingface/.hf-cache")
    p.add_argument("--skip_dnabert2",   action="store_true",
                   help="Skip Extension C (no GPU / checkpoint required)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Extension A ──────────────────────────────────────────────────────────
    euk_result = run_extension_a(args.euk_json, top_k=args.top_k)
    Path(args.out_euk_motifs).parent.mkdir(parents=True, exist_ok=True)
    json.dump(euk_result, open(args.out_euk_motifs, "w"), indent=2)
    print(f"  Saved → {args.out_euk_motifs}")

    # ── Extension B ──────────────────────────────────────────────────────────
    prok_result: Optional[Dict] = None
    if args.prok_json and Path(args.prok_json).exists():
        prok_result = run_extension_b(args.prok_json, top_k=args.top_k)
        json.dump(prok_result, open(args.out_prok_motifs, "w"), indent=2)
        print(f"  Saved → {args.out_prok_motifs}")
    else:
        print(f"\n[Extension B] Skipping — {args.prok_json} not found.")
        print("  Run first: python scripts/interpretability/run_sw_hexamer_causal.py "
              "--model generator_prokaryote "
              "--out results/sw_hexamer_causal_generator_prok.json")

    # ── Extension C ──────────────────────────────────────────────────────────
    db2_result: Optional[Dict] = None
    if not args.skip_dnabert2:
        db2_result = run_extension_c(
            args.dnabert2_ckpt, args.sw_index, args.device, args.hf_home
        )
        json.dump(db2_result, open(args.out_dnabert2, "w"), indent=2)
        print(f"  Saved → {args.out_dnabert2}")
    else:
        print("\n[Extension C] Skipped (--skip_dnabert2 flag).")

    # ── Combined figure ───────────────────────────────────────────────────────
    euk_records  = json.load(open(args.euk_json))["results"]
    prok_records = (
        json.load(open(args.prok_json))["results"]
        if args.prok_json and Path(args.prok_json).exists()
        else None
    )
    _plot_combined(
        euk_records,
        prok_records,
        euk_result["motif_enrichment"],
        prok_result["motif_enrichment"] if prok_result else None,
        args.plot,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
