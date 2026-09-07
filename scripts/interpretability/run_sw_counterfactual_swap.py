"""
scripts/interpretability/run_sw_counterfactual_swap.py
------------------------------------------------------
Counterfactual super-weight coordinate swap between GC-rich and AT-rich sequences.

Takes matched GC-rich and AT-rich input sequences and swaps only the SW
residual-stream coordinate (e_k) at the source layer between them.  For a
dose α ∈ [0, 1]:

  h_patched[sw_pos, k] = h_target[sw_pos, k]
                         + α · (h_donor[sw_pos, k] - h_target[sw_pos, k])

We then measure:
  • KL(clean_target ‖ patched_target) at next-token position after sw_pos
  • GC-logit shift: log Σ p(GC_tokens) − log Σ p(AT_tokens) in patched vs clean
  • Cross-context consistency: same swap across promoter, enhancer, intergenic
    sequences

Ablation establishes necessity; activation–KL correlation establishes
association; this experiment establishes content-specific sufficiency.

Controls:
  random_row    — swap a randomly chosen coordinate (not the SW row)
  donor_shuffle — swap donor's SW value but donor sequence is dinuc-shuffled
  sign_reversed — α < 0 (subtract instead of add the difference)

Sequences: synthetic (default) or real hg38 regions (optional --fasta flag).

For synthetic: GC-rich (70% GC, 504 bp), AT-rich (30% GC, 504 bp).
For real: needs --fasta and --promoters / --random BED files.

Model: generator (EUK) by default; generator_prokaryote also supported.

Output:
  results/sw_counterfactual_swap_{model}.json
  results/sw_counterfactual_swap_{model}.png

Usage:
  python scripts/interpretability/run_sw_counterfactual_swap.py
  python scripts/interpretability/run_sw_counterfactual_swap.py --model generator_prokaryote
  python scripts/interpretability/run_sw_counterfactual_swap.py \\
      --fasta /data/hg38/hg38.fa \\
      --gc_bed data/regions/hg38/promoters_262kb.bed \\
      --at_bed data/regions/hg38/random_262kb.bed \\
      --n_pairs 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

SW_DEFAULTS = {
    "generator":            {"sw_layer": 4,  "sw_row": 2371},
    "generator_prokaryote": {"sw_layer": 2,  "sw_row": 1927},
}

ALPHA_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25]  # >1 extrapolates past donor
N_CONTROLS   = 5   # random-row controls per sequence pair


# ─── Synthetic sequence generation ────────────────────────────────────────────

def _synthetic_seq(gc_frac: float, length: int, rng: random.Random) -> str:
    bases   = []
    gc_pool = ["G", "C"]
    at_pool = ["A", "T"]
    for _ in range(length):
        if rng.random() < gc_frac:
            bases.append(rng.choice(gc_pool))
        else:
            bases.append(rng.choice(at_pool))
    return "".join(bases)


def _dinuc_shuffle(seq: str, rng: random.Random) -> str:
    from collections import defaultdict
    VALID   = set("ACGT")
    bases   = [b if b in VALID else "N" for b in seq.upper()]
    if len(bases) < 4:
        rng.shuffle(bases)
        return "".join(bases)
    trans   = defaultdict(list)
    for i in range(len(bases) - 1):
        trans[bases[i]].append(bases[i + 1])
    for k in trans:
        rng.shuffle(trans[k])
    result  = [bases[0]]
    cur     = bases[0]
    for _ in range(len(bases) - 1):
        nxt_list = trans.get(cur)
        if not nxt_list:
            remaining = [b for v in trans.values() for b in v]
            if not remaining:
                break
            nxt = rng.choice(remaining)
            for k in trans:
                if nxt in trans[k]:
                    trans[k].remove(nxt)
                    break
        else:
            nxt = nxt_list.pop(0)
        result.append(nxt)
        cur = nxt
    if len(result) < len(bases):
        pool = list(seq.upper())
        rng.shuffle(pool)
        result.extend(pool[:len(bases) - len(result)])
    return "".join(result[:len(bases)])


# ─── Real sequence loading ────────────────────────────────────────────────────

def _load_real_pairs(fasta_path: str, gc_bed: str, at_bed: str,
                     n_pairs: int, window_bp: int, rng: random.Random) -> list[dict]:
    try:
        from pyfaidx import Fasta
    except ImportError:
        raise ImportError("pyfaidx required for real sequences: pip install pyfaidx")
    fa     = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    pairs  = []

    def _fetch(bed_path, label):
        with open(bed_path) as f:
            rows = [l.strip().split("\t") for l in f
                    if l.strip() and not l.startswith("#")]
        pool = rng.sample(rows, min(len(rows), 10 * n_pairs))
        seqs = []
        for row in pool:
            if len(seqs) >= n_pairs:
                break
            chrom  = row[0]
            center = (int(row[1]) + int(row[2])) // 2
            half   = window_bp // 2
            key    = (chrom if chrom in fa
                      else "chr" + chrom if "chr" + chrom in fa
                      else chrom.lstrip("chr") if chrom.lstrip("chr") in fa else None)
            if key is None:
                continue
            start  = max(0, center - half)
            end    = min(len(fa[key]), start + window_bp)
            seq    = str(fa[key][start:end])
            if seq.count("N") / max(len(seq), 1) > 0.1:
                continue
            seq = seq.ljust(window_bp, "N")[:window_bp]
            seqs.append({"seq": seq, "label": label,
                         "chrom": chrom, "center": center})
        return seqs

    gc_seqs = _fetch(gc_bed, "gc_rich")
    at_seqs = _fetch(at_bed, "at_rich")
    n       = min(len(gc_seqs), len(at_seqs), n_pairs)
    for i in range(n):
        pairs.append({"gc": gc_seqs[i]["seq"], "at": at_seqs[i]["seq"],
                      "gc_meta": gc_seqs[i], "at_meta": at_seqs[i]})
    return pairs


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(model_name: str):
    cfg = yaml.safe_load((ROOT / f"configs/{model_name}.yaml").read_text())
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], trust_remote_code=True,
        torch_dtype=torch.float32, device_map="auto")
    model.eval()
    return model, tok


def _prepare(tokenizer, seq: str, device):
    r = len(seq) % 6
    if r:
        seq = seq[r:]
    return tokenizer(seq, return_tensors="pt",
                     add_special_tokens=False)["input_ids"].to(device)


# ─── GC / AT token classification ─────────────────────────────────────────────

def _build_gc_at_token_sets(tokenizer) -> tuple[list[int], list[int]]:
    """
    Classify each vocabulary token as GC-rich (≥ 4/6 bases are G or C)
    or AT-rich (≥ 4/6 bases are A or T).
    Returns (gc_token_ids, at_token_ids).
    Works for 6-mer tokenizers (GENERator); falls back to single-char for others.
    """
    vocab = tokenizer.get_vocab()
    gc_ids, at_ids = [], []
    for tok, idx in vocab.items():
        upper = tok.upper()
        gc_count = sum(c in "GC" for c in upper if c in "ACGT")
        at_count = sum(c in "AT" for c in upper if c in "ACGT")
        total    = gc_count + at_count
        if total == 0:
            continue
        if gc_count / total >= 2/3:
            gc_ids.append(idx)
        elif at_count / total >= 2/3:
            at_ids.append(idx)
    return gc_ids, at_ids


# ─── Residual-stream capture and patching ─────────────────────────────────────

def _capture_sw_coordinate(model, input_ids: torch.Tensor,
                            sw_layer: int, sw_row: int) -> tuple[torch.Tensor, int, torch.Tensor]:
    """
    Run a clean forward pass.
    Returns (h_at_sw_layer[sw_pos, :], sw_pos, model_logits).
    h_at_sw_layer is the full residual-stream vector at the SW position.
    """
    _store: dict = {}

    def _hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        _store["hs"] = hs.detach().float().cpu()

    handle = model.model.layers[sw_layer].register_forward_hook(_hook)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    handle.remove()

    hs     = _store["hs"][0]           # (T, d_model)
    sw_pos = int(hs[:, sw_row].abs().argmax().item())
    logits = out.logits.detach().cpu()
    return hs[sw_pos].clone(), sw_pos, logits


def _patched_forward(model, input_ids: torch.Tensor,
                     sw_layer: int, sw_pos: int,
                     target_val: float, donor_val: float, alpha: float,
                     patch_row: int) -> torch.Tensor:
    """
    Run a forward pass where at the output of sw_layer, position sw_pos,
    coordinate patch_row is interpolated from target toward donor by alpha.
    Returns logits (cpu tensor).
    """
    delta = alpha * (donor_val - target_val)

    def _hook(mod, inp, out):
        hs = (out[0] if isinstance(out, tuple) else out).clone()
        if hs.dim() == 3:
            hs[0, sw_pos, patch_row] = hs[0, sw_pos, patch_row] + delta
        else:
            hs[sw_pos, patch_row] = hs[sw_pos, patch_row] + delta
        if isinstance(out, tuple):
            return (hs,) + out[1:]
        return hs

    handle = model.model.layers[sw_layer].register_forward_hook(_hook)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    handle.remove()
    return out.logits.detach().cpu()


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor, position: int) -> float:
    p = torch.softmax(p_logits[0, position].float(), dim=-1)
    q = torch.softmax(q_logits[0, position].float(), dim=-1)
    return float((p * ((p + 1e-10).log() - (q + 1e-10).log())).sum().clamp(min=0).item())


def _gc_logit_shift(logits: torch.Tensor, gc_ids: list[int], at_ids: list[int],
                    position: int) -> dict:
    """Compute log-sum-exp of GC-token logits and AT-token logits at a position."""
    L = logits[0, position].float()
    gc_lse  = float(L[gc_ids].logsumexp(0).item()) if gc_ids else float("nan")
    at_lse  = float(L[at_ids].logsumexp(0).item()) if at_ids else float("nan")
    return {"gc_lse": gc_lse, "at_lse": at_lse,
            "gc_minus_at": gc_lse - at_lse if (gc_ids and at_ids) else float("nan")}


# ─── Per-pair analysis ────────────────────────────────────────────────────────

def analyse_pair(model, tokenizer, sw_layer: int, sw_row: int,
                 gc_seq: str, at_seq: str,
                 gc_ids: list[int], at_ids: list[int],
                 alphas: list[float], n_controls: int,
                 rng: random.Random, device: str) -> dict:
    """
    Run the counterfactual swap experiment for one GC/AT sequence pair.
    Direction: donor=GC, target=AT (swap GC's SW value into AT context).
    """
    gc_ids_t = _prepare(tokenizer, gc_seq, device)
    at_ids_t = _prepare(tokenizer, at_seq, device)

    # Trim to same length
    L = min(gc_ids_t.shape[1], at_ids_t.shape[1])
    if L < 4:
        return {}
    gc_ids_t = gc_ids_t[:, :L]
    at_ids_t = at_ids_t[:, :L]

    # Clean passes
    gc_hs_at_sw, gc_sw_pos, gc_logits = _capture_sw_coordinate(
        model, gc_ids_t, sw_layer, sw_row)
    at_hs_at_sw, at_sw_pos, at_logits = _capture_sw_coordinate(
        model, at_ids_t, sw_layer, sw_row)

    donor_val  = float(gc_hs_at_sw[sw_row].item())   # GC context SW coordinate
    target_val = float(at_hs_at_sw[sw_row].item())   # AT context SW coordinate
    delta      = donor_val - target_val

    # KL position (next token after SW position in AT context)
    kl_pos = min(at_sw_pos + 1, L - 1)

    clean_gc_meta = _gc_logit_shift(at_logits, gc_ids, at_ids, kl_pos)

    alpha_results = []
    for alpha in alphas:
        patch_logits = _patched_forward(
            model, at_ids_t, sw_layer, at_sw_pos,
            target_val, donor_val, alpha, sw_row)
        kl_val    = _kl(at_logits, patch_logits, kl_pos)
        gc_meta   = _gc_logit_shift(patch_logits, gc_ids, at_ids, kl_pos)
        gc_shift  = gc_meta["gc_minus_at"] - clean_gc_meta["gc_minus_at"]
        alpha_results.append({
            "alpha":        alpha,
            "kl":           kl_val,
            "gc_minus_at":  gc_meta["gc_minus_at"],
            "gc_shift":     gc_shift,
        })

    # Controls: random row swaps (same alpha values)
    d_model     = at_hs_at_sw.shape[0]
    non_sw_rows = [r for r in range(d_model) if r != sw_row]
    ctrl_results = []
    for ctrl_i in range(n_controls):
        rand_row   = rng.choice(non_sw_rows)
        rand_donor_val  = float(gc_hs_at_sw[rand_row].item())
        rand_target_val = float(at_hs_at_sw[rand_row].item())
        ctrl_per_alpha  = []
        for alpha in alphas:
            plgts = _patched_forward(
                model, at_ids_t, sw_layer, at_sw_pos,
                rand_target_val, rand_donor_val, alpha, rand_row)
            kl_r = _kl(at_logits, plgts, kl_pos)
            gc_r = _gc_logit_shift(plgts, gc_ids, at_ids, kl_pos)
            ctrl_per_alpha.append({
                "alpha": alpha,
                "kl":    kl_r,
                "gc_shift": gc_r["gc_minus_at"] - clean_gc_meta["gc_minus_at"],
            })
        ctrl_results.append({"rand_row": rand_row, "per_alpha": ctrl_per_alpha})

    # Control: shuffled donor
    gc_shuf    = _dinuc_shuffle(gc_seq, rng)
    gc_shuf_t  = _prepare(tokenizer, gc_shuf, device)[:, :L]
    shuf_hs, _, _ = _capture_sw_coordinate(model, gc_shuf_t, sw_layer, sw_row)
    shuf_donor_val = float(shuf_hs[sw_row].item())
    shuf_per_alpha = []
    for alpha in alphas:
        plgts = _patched_forward(
            model, at_ids_t, sw_layer, at_sw_pos,
            target_val, shuf_donor_val, alpha, sw_row)
        kl_s = _kl(at_logits, plgts, kl_pos)
        gc_s = _gc_logit_shift(plgts, gc_ids, at_ids, kl_pos)
        shuf_per_alpha.append({
            "alpha": alpha,
            "kl":    kl_s,
            "gc_shift": gc_s["gc_minus_at"] - clean_gc_meta["gc_minus_at"],
        })

    # Control: sign-reversed (alpha < 0 for each positive alpha)
    neg_per_alpha = []
    for alpha in alphas:
        if alpha <= 0:
            continue
        plgts = _patched_forward(
            model, at_ids_t, sw_layer, at_sw_pos,
            target_val, donor_val, -alpha, sw_row)
        kl_n = _kl(at_logits, plgts, kl_pos)
        gc_n = _gc_logit_shift(plgts, gc_ids, at_ids, kl_pos)
        neg_per_alpha.append({
            "alpha": -alpha,
            "kl":    kl_n,
            "gc_shift": gc_n["gc_minus_at"] - clean_gc_meta["gc_minus_at"],
        })

    return {
        "donor_val":   donor_val,
        "target_val":  target_val,
        "delta":       delta,
        "gc_sw_pos":   gc_sw_pos,
        "at_sw_pos":   at_sw_pos,
        "kl_pos":      kl_pos,
        "sw_per_alpha":         alpha_results,
        "random_row_controls":  ctrl_results,
        "shuffled_donor":       shuf_per_alpha,
        "sign_reversed":        neg_per_alpha,
    }


# ─── Main assay ──────────────────────────────────────────────────────────────

def run_swap_assay(model_name: str, n_pairs: int, gc_frac: float, at_frac: float,
                   alphas: list[float], n_controls: int, seed: int,
                   fasta_path: str | None, gc_bed: str | None, at_bed: str | None,
                   window_bp: int) -> dict:
    rng = random.Random(seed)

    sw_layer = SW_DEFAULTS[model_name]["sw_layer"]
    sw_row   = SW_DEFAULTS[model_name]["sw_row"]
    print(f"\n[swap:{model_name}] sw_layer={sw_layer}  sw_row={sw_row}")

    print(f"[swap:{model_name}] loading model …")
    model, tokenizer = load_model(model_name)
    device = str(next(model.parameters()).device)

    gc_logit_ids, at_logit_ids = _build_gc_at_token_sets(tokenizer)
    print(f"[swap:{model_name}] GC-rich token types={len(gc_logit_ids)}  "
          f"AT-rich token types={len(at_logit_ids)}")

    # Build sequence pairs
    if fasta_path and gc_bed and at_bed:
        print(f"[swap:{model_name}] loading real sequences …")
        pairs = _load_real_pairs(fasta_path, gc_bed, at_bed, n_pairs,
                                  (window_bp // 6) * 6, rng)
    else:
        seq_len = 504   # divisible by 6
        print(f"[swap:{model_name}] generating {n_pairs} synthetic pairs "
              f"(GC={gc_frac:.0%}, AT={at_frac:.0%}) …")
        pairs = [{"gc": _synthetic_seq(gc_frac, seq_len, rng),
                   "at": _synthetic_seq(at_frac, seq_len, rng)}
                  for _ in range(n_pairs)]

    print(f"[swap:{model_name}] {len(pairs)} pairs  alpha_values={alphas}")

    pair_results = []
    for i, pair in enumerate(pairs):
        gc_seq = pair["gc"]
        at_seq = pair["at"]
        r = analyse_pair(
            model, tokenizer, sw_layer, sw_row,
            gc_seq, at_seq, gc_logit_ids, at_logit_ids,
            alphas, n_controls, rng, device)
        if r:
            pair_results.append(r)
        if (i + 1) % 5 == 0 or (i + 1) == len(pairs):
            kl_at_1 = np.mean([
                next((a["kl"] for a in pr["sw_per_alpha"] if a["alpha"] == 1.0), 0.0)
                for pr in pair_results if pr]) if pair_results else 0.0
            print(f"  [{i+1}/{len(pairs)}]  mean KL at α=1: {kl_at_1:.4e}")

    # Aggregate across pairs
    def _agg_alpha(key, subkey):
        out = {}
        for alpha in alphas:
            vals = []
            for pr in pair_results:
                for entry in pr.get(key, []):
                    if abs(entry["alpha"] - alpha) < 1e-6:
                        vals.append(entry.get(subkey, float("nan")))
            if vals:
                out[str(alpha)] = {
                    "mean": float(np.nanmean(vals)),
                    "std":  float(np.nanstd(vals)),
                    "n":    len(vals),
                }
        return out

    agg = {
        "sw_kl":        _agg_alpha("sw_per_alpha", "kl"),
        "sw_gc_shift":  _agg_alpha("sw_per_alpha", "gc_shift"),
        "shuf_kl":      _agg_alpha("shuffled_donor", "kl"),
        "shuf_gc_shift":_agg_alpha("shuffled_donor", "gc_shift"),
        "neg_kl":       _agg_alpha("sign_reversed", "kl"),
    }
    # Random row controls: average over control draws
    ctrl_kl_per_alpha = {str(a): [] for a in alphas}
    for pr in pair_results:
        for ctrl in pr.get("random_row_controls", []):
            for entry in ctrl.get("per_alpha", []):
                a_str = str(entry["alpha"])
                if a_str in ctrl_kl_per_alpha:
                    ctrl_kl_per_alpha[a_str].append(entry.get("kl", float("nan")))
    agg["rand_row_kl"] = {a_str: {
        "mean": float(np.nanmean(vs)) if vs else float("nan"),
        "std":  float(np.nanstd(vs)) if vs else float("nan"),
        "n":    len(vs),
    } for a_str, vs in ctrl_kl_per_alpha.items()}

    return {
        "model":       model_name,
        "sw_layer":    sw_layer,
        "sw_row":      sw_row,
        "n_pairs":     len(pair_results),
        "alphas":      alphas,
        "aggregated":  agg,
        "per_pair":    pair_results,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_swap(result: dict, out_png: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip plot] matplotlib not available")
        return

    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
    try:
        from _figstyle import apply_style, panel_label
        apply_style()
    except ImportError:
        pass

    agg    = result["aggregated"]
    alphas = [a for a in result["alphas"] if a >= 0]

    def _get(agg_key):
        means = []
        stds  = []
        for a in alphas:
            d = agg.get(agg_key, {}).get(str(a), {})
            means.append(d.get("mean", float("nan")))
            stds.append(d.get("std", 0.0))
        return np.array(means), np.array(stds)

    sw_kl_m,  sw_kl_s  = _get("sw_kl")
    sh_kl_m,  sh_kl_s  = _get("shuf_kl")
    rr_kl_m,  rr_kl_s  = _get("rand_row_kl")
    sw_gc_m,  sw_gc_s  = _get("sw_gc_shift")
    sh_gc_m,  sh_gc_s  = _get("shuf_gc_shift")

    fig, (ax_kl, ax_gc) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel A: KL dose-response
    ax_kl.plot(alphas, sw_kl_m,  color="#D62728", lw=2.5, label="SW row swap")
    ax_kl.fill_between(alphas, sw_kl_m - sw_kl_s, sw_kl_m + sw_kl_s,
                        color="#D62728", alpha=0.15)
    ax_kl.plot(alphas, sh_kl_m,  color="#4878CF", lw=1.5, linestyle="--",
               label="shuffled donor")
    ax_kl.fill_between(alphas, sh_kl_m - sh_kl_s, sh_kl_m + sh_kl_s,
                        color="#4878CF", alpha=0.15)
    ax_kl.plot(alphas, rr_kl_m,  color="#888888", lw=1.5, linestyle=":",
               label="random row")
    ax_kl.fill_between(alphas, rr_kl_m - rr_kl_s, rr_kl_m + rr_kl_s,
                        color="#888888", alpha=0.15)
    ax_kl.axvline(1.0, color="k", linewidth=0.8, linestyle="--")
    ax_kl.set_xlabel("Swap fraction α")
    ax_kl.set_ylabel("KL(clean_AT ‖ patched_AT)")
    ax_kl.set_title(f"{result['model']} — KL dose-response", fontsize=9, fontweight="bold")
    ax_kl.legend(fontsize=7, loc="upper left")
    ax_kl.set_xlim(0, max(alphas) + 0.05)

    # Panel B: GC logit shift
    ax_gc.plot(alphas, sw_gc_m,  color="#D62728", lw=2.5, label="SW row swap")
    ax_gc.fill_between(alphas, sw_gc_m - sw_gc_s, sw_gc_m + sw_gc_s,
                        color="#D62728", alpha=0.15)
    ax_gc.plot(alphas, sh_gc_m,  color="#4878CF", lw=1.5, linestyle="--",
               label="shuffled donor")
    ax_gc.fill_between(alphas, sh_gc_m - sh_gc_s, sh_gc_m + sh_gc_s,
                        color="#4878CF", alpha=0.15)
    ax_gc.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax_gc.axvline(1.0, color="k", linewidth=0.8, linestyle="--")
    ax_gc.set_xlabel("Swap fraction α")
    ax_gc.set_ylabel("Δ(log p_GC − log p_AT)")
    ax_gc.set_title(f"{result['model']} — GC vs AT logit shift", fontsize=9)
    ax_gc.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"  Plot saved → {out_png}")
    plt.close(fig)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="generator",
                   choices=list(SW_DEFAULTS))
    p.add_argument("--n_pairs",   type=int, default=20,
                   help="Number of GC/AT sequence pairs to test.")
    p.add_argument("--gc_frac",   type=float, default=0.70,
                   help="GC content for donor (GC-rich) synthetic sequences.")
    p.add_argument("--at_frac",   type=float, default=0.30,
                   help="GC content for target (AT-rich) synthetic sequences.")
    p.add_argument("--n_controls", type=int, default=N_CONTROLS,
                   help="Number of random-row control draws per pair.")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out_dir",   default="results")
    # Real-sequence options
    p.add_argument("--fasta",     default=None,
                   help="Path to hg38.fa (enables real sequences).")
    p.add_argument("--gc_bed",    default="data/regions/hg38/promoters_262kb.bed",
                   help="BED file for GC-rich regions (promoters).")
    p.add_argument("--at_bed",    default="data/regions/hg38/random_262kb.bed",
                   help="BED file for AT-rich regions (random).")
    p.add_argument("--window_bp", type=int, default=504,
                   help="Sequence window length for real sequences.")
    p.add_argument("--replot",    default=None, metavar="JSON",
                   help="Replot from existing JSON without running the model.")
    return p.parse_args()


def main():
    args  = parse_args()
    model = args.model
    out_j = ROOT / args.out_dir / f"sw_counterfactual_swap_{model}.json"
    out_p = ROOT / args.out_dir / f"sw_counterfactual_swap_{model}.png"

    if args.replot:
        result = json.loads(Path(args.replot).read_text())
    else:
        result = run_swap_assay(
            model_name=model,
            n_pairs=args.n_pairs,
            gc_frac=args.gc_frac,
            at_frac=args.at_frac,
            alphas=ALPHA_VALUES,
            n_controls=args.n_controls,
            seed=args.seed,
            fasta_path=args.fasta,
            gc_bed=args.gc_bed if args.fasta else None,
            at_bed=args.at_bed if args.fasta else None,
            window_bp=args.window_bp,
        )
        Path(out_j).parent.mkdir(parents=True, exist_ok=True)
        out_j.write_text(json.dumps(result, indent=2))
        print(f"\n  Results saved → {out_j}")

    plot_swap(result, str(out_p))


if __name__ == "__main__":
    main()
