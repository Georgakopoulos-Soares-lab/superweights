"""
run_prok_stratified_lifecycle.py
---------------------------------
Tests the "early-gating" hypothesis for GENERator-PROK super-weight row 1927.

The hypothesis: because PROK SW sits at layer 2 (earlier than EUK's layer 4),
it may act as a compositional gate — routing/suppressing signal BEFORE it
propagates, rather than driving predictions directly. This would explain the
sign-reversed activation/ablation-cost correlation (r = −0.710):
  - AT-rich hexamers (Q4: high sw_activation, near-zero KL) → weakly activate
    row 1927 → gate stays open → downstream layers proceed normally →
    ablating the row has little effect.
  - GC-rich hexamers (Q1: low sw_activation, high KL) → strongly activate
    row 1927 (large negative value) → gate closes / suppresses → ablating the
    row disrupts gating → high ablation cost.

What we track:
  For each representative hexamer in Q1 (high |sw_act|, high KL) and Q4
  (low |sw_act|, low KL):
    1. Build a repeated 504-bp sequence (hexamer × 84) so every token is the
       same hexamer.
    2. Run a single forward pass with LayerProbe hooks.
    3. At each layer, record:
         - mean |H[:, 1927]| over all token positions
         - mean |delta[:, 1927]| (block contribution)
    4. Compute persistence ratio = mean|H[L>2, 1927]| / |H[L=2, 1927]|

Gate-like signature: Q1 shows higher |H[L=2, 1927]| but LOWER or equal
  persistence ratio compared to Q4. The signal is written but not propagated
  selectively for GC-rich hexamers.

Driver-like signature: Q1 shows both higher |H[L=2, 1927]| AND higher
  persistence ratio. The signal is written and amplified.

Outputs:
  results/prok_stratified_lifecycle.json

Usage (from repo root):
  python scripts/analysis/run_prok_stratified_lifecycle.py [--n_sample 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── constants ─────────────────────────────────────────────────────────────────
SW_ROW   = 1927
SW_LAYER = 2
MODEL_KEY = "generator_prokaryote"
HEXAMER_DATA = ROOT / "results" / "sw_hexamer_causal_generator_prok.json"
OUT_PATH     = ROOT / "results" / "prok_stratified_lifecycle.json"


# ── reuse LayerProbe from run_activation_lifecycle ────────────────────────────
class LayerProbe:
    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self._inputs  = [None] * n_layers
        self._outputs = [None] * n_layers
        self._hooks   = []

    def _make_pre(self, i):
        def hook(module, args):
            self._inputs[i] = args[0].detach().float().cpu()
        return hook

    def _make_post(self, i):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self._outputs[i] = h.detach().float().cpu()
        return hook

    def register(self, model):
        for i, layer in enumerate(model.model.layers):
            self._hooks.append(layer.register_forward_pre_hook(self._make_pre(i)))
            self._hooks.append(layer.register_forward_hook(self._make_post(i)))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def deltas(self):
        out = []
        for inp, outp in zip(self._inputs, self._outputs):
            if inp is not None and outp is not None:
                out.append(outp.cpu() - inp.cpu())
            else:
                out.append(None)
        return out


def load_model_and_tokenizer(config: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {"float16": torch.float16, "float32": torch.float32,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(config.get("dtype", "float32"), torch.float32)
    tok = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], torch_dtype=dtype,
        trust_remote_code=True, device_map="auto")
    model.eval()
    return model, tok


def prepare_seq(seq: str, tokenizer) -> str:
    remainder = len(seq) % 6
    if remainder:
        seq = seq[remainder:]
    bos = tokenizer.bos_token or ""
    return bos + seq


def run_one_hexamer(model, tokenizer, hexamer: str, n_layers: int,
                    seq_len: int = 504) -> dict:
    """
    Build a repeated-hexamer sequence of length seq_len, run forward pass,
    return per-layer stats at SW row 1927.
    """
    reps   = (seq_len // len(hexamer)) + 1
    seq    = (hexamer * reps)[:seq_len]
    seq    = seq[: (len(seq) // 6) * 6]   # trim to multiple-of-6

    probe  = LayerProbe(n_layers)
    probe.register(model)
    try:
        seq_str = prepare_seq(seq, tokenizer)
        inputs  = tokenizer(seq_str, return_tensors="pt",
                            add_special_tokens=False).to(model.device)
        with torch.no_grad():
            model(**inputs)
    finally:
        probe.remove()

    deltas = probe.deltas()

    mean_h_sw    = []   # mean |H[:, SW_ROW]| over tokens, per layer
    mean_delt_sw = []   # mean |delta[:, SW_ROW]| over tokens, per layer
    signed_h_sw  = []   # signed mean H[:, SW_ROW] (to track polarity)

    for i in range(n_layers):
        outp = probe._outputs[i]
        delt = deltas[i]
        if outp is None:
            mean_h_sw.append(0.0)
            mean_delt_sw.append(0.0)
            signed_h_sw.append(0.0)
            continue
        h = outp[0].cpu().numpy()    # (T, d_model)
        mean_h_sw.append(float(np.abs(h[:, SW_ROW]).mean()))
        signed_h_sw.append(float(h[:, SW_ROW].mean()))
        if delt is not None:
            d = delt[0].cpu().numpy()
            mean_delt_sw.append(float(np.abs(d[:, SW_ROW]).mean()))
        else:
            mean_delt_sw.append(0.0)

    return {
        "mean_h_sw":    mean_h_sw,
        "signed_h_sw":  signed_h_sw,
        "mean_delt_sw": mean_delt_sw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sample", type=int, default=100,
                        help="Hexamers to sample from each quartile")
    parser.add_argument("--seq_len",  type=int, default=504)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── load hexamer data ─────────────────────────────────────────────────────
    data = json.loads(HEXAMER_DATA.read_text())
    results = data["results"]
    acts = np.array([r["sw_activation"] for r in results])

    q25 = np.percentile(acts, 25)
    q75 = np.percentile(acts, 75)

    q1_all = [r for r in results if r["sw_activation"] <= q25]   # GC-rich, high |act|, high KL
    q4_all = [r for r in results if r["sw_activation"] >= q75]   # AT-rich, low  |act|, low KL

    # Sample
    n1 = min(args.n_sample, len(q1_all))
    n4 = min(args.n_sample, len(q4_all))
    idx1 = rng.choice(len(q1_all), n1, replace=False)
    idx4 = rng.choice(len(q4_all), n4, replace=False)
    q1 = [q1_all[i] for i in idx1]
    q4 = [q4_all[i] for i in idx4]

    q1_mean_act = float(np.mean([r["sw_activation"] for r in q1]))
    q4_mean_act = float(np.mean([r["sw_activation"] for r in q4]))
    q1_mean_kl  = float(np.mean([r["kl_clean_ablated"] for r in q1]))
    q4_mean_kl  = float(np.mean([r["kl_clean_ablated"] for r in q4]))
    q1_mean_gc  = float(np.mean([r["gc_frac"] for r in q1]))
    q4_mean_gc  = float(np.mean([r["gc_frac"] for r in q4]))

    print(f"Q1 (GC-rich / high |act|): n={n1}, mean_act={q1_mean_act:.0f}, mean_kl={q1_mean_kl:.3f}, mean_gc={q1_mean_gc:.3f}")
    print(f"Q4 (AT-rich / low  |act|): n={n4}, mean_act={q4_mean_act:.0f}, mean_kl={q4_mean_kl:.3f}, mean_gc={q4_mean_gc:.3f}")

    # ── load model ────────────────────────────────────────────────────────────
    config_path = ROOT / "configs" / f"{MODEL_KEY}.yaml"
    config = yaml.safe_load(config_path.read_text())
    n_layers = config["num_layers"]

    print(f"\nLoading model from {config['model_id']} …")
    model, tokenizer = load_model_and_tokenizer(config)
    print("Model loaded.\n")

    # ── run per-hexamer forward passes ────────────────────────────────────────
    def run_group(group: list, label: str):
        profiles_h    = []
        profiles_delt = []
        profiles_sign = []
        for idx, rec in enumerate(group):
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{label}] {idx+1}/{len(group)} — {rec['kmer']} …")
            out = run_one_hexamer(model, tokenizer, rec["kmer"],
                                  n_layers, args.seq_len)
            profiles_h.append(out["mean_h_sw"])
            profiles_delt.append(out["mean_delt_sw"])
            profiles_sign.append(out["signed_h_sw"])
        return np.array(profiles_h), np.array(profiles_delt), np.array(profiles_sign)

    print("Running Q1 (GC-rich) …")
    q1_h, q1_d, q1_s = run_group(q1, "Q1")
    print("Running Q4 (AT-rich) …")
    q4_h, q4_d, q4_s = run_group(q4, "Q4")

    layers = list(range(n_layers))

    def persistence_ratio(h_matrix: np.ndarray, sw_layer: int) -> np.ndarray:
        """For each hexamer: mean |H| over layers > sw_layer / |H| at sw_layer."""
        at_sw     = h_matrix[:, sw_layer]                        # (N,)
        post_mean = h_matrix[:, sw_layer + 1:].mean(axis=1)     # (N,)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(at_sw > 0, post_mean / at_sw, np.nan)
        return ratio

    q1_persist = persistence_ratio(q1_h, SW_LAYER)
    q4_persist = persistence_ratio(q4_h, SW_LAYER)

    print(f"\nPersistence ratio (mean |H[L>2]| / |H[L=2]|):")
    print(f"  Q1 (GC-rich): {np.nanmean(q1_persist):.4f} ± {np.nanstd(q1_persist):.4f}")
    print(f"  Q4 (AT-rich): {np.nanmean(q4_persist):.4f} ± {np.nanstd(q4_persist):.4f}")

    print(f"\n|H[L=2, row 1927]| at the SW layer:")
    print(f"  Q1 (GC-rich): {q1_h[:, SW_LAYER].mean():.1f} ± {q1_h[:, SW_LAYER].std():.1f}")
    print(f"  Q4 (AT-rich): {q4_h[:, SW_LAYER].mean():.1f} ± {q4_h[:, SW_LAYER].std():.1f}")

    # ── determine gate vs driver ───────────────────────────────────────────────
    q1_l2 = q1_h[:, SW_LAYER].mean()
    q4_l2 = q4_h[:, SW_LAYER].mean()
    q1_pr = float(np.nanmean(q1_persist))
    q4_pr = float(np.nanmean(q4_persist))

    if q1_l2 > q4_l2 and q1_pr <= q4_pr:
        verdict = "GATE-LIKE: Q1 writes more at L2 but persists LESS (or equally) downstream. High ablation cost despite lower persistence — the gate state itself is the critical signal."
    elif q1_l2 > q4_l2 and q1_pr > q4_pr:
        verdict = "DRIVER-LIKE: Q1 writes more at L2 AND persists more downstream. Similar to EUK but with sign-reversed input correlation."
    elif q1_l2 <= q4_l2:
        verdict = "UNEXPECTED: Q4 (AT-rich) drives larger values at L2. Re-examine sw_activation sign convention."
    else:
        verdict = "AMBIGUOUS: differences are small."
    print(f"\nVerdict: {verdict}")

    # ── write output ──────────────────────────────────────────────────────────
    out = {
        "model": MODEL_KEY,
        "sw_row": SW_ROW,
        "sw_layer": SW_LAYER,
        "n_sample_per_quartile": args.n_sample,
        "layers": layers,
        "quartile_stats": {
            "Q1": {
                "label": "GC-rich / high |sw_act| / high KL",
                "n": n1,
                "mean_sw_activation": q1_mean_act,
                "mean_kl": q1_mean_kl,
                "mean_gc": q1_mean_gc,
                "hexamers": [r["kmer"] for r in q1],
            },
            "Q4": {
                "label": "AT-rich / low |sw_act| / low KL",
                "n": n4,
                "mean_sw_activation": q4_mean_act,
                "mean_kl": q4_mean_kl,
                "mean_gc": q4_mean_gc,
                "hexamers": [r["kmer"] for r in q4],
            },
        },
        "profiles": {
            "Q1": {
                "mean_h_sw":     q1_h.mean(axis=0).tolist(),
                "std_h_sw":      q1_h.std(axis=0).tolist(),
                "mean_delt_sw":  q1_d.mean(axis=0).tolist(),
                "mean_signed_sw": q1_s.mean(axis=0).tolist(),
                "per_hexamer_h": q1_h.tolist(),
            },
            "Q4": {
                "mean_h_sw":     q4_h.mean(axis=0).tolist(),
                "std_h_sw":      q4_h.std(axis=0).tolist(),
                "mean_delt_sw":  q4_d.mean(axis=0).tolist(),
                "mean_signed_sw": q4_s.mean(axis=0).tolist(),
                "per_hexamer_h": q4_h.tolist(),
            },
        },
        "persistence_ratio": {
            "Q1_mean": float(np.nanmean(q1_persist)),
            "Q1_std":  float(np.nanstd(q1_persist)),
            "Q4_mean": float(np.nanmean(q4_persist)),
            "Q4_std":  float(np.nanstd(q4_persist)),
        },
        "sw_layer_magnitude": {
            "Q1_mean": float(q1_h[:, SW_LAYER].mean()),
            "Q1_std":  float(q1_h[:, SW_LAYER].std()),
            "Q4_mean": float(q4_h[:, SW_LAYER].mean()),
            "Q4_std":  float(q4_h[:, SW_LAYER].std()),
        },
        "verdict": verdict,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
