"""
scripts/analysis/run_activation_lifecycle.py
---------------------------------------------
Track the "rise-plateau-fall" lifecycle of super-weight (SW) channel magnitudes
across all Transformer layers of GENERator EUK or PROK, reproducing the analysis
in Sun et al. 2026 (arXiv 2603.05498) Fig. 1 for genomic LMs.

For each forward pass we register hooks on every model.layers[i] to capture:
  - H_{i+1}  : post-residual hidden state   (= input to next layer)
  - delta_i   : block contribution H_{i+1} - H_i

We then track:
  - Per-SW-channel: magnitude of that channel in H and in delta
  - Top-3 global max channels in H (to verify SW channel is indeed the global max)

Outputs
-------
  results/activation_lifecycle_{model}.json
    {
      "model": "generator",
      "sw_layer": 4,
      "sw_rows": [2371, 1522],
      "n_seqs": 20,
      "layers": [0, 1, ..., 29],
      "mean_post_residual_sw0":  [...],   # mean over seqs of |H[:, sw_rows[0]]|.max(token)
      "mean_post_residual_sw1":  [...],
      "mean_post_residual_top3": [..., ...],  # list of 3 lists
      "mean_block_delta_sw0":    [...],
      "mean_block_delta_sw1":    [...],
    }

Usage
-----
  # From within the genomic-super-weights project dir:
  python scripts/analysis/run_activation_lifecycle.py --model generator
  python scripts/analysis/run_activation_lifecycle.py --model generator_prokaryote
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── project path ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── SW positions (from super_weight_index.json) ───────────────────────────────
_SW_DEFAULTS = {
    "generator":             {"layer": 4,  "rows": [2371, 1522]},
    "generator_prokaryote":  {"layer": 2,  "rows": [1927]},
}

# ── probe sequences ───────────────────────────────────────────────────────────
# Human ACTB (504 bp, divisible by 6) — standard EUK probe
_ACTB_504 = (
    "ATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAAGCCGGCTTCGCGGGCGACGAT"
    "GCCCCGAGGGCCGTCTTCCCCTCCATCGTGGGGCGCCCCAGGCACCAGGGCGTGATGGTGGGCATGGGTCAGAAG"
    "GATTCCTATGTGGGCGACGAGGCCCAGAGCAAGAGAGGCATCCTCACCCTGAAGTACCCCATCGAGCACGGCATC"
    "GTCACCAACTGGGACGACATGGAGAAAATCTGGCACCACACCTTCTACAATGAGCTGCGTGTGGCTCCCGAGGAG"
    "CACCCCGTGCTGCTCACCGAGGCCCCCCTGAACCCGAAGGCCAACCGCGAGAAGATGACCCAGATCATGTTTGAG"
    "ACCTTCAATACCCCCGCCATGTACGTTGCTATCCAGGCTGTGCTATCCCTGTACGCCTCTGGCCGTACCACTGGC"
    "ATCGTGATGGACTCCGGTGACGGGGTCACCCACACTGTGCCC"
)

# E. coli rpoB fragment (504 bp, divisible by 6) — standard PROK probe
_RPOB_504 = (
    "ATGCGCAAAGCGCGTCTCGAAACCGGTGCAGAAGCGTTCGTTCAGCGCATCGAAGAAATCGACCGTCTGCGTGAA"
    "ATCGACGAAGAAGCGAAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAA"
    "GAAGCGCGTGCGAAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAAGAA"
    "GCGCGTGCGAAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAAGAAGCG"
    "CGTGCGAAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAAGAAGCGCGT"
    "GCGAAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAAGAAGCGCGTGCG"
    "AAAGCGCGTGAAGAAGCGCGTGCGAAAGCGCGTGAAGAAGCG"
)


def _random_seq(length: int, rng: np.random.Generator) -> str:
    """Generate a random DNA sequence of given length (rounded down to multiple of 6)."""
    length = (length // 6) * 6
    return "".join(rng.choice(list("ACGT"), size=length))


def _make_sequences(model_key: str, n: int, seq_len: int, seed: int = 42):
    """Return n sequences: first is the canonical probe, rest are random."""
    rng = np.random.default_rng(seed)
    probe = _ACTB_504 if "prokaryote" not in model_key else _RPOB_504
    seqs = [probe[:seq_len] if len(probe) >= seq_len else probe]
    while len(seqs) < n:
        seqs.append(_random_seq(seq_len, rng))
    return seqs[:n]


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(config: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {"float16": torch.float16, "float32": torch.float32,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(config.get("dtype", "float32"), torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def prepare_seq(seq: str, tokenizer) -> str:
    """Trim to multiple of 6 and prepend BOS if any."""
    remainder = len(seq) % 6
    if remainder:
        seq = seq[remainder:]
    bos = tokenizer.bos_token or ""
    return bos + seq


# ── hook infrastructure ───────────────────────────────────────────────────────

class LayerProbe:
    """Captures per-layer hidden state input and output for one forward pass."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self._inputs  = [None] * n_layers  # H_i before the layer
        self._outputs = [None] * n_layers  # H_{i+1} after residual
        self._hooks   = []

    def _make_pre(self, i):
        def hook(module, args):
            # args[0] is the hidden state tensor: (1, T, d_model)
            h = args[0].detach().float()
            self._inputs[i] = h
        return hook

    def _make_post(self, i):
        def hook(module, args, output):
            # For Llama layers, output is a tuple; output[0] is (1, T, d_model)
            if isinstance(output, tuple):
                h = output[0].detach().float()
            else:
                h = output.detach().float()
            self._outputs[i] = h
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
        """Return list of (H_{i+1} - H_i) for each layer."""
        out = []
        for inp, outp in zip(self._inputs, self._outputs):
            if inp is not None and outp is not None:
                out.append(outp - inp)
            else:
                out.append(None)
        return out


# ── per-sequence extraction ───────────────────────────────────────────────────

def run_one_seq(model, tokenizer, seq: str, sw_rows: list[int], n_layers: int):
    """
    Returns:
        post_residual_sw  : (n_layers, len(sw_rows))  — max |h[:, sw_row]| over tokens
        block_delta_sw    : (n_layers, len(sw_rows))
        post_residual_top3: (n_layers, 3)             — top-3 channel max magnitudes
    """
    probe = LayerProbe(n_layers)
    probe.register(model)
    try:
        seq_str = prepare_seq(seq, tokenizer)
        inputs = tokenizer(seq_str, return_tensors="pt",
                           add_special_tokens=False).to(model.device)
        with torch.no_grad():
            model(**inputs)
    finally:
        probe.remove()

    deltas = probe.deltas()

    post_sw   = np.zeros((n_layers, len(sw_rows)), dtype=np.float32)
    delta_sw  = np.zeros((n_layers, len(sw_rows)), dtype=np.float32)
    post_top3 = np.zeros((n_layers, 3), dtype=np.float32)

    for i, (outp, delt) in enumerate(zip(probe._outputs, deltas)):
        if outp is None:
            continue
        # outp: (1, T, d_model)
        h = outp[0].cpu().numpy()   # (T, d_model)
        for j, r in enumerate(sw_rows):
            post_sw[i, j]  = float(np.abs(h[:, r]).max())

        # top-3 channels by max absolute value across tokens
        channel_max = np.abs(h).max(axis=0)  # (d_model,)
        top3_idx    = np.argpartition(channel_max, -3)[-3:]
        post_top3[i, :] = np.sort(channel_max[top3_idx])[::-1]

        if delt is not None:
            d = delt[0].cpu().numpy()
            for j, r in enumerate(sw_rows):
                delta_sw[i, j] = float(np.abs(d[:, r]).max())

    return post_sw, delta_sw, post_top3


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="generator",
                        choices=["generator", "generator_prokaryote"])
    parser.add_argument("--n_seqs",   type=int, default=20)
    parser.add_argument("--seq_len",  type=int, default=504)
    parser.add_argument("--sw_index", default=None,
                        help="Path to super_weight_index.json (auto-detected if omitted)")
    parser.add_argument("--out_dir",  default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # SW positions
    sw_idx_path = args.sw_index or str(ROOT / "results" / "super_weight_index.json")
    with open(sw_idx_path) as f:
        sw_index = json.load(f)
    if args.model in sw_index:
        sw_layer = sw_index[args.model]["results"][0]["layer"]
        sw_rows  = [r["row"] for r in sw_index[args.model]["results"]]
    else:
        sw_layer = _SW_DEFAULTS[args.model]["layer"]
        sw_rows  = _SW_DEFAULTS[args.model]["rows"]

    print(f"[lifecycle] model={args.model}  sw_layer={sw_layer}  sw_rows={sw_rows}")

    config_path = ROOT / "configs" / f"{args.model}.yaml"
    config = yaml.safe_load(config_path.read_text())
    n_layers = config["num_layers"]

    print(f"[lifecycle] Loading model from {config['model_id']} …")
    model, tokenizer = load_model_and_tokenizer(config)
    print(f"[lifecycle] Model loaded. Probing {args.n_seqs} sequences …")

    sequences = _make_sequences(args.model, args.n_seqs, args.seq_len)

    all_post_sw   = []   # list of (n_layers, n_sw_rows) arrays
    all_delta_sw  = []
    all_post_top3 = []   # list of (n_layers, 3)

    for idx, seq in enumerate(sequences):
        print(f"  seq {idx+1}/{args.n_seqs} …", end="\r", flush=True)
        post_sw, delta_sw, post_top3 = run_one_seq(
            model, tokenizer, seq, sw_rows, n_layers)
        all_post_sw.append(post_sw)
        all_delta_sw.append(delta_sw)
        all_post_top3.append(post_top3)

    print()

    mean_post_sw   = np.mean(all_post_sw,   axis=0)   # (n_layers, n_sw_rows)
    mean_delta_sw  = np.mean(all_delta_sw,  axis=0)
    mean_post_top3 = np.mean(all_post_top3, axis=0)   # (n_layers, 3)

    result = {
        "model":     args.model,
        "sw_layer":  sw_layer,
        "sw_rows":   sw_rows,
        "n_seqs":    args.n_seqs,
        "layers":    list(range(n_layers)),
        # post-residual SW channel magnitude (averaged over sequences)
        "mean_post_residual_sw": mean_post_sw.tolist(),
        # block-delta SW channel magnitude
        "mean_block_delta_sw":   mean_delta_sw.tolist(),
        # top-3 global channel magnitudes in post-residual H
        "mean_post_residual_top3": mean_post_top3.tolist(),
        # per-sequence arrays for variance / individual traces
        "per_seq_post_residual_sw":   np.stack(all_post_sw).tolist(),
        "per_seq_block_delta_sw":     np.stack(all_delta_sw).tolist(),
    }

    out_path = out_dir / f"activation_lifecycle_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[lifecycle] Saved → {out_path}")


if __name__ == "__main__":
    main()
