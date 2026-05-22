# sae/collect.py
"""
Collect post-residual hidden-state activations from a frozen GENERator model.

For each target layer, the output of model.model.layers[layer_idx] is the
full post-residual hidden state — identical to the residual stream that feeds
the next layer.  We hook this output, collect [seq_len, hidden_size] tensors,
flatten across sequence positions, and save in float16 shards.

Usage
-----
  python sae/collect.py \\
      --model    generator                              \\
      --layer    4                                      \\
      --fasta    data/reference/ecoli/ecoli_k12.fna    \\
      --out_dir  data/sae_acts/generator_euk_layer4    \\
      --max_tokens  50_000_000                          \\
      --shard_tokens 500_000                            \\
      --chunk_tokens 512                                \\
      --batch_seqs   32

For EUK (hg38), supply a BED file with genomic regions and the genome FASTA:
  python sae/collect.py \\
      --model generator --layer 4 \\
      --fasta data/hg38.fa --bed data/regions/hg38/random_262kb.bed \\
      --out_dir data/sae_acts/generator_euk_layer4 \\
      --max_tokens 50_000_000

Saved shards
------------
  <out_dir>/shard_00000.npy   float16  [n_tokens, hidden_size]
  <out_dir>/shard_00001.npy
  ...
  <out_dir>/meta.json         {model, layer, hidden_size, n_shards, total_tokens}
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List

import numpy as np
import torch
import yaml

# Repo root on sys.path so we can import wrappers / configs
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from models import WRAPPER_MAP


# ──────────────────────────────────────────────────────────────────────────────
# FASTA helpers
# ──────────────────────────────────────────────────────────────────────────────

def _iter_fasta(fasta_path: str) -> Iterator[tuple[str, str]]:
    """Yield (name, sequence) pairs from a plain FASTA file."""
    name, parts = None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None and parts:
                    yield name, "".join(parts)
                name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.upper())
    if name is not None and parts:
        yield name, "".join(parts)


def _iter_fasta_indexed(fasta_path: str, bed_path: str) -> Iterator[tuple[str, str]]:
    """
    Yield (region_name, sequence) pairs extracted from a genome FASTA using
    regions specified in a BED file.  Requires pyfaidx; falls back to plain
    iteration over the FASTA if pyfaidx is not available.
    """
    try:
        from pyfaidx import Fasta
        genome = Fasta(fasta_path)
        with open(bed_path) as fh:
            for line in fh:
                parts = line.rstrip().split("\t")
                if len(parts) < 3:
                    continue
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                name = parts[3] if len(parts) > 3 else f"{chrom}:{start}-{end}"
                try:
                    seq = str(genome[chrom][start:end]).upper()
                    yield name, seq
                except (KeyError, Exception):
                    continue
    except ImportError:
        print(
            "[warn] pyfaidx not available; falling back to streaming the full FASTA. "
            "BED file ignored.  Install pyfaidx for region-based extraction.",
            flush=True,
        )
        yield from _iter_fasta(fasta_path)


def _chunk_sequence(seq: str, chunk_bp: int, min_bp: int = 60) -> List[str]:
    """
    Split a DNA sequence into non-overlapping windows of `chunk_bp` base pairs.
    Only complete windows are returned; the trailing remainder is dropped.
    `chunk_bp` must be divisible by 6 (GENERator's 6-mer tokenizer).
    Ambiguous bases (N-heavy regions) are skipped if N-fraction > 0.1.
    """
    chunks = []
    for start in range(0, len(seq) - chunk_bp + 1, chunk_bp):
        w = seq[start : start + chunk_bp]
        if len(w) < min_bp:
            continue
        n_frac = w.count("N") / len(w)
        if n_frac > 0.1:
            continue
        # Replace remaining Ns with A (rare, avoids tokenizer errors)
        w = w.replace("N", "A")
        chunks.append(w)
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Activation collection
# ──────────────────────────────────────────────────────────────────────────────

def _get_layer_module(model_nn, layer_idx: int):
    """Return the LlamaDecoderLayer at position layer_idx."""
    return model_nn.model.layers[layer_idx]


def collect_activations(
    wrapper,
    sequences: List[str],
    layer_idx: int,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run frozen forward passes on `sequences` and return the post-residual
    hidden states at `layer_idx` as float16 numpy arrays.

    Returns
    -------
    np.ndarray  shape [total_tokens, hidden_size]  dtype float16
    """
    model     = wrapper.model
    tokenizer = wrapper.tokenizer
    device    = next(model.parameters()).device

    layer_module = _get_layer_module(model, layer_idx)

    all_acts = []

    for b_start in range(0, len(sequences), batch_size):
        batch_seqs = sequences[b_start : b_start + batch_size]

        # Tokenize (GENERator needs seq len divisible by 6 — guaranteed by _chunk_sequence)
        tok_out = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids      = tok_out["input_ids"].to(device)
        attention_mask = tok_out.get("attention_mask", torch.ones_like(input_ids)).to(device)

        store = {}

        def _hook(_mod, _inp, output, _s=store):
            # LlamaDecoderLayer returns (hidden_states, ...) tuple
            h = output[0] if isinstance(output, tuple) else output
            # Clip before float16 cast to avoid Inf from super-weight overflow
            # (float16 max = 65504; super-weight channels can exceed this).
            h_f32 = h.detach().float().clamp(-60000.0, 60000.0)
            _s["h"] = h_f32.cpu().to(torch.float16)  # [B, L, D]

        handle = layer_module.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()

        h    = store["h"]                        # [B, L, D]
        mask = attention_mask.cpu().bool()        # [B, L]

        # Flatten: keep only non-padded positions
        for seq_idx in range(h.shape[0]):
            valid_len = mask[seq_idx].sum().item()
            all_acts.append(h[seq_idx, :valid_len, :].numpy())  # [L_valid, D]

    if not all_acts:
        return np.empty((0,), dtype=np.float16)

    return np.concatenate(all_acts, axis=0)   # [total_tokens, D]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect GENERator residual-stream activations for SAE training."
    )
    parser.add_argument("--model",  required=True, choices=list(WRAPPER_MAP.keys()))
    parser.add_argument("--layer",  type=int, required=True,
                        help="Layer index for activation hook (EUK=4, PROK=2)")
    parser.add_argument("--fasta",  required=True,
                        help="Path to genome FASTA (or one of multiple FASTAs)")
    parser.add_argument("--bed",    default=None,
                        help="BED file for region extraction (requires pyfaidx). "
                             "If omitted, the full FASTA is streamed.")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to save activation shards")
    parser.add_argument("--max_tokens",   type=int, default=50_000_000,
                        help="Stop after collecting this many tokens (default: 50M)")
    parser.add_argument("--shard_tokens", type=int, default=500_000,
                        help="Tokens per shard file (default: 500K)")
    parser.add_argument("--chunk_tokens", type=int, default=512,
                        help="Tokens per sequence chunk (default: 512 = 3072 bp)")
    parser.add_argument("--batch_seqs",   type=int, default=32,
                        help="Sequences per GPU batch (default: 32)")
    parser.add_argument("--n_passes",     type=int, default=1,
                        help="Number of passes over the FASTA (PROK: use >1 to reach token target)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_bp = args.chunk_tokens * 6  # 6 bp per token

    # ── Load model ─────────────────────────────────────────────────────────────
    config = yaml.safe_load(open(_REPO / f"configs/{args.model}.yaml"))
    wrapper = WRAPPER_MAP[args.model](config)
    wrapper.load()
    wrapper.model.eval()

    hidden_size = wrapper.model.config.hidden_size
    print(f"[collect] model={args.model}  layer={args.layer}  hidden_size={hidden_size}")
    print(f"[collect] chunk_bp={chunk_bp}  shard_tokens={args.shard_tokens}  max_tokens={args.max_tokens}")

    # ── Stream sequences ────────────────────────────────────────────────────────
    total_tokens = 0
    shard_idx    = 0
    shard_buffer = []   # list of float16 arrays

    def flush_shard():
        nonlocal shard_idx, shard_buffer
        arr  = np.concatenate(shard_buffer, axis=0).astype(np.float16)
        path = out_dir / f"shard_{shard_idx:05d}.npy"
        np.save(str(path), arr)
        print(f"  shard {shard_idx:05d}: {arr.shape[0]:,} tokens → {path}", flush=True)
        shard_idx   += 1
        shard_buffer = []

    seq_buffer = []   # batch of string sequences

    def process_batch():
        nonlocal total_tokens, shard_buffer, seq_buffer
        if not seq_buffer:
            return
        acts = collect_activations(wrapper, seq_buffer, args.layer, args.batch_seqs)
        seq_buffer.clear()
        if acts.shape[0] == 0:
            return
        shard_buffer.append(acts)
        total_tokens += acts.shape[0]
        # Flush shard when buffer is large enough
        shard_buf_tokens = sum(a.shape[0] for a in shard_buffer)
        if shard_buf_tokens >= args.shard_tokens:
            flush_shard()

    print(f"\n[collect] Streaming FASTA (pass 1/{args.n_passes})...", flush=True)

    for pass_idx in range(args.n_passes):
        if total_tokens >= args.max_tokens:
            break
        if args.bed:
            seq_iter = _iter_fasta_indexed(args.fasta, args.bed)
        else:
            seq_iter = _iter_fasta(args.fasta)

        for seq_name, seq in seq_iter:
            if total_tokens >= args.max_tokens:
                break
            chunks = _chunk_sequence(seq, chunk_bp)
            for chunk in chunks:
                seq_buffer.append(chunk)
                if len(seq_buffer) >= args.batch_seqs * 4:
                    process_batch()
                if total_tokens >= args.max_tokens:
                    break

        print(
            f"  pass {pass_idx+1}/{args.n_passes}: {total_tokens:,} tokens collected so far",
            flush=True,
        )

    # Flush remainder
    if seq_buffer:
        process_batch()
    if shard_buffer:
        flush_shard()

    # ── Save metadata ────────────────────────────────────────────────────────────
    meta = {
        "model":        args.model,
        "layer":        args.layer,
        "hidden_size":  hidden_size,
        "n_shards":     shard_idx,
        "total_tokens": total_tokens,
        "chunk_tokens": args.chunk_tokens,
        "dtype":        "float16",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[collect] Done.  {total_tokens:,} tokens in {shard_idx} shards → {out_dir}")


if __name__ == "__main__":
    main()
