#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

from superweights.activation_maps import ActivationMapCapturer
from superweights.borzoi import default_device, detect_seq_len, detect_seq_len_from_crop, load_borzoi, score_expression
from superweights.genome import Genome, bin_size_bp, window_start0
from superweights.io import ensure_dir, write_parquet
from superweights.variants import parse_variant_id_any
from superweights_borzoi.encoding import one_hot_encode_batch
from superweights_borzoi.genome import make_ref_alt_sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate motifs via in-silico ablation (shuffle/substitute) and re-scoring.")

    p.add_argument("--instances", required=True, help="Parquet/TSV with motif instances. Needs chrom,start0,end0 and layer,channel.")
    p.add_argument("--fasta", required=True)
    p.add_argument("--outdir", default="reports/ablation")

    p.add_argument("--model_name", default="johahi/borzoi-replicate-0")
    p.add_argument("--output_key", default=None)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=4)

    p.add_argument("--mode", choices=["shuffle", "substitute"], default="shuffle")
    p.add_argument("--sub_base", default="N", help="If mode=substitute, replace motif with this base (default N).")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument(
        "--use_variant_center",
        action="store_true",
        help="If instances include variant_id,pos/ref/alt context, center window on variant instead of motif center.",
    )

    return p.parse_args()


def _load_instances(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_parquet(path)
    return df


def _mutate_seq(seq: str, start: int, end: int, rng: np.random.Generator, mode: str, sub_base: str) -> str:
    if start < 0 or end > len(seq) or end <= start:
        raise ValueError("Invalid mutation bounds")

    motif = seq[start:end]
    if mode == "shuffle":
        arr = list(motif)
        rng.shuffle(arr)
        motif2 = "".join(arr)
    else:
        motif2 = (sub_base.upper() * (end - start))

    return seq[:start] + motif2 + seq[end:]


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    outdir = ensure_dir(args.outdir)

    df = _load_instances(args.instances)

    required = {"chrom", "start0", "end0", "layer", "channel"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Instances missing columns: {sorted(missing)}")

    device = default_device()
    wrapper = load_borzoi(args.model_name, device=device, output_key=args.output_key)

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = detect_seq_len_from_crop(wrapper.model) or detect_seq_len(wrapper.model) or 262144
    seq_len = int(seq_len)

    # dummy forward acceptance
    with torch.no_grad():
        _ = wrapper.model(torch.zeros((1, 4, seq_len), device=device, dtype=torch.float32))

    genome = Genome(Path(args.fasta))

    # Group by (layer,channel) to keep hooks simple
    results: List[dict] = []

    for (layer, channel), df_ch in df.groupby(["layer", "channel"], sort=False):
        layer = str(layer)
        channel = int(channel)
        logger.info(f"Ablation: {layer} ch{channel} instances={len(df_ch)}")

        layer_to_channels = {layer: [channel]}

        with ActivationMapCapturer(wrapper.model, layer_to_channels) as cap:
            for start in range(0, len(df_ch), int(args.batch_size)):
                batch = df_ch.iloc[start : start + int(args.batch_size)]

                seqs_orig: List[str] = []
                seqs_mut: List[str] = []
                meta_rows: List[pd.Series] = []

                for _idx, r in batch.iterrows():
                    chrom = str(r["chrom"])
                    inst_start0 = int(r["start0"])
                    inst_end0 = int(r["end0"])
                    inst_center0 = (inst_start0 + inst_end0) // 2

                    if args.use_variant_center and "variant_id" in r and pd.notna(r["variant_id"]):
                        # Build allele-specific sequence centered on the variant.
                        vid = str(r["variant_id"])
                        c, pos1, ref, alt = parse_variant_id_any(vid)
                        # Use REF allele for ablation by default (more stable).
                        try:
                            ref_seq, _alt_seq = make_ref_alt_sequence(
                                genome=genome,
                                chrom=c,
                                pos1=int(pos1),
                                ref=str(ref),
                                alt=str(alt),
                                seq_len=seq_len,
                                verify_ref=True,
                            )
                        except Exception:
                            continue
                        w_start0 = window_start0(int(pos1), seq_len)
                    else:
                        # Generic: center window on instance center.
                        w_start0 = int(inst_center0) - (seq_len // 2)
                        w_end0 = w_start0 + seq_len
                        try:
                            ref_seq = genome.fetch_sequence(chrom=chrom, start0=max(0, w_start0), end0=min(w_end0, genome.chrom_length(chrom)))
                        except Exception:
                            continue
                        # pad if needed
                        pad_left = max(0, -w_start0)
                        pad_right = max(0, w_end0 - genome.chrom_length(chrom))
                        ref_seq = ("N" * pad_left) + ref_seq + ("N" * pad_right)
                        if len(ref_seq) != seq_len:
                            continue

                    # Map instance coords into sequence offsets
                    off_start = int(inst_start0 - w_start0)
                    off_end = int(inst_end0 - w_start0)
                    if off_end <= 0 or off_start >= seq_len:
                        continue
                    off_start = max(0, off_start)
                    off_end = min(seq_len, off_end)
                    if off_end <= off_start:
                        continue

                    mut_seq = _mutate_seq(ref_seq, off_start, off_end, rng=rng, mode=args.mode, sub_base=args.sub_base)

                    seqs_orig.append(ref_seq)
                    seqs_mut.append(mut_seq)
                    meta_rows.append(r)

                if not meta_rows:
                    continue

                x = one_hot_encode_batch(seqs_orig + seqs_mut, device=device)

                cap.clear()
                with torch.no_grad():
                    out = wrapper.model(x)
                scores = score_expression(out, output_key=wrapper.output_key)
                maps = cap.pop().maps[layer]  # [2B, 1, L]

                bsz = len(meta_rows)
                score_orig = scores[:bsz].detach().cpu().numpy()
                score_mut = scores[bsz:].detach().cpu().numpy()

                m_orig = maps[:bsz, 0]
                m_mut = maps[bsz:, 0]

                L = int(m_orig.shape[-1])
                bin_bp = float(bin_size_bp(seq_len, L))

                for i, r in enumerate(meta_rows):
                    chrom = str(r["chrom"])
                    inst_start0 = int(r["start0"])
                    inst_end0 = int(r["end0"])
                    inst_center0 = (inst_start0 + inst_end0) // 2

                    if args.use_variant_center and "variant_id" in r and pd.notna(r["variant_id"]):
                        _c, pos1, _ref, _alt = parse_variant_id_any(str(r["variant_id"]))
                        w_start0 = window_start0(int(pos1), seq_len)
                    else:
                        w_start0 = int(inst_center0) - (seq_len // 2)

                    off_start = max(0, int(inst_start0 - w_start0))
                    off_end = min(seq_len, int(inst_end0 - w_start0))
                    lo = int(np.floor(off_start / bin_bp))
                    hi = int(np.ceil(off_end / bin_bp))
                    lo = max(0, lo)
                    hi = min(L, hi)

                    act_o = m_orig[i]
                    act_m = m_mut[i]

                    # region summary
                    if hi > lo:
                        reg_o = float(act_o[lo:hi].mean().item())
                        reg_m = float(act_m[lo:hi].mean().item())
                        reg_delta = reg_m - reg_o
                    else:
                        reg_o = float("nan")
                        reg_m = float("nan")
                        reg_delta = float("nan")

                    results.append(
                        {
                            "layer": layer,
                            "channel": channel,
                            "chrom": chrom,
                            "start0": inst_start0,
                            "end0": inst_end0,
                            "center1": int(inst_center0 + 1),
                            "mode": args.mode,
                            "score_orig": float(score_orig[i]),
                            "score_mut": float(score_mut[i]),
                            "delta_score": float(score_mut[i] - score_orig[i]),
                            "act_mean_region_orig": reg_o,
                            "act_mean_region_mut": reg_m,
                            "delta_act_mean_region": reg_delta,
                            "bin_bp": bin_bp,
                            "bin_lo": lo,
                            "bin_hi": hi,
                            "variant_id": str(r["variant_id"]) if "variant_id" in r and pd.notna(r["variant_id"]) else None,
                        }
                    )

    out_path = Path(outdir) / "ablation_results.parquet"
    write_parquet(pd.DataFrame(results), out_path)
    logger.info(f"Wrote: {out_path} rows={len(results)}")


if __name__ == "__main__":
    main()
