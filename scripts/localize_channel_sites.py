#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from superweights.activation_maps import ActivationMapCapturer, topk_indices
from superweights.borzoi import default_device, detect_seq_len, detect_seq_len_from_crop, load_borzoi
from superweights.genome import Genome, bin_size_bp, bin_to_genome_coords, variant_center0_in_window, window_start0
from superweights.io import ensure_dir, safe_slug, write_parquet
from superweights.variants import parse_variant_id_any
from superweights_borzoi.encoding import one_hot_encode_batch
from superweights_borzoi.genome import make_ref_alt_sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Localize top delta-activation sites for shortlisted channels.")

    p.add_argument("--channels", required=True, help="Shortlist parquet (from select_channel_shortlist.py)")
    p.add_argument("--parquet", required=True, help="Benchmark parquet (e.g. Whole_Blood_pip0p9_topgene.parquet)")
    p.add_argument("--pos_ids", required=True, help="Pos ID list")
    p.add_argument("--neg_ids", required=True, help="Neg ID list")
    p.add_argument("--fasta", required=True, help="Reference FASTA")

    p.add_argument("--model_name", default="johahi/borzoi-replicate-0")
    p.add_argument("--output_key", default=None)

    p.add_argument("--seq_len", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--n_pos", type=str, default="all")
    p.add_argument("--n_neg", type=str, default="match", help="'match' = match n_pos; or integer")
    p.add_argument("--neg_match_mode", choices=["random", "chrom"], default="chrom")

    p.add_argument("--k_near", type=int, default=5)
    p.add_argument("--k_global", type=int, default=5)
    p.add_argument("--near_window_bp", type=int, default=2048)

    p.add_argument("--outdir", default="results/posmaps")

    return p.parse_args()


def _parse_n(s: str) -> int | None:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in {"", "all", "none"}:
        return None
    return int(s)


def _load_id_list(path: str | Path) -> List[str]:
    ids: List[str] = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def _make_run_table(
    benchmark: pd.DataFrame,
    pos_ids_path: str,
    neg_ids_path: str,
    n_pos: int | None,
    n_neg: int | None,
    neg_match_mode: str,
    seed: int,
) -> pd.DataFrame:
    """Return df with columns: variant_id,label,chrom,pos,ref,alt."""
    rng = np.random.default_rng(seed)

    # Benchmark parquet has 'variant' like chr1_..._b38, and also chrom/pos1/ref/alt.
    # Create canonical variant_id.
    bench = benchmark.copy()

    # Prefer existing variant_id if already present.
    if "variant_id" not in bench.columns:
        if {"chrom", "pos1", "ref", "alt"}.issubset(set(bench.columns)):
            # canonical: chrom:pos:ref:alt
            chrom = bench["chrom"].astype(str).str.replace("^chr", "", regex=True)
            bench["variant_id"] = chrom + ":" + bench["pos1"].astype(int).astype(str) + ":" + bench["ref"].astype(str) + ":" + bench["alt"].astype(str)
        elif "variant" in bench.columns:
            bench["variant_id"] = bench["variant"].astype(str)
        else:
            raise ValueError("Benchmark parquet missing variant_id (need chrom/pos1/ref/alt or variant)")

    # Canonicalize to chrom:pos:ref:alt using our parser
    bench["variant_id"] = bench["variant_id"].astype(str).map(lambda s: ":".join(map(str, parse_variant_id_any(s))))

    pos_raw = _load_id_list(pos_ids_path)
    neg_raw = _load_id_list(neg_ids_path)
    pos_ids = np.array(sorted({":".join(map(str, parse_variant_id_any(s))) for s in pos_raw}), dtype=object)
    neg_ids = np.array(sorted({":".join(map(str, parse_variant_id_any(s))) for s in neg_raw}), dtype=object)

    # Pos are those in benchmark AND in pos list
    bench_set = set(bench["variant_id"].tolist())
    pos_in_bench = np.array([vid for vid in pos_ids if vid in bench_set], dtype=object)

    if n_pos is not None and n_pos < len(pos_in_bench):
        pos_in_bench = rng.choice(pos_in_bench, size=n_pos, replace=False)

    # Neg: sample from neg list excluding benchmark pos and excluding any pos list overlap.
    bad = set(pos_ids.tolist())
    neg_pool = np.array([vid for vid in neg_ids if vid not in bad], dtype=object)
    if len(neg_pool) == 0:
        raise ValueError("No negative variants available after filtering")

    if n_neg is None:
        # match
        n_neg = len(pos_in_bench)

    if neg_match_mode == "random":
        neg_sel = rng.choice(neg_pool, size=min(n_neg, len(neg_pool)), replace=False)
    else:
        # chrom-matched sampling
        pos_chroms = [parse_variant_id_any(v)[0] for v in pos_in_bench]
        need_by_chrom: Dict[str, int] = defaultdict(int)
        for c in pos_chroms:
            need_by_chrom[c] += 1

        # bucket neg pool by chrom
        neg_by_chrom: Dict[str, List[str]] = defaultdict(list)
        for vid in neg_pool:
            c, _, _, _ = parse_variant_id_any(vid)
            neg_by_chrom[c].append(vid)

        neg_sel_list: List[str] = []
        for chrom, need in need_by_chrom.items():
            pool = neg_by_chrom.get(chrom, [])
            if not pool:
                continue
            take = min(need, len(pool))
            picks = rng.choice(np.array(pool, dtype=object), size=take, replace=False)
            neg_sel_list.extend(picks.tolist())

        # top off if insufficient
        if len(neg_sel_list) < n_neg:
            remaining = n_neg - len(neg_sel_list)
            already = set(neg_sel_list)
            pool2 = np.array([v for v in neg_pool if v not in already], dtype=object)
            if len(pool2):
                extra = rng.choice(pool2, size=min(remaining, len(pool2)), replace=False)
                neg_sel_list.extend(extra.tolist())

        neg_sel = np.array(neg_sel_list[:n_neg], dtype=object)

    df_pos = pd.DataFrame({"variant_id": pos_in_bench, "label": 1})
    df_neg = pd.DataFrame({"variant_id": neg_sel, "label": 0})

    df = pd.concat([df_pos, df_neg], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Parse fields needed to fetch sequences
    chroms, poss, refs, alts = [], [], [], []
    for vid in df["variant_id"].tolist():
        c, p1, r, a = parse_variant_id_any(vid)
        chroms.append(c)
        poss.append(p1)
        refs.append(r)
        alts.append(a)
    df["chrom"] = chroms
    df["pos"] = poss
    df["ref"] = refs
    df["alt"] = alts

    return df


def _layer_to_channels(shortlist: pd.DataFrame) -> Dict[str, List[int]]:
    d: Dict[str, List[int]] = defaultdict(list)
    for _, row in shortlist.iterrows():
        d[str(row["layer"])].append(int(row["channel"]))
    # stable order
    return {layer: sorted(set(chs)) for layer, chs in d.items()}


def _topk_sites_for_one(
    delta: torch.Tensor,
    ref: torch.Tensor,
    alt: torch.Tensor,
    k: int,
    lo: int,
    hi: int,
) -> List[Tuple[int, float, float, float]]:
    """Return list of (bin_idx, delta, ref, alt) within [lo,hi)."""
    if hi <= lo or k <= 0:
        return []
    d = delta[lo:hi]
    idx = topk_indices(d.abs(), k=k)
    out: List[Tuple[int, float, float, float]] = []
    for j in idx.tolist():
        b = int(lo + j)
        out.append((b, float(delta[b].item()), float(ref[b].item()), float(alt[b].item())))
    return out


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = ensure_dir(args.outdir)

    device = default_device()
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")

    shortlist = pd.read_parquet(args.channels)
    if not {"layer", "channel"}.issubset(set(shortlist.columns)):
        raise ValueError("Shortlist parquet must include columns: layer, channel")

    bench = pd.read_parquet(args.parquet)

    n_pos = _parse_n(args.n_pos)
    n_neg: int | None
    if str(args.n_neg).strip().lower() == "match":
        n_neg = None
    else:
        n_neg = int(args.n_neg)

    df_run = _make_run_table(
        benchmark=bench,
        pos_ids_path=args.pos_ids,
        neg_ids_path=args.neg_ids,
        n_pos=n_pos,
        n_neg=n_neg,
        neg_match_mode=args.neg_match_mode,
        seed=args.seed,
    )
    logger.info(f"Variants: n={len(df_run)} labels={df_run['label'].value_counts().to_dict()}")

    wrapper = load_borzoi(args.model_name, device=device, output_key=args.output_key)

    seq_len = args.seq_len
    if seq_len is None:
        seq_len = detect_seq_len_from_crop(wrapper.model) or detect_seq_len(wrapper.model) or 262144
        logger.info(f"Auto seq_len={seq_len}")
    else:
        logger.info(f"Using seq_len={seq_len}")

    # acceptance: dummy forward
    dummy = torch.zeros((1, 4, int(seq_len)), device=device, dtype=torch.float32)
    with torch.no_grad():
        _ = wrapper.model(dummy)

    genome = Genome(Path(args.fasta))

    layer_to_chs = _layer_to_channels(shortlist)
    logger.info("Layers: " + ", ".join(f"{k} (n_ch={len(v)})" for k, v in layer_to_chs.items()))

    # Accumulate per (layer,channel)
    rows_by_key: Dict[Tuple[str, int], List[dict]] = defaultdict(list)

    skipped_indel = 0
    skipped_ref_mismatch = 0
    skipped_fasta_missing = 0

    with ActivationMapCapturer(wrapper.model, layer_to_chs) as cap:
        for start in tqdm(range(0, len(df_run), args.batch_size), desc="batches"):
            batch = df_run.iloc[start : start + args.batch_size]

            ref_seqs: List[str] = []
            alt_seqs: List[str] = []
            keep_rows: List[pd.Series] = []

            for _idx, row in batch.iterrows():
                ref = str(row["ref"])
                alt = str(row["alt"])
                if len(ref) != 1 or len(alt) != 1:
                    skipped_indel += 1
                    continue

                try:
                    ref_seq, alt_seq = make_ref_alt_sequence(
                        genome=genome,
                        chrom=str(row["chrom"]),
                        pos1=int(row["pos"]),
                        ref=ref,
                        alt=alt,
                        seq_len=int(seq_len),
                        verify_ref=True,
                    )
                except KeyError:
                    skipped_fasta_missing += 1
                    continue
                except ValueError as e:
                    if "Reference mismatch" in str(e):
                        skipped_ref_mismatch += 1
                        continue
                    raise

                ref_seqs.append(ref_seq)
                alt_seqs.append(alt_seq)
                keep_rows.append(row)

            if not keep_rows:
                continue

            # 2B batch: ref then alt
            x = one_hot_encode_batch(ref_seqs + alt_seqs, device=device)

            cap.clear()
            with torch.no_grad():
                y = wrapper.model(x)

            # scores not strictly required, but sometimes useful to keep for downstream
            # (we may add them later; keep local computation cheap)
            maps = cap.pop().maps  # layer -> [2B, n_ch, L]

            bsz = len(keep_rows)
            for layer, t in maps.items():
                if t.shape[0] != 2 * bsz:
                    raise RuntimeError(f"Unexpected map batch size for {layer}: {t.shape}")

                t_ref = t[:bsz]
                t_alt = t[bsz:]
                delta = t_alt - t_ref  # [B, n_ch, L]

                L = int(delta.shape[-1])
                bin_bp = bin_size_bp(int(seq_len), L)

                for bi, row in enumerate(keep_rows):
                    w_start0 = window_start0(int(row["pos"]), int(seq_len))
                    center0 = int(row["pos"]) - 1
                    # Convert near window bp -> bins
                    center_off = variant_center0_in_window(int(row["pos"]), int(seq_len))
                    center_bin = int(np.floor(center_off / bin_bp))
                    rad_bins = int(np.ceil(float(args.near_window_bp) / float(bin_bp)))
                    lo = max(0, center_bin - rad_bins)
                    hi = min(L, center_bin + rad_bins + 1)

                    for local_ch_idx, channel in enumerate(layer_to_chs[layer]):
                        d1 = delta[bi, local_ch_idx]
                        r1 = t_ref[bi, local_ch_idx]
                        a1 = t_alt[bi, local_ch_idx]

                        # near-variant topK
                        near_sites = _topk_sites_for_one(d1, r1, a1, k=int(args.k_near), lo=lo, hi=hi)
                        for bin_idx, dval, rval, aval in near_sites:
                            s0, e0, c1 = bin_to_genome_coords(w_start0, bin_idx, bin_bp)
                            rows_by_key[(layer, int(channel))].append(
                                {
                                    "variant_id": str(row["variant_id"]),
                                    "label": int(row["label"]),
                                    "chrom": str(row["chrom"]),
                                    "pos": int(row["pos"]),
                                    "layer": str(layer),
                                    "channel": int(channel),
                                    "bin_index": int(bin_idx),
                                    "bin_bp": float(bin_bp),
                                    "delta_act": float(dval),
                                    "act_ref": float(rval),
                                    "act_alt": float(aval),
                                    "window_type": "near_variant",
                                    "site_start0": int(s0),
                                    "site_end0": int(e0),
                                    "site_center1": int(c1),
                                    "window_start0": int(w_start0),
                                    "window_center0": int(center0),
                                }
                            )

                        # global topK
                        if int(args.k_global) > 0:
                            global_sites = _topk_sites_for_one(d1, r1, a1, k=int(args.k_global), lo=0, hi=L)
                            for bin_idx, dval, rval, aval in global_sites:
                                s0, e0, c1 = bin_to_genome_coords(w_start0, bin_idx, bin_bp)
                                rows_by_key[(layer, int(channel))].append(
                                    {
                                        "variant_id": str(row["variant_id"]),
                                        "label": int(row["label"]),
                                        "chrom": str(row["chrom"]),
                                        "pos": int(row["pos"]),
                                        "layer": str(layer),
                                        "channel": int(channel),
                                        "bin_index": int(bin_idx),
                                        "bin_bp": float(bin_bp),
                                        "delta_act": float(dval),
                                        "act_ref": float(rval),
                                        "act_alt": float(aval),
                                        "window_type": "global",
                                        "site_start0": int(s0),
                                        "site_end0": int(e0),
                                        "site_center1": int(c1),
                                        "window_start0": int(w_start0),
                                        "window_center0": int(center0),
                                    }
                                )

    logger.info(
        f"Done. skipped_indel={skipped_indel} skipped_ref_mismatch={skipped_ref_mismatch} skipped_fasta_missing={skipped_fasta_missing}"
    )

    # write per-channel outputs
    for (layer, channel), rows in rows_by_key.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        fname = f"{safe_slug(layer)}_{int(channel)}.parquet"
        out_path = outdir / fname
        write_parquet(df, out_path)

    logger.info(f"Wrote posmaps to: {outdir} (files={len(rows_by_key)})")


if __name__ == "__main__":
    main()
