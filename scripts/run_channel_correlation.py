from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from borzoi.stats import compute_channel_correlations
from borzoi.data import (
    ensure_variant_id,
    load_eqtl_parquet,
    load_posneg,
    parse_canonical_variant_id,
    sample_posneg_ids,
)
from borzoi.encode import one_hot_encode_batch
from borzoi.genome import Genome, make_ref_alt_sequence
from borzoi.activations import ActivationCapturer
from borzoi.model import detect_seq_len, detect_seq_len_from_crop, forward_score, load_borzoi


DEFAULT_LAYERS = [
    "conv_dna.conv_layer",
    "res_tower.0.conv_layer",
    "res_tower.2.conv_layer",
    "horizontal_conv0.conv_layer",
    "horizontal_conv1.conv_layer",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Borzoi channel correlation on GTEx Whole Blood eQTLs")

    p.add_argument("--parquet", required=True, type=str)
    p.add_argument("--pos_ids", required=True, type=str)
    p.add_argument("--neg_ids", required=True, type=str)
    p.add_argument("--fasta", required=True, type=str)

    p.add_argument("--seq_len", type=int, default=None, help="Override sequence length")
    p.add_argument(
        "--window_bp",
        type=int,
        default=None,
        help="Alias for --seq_len (window length). If both set, they must match.",
    )

    p.add_argument("--n_pos", type=str, default="all")
    p.add_argument("--n_neg", type=str, default="all")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--model_name", type=str, default="johahi/borzoi-replicate-0")
    p.add_argument("--layers", type=str, default=",".join(DEFAULT_LAYERS))
    p.add_argument("--output_key", type=str, default=None)

    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument(
        "--seq_cache",
        type=str,
        default=None,
        help="Optional parquet cache (variant_id, ref_seq, alt_seq)",
    )

    return p.parse_args()


def _parse_n(s: str) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in {"all", "none", ""}:
        return None
    return int(s)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device available: cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    split = load_posneg(args.pos_ids, args.neg_ids)
    df_parquet = load_eqtl_parquet(args.parquet)
    df_parquet = ensure_variant_id(df_parquet)

    parquet_ids = set(df_parquet["variant_id"].astype(str).tolist())
    overlap_stats = {
        "n_total_parquet": int(len(df_parquet)),
        "n_pos_list": int(len(split.pos_ids)),
        "n_neg_list": int(len(split.neg_ids)),
        "n_pos_in_parquet": int(len(split.pos_ids & parquet_ids)),
        "n_neg_in_parquet": int(len(split.neg_ids & parquet_ids)),
        "n_posneg_overlap": int(len(split.pos_ids & split.neg_ids)),
    }
    logger.info("Overlap counts: " + json.dumps(overlap_stats, indent=2, sort_keys=True).replace("\n", " "))

    n_pos = _parse_n(args.n_pos)
    n_neg = _parse_n(args.n_neg)

    df_run = sample_posneg_ids(split.pos_ids, split.neg_ids, n_pos=n_pos, n_neg=n_neg, seed=args.seed)

    # Parse variant_id into fields for FASTA + model scoring
    chroms, poss, refs, alts = [], [], [], []
    for vid in df_run["variant_id"].tolist():
        c, p1, r, a = parse_canonical_variant_id(vid)
        chroms.append(c)
        poss.append(p1)
        refs.append(r)
        alts.append(a)
    df_run["chrom"] = chroms
    df_run["pos"] = poss
    df_run["ref"] = refs
    df_run["alt"] = alts

    # Bring in metadata (beta/maf/gene_id/etc.) when available
    df_run = df_run.merge(df_parquet.drop_duplicates("variant_id"), on="variant_id", how="left", suffixes=("", "_parquet"))
    logger.info(f"Running variants (pos+neg): n={len(df_run)}")

    wrapper = load_borzoi(args.model_name, device=device, output_key=args.output_key)

    if args.seq_len is not None and args.window_bp is not None and int(args.seq_len) != int(args.window_bp):
        raise ValueError("--seq_len and --window_bp must match if both are set")

    seq_len = args.seq_len if args.seq_len is not None else args.window_bp
    if seq_len is None:
        seq_len = detect_seq_len_from_crop(wrapper.model) or detect_seq_len(wrapper.model)
        if seq_len is None:
            # Known good default for johahi/borzoi-replicate-0
            seq_len = 262144
            logger.warning("Could not detect seq_len; defaulting to 262144")
    logger.info(f"Using seq_len={seq_len}")

    # acceptance: dummy forward
    dummy = torch.zeros((1, 4, seq_len), device=device, dtype=torch.float32)
    with torch.no_grad():
        _ = wrapper.model(dummy)
    logger.info("Dummy forward succeeded")

    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    genome = Genome(Path(args.fasta))

    # optional sequence cache
    cache_path = Path(args.seq_cache) if args.seq_cache else None
    cache_df = None
    if cache_path and cache_path.exists():
        logger.info(f"Loading sequence cache: {cache_path}")
        cache_df = pd.read_parquet(cache_path)
        cache_df = cache_df[["variant_id", "ref_seq", "alt_seq"]].drop_duplicates("variant_id")

    if cache_df is not None:
        df_run = df_run.merge(cache_df, on="variant_id", how="left")
    else:
        df_run["ref_seq"] = pd.NA
        df_run["alt_seq"] = pd.NA

    cache_rows: List[dict] = []

    skipped_indel = 0
    skipped_ref_mismatch = 0
    skipped_fasta_missing = 0
    padded_left = 0
    padded_right = 0

    per_variant_rows: List[dict] = []

    feature_accum: Dict[Tuple[str, str], List[np.ndarray]] = {}

    def append_features(layer: str, feature: str, arr: np.ndarray) -> None:
        key = (layer, feature)
        if key not in feature_accum:
            feature_accum[key] = []
        feature_accum[key].append(arr)

    with ActivationCapturer(wrapper.model, layers) as cap:
        for start in tqdm(range(0, len(df_run), args.batch_size), desc="batches"):
            batch = df_run.iloc[start : start + args.batch_size]

            ref_seqs: List[str] = []
            alt_seqs: List[str] = []
            keep_rows: List[pd.Series] = []

            for _idx, row in batch.iterrows():
                ref = str(row.get("ref"))
                alt = str(row.get("alt"))
                if len(ref) != 1 or len(alt) != 1:
                    skipped_indel += 1
                    continue

                if pd.notna(row.get("ref_seq")) and pd.notna(row.get("alt_seq")):
                    ref_seq = str(row["ref_seq"])
                    alt_seq = str(row["alt_seq"])
                else:
                    try:
                        # track whether window hits chromosome ends
                        center0 = int(row.get("pos")) - 1
                        half = int(seq_len) // 2
                        start0 = center0 - half
                        end0 = start0 + int(seq_len)
                        try:
                            chrom_len = genome.chrom_length(str(row.get("chrom")))
                            if start0 < 0:
                                padded_left += 1
                            if end0 > chrom_len:
                                padded_right += 1
                        except KeyError:
                            pass

                        ref_seq, alt_seq = make_ref_alt_sequence(
                            genome=genome,
                            chrom=str(row.get("chrom")),
                            pos1=int(row.get("pos")),
                            ref=ref,
                            alt=alt,
                            seq_len=seq_len,
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

                if len(ref_seq) != seq_len or len(alt_seq) != seq_len:
                    raise ValueError("Sequence length mismatch")

                ref_seqs.append(ref_seq)
                alt_seqs.append(alt_seq)
                keep_rows.append(row)

                if cache_path is not None:
                    cache_rows.append({"variant_id": row["variant_id"], "ref_seq": ref_seq, "alt_seq": alt_seq})

            if not keep_rows:
                continue

            x_ref = one_hot_encode_batch(ref_seqs, device=device)
            cap.clear()
            score_ref = forward_score(wrapper, x_ref)
            ref_summ = cap.pop().by_layer

            x_alt = one_hot_encode_batch(alt_seqs, device=device)
            cap.clear()
            score_alt = forward_score(wrapper, x_alt)
            alt_summ = cap.pop().by_layer

            delta_expr = (score_alt - score_ref).detach().to("cpu").numpy().astype(np.float32)

            # activation delta features (alt - ref)
            for layer in layers:
                if layer not in ref_summ or layer not in alt_summ:
                    continue

                for feat in ("mean", "max", "l2"):
                    r = ref_summ[layer][feat]
                    a = alt_summ[layer][feat]
                    d = (a - r).astype(np.float32)
                    append_features(layer, f"delta_{feat}", d)

                    if feat in {"max", "l2"}:
                        append_features(layer, f"abs_delta_{feat}", np.abs(d).astype(np.float32))

            for i, row in enumerate(keep_rows):
                out_row = {
                    "variant_id": row["variant_id"],
                    "label": int(row["label"]),
                    "chrom": row.get("chrom"),
                    "pos": int(row.get("pos")),
                    "ref": row.get("ref"),
                    "alt": row.get("alt"),
                    "maf": row.get("maf", np.nan),
                    "beta": row.get("beta", np.nan),
                    "posterior_beta": row.get("posterior_beta", row.get("post_beta", np.nan)),
                    "delta_expr": float(delta_expr[i]),
                }

                # keep some gene fields if present
                for c in ("gene_id", "gene_name", "top_gene", "top_gene_id"):
                    if c in row.index:
                        out_row[c] = row.get(c)

                per_variant_rows.append(out_row)

    per_variant = pd.DataFrame(per_variant_rows)
    per_variant_out = outdir / "per_variant_scores.parquet"
    per_variant.to_parquet(per_variant_out, index=False)

    # save cache if requested
    if cache_path is not None:
        new_cache = pd.DataFrame(cache_rows).drop_duplicates("variant_id")
        if cache_df is not None:
            merged = pd.concat([cache_df, new_cache], axis=0).drop_duplicates("variant_id", keep="first")
        else:
            merged = new_cache
        merged.to_parquet(cache_path, index=False)
        logger.info(f"Wrote sequence cache: {cache_path} (n={len(merged)})")

    # stack features
    feature_mats: Dict[Tuple[str, str], np.ndarray] = {}
    for key, parts in feature_accum.items():
        feature_mats[key] = np.concatenate(parts, axis=0)

    corr_df = compute_channel_correlations(per_variant, feature_mats)
    corr_out = outdir / "channel_correlations.parquet"
    corr_df.to_parquet(corr_out, index=False)

    report = {
        "n_requested": int(len(df_run)),
        "n_scored": int(len(per_variant)),
        "skipped_indel": int(skipped_indel),
        "skipped_ref_mismatch": int(skipped_ref_mismatch),
        "skipped_fasta_missing": int(skipped_fasta_missing),
        "padded_left": int(padded_left),
        "padded_right": int(padded_right),
        "seq_len": int(seq_len),
        "layers": layers,
        "device": str(device),
        "overlap": overlap_stats,
        "outputs": {
            "per_variant_scores": str(per_variant_out),
            "channel_correlations": str(corr_out),
        },
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2))

    logger.info(f"Wrote: {per_variant_out}")
    logger.info(f"Wrote: {corr_out}")
    logger.info(f"Wrote: {outdir / 'report.json'}")


if __name__ == "__main__":
    main()
