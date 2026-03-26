#!/usr/bin/env python
from __future__ import annotations

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running as `python scripts/ablate_channels.py` without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Some HPC shells may not set $HOME; Python/Transformers may fall back to NSS/LDAP lookups,
# which can hang. Prefer a local writable directory.
if not os.environ.get("HOME"):
    os.environ["HOME"] = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or str(REPO_ROOT)

# Avoid transformers importing TensorFlow/JAX/Flax if present in the environment.
# This can be very slow or hang on some HPC nodes due to GPU plugin initialization.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from borzoi.io import write_parquet
from borzoi.data import ensure_variant_id, load_eqtl_parquet, load_posneg, parse_canonical_variant_id, sample_posneg_ids
from borzoi.encode import one_hot_encode_batch
from borzoi.genome import Genome, make_ref_alt_sequence
from borzoi.ablation import ChannelAblator
from borzoi.model import detect_seq_len, detect_seq_len_from_crop, forward_score, load_borzoi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Channel ablation (activation zero-out) on pos/neg eQTL benchmark")

    p.add_argument("--shortlist", required=True, type=str, help="Parquet with columns: layer, channel")

    p.add_argument("--parquet", required=True, type=str)
    p.add_argument("--pos_ids", required=True, type=str)
    p.add_argument("--neg_ids", required=True, type=str)
    p.add_argument("--fasta", required=True, type=str)

    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--window_bp", type=int, default=None)

    p.add_argument("--n_pos", type=str, default="all")
    p.add_argument("--n_neg", type=str, default="all")
    p.add_argument("--batch_size", type=int, default=4)

    p.add_argument("--model_name", type=str, default="johahi/borzoi-replicate-0")
    p.add_argument("--output_key", type=str, default=None)

    p.add_argument("--seq_cache", type=str, default=None, help="Optional parquet cache (variant_id, ref_seq, alt_seq)")

    p.add_argument(
        "--warmup",
        action="store_true",
        help="Run a single dummy forward pass after loading the model (can reduce first-batch latency but costs time).",
    )

    p.add_argument("--max_channels", type=int, default=None, help="Optionally only evaluate first N channels from shortlist")
    p.add_argument("--out", type=str, default="results/ablation/channel_ablation.parquet")
    p.add_argument("--report_json", type=str, default="results/ablation/report.json")

    p.add_argument("--seed", type=int, default=1337)

    return p.parse_args()


def _parse_n(s: str) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in {"all", "none", ""}:
        return None
    return int(s)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if len(x) < 2:
        return float("nan")
    r = spearmanr(x, y, nan_policy="omit")
    return float(r.correlation)


def _build_run_table(df_parquet: pd.DataFrame, pos_ids: str, neg_ids: str, n_pos: Optional[int], n_neg: Optional[int], seed: int) -> Tuple[pd.DataFrame, dict]:
    split = load_posneg(pos_ids, neg_ids)
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

    df_run = sample_posneg_ids(split.pos_ids, split.neg_ids, n_pos=n_pos, n_neg=n_neg, seed=seed)

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

    df_run = df_run.merge(df_parquet.drop_duplicates("variant_id"), on="variant_id", how="left", suffixes=("", "_parquet"))
    return df_run, overlap_stats


def _iter_batches(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


@torch.no_grad()
def _compute_delta_expr(
    df_run: pd.DataFrame,
    wrapper,
    genome: Genome,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    ablate: Optional[Tuple[str, int]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Compute delta_expr = score_alt - score_ref for all rows.

    If ablate is provided, it should be (layer_name, channel_id) and will be zeroed at that layer.

    Returns (per_variant_df, stats).
    """

    skipped_indel = 0
    skipped_ref_mismatch = 0
    skipped_fasta_missing = 0
    padded_left = 0
    padded_right = 0

    per_variant_rows: List[dict] = []

    ablator = None
    if ablate is not None:
        layer, ch = ablate
        ablator = ChannelAblator(wrapper.model, layer_name=str(layer), channels=[int(ch)])
        ablator.register()

    try:
        for batch in tqdm(_iter_batches(df_run, batch_size), desc="batches", leave=False):
            ref_seqs: List[str] = []
            alt_seqs: List[str] = []
            keep_rows: List[pd.Series] = []

            for _idx, row in batch.iterrows():
                ref = str(row.get("ref"))
                alt = str(row.get("alt"))
                if len(ref) != 1 or len(alt) != 1:
                    skipped_indel += 1
                    continue

                ref_seq = row.get("ref_seq", None)
                alt_seq = row.get("alt_seq", None)
                if pd.notna(ref_seq) and pd.notna(alt_seq):
                    ref_seq = str(ref_seq)
                    alt_seq = str(alt_seq)
                else:
                    try:
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

                ref_seqs.append(ref_seq)
                alt_seqs.append(alt_seq)
                keep_rows.append(row)

            if not keep_rows:
                continue

            x_ref = one_hot_encode_batch(ref_seqs, device=device)
            score_ref = forward_score(wrapper, x_ref)

            x_alt = one_hot_encode_batch(alt_seqs, device=device)
            score_alt = forward_score(wrapper, x_alt)

            delta_expr = (score_alt - score_ref).detach().to("cpu").numpy().astype(np.float32)

            for i, row in enumerate(keep_rows):
                per_variant_rows.append(
                    {
                        "variant_id": row["variant_id"],
                        "label": int(row["label"]),
                        "beta": row.get("beta", np.nan),
                        "delta_expr": float(delta_expr[i]),
                    }
                )

    finally:
        if ablator is not None:
            ablator.close()

    per_variant = pd.DataFrame(per_variant_rows)

    n_pos_scored = int((per_variant["label"] == 1).sum()) if not per_variant.empty and "label" in per_variant.columns else 0
    n_neg_scored = int((per_variant["label"] == 0).sum()) if not per_variant.empty and "label" in per_variant.columns else 0

    stats = {
        "n_requested": int(len(df_run)),
        "n_scored": int(len(per_variant)),
        "n_pos_scored": n_pos_scored,
        "n_neg_scored": n_neg_scored,
        "skipped_indel": int(skipped_indel),
        "skipped_ref_mismatch": int(skipped_ref_mismatch),
        "skipped_fasta_missing": int(skipped_fasta_missing),
        "padded_left": int(padded_left),
        "padded_right": int(padded_right),
    }

    return per_variant, stats


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device available: cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    shortlist = pd.read_parquet(Path(args.shortlist))
    if "layer" not in shortlist.columns or "channel" not in shortlist.columns:
        raise ValueError("Shortlist must contain columns: layer, channel")
    shortlist = shortlist[["layer", "channel"]].drop_duplicates().reset_index(drop=True)
    if args.max_channels is not None:
        shortlist = shortlist.head(int(args.max_channels)).copy()

    n_pos = _parse_n(args.n_pos)
    n_neg = _parse_n(args.n_neg)

    df_parquet = load_eqtl_parquet(args.parquet)
    df_run, overlap_stats = _build_run_table(df_parquet, args.pos_ids, args.neg_ids, n_pos=n_pos, n_neg=n_neg, seed=args.seed)
    logger.info(f"Running variants (pos+neg): n={len(df_run)}")

    # optional sequence cache (saves repeated FASTA fetch)
    cache_path = Path(args.seq_cache) if args.seq_cache else None
    if cache_path and cache_path.exists():
        logger.info(f"Loading sequence cache: {cache_path}")
        cache_df = pd.read_parquet(cache_path)
        cache_df = cache_df[["variant_id", "ref_seq", "alt_seq"]].drop_duplicates("variant_id")
        df_run = df_run.merge(cache_df, on="variant_id", how="left")
    else:
        df_run["ref_seq"] = pd.NA
        df_run["alt_seq"] = pd.NA

    wrapper = load_borzoi(args.model_name, device=device, output_key=args.output_key)

    if args.seq_len is not None and args.window_bp is not None and int(args.seq_len) != int(args.window_bp):
        raise ValueError("--seq_len and --window_bp must match if both are set")

    seq_len = args.seq_len if args.seq_len is not None else args.window_bp
    if seq_len is None:
        seq_len = detect_seq_len_from_crop(wrapper.model) or detect_seq_len(wrapper.model)
        if seq_len is None:
            seq_len = 262144
            logger.warning("Could not detect seq_len; defaulting to 262144")
    logger.info(f"Using seq_len={seq_len}")

    if args.warmup:
        logger.info("Warmup forward (dummy sequence)")
        dummy = torch.zeros((1, 4, int(seq_len)), device=device, dtype=torch.float32)
        with torch.no_grad():
            _ = wrapper.model(dummy)

    genome = Genome(Path(args.fasta))

    # baseline
    logger.info("Computing baseline delta_expr (no ablation)")
    base_df, base_stats = _compute_delta_expr(
        df_run=df_run,
        wrapper=wrapper,
        genome=genome,
        seq_len=int(seq_len),
        batch_size=int(args.batch_size),
        device=device,
        ablate=None,
    )

    y = base_df["label"].astype(int).to_numpy()
    base_abs = np.abs(base_df["delta_expr"].to_numpy(dtype=np.float32))
    baseline_auc = float(roc_auc_score(y, base_abs)) if len(np.unique(y)) == 2 else float("nan")

    pos_mask = base_df["label"].to_numpy(dtype=int) == 1
    beta = base_df.loc[pos_mask, "beta"].to_numpy(dtype=np.float32)
    delta = base_df.loc[pos_mask, "delta_expr"].to_numpy(dtype=np.float32)
    baseline_spearman_beta = _spearman(delta, beta)

    rows: List[dict] = []

    for i, r in enumerate(shortlist.itertuples(index=False), start=1):
        layer = str(r.layer)
        ch = int(r.channel)
        logger.info(f"[{i}/{len(shortlist)}] Ablating {layer} channel {ch}")

        ab_df, ab_stats = _compute_delta_expr(
            df_run=df_run,
            wrapper=wrapper,
            genome=genome,
            seq_len=int(seq_len),
            batch_size=int(args.batch_size),
            device=device,
            ablate=(layer, ch),
        )

        if len(ab_df) != len(base_df):
            logger.warning(f"Ablated run scored {len(ab_df)} vs baseline {len(base_df)}; joining on variant_id")

        merged = base_df.merge(ab_df[["variant_id", "delta_expr"]].rename(columns={"delta_expr": "delta_expr_ab"}), on="variant_id", how="inner")

        y_m = merged["label"].astype(int).to_numpy()
        abs_ab = np.abs(merged["delta_expr_ab"].to_numpy(dtype=np.float32))
        auc_ab = float(roc_auc_score(y_m, abs_ab)) if len(np.unique(y_m)) == 2 else float("nan")

        pos_m = merged["label"].to_numpy(dtype=int) == 1
        beta_m = merged.loc[pos_m, "beta"].to_numpy(dtype=np.float32)
        delta_ab_m = merged.loc[pos_m, "delta_expr_ab"].to_numpy(dtype=np.float32)
        spearman_beta_ab = _spearman(delta_ab_m, beta_m)

        abs_base_m = np.abs(merged["delta_expr"].to_numpy(dtype=np.float32))
        delta_drop = float(np.mean(abs_base_m - abs_ab))

        rows.append(
            {
                "layer": layer,
                "channel": ch,
                "n_scored": int(len(merged)),
                "baseline_auc_label": baseline_auc,
                "ablated_auc_label": auc_ab,
                "delta_auc": float(baseline_auc - auc_ab),
                "baseline_spearman_beta_pos": baseline_spearman_beta,
                "ablated_spearman_beta_pos": spearman_beta_ab,
                "delta_spearman_beta": float(baseline_spearman_beta - spearman_beta_ab),
                "mean_abs_delta_drop": delta_drop,
                "baseline_stats": json.dumps(base_stats, sort_keys=True),
                "ablated_stats": json.dumps(ab_stats, sort_keys=True),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["delta_auc", "mean_abs_delta_drop"], ascending=[False, False]).reset_index(drop=True)
    write_parquet(out_df, out_path)
    logger.info(f"Wrote: {out_path} (rows={len(out_df)})")

    report = {
        "seq_len": int(seq_len),
        "batch_size": int(args.batch_size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "baseline": {
            "auc_label": baseline_auc,
            "spearman_beta_pos": baseline_spearman_beta,
            "stats": base_stats,
        },
        "overlap": overlap_stats,
        "shortlist": {
            "path": str(args.shortlist),
            "n_channels": int(len(shortlist)),
        },
        "outputs": {
            "ablation_parquet": str(out_path),
        },
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
