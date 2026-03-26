#!/usr/bin/env python
from __future__ import annotations

import sys
import os
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Allow running as `python scripts/...py` without installing the package.
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

from superweights.io import write_parquet
from superweights_borzoi.data import (
    ensure_variant_id,
    load_eqtl_parquet,
    load_posneg,
    parse_canonical_variant_id,
    sample_posneg_ids,
)
from superweights_borzoi.encoding import one_hot_encode_batch
from superweights_borzoi.genome import Genome, make_ref_alt_sequence
from superweights_borzoi.interpret.ablation import ChannelAblator
from superweights_borzoi.models.borzoi_pt import (
    detect_seq_len,
    detect_seq_len_from_crop,
    forward_score,
    get_module_by_name,
    load_borzoi,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Null + bootstrap analysis for channel ablation. "
            "(1) build a matched-layer random-channel null distribution; "
            "(2) bootstrap variants to get CIs for ΔAUC and ΔSpearman(beta)."
        )
    )

    p.add_argument("--shortlist", required=True, type=str, help="Parquet with columns: layer, channel")

    p.add_argument("--parquet", required=True, type=str)
    p.add_argument("--pos_ids", required=True, type=str)
    p.add_argument("--neg_ids", required=True, type=str)
    p.add_argument("--fasta", required=True, type=str)

    p.add_argument("--seq_cache", type=str, default=None, help="Optional parquet cache (variant_id, ref_seq, alt_seq)")

    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--window_bp", type=int, default=None)

    p.add_argument("--n_pos", type=str, default="500")
    p.add_argument("--n_neg", type=str, default="500")
    p.add_argument("--batch_size", type=int, default=2)

    p.add_argument("--model_name", type=str, default="johahi/borzoi-replicate-0")
    p.add_argument("--output_key", type=str, default=None)

    p.add_argument("--n_null", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=500)

    p.add_argument(
        "--only_null",
        action="store_true",
        help=(
            "Only run the matched-layer null ablations and write --out_null/--report_json. "
            "Skips shortlist bootstrapping (useful when you already have shortlist point estimates)."
        ),
    )

    p.add_argument("--seed", type=int, default=1337)

    p.add_argument(
        "--out_bootstrap",
        type=str,
        default="results/ablation/ablate_top20_500_500.bootstrap.parquet",
        help="Output parquet for shortlist channels with bootstrap CIs + null p-values",
    )
    p.add_argument(
        "--out_null",
        type=str,
        default="results/ablation/null_random200_500_500.parquet",
        help="Output parquet for matched-layer random-channel null ablations",
    )
    p.add_argument(
        "--report_json",
        type=str,
        default="results/ablation/null_and_bootstrap.report.json",
        help="Write a JSON report with run parameters and summary stats",
    )

    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=0,
        help=(
            "If >0, write an intermediate parquet checkpoint every N null ablations. "
            "Useful for long runs to avoid losing all progress."
        ),
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Where to write intermediate checkpoints (parquet). Defaults to '<out_null>.partial.parquet'. "
            "Only used if --checkpoint_every > 0."
        ),
    )

    p.add_argument("--max_shortlist", type=int, default=None, help="Optionally cap shortlist to first N rows")
    p.add_argument(
        "--allow_shortlist_in_null",
        action="store_true",
        help="Allow null sampler to pick channels that are also in shortlist",
    )

    return p.parse_args()


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomically write a parquet file by writing a temp file then renaming."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _parse_n(s: str) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in {"all", "none", ""}:
        return None
    return int(s)


def _iter_batches(df: pd.DataFrame, batch_size: int):
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def _build_run_table(
    df_parquet: pd.DataFrame,
    pos_ids: str,
    neg_ids: str,
    n_pos: Optional[int],
    n_neg: Optional[int],
    seed: int,
) -> Tuple[pd.DataFrame, dict]:
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
    """Compute delta_expr = score_alt - score_ref for all rows."""

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


def _spearman_rankcorr(x: np.ndarray, y: np.ndarray) -> float:
    """Fast-ish Spearman via rank correlation."""
    from scipy.stats import rankdata

    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2:
        return float("nan")

    # Drop NaNs pairwise
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")

    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx == 0.0 or sy == 0.0:
        return float("nan")

    # Pearson correlation of ranks
    return float(np.corrcoef(rx, ry)[0, 1])


@dataclass(frozen=True)
class LayerShape:
    layer: str
    out_shape: Tuple[int, ...]
    channel_dim: int
    n_channels: int


@torch.no_grad()
def infer_layer_channels(wrapper, layer_name: str, seq_len: int) -> LayerShape:
    """Run a dummy forward pass and infer number of channels for a module output."""

    module = get_module_by_name(wrapper.model, layer_name)
    captured = {"out": None}

    def hook(_module, _inp, out):
        # Handle dict/tuple/list similarly to ChannelAblator
        t = out
        if isinstance(t, dict):
            # choose first value
            for v in t.values():
                t = v
                break
        if isinstance(t, (tuple, list)):
            t = t[0]
        captured["out"] = t

    handle = module.register_forward_hook(hook)
    try:
        dummy = torch.zeros((1, 4, int(seq_len)), device=wrapper.device, dtype=torch.float32)
        _ = wrapper.model(dummy)
    finally:
        try:
            handle.remove()
        except Exception:
            pass

    out = captured["out"]
    if out is None or not torch.is_tensor(out):
        raise RuntimeError(f"Could not capture tensor output for layer {layer_name}")

    shape = tuple(int(s) for s in out.shape)
    if out.ndim == 3:
        # Heuristic: channels is the smaller of dims 1 and 2 (C vs L).
        ch_dim = 1 if out.shape[1] <= out.shape[2] else 2
        n_ch = int(out.shape[ch_dim])
        return LayerShape(layer=layer_name, out_shape=shape, channel_dim=ch_dim, n_channels=n_ch)

    # Fallback: pick the first non-batch dim in a reasonable channel range.
    for dim in range(1, out.ndim):
        n = int(out.shape[dim])
        if 16 <= n <= 8192:
            return LayerShape(layer=layer_name, out_shape=shape, channel_dim=dim, n_channels=n)

    # Last resort: use dim=1
    if out.ndim >= 2:
        return LayerShape(layer=layer_name, out_shape=shape, channel_dim=1, n_channels=int(out.shape[1]))

    raise RuntimeError(f"Layer output too small to infer channels: {layer_name} shape={shape}")


def allocate_null_counts(layer_counts: Dict[str, int], n_null: int) -> Dict[str, int]:
    total = int(sum(layer_counts.values()))
    if total <= 0:
        raise ValueError("Empty shortlist layer distribution")

    # Proportional allocation, then distribute remainder.
    raw = {layer: (n_null * c / total) for layer, c in layer_counts.items()}
    alloc = {layer: int(np.floor(v)) for layer, v in raw.items()}
    used = int(sum(alloc.values()))
    rem = int(n_null - used)

    if rem > 0:
        # Give remaining slots to largest fractional parts.
        frac = sorted(((layer, raw[layer] - alloc[layer]) for layer in raw), key=lambda x: x[1], reverse=True)
        for i in range(rem):
            alloc[frac[i % len(frac)][0]] += 1

    # Remove zeros to keep loops clean
    return {k: int(v) for k, v in alloc.items() if int(v) > 0}


def bootstrap_auc(y: np.ndarray, score: np.ndarray, bs_idx_all: np.ndarray) -> np.ndarray:
    out = np.empty((bs_idx_all.shape[0],), dtype=np.float32)
    for i in range(bs_idx_all.shape[0]):
        idx = bs_idx_all[i]
        yb = y[idx]
        sb = score[idx]
        if len(np.unique(yb)) < 2:
            out[i] = np.nan
        else:
            out[i] = float(roc_auc_score(yb, sb))
    return out


def bootstrap_spearman(delta_pos: np.ndarray, beta_pos: np.ndarray, bs_idx_pos: np.ndarray) -> np.ndarray:
    out = np.empty((bs_idx_pos.shape[0],), dtype=np.float32)
    for i in range(bs_idx_pos.shape[0]):
        idx = bs_idx_pos[i]
        out[i] = float(_spearman_rankcorr(delta_pos[idx], beta_pos[idx]))
    return out


def summarize_ci(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "frac_gt0": float("nan")}
    return {
        "median": float(np.quantile(x, 0.5)),
        "ci_low": float(np.quantile(x, 0.025)),
        "ci_high": float(np.quantile(x, 0.975)),
        "frac_gt0": float(np.mean(x > 0)),
    }


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_bootstrap = Path(args.out_bootstrap)
    out_null = Path(args.out_null)
    out_bootstrap.parent.mkdir(parents=True, exist_ok=True)
    out_null.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_every = int(args.checkpoint_every)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else Path(str(out_null) + ".partial.parquet")
    if checkpoint_every > 0:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpointing enabled: every {checkpoint_every} null ablations -> {checkpoint_path}")

    logger.info(f"Device available: cuda={torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    shortlist = pd.read_parquet(Path(args.shortlist))
    if "layer" not in shortlist.columns or "channel" not in shortlist.columns:
        raise ValueError("Shortlist must contain columns: layer, channel")
    shortlist = shortlist[["layer", "channel"]].drop_duplicates().reset_index(drop=True)
    if args.max_shortlist is not None:
        shortlist = shortlist.head(int(args.max_shortlist)).copy()

    layer_counts = shortlist["layer"].astype(str).value_counts().to_dict()
    logger.info(f"Shortlist channels: n={len(shortlist)} across {len(layer_counts)} layer(s)")

    n_pos = _parse_n(args.n_pos)
    n_neg = _parse_n(args.n_neg)

    df_parquet = load_eqtl_parquet(args.parquet)
    df_run, overlap_stats = _build_run_table(df_parquet, args.pos_ids, args.neg_ids, n_pos=n_pos, n_neg=n_neg, seed=args.seed)
    logger.info(f"Running variants (pos+neg): n={len(df_run)}")

    # Optional sequence cache (saves repeated FASTA fetch)
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

    genome = Genome(Path(args.fasta))

    # Baseline
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
    base_df = base_df.sort_values("variant_id").reset_index(drop=True)

    y = base_df["label"].astype(int).to_numpy()
    abs_base = np.abs(base_df["delta_expr"].to_numpy(dtype=np.float32))
    baseline_auc = float(roc_auc_score(y, abs_base)) if len(np.unique(y)) == 2 else float("nan")

    pos_mask = y == 1
    beta_pos = base_df.loc[pos_mask, "beta"].to_numpy(dtype=np.float32)
    delta_pos = base_df.loc[pos_mask, "delta_expr"].to_numpy(dtype=np.float32)
    baseline_spearman_beta = _spearman_rankcorr(delta_pos, beta_pos)

    logger.info(f"Baseline: AUC={baseline_auc:.5f} spearman_beta_pos={baseline_spearman_beta:.5f} n_scored={len(base_df)}")
    logger.info(
        f"Scored label counts: pos={int(pos_mask.sum())} neg={int((~pos_mask).sum())} "
        f"(requested pos={n_pos} neg={n_neg})"
    )

    # Bootstrap indices (once)
    b = int(args.bootstrap)
    bs_idx_all = None
    bs_idx_pos = None
    baseline_auc_boot = None
    baseline_spear_boot = None

    if not args.only_null:
        if b <= 0:
            raise ValueError("--bootstrap must be > 0 (or use --only_null)")

        bs_idx_all = rng.integers(low=0, high=len(base_df), size=(b, len(base_df)), dtype=np.int64)
        n_pos_scored = int(pos_mask.sum())
        if n_pos_scored < 2:
            raise RuntimeError("Need at least 2 positive variants for Spearman bootstrap")
        bs_idx_pos = rng.integers(low=0, high=n_pos_scored, size=(b, n_pos_scored), dtype=np.int64)

        baseline_auc_boot = bootstrap_auc(y, abs_base, bs_idx_all)
        baseline_spear_boot = bootstrap_spearman(delta_pos, beta_pos, bs_idx_pos)

    # Probe layers to sample null channels
    layer_shapes: Dict[str, LayerShape] = {}
    for layer in sorted(layer_counts.keys()):
        ls = infer_layer_channels(wrapper, layer, seq_len=int(seq_len))
        layer_shapes[layer] = ls
        logger.info(f"Layer {layer}: out_shape={ls.out_shape} channel_dim={ls.channel_dim} n_channels={ls.n_channels}")

    # Null sampling
    null_alloc = allocate_null_counts(layer_counts, n_null=int(args.n_null))
    logger.info(f"Sampling null channels (matched by layer): {null_alloc}")

    shortlist_set = {(str(r.layer), int(r.channel)) for r in shortlist.itertuples(index=False)}
    null_pairs: List[Tuple[str, int]] = []
    for layer, n_layer in null_alloc.items():
        n_ch = layer_shapes[layer].n_channels
        all_channels = np.arange(n_ch, dtype=np.int32)

        if not args.allow_shortlist_in_null:
            used = sorted({ch for (ly, ch) in shortlist_set if ly == layer})
            if used:
                mask = np.ones((n_ch,), dtype=bool)
                mask[np.array(used, dtype=np.int32)] = False
                all_channels = all_channels[mask]

        replace = len(all_channels) < int(n_layer)
        sampled = rng.choice(all_channels, size=int(n_layer), replace=replace)
        for ch in sampled.tolist():
            null_pairs.append((layer, int(ch)))

    # Deduplicate (rare unless replace=True), then top up if needed
    null_pairs = list(dict.fromkeys(null_pairs))
    while len(null_pairs) < int(args.n_null):
        layer = rng.choice(list(null_alloc.keys()))
        n_ch = layer_shapes[layer].n_channels
        ch = int(rng.integers(low=0, high=n_ch))
        pair = (layer, ch)
        if (not args.allow_shortlist_in_null) and (pair in shortlist_set):
            continue
        if pair in null_pairs:
            continue
        null_pairs.append(pair)
    null_pairs = null_pairs[: int(args.n_null)]

    # Evaluate ablations (shortlist + null)
    def eval_ablation(layer: str, ch: int) -> Tuple[dict, np.ndarray, np.ndarray]:
        ab_df, ab_stats = _compute_delta_expr(
            df_run=df_run,
            wrapper=wrapper,
            genome=genome,
            seq_len=int(seq_len),
            batch_size=int(args.batch_size),
            device=device,
            ablate=(layer, ch),
        )

        ab_df = ab_df[["variant_id", "delta_expr"]].rename(columns={"delta_expr": "delta_expr_ab"})
        merged = base_df.merge(ab_df, on="variant_id", how="inner")
        y_m = merged["label"].astype(int).to_numpy()
        abs_ab = np.abs(merged["delta_expr_ab"].to_numpy(dtype=np.float32))

        auc_ab = float(roc_auc_score(y_m, abs_ab)) if len(np.unique(y_m)) == 2 else float("nan")

        pos_m = y_m == 1
        beta_m = merged.loc[pos_m, "beta"].to_numpy(dtype=np.float32)
        delta_ab_pos = merged.loc[pos_m, "delta_expr_ab"].to_numpy(dtype=np.float32)
        spearman_beta_ab = _spearman_rankcorr(delta_ab_pos, beta_m)

        abs_base_m = np.abs(merged["delta_expr"].to_numpy(dtype=np.float32))
        delta_drop = float(np.mean(abs_base_m - abs_ab))

        row = {
            "layer": str(layer),
            "channel": int(ch),
            "n_scored": int(len(merged)),
            "baseline_auc_label": float(baseline_auc),
            "ablated_auc_label": float(auc_ab),
            "delta_auc": float(baseline_auc - auc_ab),
            "baseline_spearman_beta_pos": float(baseline_spearman_beta),
            "ablated_spearman_beta_pos": float(spearman_beta_ab),
            "delta_spearman_beta": float(baseline_spearman_beta - spearman_beta_ab),
            "mean_abs_delta_drop": float(delta_drop),
            "baseline_stats": json.dumps(base_stats, sort_keys=True),
            "ablated_stats": json.dumps(ab_stats, sort_keys=True),
        }

        # Return abs_ab aligned to base_df (via merge order) and delta_ab_pos aligned to beta_pos.
        # We rely on base_df sorted by variant_id, and merge(inner) keeps that order.
        return row, abs_ab.astype(np.float32), delta_ab_pos.astype(np.float32)

    shortlist_rows: List[dict] = []
    shortlist_abs_ab: Dict[Tuple[str, int], np.ndarray] = {}
    shortlist_delta_pos_ab: Dict[Tuple[str, int], np.ndarray] = {}

    if not args.only_null:
        logger.info("Evaluating shortlist ablations (for bootstrap)")
        for i, r in enumerate(shortlist.itertuples(index=False), start=1):
            layer = str(r.layer)
            ch = int(r.channel)
            logger.info(f"[shortlist {i}/{len(shortlist)}] {layer} ch={ch}")
            row, abs_ab, delta_ab_pos = eval_ablation(layer, ch)
            shortlist_rows.append(row)
            shortlist_abs_ab[(layer, ch)] = abs_ab
            shortlist_delta_pos_ab[(layer, ch)] = delta_ab_pos

    logger.info("Evaluating null ablations")
    null_rows: List[dict] = []
    null_delta_auc: List[float] = []

    for i, (layer, ch) in enumerate(null_pairs, start=1):
        logger.info(f"[null {i}/{len(null_pairs)}] {layer} ch={ch}")
        row, _abs_ab, _delta_ab_pos = eval_ablation(layer, ch)
        row["is_null"] = True
        row["null_rank"] = i
        null_rows.append(row)
        null_delta_auc.append(float(row["delta_auc"]))

        if checkpoint_every > 0 and (i % checkpoint_every == 0 or i == len(null_pairs)):
            _atomic_write_parquet(pd.DataFrame(null_rows), checkpoint_path)
            logger.info(f"Wrote null checkpoint: {checkpoint_path} (rows={len(null_rows)})")

    null_df = pd.DataFrame(null_rows).sort_values("delta_auc", ascending=False).reset_index(drop=True)
    write_parquet(null_df, out_null)
    logger.info(f"Wrote null parquet: {out_null} (rows={len(null_df)})")

    null_delta_auc_arr = np.asarray(null_delta_auc, dtype=np.float64)
    null_mean = float(np.mean(null_delta_auc_arr))
    null_std = float(np.std(null_delta_auc_arr))

    if not args.only_null:
        logger.info("Bootstrapping CIs for shortlist channels")
        boot_rows: List[dict] = []

        for row in shortlist_rows:
            layer = str(row["layer"])
            ch = int(row["channel"])
            key = (layer, ch)

            abs_ab = shortlist_abs_ab[key]
            delta_ab_pos = shortlist_delta_pos_ab[key]

            auc_ab_boot = bootstrap_auc(y, abs_ab, bs_idx_all)
            delta_auc_boot = baseline_auc_boot - auc_ab_boot

            spear_ab_boot = bootstrap_spearman(delta_ab_pos, beta_pos, bs_idx_pos)
            delta_spear_boot = baseline_spear_boot - spear_ab_boot

            auc_ci = summarize_ci(delta_auc_boot)
            spear_ci = summarize_ci(delta_spear_boot)

            # Empirical one-sided p-value vs null (greater ΔAUC = more harmful to ablate)
            point = float(row["delta_auc"])
            p_null = float((1.0 + float(np.sum(null_delta_auc_arr >= point))) / (len(null_delta_auc_arr) + 1.0))
            percentile = float(np.mean(null_delta_auc_arr <= point))

            boot_rows.append(
                {
                    **row,
                    "is_null": False,
                    "bootstrap": int(b),
                    "delta_auc_ci_low": auc_ci["ci_low"],
                    "delta_auc_ci_high": auc_ci["ci_high"],
                    "delta_auc_median": auc_ci["median"],
                    "delta_auc_frac_gt0": auc_ci["frac_gt0"],
                    "delta_spearman_ci_low": spear_ci["ci_low"],
                    "delta_spearman_ci_high": spear_ci["ci_high"],
                    "delta_spearman_median": spear_ci["median"],
                    "delta_spearman_frac_gt0": spear_ci["frac_gt0"],
                    "null_n": int(len(null_delta_auc_arr)),
                    "null_delta_auc_mean": null_mean,
                    "null_delta_auc_std": null_std,
                    "null_p_ge": p_null,
                    "null_percentile": percentile,
                }
            )

        boot_df = pd.DataFrame(boot_rows).sort_values("delta_auc", ascending=False).reset_index(drop=True)
        write_parquet(boot_df, out_bootstrap)
        logger.info(f"Wrote bootstrap parquet: {out_bootstrap} (rows={len(boot_df)})")

    report = {
        "seed": int(args.seed),
        "seq_len": int(seq_len),
        "batch_size": int(args.batch_size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "bootstrap": int(b),
        "only_null": bool(args.only_null),
        "n_null": int(len(null_df)),
        "baseline": {
            "auc_label": float(baseline_auc),
            "spearman_beta_pos": float(baseline_spearman_beta),
            "stats": base_stats,
        },
        "overlap": overlap_stats,
        "shortlist": {
            "path": str(args.shortlist),
            "n_channels": int(len(shortlist)),
            "layer_counts": {k: int(v) for k, v in layer_counts.items()},
        },
        "null": {
            "allocation": {k: int(v) for k, v in null_alloc.items()},
            "delta_auc_mean": null_mean,
            "delta_auc_std": null_std,
        },
        "outputs": {
            "bootstrap_parquet": (str(out_bootstrap) if not args.only_null else None),
            "null_parquet": str(out_null),
        },
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
