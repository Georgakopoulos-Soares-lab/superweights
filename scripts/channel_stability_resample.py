#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Resampling stability for top channels exported by export_top_channel_deltas.py. "
            "Computes Spearman rho and abs-rank across channels on repeated subsamples of variants."
        )
    )

    p.add_argument("--run_dir", required=True, type=str, help="Run directory containing per_variant_delta_activation_top*.parquet")
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer names to process. Default: infer from filenames in run_dir.",
    )
    p.add_argument("--topk", type=int, default=50, help="Top-K channels per layer that were exported")
    p.add_argument("--n_resamples", type=int, default=20)
    p.add_argument("--frac", type=float, default=0.7, help="Fraction of variants per resample")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--target",
        type=str,
        default="signed",
        choices=["signed", "abs"],
        help="Which task readout to correlate against: signed delta or abs(delta).",
    )

    return p.parse_args()


def _infer_layers(run_dir: Path, topk: int) -> List[str]:
    layers: List[str] = []
    prefix = f"per_variant_delta_activation_top{int(topk)}_"
    for p in sorted(run_dir.glob(f"{prefix}*.parquet")):
        name = p.name[len(prefix) : -len(".parquet")]
        layers.append(name)
    return layers


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if int(m.sum()) < 10:
        return float("nan")
    rho, _p = spearmanr(x[m], y[m])
    return float(rho)


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Missing run_dir: {run_dir}")

    rng = np.random.default_rng(int(args.seed))

    if args.layers is None:
        layer_keys = _infer_layers(run_dir, int(args.topk))
        if not layer_keys:
            raise SystemExit(f"No exported parquet files found in {run_dir}")
    else:
        layer_keys = [s.strip().replace(".", "_") for s in str(args.layers).split(",") if s.strip()]

    all_summaries: List[pd.DataFrame] = []

    for layer_key in layer_keys:
        in_path = run_dir / f"per_variant_delta_activation_top{int(args.topk)}_{layer_key}.parquet"
        if not in_path.exists():
            raise SystemExit(f"Missing exported delta activation file: {in_path}")

        df = pd.read_parquet(in_path)
        # normalize types
        df["variant_id"] = df["variant_id"].astype(str)
        df["channel"] = df["channel"].astype(int)

        variant_ids = df["variant_id"].unique().tolist()
        N = len(variant_ids)
        n_take = int(np.floor(float(args.frac) * float(N)))
        n_take = max(10, min(N, n_take))

        channels = sorted(df["channel"].unique().tolist())
        logger.info(f"Layer {layer_key}: variants={N} channels={len(channels)} resamples={int(args.n_resamples)} frac={args.frac}")

        # Prepare per-channel vectors aligned by variant_id using merge/pivot.
        # df is long: one row per (variant_id, channel)
        # We'll build per-channel arrays by sorting by variant_id once per resample.
        df_small = df[["variant_id", "channel", "delta_act", "delta_task_readout", "rho_full", "abs_rho_full", "rank_full", "pad_bins", "shift_bins"]].copy()

        # Pre-split by channel to speed up
        by_channel: Dict[int, pd.DataFrame] = {c: g.sort_values("variant_id").reset_index(drop=True) for c, g in df_small.groupby("channel")}
        # shared ordering
        base_vid = by_channel[channels[0]]["variant_id"].to_numpy()
        for c in channels[1:]:
            if not np.array_equal(base_vid, by_channel[c]["variant_id"].to_numpy()):
                raise RuntimeError(f"Variant_id ordering mismatch for channel {c}; export file must contain all channels for all variants")

        if str(args.target).lower() == "abs":
            if "target_task_readout" in by_channel[channels[0]].columns:
                y_full = by_channel[channels[0]]["target_task_readout"].to_numpy(dtype=np.float64)
            else:
                y_full = np.abs(by_channel[channels[0]]["delta_task_readout"].to_numpy(dtype=np.float64))
        else:
            y_full = by_channel[channels[0]]["delta_task_readout"].to_numpy(dtype=np.float64)

        # resample bookkeeping
        ranks_mat = np.full((int(args.n_resamples), len(channels)), np.nan, dtype=np.float64)
        absrho_mat = np.full((int(args.n_resamples), len(channels)), np.nan, dtype=np.float64)

        for ri in range(int(args.n_resamples)):
            take = rng.choice(N, size=n_take, replace=False)

            rhos = np.full((len(channels),), np.nan, dtype=np.float64)
            for ci, c in enumerate(channels):
                x = by_channel[c]["delta_act"].to_numpy(dtype=np.float64)
                rho = _spearman(x[take], y_full[take])
                rhos[ci] = rho

            abs_rho = np.abs(rhos)
            order = np.argsort(-abs_rho)  # descending
            rank = np.empty_like(order, dtype=np.int32)
            rank[order] = np.arange(1, len(channels) + 1, dtype=np.int32)

            ranks_mat[ri, :] = rank.astype(np.float64)
            absrho_mat[ri, :] = abs_rho

        # summarize
        df_full_meta = (
            df_small.drop_duplicates("channel")
            .set_index("channel")
            .loc[channels]
            .reset_index()
        )

        mean_rank = np.nanmean(ranks_mat, axis=0)
        std_rank = np.nanstd(ranks_mat, axis=0)
        mean_absrho = np.nanmean(absrho_mat, axis=0)
        std_absrho = np.nanstd(absrho_mat, axis=0)

        frac_top5 = np.nanmean((ranks_mat <= 5).astype(np.float64), axis=0)
        frac_top10 = np.nanmean((ranks_mat <= 10).astype(np.float64), axis=0)

        df_sum = pd.DataFrame(
            {
                "layer": df["layer"].iloc[0] if "layer" in df.columns else layer_key.replace("_", "."),
            "target": str(args.target).lower(),
                "channel": channels,
                "pad_bins": df_full_meta["pad_bins"].astype(int).to_list(),
                "shift_bins": df_full_meta["shift_bins"].astype(int).to_list(),
                "rho_full": df_full_meta["rho_full"].astype(float).to_list(),
                "abs_rho_full": df_full_meta["abs_rho_full"].astype(float).to_list(),
                "rank_full": df_full_meta["rank_full"].astype(int).to_list(),
                "mean_abs_rho": mean_absrho,
                "std_abs_rho": std_absrho,
                "mean_rank": mean_rank,
                "std_rank": std_rank,
                "frac_top5": frac_top5,
                "frac_top10": frac_top10,
                "stability_score": frac_top10,
            }
        )

        df_sum = df_sum.sort_values(["stability_score", "mean_abs_rho"], ascending=[False, False]).reset_index(drop=True)

        out_pq = run_dir / (
            f"stability_{layer_key}_top{int(args.topk)}_{str(args.target).lower()}_n{int(args.n_resamples)}_frac{args.frac:.2f}.parquet"
        )
        out_tsv = run_dir / (
            f"stability_{layer_key}_top{int(args.topk)}_{str(args.target).lower()}_n{int(args.n_resamples)}_frac{args.frac:.2f}.tsv"
        )
        df_sum.to_parquet(out_pq, index=False)
        df_sum.to_csv(out_tsv, sep="\t", index=False)
        logger.info(f"Wrote stability summary: {out_tsv}")

        all_summaries.append(df_sum)

    if all_summaries:
        df_all = pd.concat(all_summaries, axis=0, ignore_index=True)
        out_all = run_dir / (
            f"stability_ALL_top{int(args.topk)}_{str(args.target).lower()}_n{int(args.n_resamples)}_frac{args.frac:.2f}.tsv"
        )
        df_all.to_csv(out_all, sep="\t", index=False)
        logger.info(f"Wrote combined summary: {out_all}")


if __name__ == "__main__":
    main()
