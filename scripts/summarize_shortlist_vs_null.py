#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from superweights.io import write_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare shortlist ablation deltas vs a null ablation distribution")
    p.add_argument("--shortlist_ablation", required=True, type=str, help="Parquet from scripts/ablate_channels.py")
    p.add_argument("--null_ablation", required=True, type=str, help="Parquet from scripts/ablation_null_and_bootstrap.py --only_null")
    p.add_argument(
        "--out",
        type=str,
        default="results/ablation/shortlist_vs_null.parquet",
        help="Output parquet with null percentiles and empirical p-values",
    )
    return p.parse_args()


def empirical_p_ge(null: np.ndarray, x: float) -> float:
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(x):
        return float("nan")
    # add-one smoothing
    return float((1.0 + float(np.sum(null >= x))) / (len(null) + 1.0))


def percentile(null: np.ndarray, x: float) -> float:
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(x):
        return float("nan")
    return float(np.mean(null <= x))


def main() -> None:
    args = parse_args()

    df_s = pd.read_parquet(Path(args.shortlist_ablation))
    df_n = pd.read_parquet(Path(args.null_ablation))

    for col in ("layer", "channel", "delta_auc"):
        if col not in df_s.columns:
            raise ValueError(f"shortlist_ablation missing column: {col}")
    for col in ("layer", "channel", "delta_auc"):
        if col not in df_n.columns:
            raise ValueError(f"null_ablation missing column: {col}")

    df_s = df_s.copy()
    df_s["layer"] = df_s["layer"].astype(str)
    df_s["channel"] = df_s["channel"].astype(int)

    df_n = df_n.copy()
    df_n["layer"] = df_n["layer"].astype(str)
    df_n["channel"] = df_n["channel"].astype(int)

    null_all = df_n["delta_auc"].to_numpy(dtype=np.float64)
    null_by_layer = {
        layer: g["delta_auc"].to_numpy(dtype=np.float64) for layer, g in df_n.groupby("layer", sort=False)
    }

    rows = []
    for r in df_s.itertuples(index=False):
        layer = str(getattr(r, "layer"))
        ch = int(getattr(r, "channel"))
        da = float(getattr(r, "delta_auc"))

        nl = null_by_layer.get(layer, null_all)
        rows.append(
            {
                "layer": layer,
                "channel": ch,
                "delta_auc": da,
                "null_n": int(np.isfinite(null_all).sum()),
                "null_layer_n": int(np.isfinite(nl).sum()),
                "null_p_ge": empirical_p_ge(null_all, da),
                "null_percentile": percentile(null_all, da),
                "null_layer_p_ge": empirical_p_ge(nl, da),
                "null_layer_percentile": percentile(nl, da),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["delta_auc"], ascending=False).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(out_df, out_path)
    logger.info(f"Wrote: {out_path} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
