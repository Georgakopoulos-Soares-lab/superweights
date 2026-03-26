from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize top Borzoi channels from channel_correlations.parquet")
    p.add_argument("--corr_parquet", required=True, type=str)
    p.add_argument("--outdir", required=True, type=str)
    p.add_argument("--topk", type=int, default=50)
    return p.parse_args()


def _top(df: pd.DataFrame, sort_col: str, topk: int) -> pd.DataFrame:
    d = df.copy()
    d = d[np.isfinite(d[sort_col].to_numpy(dtype=float))]
    d = d.sort_values(sort_col, ascending=False).head(topk)
    return d


def _top_abs(df: pd.DataFrame, col: str, topk: int) -> pd.DataFrame:
    d = df.copy()
    arr = d[col].to_numpy(dtype=float)
    d = d[np.isfinite(arr)]
    d = d.assign(_abs=np.abs(d[col].to_numpy(dtype=float)))
    d = d.sort_values("_abs", ascending=False).head(topk).drop(columns=["_abs"])
    return d


def main() -> None:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    corr = pd.read_parquet(args.corr_parquet)

    # prefer features that are typically most discriminative for SNP impact
    prefer = corr[corr["feature_type"].isin(["abs_delta_l2", "abs_delta_max", "delta_l2", "delta_max", "delta_mean"])]

    top_auc = _top(prefer, "auc_label", args.topk)
    top_beta = _top_abs(prefer, "spearman_beta", args.topk)
    top_delta = _top_abs(prefer, "spearman_delta", args.topk)

    top_auc.to_parquet(outdir / "top_auc_label.parquet", index=False)
    top_beta.to_parquet(outdir / "top_abs_spearman_beta.parquet", index=False)
    top_delta.to_parquet(outdir / "top_abs_spearman_delta.parquet", index=False)

    summary = {
        "n_rows": int(len(corr)),
        "n_layers": int(corr["layer"].nunique()),
        "n_feature_types": int(corr["feature_type"].nunique()),
        "topk": int(args.topk),
        "outputs": {
            "top_auc_label": str(outdir / "top_auc_label.parquet"),
            "top_abs_spearman_beta": str(outdir / "top_abs_spearman_beta.parquet"),
            "top_abs_spearman_delta": str(outdir / "top_abs_spearman_delta.parquet"),
        },
    }

    (outdir / "top_channels_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
