#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from borzoi.io import df_to_markdown_table, ensure_parent_dir, read_parquet, write_parquet, write_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize per-channel position-map outputs (posmaps).")
    p.add_argument("--posmaps_glob", default="results/posmaps/*.parquet")
    p.add_argument("--shortlist", default="results/shortlist_channels.parquet")
    p.add_argument("--out_parquet", default="reports/posmaps_channel_summary.parquet")
    p.add_argument("--out_md", default="reports/posmaps_channel_summary.md")
    p.add_argument("--window_type", default="near_variant", choices=["near_variant", "global", "all"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    paths = sorted(glob.glob(args.posmaps_glob))
    if not paths:
        raise FileNotFoundError(f"No posmaps matched: {args.posmaps_glob}")

    rows = []
    for p in paths:
        df = pd.read_parquet(p)
        layer = str(df["layer"].iloc[0])
        channel = int(df["channel"].iloc[0])

        if args.window_type != "all":
            df = df[df["window_type"] == args.window_type].copy()

        if len(df) == 0:
            continue

        df["abs_delta"] = df["delta_act"].abs()

        y = df["label"].to_numpy()
        x = df["abs_delta"].to_numpy()

        auc = float(roc_auc_score(y, x)) if len(np.unique(y)) == 2 and np.std(x) > 0 else np.nan

        coord_key = df["chrom"].astype(str) + ":" + df["site_center1"].astype(int).astype(str)

        med_pos = float(np.median(x[y == 1])) if (y == 1).any() else np.nan
        med_neg = float(np.median(x[y == 0])) if (y == 0).any() else np.nan

        rows.append(
            {
                "layer": layer,
                "channel": channel,
                "window_type": args.window_type,
                "n_sites": int(len(df)),
                "n_pos_sites": int((df["label"] == 1).sum()),
                "n_neg_sites": int((df["label"] == 0).sum()),
                "n_unique_coords": int(coord_key.nunique()),
                "auc_pos_vs_neg_abs_delta": auc,
                "median_abs_delta_pos": med_pos,
                "median_abs_delta_neg": med_neg,
                "median_ratio_pos_over_neg": (med_pos / (med_neg + 1e-12))
                if np.isfinite(med_pos) and np.isfinite(med_neg)
                else np.nan,
            }
        )

    summ = pd.DataFrame(rows)

    if Path(args.shortlist).exists():
        short = read_parquet(args.shortlist)
        summ = summ.merge(short, on=["layer", "channel"], how="left")

    summ = summ.sort_values(["auc_label", "abs_spearman_beta", "auc_pos_vs_neg_abs_delta"], ascending=[False, False, False])

    write_parquet(summ, args.out_parquet)

    md = []
    md.append("# Position-map summary\n")
    md.append(f"Input glob: `{args.posmaps_glob}`\n")
    md.append(f"Window type: `{args.window_type}`\n")
    md.append(f"Channels summarized: `{len(summ)}`\n\n")

    cols = [
        "layer",
        "channel",
        "auc_label",
        "abs_spearman_beta",
        "spearman_beta",
        "feature_corr_beta",
        "auc_pos_vs_neg_abs_delta",
        "median_ratio_pos_over_neg",
        "n_unique_coords",
        "n_sites",
    ]
    cols = [c for c in cols if c in summ.columns]

    md.append("## Top channels (by auc_label, |corr_beta|, site_auc)\n")
    md.append(df_to_markdown_table(summ, columns=cols, max_rows=20).render())

    write_text(args.out_md, "".join(md))


if __name__ == "__main__":
    main()
