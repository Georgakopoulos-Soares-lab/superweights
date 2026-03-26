#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from borzoi.io import df_to_markdown_table, ensure_parent_dir, read_parquet, write_parquet, write_text


def _best_by_abs(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp["__abs"] = tmp[value_col].abs()
    idx = tmp.groupby(group_cols, sort=False)["__abs"].idxmax()
    out = df.loc[idx].copy()
    out = out.drop(columns=[c for c in ("__abs",) if c in out.columns], errors="ignore")
    return out


def build_shortlist_table(df: pd.DataFrame) -> pd.DataFrame:
    # best AUC (per layer,channel)
    idx_auc = df.groupby(["layer", "channel"], sort=False)["auc_label"].idxmax()
    best_auc = df.loc[idx_auc, ["layer", "channel", "feature_type", "auc_label", "n"]].copy()
    best_auc = best_auc.rename(columns={"feature_type": "feature_auc"})

    # best |corr_beta|
    best_beta = _best_by_abs(df, value_col="spearman_beta", group_cols=["layer", "channel"])
    best_beta = best_beta[["layer", "channel", "feature_type", "spearman_beta", "p_spearman_beta"]].copy()
    best_beta = best_beta.rename(columns={"feature_type": "feature_corr_beta"})
    best_beta["abs_spearman_beta"] = best_beta["spearman_beta"].abs()

    # best |corr_delta_expr|
    best_delta = _best_by_abs(df, value_col="spearman_delta", group_cols=["layer", "channel"])
    best_delta = best_delta[["layer", "channel", "feature_type", "spearman_delta", "p_spearman_delta"]].copy()
    best_delta = best_delta.rename(columns={"feature_type": "feature_corr_delta"})
    best_delta["abs_spearman_delta"] = best_delta["spearman_delta"].abs()

    out = best_auc.merge(best_beta, on=["layer", "channel"], how="left").merge(best_delta, on=["layer", "channel"], how="left")
    out = out.sort_values(["auc_label", "abs_spearman_beta"], ascending=[False, False]).reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Select a shortlist of channels for motif discovery/validation.")
    ap.add_argument("--in_parquet", default="results/full_v1/channel_correlations.parquet")
    ap.add_argument("--min_auc", type=float, default=0.60, help="Keep channels with auc_label >= this unless --top_k is set")
    ap.add_argument("--min_abs_corr_beta", type=float, default=0.10, help="Keep channels with |spearman_beta| >= this")
    ap.add_argument("--top_k", type=int, default=None, help="If set, ignore thresholds and keep top-K by auc_label")
    ap.add_argument("--out", default="results/shortlist_channels.parquet")

    ap.add_argument(
        "--drop_scalar_tracking",
        action="store_true",
        help="Optionally drop channels where |corr_delta_expr| is huge but |corr_beta| is small",
    )
    ap.add_argument("--scalar_delta_thr", type=float, default=0.50)
    ap.add_argument("--scalar_beta_thr", type=float, default=0.05)

    ap.add_argument("--report_md", default="reports/shortlist.md")

    args = ap.parse_args()

    in_path = Path(args.in_parquet)
    df = read_parquet(in_path)

    required = {"layer", "channel", "feature_type", "spearman_beta", "spearman_delta", "auc_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    tbl = build_shortlist_table(df)

    if args.drop_scalar_tracking:
        before = len(tbl)
        mask = ~((tbl["abs_spearman_delta"] >= args.scalar_delta_thr) & (tbl["abs_spearman_beta"] < args.scalar_beta_thr))
        tbl = tbl[mask].copy().reset_index(drop=True)
        logger.info(f"Dropped scalar-tracking channels: {before - len(tbl)}")

    if args.top_k is not None:
        shortlist = tbl.head(int(args.top_k)).copy()
    else:
        shortlist = tbl[(tbl["auc_label"] >= float(args.min_auc)) & (tbl["abs_spearman_beta"] >= float(args.min_abs_corr_beta))].copy()

    # Keep both metrics visible; also keep which feature_type achieved them.
    shortlist = shortlist.sort_values(["auc_label", "abs_spearman_beta"], ascending=[False, False]).reset_index(drop=True)

    out_path = write_parquet(shortlist, args.out)
    logger.info(f"Wrote shortlist: {out_path} rows={len(shortlist)}")

    # Markdown report
    report_lines = []
    report_lines.append("# Channel shortlist\n")
    report_lines.append(f"Input: `{in_path}`\n")
    report_lines.append("\n## Decision rules (encoded)\n")
    if args.top_k is not None:
        report_lines.append(f"- Keep top `{args.top_k}` channels by `auc_label` (best over feature types).\n")
    else:
        report_lines.append(f"- Keep channels with `auc_label >= {args.min_auc:.3f}` (best over feature types).\n")
        report_lines.append(f"- AND `|spearman_beta| >= {args.min_abs_corr_beta:.3f}` (best over feature types).\n")
    if args.drop_scalar_tracking:
        report_lines.append(
            f"- Drop scalar-tracking: `|spearman_delta| >= {args.scalar_delta_thr:.3f}` AND `|spearman_beta| < {args.scalar_beta_thr:.3f}`.\n"
        )
    report_lines.append("\n## Why these filters\n")
    report_lines.append("- Primary: `auc_label` (causal sensitivity, pos vs neg).\n")
    report_lines.append("- Secondary: `corr_beta` (biological meaning / effect-size alignment).\n")

    report_lines.append("\n## Summary\n")
    report_lines.append(f"- Total unique channels: `{tbl.shape[0]}`\n")
    report_lines.append(f"- Shortlisted channels: `{shortlist.shape[0]}`\n")

    report_lines.append("\n## Top channels (preview)\n")
    cols = [
        "layer",
        "channel",
        "auc_label",
        "feature_auc",
        "spearman_beta",
        "abs_spearman_beta",
        "feature_corr_beta",
        "spearman_delta",
        "abs_spearman_delta",
        "feature_corr_delta",
    ]
    table = df_to_markdown_table(shortlist, columns=cols, max_rows=20)
    report_lines.append(table.render())

    report_path = write_text(args.report_md, "".join(report_lines))
    logger.info(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
