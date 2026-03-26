#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize ch130 multi-seed ablation and compare to a null distribution.")
    p.add_argument("--ch_parquet", required=True, type=str, help="Per-seed parquet from scripts/ablate_one_channel.py")
    p.add_argument("--ch_report", required=False, type=str, default=None, help="Report JSON from scripts/ablate_one_channel.py")
    p.add_argument("--null_parquet", required=False, type=str, default=None, help="Null parquet (or partial parquet)")
    p.add_argument("--out_md", required=False, type=str, default="reports/ch130_ablation_summary.md")
    p.add_argument("--out_json", required=False, type=str, default="reports/ch130_ablation_summary.json")
    return p.parse_args()


def _empirical_p_ge(null_vals: np.ndarray, x: float) -> float:
    null_vals = np.asarray(null_vals, dtype=np.float64)
    null_vals = null_vals[np.isfinite(null_vals)]
    if len(null_vals) == 0 or not np.isfinite(x):
        return float("nan")
    ge = float(np.sum(null_vals >= x))
    return float((1.0 + ge) / (1.0 + len(null_vals)))


def _percentile(null_vals: np.ndarray, x: float) -> float:
    null_vals = np.asarray(null_vals, dtype=np.float64)
    null_vals = null_vals[np.isfinite(null_vals)]
    if len(null_vals) == 0 or not np.isfinite(x):
        return float("nan")
    return float(100.0 * np.mean(null_vals < x))


def main() -> None:
    args = parse_args()

    ch_df = pd.read_parquet(Path(args.ch_parquet))
    ch_df = ch_df.sort_values("seed").reset_index(drop=True)

    delta_auc = ch_df["delta_auc"].astype(float).to_numpy()
    delta_spear = ch_df["delta_spearman_beta"].astype(float).to_numpy()

    out = {
        "n_seeds": int(len(ch_df)),
        "delta_auc": {
            "mean": float(np.nanmean(delta_auc)),
            "std": float(np.nanstd(delta_auc, ddof=1)) if len(delta_auc) > 1 else float("nan"),
            "min": float(np.nanmin(delta_auc)),
            "max": float(np.nanmax(delta_auc)),
        },
        "delta_spearman_beta": {
            "mean": float(np.nanmean(delta_spear)),
            "std": float(np.nanstd(delta_spear, ddof=1)) if len(delta_spear) > 1 else float("nan"),
            "min": float(np.nanmin(delta_spear)),
            "max": float(np.nanmax(delta_spear)),
        },
        "per_seed": ch_df[["seed", "baseline_auc_label", "ablated_auc_label", "delta_auc", "baseline_spearman_beta_pos", "ablated_spearman_beta_pos", "delta_spearman_beta", "n_scored"]]
        .to_dict(orient="records"),
    }

    if args.ch_report and Path(args.ch_report).exists():
        out["report_json"] = json.loads(Path(args.ch_report).read_text())

    if args.null_parquet and Path(args.null_parquet).exists():
        null_df = pd.read_parquet(Path(args.null_parquet))
        if "delta_auc" in null_df.columns:
            null_vals = null_df["delta_auc"].astype(float).to_numpy()
            point = float(out["delta_auc"]["mean"])
            out["null"] = {
                "path": str(args.null_parquet),
                "n": int(len(null_df)),
                "mean": float(np.nanmean(null_vals)),
                "std": float(np.nanstd(null_vals, ddof=1)) if len(null_vals) > 1 else float("nan"),
                "min": float(np.nanmin(null_vals)),
                "max": float(np.nanmax(null_vals)),
                "p_ge_mean_delta_auc": _empirical_p_ge(null_vals, point),
                "percentile_mean_delta_auc": _percentile(null_vals, point),
            }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))

    md_lines = []
    md_lines.append("# ch130 ablation summary\n\n")
    md_lines.append(f"Input: `{args.ch_parquet}`\n\n")
    md_lines.append("## ΔAUC (baseline - ablated)\n\n")
    md_lines.append(
        f"- mean: {out['delta_auc']['mean']:.6f}\n- std: {out['delta_auc']['std']:.6f}\n- min/max: {out['delta_auc']['min']:.6f} / {out['delta_auc']['max']:.6f}\n\n"
    )
    md_lines.append("## ΔSpearman(beta) on positives (baseline - ablated)\n\n")
    md_lines.append(
        f"- mean: {out['delta_spearman_beta']['mean']:.6f}\n- std: {out['delta_spearman_beta']['std']:.6f}\n- min/max: {out['delta_spearman_beta']['min']:.6f} / {out['delta_spearman_beta']['max']:.6f}\n\n"
    )

    if "null" in out:
        n = out["null"]
        md_lines.append("## Null comparison (empirical)\n\n")
        md_lines.append(f"- null parquet: `{n['path']}`\n")
        md_lines.append(f"- null n={n['n']} mean={n['mean']:.6f} std={n['std']:.6f} max={n['max']:.6f}\n")
        md_lines.append(f"- p(null ≥ mean ΔAUC): {n['p_ge_mean_delta_auc']:.6f}\n")
        md_lines.append(f"- percentile(mean ΔAUC): {n['percentile_mean_delta_auc']:.1f}%\n\n")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md_lines))


if __name__ == "__main__":
    main()
