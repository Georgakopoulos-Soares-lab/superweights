#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare signed-target vs abs-target channel rankings and stability. "
            "Merges best_across_settings and stability outputs for each layer/channel."
        )
    )

    p.add_argument("--run_signed", required=True, type=str, help="Signed run directory")
    p.add_argument("--run_abs", required=True, type=str, help="Abs run directory")
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layers to compare (default: infer from best_* files in run_signed)",
    )
    p.add_argument("--topk", type=int, default=50, help="Top-K channels per run to include")

    p.add_argument(
        "--stability_signed",
        type=str,
        default=None,
        help="Optional: path to signed stability_ALL or per-layer file. If omitted, comparison uses best-only.",
    )
    p.add_argument(
        "--stability_abs",
        type=str,
        default=None,
        help="Optional: path to abs stability_ALL or per-layer file. If omitted, comparison uses best-only.",
    )

    p.add_argument(
        "--motif_channels",
        type=str,
        default=None,
        help="Optional comma-separated list like 'horizontal_conv1.conv_layer:130,final_joined_convs.0.conv_layer:1599'",
    )

    p.add_argument("--out_tsv", required=True, type=str, help="Output TSV path")

    return p.parse_args()


def _parse_layers(run_dir: Path) -> List[str]:
    layers = []
    for p in run_dir.glob("best_*_across_settings.parquet"):
        # best_{layer.replace('.', '_')}_across_settings.parquet
        key = p.name[len("best_") : -len("_across_settings.parquet")]
        layers.append(key)
    return sorted(set(layers))


def _parse_layer_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    out = [x.strip() for x in str(s).split(",") if x.strip()]
    return out or None


def _load_best(run_dir: Path, layer_key: str, topk: int) -> pd.DataFrame:
    p = run_dir / f"best_{layer_key}_across_settings.parquet"
    df = pd.read_parquet(p)
    df = df.sort_values("abs_rho", ascending=False).head(int(topk)).copy()
    df["layer_key"] = layer_key
    return df


def _load_stability(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df


def _parse_motif_channels(s: Optional[str]) -> pd.DataFrame:
    if not s:
        return pd.DataFrame(columns=["layer", "channel", "is_motif_channel"]).astype({"channel": "int64"})

    rows = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"Bad motif_channels entry (expected layer:channel): {part}")
        layer, ch = part.split(":", 1)
        rows.append({"layer": layer.strip(), "channel": int(ch), "is_motif_channel": True})

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    run_signed = Path(args.run_signed)
    run_abs = Path(args.run_abs)
    out_tsv = Path(args.out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    layers = _parse_layer_list(args.layers)
    if layers is None:
        layers = _parse_layers(run_signed)
        if not layers:
            raise SystemExit(f"No best_* files found in {run_signed}")

    # convert layer keys to layer names to merge consistently
    def key_to_layer(layer_key: str) -> str:
        return layer_key.replace("_", ".")

    signed_parts = []
    abs_parts = []
    for layer_key in layers:
        signed_parts.append(_load_best(run_signed, layer_key, int(args.topk)))
        abs_parts.append(_load_best(run_abs, layer_key, int(args.topk)))

    df_s = pd.concat(signed_parts, ignore_index=True)
    df_a = pd.concat(abs_parts, ignore_index=True)

    # normalize
    df_s["layer"] = df_s["layer"].astype(str)
    df_a["layer"] = df_a["layer"].astype(str)

    df_s = df_s.rename(
        columns={
            "rho": "rho_signed",
            "abs_rho": "abs_rho_signed",
            "rank": "rank_signed",
            "pad_bins": "pad_bins_signed",
            "shift_bins": "shift_bins_signed",
            "p": "p_signed",
            "q": "q_signed",
            "p_emp": "p_emp_signed",
        }
    )
    df_a = df_a.rename(
        columns={
            "rho": "rho_abs",
            "abs_rho": "abs_rho_abs",
            "rank": "rank_abs",
            "pad_bins": "pad_bins_abs",
            "shift_bins": "shift_bins_abs",
            "p": "p_abs",
            "q": "q_abs",
            "p_emp": "p_emp_abs",
        }
    )

    # Merge on (layer, channel)
    keep_s = [
        "layer",
        "channel",
        "rho_signed",
        "abs_rho_signed",
        "rank_signed",
        "pad_bins_signed",
        "shift_bins_signed",
        "p_signed",
        "q_signed",
        "p_emp_signed",
    ]
    keep_a = [
        "layer",
        "channel",
        "rho_abs",
        "abs_rho_abs",
        "rank_abs",
        "pad_bins_abs",
        "shift_bins_abs",
        "p_abs",
        "q_abs",
        "p_emp_abs",
    ]

    df = pd.merge(df_s[keep_s], df_a[keep_a], on=["layer", "channel"], how="outer")

    # Optional stability merges
    if args.stability_signed:
        st_s = _load_stability(Path(args.stability_signed)).rename(
            columns={
                "mean_abs_rho": "mean_abs_rho_signed",
                "std_abs_rho": "std_abs_rho_signed",
                "mean_rank": "mean_rank_signed",
                "std_rank": "std_rank_signed",
                "frac_top5": "frac_top5_signed",
                "frac_top10": "frac_top10_signed",
                "stability_score": "stability_score_signed",
            }
        )
        df = df.merge(st_s[[c for c in st_s.columns if c in {"layer", "channel", "target"} or c.endswith("_signed")]], on=["layer", "channel"], how="left")

    if args.stability_abs:
        st_a = _load_stability(Path(args.stability_abs)).rename(
            columns={
                "mean_abs_rho": "mean_abs_rho_abs",
                "std_abs_rho": "std_abs_rho_abs",
                "mean_rank": "mean_rank_abs",
                "std_rank": "std_rank_abs",
                "frac_top5": "frac_top5_abs",
                "frac_top10": "frac_top10_abs",
                "stability_score": "stability_score_abs",
            }
        )
        df = df.merge(st_a[[c for c in st_a.columns if c in {"layer", "channel", "target"} or c.endswith("_abs")]], on=["layer", "channel"], how="left")

    # Motif channel flags
    motif = _parse_motif_channels(args.motif_channels)
    if not motif.empty:
        df = df.merge(motif, on=["layer", "channel"], how="left")
    if "is_motif_channel" not in df.columns:
        df["is_motif_channel"] = False
    df["is_motif_channel"] = df["is_motif_channel"].fillna(False).astype(bool)

    # simple diffs
    if "rank_signed" in df.columns and "rank_abs" in df.columns:
        df["rank_signed_minus_abs"] = df["rank_signed"] - df["rank_abs"]
    if "abs_rho_signed" in df.columns and "abs_rho_abs" in df.columns:
        df["abs_rho_abs_minus_signed"] = df["abs_rho_abs"] - df["abs_rho_signed"]

    # sort by abs ranking strength if present
    sort_cols = []
    if "abs_rho_abs" in df.columns:
        sort_cols.append("abs_rho_abs")
    if "abs_rho_signed" in df.columns:
        sort_cols.append("abs_rho_signed")
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)

    df.to_csv(out_tsv, sep="\t", index=False)


if __name__ == "__main__":
    main()
