#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from borzoi.genome import Genome, fetch_with_padding
from borzoi.io import ensure_dir, safe_slug


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract motif FASTA sequences around localized channel sites.")
    p.add_argument("--posmaps", nargs="+", required=True, help="One or more posmap parquet files (results/posmaps/*.parquet)")
    p.add_argument("--fasta", required=True)
    p.add_argument("--flank_bp", type=int, default=200, help="Extract +/- flank_bp around site_center1")
    p.add_argument("--max_per_channel", type=int, default=1000, help="Max sequences per channel per label")
    p.add_argument("--outdir", default="motifs")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def _write_fasta(path: Path, records: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            # wrap at 80
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def _select_top_sites(df: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    # Deduplicate by genomic coordinate + channel
    df = df.copy()
    df["coord_key"] = df["chrom"].astype(str) + ":" + df["site_center1"].astype(int).astype(str)
    df = df.sort_values("abs_delta", ascending=False)
    df = df.drop_duplicates(subset=["layer", "channel", "coord_key"], keep="first")

    if len(df) > max_n:
        # stable top-N by abs_delta (deterministic)
        df = df.head(max_n)
    return df


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)

    outdir = ensure_dir(args.outdir)
    genome = Genome(Path(args.fasta))

    # load all posmaps
    dfs = []
    for p in args.posmaps:
        df = pd.read_parquet(p)
        dfs.append(df)
    df_all = pd.concat(dfs, axis=0, ignore_index=True)

    required = {"layer", "channel", "label", "chrom", "site_center1", "delta_act", "window_type"}
    missing = required - set(df_all.columns)
    if missing:
        raise ValueError(f"Posmaps missing columns: {sorted(missing)}")

    df_all["abs_delta"] = df_all["delta_act"].abs()

    # group per channel
    n_written = 0
    for (layer, channel), df_ch in df_all.groupby(["layer", "channel"], sort=False):
        df_pos = df_ch[df_ch["label"] == 1].copy()
        df_neg = df_ch[df_ch["label"] == 0].copy()

        df_pos = _select_top_sites(df_pos, max_n=int(args.max_per_channel), seed=args.seed)
        df_neg = _select_top_sites(df_neg, max_n=int(args.max_per_channel), seed=args.seed)

        def make_records(df: pd.DataFrame, label: str) -> List[Tuple[str, str]]:
            recs: List[Tuple[str, str]] = []
            for _, r in df.iterrows():
                chrom = str(r["chrom"])
                center1 = int(r["site_center1"])
                center0 = center1 - 1
                start0 = center0 - int(args.flank_bp)
                end0 = center0 + int(args.flank_bp) + 1
                seq = fetch_with_padding(genome, chrom=chrom, start0=start0, end0=end0)

                header = (
                    f"layer={safe_slug(layer)}|channel={int(channel)}|label={label}|"
                    f"variant={r.get('variant_id','NA')}|window={r.get('window_type','NA')}|"
                    f"chrom={chrom}|center1={center1}|delta_act={float(r['delta_act']):.6g}"
                )
                recs.append((header, seq))
            return recs

        base = f"{safe_slug(layer)}_{int(channel)}"
        pos_path = outdir / f"{base}_pos.fa"
        neg_path = outdir / f"{base}_neg.fa"

        _write_fasta(pos_path, make_records(df_pos, label="pos"))
        _write_fasta(neg_path, make_records(df_neg, label="neg"))

        n_written += 1

    logger.info(f"Wrote FASTA for channels: {n_written} into {outdir}")


if __name__ == "__main__":
    main()
