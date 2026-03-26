#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Allow running from any cwd.
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from scripts.regions._common import (
    Bed6,
    chrom_prefix_mode_to_target,
    ensure_dir,
    ensure_fai,
    filter_to_bounds,
    normalize_chrom,
    read_fai,
    sort_bed,
    write_bed6,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate deterministic random fixed-size windows for hg38.")
    p.add_argument("--genome_fasta", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="data/regions/hg38")
    p.add_argument("--seq_len_bp", type=int, default=262144)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n_random", type=int, default=50000)
    p.add_argument("--exclude_chroms", type=str, default="chrM", help="Comma-separated list")
    p.add_argument("--chrom_prefix_mode", type=str, default="auto", choices=["auto", "add_chr", "strip_chr"])
    p.add_argument("--out_bed", type=str, default=None)
    return p.parse_args()


def _eligible_chroms(
    chrom_sizes: Dict[str, int], *, seq_len_bp: int, exclude: List[str]
) -> Tuple[List[str], np.ndarray]:
    chroms: List[str] = []
    weights: List[float] = []
    for c, L in chrom_sizes.items():
        if c in exclude:
            continue
        max_start = int(L) - int(seq_len_bp)
        if max_start <= 0:
            continue
        chroms.append(str(c))
        weights.append(float(max_start + 1))
    if not chroms:
        raise RuntimeError("No eligible chromosomes for random windows")
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return chroms, w


def build_random(
    *,
    genome_fasta: str | Path,
    out_dir: str | Path,
    seq_len_bp: int = 262144,
    seed: int = 1,
    n_random: int = 50000,
    exclude_chroms: str = "chrM",
    chrom_prefix_mode: str = "auto",
    out_bed: Optional[str | Path] = None,
) -> Tuple[Path, dict]:
    out_dir = ensure_dir(out_dir)

    fai = ensure_fai(genome_fasta)
    chrom_sizes = read_fai(fai)
    target_has_chr = chrom_prefix_mode_to_target(chrom_sizes)

    exclude = [x.strip() for x in str(exclude_chroms).split(",") if x.strip()]
    # Normalize exclude list to target chrom naming.
    exclude = [normalize_chrom(x, mode=chrom_prefix_mode, target_has_chr=target_has_chr) for x in exclude]

    chroms, w = _eligible_chroms(chrom_sizes, seq_len_bp=int(seq_len_bp), exclude=exclude)

    rng = np.random.default_rng(int(seed))
    chrom_idx = rng.choice(np.arange(len(chroms)), size=int(n_random), replace=True, p=w)

    recs: List[Bed6] = []
    for i, ci in enumerate(chrom_idx, start=1):
        chrom = chroms[int(ci)]
        L = int(chrom_sizes[chrom])
        max_start = int(L) - int(seq_len_bp)
        start0 = int(rng.integers(0, max_start + 1))
        end0 = int(start0) + int(seq_len_bp)
        recs.append(Bed6(chrom=chrom, start0=start0, end0=end0, name=f"random_{i:07d}", score=0, strand="."))

    before = len(recs)
    recs = list(filter_to_bounds(recs, chrom_sizes))
    if len(recs) != before:
        logger.warning(f"Filtered random windows to bounds: {len(recs)}/{before}")

    recs = sort_bed(recs, chrom_sizes)
    out_bed_path = Path(out_bed) if out_bed is not None else (out_dir / f"random_{int(seq_len_bp)}bp.bed")
    write_bed6(out_bed_path, recs)

    stats = {
        "n_requested": int(n_random),
        "n_written": int(len(recs)),
        "seq_len_bp": int(seq_len_bp),
        "seed": int(seed),
        "exclude_chroms": exclude,
        "n_chroms_eligible": int(len(chroms)),
    }
    logger.info(f"Wrote random windows: {out_bed_path} n={len(recs)}")
    return out_bed_path, stats


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_bed = args.out_bed
    if out_bed is None:
        out_bed = str(out_dir / "random_262kb.bed")
    bed, stats = build_random(
        genome_fasta=args.genome_fasta,
        out_dir=out_dir,
        seq_len_bp=int(args.seq_len_bp),
        seed=int(args.seed),
        n_random=int(args.n_random),
        exclude_chroms=str(args.exclude_chroms),
        chrom_prefix_mode=str(args.chrom_prefix_mode),
        out_bed=out_bed,
    )
    logger.info(stats)


if __name__ == "__main__":
    main()
