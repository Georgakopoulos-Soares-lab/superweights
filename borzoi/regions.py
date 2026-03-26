from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from loguru import logger

from borzoi.genome import Genome


@dataclass(frozen=True)
class Region:
    chrom: str
    start0: int
    end0: int
    strand: str = "."
    region_set: str = "regions"

    @property
    def width(self) -> int:
        return int(self.end0) - int(self.start0)


def _open_text_maybe_gzip(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def read_bed(path: str | Path, *, region_set: str, max_rows: Optional[int] = None) -> List[Region]:
    """Read a BED3/BED6 file.

    Accepts BED3: chrom start end
    Accepts BED6: chrom start end name score strand

    Ignores header/comment lines starting with '#'.
    """
    regions: List[Region] = []
    with _open_text_maybe_gzip(path) as f:
        for line in f:
            if max_rows is not None and len(regions) >= int(max_rows):
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start0 = int(parts[1])
                end0 = int(parts[2])
            except Exception:
                continue
            strand = "."
            if len(parts) >= 6 and parts[5] in {"+", "-", "."}:
                strand = parts[5]
            if end0 <= start0:
                continue
            regions.append(Region(chrom=str(chrom), start0=int(start0), end0=int(end0), strand=strand, region_set=str(region_set)))

    logger.info(f"Loaded BED: {path} set={region_set} n={len(regions)}")
    return regions


def resize_to_width(r: Region, width: int) -> Region:
    """Center-resize a region to exactly `width` bp."""
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")

    mid = (int(r.start0) + int(r.end0)) // 2
    start0 = int(mid) - (width // 2)
    end0 = int(start0) + width
    return Region(chrom=r.chrom, start0=start0, end0=end0, strand=r.strand, region_set=r.region_set)


def filter_in_bounds(regions: Sequence[Region], genome: Genome) -> List[Region]:
    """Keep only regions fully contained within chromosome bounds."""
    kept: List[Region] = []
    skipped = 0
    for r in regions:
        try:
            chrom_len = genome.chrom_length(r.chrom)
        except KeyError:
            skipped += 1
            continue
        if r.start0 < 0 or r.end0 > int(chrom_len):
            skipped += 1
            continue
        kept.append(r)
    if skipped:
        logger.info(f"Skipped out-of-bounds regions: {skipped}/{len(regions)}")
    return kept


def sample_regions(regions: Sequence[Region], n: int, *, seed: int) -> List[Region]:
    n = int(n)
    if n <= 0:
        return []
    if len(regions) <= n:
        return list(regions)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(np.arange(len(regions)), size=n, replace=False)
    return [regions[int(i)] for i in idx.tolist()]


def prepare_region_set(
    bed_path: str | Path,
    *,
    genome: Genome,
    region_set: str,
    width: int,
    n: int,
    seed: int,
) -> List[Region]:
    regs = read_bed(bed_path, region_set=region_set)
    regs = [resize_to_width(r, width=int(width)) for r in regs]
    regs = filter_in_bounds(regs, genome)
    regs = sample_regions(regs, n=int(n), seed=int(seed))
    return regs


def iter_region_batches(regions: Sequence[Region], batch_size: int) -> Iterable[List[Region]]:
    bs = int(batch_size)
    if bs <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(regions), bs):
        yield list(regions[i : i + bs])


__all__ = [
    "Region",
    "read_bed",
    "resize_to_width",
    "filter_in_bounds",
    "sample_regions",
    "prepare_region_set",
    "iter_region_batches",
]
