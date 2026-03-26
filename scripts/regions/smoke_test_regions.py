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
from typing import Dict, List

from loguru import logger

from scripts.regions._common import ensure_fai, open_text_maybe_gzip, read_fai


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test region BEDs against a genome FASTA index.")
    p.add_argument("--genome_fasta", type=str, required=True)
    p.add_argument("--regions_dir", type=str, default="data/regions/hg38")
    p.add_argument("--seq_len_bp", type=int, default=262144)
    return p.parse_args()


def _read_bed_lines(path: Path, n: int = 3) -> List[str]:
    out: List[str] = []
    with open_text_maybe_gzip(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            out.append(line.rstrip("\n"))
            if len(out) >= n:
                break
    return out


def _validate_bed(path: Path, chrom_sizes: Dict[str, int], seq_len_bp: int) -> int:
    n = 0
    bad = 0
    with open_text_maybe_gzip(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom not in chrom_sizes:
                bad += 1
                continue
            try:
                start0 = int(parts[1])
                end0 = int(parts[2])
            except Exception:
                bad += 1
                continue
            if end0 - start0 != int(seq_len_bp):
                bad += 1
                continue
            if start0 < 0 or end0 > int(chrom_sizes[chrom]):
                bad += 1
                continue
            n += 1
    if bad:
        raise RuntimeError(f"Validation failed for {path}: bad={bad} good={n}")
    return n


def main() -> None:
    args = parse_args()
    fai = ensure_fai(args.genome_fasta)
    chrom_sizes = read_fai(fai)

    regions_dir = Path(args.regions_dir)
    beds = {
        "promoters": regions_dir / "promoters_262kb.bed",
        "enhancers": regions_dir / "enhancers_ccre_262kb.bed",
        "random": regions_dir / "random_262kb.bed",
    }

    for name, path in beds.items():
        if not path.exists():
            raise SystemExit(f"Missing BED: {path}")
        n = _validate_bed(path, chrom_sizes, int(args.seq_len_bp))
        logger.info(f"{name}: n={n} path={path}")
        for l in _read_bed_lines(path, n=3):
            print(f"{name}\t{l}")

    logger.info("Smoke test OK")


if __name__ == "__main__":
    main()
