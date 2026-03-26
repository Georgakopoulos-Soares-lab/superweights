from __future__ import annotations

from pathlib import Path
from typing import Tuple

from superweights_borzoi.genome import Genome, make_ref_alt_sequence


__all__ = [
    "Genome",
    "make_ref_alt_sequence",
    "fetch_with_padding",
]


def fetch_with_padding(genome: Genome, chrom: str, start0: int, end0: int, pad_base: str = "N") -> str:
    """Fetch [start0,end0) with N-padding if it runs off chromosome ends."""
    if end0 <= start0:
        raise ValueError("end0 <= start0")

    chrom_len = genome.chrom_length(chrom)
    pad_left = max(0, -start0)
    pad_right = max(0, end0 - chrom_len)
    fetch_start0 = max(0, start0)
    fetch_end0 = min(end0, chrom_len)

    seq_mid = genome.fetch_sequence(chrom=chrom, start0=fetch_start0, end0=fetch_end0)
    return (pad_base * pad_left) + seq_mid + (pad_base * pad_right)


def window_start0(pos1: int, seq_len: int) -> int:
    center0 = int(pos1) - 1
    return center0 - (seq_len // 2)


def window_end0(pos1: int, seq_len: int) -> int:
    return window_start0(pos1, seq_len) + int(seq_len)


def variant_center0_in_window(pos1: int, seq_len: int) -> int:
    return int(pos1) - 1 - window_start0(pos1, seq_len)


def bin_size_bp(seq_len: int, n_bins: int) -> float:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    return float(seq_len) / float(n_bins)


def bin_to_genome_coords(window_start0: int, bin_idx: int, bin_bp: float) -> Tuple[int, int, int]:
    """Return (start0,end0,center1) for a given bin index."""
    start0 = int(window_start0 + int(bin_idx * bin_bp))
    end0 = int(window_start0 + int((bin_idx + 1) * bin_bp))
    if end0 <= start0:
        end0 = start0 + 1
    center0 = (start0 + end0) // 2
    center1 = center0 + 1
    return start0, end0, center1
