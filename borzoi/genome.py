from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger
from pyfaidx import Fasta


def _try_match_chrom(fasta: Fasta, chrom: str) -> Optional[str]:
    """Return a chrom key that exists in FASTA, or None."""
    if chrom in fasta:
        return chrom
    if chrom.lower().startswith("chr"):
        alt = chrom[3:]
        if alt in fasta:
            return alt
    else:
        alt = "chr" + chrom
        if alt in fasta:
            return alt
    # mitochondrial common variants
    if chrom in {"M", "MT", "chrM", "chrMT"}:
        for cand in ("MT", "M", "chrM", "chrMT"):
            if cand in fasta:
                return cand
    return None


@dataclass
class Genome:
    fasta_path: Path
    as_raw: bool = True

    def __post_init__(self) -> None:
        self.fasta_path = Path(self.fasta_path)
        self._fasta: Optional[Fasta] = None

    @property
    def fasta(self) -> Fasta:
        if self._fasta is None:
            self._fasta = Fasta(str(self.fasta_path), as_raw=self.as_raw, sequence_always_upper=True)
        return self._fasta

    def resolve_chrom_key(self, chrom: str) -> str:
        key = _try_match_chrom(self.fasta, str(chrom))
        if key is None:
            raise KeyError(f"chrom not found in FASTA: {chrom}")
        return key

    def chrom_length(self, chrom: str) -> int:
        key = self.resolve_chrom_key(chrom)
        return int(len(self.fasta[key]))

    def fetch_sequence(self, chrom: str, start0: int, end0: int) -> str:
        if start0 < 0:
            raise ValueError(f"start0 < 0: {start0}")
        if end0 <= start0:
            raise ValueError(f"end0 <= start0: {end0} <= {start0}")

        key = self.resolve_chrom_key(chrom)

        # pyfaidx slices are 0-based, end-exclusive
        seq = self.fasta[key][start0:end0]
        return str(seq)


def make_ref_alt_sequence(
    genome: Genome,
    chrom: str,
    pos1: int,
    ref: str,
    alt: str,
    seq_len: int,
    verify_ref: bool = True,
) -> Tuple[str, str]:
    """Return (ref_seq, alt_seq) of exactly seq_len, centered on pos1.

    Assumes SNPs (len(ref)==len(alt)==1). Coordinates:
    - pos1 is 1-based genomic position of the SNP.
    - We build a window [start0, end0) of length seq_len.
    """
    if len(ref) != 1 or len(alt) != 1:
        raise ValueError("Only SNPs supported")

    center0 = int(pos1) - 1
    half = seq_len // 2
    start0 = center0 - half
    end0 = start0 + seq_len

    # Handle chromosome ends by padding with N. This keeps SNP centered while allowing
    # variants close to chrom starts/ends.
    chrom_len = genome.chrom_length(chrom)
    if center0 < 0 or center0 >= chrom_len:
        raise ValueError(f"Variant position out of bounds: {chrom}:{pos1} (chrom_len={chrom_len})")

    pad_left = max(0, -start0)
    pad_right = max(0, end0 - chrom_len)
    fetch_start0 = max(0, start0)
    fetch_end0 = min(end0, chrom_len)

    seq_mid = genome.fetch_sequence(chrom=chrom, start0=fetch_start0, end0=fetch_end0)
    seq = ("N" * pad_left) + seq_mid + ("N" * pad_right)
    if len(seq) != seq_len:
        raise ValueError(f"Padded FASTA window wrong length: {len(seq)} != {seq_len}")

    offset = center0 - start0
    base = seq[offset : offset + 1]

    if verify_ref:
        if base.upper() != ref.upper():
            raise ValueError(f"Reference mismatch at {chrom}:{pos1}: FASTA={base} expected_ref={ref}")

    ref_seq = seq
    alt_seq = seq[:offset] + alt.upper() + seq[offset + 1 :]
    return ref_seq, alt_seq


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


__all__ = [
    "Genome",
    "make_ref_alt_sequence",
    "fetch_with_padding",
    "window_start0",
    "window_end0",
    "variant_center0_in_window",
    "bin_size_bp",
    "bin_to_genome_coords",
]
