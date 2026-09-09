"""
analysis/probe_windows.py
--------------------------
Shared FASTA/BED window-sampling helpers for the held-out probe set used
throughout Tier 1+ of the mechanism study: N non-overlapping-ish windows of
window_bp length, drawn from real genomic regions (hg38 for EUK models,
E. coli K-12 for PROK models).

This factors out the duplicated logic in
scripts/interpretability/run_sw_gradient_attribution.py and
scripts/interpretability/run_sw_causal_tracing.py (both copy-pasted the same
_open_fasta / _fetch_window / _sample_bed_windows trio) so Tier 1's
channel-survival driver (and anything downstream) has one source of truth.
Behavior is unchanged from the original implementations.
"""
from __future__ import annotations

import random


def open_fasta(path: str):
    try:
        from pyfaidx import Fasta
        return Fasta(path, as_raw=True, sequence_always_upper=True)
    except ImportError:
        raise ImportError("pyfaidx required: pip install pyfaidx")


def fetch_window(fasta, chrom: str, center: int, window_bp: int) -> str | None:
    half = window_bp // 2
    start = max(0, center - half)
    end = start + window_bp
    key = (chrom if chrom in fasta
           else "chr" + chrom if "chr" + chrom in fasta
           else chrom.lstrip("chr") if chrom.lstrip("chr") in fasta
           else None)
    if key is None:
        return None
    chrom_len = len(fasta[key])
    if end > chrom_len:
        end = chrom_len
        start = max(0, end - window_bp)
    seq = str(fasta[key][start:end])
    if len(seq) < window_bp:
        seq = seq + "N" * (window_bp - len(seq))
    return seq


def sample_bed_windows(bed_path: str, fasta, n: int, window_bp: int,
                        rng: random.Random, label: str = "region") -> list:
    """Draw n windows centered on random BED rows, skipping high-N windows."""
    with open(bed_path) as fh:
        lines = [l.strip().split("\t") for l in fh
                 if l.strip() and not l.startswith("#")]
    candidate_pool = rng.sample(lines, min(len(lines), 10 * n))
    results = []
    for row in candidate_pool:
        if len(results) >= n:
            break
        chrom = row[0]
        s, e = int(row[1]), int(row[2])
        center = (s + e) // 2
        seq = fetch_window(fasta, chrom, center, window_bp)
        if seq is None:
            continue
        if seq.count("N") / len(seq) > 0.1:
            continue
        results.append({"label": label, "chrom": chrom, "center": center, "seq": seq})
    return results


def sample_probe_windows(kingdom: str, n: int = 30, window_bp: int = 3072,
                          seed: int = 42, root=None) -> list:
    """
    Sample n probe windows for the given kingdom ('euk' or 'prok'), pooling
    across the standard region BED files (promoters/enhancers/random for
    hg38; promoters/random for E. coli), split as evenly as possible.

    Returns a list of dicts: {"label", "chrom", "center", "seq"}.
    """
    from pathlib import Path
    if root is None:
        root = Path(__file__).resolve().parent.parent
    else:
        root = Path(root)
    rng = random.Random(seed)

    if kingdom == "euk":
        fasta_path = "/data/nvidia/data/hg38/hg38.fa"
        bed_files = [
            (str(root / "data/regions/hg38/promoters_262kb.bed"), "promoter"),
            (str(root / "data/regions/hg38/enhancers_ccre_262kb.bed"), "enhancer"),
            (str(root / "data/regions/hg38/random_262kb.bed"), "random"),
        ]
    elif kingdom == "prok":
        fasta_path = str(root / "data/reference/ecoli/ecoli_k12.fna")
        bed_files = [
            (str(root / "data/regions/ecoli/promoters_ecoli.bed"), "promoter"),
            (str(root / "data/regions/ecoli/random_ecoli.bed"), "random"),
        ]
    else:
        raise ValueError(f"Unknown kingdom '{kingdom}', expected 'euk' or 'prok'")

    fasta = open_fasta(fasta_path)
    per_file = -(-n // len(bed_files))  # ceil division, then trim to n
    windows = []
    for bed_path, label in bed_files:
        windows.extend(sample_bed_windows(bed_path, fasta, per_file, window_bp, rng, label))
    rng.shuffle(windows)
    return windows[:n]
