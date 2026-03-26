#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Allow running from any cwd.
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from loguru import logger

from scripts.regions._common import (
    Bed6,
    center_to_window,
    chrom_prefix_mode_to_target,
    download_url,
    ensure_dir,
    ensure_fai,
    filter_to_bounds,
    normalize_chrom,
    open_text_maybe_gzip,
    read_fai,
    sort_bed,
    write_bed6,
)


_DEFAULT_GENCODE_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download GENCODE v44 and build fixed-window promoter BEDs (TSS-centered).")
    p.add_argument("--genome_fasta", type=str, required=True, help="hg38/GRCh38 FASTA (used for chrom sizes via .fai)")
    p.add_argument("--out_dir", type=str, default="data/regions/hg38", help="Output directory")
    p.add_argument("--raw_dir", type=str, default=None, help="Directory for downloaded raw inputs")
    p.add_argument("--gencode_gtf_url", type=str, default=_DEFAULT_GENCODE_GTF_URL)
    p.add_argument("--seq_len_bp", type=int, default=262144)
    p.add_argument("--chrom_prefix_mode", type=str, default="auto", choices=["auto", "add_chr", "strip_chr"])
    p.add_argument(
        "--feature",
        type=str,
        default="transcript",
        choices=["transcript", "gene"],
        help="Which GTF feature type to use for TSS extraction.",
    )
    p.add_argument("--dedup", action="store_true", help="Deduplicate identical windows (chrom,start,end,strand)")
    p.add_argument("--max_rows", type=int, default=None, help="Optional cap on number of parsed features (for smoke tests)")
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--out_bed", type=str, default=None, help="Override output BED path")
    return p.parse_args()


_ATTR_RE = re.compile(r"(\S+)\s+\"([^\"]+)\";")


def _parse_gtf_attrs(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for m in _ATTR_RE.finditer(s):
        out[m.group(1)] = m.group(2)
    return out


def iter_tss_from_gtf(
    gtf_gz_path: str | Path,
    *,
    chrom_sizes: Dict[str, int],
    target_has_chr: bool,
    chrom_prefix_mode: str,
    feature: str,
    seq_len_bp: int,
    max_rows: Optional[int] = None,
) -> Iterator[Bed6]:
    n = 0
    kept = 0
    with open_text_maybe_gzip(gtf_gz_path) as f:
        for line in f:
            if max_rows is not None and n >= int(max_rows):
                break
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != feature:
                continue

            chrom = normalize_chrom(parts[0], mode=chrom_prefix_mode, target_has_chr=target_has_chr)
            if chrom not in chrom_sizes:
                continue

            try:
                start1 = int(parts[3])
                end1 = int(parts[4])
            except Exception:
                continue

            strand = parts[6]
            if strand not in {"+", "-"}:
                continue

            attrs = _parse_gtf_attrs(parts[8])
            tx_id = attrs.get("transcript_id") or attrs.get("gene_id") or f"row{n}"

            # GTF positions are 1-based inclusive.
            tss1 = start1 if strand == "+" else end1
            center0 = int(tss1) - 1
            win_start0, win_end0 = center_to_window(center0, int(seq_len_bp))

            name = f"promoter_{feature}_{tx_id}"
            # We'll re-name sequentially after dedup/sort to keep compact and stable.
            r = Bed6(chrom=chrom, start0=win_start0, end0=win_end0, name=name, score=0, strand=strand)
            n += 1
            kept += 1
            yield r


def build_promoters(
    *,
    genome_fasta: str | Path,
    out_dir: str | Path,
    raw_dir: Optional[str | Path] = None,
    gencode_gtf_url: str = _DEFAULT_GENCODE_GTF_URL,
    seq_len_bp: int = 262144,
    chrom_prefix_mode: str = "auto",
    feature: str = "transcript",
    dedup: bool = True,
    max_rows: Optional[int] = None,
    force_download: bool = False,
    out_bed: Optional[str | Path] = None,
) -> Tuple[Path, Dict[str, int]]:
    out_dir = ensure_dir(out_dir)
    raw_dir_p = ensure_dir(raw_dir or (out_dir / "raw"))

    fai = ensure_fai(genome_fasta)
    chrom_sizes = read_fai(fai)
    target_has_chr = chrom_prefix_mode_to_target(chrom_sizes)

    gtf_path = raw_dir_p / "gencode.v44.annotation.gtf.gz"
    download_url(gencode_gtf_url, gtf_path, force=force_download)

    records = list(
        iter_tss_from_gtf(
            gtf_path,
            chrom_sizes=chrom_sizes,
            target_has_chr=target_has_chr,
            chrom_prefix_mode=chrom_prefix_mode,
            feature=feature,
            seq_len_bp=int(seq_len_bp),
            max_rows=max_rows,
        )
    )

    before = len(records)
    records = list(filter_to_bounds(records, chrom_sizes))

    if dedup:
        uniq = {}
        for r in records:
            key = (r.chrom, int(r.start0), int(r.end0), str(r.strand))
            if key not in uniq:
                uniq[key] = r
        records = list(uniq.values())

    records = sort_bed(records, chrom_sizes)

    # Re-name sequentially for portability.
    out_records: List[Bed6] = []
    for i, r in enumerate(records, start=1):
        name = f"promoter_TSS{i:07d}"
        out_records.append(Bed6(chrom=r.chrom, start0=r.start0, end0=r.end0, name=name, score=0, strand=r.strand))

    for r in out_records[:3]:
        logger.info(f"Example promoter window: {r}")

    out_bed_path = Path(out_bed) if out_bed is not None else (out_dir / f"promoters_{int(seq_len_bp)}bp.bed")
    write_bed6(out_bed_path, out_records)

    stats = {
        "n_parsed_features": int(before),
        "n_in_bounds": int(len(records)),
        "n_written": int(len(out_records)),
        "seq_len_bp": int(seq_len_bp),
        "feature": str(feature),
        "dedup": bool(dedup),
    }
    logger.info(f"Wrote promoters: {out_bed_path} n={len(out_records)}")
    return out_bed_path, stats


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_bed = args.out_bed
    if out_bed is None:
        out_bed = str(out_dir / "promoters_262kb.bed")

    bed, stats = build_promoters(
        genome_fasta=args.genome_fasta,
        out_dir=out_dir,
        raw_dir=args.raw_dir,
        gencode_gtf_url=args.gencode_gtf_url,
        seq_len_bp=int(args.seq_len_bp),
        chrom_prefix_mode=str(args.chrom_prefix_mode),
        feature=str(args.feature),
        dedup=bool(args.dedup) or True,
        max_rows=args.max_rows,
        force_download=bool(args.force_download),
        out_bed=out_bed,
    )
    logger.info(stats)


if __name__ == "__main__":
    main()
