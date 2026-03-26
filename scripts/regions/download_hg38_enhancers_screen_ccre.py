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
from typing import Dict, Iterator, List, Optional, Tuple

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


_DEFAULT_CCRE_URLS = [
    # Weng Lab Registry mirror (works without SCREEN UI redirects).
    "https://downloads.wenglab.org/Registry-V3/GRCh38-cCREs.bed",
    # SCREEN download endpoint currently serves HTML for .bed.gz on some networks.
    "https://screen.encodeproject.org/downloads/GRCh38-cCREs.bed.gz",
]


def _looks_like_gzip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except Exception:
        return False


def _validate_ccre_download(path: Path, url: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Download missing: {path}")
    if str(path).endswith(".gz") and not _looks_like_gzip(path):
        # Common failure mode: SCREEN endpoint returns an HTML landing page.
        with open(path, "rb") as f:
            head = f.read(64)
        raise RuntimeError(
            "Downloaded cCRE file does not look like gzip (likely HTML redirect/landing page). "
            f"url={url} path={path} head={head!r}. "
            "Provide --ccre_url pointing to a direct BED download (e.g. Weng Lab Registry)."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ENCODE SCREEN cCREs and build fixed-window enhancer BEDs.")
    p.add_argument("--genome_fasta", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="data/regions/hg38")
    p.add_argument("--raw_dir", type=str, default=None)
    p.add_argument(
        "--ccre_url",
        type=str,
        default=None,
        help="Direct URL to GRCh38-cCREs.bed.gz. If omitted, tries a small list of known mirrors.",
    )
    p.add_argument("--seq_len_bp", type=int, default=262144)
    p.add_argument("--chrom_prefix_mode", type=str, default="auto", choices=["auto", "add_chr", "strip_chr"])
    p.add_argument(
        "--keep_classes",
        type=str,
        default="ELS,dELS,pELS",
        help="Comma-separated class tokens to keep (matched as substring across fields).",
    )
    p.add_argument("--dedup", action="store_true", help="Deduplicate identical windows (chrom,start,end)")
    p.add_argument("--max_rows", type=int, default=None, help="Optional cap on number of parsed rows (for smoke tests)")
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--out_bed", type=str, default=None)
    return p.parse_args()


def _iter_ccre_enhancers(
    ccre_bed_gz: str | Path,
    *,
    chrom_sizes: Dict[str, int],
    target_has_chr: bool,
    chrom_prefix_mode: str,
    seq_len_bp: int,
    keep_tokens: List[str],
    max_rows: Optional[int] = None,
) -> Iterator[Bed6]:
    n = 0
    with open_text_maybe_gzip(ccre_bed_gz) as f:
        for line in f:
            if max_rows is not None and n >= int(max_rows):
                break
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = normalize_chrom(parts[0], mode=chrom_prefix_mode, target_has_chr=target_has_chr)
            if chrom not in chrom_sizes:
                continue
            try:
                start0 = int(parts[1])
                end0 = int(parts[2])
            except Exception:
                continue
            if end0 <= start0:
                continue

            # SCREEN cCRE beds vary a bit. We'll accept rows that contain any keep token
            # as a substring in any field (case-sensitive to match canonical labels).
            row_text = "\t".join(parts)
            if keep_tokens and not any(tok in row_text for tok in keep_tokens):
                continue

            center0 = (start0 + end0) // 2
            win_start0, win_end0 = center_to_window(int(center0), int(seq_len_bp))
            name = parts[3] if len(parts) >= 4 and parts[3] else f"ccre_row{n}"
            yield Bed6(chrom=chrom, start0=win_start0, end0=win_end0, name=name, score=0, strand=".")
            n += 1


def build_enhancers(
    *,
    genome_fasta: str | Path,
    out_dir: str | Path,
    raw_dir: Optional[str | Path] = None,
    ccre_url: Optional[str] = None,
    seq_len_bp: int = 262144,
    chrom_prefix_mode: str = "auto",
    keep_classes: str = "ELS,dELS,pELS",
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

    url_candidates = [ccre_url] if ccre_url else list(_DEFAULT_CCRE_URLS)
    url_candidates = [u for u in url_candidates if u]
    if not url_candidates:
        raise RuntimeError("No cCRE URL candidates")

    # Choose output filename by URL suffix, but keep it stable.
    bed_path = raw_dir_p / ("GRCh38-cCREs.bed.gz" if str(url_candidates[0]).endswith(".gz") else "GRCh38-cCREs.bed")
    last_err = None
    for u in url_candidates:
        try:
            # If URL suffix differs from current bed_path, adjust output name.
            desired = raw_dir_p / ("GRCh38-cCREs.bed.gz" if str(u).endswith(".gz") else "GRCh38-cCREs.bed")
            download_url(str(u), desired, force=force_download)
            _validate_ccre_download(desired, str(u))
            bed_path = desired
            last_err = None
            break
        except Exception as e:
            last_err = e
            logger.warning(f"Download failed for {u}: {e}")
    if last_err is not None and not bed_path.exists():
        raise RuntimeError(
            f"Failed to download cCRE bed from all candidates. Provide --ccre_url explicitly. Last error: {last_err}"
        )

    keep_tokens = [t.strip() for t in str(keep_classes).split(",") if t.strip()]

    recs = list(
        _iter_ccre_enhancers(
            bed_path,
            chrom_sizes=chrom_sizes,
            target_has_chr=target_has_chr,
            chrom_prefix_mode=str(chrom_prefix_mode),
            seq_len_bp=int(seq_len_bp),
            keep_tokens=keep_tokens,
            max_rows=max_rows,
        )
    )
    before = len(recs)
    recs = list(filter_to_bounds(recs, chrom_sizes))

    if dedup:
        uniq = {}
        for r in recs:
            key = (r.chrom, int(r.start0), int(r.end0))
            if key not in uniq:
                uniq[key] = r
        recs = list(uniq.values())

    recs = sort_bed(recs, chrom_sizes)

    out_records: List[Bed6] = []
    for i, r in enumerate(recs, start=1):
        name = f"ccre_ELS{i:07d}"
        out_records.append(Bed6(chrom=r.chrom, start0=r.start0, end0=r.end0, name=name, score=0, strand="."))

    out_bed_path = Path(out_bed) if out_bed is not None else (out_dir / f"enhancers_ccre_{int(seq_len_bp)}bp.bed")
    write_bed6(out_bed_path, out_records)

    stats = {
        "n_parsed_rows": int(before),
        "n_in_bounds": int(len(recs)),
        "n_written": int(len(out_records)),
        "seq_len_bp": int(seq_len_bp),
        "keep_classes": str(keep_classes),
        "dedup": bool(dedup),
        "download_url_used": str(url_candidates[0] if ccre_url else "(auto list)"),
    }
    logger.info(f"Wrote enhancers: {out_bed_path} n={len(out_records)}")
    return out_bed_path, stats


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_bed = args.out_bed
    if out_bed is None:
        out_bed = str(out_dir / "enhancers_ccre_262kb.bed")

    bed, stats = build_enhancers(
        genome_fasta=args.genome_fasta,
        out_dir=out_dir,
        raw_dir=args.raw_dir,
        ccre_url=args.ccre_url,
        seq_len_bp=int(args.seq_len_bp),
        chrom_prefix_mode=str(args.chrom_prefix_mode),
        keep_classes=str(args.keep_classes),
        dedup=bool(args.dedup) or True,
        max_rows=args.max_rows,
        force_download=bool(args.force_download),
        out_bed=out_bed,
    )
    logger.info(stats)


if __name__ == "__main__":
    main()
