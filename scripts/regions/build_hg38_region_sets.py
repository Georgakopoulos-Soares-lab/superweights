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
from typing import Dict, Optional

from loguru import logger

from scripts.regions._common import atomic_write_json, ensure_dir, now_timestamp, sha256_file
from scripts.regions.download_hg38_enhancers_screen_ccre import build_enhancers
from scripts.regions.download_hg38_promoters_gencode import build_promoters
from scripts.regions.make_hg38_random_windows import build_random


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build hg38 promoter/enhancer/random 262kb BED sets for the superactivations scan.")
    p.add_argument("--genome_fasta", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="data/regions/hg38")
    p.add_argument("--seq_len_bp", type=int, default=262144)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n_random", type=int, default=50000)
    p.add_argument("--exclude_chroms", type=str, default="chrM")
    p.add_argument("--chrom_prefix_mode", type=str, default="auto", choices=["auto", "add_chr", "strip_chr"])

    # download urls override
    p.add_argument("--gencode_gtf_url", type=str, default=None)
    p.add_argument("--ccre_url", type=str, default=None)

    # toggles
    p.add_argument("--force_download", action="store_true")
    p.add_argument("--bgzip_tabix", action="store_true", help="Optionally bgzip+tabix the resulting BEDs if tools exist")

    # dev/smoke
    p.add_argument("--max_promoters", type=int, default=None)
    p.add_argument("--max_enhancers", type=int, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    raw_dir = ensure_dir(out_dir / "raw")

    logger.info(f"Building region sets under: {out_dir}")

    # Build promoters
    promoters_bed, promoters_stats = build_promoters(
        genome_fasta=args.genome_fasta,
        out_dir=out_dir,
        raw_dir=raw_dir,
        gencode_gtf_url=args.gencode_gtf_url
        or "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz",
        seq_len_bp=int(args.seq_len_bp),
        chrom_prefix_mode=str(args.chrom_prefix_mode),
        feature="transcript",
        dedup=True,
        max_rows=args.max_promoters,
        force_download=bool(args.force_download),
        out_bed=out_dir / "promoters_262kb.bed",
    )

    # Build enhancers
    enhancers_bed, enhancers_stats = build_enhancers(
        genome_fasta=args.genome_fasta,
        out_dir=out_dir,
        raw_dir=raw_dir,
        ccre_url=args.ccre_url,
        seq_len_bp=int(args.seq_len_bp),
        chrom_prefix_mode=str(args.chrom_prefix_mode),
        keep_classes="ELS,dELS,pELS",
        dedup=True,
        max_rows=args.max_enhancers,
        force_download=bool(args.force_download),
        out_bed=out_dir / "enhancers_ccre_262kb.bed",
    )

    # Build random
    random_bed, random_stats = build_random(
        genome_fasta=args.genome_fasta,
        out_dir=out_dir,
        seq_len_bp=int(args.seq_len_bp),
        seed=int(args.seed),
        n_random=int(args.n_random),
        exclude_chroms=str(args.exclude_chroms),
        chrom_prefix_mode=str(args.chrom_prefix_mode),
        out_bed=out_dir / "random_262kb.bed",
    )

    # Raw input checksums (best-effort; files should exist after successful build).
    raw_inputs = {}
    gtf_path = raw_dir / "gencode.v44.annotation.gtf.gz"
    if gtf_path.exists():
        raw_inputs["gencode_v44_gtf_gz"] = {"path": str(gtf_path), "sha256": sha256_file(gtf_path)}
    ccre_bed = raw_dir / "GRCh38-cCREs.bed"
    ccre_gz = raw_dir / "GRCh38-cCREs.bed.gz"
    if ccre_bed.exists():
        raw_inputs["ccres_bed"] = {"path": str(ccre_bed), "sha256": sha256_file(ccre_bed)}
    if ccre_gz.exists():
        raw_inputs["ccres_bed_gz"] = {"path": str(ccre_gz), "sha256": sha256_file(ccre_gz)}

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "timestamp": now_timestamp(),
        "genome_fasta": str(args.genome_fasta),
        "raw_inputs": raw_inputs,
        "params": {
            "seq_len_bp": int(args.seq_len_bp),
            "seed": int(args.seed),
            "n_random": int(args.n_random),
            "exclude_chroms": str(args.exclude_chroms),
            "chrom_prefix_mode": str(args.chrom_prefix_mode),
        },
        "sources": {
            "gencode_gtf_url": str(args.gencode_gtf_url)
            if args.gencode_gtf_url
            else "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz",
            "ccre_url": str(args.ccre_url) if args.ccre_url else "(auto mirror list)",
        },
        "outputs": {
            "promoters_262kb": {
                "path": str(promoters_bed),
                "sha256": sha256_file(promoters_bed),
                "stats": promoters_stats,
            },
            "enhancers_ccre_262kb": {
                "path": str(enhancers_bed),
                "sha256": sha256_file(enhancers_bed),
                "stats": enhancers_stats,
            },
            "random_262kb": {
                "path": str(random_bed),
                "sha256": sha256_file(random_bed),
                "stats": random_stats,
            },
        },
    }

    atomic_write_json(manifest_path, manifest)
    logger.info(f"Wrote manifest: {manifest_path}")

    if bool(args.bgzip_tabix):
        from scripts.regions._common import maybe_bgzip_and_tabix

        maybe_bgzip_and_tabix(promoters_bed)
        maybe_bgzip_and_tabix(enhancers_bed)
        maybe_bgzip_and_tabix(random_bed)

    logger.info("Done.")


if __name__ == "__main__":
    main()
