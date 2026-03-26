from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from loguru import logger


@dataclass(frozen=True)
class Bed6:
    chrom: str
    start0: int
    end0: int
    name: str
    score: int = 0
    strand: str = "."

    @property
    def width(self) -> int:
        return int(self.end0) - int(self.start0)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(int(chunk_bytes))
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def download_url(url: str, out_path: str | Path, *, force: bool = False, timeout_s: int = 60) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    if out_path.exists() and not force:
        logger.info(f"Download exists, skipping: {out_path}")
        return out_path

    logger.info(f"Downloading: {url} -> {out_path}")
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r, open(tmp, "wb") as w:
            shutil.copyfileobj(r, w)
        os.replace(tmp, out_path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
    return out_path


def open_text_maybe_gzip(path: str | Path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def read_fai(fai_path: str | Path) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    with open(fai_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            chrom = parts[0]
            try:
                length = int(parts[1])
            except Exception:
                continue
            sizes[str(chrom)] = int(length)
    if not sizes:
        raise RuntimeError(f"No chrom sizes parsed from fai: {fai_path}")
    return sizes


def ensure_fai(genome_fasta: str | Path) -> Path:
    fasta = Path(genome_fasta)
    fai = fasta.with_suffix(fasta.suffix + ".fai")
    if fai.exists():
        return fai

    logger.warning(f"FASTA index not found: {fai}")
    samtools = shutil.which("samtools")
    if samtools is None:
        raise RuntimeError(
            f"Missing FASTA index {fai}. Create it with 'samtools faidx {fasta}' or provide --chrom_sizes."
        )

    logger.info(f"Indexing FASTA with samtools faidx: {fasta}")
    subprocess.run([samtools, "faidx", str(fasta)], check=True)
    if not fai.exists():
        raise RuntimeError(f"samtools faidx did not create: {fai}")
    return fai


def chrom_prefix_mode_to_target(chrom_sizes: Dict[str, int]) -> bool:
    # True if targets use 'chr' prefix.
    for k in chrom_sizes.keys():
        return str(k).startswith("chr")
    return True


def normalize_chrom(chrom: str, *, mode: str, target_has_chr: bool) -> str:
    c = str(chrom)
    has_chr = c.startswith("chr")
    mode = str(mode)
    if mode not in {"auto", "add_chr", "strip_chr"}:
        raise ValueError(f"Invalid chrom_prefix_mode: {mode}")

    if mode == "add_chr":
        return c if has_chr else ("chr" + c)
    if mode == "strip_chr":
        return c[3:] if has_chr else c

    # auto: adapt to target
    if target_has_chr:
        return c if has_chr else ("chr" + c)
    return c[3:] if has_chr else c


def center_to_window(center0: int, seq_len: int) -> Tuple[int, int]:
    half = int(seq_len) // 2
    start0 = int(center0) - half
    end0 = start0 + int(seq_len)
    return int(start0), int(end0)


def filter_to_bounds(records: Iterable[Bed6], chrom_sizes: Dict[str, int]) -> Iterator[Bed6]:
    for r in records:
        if r.chrom not in chrom_sizes:
            continue
        L = int(chrom_sizes[r.chrom])
        if r.start0 < 0 or r.end0 > L:
            continue
        if r.end0 <= r.start0:
            continue
        yield r


def sort_bed(records: Sequence[Bed6], chrom_sizes: Dict[str, int]) -> List[Bed6]:
    chrom_order = {c: i for i, c in enumerate(chrom_sizes.keys())}

    def key(r: Bed6):
        return (chrom_order.get(r.chrom, 10**9), int(r.start0), int(r.end0), str(r.strand))

    return sorted(list(records), key=key)


def write_bed6(path: str | Path, records: Sequence[Bed6]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(
                "\t".join(
                    [
                        str(r.chrom),
                        str(int(r.start0)),
                        str(int(r.end0)),
                        str(r.name),
                        str(int(r.score)),
                        str(r.strand),
                    ]
                )
                + "\n"
            )
    os.replace(tmp, path)


def maybe_bgzip_and_tabix(bed_path: str | Path, *, force: bool = False) -> Optional[Tuple[Path, Path]]:
    bed_path = Path(bed_path)
    bgzip = shutil.which("bgzip")
    tabix = shutil.which("tabix")
    if bgzip is None or tabix is None:
        logger.info("bgzip/tabix not available; skipping compression/index")
        return None

    gz = bed_path.with_suffix(bed_path.suffix + ".gz")
    tbi = gz.with_suffix(gz.suffix + ".tbi")
    if gz.exists() and tbi.exists() and not force:
        return gz, tbi

    logger.info(f"bgzip: {bed_path} -> {gz}")
    subprocess.run([bgzip, "-f", "-c", str(bed_path)], check=True, stdout=open(gz, "wb"))
    logger.info(f"tabix: {gz}")
    subprocess.run([tabix, "-f", "-p", "bed", str(gz)], check=True)
    if not gz.exists() or not tbi.exists():
        raise RuntimeError("bgzip/tabix did not produce expected outputs")
    return gz, tbi


def now_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
