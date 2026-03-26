#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rewrite a FASTA file with short, whitespace-free sequence IDs suitable for FIMO, and write a TSV map "
            "from sanitized id -> original FASTA header."
        )
    )
    p.add_argument("--in_fa", required=True, help="Input FASTA")
    p.add_argument("--out_fa", required=True, help="Output FASTA with sanitized IDs")
    p.add_argument("--map_tsv", required=True, help="Output mapping TSV with columns: id, original_header")
    p.add_argument("--prefix", default="s", help="Sequence id prefix (default: s)")
    p.add_argument("--max_records", type=int, default=None, help="Optionally cap the number of records")
    return p.parse_args()


def _read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: str | None = None
    seq_chunks: List[str] = []

    with path.open("r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())

    if header is not None:
        records.append((header, "".join(seq_chunks)))
    return records


def _write_fasta(path: Path, records: Iterable[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for header, seq in records:
            f.write(f">{header}\n")
            # keep single-line sequences (fine for MEME-suite)
            f.write(seq + "\n")


def main() -> None:
    args = parse_args()

    in_fa = Path(args.in_fa)
    out_fa = Path(args.out_fa)
    map_tsv = Path(args.map_tsv)

    records = _read_fasta(in_fa)
    if args.max_records is not None:
        records = records[: int(args.max_records)]

    sanitized: List[Tuple[str, str]] = []
    map_rows: List[Tuple[str, str]] = []
    for i, (orig_header, seq) in enumerate(records):
        sid = f"{args.prefix}{i}"  # no whitespace, short
        sanitized.append((sid, seq))
        map_rows.append((sid, orig_header))

    _write_fasta(out_fa, sanitized)

    map_tsv.parent.mkdir(parents=True, exist_ok=True)
    with map_tsv.open("w") as f:
        f.write("id\toriginal_header\n")
        for sid, orig_header in map_rows:
            f.write(f"{sid}\t{orig_header}\n")

    print(f"wrote fasta={out_fa} records={len(sanitized)}")
    print(f"wrote map={map_tsv} rows={len(map_rows)}")


if __name__ == "__main__":
    main()
