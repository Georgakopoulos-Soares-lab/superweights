#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rewrite FASTA headers to have unique, FIMO-safe IDs.")
    p.add_argument("--in_fa", required=True, help="Input FASTA")
    p.add_argument("--out_fa", required=True, help="Output FASTA with rewritten IDs")
    p.add_argument(
        "--map_tsv",
        required=True,
        help="TSV mapping file with columns: id\toriginal_header (without leading '>')",
    )
    p.add_argument("--prefix", default="seq", help="ID prefix; IDs become '<prefix><index:06d>'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_fa)
    out_path = Path(args.out_fa)
    map_path = Path(args.map_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    out_lines: list[str] = []
    map_lines: list[str] = ["id\toriginal_header\n"]

    seq_id = 0
    pending_header: str | None = None
    pending_seq: list[str] = []

    def flush() -> None:
        nonlocal seq_id, pending_header, pending_seq
        if pending_header is None:
            return
        seq_id += 1
        new_id = f"{args.prefix}{seq_id:06d}"
        header_clean = pending_header.strip()
        map_lines.append(f"{new_id}\t{header_clean}\n")
        out_lines.append(f">{new_id} {header_clean}\n")
        out_lines.append("".join(pending_seq).replace("\n", "").strip() + "\n")
        pending_header = None
        pending_seq = []

    with in_path.open() as f:
        for line in f:
            if line.startswith(">"):
                flush()
                pending_header = line[1:].rstrip("\n")
            else:
                pending_seq.append(line)
        flush()

    out_path.write_text("".join(out_lines))
    map_path.write_text("".join(map_lines))


if __name__ == "__main__":
    main()
