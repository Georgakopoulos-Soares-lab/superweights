#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List Borzoi output tracks (targets) by regex/assay and print track indices.")

    p.add_argument(
        "--targets_tsv",
        required=True,
        help="Path to targets.txt/targets.tsv (tab-separated) with at least 'identifier' and optionally 'description'.",
    )
    p.add_argument(
        "--regex",
        default=None,
        help="Case-insensitive regex matched against identifier/description (depending on --field).",
    )
    p.add_argument(
        "--field",
        choices=["identifier", "description", "both"],
        default="both",
        help="Which column(s) to match --regex against.",
    )
    p.add_argument(
        "--assay",
        default=None,
        help=(
            "Optional assay filter based on 'description' prefix (before ':'). "
            "Example: CAGE or RNA. Comma-separated allowed (e.g. 'CAGE,RNA')."
        ),
    )
    p.add_argument(
        "--prefer_plus_strand",
        action="store_true",
        help=(
            "If strand_pair column exists, keep only one track per pair by preferring the '+' identifier when present."
        ),
    )
    p.add_argument("--limit", type=int, default=50, help="Max rows to print (table mode).")
    p.add_argument(
        "--format",
        choices=["table", "csv"],
        default="table",
        help="Output format: table prints rows; csv prints comma-separated indices only.",
    )

    return p.parse_args()


def _infer_track_index(df: pd.DataFrame) -> pd.Series:
    # Common: an unnamed leading index column.
    first = str(df.columns[0])
    if first.startswith("Unnamed"):
        return df.iloc[:, 0].astype(int)
    if "index" in df.columns:
        return df["index"].astype(int)
    if "track_index" in df.columns:
        return df["track_index"].astype(int)
    return pd.Series(range(len(df)), dtype=int)


def main() -> None:
    args = parse_args()

    path = Path(args.targets_tsv)
    if not path.exists():
        raise SystemExit(f"targets_tsv not found: {path}")

    df = pd.read_csv(path, sep="\t")
    if "identifier" not in df.columns:
        raise SystemExit(f"targets_tsv missing required column 'identifier': {path}")
    if "description" not in df.columns:
        df["description"] = ""

    df = df.copy()
    df["track_index"] = _infer_track_index(df).astype(int)

    # Optional assay filter.
    if args.assay:
        assays = {a.strip().upper() for a in str(args.assay).split(",") if a.strip()}
        desc = df["description"].astype(str)
        df["_assay"] = desc.str.split(":", n=1).str[0].fillna("").str.upper()
        df = df[df["_assay"].isin(assays)].copy()

    # Optional regex filter.
    if args.regex and str(args.regex).strip() != "":
        pat = re.compile(str(args.regex), flags=re.IGNORECASE)

        def match_row(r: pd.Series) -> bool:
            ident = str(r.get("identifier", ""))
            desc = str(r.get("description", ""))
            if args.field == "identifier":
                return pat.search(ident) is not None
            if args.field == "description":
                return pat.search(desc) is not None
            return (pat.search(ident) is not None) or (pat.search(desc) is not None)

        m = df.apply(match_row, axis=1).astype(bool).to_numpy()
        df = df.loc[m].copy()

    # Optionally collapse strand pairs.
    if args.prefer_plus_strand and "strand_pair" in df.columns:
        df["strand_pair"] = pd.to_numeric(df["strand_pair"], errors="coerce").astype("Int64")

        def pair_key_row(r: pd.Series):
            if pd.isna(r.get("strand_pair")):
                return (int(r["track_index"]),)
            a = int(r["track_index"])
            b = int(r["strand_pair"])
            return (a, b) if a < b else (b, a)

        df["_pair_key"] = df.apply(pair_key_row, axis=1)
        # prefer '+' in identifier, else smallest track_index
        df["_prefer"] = df["identifier"].astype(str).str.endswith("+").astype(int)
        df = (
            df.sort_values(["_pair_key", "_prefer", "track_index"], ascending=[True, False, True])
            .drop_duplicates(subset=["_pair_key"], keep="first")
            .copy()
        )

    df = df.sort_values(["track_index"]).reset_index(drop=True)

    if args.format == "csv":
        print(",".join(map(str, df["track_index"].astype(int).tolist())))
        return

    cols = ["track_index", "identifier", "description"]
    if "strand_pair" in df.columns:
        cols.append("strand_pair")
    if "_assay" in df.columns:
        cols.insert(2, "_assay")

    show = df[cols].head(int(args.limit)).copy()
    print(f"matches={len(df)}")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
