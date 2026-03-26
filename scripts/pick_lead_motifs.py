#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick lead motifs from Tomtom matches and annotate with TF names.")
    p.add_argument("--tomtom_tsv", required=True, help="Tomtom TSV (e.g. results/motifs/.../tomtom.tsv)")
    p.add_argument(
        "--jaspar_meme",
        default="/scratch/10906/arisk/envs/superweights-borzoi/share/meme-5.5.9/doc/examples/example-datasets/JASPAR2018_CORE_non-redundant.meme",
        help="JASPAR MEME-format DB used for Tomtom (to map Target_ID -> TF name).",
    )
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--out_tsv", default=None, help="Optional output TSV path")
    return p.parse_args()


def _load_jaspar_id_to_name(meme_path: Path) -> dict[str, str]:
    id_to_name: dict[str, str] = {}
    motif_re = re.compile(r"^MOTIF\s+(\S+)\s+(.*)\s*$")
    with meme_path.open() as f:
        for line in f:
            m = motif_re.match(line)
            if not m:
                continue
            motif_id = m.group(1)
            name = m.group(2).strip()
            if motif_id and name:
                id_to_name[motif_id] = name
    return id_to_name


def _infer_family(tf_name: str) -> str:
    # Heuristic family tags for blood-relevant interpretation.
    n = tf_name.upper()
    if any(x in n for x in ["SPI", "ETS", "ELK", "ERG", "ETV", "GABP", "FLI", "PU.1"]):
        return "ETS"
    if n.startswith("GATA"):
        return "GATA"
    if n.startswith("RUNX"):
        return "RUNX"
    if n.startswith("IRF"):
        return "IRF"
    if any(x in n for x in ["NFKB", "RELA", "RELB", "REL", "NFKB1", "NFKB2"]):
        return "NFkB"
    if n.startswith("CEBP") or n.startswith("C/EBP"):
        return "CEBP"
    if n.startswith("STAT"):
        return "STAT"
    return "other"


def main() -> None:
    args = parse_args()

    tomtom_tsv = Path(args.tomtom_tsv)
    jaspar_meme = Path(args.jaspar_meme)

    df = pd.read_csv(tomtom_tsv, sep="\t", comment="#")
    if df.empty:
        raise SystemExit(f"No Tomtom hits found in: {tomtom_tsv}")

    id_to_name = _load_jaspar_id_to_name(jaspar_meme) if jaspar_meme.exists() else {}

    # Best match per discovered motif by q-value then E-value.
    df_best = (
        df.sort_values(["Query_ID", "q-value", "E-value", "p-value"], ascending=[True, True, True, True])
        .groupby("Query_ID", as_index=False)
        .first()
    )

    df_best["tf_name"] = df_best["Target_ID"].map(id_to_name).fillna(df_best["Target_ID"])
    df_best["family"] = df_best["tf_name"].map(_infer_family)

    df_out = df_best.sort_values(["q-value", "E-value", "p-value"], ascending=[True, True, True]).head(int(args.top_k)).copy()
    df_out.insert(0, "rank", range(1, len(df_out) + 1))

    cols = [
        "rank",
        "Query_ID",
        "Target_ID",
        "tf_name",
        "family",
        "q-value",
        "E-value",
        "p-value",
        "Query_consensus",
        "Target_consensus",
        "Orientation",
    ]
    df_out = df_out.loc[:, [c for c in cols if c in df_out.columns]]

    if args.out_tsv:
        out_path = Path(args.out_tsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, sep="\t", index=False)

    # Always print to stdout for quick selection.
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
