#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract fixed-length sequence windows from a posmaps parquet and write FASTA for motif discovery. "
            "Uses seq_cache (variant_id, ref_seq, alt_seq) to avoid FASTA lookups."
        )
    )

    p.add_argument("--posmaps", required=True, type=str, help="Posmaps parquet (e.g. results/posmaps/<layer>_<ch>.parquet)")
    p.add_argument("--seq_cache", required=True, type=str, help="Parquet with variant_id, ref_seq, alt_seq")

    p.add_argument("--window_type", type=str, default="near_variant", choices=["near_variant", "global", "all"])
    p.add_argument("--window_len", type=int, default=211, help="FASTA window length in bp (odd recommended)")

    p.add_argument(
        "--use_allele",
        type=str,
        default="max_act",
        choices=["ref", "alt", "max_act"],
        help=(
            "Which allele sequence to extract. 'max_act' chooses the allele with larger activation magnitude "
            "(|act_ref| vs |act_alt|) for that site."
        ),
    )

    p.add_argument(
        "--score",
        type=str,
        default="abs_delta_act",
        choices=["abs_delta_act", "delta_act", "abs_act_alt", "abs_act_ref"],
        help="How to rank sites within a variant (used with --max_per_variant).",
    )

    p.add_argument("--max_per_variant", type=int, default=1, help="Keep at most N sites per variant per label")
    p.add_argument("--max_total", type=int, default=5000, help="Cap total windows per label")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--out_pos_fa", required=True, type=str)
    p.add_argument("--out_neg_fa", required=True, type=str)

    return p.parse_args()


def _pick_seq(row: pd.Series, cache: Dict[str, Tuple[str, str]], use: str) -> Optional[str]:
    vid = str(row["variant_id"])
    pair = cache.get(vid)
    if pair is None:
        return None
    ref_seq, alt_seq = pair

    if use == "ref":
        return ref_seq
    if use == "alt":
        return alt_seq

    # max_act
    act_ref = float(row.get("act_ref", np.nan))
    act_alt = float(row.get("act_alt", np.nan))
    if not np.isfinite(act_ref) and not np.isfinite(act_alt):
        return alt_seq
    if abs(act_alt) >= abs(act_ref):
        return alt_seq
    return ref_seq


def _extract_window(seq: str, center_index0: int, window_len: int) -> Optional[str]:
    half = window_len // 2
    start = int(center_index0) - half
    end = start + int(window_len)
    if start < 0 or end > len(seq):
        return None
    return seq[start:end]


def _site_score(row: pd.Series, score: str) -> float:
    if score == "abs_delta_act":
        return float(abs(row["delta_act"]))
    if score == "delta_act":
        return float(row["delta_act"])
    if score == "abs_act_alt":
        return float(abs(row.get("act_alt", np.nan)))
    if score == "abs_act_ref":
        return float(abs(row.get("act_ref", np.nan)))
    raise ValueError(score)


def _select_sites(df: pd.DataFrame, label: int, score: str, max_per_variant: int, max_total: int, seed: int) -> pd.DataFrame:
    d = df[df["label"] == int(label)].copy()
    if len(d) == 0:
        return d

    d["_score"] = d.apply(lambda r: _site_score(r, score), axis=1)

    # take top-k per variant
    d = d.sort_values(["variant_id", "_score"], ascending=[True, False])
    if max_per_variant is not None and int(max_per_variant) > 0:
        d = d.groupby("variant_id", sort=False).head(int(max_per_variant)).reset_index(drop=True)

    # cap total
    if max_total is not None and len(d) > int(max_total):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(np.arange(len(d)), size=int(max_total), replace=False)
        d = d.iloc[np.sort(idx)].reset_index(drop=True)

    return d


def _write_fasta(rows: Iterable[Tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for header, seq in rows:
            f.write(f">{header}\n")
            f.write(f"{seq}\n")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    posmaps = pd.read_parquet(Path(args.posmaps))
    if args.window_type != "all":
        posmaps = posmaps[posmaps["window_type"] == args.window_type].copy()

    # Minimal required columns
    required = {"variant_id", "label", "site_center1", "window_start0", "delta_act"}
    missing = [c for c in sorted(required) if c not in posmaps.columns]
    if missing:
        raise ValueError(f"posmaps missing columns: {missing}")

    cache_df = pd.read_parquet(Path(args.seq_cache))
    cache_df = cache_df[["variant_id", "ref_seq", "alt_seq"]].drop_duplicates("variant_id")
    cache: Dict[str, Tuple[str, str]] = {
        str(r.variant_id): (str(r.ref_seq), str(r.alt_seq)) for r in cache_df.itertuples(index=False)
    }

    pos_sel = _select_sites(posmaps, label=1, score=args.score, max_per_variant=args.max_per_variant, max_total=args.max_total, seed=args.seed)
    neg_sel = _select_sites(posmaps, label=0, score=args.score, max_per_variant=args.max_per_variant, max_total=args.max_total, seed=args.seed)

    def build_records(df: pd.DataFrame) -> List[Tuple[str, str]]:
        recs: List[Tuple[str, str]] = []
        for r in df.itertuples(index=False):
            row = pd.Series(r._asdict())
            seq = _pick_seq(row, cache, use=str(args.use_allele))
            if seq is None:
                continue
            # Convert genomic coordinate to index in the cached window
            center0 = int(row["site_center1"]) - 1
            window_start0 = int(row["window_start0"])  # 0-based
            idx0 = center0 - window_start0
            w = _extract_window(seq, idx0, int(args.window_len))
            if w is None:
                continue

            header = (
                f"{row['variant_id']}|label={int(row['label'])}|site_center1={int(row['site_center1'])}"
                f"|bin={int(row.get('bin_index', -1))}|delta_act={float(row.get('delta_act', np.nan)):.6g}"
                f"|act_ref={float(row.get('act_ref', np.nan)):.6g}|act_alt={float(row.get('act_alt', np.nan)):.6g}"
                f"|use={args.use_allele}|window_type={row.get('window_type', 'NA')}"
            )
            recs.append((header, w))
        return recs

    pos_records = build_records(pos_sel)
    neg_records = build_records(neg_sel)

    _write_fasta(pos_records, Path(args.out_pos_fa))
    _write_fasta(neg_records, Path(args.out_neg_fa))

    print(f"wrote pos={len(pos_records)} -> {args.out_pos_fa}")
    print(f"wrote neg={len(neg_records)} -> {args.out_neg_fa}")


if __name__ == "__main__":
    main()
