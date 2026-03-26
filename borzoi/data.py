from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger


def normalize_chrom_str(chrom: str) -> str:
    if chrom is None:
        return chrom
    chrom = str(chrom)
    chrom = chrom.strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    # Keep special chroms stable
    if chrom in {"M", "MT"}:
        chrom = "MT"
    return chrom


def make_variant_id(chrom: str, pos1: int, ref: str, alt: str) -> str:
    chrom = normalize_chrom_str(chrom)
    return f"{chrom}:{int(pos1)}:{ref}:{alt}"


def canonicalize_variant_id(variant_id: str) -> str:
    """Normalize a variant_id into chrom:pos:ref:alt with chrom stripped of 'chr'.

    If the input can't be parsed, returns it unchanged.
    """
    if variant_id is None:
        return variant_id
    s = str(variant_id).strip()

    # Format 1: chrom:pos:ref:alt
    if ":" in s:
        parts = s.split(":")
        if len(parts) != 4:
            return s
        chrom, pos, ref, alt = parts
        try:
            pos1 = int(pos)
        except Exception:
            return s
        return make_variant_id(chrom, pos1, ref, alt)

    # Format 2: chrom_pos_ref_alt(_b38)
    if "_" in s:
        parts = s.split("_")
        if len(parts) < 4:
            return s
        chrom, pos, ref, alt = parts[:4]
        try:
            pos1 = int(pos)
        except Exception:
            return s
        return make_variant_id(chrom, pos1, ref, alt)

    return s


def parse_canonical_variant_id(variant_id: str) -> Tuple[str, int, str, str]:
    """Parse canonical chrom:pos:ref:alt into fields.

    chrom returned without 'chr' prefix.
    """
    parts = str(variant_id).split(":")
    if len(parts) != 4:
        raise ValueError(f"Not canonical variant_id: {variant_id}")
    chrom, pos, ref, alt = parts
    return normalize_chrom_str(chrom), int(pos), str(ref), str(alt)


def parse_variant_id_any(variant_id: str) -> Tuple[str, int, str, str]:
    """Parse a variant id in any supported format.

    Returns canonicalized (chrom_no_chr, pos1, ref, alt).
    """
    return parse_canonical_variant_id(canonicalize_variant_id(variant_id))


@dataclass(frozen=True)
class PosNegSplit:
    pos_ids: set[str]
    neg_ids: set[str]


def load_id_list(path: str | Path) -> set[str]:
    path = Path(path)
    ids: set[str] = set()
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(canonicalize_variant_id(line))
    return ids


def load_posneg(pos_ids_path: str | Path, neg_ids_path: str | Path) -> PosNegSplit:
    pos_ids = load_id_list(pos_ids_path)
    neg_ids = load_id_list(neg_ids_path)
    overlap = len(pos_ids & neg_ids)
    if overlap:
        logger.warning(f"pos/neg ID lists overlap: {overlap}")
    return PosNegSplit(pos_ids=pos_ids, neg_ids=neg_ids)


def load_eqtl_parquet(parquet_path: str | Path) -> pd.DataFrame:
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    return df


def ensure_variant_id(df: pd.DataFrame) -> pd.DataFrame:
    if "variant_id" in df.columns:
        out = df.copy()
        out["variant_id"] = out["variant_id"].astype(str).map(canonicalize_variant_id)
        return out

    if "chrom" not in df.columns:
        raise ValueError("Parquet missing required column: chrom")
    if "ref" not in df.columns or "alt" not in df.columns:
        raise ValueError("Parquet missing required columns: ref/alt")

    pos_col = None
    for cand in ("pos", "pos1", "position", "bp", "variant_pos"):
        if cand in df.columns:
            pos_col = cand
            break
    if pos_col is None:
        raise ValueError("Parquet missing required position column (pos/pos1/position/bp/variant_pos)")

    out = df.copy()
    # standardize to 1-based pos column name
    out["pos"] = out[pos_col].astype(int)
    out["chrom_norm"] = out["chrom"].astype(str).map(normalize_chrom_str)
    out["variant_id"] = (
        out["chrom_norm"].astype(str)
        + ":"
        + out["pos"].astype(int).astype(str)
        + ":"
        + out["ref"].astype(str)
        + ":"
        + out["alt"].astype(str)
    )
    return out


def attach_labels(df: pd.DataFrame, split: PosNegSplit) -> Tuple[pd.DataFrame, dict]:
    """Attach label column: 1=pos, 0=neg, NaN=neither."""
    out = df.copy()
    out["is_pos"] = out["variant_id"].isin(split.pos_ids)
    out["is_neg"] = out["variant_id"].isin(split.neg_ids)

    out["label"] = pd.NA
    out.loc[out["is_pos"], "label"] = 1
    out.loc[out["is_neg"], "label"] = 0

    stats = {
        "n_total": int(len(out)),
        "n_pos_in_parquet": int(out["is_pos"].sum()),
        "n_neg_in_parquet": int(out["is_neg"].sum()),
        "n_pos_list": int(len(split.pos_ids)),
        "n_neg_list": int(len(split.neg_ids)),
        "n_posneg_overlap": int(len(split.pos_ids & split.neg_ids)),
    }

    return out, stats


def sample_posneg(
    df_labeled: pd.DataFrame,
    n_pos: Optional[int],
    n_neg: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Return a dataframe with only labeled rows, sampled to requested sizes."""
    pos_df = df_labeled[df_labeled["label"] == 1]
    neg_df = df_labeled[df_labeled["label"] == 0]

    if n_pos is not None and n_pos < len(pos_df):
        pos_df = pos_df.sample(n=n_pos, random_state=seed)
    if n_neg is not None and n_neg < len(neg_df):
        neg_df = neg_df.sample(n=n_neg, random_state=seed)

    out = pd.concat([pos_df, neg_df], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def sample_posneg_ids(
    pos_ids: set[str],
    neg_ids: set[str],
    n_pos: Optional[int],
    n_neg: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Sample directly from ID sets and return a minimal run table.

    Columns: variant_id, label
    """
    rng = np.random.default_rng(seed)

    pos = np.array(sorted(pos_ids), dtype=object)
    neg = np.array(sorted(neg_ids), dtype=object)

    if n_pos is not None and n_pos < len(pos):
        pos = rng.choice(pos, size=n_pos, replace=False)
    if n_neg is not None and n_neg < len(neg):
        neg = rng.choice(neg, size=n_neg, replace=False)

    df_pos = pd.DataFrame({"variant_id": pos, "label": 1})
    df_neg = pd.DataFrame({"variant_id": neg, "label": 0})
    out = pd.concat([df_pos, df_neg], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


__all__ = [
    "normalize_chrom_str",
    "make_variant_id",
    "canonicalize_variant_id",
    "parse_canonical_variant_id",
    "parse_variant_id_any",
    "PosNegSplit",
    "load_id_list",
    "load_posneg",
    "load_eqtl_parquet",
    "ensure_variant_id",
    "attach_labels",
    "sample_posneg",
    "sample_posneg_ids",
]
