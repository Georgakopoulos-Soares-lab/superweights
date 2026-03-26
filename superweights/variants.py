from __future__ import annotations

from typing import Tuple

from superweights_borzoi.data import (
    canonicalize_variant_id,
    make_variant_id,
    normalize_chrom_str,
    parse_canonical_variant_id,
)


__all__ = [
    "normalize_chrom_str",
    "make_variant_id",
    "canonicalize_variant_id",
    "parse_canonical_variant_id",
]


def parse_variant_id_any(variant_id: str) -> Tuple[str, int, str, str]:
    """Parse a variant id in any supported format.

    Returns canonicalized (chrom_no_chr, pos1, ref, alt).
    """
    return parse_canonical_variant_id(canonicalize_variant_id(variant_id))
