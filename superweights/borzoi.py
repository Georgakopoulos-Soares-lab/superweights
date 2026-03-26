from __future__ import annotations

import torch

from superweights_borzoi.models.borzoi_pt import (
    BorzoiWrapper,
    detect_seq_len,
    detect_seq_len_from_crop,
    forward_score,
    get_module_by_name,
    load_borzoi,
    score_expression,
)


__all__ = [
    "BorzoiWrapper",
    "load_borzoi",
    "detect_seq_len",
    "detect_seq_len_from_crop",
    "forward_score",
    "score_expression",
    "get_module_by_name",
]


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
