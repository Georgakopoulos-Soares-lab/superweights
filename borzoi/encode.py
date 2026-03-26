from __future__ import annotations

from typing import List

import numpy as np
import torch


_BASE_TO_IDX = np.full(256, 4, dtype=np.uint8)
for base, idx in [("A", 0), ("C", 1), ("G", 2), ("T", 3)]:
    _BASE_TO_IDX[ord(base)] = idx
    _BASE_TO_IDX[ord(base.lower())] = idx


def one_hot_encode_batch(seqs: List[str], device: torch.device) -> torch.Tensor:
    """One-hot encode DNA strings.

    Output: float32 tensor [B, 4, L] on `device`.
    A,C,G,T channel order. Non-ACGT -> all zeros.
    """
    if not seqs:
        raise ValueError("Empty seqs")

    length = len(seqs[0])
    for s in seqs:
        if len(s) != length:
            raise ValueError("All sequences must have same length")

    # CPU: convert bytes -> indices fast
    idx_np = np.empty((len(seqs), length), dtype=np.uint8)
    for i, s in enumerate(seqs):
        b = s.encode("ascii", errors="ignore")
        if len(b) != length:
            # fallback, still enforce length
            b = (s.upper()[:length]).ljust(length, "N").encode("ascii")
        idx_np[i] = _BASE_TO_IDX[np.frombuffer(b, dtype=np.uint8)]

    idx = torch.from_numpy(idx_np.astype(np.int64)).to(device=device, non_blocking=True)

    # GPU: scatter then mask invalid bases
    out = torch.zeros((idx.shape[0], 4, idx.shape[1]), device=device, dtype=torch.float32)
    valid = idx < 4
    out.scatter_(1, idx.clamp(max=3).unsqueeze(1), 1.0)
    out *= valid.unsqueeze(1).to(out.dtype)
    return out


__all__ = ["one_hot_encode_batch"]
