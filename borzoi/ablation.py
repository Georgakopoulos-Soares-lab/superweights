from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from loguru import logger

from borzoi.model import get_module_by_name


def _as_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
        return x[0]
    return None


def _replace_first(out, new0: torch.Tensor):
    if torch.is_tensor(out):
        return new0
    if isinstance(out, tuple):
        return (new0,) + tuple(out[1:])
    if isinstance(out, list):
        out2 = list(out)
        out2[0] = new0
        return out2
    return out


def _infer_channel_dim(t: torch.Tensor, channels: Sequence[int]) -> Optional[int]:
    if t.ndim < 3:
        return None

    max_ch = int(max(channels)) if channels else 0

    # Common in this repo: [B, C, L]
    if t.ndim == 3:
        if max_ch < t.shape[1]:
            return 1
        if max_ch < t.shape[2]:
            return 2
        return None

    # If higher-rank, try to find a dimension that can hold channels
    for dim in range(1, t.ndim):
        if max_ch < t.shape[dim]:
            return dim
    return None


@dataclass
class ChannelAblator:
    """Ablate (zero-out) one or more channels at a named module output.

    Implemented as a forward hook that replaces the module output.
    """

    model: torch.nn.Module
    layer_name: str
    channels: List[int]

    _handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def register(self) -> None:
        if self._handle is not None:
            return

        module = get_module_by_name(self.model, self.layer_name)
        channels = [int(c) for c in self.channels]

        def hook(_module, _inp, out):
            t = _as_tensor(out)
            if t is None:
                logger.warning(f"Ablation hook {self.layer_name}: non-tensor output")
                return out

            ch_dim = _infer_channel_dim(t, channels)
            if ch_dim is None:
                logger.warning(f"Ablation hook {self.layer_name}: cannot infer channel dim for shape {tuple(t.shape)}")
                return out

            # Avoid in-place mutation on tensors that might be reused by autograd/other hooks.
            x = t.clone()

            for ch in channels:
                if ch < 0 or ch >= x.shape[ch_dim]:
                    logger.warning(
                        f"Ablation hook {self.layer_name}: channel out of range: {ch} (dim={ch_dim}, size={x.shape[ch_dim]})"
                    )
                    continue

                idx = [slice(None)] * x.ndim
                idx[ch_dim] = ch
                x[tuple(idx)] = 0

            return _replace_first(out, x)

        self._handle = module.register_forward_hook(hook)

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.remove()
        except Exception:
            pass
        self._handle = None

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


__all__ = ["ChannelAblator"]
