from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from loguru import logger

from superweights.borzoi import get_module_by_name


def _as_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
        return x[0]
    return None


@dataclass
class ActivationMapBatch:
    # layer -> tensor [B, n_channels, L]
    maps: Dict[str, torch.Tensor]


class ActivationMapCapturer:
    """Forward-hook capture of activation maps for selected (layer,channels).

    Unlike v1 summaries, this keeps only the selected channels and keeps the
    length dimension to support top-K localization.
    """

    def __init__(self, model: torch.nn.Module, layer_to_channels: Dict[str, Sequence[int]]) -> None:
        self.model = model
        self.layer_to_channels = {k: list(map(int, v)) for k, v in layer_to_channels.items()}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._current: Dict[str, torch.Tensor] = {}

    def _hook(self, layer_name: str, channels: list[int]):
        ch_idx = torch.tensor(channels, dtype=torch.long)

        def fn(_module, _inp, out):
            t = _as_tensor(out)
            if t is None:
                logger.warning(f"Layer {layer_name} hook: non-tensor output")
                return
            if t.ndim < 3:
                logger.warning(f"Layer {layer_name} hook: expected [B,C,L], got {tuple(t.shape)}")
                return

            x = t
            if x.device.type != "cuda" and x.device.type != "cpu":
                # should never happen, but keep it safe
                x = x.to("cpu")

            # Move channel index tensor to same device lazily
            idx = ch_idx.to(device=x.device)
            # Select only requested channels: [B, n_ch, L]
            sel = x.index_select(dim=1, index=idx)
            self._current[layer_name] = sel

        return fn

    def register(self) -> None:
        self.clear()
        for layer_name, channels in self.layer_to_channels.items():
            if not channels:
                continue
            module = get_module_by_name(self.model, layer_name)
            handle = module.register_forward_hook(self._hook(layer_name, list(channels)))
            self._handles.append(handle)
        logger.info(f"Registered {len(self._handles)} activation-map hooks")

    def clear(self) -> None:
        self._current = {}

    def pop(self) -> ActivationMapBatch:
        out = self._current
        self._current = {}
        return ActivationMapBatch(maps=out)

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return indices of top-k values in 1D tensor (ties arbitrary)."""
    if k <= 0:
        return torch.empty((0,), dtype=torch.long, device=x.device)
    k2 = min(int(k), int(x.numel()))
    return torch.topk(x, k=k2, dim=0, largest=True, sorted=False).indices
