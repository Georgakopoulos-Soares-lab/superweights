from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from superweights_borzoi.models.borzoi_pt import get_module_by_name


def _as_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)) and x and torch.is_tensor(x[0]):
        return x[0]
    return None


@dataclass
class ActivationBatchSummary:
    # layer -> feature -> np.ndarray [B, C]
    by_layer: Dict[str, Dict[str, np.ndarray]]


class ActivationCapturer:
    """Forward-hook activation summaries for selected layers.

    Stores only channel summaries per batch (mean/max/L2 across length).
    Does NOT store full activation tensors.
    """

    def __init__(self, model: torch.nn.Module, layer_names: List[str]) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._current: Dict[str, Dict[str, np.ndarray]] = {}

    def _hook(self, layer_name: str):
        def fn(_module, _inp, out):
            t = _as_tensor(out)
            if t is None:
                logger.warning(f"Layer {layer_name} hook: non-tensor output")
                return
            if t.ndim < 3:
                logger.warning(f"Layer {layer_name} hook: expected [B,C,L], got shape {tuple(t.shape)}")
                return

            x = t
            # summarize immediately to avoid keeping big tensors
            mean = x.mean(dim=-1)
            maxv = x.amax(dim=-1)
            l2 = torch.linalg.vector_norm(x, ord=2, dim=-1)

            self._current[layer_name] = {
                "mean": mean.detach().to("cpu", dtype=torch.float32).numpy(),
                "max": maxv.detach().to("cpu", dtype=torch.float32).numpy(),
                "l2": l2.detach().to("cpu", dtype=torch.float32).numpy(),
            }

        return fn

    def register(self) -> None:
        self.clear()
        for layer_name in self.layer_names:
            module = get_module_by_name(self.model, layer_name)
            handle = module.register_forward_hook(self._hook(layer_name))
            self._handles.append(handle)
        logger.info(f"Registered {len(self._handles)} activation hooks")

    def clear(self) -> None:
        self._current = {}

    def pop(self) -> ActivationBatchSummary:
        out = self._current
        self._current = {}
        return ActivationBatchSummary(by_layer=out)

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


class ActivationWindowCapturer:
    """Forward-hook activation summaries within a window for selected layers.

    Computes per-channel mean over a *bp-defined* window (mapped to each layer's
    length dimension). This avoids keeping large [B,C,L] tensors while allowing
    localized summaries aligned to a task readout window.

    Notes:
    - The same bp window is applied to every sample in the batch.
    - The bp window is defined in *input sequence coordinates* [0, seq_len).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        *,
        seq_len: int,
        win_start_bp: float,
        win_end_bp: float,
    ) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self.seq_len = int(seq_len)
        self.win_start_bp = float(win_start_bp)
        self.win_end_bp = float(win_end_bp)
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._current: Dict[str, Dict[str, np.ndarray]] = {}

    def _hook(self, layer_name: str):
        def fn(_module, _inp, out):
            t = _as_tensor(out)
            if t is None:
                logger.warning(f"Layer {layer_name} hook: non-tensor output")
                return
            if t.ndim < 3:
                logger.warning(f"Layer {layer_name} hook: expected [B,C,L], got shape {tuple(t.shape)}")
                return

            x = t
            L = int(x.shape[-1])
            if L <= 0:
                logger.warning(f"Layer {layer_name} hook: empty length dimension")
                return

            # Map bp window to layer bins.
            bin_bp = float(self.seq_len) / float(L)
            lo = int(np.floor(self.win_start_bp / bin_bp))
            hi = int(np.ceil(self.win_end_bp / bin_bp))
            lo = max(0, min(lo, L))
            hi = max(0, min(hi, L))
            if hi <= lo:
                # Fallback to a single bin around the mapped center.
                center = int(np.floor(((self.win_start_bp + self.win_end_bp) / 2.0) / bin_bp))
                center = max(0, min(center, L - 1))
                lo, hi = center, center + 1

            # summarize immediately to avoid keeping big tensors
            mean_win = x[..., lo:hi].mean(dim=-1)

            self._current[layer_name] = {
                "mean_win": mean_win.detach().to("cpu", dtype=torch.float32).numpy(),
                "win_lo": np.asarray([lo], dtype=np.int32),
                "win_hi": np.asarray([hi], dtype=np.int32),
                "win_bin_bp": np.asarray([bin_bp], dtype=np.float32),
            }

        return fn

    def register(self) -> None:
        self.clear()
        for layer_name in self.layer_names:
            module = get_module_by_name(self.model, layer_name)
            handle = module.register_forward_hook(self._hook(layer_name))
            self._handles.append(handle)
        logger.info(f"Registered {len(self._handles)} activation-window hooks")

    def clear(self) -> None:
        self._current = {}

    def pop(self) -> ActivationBatchSummary:
        out = self._current
        self._current = {}
        return ActivationBatchSummary(by_layer=out)

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


class ActivationMultiWindowCapturer:
    """Forward-hook activation summaries within multiple bp windows for selected layers.

    Computes per-channel mean over each bp-defined window (mapped to each layer's
    length dimension). Returns per layer:

      - mean_win: np.ndarray [B, C, W]

    where W is the number of windows.

    This is intended for small window grids (e.g. pad/shift sweeps) while avoiding
    storing full activation tensors.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: List[str],
        *,
        seq_len: int,
        windows_bp: List[Tuple[float, float]],
        channels: Optional[List[int]] = None,
    ) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self.seq_len = int(seq_len)
        self.windows_bp = [(float(a), float(b)) for a, b in windows_bp]
        if not self.windows_bp:
            raise ValueError("windows_bp must be non-empty")

        self.channels = None if channels is None else [int(c) for c in channels]
        self._channels_t: Optional[torch.Tensor] = None

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._current: Dict[str, Dict[str, np.ndarray]] = {}

        # Cache per layer: (L, lo_idx_tensor[W], hi_idx_tensor[W], denom_tensor[W], lo_mask_tensor[W])
        self._bin_cache: Dict[str, Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _compute_bin_indices(self, layer_name: str, L: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bin_bp = float(self.seq_len) / float(L)
        lo_list: List[int] = []
        hi_list: List[int] = []
        for a_bp, b_bp in self.windows_bp:
            lo = int(np.floor(a_bp / bin_bp))
            hi = int(np.ceil(b_bp / bin_bp))
            lo = max(0, min(lo, L))
            hi = max(0, min(hi, L))
            if hi <= lo:
                center = int(np.floor(((a_bp + b_bp) / 2.0) / bin_bp))
                center = max(0, min(center, L - 1))
                lo, hi = center, center + 1
            lo_list.append(int(lo))
            hi_list.append(int(hi))

        lo_t = torch.tensor(lo_list, dtype=torch.long, device=device)
        hi_t = torch.tensor(hi_list, dtype=torch.long, device=device)
        denom = (hi_t - lo_t).clamp(min=1).to(dtype=torch.float32)
        has_lo = (lo_t > 0).to(dtype=torch.float32)

        # cache uses original device, but we will re-create if device changes.
        self._bin_cache[layer_name] = (int(L), lo_t, hi_t, denom, has_lo)
        return lo_t, hi_t, denom, has_lo

    def _hook(self, layer_name: str):
        def fn(_module, _inp, out):
            t = _as_tensor(out)
            if t is None:
                logger.warning(f"Layer {layer_name} hook: non-tensor output")
                return
            if t.ndim < 3:
                logger.warning(f"Layer {layer_name} hook: expected [B,C,L], got shape {tuple(t.shape)}")
                return

            x = t
            L = int(x.shape[-1])
            if L <= 0:
                logger.warning(f"Layer {layer_name} hook: empty length dimension")
                return

            cached = self._bin_cache.get(layer_name)
            if cached is None or int(cached[0]) != int(L) or cached[1].device != x.device:
                lo_t, hi_t, denom, has_lo = self._compute_bin_indices(layer_name, L, x.device)
            else:
                _L, lo_t, hi_t, denom, has_lo = cached

            # prefix sums along length: [B,C,L]
            pref = x.cumsum(dim=-1)

            # gather endpoints for all windows at once -> [B,C,W]
            W = int(lo_t.numel())
            idx_hi = (hi_t - 1).clamp(min=0, max=L - 1).view(1, 1, W).expand(int(x.shape[0]), int(x.shape[1]), W)
            idx_lo = (lo_t - 1).clamp(min=0, max=L - 1).view(1, 1, W).expand(int(x.shape[0]), int(x.shape[1]), W)

            pref_hi = pref.gather(dim=-1, index=idx_hi)
            pref_lo = pref.gather(dim=-1, index=idx_lo)
            pref_lo = pref_lo * has_lo.view(1, 1, W)

            sums = pref_hi - pref_lo
            mean_win = sums / denom.view(1, 1, W)

            if self.channels is not None:
                if self._channels_t is None or self._channels_t.device != mean_win.device:
                    self._channels_t = torch.tensor(self.channels, dtype=torch.long, device=mean_win.device)
                mean_win = mean_win.index_select(dim=1, index=self._channels_t)

            self._current[layer_name] = {
                "mean_win": mean_win.detach().to("cpu", dtype=torch.float32).numpy(),
            }

        return fn

    def register(self) -> None:
        self.clear()
        for layer_name in self.layer_names:
            module = get_module_by_name(self.model, layer_name)
            handle = module.register_forward_hook(self._hook(layer_name))
            self._handles.append(handle)
        logger.info(f"Registered {len(self._handles)} activation-multiwindow hooks")

    def clear(self) -> None:
        self._current = {}

    def pop(self) -> ActivationBatchSummary:
        out = self._current
        self._current = {}
        return ActivationBatchSummary(by_layer=out)

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
