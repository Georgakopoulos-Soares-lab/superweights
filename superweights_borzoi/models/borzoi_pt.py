from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import os
import torch
from loguru import logger


@dataclass
class BorzoiWrapper:
    model: torch.nn.Module
    device: torch.device
    output_key: Optional[str] = None


def load_borzoi(model_name: str, device: torch.device, output_key: Optional[str] = None) -> BorzoiWrapper:
    logger.info(f"Loading Borzoi: {model_name}")

    # Import can be slow (transformers, safetensors, etc.). Log before doing it.
    # Some HPC environments may not set $HOME; transformers may fall back to NSS/LDAP lookups.
    if not os.environ.get("HOME"):
        os.environ["HOME"] = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or "/tmp"

    # Avoid transformers importing TensorFlow/Flax (can hang/slow due to GPU init).
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    from borzoi_pytorch import Borzoi  # local env dependency

    logger.info("Downloading/loading weights (from_pretrained)...")
    model = Borzoi.from_pretrained(model_name)
    logger.info("Model loaded; moving to device...")
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    return BorzoiWrapper(model=model, device=device, output_key=output_key)


def detect_seq_len(model: torch.nn.Module) -> Optional[int]:
    """Best-effort seq_len detection from model/config."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        cfg = None

    # config might be dict-like
    for key in ("seq_len", "seq_length", "sequence_length", "input_length", "length"):
        try:
            if isinstance(cfg, dict) and key in cfg:
                v = cfg[key]
                if v is not None:
                    return int(v)
            if hasattr(cfg, key):
                v = getattr(cfg, key)
                if v is not None:
                    return int(v)
        except Exception:
            continue

    return None


def detect_seq_len_from_crop(model: torch.nn.Module) -> Optional[int]:
    """Infer input sequence length from a TargetLengthCrop module if present.

    For borzoi_pytorch models, there is typically a `model.crop.target_length` in *bins*.
    Empirically, the pre-crop bin length is a power-of-two (e.g. 8192) and the bin size is 32bp.
    We approximate:
      input_len_bp = next_pow2(target_bins) * 32
    """

    crop = getattr(model, "crop", None)
    target = getattr(crop, "target_length", None)
    if target is None:
        return None
    try:
        target = int(target)
    except Exception:
        return None
    if target <= 0:
        return None

    # next power of 2 >= target
    bins = 1
    while bins < target:
        bins *= 2
    return int(bins * 32)


def score_expression(output: Any, output_key: Optional[str] = None) -> torch.Tensor:
    """Reduce model output to scalar per batch element.

    Returns tensor [B] on the same device as output.
    """
    if isinstance(output, dict):
        if output_key is not None:
            if output_key not in output:
                raise KeyError(f"output_key not in output dict: {output_key}")
            output = output[output_key]
        else:
            # pick first tensor-like value
            for v in output.values():
                output = v
                break

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported output type: {type(output)}")

    out = output.float()
    if out.ndim == 1:
        return out
    reduce_dims = tuple(range(1, out.ndim))
    return out.mean(dim=reduce_dims)


@torch.no_grad()
def forward_score(wrapper: BorzoiWrapper, x: torch.Tensor) -> torch.Tensor:
    y = wrapper.model(x)
    return score_expression(y, output_key=wrapper.output_key)


def get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    for name, module in model.named_modules():
        if name == module_name:
            return module
    raise KeyError(f"Module not found: {module_name}")
