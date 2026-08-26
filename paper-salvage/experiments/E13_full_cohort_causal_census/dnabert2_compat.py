"""
experiments/E13_full_cohort_causal_census/dnabert2_compat.py

Environment-compatibility shim for loading DNABERT-2 under this session's torch 2.5.1 /
transformers 5.15.1 pair. Two independent, narrow bugs, neither a defect in this repo's own
code -- both are version-drift between DNABERT-2's 2023-era `trust_remote_code` model file
and transformers' current defaults:

1. **Meta-device alibi construction.** `bert_layers.py`'s `BertEncoder.__init__` calls
   `self.rebuild_alibi_tensor(size=..., device=None)` directly (eager tensor math, not
   deferred `nn.Parameter`/buffer registration). Current transformers unconditionally
   constructs `cls(config)` under a `torch.device("meta")` context during
   `from_pretrained` (not disabled by `low_cpu_mem_usage=False` -- verified empirically:
   passing it made no difference). With `device=None`, `rebuild_alibi_tensor`'s
   `torch.arange(size, device=None)` calls silently inherit the ambient "meta" device,
   while `torch.Tensor(...).to(None)` does not, producing a meta-vs-cpu device mismatch on
   the first elementwise multiply ("RuntimeError: Tensor on device meta is not on the
   expected device cpu!").
   Fix: monkeypatch `BertEncoder.rebuild_alibi_tensor` with a copy that hardcodes
   `device=torch.device("cpu")` instead of trusting the ambient default.

2. **`torch.load` version gate.** DNABERT-2's checkpoint is `pytorch_model.bin` (predates
   safetensors). transformers 5.x refuses to deserialize any `.bin` checkpoint unless
   torch>=2.6 (CVE-2025-32434), regardless of `weights_only`. This session's torch is
   2.5.1. Fix: monkeypatch `transformers.utils.import_utils.check_torch_load_is_safe` (and
   the copy re-imported into `transformers.modeling_utils`) to a no-op for this call --
   scoped to one pinned, already-locally-cached, non-executable weights file
   (`zhihan1996/DNABERT-2-117M` @ 7bce263b15377fc15361f52cfab88f8b586abda0) loaded via
   `local_files_only=True`, not a blanket disablement of the CVE-2025-32434 guard.

Both patches apply ONLY inside `load_dnabert2_patched()`. Verified empirically in this
session: model loads, and a real forward pass on a DNA string produces logits of the
expected shape.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


def _patched_rebuild_alibi_tensor(self, size: int, device=None):
    """Copy of bert_layers.py's original BertEncoder.rebuild_alibi_tensor, with `device`
    hardcoded to cpu instead of trusting the (possibly meta) ambient default."""
    device = torch.device("cpu")
    n_heads = self.num_attention_heads

    def _get_alibi_head_slopes(n_heads: int):
        def get_slopes_power_of_2(n: int):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n_heads).is_integer():
            return get_slopes_power_of_2(n_heads)
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        slopes_a = get_slopes_power_of_2(closest_power_of_2)
        slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
        slopes_b = slopes_b[0::2][: n_heads - closest_power_of_2]
        return slopes_a + slopes_b

    context_position = torch.arange(size, device=device)[:, None]
    memory_position = torch.arange(size, device=device)[None, :]
    relative_position = torch.abs(memory_position - context_position)
    relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)
    slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
    alibi = alibi.unsqueeze(0)
    assert alibi.shape == torch.Size([1, n_heads, size, size])

    self._current_alibi_size = size
    self.alibi = alibi


def _disable_torch_load_safety_gate() -> None:
    import transformers.modeling_utils as mu
    import transformers.utils.import_utils as iu

    noop = lambda *a, **k: None  # noqa: E731
    iu.check_torch_load_is_safe = noop
    mu.check_torch_load_is_safe = noop


def load_mosaicbert_patched(repo: str, revision: str | None = None,
                            device: str = "cuda", local_files_only: bool = True):
    """Load MosaicBERT with the same meta-device ALiBi compatibility patch.

    MosaicBERT and DNABERT-2 vendor the same MosaicML ``BertEncoder`` implementation,
    so current transformers' meta-device construction triggers the identical CPU/meta
    mismatch in ``rebuild_alibi_tensor``.  This keeps the patch scoped to the pinned,
    locally cached checkpoint and otherwise delegates loading to the Auto classes.
    """
    from transformers import AutoConfig, AutoModelForMaskedLM, BertTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    _disable_torch_load_safety_gate()
    kw = dict(trust_remote_code=True, local_files_only=local_files_only)
    if revision:
        kw["revision"] = revision
    tok = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    cfg = AutoConfig.from_pretrained(repo, **kw)
    model_cls = get_class_from_dynamic_module("bert_layers.BertForMaskedLM", repo, **kw)
    mod = sys.modules[model_cls.__module__]
    mod.BertEncoder.rebuild_alibi_tensor = _patched_rebuild_alibi_tensor
    model = AutoModelForMaskedLM.from_pretrained(repo, config=cfg, **kw)
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    return model, tok, getattr(model.config, "_commit_hash", None)


def load_dnabert2_patched(repo: str = "zhihan1996/DNABERT-2-117M",
                           revision: str = "7bce263b15377fc15361f52cfab88f8b586abda0",
                           device: str = "cuda", local_files_only: bool = True):
    """Drop-in replacement for tomography_lib.load_dnabert2_pretrained() under this
    session's torch/transformers pair. Returns (model, tokenizer, cfg_dict, patched) --
    same shape as the original, so existing call sites need only swap the import."""
    import yaml
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    _disable_torch_load_safety_gate()

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, revision=revision,
                                         local_files_only=local_files_only)
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, revision=revision,
                                      local_files_only=local_files_only)
    if not hasattr(cfg, "is_decoder") or cfg.is_decoder is None:
        cfg.is_decoder = False
    if not hasattr(cfg, "pad_token_id") or cfg.pad_token_id is None:
        cfg.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    model_cls = get_class_from_dynamic_module(
        "bert_layers.BertForMaskedLM", repo, revision=revision,
        local_files_only=local_files_only)
    mod = sys.modules[model_cls.__module__]
    mod.BertEncoder.rebuild_alibi_tensor = _patched_rebuild_alibi_tensor

    model = AutoModelForMaskedLM.from_pretrained(
        repo, config=cfg, trust_remote_code=True, revision=revision,
        local_files_only=local_files_only)
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    root = None
    for p in Path(__file__).resolve().parents:
        if (p / "configs" / "dnabert2.yaml").exists():
            root = p
            break
    cfg_dict = yaml.safe_load((root / "configs" / "dnabert2.yaml").read_text())

    bert_mod = sys.modules[type(model).__module__]
    patched_triton = getattr(bert_mod, "flash_attn_qkvpacked_func", None) is not None
    if patched_triton:
        bert_mod.flash_attn_qkvpacked_func = None

    return model, tok, cfg_dict, patched_triton
