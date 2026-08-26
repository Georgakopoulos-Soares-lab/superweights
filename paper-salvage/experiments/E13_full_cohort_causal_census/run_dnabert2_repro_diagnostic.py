#!/usr/bin/env python3
"""Narrow DNABERT-2 discovery-vs-E13 execution-path diagnostic.

This does not intervene on weights and does not search for a replacement basis.
It records the complete row-wise detector under one requested attention/input
combination. Run each combination in a fresh process because DNABERT-2's
attention implementation is selected through a remote-module global.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SALVAGE = HERE.parents[1]
ROOT = SALVAGE.parent
sys.path[:0] = [str(HERE), str(SALVAGE / "experiments" / "E7_exact_dimensionality"),
                str(SALVAGE / "experiments" / "E11_scale_ladder")]

from dnabert2_compat import (  # noqa: E402
    _disable_torch_load_safety_gate,
    _patched_rebuild_alibi_tensor,
)
from run_model import _ACTB_500  # noqa: E402
from spectral_lib import row_spectral_metrics  # noqa: E402

MODEL_ID = "zhihan1996/DNABERT-2-117M"
REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
TARGET = (5, 603)
THRESHOLD = 5.0
HISTORICAL_COORDS = [(5,603),(3,86),(3,399),(9,264),(9,294),(3,603),(3,641),(7,603),(6,603),(5,86)]
OUT = ROOT / "results" / "E13_dnabert2_reproducibility"


class Recorder:
    def __init__(self):
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []

    def add(self, layer: int, module) -> None:
        self.handles.append(module.register_forward_hook(self._hook(layer)))

    def _hook(self, layer: int):
        def hook(_module, _args, output):
            y = output[0] if isinstance(output, tuple) else output
            self.values[layer] = y.reshape(-1, y.shape[-1]).abs().float().max(dim=0).values.cpu()
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_model(attention: str):
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    import transformers

    _disable_torch_load_safety_gate()
    kw = dict(trust_remote_code=True, revision=REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, **kw)
    cfg = AutoConfig.from_pretrained(MODEL_ID, **kw)
    overrides = {}
    if not hasattr(cfg, "is_decoder") or cfg.is_decoder is None:
        cfg.is_decoder = False; overrides["is_decoder"] = False
    if not hasattr(cfg, "pad_token_id") or cfg.pad_token_id is None:
        cfg.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        overrides["pad_token_id"] = cfg.pad_token_id

    model_cls = get_class_from_dynamic_module("bert_layers.BertForMaskedLM", MODEL_ID, **kw)
    remote = sys.modules[model_cls.__module__]
    remote.BertEncoder.rebuild_alibi_tensor = _patched_rebuild_alibi_tensor
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, config=cfg, **kw)
    model.eval().cuda()
    flash_before = getattr(remote, "flash_attn_qkvpacked_func", None)
    if attention == "eager":
        remote.flash_attn_qkvpacked_func = None
    flash_after = getattr(remote, "flash_attn_qkvpacked_func", None)
    return model, tok, remote, overrides, transformers.__version__, flash_before, flash_after


def build_inputs(tok, preprocessing: str):
    if preprocessing == "historical_special_tokens":
        inputs = tok(_ACTB_500, return_tensors="pt", truncation=True, max_length=512)
    else:
        sequence = (tok.bos_token or "") + _ACTB_500
        inputs = tok(sequence, return_tensors="pt", add_special_tokens=False)
    return {k: v.cuda() for k, v in inputs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention", choices=["default", "eager"], required=True)
    ap.add_argument("--preprocessing", choices=["historical_special_tokens", "e13_no_special_tokens"], required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model, tok, remote, overrides, transformers_version, flash_before, flash_after = load_model(args.attention)
    inputs = build_inputs(tok, args.preprocessing)
    rec = Recorder()
    for layer in range(12):
        rec.add(layer, model.bert.encoder.layer[layer].mlp.wo)
    with torch.no_grad():
        model(**inputs)
    rec.close()

    rows = []
    for layer, values in sorted(rec.values.items()):
        median = float(values.median())
        for row, activation_max in enumerate(values.tolist()):
            rows.append({
                "layer": layer, "row": row, "activation_max": activation_max,
                "activation_median": median,
                "detector_ratio": activation_max / median if median > 0 else float("inf"),
            })
    by_score = sorted(rows, key=lambda x: (-x["detector_ratio"], x["layer"], x["row"]))
    by_activation = sorted(rows, key=lambda x: (-x["activation_max"], x["layer"], x["row"]))
    lookup = {(r["layer"], r["row"]): r for r in rows}
    for rank, r in enumerate(by_score, 1):
        r["ratio_rank"] = rank
    for rank, r in enumerate(by_activation, 1):
        r["activation_rank"] = rank

    mlp = model.bert.encoder.layer[TARGET[0]].mlp
    packed = mlp.gated_layers.weight.detach().cpu()
    n = packed.shape[0] // 2
    metrics, _ = row_spectral_metrics(packed[:n], packed[n:], mlp.wo.weight[TARGET[1]].detach().cpu())
    target = lookup[TARGET]
    important = []
    for coord in HISTORICAL_COORDS:
        item = dict(lookup[coord])
        item["historical_coordinate"] = True
        important.append(item)

    payload = {
        "label": args.label,
        "attention_requested": args.attention,
        "preprocessing": args.preprocessing,
        "model_id": MODEL_ID,
        "checkpoint_revision": REVISION,
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "tokenizer_class": f"{type(tok).__module__}.{type(tok).__name__}",
        "trust_remote_code": True,
        "local_files_only": True,
        "config_overrides": overrides,
        "eval_mode": not model.training,
        "model_parameter_dtype": str(next(model.parameters()).dtype),
        "autocast_enabled": torch.is_autocast_enabled(),
        "device": str(next(model.parameters()).device),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "python_version": platform.python_version(),
        "remote_module": remote.__name__,
        "flash_function_present_before_choice": flash_before is not None,
        "flash_function_present_during_forward": flash_after is not None,
        "attention_runtime": "remote-code default/flash" if flash_after is not None else "eager PyTorch fallback",
        "input_bp": len(_ACTB_500),
        "input_sha256": hashlib.sha256(_ACTB_500.encode()).hexdigest(),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "input_ids": inputs["input_ids"][0].detach().cpu().tolist(),
        "attention_mask": inputs.get("attention_mask", torch.empty(0)).detach().cpu().tolist(),
        "hook_target": "model.bert.encoder.layer.{i}.mlp.wo output",
        "detector_formula": "max_tokens(abs(output[..., row])) / median_rows(max_tokens(abs(output[..., row]))) within layer",
        "threshold": THRESHOLD,
        "eligible_rows": len(rows),
        "k": sum(r["detector_ratio"] >= THRESHOLD for r in rows),
        "target": {**target, "passes": target["detector_ratio"] >= THRESHOLD},
        "global_max_ratio": by_score[0],
        "global_max_activation": by_activation[0],
        "top20_by_ratio": by_score[:20],
        "top20_by_activation": by_activation[:20],
        "historical_coordinates": important,
        "target_structural_metrics": {
            "q1": metrics.q1, "pr_spec": metrics.pr_spec,
            "frob_norm": metrics.frob_norm, "stable_rank": metrics.stable_rank,
        },
        "rows": rows,
        "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain"),
        "script_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
    }
    out = OUT / f"{args.label}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out)
    print(json.dumps({"output": str(out), "target": payload["target"], "k": payload["k"],
                      "global_max_ratio": payload["global_max_ratio"],
                      "global_max_activation": payload["global_max_activation"]}, indent=2))


if __name__ == "__main__":
    main()
