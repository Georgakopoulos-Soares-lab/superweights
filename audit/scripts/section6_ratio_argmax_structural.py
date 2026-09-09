#!/usr/bin/env python
"""Section 6 point 4: q1 and ||U_k||_F of the ratio-argmax coordinate (8,603) under
the canonical (historical special-token) DNABERT-2 detector path, per the
DNABERT2_EXECUTION_PATH_DIAGNOSTIC.md finding that (8,603) -- not (5,603) -- is the
global ratio-argmax under that path. Structural metrics are weight-only (do not
depend on tokenization/preprocessing, per that diagnostic's Part C), so no forward
pass or tokenizer input is needed here -- just load weights, call row_spectral_metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "manuscript/experiments/E7_exact_dimensionality"))
sys.path.insert(0, str(ROOT / "manuscript/experiments/E13_full_cohort_causal_census"))

from spectral_lib import row_spectral_metrics  # noqa: E402
from dnabert2_compat import load_dnabert2_patched  # noqa: E402

TARGETS = [(5, 603), (8, 603)]  # activation-argmax (frozen candidate) vs ratio-argmax


def main():
    model, tok, cfg, patched = load_dnabert2_patched(device="cpu")
    model = model.eval()

    out = {}
    for layer, row in TARGETS:
        mlp = model.bert.encoder.layer[layer].mlp
        packed = mlp.gated_layers.weight.detach().cpu()
        n = packed.shape[0] // 2
        metrics, _ = row_spectral_metrics(packed[:n], packed[n:], mlp.wo.weight[row].detach().cpu())
        out[f"L{layer}r{row}"] = dict(q1=metrics.q1, pr_spec=metrics.pr_spec,
                                       frob_norm=metrics.frob_norm, stable_rank=metrics.stable_rank)
        print(f"L{layer}/r{row}: q1={metrics.q1:.6f} PR_spec={metrics.pr_spec:.6f} "
              f"||U_k||_F={metrics.frob_norm:.6f} stable_rank={metrics.stable_rank:.6f}")

    (ROOT / "audit" / "section6_ratio_argmax_structural.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
