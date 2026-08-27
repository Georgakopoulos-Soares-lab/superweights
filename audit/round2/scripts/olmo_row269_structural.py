#!/usr/bin/env python
"""Follow-up: exact q1, PR_spec, ||U_k||_F for OLMo-7B's published coordinate (L1/r269)
alongside the two alternative coordinates the current ratio/activation detector prefers
(L2/r269 ratio-argmax, L30/r269 activation-argmax). Structural only -- weight-derived
metrics, no forward pass, no causal measurement. Appends to
audit/round2/section3_text_decoder_calibration.json under a new "structural_comparison" key.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "paper-salvage/experiments/E7_exact_dimensionality"))
from spectral_lib import row_spectral_metrics  # noqa: E402

REPO = "allenai/OLMo-7B-0724-hf"
REVISION = "1ee306df318ee15bfe4a76ebd5c002b0105b1ab6"
PATTERN = "model.layers.{i}.mlp.down_proj"
COORDS = [(1, 269, "published (Yu et al.)"), (2, 269, "ratio-argmax"), (30, 269, "activation-argmax")]
OUT = ROOT / "audit/round2/section3_text_decoder_calibration.json"


def main():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION, trust_remote_code=True,
                                                   torch_dtype=torch.float32)
    model.eval()
    mods = dict(model.named_modules())

    results = {}
    for layer, row, role in COORDS:
        base = PATTERN.format(i=layer)
        gate = mods[base.replace("down_proj", "gate_proj")].weight.detach().cpu().double()
        up = mods[base.replace("down_proj", "up_proj")].weight.detach().cpu().double()
        down_row = mods[base].weight[row].detach().cpu().double()
        metrics, _ = row_spectral_metrics(gate, up, down_row, device="cpu")
        results[f"L{layer}r{row}"] = dict(role=role, q1=metrics.q1, pr_spec=metrics.pr_spec,
                                           frob_norm=metrics.frob_norm)
        print(f"L{layer}/r{row} ({role}): q1={metrics.q1:.6f} PR_spec={metrics.pr_spec:.6f} "
              f"||U_k||_F={metrics.frob_norm:.6f}")

    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing.setdefault("OLMo-7B-0724-hf", {})["structural_comparison"] = results
    OUT.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nwrote structural_comparison into {OUT}")


if __name__ == "__main__":
    main()
