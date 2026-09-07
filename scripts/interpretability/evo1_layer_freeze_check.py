"""
scripts/interpretability/evo1_layer_freeze_check.py
----------------------------------------------------
Follow-up to the STEP 2 gate, NOT a new experiment: it re-reads the same forward pass
the gate already ran and answers one factual question the gate's secondary table raised.

The gate found that L13-vs-L20, L13-vs-L25 and L13-vs-L31 give byte-identical difference
statistics for one input. That is only possible if the residual stream stops changing
somewhere around L20. This script measures the consecutive-layer delta directly:

    ||h_{l+1} - h_l||  for every l

and reports at which layer the block outputs stop registering in the residual at all.

Why it matters: the harness casts every parameter except poles/residues to bf16. bf16 ULP
at |h| ~ 4.2e6 is ~3.3e4. If a block's own output is smaller than that, the residual add
rounds it away and the block becomes a no-op — the same precision floor that killed the
fixed-eps injection, but applied to the model's own computation rather than to ours.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.interpretability.evo1_input_sensitivity_gate import (  # noqa: E402
    SW_ROW, capture_blocks, load_evo1,
)


def main() -> int:
    from probes.dna_probes import get_probe

    seq = get_probe("actb_500")
    print("loading Evo1 (bf16 except poles/residues) ...", flush=True)
    model, tokenizer = load_evo1("cuda")
    caps = capture_blocks(model, tokenizer, seq, "cuda")

    layers = sorted(caps.keys())
    report = {"probe": "actb_500", "consecutive": {}, "vs_final": {}}

    print("\n=== consecutive-layer change in the residual stream ===")
    print(f"  {'l -> l+1':>10}  {'max|Δ|':>12}  {'mean|Δ|':>12}  {'frac≠0':>8}  {'identical':>9}")
    for a, b in zip(layers, layers[1:]):
        x, y = caps[a].float(), caps[b].float()
        d = (x - y).abs()
        ident = bool(torch.equal(x, y))
        st = {
            "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "frac_coords_differ": float((d > 0).sum().item() / d.numel()),
            "bitwise_identical": ident,
        }
        report["consecutive"][f"L{a}_to_L{b}"] = st
        print(f"  {f'L{a}->L{b}':>10}  {st['max_abs_diff']:12.4e}  "
              f"{st['mean_abs_diff']:12.4e}  {st['frac_coords_differ']:8.4f}  {str(ident):>9}")

    final = layers[-1]
    print(f"\n=== every layer vs final layer L{final} ===")
    first_frozen = None
    for li in layers:
        x, y = caps[li].float(), caps[final].float()
        ident = bool(torch.equal(x, y))
        st = {"max_abs_diff": float((x - y).abs().max()), "bitwise_identical": ident}
        report["vs_final"][f"L{li}"] = st
        if ident and first_frozen is None:
            first_frozen = li
        print(f"  L{li:2d} vs L{final}  max|Δ|={st['max_abs_diff']:12.4e}  identical={ident}")

    report["first_layer_identical_to_final"] = first_frozen
    n_frozen = (final - first_frozen) if first_frozen is not None else 0
    report["n_frozen_blocks"] = n_frozen
    print(f"\n  first layer bit-identical to final: {first_frozen}")
    print(f"  blocks whose output does not register in the residual: {n_frozen}")

    # bf16 ULP at the residual's operating magnitude, for the write-up
    med = float(caps[final].float().abs().median())
    ulp = 2.0 ** (torch.tensor(med).log2().floor().item() - 7)  # bf16: 8 mantissa bits
    report["median_abs_h_final"] = med
    report["bf16_ulp_at_median"] = ulp
    print(f"  median |h| at L{final} = {med:.4e}   bf16 ULP there ≈ {ulp:.4e}")
    print(f"  SW row {SW_ROW} median |h| = {float(caps[final][:, SW_ROW].abs().median()):.4e}")

    out = ROOT / "results" / "evo1_layer_freeze_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
