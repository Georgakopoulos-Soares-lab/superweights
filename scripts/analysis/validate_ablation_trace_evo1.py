"""
scripts/analysis/validate_ablation_trace_evo1.py
---------------------------------------------------
Rule 6 sanity check: confirm hooks/ablation_trace.py's generalized
clean-forward block capture reproduces the already-validated Evo1 instance
in results/sw_residual_attribution_evo1_fp32.json before trusting the
primitive on new models.

That file used a bespoke clean-only hook (no ablation) on model.blocks[li],
row 3776, probe actb_500, with the model cast to bf16-except-poles/residues.
This script runs the SAME setup through hooks.ablation_trace.run_clean_trace
and diffs the per-layer (sw_pos, block_residual_out[row]) values.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models import WRAPPER_MAP  # noqa: E402
from src.dna_probes import get_probe  # noqa: E402
from src.ablation_trace import run_clean_trace  # noqa: E402

ROW = 3776

config = {"model_id": "evo-1-8k-base", "device": "cuda"}
wrapper = WRAPPER_MAP["evo1"](config)
wrapper.load()
model = wrapper.model

# Match the reference script's precision handling exactly: force everything
# except poles/residues to bf16 (pure fp32 is impossible -- flash-attn
# asserts fp16/bf16 -- and bf16 shares fp32's exponent range so it doesn't
# overflow like fp16 does past layer 10).
for n, p in model.named_parameters():
    if any(k in n for k in ("poles", "residues")):
        continue
    p.data = p.data.to(torch.bfloat16)

seq = get_probe("actb_500")
clean = run_clean_trace(wrapper, "evo1", seq)

ref = json.loads((ROOT / "results/sw_residual_attribution_evo1_fp32.json").read_text())
ref_by_layer = {r["layer"]: r for r in ref["per_layer"]}

print(f"{'layer':>5} {'ref_pos':>8} {'my_pos':>8} {'ref_res_out':>14} {'my_res_out':>14} {'abs_diff':>12} {'rel_diff':>10}")
print("-" * 80)
max_rel_diff = 0.0
for li in range(len(clean)):
    h = clean[li]
    if h is None:
        continue
    pos_norms = (h ** 2).sum(dim=1)
    my_pos = int(pos_norms.argmax())  # note: reference uses argmax of |h[:,row]|, not full-position norm
    my_pos_row = int(h[:, ROW].abs().argmax())
    my_val = float(h[my_pos_row, ROW])
    r = ref_by_layer.get(li)
    if r is None:
        continue
    ref_val = r["block_residual_out"]
    ref_pos = r["sw_pos"]
    abs_diff = abs(my_val - ref_val)
    rel_diff = abs_diff / (abs(ref_val) + 1e-8)
    max_rel_diff = max(max_rel_diff, rel_diff)
    print(f"{li:5d} {ref_pos:8d} {my_pos_row:8d} {ref_val:14.3f} {my_val:14.3f} {abs_diff:12.4f} {rel_diff:10.4%}")

print(f"\nmax relative diff across all layers: {max_rel_diff:.4%}")
print("PASS -- primitive reproduces the validated Evo1 trace" if max_rel_diff < 0.01
      else "FAIL -- primitive does NOT reproduce the validated Evo1 trace, investigate before trusting on new models")
