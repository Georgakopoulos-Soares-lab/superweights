"""
scripts/detection/run_caduceus_detection.py
-----------------------------------------------
T2.1 -- super-weight-style activation-spike detection for Caduceus.

Caduceus has no down_proj/FFN, so the standard detection/sweep.py (which
hooks a registered down_proj module) doesn't apply. This hooks every block's
combined (hidden_states + residual) output directly -- the same OUTER
512-dim residual-stream space hooks.ablation_trace operates on -- and finds
the (layer, channel) with the largest |activation| across a probe sequence,
using the same iterative peel-and-resweep algorithm as
detection/iterative_finder.py (zero the found channel via
hooks.ablation_trace.install_ablation_hook, re-sweep, repeat).

Output: results/mechanism/caduceus_sw_detected.json (NOT written to the
canonical results/super_weight_index.json, since Caduceus is not yet a
first-class registered model there and this is exploratory Tier-2 work).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models import WRAPPER_MAP  # noqa: E402
from src.ablation_trace import get_blocks, unwrap_block_output, install_ablation_hook  # noqa: E402
from src.dna_probes import get_probe  # noqa: E402

MODEL_KEY = "caduceus"
THRESHOLD = 0.1
MAX_ITER = 10

config = yaml.safe_load((ROOT / "configs/caduceus.yaml").read_text())
wrapper = WRAPPER_MAP[MODEL_KEY](config)
wrapper.load()
blocks = get_blocks(MODEL_KEY, wrapper)
n_layers = len(blocks)


def sweep(sequence: str):
    """Return per-layer max|activation| and argmax channel, via block hooks."""
    outputs = [None] * n_layers

    def make_hook(i):
        def hook(mod, inp, out):
            outputs[i] = unwrap_block_output(MODEL_KEY, out)
        return hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(blocks)]
    wrapper.forward(sequence)
    for h in handles:
        h.remove()
    return outputs


probe = get_probe("actb_500")  # Caduceus is EUK-trained (human hg38)
found = []
zeroed = []  # (layer, row) already ablated

outputs = sweep(probe)
initial_max = max(float(o.abs().max()) for o in outputs if o is not None)
print(f"[caduceus_detect] initial max|activation| across all layers = {initial_max:.4f}")

for it in range(MAX_ITER):
    outputs = sweep(probe)
    per_layer_max = [float(o.abs().max()) if o is not None else 0.0 for o in outputs]
    spike_layer = int(np.argmax(per_layer_max))
    spike_val = per_layer_max[spike_layer]
    if spike_val < THRESHOLD * initial_max:
        print(f"[iter {it}] spike suppressed below threshold ({spike_val:.4f} < {THRESHOLD*initial_max:.4f}). Done.")
        break
    o = outputs[spike_layer]
    flat_idx = int(o.abs().argmax())
    pos, row = divmod(flat_idx, o.shape[-1])
    if (spike_layer, row) in zeroed:
        print(f"[iter {it}] same (layer={spike_layer}, row={row}) re-detected. Done.")
        break
    zeroed.append((spike_layer, row))
    found.append({"layer": spike_layer, "row": row, "pos": pos, "out_max": spike_val})
    print(f"[iter {it}] layer={spike_layer} row={row} pos={pos} out_max={spike_val:.4f}")
    # install a PERMANENT ablation hook for this round's channel so the next
    # sweep reveals the NEXT-largest spike (mirrors iterative_finder.py)
    install_ablation_hook(wrapper, spike_layer, row, model_key=MODEL_KEY)

out = {"model": "caduceus", "mode": "block-output (no down_proj exists)", "results": found}
out_path = ROOT / "results/mechanism/caduceus_sw_detected.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n[caduceus_detect] wrote {out_path}")
print(json.dumps(found, indent=2))
