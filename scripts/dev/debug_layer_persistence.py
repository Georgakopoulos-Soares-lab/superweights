"""
debug_layer_persistence.py — Does the super activation at channel 2371 persist
across all 30 layers (like Llama), or does it decay after layer 4?

Yu et al. show the super activation propagates at CONSTANT magnitude through all
later layers via skip connections.  If GENERator behaves differently, that is a
key mechanistic distinction to report.

Method: hook every down_proj INPUT across all 30 layers and record:
  - The value at the super activation channel (2371) — all token positions
  - The global max of the full input tensor

Run twice: with row 2371 intact, and with row 2371 zeroed. This confirms
(a) where the spike lives in the layer stack, and
(b) whether removing the super row damps the activation downstream.

Usage:
  python debug_layer_persistence.py
"""
import torch
import yaml
from models import WRAPPER_MAP
from probes.dna_probes import PROBES

SUPER_ROW  = 2371
SUPER_COL  = 2536   # the channel that gets spiked
PROBE_KEY  = "actb_full"
NUM_LAYERS = 30

config  = yaml.safe_load(open("configs/generator.yaml"))
wrapper = WRAPPER_MAP["generator"](config)
wrapper.load()
model   = wrapper.model

down_proj_modules = [wrapper.get_target_module(i) for i in range(NUM_LAYERS)]


def run_sweep(label: str) -> list:
    """
    Register pre-hooks on all down_proj layers to capture activation at channel
    SUPER_COL, plus the global max.  Returns list of per-layer dicts.
    """
    records  = [None] * NUM_LAYERS
    handles  = []

    for i, mod in enumerate(down_proj_modules):
        def make_hook(layer_idx):
            def hook(m, args):
                x    = args[0].detach().float().squeeze(0)   # [seq, intermediate]
                ch   = x[:, SUPER_COL]                       # [seq] — super activation channel
                records[layer_idx] = {
                    "ch_max":    ch.abs().max().item(),
                    "ch_mean":   ch.abs().mean().item(),
                    "global_max": x.abs().max().item(),
                }
            return hook
        handles.append(mod.register_forward_pre_hook(make_hook(i)))

    seq    = wrapper._prepare_sequence(PROBES[PROBE_KEY])
    inputs = wrapper.tokenizer(seq, return_tensors="pt",
                                add_special_tokens=False).to(model.device)
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return records


# ── Run 1: intact model ───────────────────────────────────────────────────────
records_intact = run_sweep("intact")

# ── Run 2: super row zeroed ───────────────────────────────────────────────────
sr_module   = wrapper.get_target_module(4)
sr_saved    = sr_module.weight.data[SUPER_ROW, :].clone()
with torch.no_grad():
    sr_module.weight.data[SUPER_ROW, :] = 0.0

records_zeroed = run_sweep("zeroed")

with torch.no_grad():
    sr_module.weight.data[SUPER_ROW, :] = sr_saved   # restore

# ── Print comparison ──────────────────────────────────────────────────────────
print("=" * 80)
print(f"Super activation persistence — channel {SUPER_COL} at down_proj input across layers")
print(f"Super row: layer=4, row={SUPER_ROW}  |  Probe: {PROBE_KEY} ({len(PROBES[PROBE_KEY])} bp)")
print("=" * 80)
print(f"{'Layer':>6}  {'ch_max (intact)':>18}  {'ch_max (zeroed)':>18}  "
      f"{'global_max (intact)':>20}  {'note'}")
print("-" * 80)

for i in range(NUM_LAYERS):
    ri = records_intact[i]
    rz = records_zeroed[i]
    ratio = rz["ch_max"] / ri["ch_max"] if ri["ch_max"] > 0 else float("nan")
    flag  = ""
    if i == 4:
        flag = " ← super row layer"
    elif ri["ch_max"] > 1000:
        flag = " ← still large"
    print(f"  {i:>4}  {ri['ch_max']:>18.1f}  {rz['ch_max']:>18.1f}  "
          f"{ri['global_max']:>20.1f}  {flag}")

print()
# Summary: does the activation persist?
post_layer4 = [records_intact[i]["ch_max"] for i in range(5, NUM_LAYERS)]
max_post    = max(post_layer4)
mean_post   = sum(post_layer4) / len(post_layer4)
layer4_val  = records_intact[4]["ch_max"]

print(f"Layer 4 (origin)  ch_max = {layer4_val:.1f}")
print(f"Layers 5–29       ch_max: max={max_post:.1f}  mean={mean_post:.1f}")
print()
if max_post > 0.1 * layer4_val:
    print("→ Super activation PERSISTS downstream (max post-layer4 > 10% of origin)")
else:
    print("→ Super activation DECAYS after layer 4 — does NOT persist like Llama")
    print("  This is a key mechanistic difference from NLP super weights.")
