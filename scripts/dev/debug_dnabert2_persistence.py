"""
debug_dnabert2_persistence.py — Does the super activation at channel 1062 persist
across all 12 layers via skip connections?

Mirrors debug_layer_persistence.py for DNABERT-2.

In GENERator (Llama-style decoder), Yu et al. show the super activation persists
at CONSTANT magnitude through all later layers via residual connections.
This script checks whether DNABERT-2 behaves the same way.

Method:
  Hook the INPUT of every bert.encoder.layer.{i}.mlp.wo across all 12 layers.
  Record:
    - ch_max:    abs-max of intermediate channel 1062 across all token positions
    - ch_mean:   abs-mean of intermediate channel 1062 across all token positions
    - global_max: abs-max of the entire input tensor at that layer (all channels)

  Run twice:
    intact:  normal forward pass
    zeroed:  row 603 zeroed in layer 5's wo weight

  Print comparison table.  If the spike persists (like Llama), ch_max should
  remain large at layers 6–11 even in the intact case.
  If it decays, DNABERT-2's super activation is LOCAL (only at L5), which is a
  key mechanistic difference from NLP super weights.

  Also check: does zeroing row 603 at L5 damp ch_max at layers 6–11?
  If yes, the super activation flows through skip connections.

Target module: bert.encoder.layer.{i}.mlp.wo  (in=3072, out=768)

Usage:
  cd genomic-super-weights
  CUDA_VISIBLE_DEVICES=5 python debug_dnabert2_persistence.py
"""
import torch
import yaml
from models import WRAPPER_MAP
from probes.dna_probes import PROBES

SUPER_ROW  = 603
SUPER_COL  = 1062   # intermediate channel to track at wo input (3072-dim space)
LAYER      = 5      # the spike layer where super row lives
PROBE_KEY  = "actb_full"
NUM_LAYERS = 12

config  = yaml.safe_load(open("configs/dnabert2.yaml"))
wrapper = WRAPPER_MAP["dnabert2"](config)
wrapper.load()
model   = wrapper.model

wo_modules = [wrapper.get_target_module(i) for i in range(NUM_LAYERS)]


def run_sweep(label: str) -> list:
    """
    Register pre-hooks on all wo layers to record, per layer:
      - ch_max:     max |activation| at intermediate dimension SUPER_COL
      - ch_mean:    mean |activation| at intermediate dimension SUPER_COL
      - global_max: max |activation| across the entire input tensor
    Returns list of per-layer dicts (indexed 0..NUM_LAYERS-1).
    """
    records = [None] * NUM_LAYERS
    handles = []

    for i, mod in enumerate(wo_modules):
        def make_hook(layer_idx):
            def hook(m, args):
                # args[0]: input to wo, shape [batch, seq, intermediate=3072]
                x    = args[0].detach().float().squeeze(0)   # [seq, 3072]
                ch   = x[:, SUPER_COL]                       # [seq]
                records[layer_idx] = {
                    "ch_max":     ch.abs().max().item(),
                    "ch_mean":    ch.abs().mean().item(),
                    "global_max": x.abs().max().item(),
                }
            return hook
        handles.append(mod.register_forward_pre_hook(make_hook(i)))

    tokenizer = wrapper.tokenizer
    inputs    = tokenizer(
        PROBES[PROBE_KEY], return_tensors="pt", truncation=True, max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return records


# ── Run 1: intact model ───────────────────────────────────────────────────────
records_intact = run_sweep("intact")

# ── Run 2: super row zeroed ───────────────────────────────────────────────────
sr_module = wrapper.get_target_module(LAYER)
sr_saved  = sr_module.weight.data[SUPER_ROW, :].clone()
with torch.no_grad():
    sr_module.weight.data[SUPER_ROW, :] = 0.0

records_zeroed = run_sweep("zeroed")

with torch.no_grad():
    sr_module.weight.data[SUPER_ROW, :] = sr_saved   # restore

# ── Print comparison ──────────────────────────────────────────────────────────
print("=" * 88)
print(f"Super activation persistence — intermediate channel {SUPER_COL} at wo input")
print(f"Super row: layer={LAYER}, row={SUPER_ROW}  |  Probe: {PROBE_KEY}")
print(f"Model: DNABERT-2 (12-layer encoder, wo: 3072 → 768)")
print("=" * 88)
print(f"  {'Layer':>6}  {'ch_max (intact)':>18}  {'ch_max (zeroed)':>18}  "
      f"{'global_max (intact)':>21}  note")
print("  " + "-" * 86)

for i in range(NUM_LAYERS):
    ri   = records_intact[i]
    rz   = records_zeroed[i]
    flag = ""
    if i == LAYER:
        flag = " ← super row layer"
    elif ri["ch_max"] > 0.5 * records_intact[LAYER]["ch_max"]:
        flag = " ← large (>50% of origin)"
    print(f"  {i:>6}  {ri['ch_max']:>18.2f}  {rz['ch_max']:>18.2f}  "
          f"{ri['global_max']:>21.2f}  {flag}")

# ── Summary: does spike persist? ─────────────────────────────────────────────
print()
origin_val = records_intact[LAYER]["ch_max"]
post_vals  = [records_intact[i]["ch_max"] for i in range(LAYER + 1, NUM_LAYERS)]

if post_vals:
    max_post  = max(post_vals)
    mean_post = sum(post_vals) / len(post_vals)

    print(f"Origin (L{LAYER})     ch_max = {origin_val:.2f}")
    print(f"Layers {LAYER+1}–{NUM_LAYERS-1}  ch_max: max={max_post:.2f}  mean={mean_post:.2f}")
    ratio = max_post / origin_val if origin_val > 0 else 0.0
    print()
    if ratio > 0.5:
        print(f"→ Super activation PERSISTS downstream (post-spike max = {ratio*100:.0f}% of origin)")
        print("  Consistent with Yu et al. (2024) skip-connection amplification.")
    elif ratio > 0.1:
        print(f"→ Super activation ATTENUATES downstream ({ratio*100:.0f}% of origin)")
        print("  Partial persistence — possibly layer-norm dampening.")
    else:
        print(f"→ Super activation DECAYS after L{LAYER} ({ratio*100:.1f}% of origin)")
        print("  DNABERT-2 super activation is LOCAL — distinct from Llama/GENERator behaviour.")

# ── Zeroing effect: does removing super row collapse downstream? ──────────────
print()
print("── Effect of zeroing row 603 on downstream channels ──")
print(f"  {'Layer':>6}  {'intact ch_max':>16}  {'zeroed ch_max':>16}  {'ratio (z/i)':>14}")
for i in range(LAYER, NUM_LAYERS):
    ri    = records_intact[i]
    rz    = records_zeroed[i]
    ratio = rz["ch_max"] / ri["ch_max"] if ri["ch_max"] > 0 else float("nan")
    flag  = " ← zeroed here" if i == LAYER else ""
    print(f"  {i:>6}  {ri['ch_max']:>16.2f}  {rz['ch_max']:>16.2f}  "
          f"{ratio:>14.4f}{flag}")
