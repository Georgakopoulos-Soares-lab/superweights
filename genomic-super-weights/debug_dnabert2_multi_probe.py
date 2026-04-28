"""
debug_dnabert2_multi_probe.py — Row importance vs. random + multi-probe consistency.

Mirrors debug_super_row.py for DNABERT-2, two tests:

Test 1 – Row importance
  Entropy baseline → zero row 603 at L5 → measure entropy delta.
  Repeat for N random rows at the same layer.  If row 603 is unique,
  its delta should be >> random controls (which should be ~0%).

Test 2 – Multi-probe spike channel consistency
  Run the activation sweep on multiple DNA probes.  For GENERator, the same
  out_channel (2371) is detected every time — probe-invariant.
  Check whether the same set of rows is detected for DNABERT-2.
  Each probe reports which (layer, out_channel) is identified as the spike.

DNABERT-2 is an encoder (masked LM), so perplexity is replaced by
masked_token_entropy from analysis.ablation.

Usage:
  cd genomic-super-weights
  CUDA_VISIBLE_DEVICES=0 python debug_dnabert2_multi_probe.py
"""
import random
import torch
import yaml

from models import WRAPPER_MAP
from probes.dna_probes import PROBES
from detection.sweep import sweep
from detection.identify_spikes import find_spike_layer
from detection.iterative_finder import find_all_super_weights
from analysis.ablation import masked_token_entropy

LAYER       = 5      # expected spike layer
SUPER_ROW   = 603    # expected super row (out_channel at L5)
N_RANDOM    = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

config  = yaml.safe_load(open("configs/dnabert2.yaml"))
wrapper = WRAPPER_MAP["dnabert2"](config)
wrapper.load()

module = wrapper.get_target_module(LAYER)
W_orig = module.weight.data.clone()
nrows  = W_orig.shape[0]   # 768


def restore():
    with torch.no_grad():
        module.weight.data.copy_(W_orig)


def entropy(probe_key: str = "actb_full") -> float:
    return masked_token_entropy(wrapper, PROBES[probe_key])


# ══ Test 1: Row importance comparison ════════════════════════════════════════
print("=" * 66)
print("Test 1: Row-level importance — super row vs. random rows")
print("=" * 66)

base = entropy()
print(f"  {'Baseline (actb_full)':<32}  entropy={base:.4f}")

# Super row
with torch.no_grad():
    module.weight.data[SUPER_ROW, :] = 0.0
e_super = entropy()
restore()
print(f"  {'Prune row 603 (super row)':<32}  entropy={e_super:.4f}  "
      f"({(e_super - base) / base * 100:+.1f}%)")

# Random rows at the same layer
random_rows = random.sample([r for r in range(nrows) if r != SUPER_ROW], N_RANDOM)
for rr in random_rows:
    with torch.no_grad():
        module.weight.data[rr, :] = 0.0
    e_rand = entropy()
    restore()
    print(f"    Prune row {rr:<22}  entropy={e_rand:.4f}  "
          f"({(e_rand - base) / base * 100:+.1f}%)")


# ══ Test 2: Multi-probe spike channel consistency ════════════════════════════
print()
print("=" * 66)
print("Test 2: Multi-probe out_channel consistency (single spike layer)")
print("=" * 66)
print(f"  {'Probe':<22}  {'spike_layer':>12}  {'out_channel':>12}  "
      f"{'out_max':>12}  matches?")
print("  " + "-" * 66)

# Skip poly_a (degenerate; known to behave differently)
probe_keys = [k for k in PROBES if k != "poly_a"]
for name in probe_keys:
    records     = sweep(wrapper, PROBES[name])
    spike_layer = find_spike_layer(records)
    out_ch      = records[spike_layer]["out_channel"]
    out_max     = records[spike_layer]["out_max"]
    match       = "YES ← SUPER ROW" if out_ch == SUPER_ROW else "   "
    print(f"  {name:<22}  {spike_layer:>12}  {out_ch:>12}  {out_max:>12.1f}  {match}")


# ══ Test 3: Full iterative detection on each probe (all 10 super rows?) ══════
print()
print("=" * 66)
print("Test 3: Iterative super-row detection on each probe")
print("  (How many rows are recovered? Same set every time?)")
print("=" * 66)

detected_sets = {}
for name in probe_keys:
    restore()   # ensure clean slate between probes
    found = find_all_super_weights(wrapper, probe=PROBES[name], use_row_zeroing=True)
    restore()   # clean after iterative zeroing
    coords = [(sw["layer"], sw["row"]) for sw in found]
    detected_sets[name] = coords
    print(f"  {name:<22}  {len(found)} rows:  "
          + ", ".join(f"L{l}r{r}" for l, r in coords))

# Intersection across all probes
all_sets = [set(v) for v in detected_sets.values()]
common   = set.intersection(*all_sets) if all_sets else set()
union    = set.union(*all_sets)        if all_sets else set()
print(f"\n  Common across all probes ({len(common)}): "
      + (", ".join(f"L{l}r{r}" for l, r in sorted(common)) or "none"))
print(f"  Union ({len(union)}): "
      + ", ".join(f"L{l}r{r}" for l, r in sorted(union)))

# Final restore
restore()
