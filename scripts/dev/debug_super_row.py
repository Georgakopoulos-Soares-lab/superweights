"""
debug_super_row.py — Verify that row 2371 is a uniquely important "super row".

Tests:
  1. Prune row 2371 vs. prune N random other rows — if row 2371 is uniquely
     catastrophic, this is the super row finding.
  2. Multi-probe consistency — run all probes and check which out_channel is
     identified as the spike. If it's 2371 every time, the super activation
     channel is probe-invariant (key property from Yu et al.).

Usage:
  python debug_super_row.py
"""
import random
import torch
import yaml

from models import WRAPPER_MAP
from probes.dna_probes import PROBES
from detection.sweep import sweep
from detection.identify_spikes import find_spike_layer

LAYER       = 4
SUPER_ROW   = 2371
N_RANDOM    = 5     # number of random rows to test as control
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

config  = yaml.safe_load(open("configs/generator.yaml"))
wrapper = WRAPPER_MAP["generator"](config)
wrapper.load()

module = wrapper.get_target_module(LAYER)
W_orig = module.weight.data.clone()
nrows  = W_orig.shape[0]


def ppl(probe_name="actb_full"):
    from analysis.ablation import causal_perplexity
    return causal_perplexity(wrapper, PROBES[probe_name])


def restore():
    with torch.no_grad():
        module.weight.data.copy_(W_orig)


# ══ Test 1: Row importance comparison ══════════════════════════════════════════
print("=" * 62)
print("Test 1: Row-level importance — super row vs. random rows")
print("=" * 62)

base = ppl()
print(f"{'Baseline':<30} {base:.4f}")

# Super row
with torch.no_grad():
    module.weight.data[SUPER_ROW, :] = 0.0
p_super = ppl()
restore()
print(f"{'Prune row 2371 (super row)':<30} {p_super:.4f}  "
      f"({(p_super - base) / base * 100:+.1f}%)")

# Random rows
random_rows = random.sample([r for r in range(nrows) if r != SUPER_ROW], N_RANDOM)
for rr in random_rows:
    with torch.no_grad():
        module.weight.data[rr, :] = 0.0
    p_rand = ppl()
    restore()
    print(f"  Prune row {rr:<20} {p_rand:.4f}  "
          f"({(p_rand - base) / base * 100:+.1f}%)")


# ══ Test 2: Multi-probe spike channel consistency ══════════════════════════════
print()
print("=" * 62)
print("Test 2: Multi-probe out_channel consistency at spike layer")
print("=" * 62)
print(f"{'Probe':<20} {'spike_layer':>12} {'out_channel':>12} {'out_max':>12}")
print("-" * 62)

probe_keys = [k for k in PROBES if k != "poly_a"]   # skip degenerate poly-A
for name in probe_keys:
    records     = sweep(wrapper, PROBES[name])
    spike_layer = find_spike_layer(records)
    out_ch      = records[spike_layer]["out_channel"]
    out_max     = records[spike_layer]["out_max"]
    marker      = " ← SUPER ROW" if out_ch == SUPER_ROW else ""
    print(f"  {name:<18} {spike_layer:>12} {out_ch:>12} {out_max:>12.1f}{marker}")
