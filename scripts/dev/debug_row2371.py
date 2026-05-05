"""
debug_row2371.py — Search for a hidden scalar super weight inside row 2371.

The detection pipeline identifies col=2536 as SW because channel 2536 has the
largest input activation at that forward pass.  But the actual dominant term in
Y[2371] = sum_k X[k] * W[2371,k] is determined by the PRODUCT x[k]*W[2371,k],
not by x[k] alone.  A smaller-magnitude weight paired with a large activation
can dominate over a large-magnitude weight paired with a small activation.

This script:
  1. Captures the actual input vector to down_proj layer 4 (per token, take peak).
  2. Ranks all 8448 columns by actual contribution |x[k] * W[2371,k]|.
  3. Runs single-scalar ablations on the top-10 contributing columns (not just #1)
     to find out if any one scalar is catastrophically important.
  4. Runs the same on N_RANDOM random columns as a noise floor.

Usage:
  python debug_row2371.py
"""
import torch
import yaml
from models import WRAPPER_MAP
from probes.dna_probes import PROBES

LAYER       = 4
SUPER_ROW   = 2371
KNOWN_COL   = 2536   # currently detected SW col
N_TOP       = 10     # test top-N contributing columns
N_RANDOM    = 5
PROBE_KEY   = "actb_full"

config  = yaml.safe_load(open("configs/generator.yaml"))
wrapper = WRAPPER_MAP["generator"](config)
wrapper.load()

module = wrapper.get_target_module(LAYER)
W_orig = module.weight.data.clone()


# ── Step 1: capture input vector at layer 4 ───────────────────────────────────
input_captures = {}

def _capture_hook(mod, args):
    x = args[0].squeeze(0).detach().float()   # [seq_len, H]
    # use the token position with the largest L2 norm (most activated)
    peak_token = x.norm(dim=-1).argmax().item()
    input_captures["x"]           = x[peak_token]
    input_captures["x_all"]       = x                             # keep all tokens
    input_captures["peak_token"]  = peak_token

handle = module.register_forward_pre_hook(_capture_hook)
wrapper.forward(PROBES[PROBE_KEY])
handle.remove()

x     = input_captures["x"]                         # [8448]
w_row = module.weight.data[SUPER_ROW].float()       # [8448]


# ── Step 2: rank columns by contribution ─────────────────────────────────────
contributions = (x * w_row).abs()   # [8448]
topk          = torch.topk(contributions, N_TOP)

print("=" * 70)
print(f"Input captured at token position {input_captures['peak_token']} "
      f"(peak norm, seq_len={input_captures['x_all'].shape[0]})")
print(f"\nTOP-{N_TOP} COLUMNS BY ACTUAL CONTRIBUTION TO NEURON {SUPER_ROW}")
print(f"{'Col':>6}  {'x[col]':>10}  {'W[row,col]':>12}  {'|x*W|':>10}  {'rank':>6}")
print("-" * 70)
for rank, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), 1):
    marker = " ← KNOWN" if idx == KNOWN_COL else ""
    print(f"{idx:>6}  {x[idx].item():>10.3f}  {w_row[idx].item():>12.4f}  "
          f"{val:>10.2f}  {rank:>6}{marker}")

sw_contrib   = contributions[KNOWN_COL].item()
sw_rank      = int((contributions > sw_contrib).sum().item()) + 1
print(f"\nKnown col={KNOWN_COL}: |x*W|={sw_contrib:.2f}  rank={sw_rank}/{len(contributions)}")


# ── Step 3: perplexity check — top contributing columns ──────────────────────
def ppl():
    from analysis.ablation import causal_perplexity
    return causal_perplexity(wrapper, PROBES[PROBE_KEY])

def restore():
    with torch.no_grad():
        module.weight.data.copy_(W_orig)

base = ppl()
print(f"\n{'=' * 70}")
print(f"SCALAR ABLATION — single-column prune within row {SUPER_ROW}")
print(f"{'Condition':<35} {'PPL':>8}  {'Delta':>8}")
print("-" * 70)
print(f"{'Baseline':<35} {base:>8.4f}")

# Top-N contributing columns
for rank, (val, col) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), 1):
    with torch.no_grad():
        module.weight.data[SUPER_ROW, col] = 0.0
    p = ppl()
    restore()
    marker = " ← KNOWN" if col == KNOWN_COL else ""
    print(f"  Prune col {col:5d} (contrib rank {rank:2d}){marker:<10}  "
          f"{p:>8.4f}  {(p-base)/base*100:>+7.1f}%")

# N_RANDOM random columns as noise floor
import random; random.seed(99)
top_cols = set(topk.indices.tolist())
rand_cols = random.sample([c for c in range(w_row.shape[0]) if c not in top_cols], N_RANDOM)
print(f"\n  --- random column baseline (noise floor) ---")
rand_ppls = []
for col in rand_cols:
    with torch.no_grad():
        module.weight.data[SUPER_ROW, col] = 0.0
    p = ppl()
    restore()
    rand_ppls.append(p)
    print(f"  Prune col {col:5d} (random)              "
          f"    {p:>8.4f}  {(p-base)/base*100:>+7.1f}%")

avg_rand = sum(rand_ppls) / len(rand_ppls)
print(f"\n  Average random-col perplexity: {avg_rand:.4f}  "
      f"({(avg_rand-base)/base*100:+.1f}%)")

print(f"\n{'=' * 70}")
print("Interpretation:")
print("  If one top-contrib column gives catastrophic PPL → hidden scalar SW")
print("  If all top-contrib columns give ~same small delta → effect is distributed")
print("  If random cols match top-contrib → row 2371 is uniformly important")
