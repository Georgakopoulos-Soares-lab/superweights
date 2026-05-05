"""
debug_gate_up.py — Trace the origin of x[2536]=67,550 entering down_proj.

In SwiGLU: down_proj_input[ch] = silu(gate_proj_out[ch]) * up_proj_out[ch]
The 67,550 at channel 2536 comes from those two scalars.
This script finds which weight scalar(s) inside gate_proj and up_proj are
responsible — that would be the true scalar super weight if one exists.

Plan:
  1. Capture gate_proj and up_proj outputs at all token positions.
  2. Identify the token + channel that produces the 67,550 spike.
  3. Check the weight rows (output channel 2536) in gate_proj and up_proj
     for an anomalous scalar — magnitude rank, top contributors.
  4. Single-scalar ablations on top-contributing columns in both matrices.

Usage:
  python debug_gate_up.py
"""
import torch
import yaml
from models import WRAPPER_MAP
from probes.dna_probes import PROBES

LAYER     = 4
CHANNEL   = 2536   # the intermediate channel with the 67,550 spike
PROBE_KEY = "actb_full"
N_TOP     = 10
N_RANDOM  = 5

config  = yaml.safe_load(open("configs/generator.yaml"))
wrapper = WRAPPER_MAP["generator"](config)
wrapper.load()

model    = wrapper.model
mlp      = model.model.layers[LAYER].mlp
gate_mod = mlp.gate_proj
up_mod   = mlp.up_proj
down_mod = mlp.down_proj

gate_captured = {}
up_captured   = {}
down_input    = {}

def hook_gate(mod, args, out):
    gate_captured["val"] = out.detach().float().squeeze(0)   # [seq, intermediate]

def hook_up(mod, args, out):
    up_captured["val"] = out.detach().float().squeeze(0)

def hook_down_pre(mod, args):
    x = args[0].detach().float().squeeze(0)                  # [seq, intermediate]
    down_input["val"] = x

h1 = gate_mod.register_forward_hook(hook_gate)
h2 = up_mod.register_forward_hook(hook_up)
h3 = down_mod.register_forward_pre_hook(hook_down_pre)
wrapper.forward(PROBES[PROBE_KEY])
h1.remove(); h2.remove(); h3.remove()

gate = gate_captured["val"]   # [seq, intermediate_size]
up   = up_captured["val"]
x_down = down_input["val"]    # [seq, intermediate_size]  (= silu(gate) * up)

# Find the token with the largest spike at channel CHANNEL
spike_per_token = x_down[:, CHANNEL].abs()
peak_token      = spike_per_token.argmax().item()
peak_val        = x_down[peak_token, CHANNEL].item()

print("=" * 68)
print(f"Channel {CHANNEL} spike in down_proj input:")
print(f"  Peak token: {peak_token}  |x_down[{peak_token},{CHANNEL}]| = {peak_val:.2f}")
print(f"  gate_proj out [{peak_token},{CHANNEL}] = {gate[peak_token, CHANNEL].item():.4f}")
print(f"  up_proj   out [{peak_token},{CHANNEL}] = {up[peak_token, CHANNEL].item():.4f}")

import torch.nn.functional as F
silu_gate = F.silu(gate[peak_token, CHANNEL])
product   = (silu_gate * up[peak_token, CHANNEL]).item()
print(f"  silu(gate)*up (≈ down input)  = {product:.2f}")

# ── Are gate or up outputs at channel 2536 anomalous? ────────────────────────
print()
print("─" * 68)
print(f"Gate_proj output at channel {CHANNEL} — all tokens:")
gate_ch = gate[:, CHANNEL].abs()
print(f"  max={gate_ch.max().item():.2f}  mean={gate_ch.mean().item():.4f}  "
      f"std={gate_ch.std().item():.4f}")
print(f"  max across ALL channels (all tokens): {gate.abs().max().item():.2f}")

up_ch = up[:, CHANNEL].abs()
print(f"Up_proj output at channel {CHANNEL} — all tokens:")
print(f"  max={up_ch.max().item():.2f}  mean={up_ch.mean().item():.4f}  "
      f"std={up_ch.std().item():.4f}")
print(f"  max across ALL channels (all tokens): {up.abs().max().item():.2f}")

# ── Examine gate_proj weight row CHANNEL ─────────────────────────────────────
# gate_proj: shape [intermediate, hidden], output row CHANNEL produces gate[ch]
print()
print("─" * 68)
print(f"GATE_PROJ weight analysis — output row {CHANNEL}")
Wg = gate_mod.weight.data[CHANNEL].float()   # [hidden_size=3072]
Wg_flat = gate_mod.weight.data.abs().flatten()
g_rank = int((Wg_flat > Wg.abs().max()).sum().item()) + 1
print(f"  Row {CHANNEL} max magnitude: {Wg.abs().max().item():.4f}  "
      f"(rank {g_rank} / {Wg_flat.numel()})")
print(f"  Row {CHANNEL} mean magnitude: {Wg.abs().mean().item():.4f}")

# Get the input to gate_proj at peak token
gate_input = {}
def hook_gate_pre(mod, args):
    gate_input["val"] = args[0].detach().float().squeeze(0)
h_gp = gate_mod.register_forward_pre_hook(hook_gate_pre)
wrapper.forward(PROBES[PROBE_KEY])
h_gp.remove()
x_gate = gate_input["val"][peak_token]   # [hidden_size=3072]

gate_contribs = (x_gate * Wg).abs()
topk_g = torch.topk(gate_contribs, N_TOP)
print(f"\n  Top-{N_TOP} columns by |x*W| contribution to gate_proj output {CHANNEL}:")
print(f"  {'Col':>6}  {'x[col]':>10}  {'W[ch,col]':>12}  {'|x*W|':>10}")
print("  " + "-" * 48)
for val, idx in zip(topk_g.values.tolist(), topk_g.indices.tolist()):
    col = idx
    print(f"  {col:>6}  {x_gate[col].item():>10.3f}  {Wg[col].item():>12.4f}  {val:>10.2f}")

# ── Examine up_proj weight row CHANNEL ───────────────────────────────────────
print()
print("─" * 68)
print(f"UP_PROJ weight analysis — output row {CHANNEL}")
Wu = up_mod.weight.data[CHANNEL].float()   # [hidden_size=3072]
Wu_flat = up_mod.weight.data.abs().flatten()
u_rank = int((Wu_flat > Wu.abs().max()).sum().item()) + 1
print(f"  Row {CHANNEL} max magnitude: {Wu.abs().max().item():.4f}  "
      f"(rank {u_rank} / {Wu_flat.numel()})")
print(f"  Row {CHANNEL} mean magnitude: {Wu.abs().mean().item():.4f}")

up_contribs = (x_gate * Wu).abs()   # same input x to both gate and up projections
topk_u = torch.topk(up_contribs, N_TOP)
print(f"\n  Top-{N_TOP} columns by |x*W| contribution to up_proj output {CHANNEL}:")
print(f"  {'Col':>6}  {'x[col]':>10}  {'W[ch,col]':>12}  {'|x*W|':>10}")
print("  " + "-" * 48)
for val, idx in zip(topk_u.values.tolist(), topk_u.indices.tolist()):
    col = idx
    print(f"  {col:>6}  {x_gate[col].item():>10.3f}  {Wu[col].item():>12.4f}  {val:>10.2f}")

# ── Single-scalar ablations in gate_proj and up_proj ─────────────────────────
def ppl():
    from analysis.ablation import causal_perplexity
    return causal_perplexity(wrapper, PROBES[PROBE_KEY])

Wg_orig = gate_mod.weight.data.clone()
Wu_orig = up_mod.weight.data.clone()

def restore():
    with torch.no_grad():
        gate_mod.weight.data.copy_(Wg_orig)
        up_mod.weight.data.copy_(Wu_orig)

restore()   # ensure clean state before ablations

base = ppl()
print()
print("=" * 68)
print(f"SCALAR ABLATIONS — gate_proj and up_proj row {CHANNEL}")
print(f"{'Condition':<42} {'PPL':>8}  {'Delta':>8}")
print("-" * 68)
print(f"{'Baseline':<42} {base:>8.4f}")

# Check if either row is anomalous overall (zero entire row)
with torch.no_grad():
    gate_mod.weight.data[CHANNEL, :] = 0.0
p = ppl(); restore()
print(f"  Zero gate_proj row {CHANNEL} ({Wg.shape[0]} weights)  "
      f"         {p:>8.4f}  {(p-base)/base*100:>+7.1f}%")

with torch.no_grad():
    up_mod.weight.data[CHANNEL, :] = 0.0
p = ppl(); restore()
print(f"  Zero up_proj   row {CHANNEL} ({Wu.shape[0]} weights)  "
      f"         {p:>8.4f}  {(p-base)/base*100:>+7.1f}%")

# Top-N scalar ablations in gate_proj row CHANNEL
print(f"\n  --- gate_proj row {CHANNEL} top-{N_TOP} contributing scalars ---")
for rank, (val, col) in enumerate(zip(topk_g.values.tolist(), topk_g.indices.tolist()), 1):
    with torch.no_grad():
        gate_mod.weight.data[CHANNEL, col] = 0.0
    p = ppl(); restore()
    print(f"  gate_proj[{CHANNEL},{col:5d}] (rank {rank:2d})  "
          f"              {p:>8.4f}  {(p-base)/base*100:>+7.1f}%")

# Top-N scalar ablations in up_proj row CHANNEL
print(f"\n  --- up_proj   row {CHANNEL} top-{N_TOP} contributing scalars ---")
for rank, (val, col) in enumerate(zip(topk_u.values.tolist(), topk_u.indices.tolist()), 1):
    with torch.no_grad():
        up_mod.weight.data[CHANNEL, col] = 0.0
    p = ppl(); restore()
    print(f"  up_proj  [{CHANNEL},{col:5d}] (rank {rank:2d})  "
          f"              {p:>8.4f}  {(p-base)/base*100:>+7.1f}%")

print()
print("=" * 68)
print("Interpretation:")
print("  If zero entire gate/up row → catastrophic: that row is the super entity")
print("  If one scalar ablation → catastrophic: that is the scalar super weight")
print("  If all scalars → small delta: effect is distributed; no scalar SW in gate/up")
