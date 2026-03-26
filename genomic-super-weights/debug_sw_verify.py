"""
debug_sw_verify.py — Definitive super weight verification.

Runs a series of targeted ablations at the candidate super weight location
(layer=4, row=2371, col=2536) to determine whether:
  1. The single scalar truly matters catastrophically
  2. The row/col mapping is correct
  3. The whole layer is load-bearing or not

Usage:
  python debug_sw_verify.py
"""
import json
import yaml
import torch
from pathlib import Path
from models import WRAPPER_MAP
from probes.dna_probes import get_probe

LAYER = 4
ROW   = 2371
COL   = 2536
PROBE = get_probe("actb_full")   # 1128 bp = 188 tokens

config = yaml.safe_load(open("configs/generator.yaml"))
wrapper = WRAPPER_MAP["generator"](config)
wrapper.load()
model = wrapper.model

def ppl(seq=PROBE):
    wrapper_seq = wrapper._prepare_sequence(seq)
    inputs = wrapper.tokenizer(wrapper_seq, return_tensors="pt",
                                add_special_tokens=False).to(model.device)
    ids = inputs["input_ids"]
    with torch.no_grad():
        out = model(**inputs, labels=ids)
    return torch.exp(out.loss).item()

def restore(module, saved):
    with torch.no_grad():
        module.weight.data.copy_(saved)

module = wrapper.get_target_module(LAYER)
W_orig = module.weight.data.clone()

print(f"\n{'='*60}")
print(f"Target: layer={LAYER}, row={ROW}, col={COL}")
print(f"W[{ROW},{COL}] = {W_orig[ROW, COL].item():.6f}")
print(f"down_proj weight shape: {tuple(W_orig.shape)}")

# ── Weight magnitude context ──────────────────────────────────────────────────
flat    = W_orig.abs().flatten()
topk_v, topk_i = flat.topk(20)
print(f"\nTop-20 magnitudes in layer {LAYER} down_proj:")
for v, idx in zip(topk_v.tolist(), topk_i.tolist()):
    r, c = divmod(idx.item() if hasattr(idx, 'item') else idx, W_orig.shape[1])
    marker = " ← SW" if (r == ROW and c == COL) else ""
    print(f"  [{r:5d},{c:5d}]  {v:.4f}{marker}")

sw_rank = (flat > W_orig[ROW, COL].abs()).sum().item() + 1
print(f"\nSW rank by magnitude: {sw_rank} / {flat.numel()} "
      f"(top {sw_rank/flat.numel()*100:.4f}%)")

# ── Baseline ──────────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
base = ppl()
print(f"Baseline perplexity:                         {base:.4f}")

# ── Test 1: Prune single scalar ───────────────────────────────────────────────
with torch.no_grad():
    module.weight.data[ROW, COL] = 0.0
p1 = ppl()
restore(module, W_orig)
print(f"Prune W[{ROW},{COL}] (single scalar):         {p1:.4f}  ({(p1-base)/base*100:+.1f}%)")

# ── Test 2: Prune ENTIRE ROW (output channel ROW) ────────────────────────────
with torch.no_grad():
    module.weight.data[ROW, :] = 0.0
p2 = ppl()
restore(module, W_orig)
print(f"Prune entire row {ROW} ({W_orig.shape[1]} weights):      {p2:.4f}  ({(p2-base)/base*100:+.1f}%)")

# ── Test 3: Prune ENTIRE COLUMN (input channel COL) ──────────────────────────
with torch.no_grad():
    module.weight.data[:, COL] = 0.0
p3 = ppl()
restore(module, W_orig)
print(f"Prune entire col {COL} ({W_orig.shape[0]} weights):       {p3:.4f}  ({(p3-base)/base*100:+.1f}%)")

# ── Test 4: Prune ENTIRE layer 4 down_proj ───────────────────────────────────
with torch.no_grad():
    module.weight.data.zero_()
p4 = ppl()
restore(module, W_orig)
print(f"Zero all of layer {LAYER} down_proj:                {p4:.4f}  ({(p4-base)/base*100:+.1f}%)")

# ── Test 5: Try swapped row/col (in case paper caption is right, not math) ───
with torch.no_grad():
    module.weight.data[COL, ROW] = 0.0   # swapped
p5 = ppl()
restore(module, W_orig)
print(f"Prune W[{COL},{ROW}] (SWAPPED coords):         {p5:.4f}  ({(p5-base)/base*100:+.1f}%)")

# ── Test 6: Prune the ACTUAL MAX MAGNITUDE scalar in this layer ───────────────
max_idx = W_orig.abs().argmax().item()
max_r, max_c = divmod(max_idx, W_orig.shape[1])
max_val = W_orig[max_r, max_c].item()
with torch.no_grad():
    module.weight.data[max_r, max_c] = 0.0
p6 = ppl()
restore(module, W_orig)
print(f"Prune max-magnitude W[{max_r},{max_c}]={max_val:.4f}:  {p6:.4f}  ({(p6-base)/base*100:+.1f}%)")

print(f"\n{'='*60}")
print("Interpretation guide:")
print("  If Test 1 ≈ baseline → SW scalar doesn't drive perplexity")
print("  If Test 2 >> baseline → correct row, but single scalar not enough")
print("  If Test 4 ≈ baseline → layer 4 down_proj is not load-bearing at all")
print("  If Test 4 >> Test 1  → layer matters but wrong scalar identified")
print("  If Test 5 >> Test 1  → paper caption coords are correct, not math")
