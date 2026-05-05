"""
debug_dnabert2_profile.py — Full 12-layer activation profile for DNABERT-2.

Mirrors debug_generator.py for DNABERT-2, answering:
  1. Which layer is the spike layer?  (Expected: L5, out_channel=603)
  2. Does the spike collapse completely after zeroing row 603 at L5?
  3. Does L5's out_max equal L6's in_max?  (skip-connection propagation check)
  4. What is the weight magnitude of W[L5][603, 1062] and its global rank?

Target module (from dnabert2.yaml):
  bert.encoder.layer.{i}.mlp.wo   (in=3072, out=768)

Usage:
  cd genomic-super-weights
  CUDA_VISIBLE_DEVICES=0 python debug_dnabert2_profile.py
"""
import torch
import yaml
from models import WRAPPER_MAP
from hooks.activation_hooks import ActivationRecorder
from probes.dna_probes import DEFAULT_PROBE

LAYER     = 5      # confirmed spike layer (0-indexed)
ROW       = 603    # super row (out_channel at L5)
COL       = 1062   # in_channel at L5 (SW column in intermediate space)

config  = yaml.safe_load(open("configs/dnabert2.yaml"))
wrapper = WRAPPER_MAP["dnabert2"](config)
wrapper.load()
model = wrapper.model

probe = DEFAULT_PROBE   # ACTB CDS 1128 bp — works for any tokenizer


# ── 1. Full profile — intact model ───────────────────────────────────────────
recorder = ActivationRecorder()
recorder.register(wrapper, list(range(wrapper.num_layers)))
wrapper.forward(probe)
recorder.remove_all()

print("=== FULL ACTIVATION PROFILE — INTACT MODEL (all 12 layers) ===")
print(f"  {'layer':>5}  {'in_max':>14}  {'out_max':>14}  {'in_ch':>8}  {'out_ch':>8}")
print("  " + "-" * 60)
for i in sorted(recorder.records.keys()):
    r = recorder.records[i]
    marker = "  ← SPIKE" if i == LAYER else ""
    print(f"  {i:>5}  {r.get('in_max', 0):>14.2f}  {r.get('out_max', 0):>14.2f}  "
          f"{r.get('in_channel', '?'):>8}  {r.get('out_channel', '?'):>8}{marker}")


# ── 2. Weight magnitude context at (LAYER, ROW, COL) ─────────────────────────
module = wrapper.get_target_module(LAYER)
W_orig = module.weight.data.clone()
sw_val = W_orig[ROW, COL].item()
flat   = W_orig.abs().flatten()
topk_v, topk_i = flat.topk(20)
sw_rank = int((flat > W_orig[ROW, COL].abs()).sum().item()) + 1

print(f"\n=== WEIGHT CONTEXT: layer={LAYER}, wo shape={tuple(W_orig.shape)} ===")
print(f"  W[{ROW},{COL}] = {sw_val:.6f}  (global rank #{sw_rank} / {flat.numel()})")
print(f"  Top-20 magnitudes:")
for v, idx in zip(topk_v.tolist(), topk_i.tolist()):
    r, c = divmod(idx.item() if hasattr(idx, "item") else int(idx), W_orig.shape[1])
    marker = "  ← SW (detected)" if (r == ROW and c == COL) else ""
    print(f"    [{r:5d},{c:5d}]  {v:.4f}{marker}")


# ── 3. Zero row 603, re-run profile ──────────────────────────────────────────
with torch.no_grad():
    module.weight.data[ROW, :] = 0.0

recorder2 = ActivationRecorder()
recorder2.register(wrapper, list(range(wrapper.num_layers)))
wrapper.forward(probe)
recorder2.remove_all()

with torch.no_grad():
    module.weight.data.copy_(W_orig)   # restore

print(f"\n=== FULL ACTIVATION PROFILE — AFTER ZEROING row {ROW} at L{LAYER} ===")
print(f"  {'layer':>5}  {'in_max (intact)':>17}  {'in_max (zeroed)':>17}  "
      f"{'out_max (intact)':>18}  {'out_max (zeroed)':>18}")
print("  " + "-" * 80)
for i in sorted(recorder.records.keys()):
    ri = recorder.records[i]
    rz = recorder2.records[i]
    flag = "  ← zeroed layer" if i == LAYER else ""
    print(f"  {i:>5}  {ri.get('in_max', 0):>17.2f}  {rz.get('in_max', 0):>17.2f}  "
          f"{ri.get('out_max', 0):>18.2f}  {rz.get('out_max', 0):>18.2f}{flag}")


# ── 4. Propagation check: L5 out_max == L6 in_max? ───────────────────────────
r5 = recorder.records.get(LAYER, {})
r6 = recorder.records.get(LAYER + 1, {})
print(f"\n=== PROPAGATION CHECK (skip-connection) ===")
print(f"  L5 out_max = {r5.get('out_max', 'N/A'):.2f}  (out_ch={r5.get('out_channel')})")
print(f"  L6 in_max  = {r6.get('in_max',  'N/A'):.2f}  (in_ch ={r6.get('in_channel')})")
diff = abs(r5.get("out_max", 0) - r6.get("in_max", 0))
print(f"  |diff| = {diff:.4f}  → "
      + ("MATCH (spike propagates via skip)" if diff < 1.0
         else "MISMATCH — spike may attenuate before L6"))


# ── 5. Compare out_channel across layers ─────────────────────────────────────
print(f"\n=== OUT_CHANNEL PROPAGATION — is channel {ROW} consistently the max? ===")
print(f"  {'layer':>5}  {'out_channel':>12}  {'out_max':>12}  {'matches SW row?':>16}")
for i in sorted(recorder.records.keys()):
    r   = recorder.records[i]
    och = r.get("out_channel", -1)
    match = "YES" if och == ROW else "   "
    print(f"  {i:>5}  {och:>12}  {r.get('out_max', 0):>12.2f}  {match:>16}")
