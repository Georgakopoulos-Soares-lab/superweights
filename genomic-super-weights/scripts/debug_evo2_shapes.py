"""
Run this inside the evo2 apptainer to understand the model's forward output structure.
  apptainer exec --nv --cleanenv --bind /work,/tmp \
    --env HF_HUB_OFFLINE=1 --env PYTHONNOUSERSITE=1 \
    --env PYTHONPATH=/work/11034/atzanakak/glm_super_weight/genomic-super-weights \
    /work/11034/atzanakak/ls6/containers/evo2.sif \
    python3 /work/11034/atzanakak/glm_super_weight/genomic-super-weights/scripts/debug_evo2_shapes.py
"""
import sys, torch, yaml
sys.path.insert(0, "/work/11034/atzanakak/glm_super_weight/genomic-super-weights")

from models import WRAPPER_MAP

config = yaml.safe_load(open("configs/evo2.yaml"))
wrapper = WRAPPER_MAP["evo2"](config)
wrapper.load()

SEQ = "ATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAAGCCGGC"  # 60 bp

# ── 1. Tokenize ────────────────────────────────────────────────────────────────
input_ids = wrapper._tokenize(SEQ)
print(f"\n[1] input_ids: {input_ids.shape}  dtype={input_ids.dtype}")

# ── 2. Raw model output ────────────────────────────────────────────────────────
with torch.no_grad():
    out = wrapper.model(input_ids)

print(f"\n[2] type(out): {type(out)}")
if isinstance(out, (tuple, list)):
    for i, x in enumerate(out):
        if hasattr(x, "shape"):
            flt = x.float()
            print(f"    out[{i}]  shape={x.shape}  dtype={x.dtype}  "
                  f"min={flt.min().item():.3f}  max={flt.max().item():.3f}")
        else:
            print(f"    out[{i}]  type={type(x)}")
else:
    flt = out.float()
    print(f"    shape={out.shape}  dtype={out.dtype}  "
          f"min={flt.min().item():.3f}  max={flt.max().item():.3f}")

# ── 3. Unembed weight ──────────────────────────────────────────────────────────
if wrapper.unembed is not None:
    print(f"\n[3] unembed found: {type(wrapper.unembed)}")
    for name, p in wrapper.unembed.named_parameters():
        print(f"    param '{name}'  shape={p.shape}  dtype={p.dtype}")
else:
    print("\n[3] unembed: None")

# ── 4. score_sequences (skipped — returns numpy floats, not tensors) ───────────
print("\n[4] skipped")

# ── 5. Hook test: does zeroing output channel affect anything? ─────────────────
print("\n[5] Hook test ...")
target = wrapper.get_target_module(29)
print(f"    target module type: {type(target)}")
print(f"    target module: {target}")

fired = []

def _probe_hook(mod, inp, out):
    o = out[0] if isinstance(out, tuple) else out
    print(f"      hook fired! out type={type(o)}  shape={o.shape}  "
          f"channel_1084_max={o[..., 1084].abs().max().item():.2f}")
    fired.append(o[..., 1084].abs().max().item())
    o = o.clone()  # avoid in-place on TE internal buffer
    o[..., 1084] = 0.0
    return (o,) + out[1:] if isinstance(out, tuple) else o

h = target.register_forward_hook(_probe_hook)
with torch.no_grad():
    out2 = wrapper.model(input_ids)
h.remove()

print(f"\n    hook fired {len(fired)} times")
if not fired:
    print("    ** HOOK NEVER FIRED — module path is wrong or hook not reaching forward **")

v1 = (out[0] if isinstance(out, (tuple, list)) else out).float()
v2 = (out2[0] if isinstance(out2, (tuple, list)) else out2).float()
diff = (v1 - v2).abs()
print(f"    max  |logits_baseline - logits_ablated| : {diff.max().item():.6f}")
print(f"    mean |logits_baseline - logits_ablated| : {diff.mean().item():.6f}")
print(f"    (0.0 everywhere = hook fired but logits unchanged)")

# ── 5b. Same test but hook ALL 32 layers at once ──────────────────────────────
print("\n[5b] Hook all 32 layers at channel 1084 ...")
all_handles = []
for i in range(32):
    m = wrapper.get_target_module(i)
    def _make(layer_i):
        def _h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o = o.clone()
            o[..., 1084] = 0.0
            return (o,) + out[1:] if isinstance(out, tuple) else o
        return _h
    all_handles.append(m.register_forward_hook(_make(i)))

with torch.no_grad():
    out3 = wrapper.model(input_ids)

for h in all_handles:
    h.remove()

v3 = (out3[0] if isinstance(out3, (tuple, list)) else out3).float()
diff3 = (v1 - v3).abs()
print(f"    max  |logits_baseline - all_layers_ablated| : {diff3.max().item():.6f}")
print(f"    mean |logits_baseline - all_layers_ablated| : {diff3.mean().item():.6f}")

# ── 6. Per-token CE change with superrow hook on layer 29 ──────────────────────
print("\n[6] Per-token CE change (superrow ablation, layer 29) ...")
import torch.nn.functional as F

# Baseline CE per token
v1_logits = (out[0] if isinstance(out, (tuple, list)) else out)[0].float()  # [T, vocab]
labels = input_ids[0]  # [T]
ce_base = F.cross_entropy(v1_logits[:-1], labels[1:], reduction='none')  # [T-1]

# Ablated CE per token
v2_logits = (out2[0] if isinstance(out2, (tuple, list)) else out2)[0].float()
ce_abl  = F.cross_entropy(v2_logits[:-1], labels[1:], reduction='none')

delta_ce = (ce_abl - ce_base)
print(f"    tokens: {len(delta_ce)}")
print(f"    mean   CE change : {delta_ce.mean().item():+.6f}")
print(f"    max    CE change : {delta_ce.max().item():+.6f}  at position {delta_ce.argmax().item()}")
print(f"    min    CE change : {delta_ce.min().item():+.6f}  at position {delta_ce.argmin().item()}")
print(f"    PPL baseline     : {torch.exp(ce_base.mean()).item():.6f}")
print(f"    PPL ablated      : {torch.exp(ce_abl.mean()).item():.6f}")

# top 5 most affected tokens
topk = delta_ce.abs().topk(5)
print(f"\n    Top-5 most affected positions:")
for rank, (pos, val) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
    print(f"      rank {rank+1}: pos={pos}  delta_CE={val:+.4f}  "
          f"base_CE={ce_base[pos].item():.4f}  abl_CE={ce_abl[pos].item():.4f}")


