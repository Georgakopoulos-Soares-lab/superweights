# debug_layer4.py
import torch
import yaml
from models.generator_wrapper import GeneratorWrapper
from probes.dna_probes import DEFAULT_PROBE

config = yaml.safe_load(open("configs/generator.yaml"))
wrapper = GeneratorWrapper(config)
wrapper.load()

# Navigate to the full layer 4 block
pattern = config["down_proj_pattern"].replace("{i}", "4")
block_path = ".".join(pattern.split(".")[:-2])  # e.g. "model.layers.4"
block = wrapper.model
for attr in block_path.split("."):
    block = getattr(block, attr)

print("=== ALL LINEAR LAYERS INSIDE LAYER 4 BLOCK ===")
records = {}
handles = []

for name, module in block.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"  {name}  weight={module.weight.shape}")

        def make_hooks(n, m):
            def pre(mod, args):
                x = args[0].reshape(-1, args[0].shape[-1]).abs()
                records[n] = {
                    "in_max": x.max().item(),
                    "in_ch": x.max(dim=0).values.argmax().item(),
                }
            def post(mod, args, out):
                y = out[0] if isinstance(out, tuple) else out
                y = y.reshape(-1, y.shape[-1]).abs()
                records[n]["out_max"] = y.max().item()
                records[n]["out_ch"] = y.max(dim=0).values.argmax().item()
            handles.append(m.register_forward_pre_hook(pre))
            handles.append(m.register_forward_hook(post))

        make_hooks(name, module)

wrapper.forward(DEFAULT_PROBE)
for h in handles:
    h.remove()

print("\n=== ACTIVATION STATS PER SUB-LAYER IN LAYER 4 ===")
for name, stats in sorted(records.items()):
    print(
        f"  {name:<30}  "
        f"in_max={stats.get('in_max', 0):>12.2f}  "
        f"out_max={stats.get('out_max', 0):>12.2f}  "
        f"in_ch={stats.get('in_ch')}  "
        f"out_ch={stats.get('out_ch')}"
    )

# Highlight the sub-layer closest to the 67551 spike
print("\n=== SPIKE ORIGIN CANDIDATE (in_max closest to 67551) ===")
best = max(records.items(), key=lambda kv: kv[1].get("in_max", 0))
print(f"  Sub-layer : {best[0]}")
print(f"  in_max    : {best[1].get('in_max'):.2f}")
print(f"  out_max   : {best[1].get('out_max'):.2f}")
print(f"  in_ch     : {best[1].get('in_ch')}")
print(f"  out_ch    : {best[1].get('out_ch')}")
print(f"\n  → Suggested down_proj_pattern: model.layers.{{i}}.{best[0]}")
