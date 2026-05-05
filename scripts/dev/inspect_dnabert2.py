"""
Print DNABERT-2 model architecture so we can confirm:
  1. The down-proj layer path (bert.encoder.layer.{i}.output.dense ?)
  2. Number of layers
  3. Hidden / intermediate dims

Run: bash run.sh inspect_dnabert2.py
"""
import torch
import yaml
from models import WRAPPER_MAP

config = yaml.safe_load(open("configs/dnabert2.yaml"))
wrapper = WRAPPER_MAP["dnabert2"](config)
wrapper.load()

model = wrapper.model
print(model)
print()

# Walk all named modules and print Linear layers with their shapes
print("=== Linear layers ===")
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        print(f"  {name:60s}  in={mod.in_features}, out={mod.out_features}")

print()
# Spot-check: try resolving the configured pattern for layer 0
try:
    m = wrapper.get_target_module(0)
    print(f"get_target_module(0) → {type(m).__name__}  weight shape: {m.weight.shape}")
    print("Config pattern is CORRECT.")
except Exception as e:
    print(f"get_target_module(0) FAILED: {e}")
    print("Config pattern needs updating.")
