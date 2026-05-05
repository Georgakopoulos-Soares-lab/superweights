"""
inspect_hybridna.py — Load HybriDNA-7B on CPU and inspect architecture.

Confirms:
  - The down_proj pattern: model.layers.{i}.feed_forward.down_proj
  - Number of hidden layers (should be 32)
  - Layer shapes

Run with:
  bash run.sh inspect_hybridna.py
"""
import re
from transformers import AutoModelForCausalLM

MODEL_ID = "Mishamq/HybriDNA-7B"

print(f"Loading {MODEL_ID} on GPU (device_map=auto, use_mamba_kernels=False) ...")
m = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
    use_mamba_kernels=False,
)

names = [n for n, _ in m.named_modules()]

# Candidates for MLP / projection layers
candidates = [n for n in names if any(k in n.lower() for k in ["mlp", "feed_forward", "down", "fc2", "proj"])]
print("\n--- MLP / projection candidates (first 60) ---")
print("\n".join(candidates[:60]))

# Layer count
indices = set(re.findall(r'layers\.(\d+)', " ".join(names)))
if indices:
    num_layers = max(int(i) for i in indices) + 1
    print(f"\nnum_layers: {num_layers}")

# Inspect first and last layer down_proj shapes
print("\n--- down_proj shapes ---")
for idx in [0, 1, 4, 15, 31]:
    try:
        mod = m.model.layers[idx].feed_forward.down_proj
        print(f"  layer {idx:2d}: weight shape = {tuple(mod.weight.shape)}")
    except AttributeError as e:
        print(f"  layer {idx:2d}: ERROR — {e}")

# Print full model structure summary (top-level only)
print("\n--- Top-level model structure ---")
for name, mod in m.named_children():
    print(f"  {name}: {type(mod).__name__}")
