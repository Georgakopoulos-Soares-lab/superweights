"""
inspect_ntv3.py — Load NTv3_650M_pre and confirm down_proj pattern + layer count.

Run with:
  bash run.sh inspect_ntv3.py
"""
import re
import torch
from transformers import AutoModelForMaskedLM

MODEL_ID = "InstaDeepAI/NTv3_650M_pre"

print(f"Loading {MODEL_ID} ...")
m = AutoModelForMaskedLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
)

names = [n for n, _ in m.named_modules()]

# Show all MLP / projection candidates
candidates = [n for n in names if any(k in n.lower() for k in ["mlp", "ffn", "down", "fc2", "dense", "proj", "out_proj"])]
print("\n--- MLP / projection candidates (first 80) ---")
print("\n".join(candidates[:80]))

# Layer count
indices = set(re.findall(r'\.(\d+)\.', " ".join(names)))
if indices:
    print(f"\nmax layer index seen: {max(int(i) for i in indices)}")

# Print full top-level structure
print("\n--- Top-level structure ---")
for name, mod in m.named_children():
    print(f"  {name}: {type(mod).__name__}")

# Print core children if present
if hasattr(m, "core"):
    print("\n--- m.core children ---")
    for name, mod in m.core.named_children():
        print(f"  {name}: {type(mod).__name__}")
