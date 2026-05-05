from transformers import AutoModelForCausalLM
import re

m = AutoModelForCausalLM.from_pretrained(
    "GenerTeam/GENERator-v2-eukaryote-3b-base",
    trust_remote_code=True,
    device_map="cpu",
)

names = [n for n, _ in m.named_modules()]
mlp = [n for n in names if any(k in n.lower() for k in ["mlp", "down", "fc2", "proj"])]
print("--- MLP candidates ---")
print("\n".join(mlp[:40]))

indices = set(re.findall(r'layers\.(\d+)', " ".join(names)))
if indices:
    print(f"\nnum_layers: {max(int(i) for i in indices) + 1}")
