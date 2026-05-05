"""
Print Evo 2 model architecture to confirm:
  1. The down-proj layer path (backbone.blocks.{i}.mlp.l2 ?)
  2. Which block indices are attention blocks (vs Hyena/conv — skip those)
  3. Correct num_layers count

Run inside container:
  apptainer exec --nv --cleanenv --bind /work,/tmp \
    --env PYTHONPATH=$REPO "${SIF}" python3 "${REPO}/inspect_evo2.py"
"""
import torch
from evo2 import Evo2

print("Loading evo2_7b ...")
evo2 = Evo2("evo2_7b")

# Evo2 wraps the actual torch model — find it
inner = None
for attr in ("model", "net", "backbone", "_model"):
    if hasattr(evo2, attr):
        inner = getattr(evo2, attr)
        print(f"Inner model at: evo2.{attr}  → {type(inner).__name__}")
        break

if inner is None:
    print("Could not find inner model — printing evo2 attributes:")
    print(dir(evo2))
else:
    print()
    print(inner)
    print()
    print("=== Linear layers ===")
    for name, mod in inner.named_modules():
        if isinstance(mod, torch.nn.Linear):
            print(f"  {name:70s}  in={mod.in_features}, out={mod.out_features}")

    print()
    print("=== Block types ===")
    for name, mod in inner.named_modules():
        parts = name.split(".")
        if "blocks" in parts and len(parts) == 2:
            print(f"  {name:50s}  {type(mod).__name__}")

    print()
    # Try resolving the expected down-proj path for block 0
    def resolve(model, path):
        m = model
        for attr in path.split("."):
            m = getattr(m, attr)
        return m

    for candidate in ["backbone.blocks.0.mlp.l2", "blocks.0.mlp.l2",
                       "backbone.blocks.0.mlp.out_proj", "blocks.0.mlp.out_proj"]:
        try:
            m = resolve(inner, candidate)
            print(f"  Path '{candidate}' → {type(m).__name__}  shape: {m.weight.shape}  ✓")
        except AttributeError:
            print(f"  Path '{candidate}' → NOT FOUND")
