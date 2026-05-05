"""
inspect_megadna.py — Inspect megaDNA model architecture and verify wrapper assumptions.

Run with the megadna venv (NOT grlm conda):
  source /work/11034/atzanakak/ls6/venvs/megadna/bin/activate
  export PYTHONPATH=/work/11034/atzanakak/ls6/megaDNA:$PYTHONPATH
  python inspect_megadna.py

Or via the megadna run script:
  bash run_megadna.sh inspect_megadna.py

Outputs:
  - Model class / stage structure
  - Total parameters
  - FF module structure confirming down-proj at index [4]
  - Vocab encoding check
  - Baseline perplexity on phage probe
"""
import sys
import torch

MODEL_PATH  = "/work/11034/atzanakak/ls6/megaDNA/megaDNA_phage_145M.pt"
MEGADNA_SRC = "/work/11034/atzanakak/ls6/megaDNA"

sys.path.insert(0, MEGADNA_SRC)
from megaDNA.megadna import MEGADNA  # noqa: F401 — registers class for unpickling

print(f"Loading {MODEL_PATH} ...")
model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()
print(f"  Model class   : {type(model).__name__}")

# ── Stage / layer structure ───────────────────────────────────────────────────
n_stages = len(model.transformers)
print(f"\n  stages        : {model.stages}")
print(f"  transformers  : {n_stages}")
for s, t in enumerate(model.transformers):
    n_layers = len(t.layers)
    print(f"    stage {s}: {n_layers} layers")

# ── Parameter count ───────────────────────────────────────────────────────────
total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Total params  : {total_params / 1e6:.1f}M")

# ── FF module structure (stage 0, layer 0) ────────────────────────────────────
print("\n--- FF module structure (stage 0, layer 0) ---")
attn0, ff0 = model.transformers[0].layers[0]
for i, mod in enumerate(ff0):
    print(f"  ff[{i}]: {mod}")

down_proj = ff0[4]
print(f"\n  Down-proj (ff[4])  : in={down_proj.weight.shape[1]}  out={down_proj.weight.shape[0]}")
print(f"  Max abs weight     : {down_proj.weight.data.abs().max().item():.4f}")

# ── Confirm all stages have uniform FF structure ──────────────────────────────
print("\n--- Checking all stages/layers for ff[4] = Linear ---")
ok = True
for s, t in enumerate(model.transformers):
    for l, (_, ff) in enumerate(t.layers):
        m = ff[4]
        if not isinstance(m, torch.nn.Linear):
            print(f"  UNEXPECTED: stage={s} layer={l} ff[4] is {type(m).__name__}")
            ok = False
if ok:
    print("  All ff[4] modules are nn.Linear ✓")

# ── Vocabulary check ──────────────────────────────────────────────────────────
print("\n--- Vocabulary ---")
BASE_TO_TOKEN = {"A": 1, "T": 2, "C": 3, "G": 4}
print(f"  Encoding: {BASE_TO_TOKEN}")

seq = "ATGGATGATGATATCGCCGCG"
ids = torch.tensor([BASE_TO_TOKEN[b] for b in seq if b in BASE_TO_TOKEN],
                   dtype=torch.long).unsqueeze(0)
print(f"  Test sequence ({len(seq)} bp) → {ids.shape[1]} tokens")

# ── State dict key patterns ───────────────────────────────────────────────────
print("\n--- State dict key patterns (first 12 transformers keys) ---")
sd = model.state_dict()
t_keys = [k for k in sd if k.startswith("transformers")][:12]
for k in t_keys:
    print(f"  {k}  {list(sd[k].shape)}")

# ── Baseline perplexity on a short phage sequence ────────────────────────────
print("\n--- Baseline perplexity test ---")
try:
    from probes.dna_probes import PROBES
    probe = PROBES["phage"][:600]
    ids_probe = torch.tensor(
        [BASE_TO_TOKEN[b] for b in probe if b in BASE_TO_TOKEN],
        dtype=torch.long
    ).unsqueeze(0)
    with torch.no_grad():
        loss = model(ids_probe, return_loss=True)
    ppl = torch.exp(loss).item()
    print(f"  Probe    : phage (first 600 bp)")
    print(f"  Loss     : {loss.item():.4f}")
    print(f"  PPL      : {ppl:.2f}")
except Exception as e:
    print(f"  (skipped — {e})")

print("\nDone.")
