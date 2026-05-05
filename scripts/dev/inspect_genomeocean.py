"""
inspect_genomeocean.py — Inspect GenomeOcean model architecture.

Run inside the genomeocean apptainer:
  apptainer exec --nv --cleanenv \\
    --bind /work,/tmp \\
    --env HF_HOME=/work/11034/atzanakak/ls6/huggingface/.hf-cache \\
    --env HF_HUB_CACHE=/work/11034/atzanakak/ls6/huggingface/.hf-cache/hub \\
    --env PYTHONPATH=/work/11034/atzanakak/glm_super_weight/genomic-super-weights \\
    /work/11034/atzanakak/genomeocean/apptainer/genomeocean.sif \\
    python3 /work/11034/atzanakak/glm_super_weight/genomic-super-weights/inspect_genomeocean.py

or from the repo root with run.sh (if grlm env has genomeocean installed):
  bash run.sh inspect_genomeocean.py

Outputs:
  - Model class name
  - Number of layers
  - Hidden size
  - MLP module names (to confirm down_proj_pattern in configs/genomeocean.yaml)
"""
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "pGenomeOcean/GenomeOcean-4B"

print(f"Loading tokenizer from {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"  Tokenizer class : {type(tok).__name__}")
print(f"  Vocab size      : {tok.vocab_size}")
print(f"  Padding side    : {tok.padding_side}")

print(f"\nLoading model (cpu, float32 for inspection) ...")
m = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="cpu",
)
print(f"  Model class     : {type(m).__name__}")

names = [n for n, _ in m.named_modules()]

# --- MLP candidates ---
mlp = [n for n in names if any(k in n.lower() for k in ["mlp", "down", "fc2", "dense_4h", "proj"])]
print("\n--- MLP candidates (first 40) ---")
print("\n".join(mlp[:40]))

# --- Number of layers ---
indices = set(re.findall(r'layers\.(\d+)', " ".join(names)))
if indices:
    num_layers = max(int(i) for i in indices) + 1
    print(f"\nnum_layers : {num_layers}")

# --- Hidden / intermediate sizes ---
cfg = m.config
print(f"hidden_size        : {getattr(cfg, 'hidden_size', 'N/A')}")
print(f"intermediate_size  : {getattr(cfg, 'intermediate_size', 'N/A')}")
print(f"num_hidden_layers  : {getattr(cfg, 'num_hidden_layers', 'N/A')}")
print(f"num_attention_heads: {getattr(cfg, 'num_attention_heads', 'N/A')}")
print(f"max_position_embeds: {getattr(cfg, 'max_position_embeddings', 'N/A')}")
print(f"model_type (config): {getattr(cfg, 'model_type', 'N/A')}")
print(f"architectures      : {getattr(cfg, 'architectures', 'N/A')}")

# --- Quick tokenization check ---
seq = "ATGGATGATGATATCGCCGCGCTCGTCGTCGAC"
enc = tok(seq, return_tensors="pt", add_special_tokens=False)
print(f"\nTest sequence ({len(seq)} bp) → {enc['input_ids'].shape[1]} tokens")
print(f"  Tokens per bp  : {enc['input_ids'].shape[1] / len(seq):.2f}")
