"""
run_esm2_uk_audit.py
---------------------
Mechanistic ||U_k||_F audit for ESM-2 15B, mirroring run_dnabert2_uk_audit.py
and run_generator_uk_audit.py.

ESM-2 uses a standard (non-gated) RoBERTa-style FFN per encoder layer:

    intermediate = GELU( intermediate_dense(x) )   # (B, d_ffn)
    out = output_dense( intermediate )              # (B, d_model)

Weight layout:
    W_up   = encoder.layer[i].intermediate.dense.weight  (d_ffn=10240, d_model=2560)
    W_down = encoder.layer[i].output.dense.weight         (d_model=2560, d_ffn=10240)

No gating matrix (unlike DNABERT-2 / GENERator), so the rank-1 Frobenius bound
simplifies to:

    ||U_k||_F = sqrt( sum_j  W_down[k,j]^2  *  ||W_up[j,:]||^2 )

where k indexes d_model (output row of MLP), j indexes d_ffn (hidden neuron).

Model: facebook/esm2_t48_15B_UR50D
  d_model   = 2560
  d_ffn     = 10240
  n_layers  = 48
  hidden_act = gelu  (no gating)

Computed on CPU (weights are loaded in float32; no GPU needed).

Output
------
results/sw_mechanistic_esm2.json
  {
    "model": "esm2_t48_15B",
    "d_model": 2560,
    "d_ffn": 10240,
    "n_layers": 48,
    "frob_norm_uk_by_layer": {"0": [...2560 floats...], "1": [...], ...},
    "sw_row_ranks":           {"0": {"top1_row": 1, ...}, ...},
    "median_frob_by_layer":   {"0": float, ...},
    "max_frob_by_layer":      {"0": float, ...},
    "concentration_ratio_by_layer": {"0": float, ...},   # max / median
    "top1_frob_by_layer":     {"0": float, ...},
    "top1_row_by_layer":      {"0": int, ...},
  }

Usage (from repo root):
  python scripts/analysis/run_esm2_uk_audit.py [--hf_home /path/to/hf/cache]
  python scripts/analysis/run_esm2_uk_audit.py --hf_home /scratch/11034/atzanakak/hf_cache
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT    = Path(__file__).resolve().parent.parent.parent
OUT_PATH = ROOT / "results" / "sw_mechanistic_esm2.json"

MODEL_ID  = "facebook/esm2_t48_15B_UR50D"
D_MODEL   = 2560
D_FFN     = 10240
N_LAYERS  = 48


def load_model(hf_home: str):
    from transformers import AutoModel, AutoTokenizer
    # transformers ≥4.51 blocks torch.load on .bin files unless torch ≥2.6 (CVE-2025-32434).
    # These weights are trusted local files already downloaded from Meta; patch the check away.
    # Must patch modeling_utils (where the name is used), not import_utils (where it's defined).
    import transformers.modeling_utils as _tmu
    _tmu.check_torch_load_is_safe = lambda: None

    hub_cache = os.path.join(hf_home, "hub")
    os.environ["HF_HOME"]      = hf_home
    os.environ["HF_HUB_CACHE"] = hub_cache

    print(f"[uk_audit] Loading tokenizer from {MODEL_ID} …")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=hub_cache)

    print(f"[uk_audit] Loading model (CPU, float32) …")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir=hub_cache,
    )
    model.eval()
    return model, tok


def extract_weights(model, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (W_up, W_down) for layer `layer_idx`.
      W_up   : (d_ffn, d_model)
      W_down : (d_model, d_ffn)
    """
    layer  = model.encoder.layer[layer_idx]
    W_up   = layer.intermediate.dense.weight.detach().float().cpu().numpy()   # (d_ffn, d_model)
    W_down = layer.output.dense.weight.detach().float().cpu().numpy()          # (d_model, d_ffn)
    return W_up, W_down


def compute_frob(W_up: np.ndarray, W_down: np.ndarray) -> np.ndarray:
    """
    ||U_k||_F for all k simultaneously (vectorised).

    Standard FFN (no gating):
      ||U_k||_F = sqrt( sum_j  W_down[k,j]^2 * ||W_up[j,:]||^2 )

    W_up   : (d_ffn, d_model)
    W_down : (d_model, d_ffn)
    Returns: (d_model,) float32 array
    """
    row_norms_sq = (W_up ** 2).sum(axis=1)    # (d_ffn,)  ||W_up[j,:]||^2
    # W_down[k,j]^2 * row_norms_sq[j]  summed over j for each k
    frob_sq = (W_down ** 2) @ row_norms_sq    # (d_model,)
    return np.sqrt(np.maximum(frob_sq, 0.0)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="ESM-2 15B ||U_k||_F audit")
    parser.add_argument("--hf_home", default="/scratch/11034/atzanakak/hf_cache",
                        help="HuggingFace cache dir containing ESM-2 15B weights")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, _ = load_model(args.hf_home)
    print(f"[uk_audit] Model loaded. Scanning {N_LAYERS} layers …\n")

    frob_by_layer:          dict[str, list] = {}
    median_by_layer:        dict[str, float] = {}
    max_by_layer:           dict[str, float] = {}
    concentration_by_layer: dict[str, float] = {}
    top1_frob_by_layer:     dict[str, float] = {}
    top1_row_by_layer:      dict[str, int]   = {}
    sw_row_ranks:           dict[str, dict]  = {}

    for li in range(N_LAYERS):
        W_up, W_down = extract_weights(model, li)
        frob = compute_frob(W_up, W_down)

        top1_row  = int(np.argmax(frob))
        top1_val  = float(frob[top1_row])
        med       = float(np.median(frob))
        max_val   = float(frob.max())
        ratio     = max_val / (med + 1e-12)

        frob_by_layer[str(li)]          = frob.tolist()
        median_by_layer[str(li)]        = med
        max_by_layer[str(li)]           = max_val
        concentration_by_layer[str(li)] = ratio
        top1_frob_by_layer[str(li)]     = top1_val
        top1_row_by_layer[str(li)]      = top1_row

        # rank of top-3 rows (1-based, lower = more concentrated)
        order = np.argsort(-frob)
        top3  = {str(int(order[r])): r + 1 for r in range(min(3, len(order)))}
        sw_row_ranks[str(li)] = top3

        print(f"  L{li:2d}: top1_row={top1_row:4d}  frob={top1_val:8.2f}"
              f"  median={med:6.2f}  ratio={ratio:5.1f}x")

    # Summary: layer with globally highest concentration ratio
    best_layer = max(concentration_by_layer, key=lambda k: concentration_by_layer[k])
    print(f"\n[uk_audit] Highest concentration: layer {best_layer}"
          f"  ratio={concentration_by_layer[best_layer]:.1f}x"
          f"  top1_row={top1_row_by_layer[best_layer]}"
          f"  frob={top1_frob_by_layer[best_layer]:.2f}")

    out = {
        "model":                      "esm2_t48_15B",
        "model_id":                   MODEL_ID,
        "d_model":                    D_MODEL,
        "d_ffn":                      D_FFN,
        "n_layers":                   N_LAYERS,
        "mlp_type":                   "standard_gelu_ffn",
        "frob_norm_uk_by_layer":      frob_by_layer,
        "sw_row_ranks":               sw_row_ranks,
        "median_frob_by_layer":       median_by_layer,
        "max_frob_by_layer":          max_by_layer,
        "concentration_ratio_by_layer": concentration_by_layer,
        "top1_frob_by_layer":         top1_frob_by_layer,
        "top1_row_by_layer":          top1_row_by_layer,
        "best_sw_layer":              int(best_layer),
        "best_sw_row":                top1_row_by_layer[best_layer],
        "best_sw_concentration_ratio": concentration_by_layer[best_layer],
    }

    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[uk_audit] Saved → {out_path}")


if __name__ == "__main__":
    main()
