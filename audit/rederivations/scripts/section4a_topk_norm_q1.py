#!/usr/bin/env python
"""Round-2 Section 4a, prioritized sub-question: does the candidate's spectral
concentration (q1) remain exceptional conditional on being high-norm, or does q1 simply
track norm? For each model: compute ||U_k||_F for every row in the candidate's layer
(Gram-identity, cheap), take the candidate + top-5-by-norm OTHER rows, compute exact q1
for all 6 via full SVD (row_spectral_metrics), compare.

Batch 1: the 12 E11-panel models (reuses E11_scale_ladder/run_model.py's loader/PANEL
unchanged). Appends to audit/rederivations/topk_norm_controls.csv and
audit/rederivations/topk_norm_q1_summary.csv; checkpointed so it can be resumed/extended to more
models without recomputing finished ones.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
E11_DIR = ROOT / "experiments/frozen/E11_scale_ladder"
E5_DIR = ROOT / "experiments/frozen/E5_dimensionality"
E7_DIR = ROOT / "experiments/frozen/E7_exact_dimensionality"
sys.path.insert(0, str(E11_DIR))
sys.path.insert(0, str(E5_DIR))
sys.path.insert(0, str(E7_DIR))

import run_model as rm  # noqa: E402
from dimensionality_lib import exact_uk_all_rows  # noqa: E402
from spectral_lib import row_spectral_metrics  # noqa: E402


def _patch_rope_default():
    """transformers 5.15.1 dropped the 'default' key from ROPE_INIT_FUNCTIONS (the
    no-scaling base-RoPE case is now computed inline in the library's own RotaryEmbedding
    classes rather than via this dict). EuroBERT's frozen remote code
    (modeling_eurobert.py's EuroBertRotaryEmbedding.__init__) still indexes
    ROPE_INIT_FUNCTIONS['default'] directly -- KeyError otherwise. This restores exactly
    the standard, unscaled RoPE computation (attention_factor=1.0, no scaling applied),
    faithful to what "default" always meant; it does not change any science, only
    restores a removed no-op entry so old remote code can still find it."""
    import torch as _torch
    from transformers import modeling_rope_utils as _rope_utils

    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, layer_type=None, **_):
        base = getattr(config, "rope_theta", 10000.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (_torch.arange(0, dim, 2, dtype=_torch.int64).to(
            device=device, dtype=_torch.float) / dim))
        return inv_freq, 1.0

    _rope_utils.ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)

OUT_CSV = ROOT / "audit/rederivations/topk_norm_controls.csv"
SUMMARY_CSV = ROOT / "audit/rederivations/topk_norm_q1_summary.csv"
CANDIDATE_CSV = ROOT / "results/experiments/E11/scale_ladder.csv"
CONTROLS_CSV = ROOT / "results/experiments/E11/scale_ladder_controls.csv"

FIELDS_MAIN = ["model", "layer", "candidate_row", "candidate_norm", "layer_median_norm",
               "rank2_row", "rank3_row", "rank4_row", "rank5_row", "rank6_row",
               "rank2_norm", "rank3_norm", "rank4_norm", "rank5_norm", "rank6_norm",
               "ratio_rank2_to_candidate", "existing_random_control_rows",
               "existing_control_norm_percentiles"]
FIELDS_Q1 = ["model", "coord_role", "layer", "row", "norm", "q1", "pr_spec"]


def load_checkpoint(path, key_field="model"):
    done = set()
    if path.exists():
        for r in csv.DictReader(open(path)):
            done.add(r[key_field])
    return done


def append_rows(path, fields, rows):
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def main():
    _patch_rope_default()
    candidates = {r["model"]: r for r in csv.DictReader(open(CANDIDATE_CSV))}
    controls_by_model = {}
    for r in csv.DictReader(open(CONTROLS_CSV)):
        controls_by_model.setdefault(r["model"], []).append(r)

    key_to_full = {"qwen25-0.5b": "Qwen/Qwen2.5-0.5B", "qwen25-1.5b": "Qwen/Qwen2.5-1.5B",
                   "qwen25-3b": "Qwen/Qwen2.5-3B", "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
                   "smollm2-360m": "HuggingFaceTB/SmolLM2-360M", "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
                   "generator-prok-1.2b": "GenerTeam/GENERator-v2-prokaryote-1.2b-base",
                   "generator-prok-3b": "GenerTeam/GENERator-v2-prokaryote-3b-base",
                   "eurobert-210m": "EuroBERT/EuroBERT-210m", "eurobert-610m": "EuroBERT/EuroBERT-610m",
                   "eurobert-2.1b": "EuroBERT/EuroBERT-2.1B", "modernbert-large": "answerdotai/ModernBERT-large"}

    done_main = load_checkpoint(OUT_CSV)
    done_q1 = load_checkpoint(SUMMARY_CSV)

    for key in rm.PANEL_ORDER:
        full_name = key_to_full[key]
        if full_name in done_main and full_name in done_q1:
            print(f"skip {full_name} (checkpointed)")
            continue

        t0 = time.time()
        spec = rm.PANEL[key]
        model, tok, cfg, device, dtype_used, resolved_rev = rm.load_model_and_tokenizer(spec)
        stack = rm.get_decoder_stack(model, spec["model_class"])

        cand = candidates[full_name]
        layer, cand_row = int(cand["layer"]), int(cand["row"])
        layer_mod = stack[layer]
        Wg, Wu, Wd = rm.gate_up_down_weights(layer_mod, spec["ffn_kind"])

        norms = exact_uk_all_rows(Wg, Wu, Wd, device=device).numpy()
        d_model = norms.shape[0]
        cand_norm = float(norms[cand_row])
        layer_median = float(np.median(norms))

        order = np.argsort(-norms)  # descending
        top_rows = [int(r) for r in order if r != cand_row][:5]
        top_norms = [float(norms[r]) for r in top_rows]

        existing_ctrl = controls_by_model.get(full_name, [])
        existing_ctrl_rows = [int(c["row"]) for c in existing_ctrl]
        existing_ctrl_pct = [float(100 * (norms <= norms[r]).sum() / d_model) for r in existing_ctrl_rows]

        row_main = dict(model=full_name, layer=layer, candidate_row=cand_row,
                         candidate_norm=cand_norm, layer_median_norm=layer_median,
                         **{f"rank{i+2}_row": top_rows[i] for i in range(5)},
                         **{f"rank{i+2}_norm": top_norms[i] for i in range(5)},
                         ratio_rank2_to_candidate=top_norms[0] / cand_norm if cand_norm else float("nan"),
                         existing_random_control_rows=";".join(map(str, existing_ctrl_rows)),
                         existing_control_norm_percentiles=";".join(f"{p:.1f}" for p in existing_ctrl_pct))
        append_rows(OUT_CSV, FIELDS_MAIN, [row_main])

        # exact q1 for candidate + top-5-by-norm rows
        q1_rows = []
        Wg64, Wu64 = Wg.double(), Wu.double()
        for role, row in [("candidate", cand_row)] + [(f"rank{i+2}_by_norm", r) for i, r in enumerate(top_rows)]:
            d_row = Wd[row].double()
            metrics, _ = row_spectral_metrics(Wg64, Wu64, d_row, device="cpu")
            q1_rows.append(dict(model=full_name, coord_role=role, layer=layer, row=row,
                                 norm=metrics.frob_norm, q1=metrics.q1, pr_spec=metrics.pr_spec))
        append_rows(SUMMARY_CSV, FIELDS_Q1, q1_rows)

        cand_q1 = q1_rows[0]["q1"]
        topk_q1s = [r["q1"] for r in q1_rows[1:]]
        dt = time.time() - t0
        print(f"{full_name}: candidate q1={cand_q1:.4f} (norm={cand_norm:.2f})  "
              f"top5-by-norm q1s={[f'{q:.4f}' for q in topk_q1s]} "
              f"(norms={[f'{n:.2f}' for n in top_norms]})  "
              f"candidate_q1_rank_among_6={sorted([cand_q1]+topk_q1s, reverse=True).index(cand_q1)+1}  "
              f"[{dt:.1f}s]", flush=True)

        del model, Wg, Wu, Wd, Wg64, Wu64, norms
        if device == "cuda":
            torch.cuda.empty_cache()

    print("=== BATCH COMPLETE ===")


if __name__ == "__main__":
    main()
