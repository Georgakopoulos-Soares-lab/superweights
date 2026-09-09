"""
experiments/E10_nlp_architecture_causal/rank_encoder_rows.py

E10 Phase 2 (Case B): rank every gated-FFN output row of MosaicBERT / ModernBERT by the
exact E7 spectral score (q1), using the identical adapters and row_spectral_metrics already
used by E7/E8 -- no new structural metric is introduced. Weight-only; no forward pass.

For each layer i in range(n_layers), each row k in range(d_model):
    q1(layer=i, row=k) = spectral_lib.row_spectral_metrics(W_gate_i, W_up_i, W_down_i[k])

Ranks all (layer, row) pairs descending by q1, writes the full ranking plus the top-K=10
selection to results/e10_encoder_row_ranking_<model>.json.

Usage:
  python3 rank_encoder_rows.py --model mosaicbert
  python3 rank_encoder_rows.py --model modernbert
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE.parent
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E7_exact_dimensionality"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E8_encoder_decoder"))

from spectral_lib import row_spectral_metrics  # noqa: E402  (E7, reused unmodified)
from run_detection_and_spectral import adapter_mosaicbert, adapter_modernbert  # noqa: E402

CANONICAL = {"mosaicbert": (9, 287), "modernbert": (15, 251)}
TOP_K = 10


def load_mosaicbert():
    from transformers import AutoModelForMaskedLM, BertTokenizer
    repo = "mosaicml/mosaic-bert-base"
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    return model, repo, model.config.num_hidden_layers, adapter_mosaicbert


def load_modernbert():
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    repo = "answerdotai/ModernBERT-base"
    AutoTokenizer.from_pretrained(repo)  # not used for ranking; kept for parity/logging only
    model = AutoModelForMaskedLM.from_pretrained(repo, torch_dtype=torch.float32)
    model.eval()
    return model, repo, model.config.num_hidden_layers, adapter_modernbert


def rank_model(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = {"mosaicbert": load_mosaicbert, "modernbert": load_modernbert}[name]
    t0 = time.time()
    model, repo, n_layers, adapter_fn = loader()
    resolved_rev = getattr(model.config, "_commit_hash", None)
    print(f"loaded {repo} ({n_layers} layers) in {time.time()-t0:.1f}s, resolved_revision={resolved_rev}")

    all_rows = []
    for layer in range(n_layers):
        Wg, Wu, Wd = adapter_fn(model, layer)
        Wg, Wu, Wd = Wg.detach().cpu(), Wu.detach().cpu(), Wd.detach().cpu()
        d_model = Wd.shape[0]
        t_layer = time.time()
        for row in range(d_model):
            metrics, _ = row_spectral_metrics(Wg, Wu, Wd[row], device=device)
            all_rows.append(dict(layer=layer, row=row, q1=metrics.q1, pr_spec=metrics.pr_spec,
                                  stable_rank=metrics.stable_rank, frob_norm=metrics.frob_norm))
        print(f"  layer {layer}: {d_model} rows scored in {time.time()-t_layer:.1f}s")

    all_rows.sort(key=lambda r: r["q1"], reverse=True)
    for rank, r in enumerate(all_rows):
        r["rank"] = rank

    top_k = all_rows[:TOP_K]
    canon_layer, canon_row = CANONICAL[name]
    canon_rank = next(r["rank"] for r in all_rows if r["layer"] == canon_layer and r["row"] == canon_row)
    canon_in_topk = canon_rank < TOP_K

    result = dict(
        model=name, repo=repo, resolved_revision=resolved_rev, n_layers=n_layers,
        n_rows_scored=len(all_rows), top_k=TOP_K,
        canonical_row=dict(layer=canon_layer, row=canon_row, rank=canon_rank, in_top_k=canon_in_topk,
                            q1=next(r["q1"] for r in all_rows if r["layer"] == canon_layer and r["row"] == canon_row)),
        top_k_rows=top_k,
        full_ranking=all_rows,
    )

    out_path = ROOT / "results" / f"e10_encoder_row_ranking_{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n{name}: canonical (L{canon_layer}/r{canon_row}) rank={canon_rank} "
          f"(0=best of {len(all_rows)}), in_top_{TOP_K}={canon_in_topk}")
    print(f"top {TOP_K}:")
    for r in top_k:
        print(f"  rank={r['rank']:2d} L{r['layer']}/r{r['row']}: q1={r['q1']:.6f}")
    print(f"wrote {out_path}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["mosaicbert", "modernbert"])
    args = p.parse_args()
    rank_model(args.model)


if __name__ == "__main__":
    main()
