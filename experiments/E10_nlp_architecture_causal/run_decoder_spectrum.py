#!/usr/bin/env python
"""
E10 ARM A / Step D2 -- decoder singleton causal spectrum.

For each decoder, measures the intrinsic causal-LM endpoint (mean per-token NLL on the
frozen 100-window WikiText-2 set) under:
  - baseline (alpha=1.0)
  - each frozen structural row at alpha=0.0            (the singleton spectrum)
  - the rank-1 row additionally at alpha=0.5           (retained dose point)
  - each of 5 frozen same-layer control rows at alpha=0.0

Rows, controls, endpoint, alpha grid and batch-size policy all come from the locked v2
prereg (PREREG_E10_nlp_architecture_causal_v2.md, sha256 3e4b991d...) and
DECODER_INTERVENTION_FREEZE.md. Nothing is selected or tuned here.

Streaming design: the outer loop is over BATCHES, the inner loop over conditions. This
keeps only one batch of baseline log-probs resident (needed for logit KL) instead of the
whole corpus, and makes every condition paired on identical contexts by construction.

Usage:  python run_decoder_spectrum.py --model llama
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SALVAGE = HERE.parents[1]
ROOT = SALVAGE  # repo root; harness paths are written relative to it
sys.path.insert(0, str(HERE))

import e10_lib as L  # noqa: E402

N_WINDOWS = 100
MAX_TOKENS = 512
PATTERN = "model.layers.{i}.mlp.down_proj"

# Frozen by DECODER_INTERVENTION_FREEZE.md v2 (ranked by exact ||U_k||_F / layer median).
PANEL = {
    "llama": dict(
        repo="huggyllama/llama-7b", revision=None, batch_size=8,
        rows=[(2, 3968)],
        controls=(2, [892, 2034, 2379, 3729, 3751]),
    ),
    "mistral": dict(
        repo="mistralai/Mistral-7B-v0.1", revision=None, batch_size=8,
        rows=[(1, 2070)],
        controls=(1, [190, 305, 746, 1912, 2687]),
    ),
    "olmo": dict(
        repo="allenai/OLMo-7B-0724-hf", revision=None, batch_size=4,
        rows=[(24, 269), (7, 269), (1, 269), (2, 269)],
        controls=(24, [292, 437, 2877, 2908, 3160]),
    ),
    "phi3": dict(
        repo="microsoft/Phi-3-mini-4k-instruct",
        revision="f39ac1d28e925b323eae81227eaba4464caced4e", batch_size=8,
        rows=[(2, 525), (2, 1693), (2, 1113), (4, 525), (4, 1113), (4, 1693)],
        controls=(2, [356, 2345, 2759, 2830, 2983]),
    ),
    "qwen25": dict(
        repo="Qwen/Qwen2.5-7B",
        revision="d149729398750b98c0af14eb82c78cfe92750796", batch_size=2,
        rows=[(26, 458)],
        controls=(26, [62, 1257, 2926, 3419, 3553]),
    ),
}


def build_conditions(spec):
    """Frozen condition list. Order is fixed by the freeze doc, not by any response."""
    conds = []
    for rank, (layer, row) in enumerate(spec["rows"]):
        conds.append(dict(kind="target", rank=rank + 1, layer=layer, row=row, alpha=0.0))
    # alpha=0.5 retained only for the rank-1 row (v2 prereg, Intervention semantics)
    l1, r1 = spec["rows"][0]
    conds.append(dict(kind="target_half", rank=1, layer=l1, row=r1, alpha=0.5))
    clayer, crows = spec["controls"]
    for cr in crows:
        conds.append(dict(kind="control", rank=None, layer=clayer, row=cr, alpha=0.0))
    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PANEL))
    args = ap.parse_args()
    spec = PANEL[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {"dtype": torch.float32}
    if spec["revision"]:
        kw["revision"] = spec["revision"]

    print(f"loading {spec['repo']} (float32) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec["repo"], **({"revision": spec["revision"]}
                                                          if spec["revision"] else {}))
    model = AutoModelForCausalLM.from_pretrained(spec["repo"], **kw)
    model.eval().to(device)
    resolved = getattr(model.config, "_commit_hash", None)
    print(f"  loaded in {time.time()-t0:.0f}s  resolved_revision={resolved}")

    windows = L.build_windows(tok, N_WINDOWS, MAX_TOKENS)
    batches = L.batch_windows(windows, spec["batch_size"])
    print(f"  {len(windows)} windows x {MAX_TOKENS} tok -> {len(batches)} batches "
          f"(batch_size={spec['batch_size']})")

    conds = build_conditions(spec)
    print(f"  {len(conds)} intervention conditions + baseline")

    # accumulators: per-condition list of (sum_nll, n_tok), plus KL/entropy sums
    base_pb, cond_pb = [], [[] for _ in conds]
    cond_kl = [[] for _ in conds]
    cond_ent = [[] for _ in conds]
    base_ent = []

    t0 = time.time()
    for bi, ids in enumerate(batches):
        ids = ids.to(device)
        s, n, base_lp, _, ent = L.causal_lm_batch(model, ids, None)
        base_pb.append((s, n))
        base_ent.append(ent)

        for ci, c in enumerate(conds):
            with L.masked(model, PATTERN, [(c["layer"], c["row"])], [c["alpha"]]):
                s2, n2, _, kl, ent2 = L.causal_lm_batch(model, ids, base_lp)
            cond_pb[ci].append((s2, n2))
            cond_kl[ci].append(kl)
            cond_ent[ci].append(ent2)
        del base_lp
        torch.cuda.empty_cache()
        print(f"  batch {bi+1}/{len(batches)}  ({time.time()-t0:.0f}s)", flush=True)

    base_nll = L.weighted_mean(base_pb)
    results = []
    for ci, c in enumerate(conds):
        nll = L.weighted_mean(cond_pb[ci])
        results.append({
            **c,
            "nll": nll,
            "delta_nll": nll - base_nll,
            "relative_pct_change": 100.0 * (nll - base_nll) / base_nll,
            "perplexity": float(torch.exp(torch.tensor(nll))),
            "mean_logit_kl_vs_baseline": sum(cond_kl[ci]) / len(cond_kl[ci]),
            "mean_next_token_entropy": sum(cond_ent[ci]) / len(cond_ent[ci]),
            "per_batch": cond_pb[ci],
        })

    out = {
        "model": args.model, "repo": spec["repo"],
        "requested_revision": spec["revision"], "resolved_revision": resolved,
        "prereg": "PREREG_E10_nlp_architecture_causal_v2.md",
        "prereg_sha256": "3e4b991d71b38cfe15550a57c10c73323cb3b751ba3c768d6c95fc4989833a8a",
        "endpoint": "mean per-token causal-LM NLL, WikiText-2-raw-v1 test",
        "n_windows": N_WINDOWS, "max_tokens": MAX_TOKENS,
        "batch_size": spec["batch_size"], "dtype": "float32", "seed": L.SEED,
        "baseline_nll": base_nll,
        "baseline_perplexity": float(torch.exp(torch.tensor(base_nll))),
        "baseline_mean_next_token_entropy": sum(base_ent) / len(base_ent),
        "baseline_per_batch": base_pb,
        "frozen_rows": spec["rows"], "control_layer": spec["controls"][0],
        "control_rows": spec["controls"][1],
        "conditions": results,
    }
    p = ROOT / "results" / f"e10_decoder_spectrum_{args.model}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))

    print(f"\nbaseline NLL = {base_nll:.6f}  (ppl {out['baseline_perplexity']:.3f})")
    for r in results:
        tag = f"{r['kind']}" + (f" rank{r['rank']}" if r["rank"] else "")
        print(f"  L{r['layer']}/r{r['row']} alpha={r['alpha']} [{tag}]: "
              f"dNLL={r['delta_nll']:+.6f} ({r['relative_pct_change']:+.3f}%)  "
              f"KL={r['mean_logit_kl_vs_baseline']:.6f}")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
