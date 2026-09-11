#!/usr/bin/env python
"""
E10b Phase 4 -- Phi-3 multi-row tomography measurement.

Measures causal-LM NLL (+ logit KL, entropy) on the frozen 100-window WikiText-2 set
(identical construction to E10's Phi-3 singleton run -- same seed/tokenizer/count reproduces
the identical windows) under every mask in the locked design (masks_phi3.json), at
epsilon in {0.5, 1.0}, over Phi-3's frozen 6-row structural basis.

Reuses E10's e10_lib.py unmodified (causal_lm_batch, masked, build_windows, batch_windows,
alphas_for_mask). Streaming batch-outer/condition-inner, exactly as run_decoder_spectrum.py,
so every condition is paired on identical contexts and only one batch of baseline log-probs
is resident at a time.

Governing lock: PREREG_E10b_phi3_tomography.md (sha256 098fd52c...).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E10 = HERE.parent / "E10_nlp_architecture_causal"
sys.path.insert(0, str(E10))
sys.path.insert(0, str(HERE))

import e10_lib as L  # noqa: E402

REPO = "microsoft/Phi-3-mini-4k-instruct"
REVISION = "f39ac1d28e925b323eae81227eaba4464caced4e"
PATTERN = "model.layers.{i}.mlp.down_proj"
N_WINDOWS = 100
MAX_TOKENS = 512
BATCH_SIZE = 8
EPSILONS = [0.5, 1.0]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    design = json.loads((HERE / "masks_phi3.json").read_text())
    basis = [tuple(b) for b in design["basis"]]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {REPO} @ {REVISION} (float32) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REVISION, dtype=torch.float32)
    model.eval().to(device)
    resolved = getattr(model.config, "_commit_hash", None)
    print(f"  loaded in {time.time()-t0:.0f}s  resolved_revision={resolved} "
          f"(expected {REVISION})")
    assert resolved == REVISION, "resolved revision does not match the locked pin"

    windows = L.build_windows(tok, N_WINDOWS, MAX_TOKENS)  # seed=42 default, same as E10
    batches = L.batch_windows(windows, BATCH_SIZE)
    print(f"  {len(windows)} windows x {MAX_TOKENS} tok -> {len(batches)} batches "
          f"(batch_size={BATCH_SIZE})")

    conds = []
    for pool in ("singletons", "fit", "calibration", "held_out"):
        for mi, a in enumerate(design["pools"][pool]):
            for eps in EPSILONS:
                conds.append(dict(pool=pool, mask_index=mi, a=a, epsilon=eps))
    print(f"  {len(conds)} conditions ({len(conds)//len(EPSILONS)} masks x {len(EPSILONS)} eps)")

    base_pb = []
    cond_pb = [[] for _ in conds]
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
            alphas = L.alphas_for_mask(c["a"], c["epsilon"])
            active = [(coord, al) for coord, al, ai in zip(basis, alphas, c["a"]) if ai]
            coords = [x[0] for x in active]
            als = [x[1] for x in active]
            with L.masked(model, PATTERN, coords, als):
                s2, n2, _, kl, ent2 = L.causal_lm_batch(model, ids, base_lp)
            cond_pb[ci].append((s2, n2))
            cond_kl[ci].append(kl)
            cond_ent[ci].append(ent2)
        del base_lp
        torch.cuda.empty_cache()
        print(f"  batch {bi+1}/{len(batches)}  ({time.time()-t0:.0f}s)", flush=True)

    base_nll = L.weighted_mean(base_pb)
    pools_out = {p: [] for p in ("singletons", "fit", "calibration", "held_out")}
    for ci, c in enumerate(conds):
        nll = L.weighted_mean(cond_pb[ci])
        rec = next((r for r in pools_out[c["pool"]] if r["mask_index"] == c["mask_index"]), None)
        if rec is None:
            rec = {"mask_index": c["mask_index"], "a": c["a"]}
            pools_out[c["pool"]].append(rec)
        eps = c["epsilon"]
        rec[f"nll_eps{eps}"] = nll
        rec[f"dloss_eps{eps}"] = nll - base_nll
        rec[f"per_batch_eps{eps}"] = cond_pb[ci]
        rec[f"kl_eps{eps}"] = sum(cond_kl[ci]) / len(cond_kl[ci])
        rec[f"entropy_eps{eps}"] = sum(cond_ent[ci]) / len(cond_ent[ci])

    out = {
        "model": "phi3", "repo": REPO,
        "requested_revision": REVISION, "resolved_revision": resolved,
        "prereg": "PREREG_E10b_phi3_tomography.md",
        "prereg_sha256": "098fd52cd398b2f574490fb64dee3842aad057b2a367921bc8ba38926338cda0",
        "endpoint": "mean per-token causal-LM NLL, WikiText-2-raw-v1 test",
        "basis": [list(b) for b in basis],
        "n_windows": N_WINDOWS, "max_tokens": MAX_TOKENS, "batch_size": BATCH_SIZE,
        "dtype": "float32", "seed": L.SEED, "epsilons": EPSILONS,
        "baseline_nll": base_nll,
        "baseline_perplexity": float(torch.exp(torch.tensor(base_nll))),
        "baseline_mean_next_token_entropy": sum(base_ent) / len(base_ent),
        "baseline_per_batch": base_pb,
        "responses": {"pools": pools_out},
    }
    p = ROOT / "results" / "e10b_phi3_tomography_responses.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nbaseline NLL = {base_nll:.6f}")
    print("singleton dloss (eps=1.0), by basis index:")
    for r in sorted(pools_out["singletons"], key=lambda r: r["mask_index"]):
        i = r["a"].index(1)
        print(f"  idx {i} L{basis[i][0]}/r{basis[i][1]}: {r['dloss_eps1.0']:+.6f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
