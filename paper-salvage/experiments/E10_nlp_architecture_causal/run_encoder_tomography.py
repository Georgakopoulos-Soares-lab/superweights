#!/usr/bin/env python
"""
E10 ARM B -- encoder mechanistic tomography measurements (DNABERT-2 protocol replicated).

Measures pretrained MLM loss on the frozen 256-window WikiText-2 set under every mask in
the frozen design (singletons / fit / calibration / held_out) at epsilon in {0.5, 1.0},
using alpha_i = 1 - epsilon*a_i over the model's frozen 10-row structural basis.

Basis, masks, endpoint, epsilons and pool structure all come from the locked v2 prereg and
ENCODER_BASIS_FREEZE.md / masks_<model>.json. Nothing is selected or tuned here, and no
held-out response is inspected by this script -- it only records raw numbers.

Streaming design (outer loop = batches, inner = conditions): keeps one batch of baseline
masked-position log-probs resident for the logit-KL secondary, and guarantees every
condition is paired on identical masked contexts.

Usage:  python run_encoder_tomography.py --model mosaicbert
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # .../genomic-super-weights (repo root)
sys.path.insert(0, str(HERE))

import e10_lib as L  # noqa: E402

N_WINDOWS = 256
MAX_LEN = 512
BATCH_SIZE = 16
MASK_PROB = 0.15
EPSILONS = [0.5, 1.0]

PANEL = {
    "mosaicbert": dict(
        repo="mosaicml/mosaic-bert-base",
        revision="c89bbadc24278928f22bcdd7de6b61a5a2d08553",
        tokenizer="bert-base-uncased", trust_remote_code=True,
        pattern="bert.encoder.layer.{i}.mlp.wo",
    ),
    "modernbert": dict(
        repo="answerdotai/ModernBERT-base",
        revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
        tokenizer=None, trust_remote_code=False,
        pattern="model.layers.{i}.mlp.Wo",
    ),
}


def load_model(spec):
    from transformers import AutoModelForMaskedLM, AutoTokenizer, BertTokenizer
    if spec["tokenizer"]:
        tok = BertTokenizer.from_pretrained(spec["tokenizer"])
    else:
        tok = AutoTokenizer.from_pretrained(spec["repo"])
    kw = dict(dtype=torch.float32)
    if spec["trust_remote_code"]:
        kw["trust_remote_code"] = True
    model = AutoModelForMaskedLM.from_pretrained(spec["repo"], **kw)
    model.eval()
    return model, tok


@torch.no_grad()
def mlm_batch(model, b, baseline_lp=None):
    """(sum_loss, n_masked, masked_logprobs, mean_kl). Loss summed over masked tokens
    only, so a batch-level bootstrap can reweight exactly (E9's convention)."""
    logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits.float()
    lab = b["labels"]
    sel = lab != -100
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab.reshape(-1),
                           ignore_index=-100, reduction="sum")
    lp = F.log_softmax(logits[sel], dim=-1)
    kl = float("nan")
    if baseline_lp is not None:
        kl = float((baseline_lp.exp() * (baseline_lp - lp)).sum(-1).mean())
    return float(loss), int(sel.sum()), lp, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PANEL))
    args = ap.parse_args()
    spec = PANEL[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    design = json.loads((HERE / f"masks_{args.model}.json").read_text())
    basis = [tuple(b) for b in design["basis"]]
    pattern = spec["pattern"]

    print(f"loading {spec['repo']} ...")
    model, tok = load_model(spec)
    model.to(device)
    resolved = getattr(model.config, "_commit_hash", None)
    print(f"  resolved_revision={resolved} (expected {spec['revision']})")

    windows = L.build_windows(tok, N_WINDOWS, MAX_LEN)
    batches = L.build_fixed_mlm_batches(tok, windows, device, BATCH_SIZE,
                                        mask_prob=MASK_PROB, seed=L.SEED)
    tot_masked = sum(b["n_masked"] for b in batches)
    print(f"  {len(windows)} windows -> {len(batches)} batches, {tot_masked} masked tokens")

    # Frozen condition list: (pool, mask_index, epsilon). Order fixed by the design file.
    conds = []
    for pool in ("singletons", "fit", "calibration", "held_out"):
        for mi, a in enumerate(design["pools"][pool]):
            for eps in EPSILONS:
                conds.append(dict(pool=pool, mask_index=mi, a=a, epsilon=eps))
    print(f"  {len(conds)} conditions ({len(conds)//len(EPSILONS)} masks x {len(EPSILONS)} eps)")

    base_pb = []
    cond_pb = [[] for _ in conds]
    cond_kl = [[] for _ in conds]

    t0 = time.time()
    for bi, b in enumerate(batches):
        s, n, base_lp, _ = mlm_batch(model, b, None)
        base_pb.append((s, n))
        for ci, c in enumerate(conds):
            alphas = L.alphas_for_mask(c["a"], c["epsilon"])
            active = [(coord, al) for coord, al, ai in zip(basis, alphas, c["a"]) if ai]
            coords = [x[0] for x in active]
            als = [x[1] for x in active]
            with L.masked(model, pattern, coords, als):
                s2, n2, _, kl = mlm_batch(model, b, base_lp)
            cond_pb[ci].append((s2, n2))
            cond_kl[ci].append(kl)
        del base_lp
        torch.cuda.empty_cache()
        print(f"  batch {bi+1}/{len(batches)}  ({time.time()-t0:.0f}s)", flush=True)

    base_loss = L.weighted_mean(base_pb)
    pools_out = {p: [] for p in ("singletons", "fit", "calibration", "held_out")}
    for ci, c in enumerate(conds):
        loss = L.weighted_mean(cond_pb[ci])
        rec = next((r for r in pools_out[c["pool"]] if r["mask_index"] == c["mask_index"]), None)
        if rec is None:
            rec = {"mask_index": c["mask_index"], "a": c["a"]}
            pools_out[c["pool"]].append(rec)
        eps = c["epsilon"]
        rec[f"loss_eps{eps}"] = loss
        rec[f"dloss_eps{eps}"] = loss - base_loss
        rec[f"per_batch_eps{eps}"] = cond_pb[ci]
        rec[f"kl_eps{eps}"] = sum(cond_kl[ci]) / len(cond_kl[ci])

    out = {
        "model": args.model, "repo": spec["repo"],
        "requested_revision": spec["revision"], "resolved_revision": resolved,
        "prereg": "PREREG_E10_nlp_architecture_causal_v2.md",
        "prereg_sha256": "3e4b991d71b38cfe15550a57c10c73323cb3b751ba3c768d6c95fc4989833a8a",
        "endpoint": "mean MLM loss over masked positions, WikiText-2-raw-v1 test",
        "basis": [list(b) for b in basis],
        "canonical_row_basis_index": design["canonical_row_basis_index"],
        "n_windows": N_WINDOWS, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
        "mask_prob": MASK_PROB, "seed": L.SEED, "dtype": "float32",
        "epsilons": EPSILONS, "n_masked_tokens": tot_masked,
        "baseline_mlm_loss": base_loss, "baseline_per_batch": base_pb,
        "responses": {"pools": pools_out},
    }
    p = ROOT / "results" / f"e10_encoder_responses_{args.model}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\nbaseline MLM loss = {base_loss:.6f}")
    print("singleton dloss (eps=1.0), by basis index:")
    for r in sorted(pools_out["singletons"], key=lambda r: r["mask_index"]):
        i = r["a"].index(1)
        print(f"  idx {i} L{basis[i][0]}/r{basis[i][1]}: {r['dloss_eps1.0']:+.6f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
