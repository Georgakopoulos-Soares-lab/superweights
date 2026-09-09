#!/usr/bin/env python3
"""
Verify the manuscript's "Candidate provenance" claims under its OWN stated discovery input.

Why this exists
---------------
run_uniform_detector_text.py measured the ratio-argmax over 24 shuffled WikiText-2 windows
(pooled max), which is a BROADER input set than the manuscript's stated discovery protocol.
The manuscript (Methods, "Candidate provenance") specifies:

    "Text ratio-based discovery used the first 20 non-empty lines of the WikiText-2 raw test
     split, joined and tokenized with truncation to a maximum length of 512 tokens."

i.e. ONE 512-token input, not 24 windows. The 24-window result is stronger evidence, but it
is not a bit-exact reproduction of the published protocol, so this script re-checks under the
exact stated input. It also reports the global ABSOLUTE-ACTIVATION argmax, which the
uniform-detector script does not: the manuscript asserts

    "In OLMo-7B ... the ratio maximum falls at L2/r269 and the activation maximum at L30/r269"

and the activation half of that sentence is unverified anywhere in this repository.

Claims checked (stated before running)
--------------------------------------
  Llama-7B     ratio-argmax == published L2/r3968      ("independently recovers")
  Mistral-7B   ratio-argmax == published L1/r2070      ("independently recovers")
  OLMo-7B      ratio-argmax == L2/r269                 ("the ratio maximum falls at L2/r269")
  OLMo-7B      activation-argmax == L30/r269           ("the activation maximum at L30/r269")

A mismatch is a manuscript-text correction, not a harness failure -- there is no stored
census quantity to gate against here, because the census never recorded these values (the
activation_ratio column is empty for all 22 models). Report exactly what is measured.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/frozen/E10_nlp_architecture_causal"))
import e10_lib as L  # noqa: E402

OUT = ROOT / "results/paper_closing"
PATTERN = "model.layers.{i}.mlp.down_proj"
PARQUET = ROOT / "frozen_inputs/wikitext_repo/wikitext-2-raw-v1/test-00000-of-00001.parquet"

MODELS = {
    "llama":   dict(label="Llama-7B", repo="huggyllama/llama-7b",
                    revision="4782ad278652c7c71b72204d462d6d01eaaf7549", layer=2, row=3968,
                    claim_ratio=(2, 3968), claim_act=None),
    "mistral": dict(label="Mistral-7B", repo="mistralai/Mistral-7B-v0.1",
                    revision="27d67f1b5f57dc0953326b2601d68371d40ea8da", layer=1, row=2070,
                    claim_ratio=(1, 2070), claim_act=None),
    "olmo":    dict(label="OLMo-7B-0724-hf", repo="allenai/OLMo-7B-0724-hf",
                    revision="1ee306df318ee15bfe4a76ebd5c002b0105b1ab6", layer=1, row=269,
                    claim_ratio=(2, 269), claim_act=(30, 269)),
}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def published_text_input(tok):
    """The manuscript's stated discovery input, via E13's frozen_wikitext_input convention:
    first 20 non-empty lines of the WikiText-2 raw test split, joined, truncated to 512."""
    import pyarrow.parquet as pq
    lines = [str(x).strip() for x in
             pq.read_table(PARQUET, columns=["text"]).column("text").to_pylist()
             if str(x).strip()]
    text = "\n".join(lines[:20])
    return tok(text, return_tensors="pt", truncation=True, max_length=512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    args = ap.parse_args()
    M = MODELS[args.model]
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(revision=M["revision"])
    tok = AutoTokenizer.from_pretrained(M["repo"], **kw)
    model = AutoModelForCausalLM.from_pretrained(M["repo"], dtype=torch.float32, **kw).eval().cuda()
    nl = model.config.num_hidden_layers
    enc = published_text_input(tok)
    ids = enc["input_ids"].cuda()
    log(f"{M['label']}: {nl} layers, published input = {ids.shape[1]} tokens")

    store, handles = {}, []
    for li in range(nl):
        handles.append(L._resolve_module(model, PATTERN, li).register_forward_hook(
            lambda _m, _i, o, _l=li: store.__setitem__(
                _l, (o[0] if isinstance(o, tuple) else o).detach().float())))
    with torch.no_grad():
        model(input_ids=ids)
    for h in handles: h.remove()
    nr = store[0].shape[-1]
    peak = torch.stack([store[li].reshape(-1, nr).abs().max(0).values for li in range(nl)])
    ratio = (peak / peak.median(dim=1, keepdim=True).values.clamp(min=1e-12)).cpu().numpy()
    absa = peak.cpu().numpy()

    rl, rr = (int(v) for v in np.unravel_index(int(ratio.argmax()), ratio.shape))
    al, ar = (int(v) for v in np.unravel_index(int(absa.argmax()), absa.shape))
    fz = (M["layer"], M["row"])
    rec = dict(
        model=M["label"], repo=M["repo"], revision=M["revision"],
        detector_input=("first 20 non-empty WikiText-2 raw test lines, joined, "
                        f"truncation to 512 tokens (measured {int(ids.shape[1])})"),
        n_layers=nl, d_model=nr, frozen=dict(layer=fz[0], row=fz[1]),
        ratio_argmax=dict(layer=rl, row=rr, ratio=float(ratio[rl, rr])),
        activation_argmax=dict(layer=al, row=ar, activation_max=float(absa[al, ar]),
                               ratio_at_that_coord=float(ratio[al, ar])),
        frozen_ratio=float(ratio[fz]), frozen_activation_max=float(absa[fz]),
        frozen_ratio_global_rank=int((ratio.ravel() > ratio[fz]).sum()) + 1,
        frozen_activation_global_rank=int((absa.ravel() > absa[fz]).sum()) + 1,
        n_coords=int(ratio.size))
    log(f"  ratio-argmax      L{rl}/r{rr}  ratio {ratio[rl,rr]:.2f}")
    log(f"  activation-argmax L{al}/r{ar}  |a| {absa[al,ar]:.2f} (ratio there {ratio[al,ar]:.2f})")
    log(f"  frozen L{fz[0]}/r{fz[1]}: ratio {ratio[fz]:.2f} (rank {rec['frozen_ratio_global_rank']}"
        f"/{rec['n_coords']}), |a| {absa[fz]:.2f} (rank {rec['frozen_activation_global_rank']})")

    checks = {}
    if M["claim_ratio"]:
        ok = (rl, rr) == tuple(M["claim_ratio"])
        checks["manuscript_ratio_argmax"] = dict(
            claimed=f"L{M['claim_ratio'][0]}/r{M['claim_ratio'][1]}",
            measured=f"L{rl}/r{rr}", agrees=bool(ok))
    if M["claim_act"]:
        ok = (al, ar) == tuple(M["claim_act"])
        checks["manuscript_activation_argmax"] = dict(
            claimed=f"L{M['claim_act'][0]}/r{M['claim_act'][1]}",
            measured=f"L{al}/r{ar}", agrees=bool(ok))
    for k, c in checks.items():
        log(f"  CHECK {k}: claimed {c['claimed']} vs measured {c['measured']}  "
            f"{'AGREES' if c['agrees'] else '*** DISAGREES ***'}")
    rec["manuscript_checks"] = checks

    import transformers as _tf
    rec["provenance"] = dict(torch=torch.__version__, transformers=_tf.__version__,
                             pattern=PATTERN, parquet=str(PARQUET))
    rec["elapsed_seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "detector_published_protocol.json"
    prev = json.loads(p.read_text()) if p.exists() else {}
    prev[args.model] = rec
    p.write_text(json.dumps(prev, indent=2))
    log(f"saved -> {p}")


if __name__ == "__main__":
    main()
