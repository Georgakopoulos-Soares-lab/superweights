#!/usr/bin/env python
"""
T2.1 — Is the genomic super-weight an ATTENTION SINK / implicit bias?

Context
-------
We already showed positionally that GENERator's SW channel peaks at the BOS token in
100% of windows (375,361 vs 54.97 at pos 1; 45,585x the mean of positions 1-79), while
DNABERT-2's SW channels never peak at [CLS] (0/40). That is the *activation* half of the
attention-sink account. This script measures the *attention* half: do other tokens route
disproportionate attention mass TO the SW position?

Diagnostics
-----------
  sink_share      mean incoming attention mass at position 0, averaged over heads/layers,
                  compared with the mass at other positions and with the uniform
                  expectation 1/L. A sink shows share >> 1/L.
  head_fraction   fraction of (layer, head) pairs whose argmax attention target is pos 0.

A third diagnostic (content-independence via a real-vs-dinucleotide-shuffled comparison at
position 0) was removed 2026-08-26: on a causal decoder, position 0's hidden state cannot
depend on any later token regardless of shuffling, so the test was unanswerable by
construction, not merely uninformative. See the comment at its former call site (below
`main`'s attention-mass block) for the full reasoning.

Kill condition (stated first)
-----------------------------
If the decoder shows NO elevated attention mass at the SW position, the "attention sink"
label is wrong even though the activation is BOS-anchored, and we must describe it as a
BOS-anchored implicit bias WITHOUT the sink claim. Report which.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts" / "mechanism"))

from run_ensemble_encoding import read_windows  # noqa: E402

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="generator")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--channel", type=int, default=2371)
    ap.add_argument("--n_windows", type=int, default=40)
    ap.add_argument("--win_bp", type=int, default=512)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--out", default="results/mechanism/attention_sink_implicit_bias.json")
    args = ap.parse_args()

    from models import WRAPPER_MAP
    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    w = WRAPPER_MAP[args.model](cfg)
    w.load()
    model, tok = w.model, w.tokenizer
    # sdpa/flash kernels silently refuse output_attentions; eager is required to read
    # attention maps at all.
    try:
        model.config._attn_implementation = "eager"
        for m_ in model.modules():
            if hasattr(m_, "config"):
                m_.config._attn_implementation = "eager"
        print("[sink] forced eager attention for attention-map readout")
    except Exception as e:
        print(f"[sink] could not force eager: {e}")
    dev = next(model.parameters()).device

    rng = random.Random(SEED)
    # Original hardcoded path (/data/nvidia/data/hg38/hg38.fa) is absent on this
    # filesystem -- repointed to the local copy downloaded for E9 (same file,
    # same provenance as PROVENANCE_AND_BASELINES.md's hg38 fix). No science changed.
    wins = read_windows(str(ROOT / "data/reference/hg38/hg38.fa"),
                        str(ROOT / "data/regions/hg38/random_262kb.bed"),
                        args.n_windows, args.win_bp, rng)
    print(f"[sink] {args.model} layer={args.layer} channel={args.channel} "
          f"windows={len(wins)}")

    # ── attention diagnostics ────────────────────────────────────────────────
    sink_shares, uniform_exp, head_argmax0, n_heads_tot = [], [], 0, 0
    other_shares = []
    for _c, _s, seq in wins:
        enc = tok(seq, return_tensors="pt", truncation=True, max_length=args.max_len)
        ids = enc["input_ids"].to(dev)
        am = enc.get("attention_mask")
        am = am.to(dev) if am is not None else torch.ones_like(ids)
        with torch.no_grad():
            try:
                out = model(input_ids=ids, attention_mask=am, output_attentions=True)
                atts = out.attentions
            except Exception as e:
                print(f"[sink] attention not exposed: {e}")
                atts = None
        if not atts:
            break
        L = ids.shape[1]
        for a in atts:                        # (B, H, Q, K)
            a = a[0].float()
            inc = a.mean(1)                   # mean over queries -> (H, K)
            sink_shares.append(float(inc[:, 0].mean()))
            if L > 1:
                other_shares.append(float(inc[:, 1:].mean()))
            uniform_exp.append(1.0 / L)
            head_argmax0 += int((inc.argmax(-1) == 0).sum())
            n_heads_tot += inc.shape[0]

    att = None
    if sink_shares:
        att = {"sink_share_pos0": float(np.mean(sink_shares)),
               "share_other_positions": float(np.mean(other_shares)) if other_shares else None,
               "uniform_expectation": float(np.mean(uniform_exp)),
               "sink_over_uniform": float(np.mean(sink_shares) / np.mean(uniform_exp)),
               "sink_over_other": (float(np.mean(sink_shares) / np.mean(other_shares))
                                   if other_shares else None),
               "frac_heads_argmax_pos0": head_argmax0 / max(n_heads_tot, 1),
               "n_layer_head_obs": n_heads_tot}
        print(f"  sink share @pos0      = {att['sink_share_pos0']:.4f}")
        print(f"  share other positions = {att['share_other_positions']:.6f}")
        print(f"  uniform expectation   = {att['uniform_expectation']:.4f}")
        print(f"  sink/uniform          = {att['sink_over_uniform']:.1f}x")
        print(f"  sink/other            = {att['sink_over_other']:.1f}x")
        print(f"  heads whose argmax is pos0: {100*att['frac_heads_argmax_pos0']:.1f}%")
    else:
        print("  [!] attention weights unavailable for this model")

    # The implicit-bias (content-independence) test that used to run here — comparing the
    # SW channel's value at position 0 on a real vs. dinucleotide-shuffled sequence — was
    # removed 2026-08-26 (round-2 audit, `audit_2_prompt.md` Section 0d). It always
    # returned ratio=1.0 by construction, not as a finding: (1) `dinuc_shuffle` preserves
    # the first character (an Altschul-Erikson shuffle invariant, needed to keep the
    # shuffle graph Eulerian), so position 0's token is identical in both conditions
    # regardless of the rest of the sequence; and (2) GENERator is a CAUSAL decoder, so
    # position 0's hidden state can only ever depend on position 0's own token — it is
    # mathematically blind to every later token, so no shuffle of anything after position
    # 0 could move it even if the shuffle didn't also preserve the first character. A
    # content-independence test at position 0 is therefore unanswerable on a causal model
    # by design, not merely uninformative in this run; it would be meaningful on a
    # bidirectional encoder (e.g. DNABERT-2), where position 0/CLS does see the whole
    # sequence. Do not resurrect this as a "max-over-positions" variant post hoc — that
    # would be a protocol change made after seeing a null, which this project does not do.
    # The `attention` block above (sink_share_pos0, sink_over_uniform,
    # frac_heads_argmax_pos0) is unaffected and remains the manuscript's actual evidence.

    import transformers as _tf
    out = {"task": "T2.1_attention_sink_implicit_bias", "model": args.model,
           "layer": args.layer, "channel": args.channel,
           "provenance": {"seed": SEED, "torch": torch.__version__,
                          "transformers": _tf.__version__,
                          "n_windows": len(wins), "max_len": args.max_len},
           "attention": att}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(op.read_text()) if op.exists() else {}
    prev[args.model] = out
    op.write_text(json.dumps(prev, indent=2))
    print(f"\nsaved → {op}")


if __name__ == "__main__":
    main()
