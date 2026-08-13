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
  bias_test       is the SW channel content-INdependent? Compare its value across a real
                  sequence vs a dinucleotide-shuffled version of the SAME sequence
                  (seed 42). A pure implicit bias is unchanged by shuffling; a feature
                  detector moves.

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


def dinuc_shuffle(s, rng):
    """Preserve dinucleotide composition (Altschul-Erikson style, simple walk)."""
    s = s.upper()
    if len(s) < 4:
        return s
    edges = {}
    for i in range(len(s) - 1):
        edges.setdefault(s[i], []).append(s[i + 1])
    for k in edges:
        rng.shuffle(edges[k])
    out = [s[0]]
    cur = s[0]
    for _ in range(len(s) - 1):
        if not edges.get(cur):
            cur = rng.choice([c for c in edges if edges[c]] or [cur])
            if not edges.get(cur):
                break
        nxt = edges[cur].pop()
        out.append(nxt)
        cur = nxt
    return "".join(out)


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
    wins = read_windows("/data/nvidia/data/hg38/hg38.fa",
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

    # ── implicit-bias (content-independence) test ────────────────────────────
    mods = dict(model.named_modules())
    key = cfg["down_proj_pattern"].format(i=args.layer).rsplit(".", 1)[0]
    store = {}
    hh = mods[key].register_forward_hook(
        lambda _m, _i, o, _s=store: _s.__setitem__(
            "h", (o[0] if isinstance(o, tuple) else o).detach().float()))
    real0, shuf0, realmax, shufmax = [], [], [], []
    srng = random.Random(SEED)
    with torch.no_grad():
        for _c, _s, seq in wins:
            for tag, sq in (("real", seq), ("shuf", dinuc_shuffle(seq, srng))):
                enc = tok(sq, return_tensors="pt", truncation=True, max_length=args.max_len)
                ids = enc["input_ids"].to(dev)
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
                h = store["h"]
                ch = (h[:, args.channel] if h.dim() == 2 else h[0, :, args.channel]).abs()
                (real0 if tag == "real" else shuf0).append(float(ch[0]))
                (realmax if tag == "real" else shufmax).append(float(ch.max()))
    hh.remove()

    bias = {"pos0_real_mean": float(np.mean(real0)),
            "pos0_shuffled_mean": float(np.mean(shuf0)),
            "pos0_ratio_shuf_over_real": float(np.mean(shuf0) / max(np.mean(real0), 1e-9)),
            "max_real_mean": float(np.mean(realmax)),
            "max_shuffled_mean": float(np.mean(shufmax))}
    print(f"\n  BOS-position value: real {bias['pos0_real_mean']:,.1f}  "
          f"dinuc-shuffled {bias['pos0_shuffled_mean']:,.1f}  "
          f"ratio {bias['pos0_ratio_shuf_over_real']:.4f}")
    print("  (ratio ~1.0 => content-INdependent implicit bias; "
          "far from 1.0 => content-dependent feature detector)")
    print("  CAVEAT: in a CAUSAL decoder position 0 attends only to itself, so a ratio of "
          "exactly 1.0 at pos0 is structurally guaranteed once the peak is known to be at "
          "BOS. It corroborates BOS-anchoring; it is not independent evidence. The "
          "informative comparison is the max-over-positions row below.")

    import transformers as _tf
    out = {"task": "T2.1_attention_sink_implicit_bias", "model": args.model,
           "layer": args.layer, "channel": args.channel,
           "provenance": {"seed": SEED, "torch": torch.__version__,
                          "transformers": _tf.__version__,
                          "n_windows": len(wins), "max_len": args.max_len},
           "attention": att, "implicit_bias": bias}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(op.read_text()) if op.exists() else {}
    prev[args.model] = out
    op.write_text(json.dumps(prev, indent=2))
    print(f"\nsaved → {op}")


if __name__ == "__main__":
    main()
