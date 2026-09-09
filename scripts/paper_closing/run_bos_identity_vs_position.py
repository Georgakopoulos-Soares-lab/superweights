#!/usr/bin/env python3
"""
EXPERIMENT 5 — does L4/r2371's high-gain activation follow BOS TOKEN IDENTITY or POSITION 0?

Exp 4 showed rescue is confined to p=0 (99.96% at p=0, 1.1% at p=1, ~0 beyond). That
establishes position-0 specificity but conflates two things: the token at position 0 is
always BOS. This separates them.

Conditions (sequence content held identical throughout):
  1 standard      BOS + seq                 BOS present, at position 0        [frozen form]
  2 no_bos        seq                       BOS absent
  3 shifted_bos   NEUTRAL + BOS + seq       BOS present, at position 1
  4 no_bos_pad    NEUTRAL + seq             BOS absent, ordinary token at position 0
                                            (token count matches condition 1)

Condition 3 is the decisive one:
  activation moves to position 1  -> follows BOS token identity
  activation stays at position 0  -> follows absolute position
  both                            -> quantify the split

NEUTRAL is a single in-vocabulary 6-mer ('AAAAAA'), matching GENERator's 6-mer tokenizer so
no ragged token is introduced. Reuses the frozen E12 model load / revision / corpus.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path("/home/nvidia/superweights")
D = ROOT / "experiments/frozen/E12_generator_degradation_control"
sys.path.insert(0, str(D))
import run_bos_mediation as bm
import e12_lib as e12

OUT = ROOT / "results/paper_closing"; OUT.mkdir(parents=True, exist_ok=True)
NEUTRAL = "AAAAAA"   # single in-vocab 6-mer (id 32)
N_WIN = 40


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build(tok, seq, cond):
    r = len(seq) % 6
    s = seq[r:] if r else seq
    bos = tok.bos_token or ""
    if cond == "standard":    return bos + s
    if cond == "no_bos":      return s
    if cond == "shifted_bos": return NEUTRAL + bos + s
    if cond == "no_bos_pad":  return NEUTRAL + s
    raise ValueError(cond)


def main():
    t0 = time.time()
    model, tok, resolved = bm.load_model()
    log(f"revision {resolved}   bos_token={tok.bos_token!r} id={tok.bos_token_id}")
    _, damage_windows, meta = e12.build_corpora(seed=42, n_prompt=96, prompt_win_bp=170,
                                                n_damage=100, damage_win_bp=512)
    wins = damage_windows[:N_WIN]

    mod = dict(model.named_modules())[bm.PATTERN.format(i=bm.LAYER)]
    store = {}
    hh = mod.register_forward_hook(
        lambda _m, i, _o, _s=store: _s.__setitem__("inp", i[0].detach().float()))

    rows = []
    for cond in ("standard", "no_bos", "shifted_bos", "no_bos_pad"):
        recs = []
        for _c, _s, seq in wins:
            text = build(tok, seq, cond)
            # add_special_tokens=False is ESSENTIAL: this tokenizer has
            # add_bos_token=True and would otherwise auto-prepend BOS to every
            # condition, making them identical at position 0.
            ids = tok(text, return_tensors="pt",
                      add_special_tokens=False)["input_ids"].to(model.device)
            with torch.no_grad():
                out = model(input_ids=ids, labels=ids)
            nll = float(out.loss)
            # row activation = the down_proj INPUT coordinate feeding row 2371 is not the
            # observable; the row's *output* contribution is. Capture the layer output.
            h = store["inp"]
            h = h if h.dim() == 2 else h[0]           # (L, d_ff) input to down_proj
            # contribution of row r to the residual stream = w_r . h_t  (per position)
            W = mod.weight.data.float()               # (d_model, d_ff)
            contrib = (h @ W[bm.ROW]).abs()           # (L,)
            toks = ids[0].tolist()
            bos_id = tok.bos_token_id
            bos_pos = toks.index(bos_id) if (bos_id is not None and bos_id in toks) else None
            amax = float(contrib.max()); apos = int(contrib.argmax())
            med = float(contrib.median())
            recs.append(dict(nll=nll, act_max=amax, act_argmax_pos=apos,
                             act_at_pos0=float(contrib[0]),
                             act_at_bos=(float(contrib[bos_pos]) if bos_pos is not None else None),
                             bos_pos=bos_pos, ratio=amax / max(med, 1e-12), L=len(toks)))
        f = lambda k: [r[k] for r in recs if r[k] is not None]
        rows.append(dict(
            condition=cond, n=len(recs),
            nll_mean=float(np.mean(f("nll"))),
            act_max_mean=float(np.mean(f("act_max"))),
            act_at_pos0_mean=float(np.mean(f("act_at_pos0"))),
            act_at_bos_mean=(float(np.mean(f("act_at_bos"))) if f("act_at_bos") else None),
            bos_position=(int(np.median(f("bos_pos"))) if f("bos_pos") else None),
            frac_argmax_at_pos0=float(np.mean([r["act_argmax_pos"] == 0 for r in recs])),
            frac_argmax_at_bos=(float(np.mean([r["act_argmax_pos"] == r["bos_pos"]
                                for r in recs if r["bos_pos"] is not None]))
                                if f("bos_pos") else None),
            median_argmax_pos=float(np.median([r["act_argmax_pos"] for r in recs])),
            activation_ratio_mean=float(np.mean(f("ratio")))))
        r = rows[-1]
        ab = "n/a" if r["frac_argmax_at_bos"] is None else f"{r['frac_argmax_at_bos']:.2f}"
        log(f"{cond:12s} nll={r['nll_mean']:.4f} act_max={r['act_max_mean']:12.1f} "
            f"argmax@0={r['frac_argmax_at_pos0']:.2f} argmax@BOS={ab} "
            f"bos_pos={r['bos_position']} ratio={r['activation_ratio_mean']:.1f}")
    hh.remove()

    json.dump(dict(experiment="EXP5_bos_identity_vs_position", model="GENERator-EUK-3B",
                   layer=bm.LAYER, row=bm.ROW, revision=resolved, n_windows=N_WIN,
                   neutral_token=NEUTRAL, corpus_meta=meta, rows=rows,
                   elapsed_seconds=time.time() - t0),
              open(OUT / "generator_bos_identity_vs_position.json", "w"), indent=2)
    cols = ["condition","n","nll_mean","act_max_mean","act_at_pos0_mean","act_at_bos_mean",
            "bos_position","frac_argmax_at_pos0","frac_argmax_at_bos","median_argmax_pos",
            "activation_ratio_mean"]
    with open(OUT / "generator_bos_identity_vs_position.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join("" if r[c] is None else
                               (f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]))
                               for c in cols) + "\n")
    log("saved")


if __name__ == "__main__":
    main()
