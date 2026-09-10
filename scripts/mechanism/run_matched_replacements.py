#!/usr/bin/env python3
"""
EXPERIMENT 9 — structurally matched replacement controls at L4/r2371.

EXP8 showed 20/20 damage-matched RANDOM directions reproduce the GC shift. This separates
which property of the learned row matters, by replacing it with vectors that share
progressively more of its structure. All replacements are renormalised to the original row
norm so scale is held constant and only DIRECTION/STRUCTURE varies.

  random_unit        random Gaussian direction                (no structure)
  random_same_layer  another row's direction from this layer  (learned, wrong row)
  top_frobenius      the largest-norm row's direction         (learned, structurally extreme)
  permuted           the row's own entries shuffled           (same value multiset, no arrangement)
  sign_flipped       the row with random sign flips           (same magnitudes, scrambled signs)
  orthogonalised     random vector, Gram-Schmidt'd against the row (explicitly not the row)
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

OUT = ROOT / "results/analyses/mechanism_generator"; OUT.mkdir(parents=True, exist_ok=True)
N_DMG, N_GC, SCALES = 40, 12, [0.5, 1.0, 2.0]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    model, tok, resolved = bm.load_model()
    prompts, dmg, meta = e12.build_corpora(seed=42, n_prompt=96, prompt_win_bp=170,
                                           n_damage=100, damage_win_bp=512)
    dwin, pwin = dmg[:N_DMG], prompts[:N_GC]
    mod = dict(model.named_modules())[bm.PATTERN.format(i=bm.LAYER)]
    W = mod.weight.data
    orig = W[bm.ROW].clone(); nrm = float(orig.norm())
    rng = np.random.default_rng(42)

    def unit(v): return v / v.norm()
    fro = W.float().norm(dim=1)
    top_row = int(torch.argmax(torch.cat([fro[:bm.ROW], fro[bm.ROW+1:]])).item())
    top_row = top_row if top_row < bm.ROW else top_row + 1
    rand_row = int(rng.choice([r for r in range(W.shape[0]) if r != bm.ROW]))

    g = torch.tensor(rng.normal(size=orig.shape[0]), dtype=orig.dtype, device=orig.device)
    perm = orig[torch.randperm(orig.shape[0], device=orig.device)]
    signs = torch.tensor(rng.choice([-1.0, 1.0], size=orig.shape[0]),
                         dtype=orig.dtype, device=orig.device)
    o_u = unit(orig)
    orth = g - (g @ o_u) * o_u

    FAM = {"random_unit": unit(g), "random_same_layer": unit(W[rand_row].clone()),
           "top_frobenius": unit(W[top_row].clone()), "permuted": unit(perm),
           "sign_flipped": unit(orig * signs), "orthogonalised": unit(orth)}

    def gc_now():
        outs = []
        for _c, _s, seq in pwin:
            ids = bm._tokenize(tok, seq)[:, :30]
            with torch.no_grad():
                o = model.generate(input_ids=ids, max_new_tokens=48, do_sample=True,
                                   top_k=50, temperature=1.0,
                                   pad_token_id=getattr(tok, "pad_token_id", 0) or 0)
            t = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
            outs.append("".join(c for c in t.upper() if c in "ACGT"))
        gs = [(s.count("G") + s.count("C")) / max(len(s), 1) for s in outs if s]
        return float(np.mean(gs))

    intact = bm.damage(model, tok, dwin, "A_intact")
    torch.manual_seed(42); gc_i = gc_now()
    sv = e12.set_row_alpha(model, bm.PATTERN, bm.LAYER, bm.ROW, 0.0)
    ablated = bm.damage(model, tok, dwin, "A_intact")
    torch.manual_seed(42); gc_a = gc_now()
    e12.restore_row(model, bm.PATTERN, bm.LAYER, bm.ROW, sv)
    log(f"intact nll={intact:.4f} gc={gc_i:.4f} | ablated nll={ablated:.4f} gc={gc_a:.4f}")

    rows = []
    for name, d in FAM.items():
        for c in SCALES:
            with torch.no_grad(): W[bm.ROW] = d * nrm * c
            nll = bm.damage(model, tok, dwin, "A_intact")
            torch.manual_seed(42); gc = gc_now()
            with torch.no_grad(): W[bm.ROW] = orig.clone()
            cos = float(torch.nn.functional.cosine_similarity(d[None], o_u[None]).item())
            rows.append(dict(family=name, scale=c, cos_to_row=cos, nll=nll,
                             delta_nll=nll - intact, gc=gc, delta_gc=gc - gc_i,
                             frac_of_ablation_damage=(nll - intact) / max(ablated - intact, 1e-9)))
            log(f"{name:20s} c={c:<4} cos={cos:+.3f} dNLL={nll-intact:+7.4f} "
                f"({rows[-1]['frac_of_ablation_damage']:+.2f}x abl) GC={gc:.4f} dGC={gc-gc_i:+.4f}")

    json.dump(dict(experiment="EXP9_matched_replacements", revision=resolved,
                   intact_nll=intact, ablated_nll=ablated, gc_intact=gc_i, gc_ablated=gc_a,
                   random_same_layer_row=rand_row, top_frobenius_row=top_row,
                   corpus_meta=meta, rows=rows, elapsed_seconds=time.time() - t0),
              open(OUT / "generator_matched_replacements.json", "w"), indent=2)
    cols = ["family","scale","cos_to_row","nll","delta_nll","frac_of_ablation_damage","gc","delta_gc"]
    with open(OUT / "generator_matched_replacements.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    log("saved")


if __name__ == "__main__":
    main()
