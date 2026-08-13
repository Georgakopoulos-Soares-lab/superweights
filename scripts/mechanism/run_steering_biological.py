#!/usr/bin/env python
"""
E3 — Controllable generation with a biological constraint.

"We can move GC" is an ML curiosity. "We can hit a target GC while the sequence stays
genomically realistic" is a capability. This measures both.

Readouts per steering condition
-------------------------------
  gc            achieved GC of generated sequence (the control variable)
  ppl_clean     perplexity of the generated sequence scored by the UNMODIFIED model
                -- the key quality metric: did steering produce sequence the un-steered
                model still considers genomic, or did it produce degenerate text?
  dinuc_kl      KL(generated dinucleotide distribution || hg38 background) -- realism
  max_homopol   longest homopolymer run -- the classic degeneration failure mode
  cpg_oe        observed/expected CpG -- a genuine genomic signature that should be
                preserved rather than collapsing with GC

Control: identical scaling applied to matched random rows (seed 42).
"""
from __future__ import annotations
import argparse, json, math, random, sys
from collections import Counter
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "mechanism"))
from run_ensemble_encoding import read_windows, gc_frac, dinuc_freqs, cpg_ratio, longest_homopolymer  # noqa
SEED = 42


def dinuc_kl(seq, bg):
    d = dinuc_freqs(seq); eps = 1e-9
    return float(sum(d[k] * math.log((d[k] + eps) / (bg[k] + eps)) for k in bg if d[k] > 0))


@torch.no_grad()
def seq_ppl(model, tok, seq, dev):
    ids = tok(seq, return_tensors="pt")["input_ids"].to(dev)
    if ids.shape[1] < 2:
        return float("nan")
    out = model(input_ids=ids, labels=ids)
    return float(torch.exp(out.loss))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="generator")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--row", type=int, default=2371)
    ap.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--n_prompts", type=int, default=32)
    ap.add_argument("--prompt_bp", type=int, default=120)
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--n_random_rows", type=int, default=3)
    ap.add_argument("--out", default="results/mechanism/steering_biological.json")
    args = ap.parse_args()

    from models import WRAPPER_MAP
    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    w = WRAPPER_MAP[args.model](cfg); w.load()
    model, tok = w.model, w.tokenizer
    dev = next(model.parameters()).device
    mod = dict(model.named_modules())[cfg["down_proj_pattern"].format(i=args.layer)]
    nrows = mod.weight.data.shape[0]

    rng = random.Random(SEED)
    wins = read_windows("/data/nvidia/data/hg38/hg38.fa",
                        str(ROOT / "data/regions/hg38/random_262kb.bed"),
                        args.n_prompts, args.prompt_bp + 400, rng)
    prompts = [s[:args.prompt_bp] for _c, _s, s in wins]
    bg_seqs = [s[args.prompt_bp:args.prompt_bp + 300] for _c, _s, s in wins]
    bg = {k: float(np.mean([dinuc_freqs(b)[k] for b in bg_seqs])) for k in dinuc_freqs("ACGT")}
    bg_gc = float(np.mean([gc_frac(b) for b in bg_seqs]))
    bg_ppl = float(np.mean([seq_ppl(model, tok, b, dev) for b in bg_seqs[:12]]))
    bg_hp = float(np.mean([longest_homopolymer(b) for b in bg_seqs]))
    bg_cpg = float(np.mean([cpg_ratio(b) for b in bg_seqs]))
    print(f"[bg] real hg38: GC={bg_gc:.4f} ppl={bg_ppl:.3f} homopol={bg_hp:.2f} cpgOE={bg_cpg:.3f}")

    def gen(row, scale):
        saved = mod.weight.data[row].clone()
        mod.weight.data[row] = saved * scale
        seqs = []
        try:
            for i, p in enumerate(prompts):
                ids = tok(p, return_tensors="pt")["input_ids"].to(dev)
                torch.manual_seed(SEED + i)
                o = model.generate(input_ids=ids, max_new_tokens=args.max_new,
                                   do_sample=True, top_k=50, temperature=1.0,
                                   pad_token_id=getattr(tok, "pad_token_id", None) or 0)
                s = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
                s = "".join(c for c in s.upper() if c in "ACGT")
                if len(s) >= 30:
                    seqs.append(s)
        finally:
            mod.weight.data[row] = saved
        return seqs

    def score(seqs):
        if not seqs:
            return None
        ppl = [seq_ppl(model, tok, s, dev) for s in seqs[:12]]
        return {"n": len(seqs),
                "gc_mean": float(np.mean([gc_frac(s) for s in seqs])),
                "gc_sd": float(np.std([gc_frac(s) for s in seqs])),
                "ppl_clean_mean": float(np.nanmean(ppl)),
                "dinuc_kl_mean": float(np.mean([dinuc_kl(s, bg) for s in seqs])),
                "max_homopolymer_mean": float(np.mean([longest_homopolymer(s) for s in seqs])),
                "cpg_oe_mean": float(np.mean([cpg_ratio(s) for s in seqs]))}

    res = {"sw": {}, "random": {}}
    print(f"\n{'scale':>7}{'GC':>9}{'ppl_clean':>12}{'dinucKL':>10}{'homopol':>9}{'cpgOE':>8}")
    for s in args.scales:
        r = score(gen(args.row, s)); res["sw"][str(s)] = r
        if r:
            print(f"{s:>7}{r['gc_mean']:>9.4f}{r['ppl_clean_mean']:>12.3f}"
                  f"{r['dinuc_kl_mean']:>10.4f}{r['max_homopolymer_mean']:>9.2f}{r['cpg_oe_mean']:>8.3f}")
    rr = random.Random(SEED).sample([c for c in range(nrows) if c != args.row], args.n_random_rows)
    print("  random-row control:")
    for s in args.scales:
        allseq = []
        for c in rr:
            allseq.extend(gen(c, s))
        r = score(allseq); res["random"][str(s)] = r
        if r:
            print(f"{s:>7}{r['gc_mean']:>9.4f}{r['ppl_clean_mean']:>12.3f}"
                  f"{r['dinuc_kl_mean']:>10.4f}{r['max_homopolymer_mean']:>9.2f}{r['cpg_oe_mean']:>8.3f}")

    import transformers as _tf
    out = {"task": "E3_steering_biological", "model": args.model, "row": args.row,
           "background": {"gc": bg_gc, "ppl": bg_ppl, "homopolymer": bg_hp, "cpg_oe": bg_cpg},
           "results": res, "random_rows": rr,
           "provenance": {"seed": SEED, "torch": torch.__version__,
                          "transformers": _tf.__version__, "max_new": args.max_new}}
    op = ROOT / args.out; op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    with open(str(op).replace(".json", ".csv"), "w") as f:
        f.write("condition,scale,n,gc,ppl_clean,dinuc_kl,max_homopolymer,cpg_oe\n")
        for cond in ("sw", "random"):
            for s in args.scales:
                r = res[cond][str(s)]
                if r:
                    f.write(f"{cond},{s},{r['n']},{r['gc_mean']:.4f},{r['ppl_clean_mean']:.4f},"
                            f"{r['dinuc_kl_mean']:.4f},{r['max_homopolymer_mean']:.3f},{r['cpg_oe_mean']:.4f}\n")
    print(f"\nsaved → {op}")


if __name__ == "__main__":
    main()
