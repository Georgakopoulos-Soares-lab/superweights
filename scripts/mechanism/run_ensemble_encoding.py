#!/usr/bin/env python
"""
T1.1 — What does the super-weight ensemble ENCODE? (real-sequence context characterization)

Converts "a dedicated 13-feature SAE code for the SW channel exists" into "the code
computes X", by finding the actual genomic contexts where each feature fires hardest and
testing them for composition / motif / positional structure.

Two modes
---------
  --mode sae      GENERator EUK: for each SW-correlated SAE feature (|r|>0.5), stream real
                  hg38, record the top-K activating token positions and their sequence
                  windows, then run the structure battery on those windows.
  --mode direct   DNABERT-2: skip the SAE. Find the splice-test sequences that drive the
                  largest SW-channel activations for the critical pair, run the same battery.

Structure battery (identical for both modes)
--------------------------------------------
  composition : GC fraction, per-dinucleotide frequency, longest homopolymer run,
                Shannon entropy of the base distribution
  motif       : presence of splice donor (GT at +1) / acceptor (AG at -1) consensus,
                TATA box, CpG dinucleotide density, NF-kB (GGGRNNYYCC)
  position    : does the feature fire at a fixed offset within its window? Reported as the
                circular-variance of firing offsets and the modal offset share.

Every statistic is compared against a matched BACKGROUND of randomly drawn windows
(seed 42), and every correlation is reported with its n_active. Statistics whose support
is < --min_support are refused, not reported -- the n_active~1 -> +-0.99 artifact is
established and must never recur.

Kill condition (stated first)
-----------------------------
If the top-activating contexts show no structure beyond "the SW channel is large here"
-- i.e. composition indistinguishable from background AND no motif enrichment AND no
positional preference -- then the code exists but its CONTENT is not resolvable with this
instrument. Report that, plus the circularity caveat (channel 2371 is part of the SAE
input, so SW-correlation alone is not evidence of encoding).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

SEED = 42
BASES = "ACGT"


# ── structure battery ────────────────────────────────────────────────────────
def gc_frac(s):
    s = s.upper()
    n = sum(c in "ACGT" for c in s)
    return (s.count("G") + s.count("C")) / max(n, 1)


def longest_homopolymer(s):
    best = cur = 1
    for i in range(1, len(s)):
        cur = cur + 1 if s[i] == s[i - 1] else 1
        best = max(best, cur)
    return best


def base_entropy(s):
    s = s.upper()
    c = Counter(ch for ch in s if ch in BASES)
    n = sum(c.values())
    if n == 0:
        return 0.0
    return -sum((v / n) * math.log2(v / n) for v in c.values() if v)


def dinuc_freqs(s):
    s = s.upper()
    c = Counter(s[i:i + 2] for i in range(len(s) - 1)
                if s[i] in BASES and s[i + 1] in BASES)
    n = sum(c.values()) or 1
    return {a + b: c.get(a + b, 0) / n for a in BASES for b in BASES}


def cpg_ratio(s):
    """Observed/expected CpG -- the standard genomic depletion statistic."""
    s = s.upper()
    n = len(s)
    cg = s.count("CG")
    c, g = s.count("C"), s.count("G")
    exp = (c * g) / max(n, 1)
    return cg / exp if exp > 0 else 0.0


MOTIFS = {
    "splice_donor_GT":    ["GGTAAG", "GGTGAG", "AGGTAAG", "AGGTGAG"],
    "splice_acceptor_AG": ["TTTTAG", "TTTCAG", "CTTTCAG", "TTTTTAG"],
    "TATA_box":           ["TATAAA", "TATAAAA", "TATATAA"],
    "CpG_island_core":    ["CGCGCG", "GCGCGC", "CGCG"],
    "NFkB":               ["GGGACTTTCC", "GGGAATTTCC", "GGGGATTTCC"],
    "polyA_signal":       ["AATAAA", "ATTAAA"],
}


def motif_hits(s):
    s = s.upper()
    return {k: int(any(m in s for m in v)) for k, v in MOTIFS.items()}


def battery(windows):
    """Run the full structure battery over a list of sequence windows."""
    if not windows:
        return None
    gc = [gc_frac(w) for w in windows]
    hp = [longest_homopolymer(w) for w in windows]
    en = [base_entropy(w) for w in windows]
    cg = [cpg_ratio(w) for w in windows]
    dn = {k: [] for k in dinuc_freqs("ACGT")}
    for w in windows:
        d = dinuc_freqs(w)
        for k in dn:
            dn[k].append(d[k])
    mh = {k: [] for k in MOTIFS}
    for w in windows:
        h = motif_hits(w)
        for k in mh:
            mh[k].append(h[k])
    return {
        "n_windows": len(windows),
        "gc_mean": float(np.mean(gc)), "gc_sd": float(np.std(gc)),
        "homopolymer_mean": float(np.mean(hp)), "homopolymer_max": int(np.max(hp)),
        "entropy_mean": float(np.mean(en)),
        "cpg_oe_mean": float(np.mean(cg)),
        "dinuc_mean": {k: float(np.mean(v)) for k, v in dn.items()},
        "motif_frac": {k: float(np.mean(v)) for k, v in mh.items()},
    }


def compare(fg, bg):
    """Effect sizes of foreground vs background, with the support counts attached."""
    if fg is None or bg is None:
        return None
    out = {"n_fg": fg["n_windows"], "n_bg": bg["n_windows"]}
    for k in ("gc_mean", "homopolymer_mean", "entropy_mean", "cpg_oe_mean"):
        out[f"{k}_fg"] = fg[k]
        out[f"{k}_bg"] = bg[k]
        out[f"{k}_delta"] = fg[k] - bg[k]
    # Cohen's d for GC using pooled sd where available
    if fg.get("gc_sd") and bg.get("gc_sd"):
        sp = math.sqrt((fg["gc_sd"] ** 2 + bg["gc_sd"] ** 2) / 2) or 1e-9
        out["gc_cohens_d"] = (fg["gc_mean"] - bg["gc_mean"]) / sp
    out["dinuc_delta"] = {k: fg["dinuc_mean"][k] - bg["dinuc_mean"][k]
                          for k in fg["dinuc_mean"]}
    out["motif_delta"] = {k: fg["motif_frac"][k] - bg["motif_frac"][k]
                          for k in fg["motif_frac"]}
    out["motif_fg"] = fg["motif_frac"]
    out["motif_bg"] = bg["motif_frac"]
    return out


# ── sequence sources ─────────────────────────────────────────────────────────
def read_windows(fasta, bed, n, win_bp, rng):
    from pyfaidx import Fasta
    fa = Fasta(fasta, as_raw=True)
    regs = []
    with open(bed) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                regs.append((p[0], int(p[1]), int(p[2])))
    rng.shuffle(regs)
    out = []
    for c, s, e in regs:
        if len(out) >= n:
            break
        if e - s < win_bp:
            continue
        st = rng.randrange(s, e - win_bp)
        sq = str(fa[c][st:st + win_bp]).upper()
        if sq.count("N") / max(len(sq), 1) < 0.01:
            out.append((c, st, sq))
    return out


# ── mode: sae (GENERator EUK) ────────────────────────────────────────────────
def run_sae_mode(args):
    from models import WRAPPER_MAP
    sys.path.insert(0, str(ROOT))
    from sae.model import BatchTopKSAE

    feats = json.loads(Path(args.feature_json).read_text())["per_feature"]
    sel = [f for f in feats if f["r_with_sw"] is not None
           and abs(f["r_with_sw"]) > args.r_threshold
           and f["n_active"] >= args.min_support]
    sel.sort(key=lambda f: -abs(f["r_with_sw"]))
    fid = [f["feature"] for f in sel]
    print(f"[sae] {len(fid)} features with |r|>{args.r_threshold} and "
          f"n_active>={args.min_support}: {fid}")

    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    wrapper = WRAPPER_MAP[args.model](cfg)
    wrapper.load()
    model, tok = wrapper.model, wrapper.tokenizer
    dev = next(model.parameters()).device

    sae = BatchTopKSAE.load(args.sae_ckpt, device=str(dev)).eval()
    scale = getattr(sae, "data_scale", None)
    print(f"[sae] scale applied: {scale is not None}")

    rng = random.Random(SEED)
    wins = read_windows(args.fasta, str(ROOT / args.bed), args.n_windows,
                        args.win_bp, rng)
    print(f"[sae] {len(wins)} hg38 windows of {args.win_bp} bp")

    layer_mod = dict(model.named_modules())[cfg["down_proj_pattern"].format(i=args.layer)
                                            .rsplit(".", 1)[0]]
    store = {}

    def hook(_m, _i, o, _s=store):
        h = o[0] if isinstance(o, tuple) else o
        _s["h"] = h.detach().float()

    # top-K tracker per feature: list of (act, win_idx, tok_pos)
    top = {f: [] for f in fid}
    handle = layer_mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for wi, (_c, _st, seq) in enumerate(wins):
                enc = tok(seq, return_tensors="pt", truncation=True,
                          max_length=args.max_len)
                ids = enc["input_ids"].to(dev)
                am = enc.get("attention_mask")
                am = am.to(dev) if am is not None else torch.ones_like(ids)
                model(input_ids=ids, attention_mask=am)
                h = store["h"][0]                       # (L, D)
                x = h / scale.to(h.device) if scale is not None else h
                acts = sae.encode(x)["acts"]            # (L, F)
                for f in fid:
                    col = acts[:, f]
                    v, p = float(col.max()), int(col.argmax())
                    if v > 0:
                        top[f].append((v, wi, p))
                if (wi + 1) % 50 == 0:
                    print(f"   {wi+1}/{len(wins)} windows", flush=True)
    finally:
        handle.remove()

    # background windows for comparison
    bg_seqs = [w[2][:args.ctx_bp] for w in wins[:args.topk * 4]]
    bg = battery(bg_seqs)

    results = {}
    for f in fid:
        hits = sorted(top[f], key=lambda t: -t[0])[:args.topk]
        if len(hits) < args.min_support_ctx:
            results[str(f)] = {"n_contexts": len(hits),
                               "refused": f"n_contexts < {args.min_support_ctx}"}
            continue
        ctxs, offs = [], []
        for v, wi, p in hits:
            seq = wins[wi][2]
            # token position -> approximate bp offset (BPE: use proportional mapping)
            n_tok = len(tok(seq)["input_ids"])
            frac = p / max(n_tok - 1, 1)
            c = int(frac * max(len(seq) - args.ctx_bp, 1))
            ctxs.append(seq[c:c + args.ctx_bp])
            offs.append(frac)
        fg = battery(ctxs)
        cmp_ = compare(fg, bg)
        modal = Counter(round(o, 1) for o in offs).most_common(1)[0]
        cmp_["position"] = {"offset_mean": float(np.mean(offs)),
                            "offset_sd": float(np.std(offs)),
                            "modal_offset": modal[0],
                            "modal_share": modal[1] / len(offs)}
        cmp_["n_active_total"] = next(x["n_active"] for x in sel if x["feature"] == f)
        cmp_["r_with_sw"] = next(x["r_with_sw"] for x in sel if x["feature"] == f)
        cmp_["top_contexts"] = ctxs[:5]
        results[str(f)] = cmp_
        print(f"  feat {f}: GC {cmp_['gc_mean_fg']:.3f} vs bg {cmp_['gc_mean_bg']:.3f} "
              f"(d={cmp_.get('gc_cohens_d',0):+.2f})  homopol "
              f"{cmp_['homopolymer_mean_fg']:.1f} vs {cmp_['homopolymer_mean_bg']:.1f}  "
              f"n_ctx={fg['n_windows']}")
    return {"mode": "sae", "model": args.model, "layer": args.layer,
            "sae_ckpt": args.sae_ckpt, "features": fid,
            "background": bg, "per_feature": results}


def run_direct_mode(args):
    """DNABERT-2: characterise the sequences that drive the SW ensemble hardest.

    No SAE. For each critical-pair channel, score every splice test sequence by the
    channel's peak |activation|, take the top-K and bottom-K, and contrast them with the
    full structure battery. Bottom-K is the matched background (same task distribution),
    which is a stricter control than random genomic windows.
    """
    import csv as _csv
    sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
    from run_gue_multiseed import _load_model
    from run_gue_ablation import _resolve_module

    cfg = yaml.safe_load((ROOT / "configs/dnabert2.yaml").read_text())
    pattern = cfg["down_proj_pattern"]
    model, tok = _load_model(cfg["model_id"], 3, None, "cuda")
    model.load_state_dict(torch.load(ROOT / args.ckpt, map_location="cpu"), strict=False)
    model.eval()

    rows = list(_csv.reader(open(args.task_csv)))[1:]
    seqs = [r[0] for r in rows[:args.n_seqs]]
    print(f"[direct] {len(seqs)} splice sequences")

    coords = [tuple(int(x) for x in c.split(",")) for c in args.coords]
    out = {}
    for (L, R) in coords:
        mod = _resolve_module(model, pattern, L)
        st = {}
        hh = mod.register_forward_hook(
            lambda _m, _i, o, _s=st: _s.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o).detach().float()))
        peak, peakpos = [], []
        with torch.no_grad():
            for sq in seqs:
                e = tok(sq, return_tensors="pt", truncation=True, max_length=args.max_len)
                ids = e["input_ids"].cuda()
                model(input_ids=ids, attention_mask=torch.ones_like(ids))
                h = st["h"]
                ch = (h[:, R] if h.dim() == 2 else h[0, :, R]).abs()
                peak.append(float(ch.max())); peakpos.append(int(ch.argmax()))
        hh.remove()
        order = np.argsort(peak)
        lo = [seqs[i] for i in order[:args.topk]]
        hi = [seqs[i] for i in order[-args.topk:]]
        cmp_ = compare(battery(hi), battery(lo))
        pp = np.array(peakpos)
        cmp_["peak_position"] = {"frac_at_token0": float((pp == 0).mean()),
                                 "median_pos": float(np.median(pp)),
                                 "mean_pos": float(pp.mean())}
        cmp_["activation"] = {"top_mean": float(np.mean([peak[i] for i in order[-args.topk:]])),
                              "bottom_mean": float(np.mean([peak[i] for i in order[:args.topk]]))}
        out[f"L{L}r{R}"] = cmp_
        print(f"  L{L}r{R}: GC hi={cmp_['gc_mean_fg']:.3f} lo={cmp_['gc_mean_bg']:.3f} "
              f"(d={cmp_.get('gc_cohens_d',0):+.2f})  homopolΔ={cmp_['homopolymer_mean_delta']:+.2f}  "
              f"peak@tok0={100*cmp_['peak_position']['frac_at_token0']:.0f}%  "
              f"act {cmp_['activation']['bottom_mean']:.1f}->{cmp_['activation']['top_mean']:.1f}")
    return {"mode": "direct", "model": "dnabert2", "coords": args.coords,
            "n_seqs": len(seqs), "per_channel": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sae", "direct"], default="sae")
    ap.add_argument("--model", default="generator")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--sae_ckpt", default="results/sae/euk_layer4_std/sae_final_scaled.pt")
    ap.add_argument("--feature_json",
                    default="results/sae/euk_layer4_std/real_seq/real_sequence_features.json")
    ap.add_argument("--fasta", default="/data/nvidia/data/hg38/hg38.fa")
    ap.add_argument("--bed", default="data/regions/hg38/random_262kb.bed")
    ap.add_argument("--n_windows", type=int, default=300)
    ap.add_argument("--win_bp", type=int, default=512)
    ap.add_argument("--ctx_bp", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--r_threshold", type=float, default=0.5)
    ap.add_argument("--min_support", type=int, default=100)
    ap.add_argument("--min_support_ctx", type=int, default=10)
    ap.add_argument("--ckpt", default="results/gue_checkpoints_multiseed/"
                                      "dnabert2_reconstructed/seed_0/model_state.pt")
    ap.add_argument("--task_csv", default="/data/nvidia/data/gue/GUE/splice/reconstructed/test.csv")
    ap.add_argument("--coords", nargs="+", default=["9,264", "9,294", "3,603", "5,603"])
    ap.add_argument("--n_seqs", type=int, default=600)
    ap.add_argument("--out", default="results/mechanism/ensemble_encoding_characterization.json")
    args = ap.parse_args()

    if args.mode == "direct":
        out = run_direct_mode(args)
    else:
        out = run_sae_mode(args)
    import transformers as _tf
    out["provenance"] = {"seed": SEED, "torch": torch.__version__,
                         "transformers": _tf.__version__,
                         "n_windows": args.n_windows, "win_bp": args.win_bp,
                         "ctx_bp": args.ctx_bp, "topk": args.topk,
                         "min_support": args.min_support}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))

    if out.get("mode") == "direct":
        csv = str(op).replace(".json", "_direct.csv")
        with open(csv, "w") as fh:
            fh.write("channel,gc_hi,gc_lo,gc_d,homopol_delta,entropy_delta,"
                     "cpg_oe_delta,frac_peak_tok0,act_lo,act_hi\n")
            for ch, r in out["per_channel"].items():
                pp = r["peak_position"]; ac = r["activation"]
                fh.write(f"{ch},{r['gc_mean_fg']:.4f},{r['gc_mean_bg']:.4f},"
                         f"{r.get('gc_cohens_d',0):.3f},{r['homopolymer_mean_delta']:.3f},"
                         f"{r['entropy_mean_delta']:.4f},{r['cpg_oe_mean_delta']:.4f},"
                         f"{pp['frac_at_token0']:.3f},{ac['bottom_mean']:.2f},{ac['top_mean']:.2f}\n")
        print(f"\nsaved → {op}\n        {csv}")
        return
    csv = str(op).replace(".json", ".csv")
    with open(csv, "w") as fh:
        fh.write("feature,r_with_sw,n_active,n_contexts,gc_fg,gc_bg,gc_d,"
                 "homopol_fg,homopol_bg,entropy_fg,entropy_bg,cpg_oe_fg,cpg_oe_bg,"
                 "modal_offset,modal_share\n")
        for f, r in out["per_feature"].items():
            if r.get("refused"):
                continue
            p = r["position"]
            fh.write(f"{f},{r['r_with_sw']:.4f},{r['n_active_total']},{r['n_fg']},"
                     f"{r['gc_mean_fg']:.4f},{r['gc_mean_bg']:.4f},{r.get('gc_cohens_d',0):.3f},"
                     f"{r['homopolymer_mean_fg']:.2f},{r['homopolymer_mean_bg']:.2f},"
                     f"{r['entropy_mean_fg']:.4f},{r['entropy_mean_bg']:.4f},"
                     f"{r['cpg_oe_mean_fg']:.4f},{r['cpg_oe_mean_bg']:.4f},"
                     f"{p['modal_offset']},{p['modal_share']:.3f}\n")
    print(f"\nsaved → {op}\n        {csv}")


if __name__ == "__main__":
    main()
