#!/usr/bin/env python
"""
TASK 1 — Is the redundant-pair structure intrinsic to pretraining, or made by fine-tuning?

Everything measured so far is post-fine-tuning, and *which* pair is load-bearing varies by
seed (splice s0/s2/s4 -> L9r264+L9r294; s1 -> L3r603+L3r641; s3 -> L3r86+L3r399). That
seed-dependence is consistent with fine-tuning SELECTING one of several latent redundancies
-- or with fine-tuning CREATING the redundancy outright. This distinguishes them.

Design
------
  model    : pretrained DNABERT-2 (AutoModelForMaskedLM), NO task head, NO fine-tuning
  readout  : masked-LM loss -- the pretraining objective itself
  data     : held-out hg38 windows (random_262kb.bed), never used for fine-tuning
  ablation : zero down_proj rows, identical convention to run_sw_pairwise_epistasis.py
  effect   : delta MLM loss (POSITIVE = worse). Note the sign flips relative to the
             fine-tuned analysis, where the effect metric was delta accuracy (NEGATIVE
             = worse). Comparisons below are made on |effect| / superadditivity sign.

  epistasis(a,b) = dLoss(ablate a AND b) - [dLoss(a) + dLoss(b)]
  Superadditive (structure present) => epistasis > 0 (joint hurts more than the sum).

Controls
--------
  * FIXED masking: one mask realisation (seed 42) reused for every condition, so all
    deltas are paired. Without this, mask noise swamps the effect.
  * random pairs: 10 non-SW pairs, seed 42, giving the null floor.
  * k-of-N curve: cumulative ablation in the same descending-out_max order used for the
    fine-tuned curves.

Kill condition (stated first, per operating rules)
--------------------------------------------------
If structurally-related pairs are NOT superadditive above the random floor in the
pretrained model, the pair structure is MANUFACTURED by fine-tuning, and the paper's
central claim must narrow to "fine-tuning recruits a redundant pair".
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from run_gue_ablation import _resolve_module, _save_row, _zero_row, _restore_row  # noqa: E402

SEED = 42


def _pad_to_multiple(tok, seqs, multiple):
    """Right-pad each sequence with 'A' until its TOKEN length is a multiple.

    Required for NTv3: a conv/deconv U-Net with 8 stride-2 conv blocks, so the deconv
    tower's skip connections only align when the token length is a multiple of 2**8=256.
    Nucleotide-level tokenisation (vocab 11) means 1 bp -> 1 token.
    """
    if multiple <= 0:
        return seqs
    out = []
    for sq in seqs:
        n = len(tok(sq)["input_ids"])
        target = ((n + multiple - 1) // multiple) * multiple
        while len(tok(sq)["input_ids"]) < target:
            sq = sq + "A"
        out.append(sq)
    return out


def _load_pretrained_mlm(model_id, device, code_revision=None):
    """Pretrained MLM, with the PyTorch attention path forced.

    DNABERT-2 bundles a Triton flash-attn kernel that is incompatible with the installed
    Triton ('dot() got an unexpected keyword argument trans_b'). The fine-tuned
    classification runs never hit it because nonzero attention dropout routes them to the
    PyTorch implementation (bert_layers.py:161). The MLM config has zero attention dropout,
    so it reaches the broken kernel. Forcing flash_attn_qkvpacked_func=None selects the
    SAME PyTorch path all prior results used -- this is a compatibility fix, not a change
    of method.
    """
    import transformers
    tk = {"trust_remote_code": True}
    if code_revision:
        tk["code_revision"] = code_revision
    model = transformers.AutoModelForMaskedLM.from_pretrained(model_id, **tk)
    mod = sys.modules[type(model).__module__]
    patched = getattr(mod, "flash_attn_qkvpacked_func", None) is not None
    if patched:
        mod.flash_attn_qkvpacked_func = None
    tok = transformers.AutoTokenizer.from_pretrained(model_id, **tk)
    return model.to(device).eval(), tok, patched


def _read_fasta_windows(fasta, bed, n_windows, win_bp, rng):
    """Read genomic windows from a BED over an indexed FASTA."""
    from pyfaidx import Fasta
    fa = Fasta(fasta, as_raw=True)
    regions = []
    with open(bed) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                regions.append((p[0], int(p[1]), int(p[2])))
    rng.shuffle(regions)
    seqs = []
    for chrom, s, e in regions:
        if len(seqs) >= n_windows:
            break
        if e - s < win_bp:
            continue
        st = rng.randrange(s, e - win_bp)
        seq = str(fa[chrom][st:st + win_bp]).upper()
        if seq.count("N") / max(len(seq), 1) < 0.01:
            seqs.append(seq)
    return seqs


def _build_fixed_batches(tok, seqs, device, max_len, batch_size, mask_prob):
    """Tokenise once and apply ONE fixed mask realisation (seed 42) to every batch.

    Every ablation condition is then scored on byte-identical inputs, so deltas are paired
    and mask sampling contributes zero variance to the comparison.
    """
    g = torch.Generator().manual_seed(SEED)
    batches = []
    for i in range(0, len(seqs), batch_size):
        enc = tok(seqs[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        ids = enc["input_ids"]
        # NTv3's tokenizer returns only input_ids (no attention_mask); with equal-length
        # padded windows an all-ones mask is exactly correct.
        attn = enc.get("attention_mask")
        if attn is None:
            attn = torch.ones_like(ids)
        special = torch.zeros_like(ids, dtype=torch.bool)
        for sid in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id):
            if sid is not None:
                special |= ids == sid
        sel = (torch.rand(ids.shape, generator=g) < mask_prob) & (attn == 1) & ~special
        if sel.sum() == 0:
            continue
        inp = ids.clone()
        inp[sel] = tok.mask_token_id
        lab = ids.clone()
        lab[~sel] = -100
        batches.append({"input_ids": inp.to(device),
                        "attention_mask": attn.to(device),
                        "labels": lab.to(device),
                        "n_masked": int(sel.sum())})
    return batches


@torch.no_grad()
def mlm_loss(model, batches):
    """Token-weighted mean MLM loss over the fixed batches."""
    tot, n = 0.0, 0
    for b in batches:
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    labels=b["labels"])
        tot += float(out.loss) * b["n_masked"]
        n += b["n_masked"]
    return tot / max(n, 1)


def with_ablation(model, pattern, coords, fn):
    saves = [(l, r, _save_row(model, pattern, l, r)) for l, r in coords]
    for l, r in coords:
        _zero_row(model, pattern, l, r)
    try:
        return fn()
    finally:
        for l, r, s in saves:
            _restore_row(model, pattern, l, r, s)


def relation(a, b):
    if a[0] == b[0]:
        return "same_layer"
    if a[1] == b[1]:
        return "same_row"
    return "unrelated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--fasta", default="/data/nvidia/data/hg38/hg38.fa")
    ap.add_argument("--bed", default="data/regions/hg38/random_262kb.bed")
    ap.add_argument("--n_windows", type=int, default=256)
    ap.add_argument("--win_bp", type=int, default=600)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--n_rand_pairs", type=int, default=10)
    ap.add_argument("--sw_index", default="results/super_weight_index.json")
    ap.add_argument("--top_n", type=int, default=0)
    ap.add_argument("--pad_to_multiple", type=int, default=0,
                    help="Pad token length to a multiple (use 256 for NTv3).")
    ap.add_argument("--code_revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / f"configs/{args.model}.yaml").read_text())
    pattern = cfg["down_proj_pattern"]

    _swp = Path(args.sw_index)
    if not _swp.is_absolute():
        _swp = ROOT / _swp
    sw = json.loads(_swp.read_text())[args.model]["results"]
    sw = sorted(sw, key=lambda e: -(e.get("out_max") or 0))
    coords = [(int(e["layer"]), int(e["row"])) for e in sw]
    if args.top_n and args.top_n < len(coords):
        coords = coords[:args.top_n]
    N = len(coords)

    model, tok, patched = _load_pretrained_mlm(cfg["model_id"], args.device,
                                               args.code_revision)
    print(f"[t1] pretrained MLM loaded (triton_patched={patched})  N_sw={N}")

    rng = random.Random(SEED)
    seqs = _read_fasta_windows(args.fasta, str(ROOT / args.bed),
                               args.n_windows, args.win_bp, rng)
    seqs = _pad_to_multiple(tok, seqs, args.pad_to_multiple)
    batches = _build_fixed_batches(tok, seqs, args.device, args.max_len,
                                   args.batch_size, args.mask_prob)
    tot_masked = sum(b["n_masked"] for b in batches)
    print(f"[t1] {len(seqs)} windows -> {len(batches)} fixed batches, "
          f"{tot_masked:,} masked tokens (seed {SEED})")

    base = mlm_loss(model, batches)
    print(f"[t1] baseline MLM loss = {base:.6f}")

    singles = {}
    for i, c in enumerate(coords):
        singles[i] = with_ablation(model, pattern, [c],
                                   lambda: mlm_loss(model, batches)) - base
        print(f"   single L{c[0]}r{c[1]:<5} dLoss={singles[i]:+.6f}")

    pairs = {}
    for i, j in combinations(range(N), 2):
        a, b = coords[i], coords[j]
        d = with_ablation(model, pattern, [a, b],
                          lambda: mlm_loss(model, batches)) - base
        eps = d - (singles[i] + singles[j])
        pairs[f"{i}_{j}"] = {"row_a": list(a), "row_b": list(b),
                             "relation": relation(a, b),
                             "d_joint": d, "d_a": singles[i], "d_b": singles[j],
                             "epistasis": eps}

    # random non-SW pair control (seed 42)
    nrows = _resolve_module(model, pattern, 0).weight.data.shape[0]
    nlayers = cfg["num_layers"]
    pool = [(l, r) for l in range(nlayers) for r in range(nrows) if (l, r) not in set(coords)]
    rr = random.Random(SEED)
    rand_eps = []
    for _ in range(args.n_rand_pairs):
        a, b = rr.sample(pool, 2)
        da = with_ablation(model, pattern, [a], lambda: mlm_loss(model, batches)) - base
        db = with_ablation(model, pattern, [b], lambda: mlm_loss(model, batches)) - base
        dj = with_ablation(model, pattern, [a, b], lambda: mlm_loss(model, batches)) - base
        rand_eps.append(dj - (da + db))
    print(f"[t1] random-pair epistasis: mean={np.mean(rand_eps):+.6f} "
          f"sd={np.std(rand_eps, ddof=1):.6f}")

    # k-of-N cumulative curve
    kcurve = []
    for k in range(1, N + 1):
        d = with_ablation(model, pattern, coords[:k], lambda: mlm_loss(model, batches)) - base
        kcurve.append({"k": k, "d_loss": d})
        print(f"   k={k:2d}  dLoss={d:+.6f}")
    sum_parts = sum(singles.values())

    ranked = sorted(pairs.values(), key=lambda p: -p["epistasis"])
    n_rel = sum(1 for p in pairs.values() if p["relation"] != "unrelated")
    from scipy import stats as _st
    enrich = {}
    for K in (3, 5, 7):
        k_rel = sum(1 for p in ranked[:K] if p["relation"] != "unrelated")
        enrich[f"top{K}"] = {"related": k_rel, "K": K,
                             "p": float(_st.hypergeom.sf(k_rel - 1, len(pairs), n_rel, K))}

    import transformers as _tf
    out = {
        "task": "T1_pretrained_vs_finetuned_epistasis",
        "model": args.model, "model_id": cfg["model_id"],
        "readout": "masked_LM_loss (higher = worse)",
        "note": "effect sign is INVERTED vs the fine-tuned accuracy analysis; "
                "superadditivity here means epistasis > 0",
        "provenance": {"seed": SEED, "torch": torch.__version__,
                       "transformers": _tf.__version__,
                       "triton_attention_patched": patched,
                       "n_windows": len(seqs), "n_masked_tokens": tot_masked,
                       "mask_prob": args.mask_prob, "max_len": args.max_len,
                       "fasta": args.fasta, "bed": args.bed},
        "sw_coords": [list(c) for c in coords],
        "baseline_mlm_loss": base,
        "singles": {str(i): singles[i] for i in singles},
        "sum_of_parts": sum_parts,
        "pairs": pairs,
        "aggregate_sorted": ranked,
        "random_pair_epistasis": {"mean": float(np.mean(rand_eps)),
                                  "sd": float(np.std(rand_eps, ddof=1)),
                                  "values": [float(x) for x in rand_eps]},
        "k_of_n": kcurve,
        "enrichment": enrich,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    # csv
    csv_path = str(args.out).replace(".json", ".csv")
    with open(csv_path, "w") as f:
        f.write("row_a_layer,row_a_row,row_b_layer,row_b_row,relation,d_a,d_b,d_joint,epistasis\n")
        for p in ranked:
            f.write(f"{p['row_a'][0]},{p['row_a'][1]},{p['row_b'][0]},{p['row_b'][1]},"
                    f"{p['relation']},{p['d_a']:.8f},{p['d_b']:.8f},{p['d_joint']:.8f},"
                    f"{p['epistasis']:.8f}\n")

    print(f"\n[t1] TOP-7 PAIRS BY EPISTASIS (pretrained, MLM loss)")
    for p in ranked[:7]:
        print(f"   L{p['row_a'][0]}r{p['row_a'][1]} + L{p['row_b'][0]}r{p['row_b'][1]:<5} "
              f"eps={p['epistasis']:+.6f}  {p['relation']}")
    for K in (3, 5, 7):
        e = enrich[f"top{K}"]
        print(f"   top-{K}: {e['related']}/{K} related  p={e['p']:.5f}")
    print(f"\n[t1] k-of-N: all-{N} dLoss={kcurve[-1]['d_loss']:+.6f} vs "
          f"sum-of-parts {sum_parts:+.6f}")
    print(f"saved → {args.out}  and  {csv_path}")


if __name__ == "__main__":
    main()
