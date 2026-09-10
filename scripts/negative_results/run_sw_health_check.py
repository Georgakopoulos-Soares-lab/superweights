#!/usr/bin/env python
"""
A1 — Super-weight health check: a data-free model diagnostic built from our negatives.

Why this exists
---------------
Two of this project's failures generalise into a tool:

  * Evo1's "super weight" was a NUMERICAL SATURATION artifact -- an activation pinned near a
    precision ceiling, preserved-then-frozen, with ~0% ablation effect.
  * GENERator-PROK's registered SW (L2/r1927) was an OFF-DISTRIBUTION artifact -- detected
    with a human promoter probe fed to a prokaryote model; it does not reproduce under
    matched probes (|row| < 1.0) and the real SW sits elsewhere (L8/r260).

Both were found only because someone checked. This script turns those checks into a
protocol any genomic-LM developer can run in minutes, with no training data.

The four probes
---------------
  1. SATURATION   is the activation near a representable ceiling? Compares |a| to fp16 max
                  (65,504) and to the fp32 mantissa limit 2^24 (16,777,216). A genuine SW is
                  large but not pinned at a ceiling; a saturated channel sits at one.
  2. PERSISTENCE  does the spike survive across DIVERSE probes (multiple sequence families)?
                  Measured as ELEVATION -- the channel's peak divided by the layer's MEDIAN
                  channel peak -- required on EVERY probe family. Rank is deliberately not
                  used: encoder SWs are content-dependent composition detectors whose rank
                  moves with the sequence, so an argmax criterion misclassifies them.
  3. CAUSALITY    does zeroing the row change the model's own objective (perplexity /
                  MLM loss)? A genuine SW moves it; an artifact does not.
  4. STRUCTURE    is the row rank-1 (or near) by ||U_k||_F among its layer's rows? The
                  weight-only structural signature.

Decision rule (applied in this order)
-------------------------------------
  GENUINE               |causal effect| >= causal_eps -- the row does measurable work
  OFF_DISTRIBUTION      inert AND min elevation over probe families < min_elevation
                        (the spike does not reproduce outside its detection probe)
  SATURATION_ARTIFACT   inert, reproduces, AND pinned within [0.5, 2.0] of a representable
                        ceiling (fp16 max, or the fp32 mantissa limit 2^24)
  INERT                 inert, reproduces, not at a ceiling -- real but load-free

CAUSALITY is tested FIRST and dominates: a row that moves the model's own objective is
doing work whatever its elevation. This ordering matters because encoder super weights are
content-dependent detectors with modest elevation (measured 3.2-6.6x), while decoder super
weights are content-independent and enormous (257-7124x); a rank- or elevation-first rule
misclassifies the encoder family.

Output: results/mechanism/sw_health_check.{json,csv} -- one verdict row per (model, layer,
row), plus the raw evidence behind each verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SEED = 42
FP16_MAX = 65504.0
FP32_MANTISSA = 2.0 ** 24          # 16,777,216 — Evo1's regime

# Deliberately diverse probe families: if a spike is real it should not care which of
# these it sees. Kept short so the whole check is a few seconds per model.
PROBES = {
    "human_gc_rich":   "GCCGCCGCCGGGCCCGGGCCGCGGCCGGCCGCGGGCCCGCGGCCGCCGGGCCCGGGCCGCGGCCGGCCGCGGG",
    "human_at_rich":   "ATTTATTTAAATATTTATTAAATTTATTTAAATATTTATTAAATTTATTTAAATATTTATTAAATTTATTTAA",
    "ecoli_like":      "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTAC",
    "repeat_element":  "CACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACA",
    "random_uniform":  "GTACTGACTTGCAAGCTTGACTGACTTAGCATCGATCGGATCCTAGCTAGCTTACGATCGATCGGCATGCATC",
    "poly_a":          "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
}


def _forward(model, ids, am):
    """Run a forward pass across our model families.

    HF models take (input_ids=, attention_mask=); Evo1's StripedHyena takes the ids
    positionally and accepts no attention_mask.
    """
    try:
        return model(input_ids=ids, attention_mask=am)
    except TypeError:
        return model(ids)


def _loss_from(model, ids):
    """Model's own objective on `ids`. Uses labels= when supported, else next-token CE."""
    try:
        return float(model(input_ids=ids, labels=ids).loss)
    except TypeError:
        out = model(ids)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        return float(torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            ids[:, 1:].reshape(-1)))


def _encode(tok, seq, device):
    """Tokenise across the interfaces our wrappers expose.

    HF tokenizers are callable and return a dict; Evo1 ships a CharLevelTokenizer that is
    not callable and exposes .tokenize()/.encode() returning a plain id list.
    """
    if callable(tok):
        enc = tok(seq, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        am = enc.get("attention_mask")
        return ids, (am.to(device) if am is not None else torch.ones_like(ids))
    fn = getattr(tok, "tokenize", None) or getattr(tok, "encode", None)
    if fn is None:
        raise TypeError(f"no usable tokenise method on {type(tok).__name__}")
    out = fn(seq)
    if hasattr(out, "ids"):
        out = out.ids
    ids = torch.tensor([list(out)], dtype=torch.long, device=device)
    return ids, torch.ones_like(ids)


def _tok_len(tok, seq):
    if callable(tok):
        return len(tok(seq)["input_ids"])
    fn = getattr(tok, "tokenize", None) or getattr(tok, "encode", None)
    out = fn(seq)
    return len(out.ids if hasattr(out, "ids") else out)


def peak_for_probes(model, tok, layer_mod, channel, device, pad_multiple=0):
    """Peak |activation| on `channel`, and whether it is the layer argmax, per probe."""
    store = {}
    def _hk(_m, _i, o, _s=store):
        h = o
        if isinstance(h, dict):                 # NTv3 returns a dict of outputs
            h = next((v for v in h.values()
                      if torch.is_tensor(v) and v.dim() >= 2), None)
        elif isinstance(h, (tuple, list)):
            h = h[0]
        if torch.is_tensor(h):
            _s["h"] = h.detach().float()

    hh = layer_mod.register_forward_hook(_hk)
    peaks, is_argmax = {}, {}
    try:
        with torch.no_grad():
            for name, seq in PROBES.items():
                s = seq
                if pad_multiple:
                    while _tok_len(tok, s) % pad_multiple:
                        s = s + "A"
                ids, am = _encode(tok, s, device)
                _forward(model, ids, am)
                h = store["h"]
                h = h if h.dim() == 2 else h[0]
                col = h[:, channel].abs()
                peaks[name] = float(col.max())
                # ELEVATION, not rank. A genuine SW stays far above the layer's typical
                # channel on every probe family, but it need not be rank 1: DNABERT-2's
                # encoder SWs are content-DEPENDENT composition detectors, so their rank
                # moves with the sequence. An off-distribution artifact instead collapses to
                # the ordinary range on probes it was not detected with.
                per_channel_peak = h.abs().max(0).values
                med = float(per_channel_peak.median())
                is_argmax[name] = float(col.max()) / max(med, 1e-12)
    finally:
        hh.remove()
    return peaks, is_argmax   # is_argmax holds elevation ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sw_index", default="results/super_weight_index.json")
    ap.add_argument("--extra_index", default=None,
                    help="second index merged in (e.g. the deep NTv3 index)")
    ap.add_argument("--top_n", type=int, default=3, help="rows per model to check")
    ap.add_argument("--causal_eps", type=float, default=0.01,
                    help="relative objective change below which a row counts as inert")
    ap.add_argument("--cv_max", type=float, default=1.0,
                    help="peak coefficient-of-variation above which a spike is off-distribution")
    ap.add_argument("--min_elevation", type=float, default=2.0,
                    help="a genuine SW must stay at least this many times the layer's MEDIAN "
                         "channel peak on EVERY probe family. Rank is deliberately not used: "
                         "encoder SWs are content-dependent, so their rank varies by sequence.")
    ap.add_argument("--pad_multiple", type=int, default=0)
    ap.add_argument("--out", default="results/mechanism/sw_health_check.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from models import WRAPPER_MAP
    idx = json.loads((ROOT / args.sw_index).read_text())
    if args.extra_index:
        extra = json.loads((ROOT / args.extra_index).read_text())
        for k, v in extra.items():
            idx.setdefault(k, v)

    rows_out = []
    for mname in args.models:
        entry = idx.get(mname)
        if not entry or not entry.get("results"):
            print(f"[{mname}] no registered SW rows — skipping")
            continue
        cfg = yaml.safe_load((ROOT / f"configs/{mname}.yaml").read_text())
        print(f"\n=== {mname} ===")
        try:
            w = WRAPPER_MAP[mname](cfg)
            w.load()
            model, tok = w.model, w.tokenizer
        except Exception as e:
            print(f"  [!] could not load: {str(e)[:110]}")
            continue
        device = next(model.parameters()).device
        mods = dict(model.named_modules())

        sw = sorted(entry["results"], key=lambda e: -(e.get("out_max") or 0))[:args.top_n]
        for s in sw:
            L, R = int(s["layer"]), int(s["row"])
            key = cfg["down_proj_pattern"].format(i=L)
            block_key = key.rsplit(".", 1)[0]
            lm = mods.get(block_key) or mods.get(key)
            if lm is None:
                print(f"  L{L}r{R}: module not found — skipping")
                continue

            # ---- probe 1: saturation ----------------------------------------
            reported = float(s.get("out_max") or 0.0)
            sat_fp16 = reported / FP16_MAX
            sat_fp32 = reported / FP32_MANTISSA
            # Saturation means PINNED NEAR a ceiling, not merely exceeding one. A value far
            # above fp16 max is unremarkable in an fp32/bf16 model -- it only indicates a
            # problem if the model is *stored* in fp16. So a channel counts as at-a-ceiling
            # only when its magnitude sits within [0.5, 2.0] of a representable limit.
            at_ceiling = any(0.5 <= r <= 2.0 for r in (sat_fp16, sat_fp32))

            # ---- probe 2: persistence across diverse probes ------------------
            peaks, elevs = peak_for_probes(model, tok, lm, R, device, args.pad_multiple)
            vals = np.array(list(peaks.values()), dtype=np.float64)
            cv = float(vals.std() / max(vals.mean(), 1e-12))
            elev = np.array(list(elevs.values()), dtype=np.float64)
            min_elev = float(elev.min())        # worst-case elevation over probe families
            med_elev = float(np.median(elev))

            # ---- probe 3: causality on the model's own objective -------------
            wmod = mods.get(key)
            causal = None
            if wmod is not None and hasattr(wmod, "weight"):
                probe_seq = PROBES["random_uniform"]
                if args.pad_multiple:
                    while _tok_len(tok, probe_seq) % args.pad_multiple:
                        probe_seq = probe_seq + "A"
                ids, _am = _encode(tok, probe_seq, device)
                try:
                    with torch.no_grad():
                        base = _loss_from(model, ids)
                    saved = wmod.weight.data[R].clone()
                    wmod.weight.data[R].zero_()
                    with torch.no_grad():
                        abl = _loss_from(model, ids)
                    wmod.weight.data[R] = saved
                    causal = abs(abl - base) / max(abs(base), 1e-9)
                except Exception as e:
                    print(f"    (causal check unavailable: {str(e)[:60]})")

            # ---- probe 4: structural rank by ||U_k||_F ------------------------
            uk_rank = None
            if wmod is not None and hasattr(wmod, "weight"):
                W = wmod.weight.data.float()
                fro = W.norm(dim=1)
                uk_rank = int((fro > fro[R]).sum().item()) + 1

            # ---- verdict ------------------------------------------------------
            inert = causal is not None and causal < args.causal_eps
            reproduces = min_elev >= args.min_elevation
            # Order matters. Non-reproduction is the more reliable discriminator, so it is
            # tested first: an off-distribution spike fails across probe families, whereas a
            # saturated channel is FROZEN and therefore reproduces perfectly while being
            # functionally inert.
            # CAUSALITY is primary: a row that measurably moves the model's own objective
            # is doing work, whatever its elevation. The persistence and saturation probes
            # then explain WHY a non-functional row looked like a super weight.
            if not inert:
                verdict = "GENUINE"
            elif not reproduces:
                verdict = "OFF_DISTRIBUTION"
            elif at_ceiling:
                verdict = "SATURATION_ARTIFACT"
            else:
                verdict = "INERT"

            rec = {"model": mname, "layer": L, "row": R,
                   "reported_out_max": reported,
                   "saturation_ratio_fp16": sat_fp16, "saturation_ratio_fp32": sat_fp32,
                   "at_precision_ceiling": bool(at_ceiling),
                   "reproduces_across_probes": bool(reproduces),
                   "probe_peaks": peaks, "probe_peak_cv": cv,
                   "probe_min_elevation": min_elev, "probe_median_elevation": med_elev,
                   "probe_elevations": elevs,
                   "causal_rel_objective_change": causal,
                   "uk_frobenius_rank": uk_rank, "n_rows": int(W.shape[0]) if wmod is not None else None,
                   "verdict": verdict}
            rows_out.append(rec)
            print(f"  L{L}r{R:<5} peak_cv={cv:5.2f} min_elev={min_elev:8.1f}x "
                  f"causal={causal if causal is None else round(causal,4)} "
                  f"sat(fp32)={sat_fp32:.3f} Uk_rank={uk_rank} -> {verdict}")
        del model
        torch.cuda.empty_cache()

    import transformers as _tf
    out = {"task": "A1_sw_health_check",
           "decision_rule": {"causal_eps": args.causal_eps, "cv_max": args.cv_max,
                             "min_elevation": args.min_elevation,
                             "fp16_max": FP16_MAX, "fp32_mantissa": FP32_MANTISSA},
           "probes": list(PROBES), "rows": rows_out,
           "provenance": {"seed": SEED, "torch": torch.__version__,
                          "transformers": _tf.__version__}}
    op = ROOT / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(op.read_text()).get("rows", []) if op.exists() else []
    seen = {(r["model"], r["layer"], r["row"]) for r in rows_out}
    out["rows"] = rows_out + [r for r in prev if (r["model"], r["layer"], r["row"]) not in seen]
    op.write_text(json.dumps(out, indent=2))
    with open(str(op).replace(".json", ".csv"), "w") as f:
        f.write("model,layer,row,out_max,peak_cv,min_elevation,causal_rel,sat_fp32,uk_rank,verdict\n")
        for r in out["rows"]:
            f.write(f"{r['model']},{r['layer']},{r['row']},{r['reported_out_max']:.1f},"
                    f"{r['probe_peak_cv']:.4f},{r['probe_min_elevation']:.3f},"
                    f"{'' if r['causal_rel_objective_change'] is None else round(r['causal_rel_objective_change'],6)},"
                    f"{r['saturation_ratio_fp32']:.6f},{r['uk_frobenius_rank']},{r['verdict']}\n")
    print(f"\nsaved → {op}")


if __name__ == "__main__":
    main()
