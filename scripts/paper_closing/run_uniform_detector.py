#!/usr/bin/env python3
"""
EXPERIMENT 2 (partial) — apply ONE uniform selection rule and re-evaluate causally.

Universal rule (as specified after the provenance audit):
  1. statistic = LAYER-RELATIVE ratio, never absolute activation
       ratio(l,r) = max_t|a[l,r,t]| / median_r'( max_t|a[l,r',t]| )
  2. scope     = global argmax over ALL layers x rows
  3. special-token handling declared, and BOTH settings reported
  4. multiple diverse inputs, with a rank-1 stability requirement
  5. report rank + ratio of the frozen coordinate so "is it the argmax?" is answerable
     from the artifact without a re-run

Frozen census is NOT modified. For each model we report the frozen candidate and the
uniform-rule candidate, and -- where they differ -- causally evaluate BOTH under the SAME
paired endpoint so the comparison is internally valid.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path("/home/nvidia/superweights"); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/mechanism"))
OUT = ROOT / "results/paper_closing"; OUT.mkdir(parents=True, exist_ok=True)

PROBES = [
 "ACGTACGTACGTAAGGCCTTACGATCGATCGGCTAGCTAGCTTACGATCGATCGGCATGCATCGATCGTAGCTAGCTAGGCTA",
 "GCCGCCGCCGGGCCCGGGCCGCGGCCGGCCGCGGGCCCGCGGCCGCCGGGCCCGGGCCGCGGCCGGCCGCGGGCCCGCGGCCG",
 "ATTTATTTAAATATTTATTAAATTTATTTAAATATTTATTAAATTTATTTAAATATTTATTAAATTTATTTAAATATTTATTA",
 "ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACAGA",
 "CACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACACAC",
 "TTGACAGCTAGCTCAGTCCTAGGTATAATGCTAGCAAGCTTGGCATTCCGGTACTGTTGGTAAAATGGCTAGCTAGCTAGCTA",
]
FROZEN = {"ntv3": (11, 1472), "dnabert2": (5, 603)}
ENV = {"ntv3": "generator", "dnabert2": "dnabert"}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(model_name):
    t0 = time.time()
    cfg = yaml.safe_load((ROOT / f"configs/{model_name}.yaml").read_text())
    pat, NL = cfg["down_proj_pattern"], cfg["num_layers"]
    if model_name == "dnabert2":
        # MaskedLM head, NOT the 3-class classifier: this experiment needs an LM endpoint.
        # Force the PyTorch attention path (the bundled Triton flash-attn kernel is
        # incompatible with the installed Triton) -- same fix as run_pretrained_epistasis.
        import transformers
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            cfg["model_id"], trust_remote_code=True)
        _m = sys.modules[type(model).__module__]
        if getattr(_m, "flash_attn_qkvpacked_func", None) is not None:
            _m.flash_attn_qkvpacked_func = None
        tok = transformers.AutoTokenizer.from_pretrained(
            cfg["model_id"], trust_remote_code=True)
        model = model.to("cuda")
    else:
        from models import WRAPPER_MAP
        w = WRAPPER_MAP[model_name](cfg); w.load(); model, tok = w.model, w.tokenizer
    model.eval()
    dev = next(model.parameters()).device
    mods = dict(model.named_modules())
    pad = 256 if model_name == "ntv3" else 0

    def enc(s, add_special):
        ids = tok(s, return_tensors="pt", add_special_tokens=add_special)["input_ids"]
        if pad:
            while ids.shape[1] % pad: ids = torch.cat([ids, ids[:, -1:]], 1)
        return ids.to(dev)

    def sweep(add_special):
        """ratio for EVERY (layer,row); returns dict layer -> (ratio_vec, rank1_count)."""
        res = {}
        for li in range(NL):
            m = mods.get(pat.format(i=li))
            if m is None or not hasattr(m, "weight"): continue
            n = m.weight.data.shape[0]
            peak = torch.zeros(n, device=dev)
            rank1 = np.zeros(n)
            st = {}
            def hk(_m, _i, o, _s=st):
                h = o
                if isinstance(h, dict):
                    h = next((v for v in h.values() if torch.is_tensor(v) and v.dim() >= 2), None)
                elif isinstance(h, (tuple, list)): h = h[0]
                if torch.is_tensor(h): _s["h"] = h.detach().float()
            hh = m.register_forward_hook(hk)
            try:
                with torch.no_grad():
                    for s in PROBES:
                        ids = enc(s, add_special)
                        try: model(input_ids=ids, attention_mask=torch.ones_like(ids))
                        except TypeError: model(ids)
                        h = st.get("h")
                        if h is None: continue
                        h = h if h.dim() == 2 else h[0]
                        pk = h.abs().max(0).values
                        peak = torch.maximum(peak, pk)
                        rank1[int(pk.argmax())] += 1
            finally: hh.remove()
            med = float(peak.median())
            res[li] = ((peak / max(med, 1e-12)).cpu().numpy(), rank1)
        return res

    def lm_loss(ids):
        if model_name == "dnabert2":
            # MLM endpoint with a FIXED mask (seed 42) so every condition is scored on
            # byte-identical inputs and frozen-vs-uniform is a paired comparison.
            g = torch.Generator().manual_seed(42)
            sel = (torch.rand(ids.shape, generator=g) < 0.15).to(ids.device)
            if sel.sum() == 0:
                sel[0, 1] = True
            inp = ids.clone(); inp[sel] = tok.mask_token_id
            lab = ids.clone(); lab[~sel] = -100
            return float(model(input_ids=inp, attention_mask=torch.ones_like(ids),
                               labels=lab).loss)
        try: return float(model(input_ids=ids, labels=ids).loss)
        except Exception:
            out = model(input_ids=ids)
            lg = out.logits if hasattr(out, "logits") else out[0]
            return float(torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, lg.shape[-1]).float(), ids[:, 1:].reshape(-1)))

    def damage(L, R):
        m = mods[pat.format(i=L)]
        sv = m.weight.data[R].clone()
        base = np.mean([lm_loss(enc(s, True)) for s in PROBES])
        m.weight.data[R].zero_()
        abl = np.mean([lm_loss(enc(s, True)) for s in PROBES])
        m.weight.data[R] = sv
        return float(base), float(abl), float((abl - base) / max(abs(base), 1e-9))

    out = {"model": model_name, "frozen": list(FROZEN[model_name]), "conditions": {}}
    for cond, add_sp in (("default", True), ("no_special", False)):
        sw = sweep(add_sp)
        best = max(((li, int(np.argmax(v[0])), float(v[0].max())) for li, v in sw.items()),
                   key=lambda t: t[2])
        fL, fR = FROZEN[model_name]
        fv = sw[fL][0]
        frank = int((fv > fv[fR]).sum()) + 1
        glob_rank = 1 + sum(int((v[0] > fv[fR]).sum()) for v in sw.values())
        out["conditions"][cond] = dict(
            uniform_layer=best[0], uniform_row=best[1], uniform_ratio=best[2],
            frozen_ratio=float(fv[fR]), frozen_rank_in_layer=frank,
            frozen_global_rank=glob_rank,
            same_coordinate=bool(best[0] == fL and best[1] == fR),
            same_row=bool(best[1] == fR), same_layer=bool(best[0] == fL),
            uniform_rank1_fraction=float(sw[best[0]][1][best[1]] / len(PROBES)),
            frozen_rank1_fraction=float(sw[fL][1][fR] / len(PROBES)))
        c = out["conditions"][cond]
        log(f"{model_name} [{cond}] uniform=L{c['uniform_layer']}/r{c['uniform_row']} "
            f"ratio={c['uniform_ratio']:.2f} | frozen=L{fL}/r{fR} ratio={c['frozen_ratio']:.2f} "
            f"layer_rank={frank} global_rank={glob_rank} same={c['same_coordinate']}")
        if not c["same_coordinate"]:
            b1, a1, r1 = damage(fL, fR)
            b2, a2, r2 = damage(c["uniform_layer"], c["uniform_row"])
            c["causal_frozen_rel"] = r1; c["causal_uniform_rel"] = r2
            c["causal_baseline"] = b1
            log(f"   causal (paired, same probes): frozen relΔ={r1:+.5f}  uniform relΔ={r2:+.5f}")
    json.dump(out, open(OUT / f"uniform_detector_{model_name}.json", "w"), indent=2)
    log(f"saved ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
