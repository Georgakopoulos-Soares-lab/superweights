#!/usr/bin/env python3
"""
EXPERIMENT 3 — how often is the high-gain phenotype tied to a privileged structural token?

DNABERT's candidate is known to be special-token sensitive: one preprocessing path makes
L5/r603 activation-dominant, another puts its ratio near 1. This measures, for each frozen
candidate under matched sequence content:

  cond A  tokenizer default special-token behaviour
  cond B  add_special_tokens=False

recording activation max, layer-relative ratio, max-activation position, and whether that
position holds a special token (BOS/CLS/SEP/EOS) or an ordinary one.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path("/home/nvidia/superweights"); sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/mechanism_generator"; OUT.mkdir(parents=True, exist_ok=True)
SEQS = ["ACGTACGTACGTAAGGCCTTACGATCGATCGGCTAGCTAGCTTACGATCGATCGGCATGCATCGATCGTAGC",
        "GCCGCCGCCGGGCCCGGGCCGCGGCCGGCCGCGGGCCCGCGGCCGCCGGGCCCGGGCCGCGGCCGGCCGCGG",
        "ATTTATTTAAATATTTATTAAATTTATTTAAATATTTATTAAATTTATTTAAATATTTATTAAATTTATTTA"]
ALL = {"generator": [(4, 2371)], "generator_prokaryote": [(8, 260)],
       "dnabert2": [(5, 603), (9, 264), (9, 294), (3, 86)], "ntv3": [(11, 1472)]}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main(names):
    from models import WRAPPER_MAP
    MODELS = [(n, L, R, None) for n in names for (L, R) in ALL[n]]
    rows = []
    cur = None
    for name, L, R, env in MODELS:
        cfg = yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text())
        if cur is None or cur[0] != name:
            if name == "dnabert2":
                import transformers
                mdl = transformers.AutoModelForMaskedLM.from_pretrained(
                    cfg["model_id"], trust_remote_code=True)
                _m = sys.modules[type(mdl).__module__]
                if getattr(_m, "flash_attn_qkvpacked_func", None) is not None:
                    _m.flash_attn_qkvpacked_func = None
                tk = transformers.AutoTokenizer.from_pretrained(
                    cfg["model_id"], trust_remote_code=True)
                cur = (name, mdl.to("cuda").eval(), tk)
            else:
                w = WRAPPER_MAP[name](cfg); w.load(); cur = (name, w.model, w.tokenizer)
        _, model, tok = cur
        dev = next(model.parameters()).device
        key = cfg["down_proj_pattern"].format(i=L)
        mods = dict(model.named_modules())
        # Hook the down_proj module ITSELF, never its parent. For DNABERT-2 the parent is
        # BertGatedLinearUnitMLP, whose output is the gated MLP result rather than the
        # down-projection output, which understates the row's activation ~40x.
        lm = mods.get(key) or mods.get(key.rsplit(".", 1)[0])
        specials = {i for i in (getattr(tok, a + "_token_id", None) for a in
                    ("bos", "cls", "sep", "eos", "pad")) if i is not None}
        for cond, add_sp in (("default", True), ("no_special", False)):
            st = {}
            hh = lm.register_forward_hook(
                lambda _m, _i, o, _s=st: _s.__setitem__(
                    "h", (o[0] if isinstance(o, tuple) else
                          (next(v for v in o.values() if torch.is_tensor(v) and v.dim() >= 2)
                           if isinstance(o, dict) else o)).detach().float()))
            per = []
            try:
                for s in SEQS:
                    try:
                        enc = tok(s, return_tensors="pt", add_special_tokens=add_sp)
                    except TypeError:
                        continue
                    ids = enc["input_ids"].to(dev)
                    if name == "ntv3":
                        while ids.shape[1] % 256: ids = torch.cat([ids, ids[:, -1:]], 1)
                    with torch.no_grad():
                        try: model(input_ids=ids, attention_mask=torch.ones_like(ids))
                        except TypeError: model(ids)
                    h = st.get("h")
                    if h is None: continue
                    h = h if h.dim() == 2 else h[0]
                    col = h[:, R].abs()
                    pk = h.abs().max(0).values
                    pos = int(col.argmax())
                    per.append(dict(amax=float(col.max()),
                                    ratio=float(col.max() / pk.median().clamp(min=1e-12)),
                                    pos=pos, is_special=bool(ids[0, pos].item() in specials),
                                    L=ids.shape[1]))
            finally:
                hh.remove()
            if not per: continue
            rows.append(dict(model=name, layer=L, row=R, condition=cond, n_seqs=len(per),
                             activation_max=float(np.mean([p["amax"] for p in per])),
                             activation_ratio=float(np.mean([p["ratio"] for p in per])),
                             median_max_position=float(np.median([p["pos"] for p in per])),
                             frac_max_at_special=float(np.mean([p["is_special"] for p in per])),
                             frac_max_at_pos0=float(np.mean([p["pos"] == 0 for p in per]))))
            r = rows[-1]
            log(f"{name:22s} L{L}r{R:<5} {cond:11s} ratio={r['activation_ratio']:10.1f} "
                f"pos={r['median_max_position']:.0f} at_special={r['frac_max_at_special']:.2f}")
    jf = OUT / "special_token_dependence.json"
    prev = json.loads(jf.read_text())["rows"] if jf.exists() else []
    seen = {(r["model"], r["layer"], r["row"], r["condition"]) for r in rows}
    allr = rows + [r for r in prev
                   if (r["model"], r["layer"], r["row"], r["condition"]) not in seen]
    json.dump(dict(experiment="EXP3_special_token_dependence", rows=allr,
                   sequences=len(SEQS)), open(jf, "w"), indent=2)
    rows = allr
    cols = ["model","layer","row","condition","n_seqs","activation_max","activation_ratio",
            "median_max_position","frac_max_at_special","frac_max_at_pos0"]
    with open(OUT / "special_token_dependence.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    log("saved")


if __name__ == "__main__":
    main(sys.argv[1:] or list(ALL))
