#!/usr/bin/env python3
"""
EXPERIMENTS 6 + 7 — BOS attention sink in GENERator EUK, and whether L4/r2371 causes it.

EXP6 quantifies the sink with a CAUSAL-MASK-AWARE baseline. Under a causal mask query i can
attend to keys 0..i, so uniform attention gives key 0 a share of 1/(i+1); a naive 1/T
baseline would manufacture an apparent sink. We therefore report

    S_BOS(l,h)   = mean over queries i>0 of A[l,h,i,0]
    fold_sink    = S_BOS / mean_i[1/(i+1)]          (>1 means genuinely disproportionate)

plus the fraction of queries where BOS is the top-attended key, and the full mean incoming
attention profile over key positions.

EXP7 repeats the measurement under the frozen E12 intervention set to test whether the
high-gain row SUPPORTS the sink or merely co-localises with it:
    intact / full ablation / BOS-only ablation / preserve-BOS-only /
    full ablation + restore at BOS / full ablation + restore at matched non-BOS

Interpretation is pre-committed: if BOS-only ablation collapses the sink and BOS restoration
rescues it, r2371 supports the sink; if the sink is unchanged despite catastrophic NLL
damage, the two co-localise but are mechanistically separable.
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
N_WIN = 24
CONDS = ["A_intact", "B_full_ablation_weight", "C_ablate_bos_only",
         "D_ablate_all_except_bos", "E_restore_at_bos", "F_restore_at_matched_nonbos"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def attn_stats(model, tok, seq, hook_ctx=None):
    """Return per-(layer,head) BOS sink stats + key-position profile for one window."""
    ids = bm._tokenize(tok, seq)
    T = ids.shape[1]
    if T < 4:
        return None
    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True)
    atts = out.attentions
    if not atts:
        return None
    # causal-mask-aware expectation for key 0 over queries i>0
    i = np.arange(1, T)
    exp0 = float(np.mean(1.0 / (i + 1)))
    S, topfrac = [], []
    prof = np.zeros(T)
    for a in atts:                               # (B,H,Q,K)
        A = a[0].float().cpu().numpy()
        s = A[:, 1:, 0].mean(axis=1)             # (H,) mean incoming to key0 over queries>0
        S.append(s)
        top = (A[:, 1:, :].argmax(axis=-1) == 0).mean(axis=1)
        topfrac.append(top)
        prof += A[:, 1:, :].mean(axis=(0, 1))    # mean over heads+queries -> (K,)
    prof /= len(atts)
    return dict(T=T, exp0=exp0, S=np.array(S), topfrac=np.array(topfrac), prof=prof)


def run_condition(model, tok, wins, cond, cache):
    per = []
    for _c, _s, seq in wins:
        if cond == "A_intact":
            r = attn_stats(model, tok, seq)
        elif cond == "B_full_ablation_weight":
            saved = e12.set_row_alpha(model, bm.PATTERN, bm.LAYER, bm.ROW, 0.0)
            try: r = attn_stats(model, tok, seq)
            finally: e12.restore_row(model, bm.PATTERN, bm.LAYER, bm.ROW, saved)
        elif cond in ("C_ablate_bos_only", "D_ablate_all_except_bos"):
            zf = ((lambda L_, pf: [i == 0 for i in range(L_)]) if cond.startswith("C")
                  else (lambda L_, pf: [i != 0 for i in range(L_)]))
            h = bm.register(model, bm.PositionRowHook(bm.ROW, zero_fn=zf))
            try: r = attn_stats(model, tok, seq)
            finally: h.remove()
        else:
            if seq in cache: iv = cache[seq]
            else:
                iv = bm._forced_capture(model, tok, seq); cache[seq] = iv
            L = bm._tokenize(tok, seq).shape[1]
            tp = 0 if cond == "E_restore_at_bos" else L // 2
            saved = e12.set_row_alpha(model, bm.PATTERN, bm.LAYER, bm.ROW, 0.0)
            h = bm.register(model, bm.PositionRowHook(
                bm.ROW, inject_fn=lambda L_, pf, t=tp, v=iv: ([t], v[t:t + 1])))
            try: r = attn_stats(model, tok, seq)
            finally:
                h.remove(); e12.restore_row(model, bm.PATTERN, bm.LAYER, bm.ROW, saved)
        if r: per.append(r)
    if not per: return None
    S = np.stack([p["S"] for p in per])          # (win, L, H)
    TF = np.stack([p["topfrac"] for p in per])
    exp0 = float(np.mean([p["exp0"] for p in per]))
    fold = S.mean(axis=0) / exp0
    return dict(condition=cond, n_windows=len(per),
                S_bos_mean=float(S.mean()), exp0_causal_baseline=exp0,
                fold_sink_mean=float(fold.mean()), fold_sink_max=float(fold.max()),
                frac_queries_bos_top=float(TF.mean()),
                frac_layerhead_fold_gt2=float((fold > 2).mean()),
                per_layer_fold=[float(x) for x in fold.mean(axis=1)],
                fold_matrix=fold.tolist())


def main():
    t0 = time.time()
    model, tok, resolved = bm.load_model()
    try:
        model.config._attn_implementation = "eager"
        for m in model.modules():
            if hasattr(m, "config"): m.config._attn_implementation = "eager"
    except Exception as e: log(f"eager set failed: {e}")
    log(f"revision {resolved}")
    _, dmg, meta = e12.build_corpora(seed=42, n_prompt=96, prompt_win_bp=170,
                                     n_damage=100, damage_win_bp=512)
    wins = dmg[:N_WIN]
    cache, rows = {}, []
    for c in CONDS:
        r = run_condition(model, tok, wins, c, cache)
        if r is None: log(f"{c}: no data"); continue
        rows.append(r)
        log(f"{c:28s} S_BOS={r['S_bos_mean']:.4f} baseline={r['exp0_causal_baseline']:.4f} "
            f"fold={r['fold_sink_mean']:7.2f} max={r['fold_sink_max']:8.2f} "
            f"top1={r['frac_queries_bos_top']:.3f} frac_lh_fold>2={r['frac_layerhead_fold_gt2']:.3f}")
    json.dump(dict(experiment="EXP6_7_attention_sink", model="GENERator-EUK-3B",
                   layer=bm.LAYER, row=bm.ROW, revision=resolved, n_windows=N_WIN,
                   corpus_meta=meta, conditions=rows, elapsed_seconds=time.time() - t0),
              open(OUT / "generator_attention_sink.json", "w"), indent=2)
    cols = ["condition","n_windows","S_bos_mean","exp0_causal_baseline","fold_sink_mean",
            "fold_sink_max","frac_queries_bos_top","frac_layerhead_fold_gt2"]
    with open(OUT / "generator_attention_sink.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    log("saved")


if __name__ == "__main__":
    main()
