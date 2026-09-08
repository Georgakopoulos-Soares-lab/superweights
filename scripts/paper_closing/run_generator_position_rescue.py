#!/usr/bin/env python3
"""
EXPERIMENT 4 (Priority 1) — GENERator EUK L4/r2371 positional rescue curve.

Question: is the BOS mediation result genuinely BOS-SPECIFIC, or is it a consequence of
causal-mask geometry (early positions influence more downstream tokens)?

Method: fully ablate L4/r2371 (weight alpha=0), then restore its intact captured
contribution at EXACTLY ONE position p, for p in {0,1,2,4,8,16,32,64}. This generalises the
frozen E12 conditions E (restore at p=0) and F (restore at p=L//2) to a curve.

  rescue(p) = (L_ablated - L_restore(p)) / (L_ablated - L_intact)

computed per window and aggregated with the SAME paired bootstrap used by the frozen run.

Nothing frozen is modified: this imports the existing E12 harness (model load, revision,
tokenizer, corpus builder, hook class, loss, bootstrap) and only adds a parameterised
restore-at-p condition. It re-runs the provenance check first and refuses to proceed if the
checkpoint no longer reproduces the historical intact / full-ablation values.

Baseline for interpretation (NOT a mechanistic model): under a causal mask, a contribution
injected at p can influence tokens p..T-1, so accessibility(p) = (T-p)/T. If rescue(p)
tracks accessibility, the effect is broadcasting geometry; if rescue spikes only at p=0, it
is position-0/BOS specific.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/home/nvidia/superweights")
D = ROOT / "paper-salvage/experiments/E12_generator_degradation_control"
sys.path.insert(0, str(D))
import run_bos_mediation as bm   # noqa: E402
import e12_lib as e12            # noqa: E402

OUT = ROOT / "results/paper_closing"
OUT.mkdir(parents=True, exist_ok=True)
POSITIONS = [0, 1, 2, 4, 8, 16, 32, 64]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def nll_restore_at_p(model, tok, seq, p, intact_cache):
    """Full weight ablation + inject the intact row output at exactly position p."""
    ids = bm._tokenize(tok, seq)
    if ids.shape[1] < 2:
        return None
    L = ids.shape[1]
    if p >= L:
        return None                      # position not present in this window
    if seq in intact_cache:
        intact_row = intact_cache[seq]
    else:
        intact_row = bm._forced_capture(model, tok, seq)
        intact_cache[seq] = intact_row
    saved = e12.set_row_alpha(model, bm.PATTERN, bm.LAYER, bm.ROW, 0.0)
    inj = lambda L_, pf, tp=p: ([tp], intact_row[tp:tp + 1])
    h = bm.register(model, bm.PositionRowHook(bm.ROW, inject_fn=inj))
    try:
        loss = bm._run_loss(model, ids)
    finally:
        h.remove()
        e12.restore_row(model, bm.PATTERN, bm.LAYER, bm.ROW, saved)
    return loss, L


def main():
    t0 = time.time()
    log("loading model (frozen E12 revision)")
    model, tok, resolved = bm.load_model()
    log(f"  resolved revision: {resolved}")

    prompt_windows, damage_windows, corpus_meta = e12.build_corpora(
        seed=42, n_prompt=96, prompt_win_bp=170, n_damage=100, damage_win_bp=512)
    log(f"corpora: {corpus_meta}")

    # ---- provenance gate (identical to the frozen run) ----------------------
    intact_d = bm.damage(model, tok, damage_windows, "A_intact")
    abl_d = bm.damage(model, tok, damage_windows, "B_full_ablation_weight")
    ok_i = abs(intact_d - bm.HIST_DAMAGE_INTACT) <= bm.REPRO_RTOL * abs(bm.HIST_DAMAGE_INTACT)
    ok_a = abs(abl_d - bm.HIST_DAMAGE_ABLATED) <= bm.REPRO_RTOL * abs(bm.HIST_DAMAGE_ABLATED)
    log(f"  intact  {intact_d:.6f} vs historical {bm.HIST_DAMAGE_INTACT:.6f}  ok={ok_i}")
    log(f"  ablated {abl_d:.6f} vs historical {bm.HIST_DAMAGE_ABLATED:.6f}  ok={ok_a}")
    if not (ok_i and ok_a):
        json.dump({"status": "ABORTED_PROVENANCE_MISMATCH",
                   "intact_measured": intact_d, "ablated_measured": abl_d},
                  open(OUT / "generator_position_rescue.json", "w"), indent=2)
        log("PROVENANCE MISMATCH — aborting per protocol"); return

    # ---- per-window intact / ablated baselines (paired) ---------------------
    log("per-window baselines")
    intact_pw = np.array(bm.damage_per_window(model, tok, damage_windows, "A_intact"))
    abl_pw = np.array(bm.damage_per_window(model, tok, damage_windows, "B_full_ablation_weight"))
    # damage_windows are (chrom, start, seq) triples, matching bm.damage_per_window
    seq_lens = np.array([bm._tokenize(tok, w[2]).shape[1] for w in damage_windows])
    log(f"  seq_len: min={seq_lens.min()} median={int(np.median(seq_lens))} max={seq_lens.max()}")

    cache: dict = {}
    rows = []
    for p in POSITIONS:
        log(f"restore at p={p}")
        per_win, used = [], []
        for i, w in enumerate(damage_windows):
            r = nll_restore_at_p(model, tok, w[2], p, cache)
            if r is None:
                continue
            loss, L = r
            per_win.append(loss); used.append(i)
        if not per_win:
            log(f"  p={p}: no window long enough — skipped"); continue
        used = np.array(used); rp = np.array(per_win)
        ip, ap = intact_pw[used], abl_pw[used]
        resc = bm.rescue_fraction_bootstrap(rp, ap, ip) if False else None
        # rescue = (ablated - restored) / (ablated - intact), paired bootstrap
        num = ap - rp; den = ap - ip
        point = float(num.sum() / den.sum())
        rng = np.random.default_rng(42)
        bs = []
        n = len(used)
        for _ in range(5000):
            k = rng.integers(0, n, n)
            d = den[k].sum()
            if abs(d) > 1e-12:
                bs.append(num[k].sum() / d)
        lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (None, None)
        mean_T = float(seq_lens[used].mean())
        rows.append(dict(position=p, n_windows=int(n),
                         nll_restored=float(rp.mean()),
                         nll_intact=float(ip.mean()), nll_ablated=float(ap.mean()),
                         delta_vs_intact=float(rp.mean() - ip.mean()),
                         rescue_fraction=point, ci95_lo=lo, ci95_hi=hi,
                         mean_seq_len=mean_T,
                         accessibility=float(np.mean((seq_lens[used] - p) / seq_lens[used]))))
        log(f"  p={p:3d} n={n:3d} nll={rp.mean():.5f} rescue={point:+.4f} "
            f"CI[{lo:+.4f},{hi:+.4f}] access={rows[-1]['accessibility']:.3f}")

    payload = dict(experiment="EXP4_generator_position_rescue",
                   model="GENERator-EUK-3B", layer=bm.LAYER, row=bm.ROW,
                   revision=resolved, corpus_meta=corpus_meta,
                   endpoint="native causal-LM NLL (pooled per-window)",
                   provenance={"intact_measured": intact_d, "ablated_measured": abl_d,
                               "intact_historical": bm.HIST_DAMAGE_INTACT,
                               "ablated_historical": bm.HIST_DAMAGE_ABLATED,
                               "intact_ok": ok_i, "ablated_ok": ok_a},
                   positions=POSITIONS, rows=rows,
                   elapsed_seconds=time.time() - t0)
    json.dump(payload, open(OUT / "generator_position_rescue.json", "w"), indent=2)
    with open(OUT / "generator_position_rescue.tsv", "w") as f:
        cols = ["position","n_windows","nll_intact","nll_ablated","nll_restored",
                "delta_vs_intact","rescue_fraction","ci95_lo","ci95_hi",
                "accessibility","mean_seq_len"]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    log(f"saved -> {OUT/'generator_position_rescue.tsv'}")


if __name__ == "__main__":
    main()
