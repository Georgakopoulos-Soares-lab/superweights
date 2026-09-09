#!/usr/bin/env python3
"""
WITHIN-MODEL CALIBRATION SWEEP — does causal damage SCALE with activation ratio?

Every result so far is either (a) between-model, n=22, one row per model, confounded by model
identity, or (b) within-model but BINARY (candidate vs 5 random same-layer controls, which sit
at ratio ~1 by construction). Neither can show a ratio->damage SLOPE.

This measures both variables for MANY rows inside ONE layer of ONE model, so every
model-level confound (size, architecture, endpoint, baseline loss) is held fixed:

  1. one forward pass, hook layer L's down_proj -> per-row max|activation| over N windows
  2. ratio(row) = max_abs(row) / median over rows of max_abs
  3. select ~40 rows log-spaced BY RANK so the ratio range is spanned, not just the extremes
  4. ablate each row (alpha=0) and measure native NLL on the frozen damage windows
  5. Spearman(ratio, damage) WITHIN the model

Selection of the swept rows is by activation ratio only -- never by causal outcome.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import stats

ROOT = Path("/home/nvidia/superweights")
D = ROOT / "manuscript/experiments/E12_generator_degradation_control"
sys.path.insert(0, str(D))
import run_bos_mediation as bm
import e12_lib as e12

OUT = ROOT / "results/paper_closing"; OUT.mkdir(parents=True, exist_ok=True)
N_ACT_WIN, N_DMG_WIN, N_ROWS = 24, 40, 40


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    model, tok, resolved = bm.load_model()
    _, dmg, meta = e12.build_corpora(seed=42, n_prompt=96, prompt_win_bp=170,
                                     n_damage=100, damage_win_bp=512)
    act_win, dmg_win = dmg[:N_ACT_WIN], dmg[:N_DMG_WIN]
    mod = dict(model.named_modules())[bm.PATTERN.format(i=bm.LAYER)]
    n_rows = mod.weight.data.shape[0]

    # ---- 1-2. per-row activation ratio in this layer ------------------------
    log(f"measuring per-row activation over {N_ACT_WIN} windows, layer {bm.LAYER}, {n_rows} rows")
    peak = torch.zeros(n_rows, device=mod.weight.device)
    st = {}
    hh = mod.register_forward_hook(
        lambda _m, _i, o, _s=st: _s.__setitem__(
            "h", (o[0] if isinstance(o, tuple) else o).detach().float()))
    with torch.no_grad():
        for _c, _s, seq in act_win:
            ids = bm._tokenize(tok, seq)
            model(input_ids=ids)
            h = st["h"]; h = h if h.dim() == 2 else h[0]
            peak = torch.maximum(peak, h.abs().max(0).values)
    hh.remove()
    med = float(peak.median())
    ratio = (peak / max(med, 1e-12)).cpu().numpy()
    order = np.argsort(-ratio)
    log(f"  median row peak = {med:.4f}; top ratio = {ratio[order[0]]:.1f} (row {order[0]}); "
        f"frozen row {bm.ROW} ratio = {ratio[bm.ROW]:.1f} rank = {int(np.where(order==bm.ROW)[0][0])+1}")

    # ---- 3. rows log-spaced BY RANK (selection independent of any outcome) --
    ranks = np.unique(np.round(np.geomspace(1, n_rows, N_ROWS)).astype(int)) - 1
    rows_sel = [int(order[r]) for r in ranks]
    if bm.ROW not in rows_sel: rows_sel.append(bm.ROW)
    log(f"  swept rows: {len(rows_sel)} spanning ratio {ratio[rows_sel].min():.2f} .. {ratio[rows_sel].max():.1f}")

    # ---- 4. causal damage per row ------------------------------------------
    intact = bm.damage(model, tok, dmg_win, "A_intact")
    log(f"  intact NLL = {intact:.5f}")
    recs = []
    for i, r in enumerate(rows_sel):
        saved = e12.set_row_alpha(model, bm.PATTERN, bm.LAYER, r, 0.0)
        try:
            d = bm.damage(model, tok, dmg_win, "A_intact")
        finally:
            e12.restore_row(model, bm.PATTERN, bm.LAYER, r, saved)
        rel = (d - intact) / max(abs(intact), 1e-9)
        recs.append(dict(row=r, rank=int(np.where(order == r)[0][0]) + 1,
                         activation_ratio=float(ratio[r]), nll=d,
                         delta_nll=d - intact, rel_delta=rel,
                         is_frozen_candidate=bool(r == bm.ROW)))
        if i % 8 == 0 or r == bm.ROW:
            log(f"   row {r:5d} rank {recs[-1]['rank']:5d} ratio {ratio[r]:9.2f} "
                f"relΔ {rel:+.5f}{'   <- frozen candidate' if r == bm.ROW else ''}")

    # ---- 5. within-model slope ---------------------------------------------
    x = np.array([r["activation_ratio"] for r in recs])
    y = np.array([r["rel_delta"] for r in recs])
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(42); bs = []
    for _ in range(10000):
        k = rng.integers(0, len(x), len(x))
        if len(np.unique(x[k])) > 2: bs.append(stats.spearmanr(x[k], y[k]).statistic)
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    # excluding the frozen candidate: does the slope survive without the extremum?
    m = np.array([not r["is_frozen_candidate"] for r in recs])
    rho_x, p_x = stats.spearmanr(x[m], y[m])
    log(f"\n  WITHIN-MODEL Spearman(ratio, rel damage) = {rho:+.4f}  p={p:.5f}  "
        f"CI[{lo:+.3f},{hi:+.3f}]  n={len(x)}")
    log(f"  excluding the frozen candidate           = {rho_x:+.4f}  p={p_x:.5f}  n={int(m.sum())}")

    json.dump(dict(experiment="WITHIN_MODEL_SLOPE", model="GENERator-EUK-3B",
                   layer=bm.LAYER, frozen_row=bm.ROW, revision=resolved,
                   n_act_windows=N_ACT_WIN, n_damage_windows=N_DMG_WIN,
                   intact_nll=intact, median_row_peak=med, corpus_meta=meta,
                   spearman_rho=float(rho), p_value=float(p), ci95=[lo, hi], n=len(x),
                   spearman_rho_excl_candidate=float(rho_x), p_excl_candidate=float(p_x),
                   rows=recs, elapsed_seconds=time.time() - t0),
              open(OUT / "within_model_slope.json", "w"), indent=2)
    cols = ["row","rank","activation_ratio","nll","delta_nll","rel_delta","is_frozen_candidate"]
    with open(OUT / "within_model_slope.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in recs:
            f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    log("saved")


if __name__ == "__main__":
    main()
