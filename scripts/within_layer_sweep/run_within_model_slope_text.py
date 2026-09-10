#!/usr/bin/env python3
"""
WITHIN-MODEL CALIBRATION SWEEP, TEXT DECODER — does causal damage SCALE with activation ratio?

Companion to run_within_model_slope.py (GENERator-EUK-3B, a genomic decoder), which found a
threshold-like rather than linear relationship (rho=+0.454, p=0.0054, n=36). That result was
measured in ONE model of ONE domain. If the ratio->damage relation is a property of gated-FFN
super rows in general, it must reproduce inside a TEXT decoder; if it is instead linear there,
the "activation ratio is a graded severity predictor" reading fails for the cohort that the
super-weight literature actually cares about, and we need to know that before a reviewer does.

Design (identical logic to the genomic version, with this model's OWN frozen protocol)
--------------------------------------------------------------------------------------
  1. per-row max|activation| at layer L's down_proj output, over the first N_ACT frozen
     WikiText-2 windows
  2. ratio(row) = max_abs(row) / median over rows of max_abs
  3. ~40 rows log-spaced BY RANK, so the ratio range is spanned rather than only its extremes
  4. ablate each row (alpha = 1 - epsilon, i.e. alpha=0 at epsilon=1.0) and measure
     causal-LM NLL on the frozen 100-window eval pool
  5. Spearman(ratio, relative damage) WITHIN the model

Row selection is by activation ratio ONLY -- never by causal outcome.

Why this run is self-validating
-------------------------------
The eval pool, batch size, ablation parameterization and NLL reducer are the census's own
(`run_singleton_census.score_text_decoder`), so two stored census quantities must reproduce
before any sweep number is trusted:
      baseline_loss     = 2.6145340592893835
      R_cand_eps1.0     = 6.988738472947954     (layer 7, row 227)
and one stored stability quantity cross-checks the activation hook:
      activation_ratio (median over 24 inputs) = 4439.34, rank 1/2048
A mismatch means the harness is wrong, not that the science changed -- the run aborts.

Corpus provenance
-----------------
The census resolved WikiText-2 through `datasets.load_dataset`. On a machine without that
cache, --wikitext_parquet reads the identical `wikitext-2-raw-v1` test parquet
(sha256 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91) and reproduces
`e10_lib.build_windows` line-for-line: same non-empty filter, same random.Random(42) shuffle,
same fill-to-512 concatenation. This is the same file, not a substitute corpus.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import stats

ROOT = Path("/home/nvidia/superweights")
sys.path.insert(0, str(ROOT / "experiments/frozen/E10_nlp_architecture_causal"))
import e10_lib as L  # noqa: E402

OUT = ROOT / "results/within_layer_sweep"
SEED = 42
DETECTOR_THRESHOLD = 5.0   # the E11/E13 detector's own >=5 rule, used unchanged here


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_windows_local(parquet, tok, n_windows, max_tokens=512, seed=SEED):
    """Byte-for-byte reimplementation of e10_lib.build_windows off a local parquet."""
    import random, pyarrow.parquet as pq
    text = pq.read_table(parquet, columns=["text"]).column("text").to_pylist()
    lines = [t.strip() for t in text if t and t.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    windows, buf, i = [], [], 0
    while len(windows) < n_windows and i < len(lines):
        buf.append(lines[i]); i += 1
        ids = tok("\n".join(buf), return_tensors=None)["input_ids"]
        if len(ids) >= max_tokens:
            windows.append(torch.tensor(ids[:max_tokens], dtype=torch.long)); buf = []
    if len(windows) < n_windows:
        raise RuntimeError(f"only built {len(windows)}/{n_windows} windows")
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--revision", default="effd688a12921b4cc83e3312b6feb579f70f9c71")
    ap.add_argument("--model_label", default="SmolLM2-1.7B")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--row", type=int, default=227)
    ap.add_argument("--pattern", default="model.layers.{i}.mlp.down_proj")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--n_act_windows", type=int, default=24)
    ap.add_argument("--n_eval_windows", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_rows", type=int, default=40)
    ap.add_argument("--wikitext_parquet",
                    default=str(ROOT / "frozen_inputs/wikitext_repo/wikitext-2-raw-v1/"
                                       "test-00000-of-00001.parquet"))
    ap.add_argument("--expect_baseline", type=float, default=2.6145340592893835)
    ap.add_argument("--expect_r_cand", type=float, default=6.988738472947954)
    ap.add_argument("--expect_ratio", type=float, default=4439.338244560368)
    ap.add_argument("--tol", type=float, default=0.02, help="relative tolerance on gates")
    ap.add_argument("--no_gate", action="store_true", help="record gate misses, do not abort")
    ap.add_argument("--out_stem", default="within_model_slope_smollm2_1.7b")
    args = ap.parse_args()

    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"loading {args.repo}@{args.revision[:8]} fp32")
    tok = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.repo, revision=args.revision,
                                                 dtype=torch.float32)
    model.eval().cuda()

    # ---- frozen eval pool (census protocol) --------------------------------
    try:
        windows = L.build_windows(tok, args.n_eval_windows, 512, seed=SEED)
        corpus_src = "datasets.load_dataset(wikitext,wikitext-2-raw-v1,test)"
    except Exception as e:
        log(f"  load_dataset unavailable ({type(e).__name__}); using local parquet")
        windows = build_windows_local(args.wikitext_parquet, tok, args.n_eval_windows, 512, SEED)
        corpus_src = f"local parquet {args.wikitext_parquet}"
    act_win = windows[:args.n_act_windows]
    units = L.batch_windows(windows, args.batch_size)
    log(f"  {len(windows)} eval windows -> {len(units)} units of {args.batch_size}; "
        f"{len(act_win)} activation windows; corpus = {corpus_src}")

    mod = dict(model.named_modules())[args.pattern.format(i=args.layer)]
    n_rows = mod.weight.data.shape[0]

    # ---- 1-2. per-row activation ratio at this layer ------------------------
    log(f"measuring per-row activation, layer {args.layer}, {n_rows} rows")
    peak = torch.zeros(n_rows, device=mod.weight.device)
    per_input, st = [], {}
    hh = mod.register_forward_hook(
        lambda _m, _i, o, _s=st: _s.__setitem__(
            "h", (o[0] if isinstance(o, tuple) else o).detach().float()))
    with torch.no_grad():
        for ids in act_win:
            model(input_ids=ids.unsqueeze(0).cuda())
            h = st["h"]; h = h.reshape(-1, h.shape[-1])
            m = h.abs().max(0).values
            peak = torch.maximum(peak, m)
            # per-input ratio, the E13 candidate-stability convention (cross-check only)
            per_input.append(float(m[args.row] / m.median().clamp(min=1e-12)))
    hh.remove()
    med = float(peak.median())
    ratio = (peak / max(med, 1e-12)).cpu().numpy()
    order = np.argsort(-ratio)
    cand_rank = int(np.where(order == args.row)[0][0]) + 1
    ratio_median_per_input = float(np.median(per_input))
    log(f"  median row peak = {med:.4f}; top ratio = {ratio[order[0]]:.1f} (row {order[0]})")
    log(f"  frozen row {args.row}: pooled ratio = {ratio[args.row]:.1f}  rank = {cand_rank}/{n_rows}")
    log(f"  frozen row {args.row}: per-input ratio median = {ratio_median_per_input:.2f} "
        f"(census stability = {args.expect_ratio:.2f})")

    gates = {}
    gates["activation_ratio"] = {
        "measured": ratio_median_per_input, "expected": args.expect_ratio,
        "rel_err": abs(ratio_median_per_input - args.expect_ratio) / args.expect_ratio,
        "rank_measured": cand_rank, "rank_expected": 1}

    # ---- 3. rows log-spaced BY RANK (selection independent of any outcome) --
    ranks = np.unique(np.round(np.geomspace(1, n_rows, args.n_rows)).astype(int)) - 1
    rows_sel = [int(order[r]) for r in ranks]
    if args.row not in rows_sel: rows_sel.append(args.row)
    log(f"  swept rows: {len(rows_sel)} spanning ratio "
        f"{ratio[rows_sel].min():.2f} .. {ratio[rows_sel].max():.1f}")

    # ---- 4. causal damage per row (census endpoint) -------------------------
    def nll_all():
        s = n = 0
        for ids in units:
            a, b, lp, _, _ = L.causal_lm_batch(model, ids.cuda())
            s += a; n += b; del lp
        return s / max(n, 1)

    with torch.no_grad():
        intact = nll_all()
    log(f"  intact NLL = {intact:.10f}  (census baseline_loss = {args.expect_baseline:.10f})")
    gates["baseline_loss"] = {"measured": intact, "expected": args.expect_baseline,
                              "rel_err": abs(intact - args.expect_baseline) /
                                         abs(args.expect_baseline)}

    alpha = L.alphas_for_mask([1], args.epsilon)[0]
    recs = []
    for i, r in enumerate(rows_sel):
        with L.masked(model, args.pattern, [(args.layer, r)], [alpha]):
            with torch.no_grad():
                d = nll_all()
        rel = (d - intact) / max(abs(intact), 1e-9)
        recs.append(dict(row=r, rank=int(np.where(order == r)[0][0]) + 1,
                         activation_ratio=float(ratio[r]), nll=d,
                         delta_nll=d - intact, rel_delta=rel,
                         is_frozen_candidate=bool(r == args.row)))
        if i % 8 == 0 or r == args.row:
            log(f"   [{i+1:2d}/{len(rows_sel)}] row {r:5d} rank {recs[-1]['rank']:5d} "
                f"ratio {ratio[r]:10.2f} relD {rel:+.6f}"
                f"{'   <- frozen candidate' if r == args.row else ''}")

    cand = next(x for x in recs if x["is_frozen_candidate"])
    gates["R_cand"] = {"measured": cand["rel_delta"], "expected": args.expect_r_cand,
                       "rel_err": abs(cand["rel_delta"] - args.expect_r_cand) /
                                  abs(args.expect_r_cand), "epsilon": args.epsilon}
    log("\n  reproduction gates:")
    bad = []
    for k, g in gates.items():
        ok = g["rel_err"] <= args.tol
        log(f"    {k:18s} measured={g['measured']:.6g} expected={g['expected']:.6g} "
            f"rel_err={g['rel_err']:.2e}  {'PASS' if ok else 'FAIL'}")
        g["pass"] = bool(ok)
        if not ok: bad.append(k)
    if bad and not args.no_gate:
        raise SystemExit(f"[abort] reproduction gate(s) failed: {bad} -- harness is wrong, "
                         f"not the science. Fix before interpreting the sweep.")

    # ---- 5. within-model slope ---------------------------------------------
    x = np.array([r["activation_ratio"] for r in recs])
    y = np.array([r["rel_delta"] for r in recs])

    def boot(xx, yy, n=10000):
        rng = np.random.default_rng(SEED); bs = []
        for _ in range(n):
            k = rng.integers(0, len(xx), len(xx))
            if len(np.unique(xx[k])) > 2: bs.append(stats.spearmanr(xx[k], yy[k]).statistic)
        return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)),
                len(bs)) if bs else (float("nan"), float("nan"), 0)

    rho, p = stats.spearmanr(x, y); lo, hi, nb = boot(x, y)
    m = np.array([not r["is_frozen_candidate"] for r in recs])
    rho_x, p_x = stats.spearmanr(x[m], y[m])
    # threshold-like vs linear: is there any graded signal AMONG ORDINARY ROWS, i.e. below
    # the detector's own >=5 rule? Pre-specified before looking, mirroring the genomic run.
    sub = x < DETECTOR_THRESHOLD
    rho_s, p_s = (stats.spearmanr(x[sub], y[sub]) if sub.sum() >= 4 else (float("nan"),) * 2)
    log(f"\n  WITHIN-MODEL Spearman(ratio, rel damage) = {rho:+.4f}  p={p:.6f}  "
        f"CI[{lo:+.3f},{hi:+.3f}]  n={len(x)}  (bootstrap n_valid={nb})")
    log(f"  excluding the frozen candidate           = {rho_x:+.4f}  p={p_x:.6f}  n={int(m.sum())}")
    log(f"  sub-threshold rows only (ratio<{DETECTOR_THRESHOLD:g})       = {rho_s:+.4f}  "
        f"p={p_s:.6f}  n={int(sub.sum())}")
    log(f"  median rel_delta: sub-threshold = {float(np.median(y[sub])):+.6g}   "
        f"candidate = {cand['rel_delta']:+.6g}")

    import transformers as _tf
    payload = dict(
        experiment="WITHIN_MODEL_SLOPE_TEXT", model=args.model_label, repo=args.repo,
        revision=args.revision, layer=args.layer, frozen_row=args.row, n_rows=n_rows,
        epsilon=args.epsilon, pattern=args.pattern,
        n_act_windows=len(act_win), n_eval_windows=len(windows), batch_size=args.batch_size,
        corpus_source=corpus_src, intact_nll=intact, median_row_peak=med,
        candidate_pooled_ratio=float(ratio[args.row]), candidate_rank=cand_rank,
        candidate_ratio_median_per_input=ratio_median_per_input,
        candidate_ratio_per_input=per_input,
        reproduction_gates=gates,
        spearman_rho=float(rho), p_value=float(p), ci95=[lo, hi], n=len(x),
        spearman_rho_excl_candidate=float(rho_x), p_excl_candidate=float(p_x),
        n_excl_candidate=int(m.sum()),
        detector_threshold=DETECTOR_THRESHOLD,
        spearman_rho_subthreshold=float(rho_s), p_subthreshold=float(p_s),
        n_subthreshold=int(sub.sum()),
        median_rel_delta_subthreshold=float(np.median(y[sub])) if sub.sum() else None,
        provenance={"seed": SEED, "torch": torch.__version__,
                    "transformers": _tf.__version__,
                    "wikitext_parquet": args.wikitext_parquet},
        rows=recs, elapsed_seconds=time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.out_stem}.json").write_text(json.dumps(payload, indent=2))
    cols = ["row", "rank", "activation_ratio", "nll", "delta_nll", "rel_delta",
            "is_frozen_candidate"]
    with (OUT / f"{args.out_stem}.tsv").open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in recs:
            f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    log(f"saved -> {OUT / (args.out_stem + '.json')}")


if __name__ == "__main__":
    main()
