#!/usr/bin/env python3
"""
EXPERIMENT 2 (text decoders) — apply ONE uniform selection rule and re-evaluate causally.

Companion to run_uniform_detector.py, which covers the two GENOMIC legacy models (NTv3,
DNABERT-2). This script covers the three remaining legacy models, all text decoders:

    Llama-7B          frozen L2/r3968     huggyllama/llama-7b
    Mistral-7B        frozen L1/r2070     mistralai/Mistral-7B-v0.1
    OLMo-7B-0724-hf   frozen L1/r269      allenai/OLMo-7B-0724-hf

Why these three
---------------
audit/detector_provenance.csv records that 6 of 22 census candidates were chosen by the
LEGACY rule (global argmax of ABSOLUTE activation) rather than the current rule (global
argmax of the LAYER-RELATIVE ratio). Two of the six have since been checked and both were
FALSE -- the frozen coordinate is NOT the ratio-argmax (NTv3: frozen ratio_rank=3; DNABERT-2:
ratio-argmax is L8/r603, not the frozen L5/r603). One (GENERator-EUK-3B) was checked and
passed. These three were never checked, so their row of the provenance table still reads
"NOT FOUND (not verified this audit)". Until they are checked we cannot say the census used
one uniform rule, and any cohort-level statement conditioned on the selection rule is
unverified for 3/22 of its rows.

Both outcomes are publishable and neither is a crisis:
  * frozen == ratio-argmax  -> the legacy and current rules agree here; provenance closes.
  * frozen != ratio-argmax  -> exactly the DNABERT-2/NTv3 pattern, and the causal comparison
    below says whether picking the "wrong" coordinate mattered for any reported number.

Uniform rule (identical wording to run_uniform_detector.py)
-----------------------------------------------------------
  1. statistic = LAYER-RELATIVE ratio, never absolute activation
       ratio(l,r) = max_t|a[l,r,t]| / median_r'( max_t|a[l,r',t]| )
  2. scope     = global argmax over ALL layers x rows
  3. special-token handling declared, and BOTH settings reported
  4. multiple diverse inputs, with a rank-1 stability requirement
  5. report rank + ratio of the frozen coordinate so "is it the argmax?" is answerable
     from the artifact without a re-run

The frozen census is NOT modified. We report the frozen candidate and the uniform-rule
candidate and -- where they differ -- causally evaluate BOTH under the SAME paired endpoint,
plus 5 seeded same-layer controls for the uniform candidate, so the comparison is internally
valid rather than read across two different eval pools.

Self-validating: four reproduction gates
----------------------------------------
The endpoint, eval pool, batch size, ablation parameterization and control-draw RNG are the
census's own, so four stored quantities must reproduce before any new number is trusted:
    (1) baseline_loss                      census_master.csv
    (2) R_cand_eps1.0 at the frozen coord   census_master.csv
    (3) R_cand_eps0.5 at the frozen coord   census_master.csv
    (4) the 5 stored control rows           census_master.csv, redrawn from
                                            SeedSequence(42).spawn(23)[panel_index]
A gate failure means the harness is wrong, not that the science changed. The run aborts.

Usage (one model per invocation; ~25 min each, needs 1 GPU with >=40 GB)
------------------------------------------------------------------------
    python scripts/paper_closing/run_uniform_detector_text.py --model llama
    python scripts/paper_closing/run_uniform_detector_text.py --model mistral
    python scripts/paper_closing/run_uniform_detector_text.py --model olmo
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import stats  # noqa: F401  (kept for parity with the sibling scripts)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper-salvage/experiments/E10_nlp_architecture_causal"))
import e10_lib as L  # noqa: E402

OUT = ROOT / "results/paper_closing"
SEED = 42
THRESHOLD = 5.0            # the E11/E13 detector's own accept rule, used unchanged
PATTERN = "model.layers.{i}.mlp.down_proj"

# panel_index is the census's own PANEL_ORDER position; it selects the control RNG stream.
MODELS = {
    "llama": dict(
        label="Llama-7B", repo="huggyllama/llama-7b",
        revision="4782ad278652c7c71b72204d462d6d01eaaf7549",
        layer=2, row=3968, panel_index=0, batch_size=4,
        baseline_loss=2.2452528551553326,
        r_cand_eps1=3.007945593585156, r_cand_eps05=0.03855088724231734,
        controls=[892, 2034, 2379, 3729, 3751]),
    "mistral": dict(
        label="Mistral-7B", repo="mistralai/Mistral-7B-v0.1",
        revision="27d67f1b5f57dc0953326b2601d68371d40ea8da",
        layer=1, row=2070, panel_index=1, batch_size=4,
        baseline_loss=2.207224828193799,
        r_cand_eps1=2.871492377308394, r_cand_eps05=3.013789457796915,
        controls=[190, 305, 746, 1912, 2687]),
    "olmo": dict(
        label="OLMo-7B-0724-hf", repo="allenai/OLMo-7B-0724-hf",
        revision="1ee306df318ee15bfe4a76ebd5c002b0105b1ab6",
        layer=1, row=269, panel_index=2, batch_size=4,
        baseline_loss=2.4953157010610325,
        r_cand_eps1=1.1177580569022323, r_cand_eps05=0.006288051768048048,
        controls=[292, 437, 2877, 2908, 3160]),
    # Positive control, not a legacy model. SmolLM2-1.7B was selected by the CURRENT
    # ratio-argmax rule, so detector_provenance.csv records frozen == ratio-argmax "by
    # construction". EXP2 must therefore return rules_agree=True here. If it does not, the
    # detector implementation in this file is wrong -- run this before the three real
    # targets, as a harness check.
    "smollm2-1.7b": dict(
        label="HuggingFaceTB/SmolLM2-1.7B", repo="HuggingFaceTB/SmolLM2-1.7B",
        revision="effd688a12921b4cc83e3312b6feb579f70f9c71",
        layer=7, row=227, panel_index=16, batch_size=8,
        baseline_loss=2.6145340592893835,
        r_cand_eps1=6.988738472947954, r_cand_eps05=0.03905575617362741,
        controls=[317, 582, 1422, 1547, 2043]),
}
N_PANEL = 23               # len(PANEL_ORDER); fixes the spawn count, do not change
EPSILONS = (0.5, 1.0)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_windows_local(parquet, tok, n_windows, max_tokens=512, seed=SEED):
    """Byte-for-byte reimplementation of e10_lib.build_windows off a local parquet, for
    machines without the `datasets` WikiText cache. Same non-empty filter, same
    random.Random(42) shuffle, same fill-to-512 concatenation."""
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


def census_controls(d_model, layer_excluded_rows, panel_index):
    """Replicate build_candidate_manifest.py's control draw exactly."""
    seed_seqs = np.random.SeedSequence(SEED).spawn(N_PANEL)
    rng = np.random.default_rng(seed_seqs[panel_index])
    pool = np.setdiff1d(np.arange(d_model), np.asarray(sorted(layer_excluded_rows)),
                        assume_unique=True)
    return sorted(int(x) for x in rng.choice(pool, size=5, replace=False))


def measure_ratios(model, tok, windows, n_layers, add_special_tokens):
    """Per-row max|activation| at EVERY layer's down_proj output, over all windows.

    Returns (ratio[n_layers, n_rows], peak[n_layers, n_rows], per_input_rank_of_argmax).
    `add_special_tokens` re-tokenizes the decoded window text so BOTH conventions are
    reported, as the uniform rule requires.
    """
    store, handles = {}, []
    for li in range(n_layers):
        mod = L._resolve_module(model, PATTERN, li)
        handles.append(mod.register_forward_hook(
            lambda _m, _i, o, _l=li: store.__setitem__(
                _l, (o[0] if isinstance(o, tuple) else o).detach().float())))
    n_rows = L._resolve_module(model, PATTERN, 0).weight.data.shape[0]
    peak = torch.zeros(n_layers, n_rows, device="cuda")
    per_input_argmax = []
    with torch.no_grad():
        for ids in windows:
            if add_special_tokens is None:
                x = ids.unsqueeze(0).cuda()                 # census ids verbatim
            else:
                text = tok.decode(ids, skip_special_tokens=True)
                x = tok(text, return_tensors="pt", truncation=True, max_length=512,
                        add_special_tokens=add_special_tokens)["input_ids"].cuda()
            model(input_ids=x)
            cur = torch.stack([store[li].reshape(-1, n_rows).abs().max(0).values
                               for li in range(n_layers)])
            peak = torch.maximum(peak, cur)
            r = cur / cur.median(dim=1, keepdim=True).values.clamp(min=1e-12)
            per_input_argmax.append(tuple(int(v) for v in
                                          np.unravel_index(int(r.argmax()), r.shape)))
    for h in handles: h.remove()
    ratio = (peak / peak.median(dim=1, keepdim=True).values.clamp(min=1e-12)).cpu().numpy()
    return ratio, peak.cpu().numpy(), per_input_argmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--n_act_windows", type=int, default=24)
    ap.add_argument("--n_eval_windows", type=int, default=100)
    ap.add_argument("--wikitext_parquet",
                    default=str(ROOT / "frozen_inputs/wikitext_repo/wikitext-2-raw-v1/"
                                       "test-00000-of-00001.parquet"))
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--no_gate", action="store_true",
                    help="record gate misses instead of aborting (for triage only)")
    args = ap.parse_args()
    M = MODELS[args.model]
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(revision=M["revision"], local_files_only=args.local_files_only)
    log(f"loading {M['repo']}@{M['revision'][:8]} fp32")
    tok = AutoTokenizer.from_pretrained(M["repo"], **kw)
    model = AutoModelForCausalLM.from_pretrained(M["repo"], dtype=torch.float32, **kw)
    model.eval().cuda()
    n_layers = model.config.num_hidden_layers
    d_model = L._resolve_module(model, PATTERN, 0).weight.data.shape[0]
    log(f"  {n_layers} layers x {d_model} down_proj rows")

    # ---- frozen eval pool (census protocol) --------------------------------
    try:
        windows = L.build_windows(tok, args.n_eval_windows, 512, seed=SEED)
        corpus_src = "datasets.load_dataset(wikitext,wikitext-2-raw-v1,test)"
    except Exception as e:
        log(f"  load_dataset unavailable ({type(e).__name__}); using local parquet")
        windows = build_windows_local(args.wikitext_parquet, tok, args.n_eval_windows,
                                      512, SEED)
        corpus_src = f"local parquet {args.wikitext_parquet}"
    act_win = windows[:args.n_act_windows]
    units = L.batch_windows(windows, M["batch_size"])
    log(f"  {len(windows)} eval windows -> {len(units)} units of {M['batch_size']}; "
        f"{len(act_win)} activation windows; corpus = {corpus_src}")

    # ---- uniform detector, BOTH special-token conventions ------------------
    detector = {}
    for tag, ast in (("census_ids", None), ("add_special_tokens_true", True),
                     ("add_special_tokens_false", False)):
        ratio, peak, per_in = measure_ratios(model, tok, act_win, n_layers, ast)
        gl, gr = (int(v) for v in np.unravel_index(int(ratio.argmax()), ratio.shape))
        flat = ratio.ravel()
        fz = ratio[M["layer"], M["row"]]
        detector[tag] = dict(
            uniform_layer=gl, uniform_row=gr, uniform_ratio=float(ratio[gl, gr]),
            frozen_ratio=float(fz),
            frozen_global_rank=int((flat > fz).sum()) + 1, n_coords=int(flat.size),
            frozen_is_ratio_argmax=bool(gl == M["layer"] and gr == M["row"]),
            frozen_within_layer_rank=int((ratio[M["layer"]] > fz).sum()) + 1,
            frozen_passes_threshold=bool(fz >= THRESHOLD),
            per_input_argmax_stability=float(
                np.mean([c == (gl, gr) for c in per_in])),
            frac_inputs_frozen_is_argmax=float(
                np.mean([c == (M["layer"], M["row"]) for c in per_in])))
        d = detector[tag]
        log(f"  [{tag}] uniform argmax = L{gl}/r{gr} ratio {d['uniform_ratio']:.2f} | "
            f"frozen L{M['layer']}/r{M['row']} ratio {d['frozen_ratio']:.2f} "
            f"rank {d['frozen_global_rank']}/{d['n_coords']} | "
            f"frozen_is_argmax={d['frozen_is_ratio_argmax']} | "
            f"argmax stable in {100*d['per_input_argmax_stability']:.0f}% of inputs")

    primary = detector["census_ids"]
    uni = (primary["uniform_layer"], primary["uniform_row"])
    frozen = (M["layer"], M["row"])

    # ---- causal endpoint ----------------------------------------------------
    def nll_all():
        s = n = 0
        for ids in units:
            a, b, lp, _, _ = L.causal_lm_batch(model, ids.cuda())
            s += a; n += b; del lp
        return s / max(n, 1)

    with torch.no_grad():
        intact = nll_all()
    log(f"  intact NLL = {intact:.10f}  (census = {M['baseline_loss']:.10f})")

    def evaluate(layer, row, eps):
        alpha = L.alphas_for_mask([1], eps)[0]
        with L.masked(model, PATTERN, [(layer, row)], [alpha]):
            with torch.no_grad():
                d = nll_all()
        return d, (d - intact) / max(abs(intact), 1e-9)

    conds = []
    for eps in EPSILONS:
        nll, rel = evaluate(*frozen, eps)
        conds.append(dict(kind="frozen_candidate", layer=frozen[0], row=frozen[1],
                          epsilon=eps, nll=nll, rel_delta=rel))
        log(f"   frozen  L{frozen[0]}/r{frozen[1]} eps={eps}  relD {rel:+.6f}")

    ctrl_rows_frozen_layer = census_controls(d_model, [frozen[1]], M["panel_index"])
    gates = {
        "baseline_loss": dict(measured=intact, expected=M["baseline_loss"],
                              rel_err=abs(intact - M["baseline_loss"]) /
                                      abs(M["baseline_loss"])),
        "R_cand_eps1.0": dict(measured=next(c["rel_delta"] for c in conds
                                            if c["epsilon"] == 1.0),
                              expected=M["r_cand_eps1"]),
        "R_cand_eps0.5": dict(measured=next(c["rel_delta"] for c in conds
                                            if c["epsilon"] == 0.5),
                              expected=M["r_cand_eps05"]),
    }
    for k in ("R_cand_eps1.0", "R_cand_eps0.5"):
        g = gates[k]
        g["rel_err"] = abs(g["measured"] - g["expected"]) / max(abs(g["expected"]), 1e-12)
    gates["control_rows"] = dict(measured=ctrl_rows_frozen_layer, expected=M["controls"],
                                 rel_err=0.0 if ctrl_rows_frozen_layer == M["controls"]
                                         else 1.0)
    log("\n  reproduction gates:")
    bad = []
    for k, g in gates.items():
        ok = g["rel_err"] <= args.tol
        g["pass"] = bool(ok)
        log(f"    {k:16s} measured={g['measured']} expected={g['expected']} "
            f"rel_err={g['rel_err']:.2e}  {'PASS' if ok else 'FAIL'}")
        if not ok: bad.append(k)
    if bad and not args.no_gate:
        raise SystemExit(f"[abort] reproduction gate(s) failed: {bad} -- the harness is "
                         f"wrong, not the science. Fix before interpreting anything below.")

    # ---- if the uniform rule disagrees, evaluate ITS candidate too ---------
    if uni != frozen:
        log(f"\n  uniform rule DISAGREES: L{uni[0]}/r{uni[1]} vs frozen "
            f"L{frozen[0]}/r{frozen[1]} -- evaluating both on the same pool")
        for eps in EPSILONS:
            nll, rel = evaluate(*uni, eps)
            conds.append(dict(kind="uniform_candidate", layer=uni[0], row=uni[1],
                              epsilon=eps, nll=nll, rel_delta=rel))
            log(f"   uniform L{uni[0]}/r{uni[1]} eps={eps}  relD {rel:+.6f}")
        ctrl_uni = census_controls(d_model, [uni[1]], M["panel_index"])
        for cr in ctrl_uni:
            nll, rel = evaluate(uni[0], cr, 1.0)
            conds.append(dict(kind="uniform_control", layer=uni[0], row=cr,
                              epsilon=1.0, nll=nll, rel_delta=rel))
        med = float(np.median([c["rel_delta"] for c in conds
                               if c["kind"] == "uniform_control"]))
        r_uni = next(c["rel_delta"] for c in conds
                     if c["kind"] == "uniform_candidate" and c["epsilon"] == 1.0)
        log(f"   uniform controls L{uni[0]} rows {ctrl_uni}: median relD {med:+.3e}  "
            f"-> G_eps1.0 = {r_uni - med:+.6f}")
    else:
        log("\n  uniform rule AGREES with the frozen candidate -- provenance closes for "
            "this model, no second causal panel needed")
        ctrl_uni, med, r_uni = None, None, None

    import transformers as _tf
    payload = dict(
        experiment="EXP2_UNIFORM_DETECTOR_TEXT", model=M["label"], repo=M["repo"],
        revision=M["revision"], panel_index=M["panel_index"],
        frozen_candidate=dict(layer=frozen[0], row=frozen[1]),
        uniform_candidate=dict(layer=uni[0], row=uni[1]),
        rules_agree=bool(uni == frozen),
        n_layers=n_layers, d_model=d_model, detector_threshold=THRESHOLD,
        detector=detector, corpus_source=corpus_src,
        n_act_windows=len(act_win), n_eval_windows=len(windows),
        batch_size=M["batch_size"], intact_nll=intact,
        reproduction_gates=gates, conditions=conds,
        uniform_control_rows=ctrl_uni,
        uniform_median_control_eps1=med, uniform_G_eps1=None if med is None
        else float(r_uni - med),
        provenance=dict(seed=SEED, torch=torch.__version__,
                        transformers=_tf.__version__, pattern=PATTERN,
                        control_stream=f"SeedSequence(42).spawn(23)[{M['panel_index']}]",
                        wikitext_parquet=args.wikitext_parquet),
        elapsed_seconds=time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"uniform_detector_text_{args.model}.json"
    p.write_text(json.dumps(payload, indent=2))
    log(f"saved -> {p}")


if __name__ == "__main__":
    main()
