#!/usr/bin/env python3
"""
ADDENDUM to the SmolLM2-1.7B within-model sweep — characterise the SECOND critical row.

What the sweep turned up (not something we went looking for)
------------------------------------------------------------
The 36-row graded sweep of SmolLM2-1.7B layer 7 found TWO rows whose single-row ablation is
catastrophic, not one:

    row 227  ratio 3181.70  (ratio rank 1)   rel. NLL increase  +6.9887
    row 161  ratio   63.54  (ratio rank 4)   rel. NLL increase  +2.8718

and, in the same layer, a row the ratio ranks ABOVE row 161 that is nearly harmless:

    row 749  ratio  358.56  (ratio rank 2)   rel. NLL increase  +0.0221

So a row ranked 4th by activation ratio is ~130x more damaging than the row ranked 2nd. The
census reports one candidate per model, so row 161 has never appeared in any artifact.

The question this raises
------------------------
DNABERT-2's critical unit is a PAIR that is EMERGENT: each row alone is inert (-0.02 / -0.11
pp) and only the joint ablation is catastrophic (-33.76 pp), epistasis -33.63. SmolLM2's two
rows are each independently catastrophic, which is the opposite phenotype. Is there any
interaction on top of that, or are they simply two separate single points of failure?

    epistasis(a,b) = rel_delta(a AND b) - [rel_delta(a) + rel_delta(b)]

    ~0        two independent single points of failure (additive)
    negative  synergy -- joint damage exceeds the sum (the DNABERT-2 sign)
    positive  sub-additive -- they partly substitute for each other, i.e. some of the damage
              is shared/redundant rather than row-specific

No outcome here is a crisis; this is a descriptive measurement of something the sweep exposed.
Kill condition stated first: if epistasis is ~0 within the noise of the endpoint, we describe
SmolLM2 layer 7 as carrying two independent critical rows and make NO ensemble claim.

Endpoint is the census's own (frozen 100-window WikiText-2 pool, batch 8, causal-LM NLL), so
the same reproduction gates apply and are checked before anything is interpreted.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/frozen/E10_nlp_architecture_causal"))
import e10_lib as L  # noqa: E402

OUT = ROOT / "results/analyses/within_layer_sweep"
REPO, REV = "HuggingFaceTB/SmolLM2-1.7B", "effd688a12921b4cc83e3312b6feb579f70f9c71"
LAYER, PATTERN = 7, "model.layers.{i}.mlp.down_proj"
EXPECT_BASELINE, EXPECT_R227 = 2.6145340592893835, 6.988738472947954
# rows: the frozen candidate, the second critical row, the high-ratio near-harmless row
# (749 is included as the informative contrast, NOT as a control -- it is not randomly drawn)
ROWS = {"cand_227": 227, "second_161": 161, "highratio_749": 749}
RATIOS = {227: 3181.70, 161: 63.54, 749: 358.56}
SEED = 42


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    model = AutoModelForCausalLM.from_pretrained(REPO, revision=REV, dtype=torch.float32)
    model.eval().cuda()
    windows = L.build_windows(tok, 100, 512, seed=SEED)
    units = L.batch_windows(windows, 8)
    log(f"{len(windows)} windows -> {len(units)} units")

    def nll_all():
        s = n = 0
        for ids in units:
            a, b, lp, _, _ = L.causal_lm_batch(model, ids.cuda())
            s += a; n += b; del lp
        return s / max(n, 1)

    with torch.no_grad():
        intact = nll_all()
    log(f"intact NLL = {intact:.10f} (census {EXPECT_BASELINE:.10f})")

    def rel(coords):
        with L.masked(model, PATTERN, [(LAYER, r) for r in coords], [0.0] * len(coords)):
            with torch.no_grad():
                d = nll_all()
        return d, (d - intact) / max(abs(intact), 1e-9)

    singles = {}
    for name, r in ROWS.items():
        nll, rd = rel([r])
        singles[name] = dict(row=r, ratio=RATIOS[r], nll=nll, rel_delta=rd)
        log(f"  single row {r:4d} (ratio {RATIOS[r]:8.2f})  relD {rd:+.6f}")

    gates = {
        "baseline_loss": dict(measured=intact, expected=EXPECT_BASELINE,
                              rel_err=abs(intact - EXPECT_BASELINE) / EXPECT_BASELINE),
        "R_cand_eps1.0": dict(measured=singles["cand_227"]["rel_delta"], expected=EXPECT_R227,
                              rel_err=abs(singles["cand_227"]["rel_delta"] - EXPECT_R227)
                                      / EXPECT_R227)}
    bad = [k for k, g in gates.items() if g["rel_err"] > 0.02]
    for k, g in gates.items():
        g["pass"] = g["rel_err"] <= 0.02
        log(f"  gate {k:14s} rel_err={g['rel_err']:.2e} {'PASS' if g['pass'] else 'FAIL'}")
    if bad:
        raise SystemExit(f"[abort] gate(s) failed: {bad}")

    pairs = {}
    for a, b in (("cand_227", "second_161"), ("cand_227", "highratio_749"),
                 ("second_161", "highratio_749")):
        ra, rb = ROWS[a], ROWS[b]
        nll, rd = rel([ra, rb])
        eps = rd - (singles[a]["rel_delta"] + singles[b]["rel_delta"])
        pairs[f"{a}+{b}"] = dict(rows=[ra, rb], nll=nll, joint_rel_delta=rd,
                                 sum_of_singles=singles[a]["rel_delta"] + singles[b]["rel_delta"],
                                 epistasis=eps)
        log(f"  joint {ra:4d}+{rb:4d}  joint relD {rd:+.6f}  sum {pairs[f'{a}+{b}']['sum_of_singles']:+.6f}"
            f"  epistasis {eps:+.6f}")

    # SIGN CONVENTION. This endpoint is a LOSS INCREASE (higher = worse), so
    #   epistasis = joint - (a + b) > 0  =>  SUPER-additive (synergy): joint worse than sum
    #   epistasis < 0                    =>  SUB-additive: joint better than sum; strongly
    #                                        negative means one ablation MASKS/RESCUES the other.
    # This is the OPPOSITE sign to the DNABERT-2 epistasis numbers elsewhere in the project,
    # which use an ACCURACY endpoint (higher = better) where synergy is negative. Do not
    # compare the two signs without converting.
    def verdict_for(pr):
        e, sm = pr["epistasis"], pr["sum_of_singles"]
        frac = e / abs(sm) if sm else 0.0
        if frac > 0.05:
            v = "SUPER-ADDITIVE (synergy: joint damage exceeds the sum)"
        elif frac < -0.5:
            v = "MASKING (the joint ablation largely CANCELS the single-row damage)"
        elif frac < -0.05:
            v = "SUB-ADDITIVE (partly shared/overlapping damage)"
        else:
            v = "ADDITIVE within 5% (two independent single points of failure)"
        pr["joint_over_sum"] = float(pr["joint_rel_delta"] / sm) if sm else None
        pr["epistasis_frac_of_sum"] = float(frac)
        pr["verdict"] = v
        return v

    log("")
    for k, pr in pairs.items():
        v = verdict_for(pr)
        log(f"  {k:34s} joint {pr['joint_rel_delta']:+.4f} vs sum {pr['sum_of_singles']:+.4f} "
            f"({100*pr['joint_over_sum']:.1f}% of sum) -> {v}")
    verdict = pairs["cand_227+second_161"]["verdict"]

    import transformers as _tf
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "smollm2_second_row_epistasis.json").write_text(json.dumps(dict(
        experiment="SMOLLM2_SECOND_CRITICAL_ROW", model="HuggingFaceTB/SmolLM2-1.7B",
        repo=REPO, revision=REV, layer=LAYER, intact_nll=intact,
        note=("row 161 was surfaced by the 36-row graded sweep, not selected by the census "
              "detector; row 749 is the high-ratio low-damage contrast, not a random control"),
        singles=singles, pairs=pairs, verdict_227_161=verdict,
        sign_convention=("loss-increase endpoint: epistasis>0 = super-additive "
                         "synergy, epistasis<0 = sub-additive/masking. OPPOSITE sign "
                         "to the project's accuracy-endpoint epistasis numbers."),
        reproduction_gates=gates,
        provenance=dict(seed=SEED, torch=torch.__version__, transformers=_tf.__version__,
                        n_windows=len(windows), batch_size=8, pattern=PATTERN),
        elapsed_seconds=time.time() - t0), indent=2))
    log(f"saved -> {OUT / 'smollm2_second_row_epistasis.json'}")


if __name__ == "__main__":
    main()
