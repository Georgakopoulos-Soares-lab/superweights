#!/usr/bin/env python
"""
E10 Step D3/D4 -- decoder causal-concentration statistics and the mechanical per-model
decision.

Every threshold and formula below is copied verbatim from the locked v2 prereg
(PREREG_E10_nlp_architecture_causal_v2.md, sha256 3e4b991d...). Nothing is re-derived,
and nothing is tuned after seeing responses.

  C1 = |dL_top1| / sum_i |dL_i|
  C2 = (|dL_top1| + |dL_top2|) / sum_i |dL_i|
  control_normalized = target_effect / median(|control_effect| + 1e-6)

  SINGLE_COMPONENT_DOMINANT      : C1 > 0.5 AND top row's control_normalized > 3.0
  MULTI_COMPONENT_CANDIDATE      : not dominant, but >= 2 rows clear control_normalized > 3.0
  STRUCTURAL_CAUSAL_DISSOCIATION : top row's control_normalized <= 3.0
  (K=1 models: C1 = 1.0 trivially, so the decision collapses to the control test.)

Pure numpy -- no GPU, no model loading.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # .../genomic-super-weights (repo root)
RES = ROOT / "results"

MODELS = ["llama", "mistral", "olmo", "phi3", "qwen25"]
EPS_SMALL = 1e-6
C1_THRESHOLD = 0.5
CONTROL_NORM_THRESHOLD = 3.0
N_BOOT = 5000
RNG = np.random.default_rng(20260823)


def boot_ci_delta(base_pb, cond_pb):
    """Paired bootstrap over context batches for one condition's delta-NLL."""
    base_pb, cond_pb = np.array(base_pb, float), np.array(cond_pb, float)
    nb = len(base_pb)
    out = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, nb, size=nb)
        b = base_pb[idx, 0].sum() / max(base_pb[idx, 1].sum(), 1)
        c = cond_pb[idx, 0].sum() / max(cond_pb[idx, 1].sum(), 1)
        out.append(c - b)
    out = np.array(out)
    return {"ci_2.5": float(np.percentile(out, 2.5)),
            "ci_97.5": float(np.percentile(out, 97.5)),
            "excludes_zero": bool(np.percentile(out, 2.5) > 0 or np.percentile(out, 97.5) < 0)}


def analyze(model):
    d = json.loads((RES / f"e10_decoder_spectrum_{model}.json").read_text())
    base_pb = d["baseline_per_batch"]
    conds = d["conditions"]

    targets = [c for c in conds if c["kind"] == "target"]          # alpha=0, ranked
    halves = [c for c in conds if c["kind"] == "target_half"]      # alpha=0.5, rank-1 only
    controls = [c for c in conds if c["kind"] == "control"]
    targets.sort(key=lambda c: c["rank"])

    ctrl_abs = np.array([abs(c["delta_nll"]) for c in controls])
    ctrl_med = float(np.median(ctrl_abs))
    denom = ctrl_med + EPS_SMALL

    deltas = np.array([c["delta_nll"] for c in targets])
    abs_deltas = np.abs(deltas)
    tot = float(abs_deltas.sum())

    # C1/C2 use absolute effects (prereg); signed effects reported separately.
    order = np.argsort(-abs_deltas)
    c1 = float(abs_deltas[order[0]] / tot) if tot > 0 else float("nan")
    c2 = (float((abs_deltas[order[0]] + abs_deltas[order[1]]) / tot)
          if tot > 0 and len(abs_deltas) > 1 else (1.0 if len(abs_deltas) == 1 else float("nan")))

    rows = []
    for c in targets:
        cn = c["delta_nll"] / denom
        rows.append({
            "structural_rank": c["rank"], "layer": c["layer"], "row": c["row"],
            "delta_nll": c["delta_nll"], "abs_delta_nll": abs(c["delta_nll"]),
            "relative_pct_change": c["relative_pct_change"],
            "control_normalized": cn,
            "control_normalized_abs": abs(c["delta_nll"]) / denom,
            "clears_control_threshold": bool(cn > CONTROL_NORM_THRESHOLD),
            "mean_logit_kl": c["mean_logit_kl_vs_baseline"],
            "mean_next_token_entropy": c["mean_next_token_entropy"],
            "bootstrap": boot_ci_delta(base_pb, c["per_batch"]),
        })

    # Per the locked prereg, "the top row" in the D4 rule is THE SAME ROW C1 is computed
    # from -- the largest-|effect| row -- not the structurally-ranked #1 row. (Verified
    # against PREREG_E10_nlp_architecture_causal_v2.md lines 171-185: "C1 ... the single
    # largest row already accounts for..." / "AND the top row's control-normalized
    # effect" -- same referent throughout.) This is a pre-decision bugfix in the analysis
    # code to match the already-locked text, not a change to the rule itself: no decision
    # had been reported before this was caught, on OLMo's real data, mid-run.
    top_row = rows[int(order[0])]
    n_clearing = sum(1 for r in rows if r["clears_control_threshold"])

    if c1 > C1_THRESHOLD and top_row["control_normalized"] > CONTROL_NORM_THRESHOLD:
        decision = "SINGLE_COMPONENT_DOMINANT"
    elif n_clearing >= 2:
        decision = "MULTI_COMPONENT_CANDIDATE"
    elif top_row["control_normalized"] <= CONTROL_NORM_THRESHOLD:
        decision = "STRUCTURAL_CAUSAL_DISSOCIATION"
    else:
        decision = "INTERMEDIATE_NO_CLEAN_DECISION"

    half = None
    if halves:
        h = halves[0]
        half = {"layer": h["layer"], "row": h["row"], "alpha": 0.5,
                "delta_nll": h["delta_nll"],
                "monotonic_vs_full": bool(0 <= abs(h["delta_nll"]) <= abs(top_row["delta_nll"]))
                if top_row["delta_nll"] != 0 else None,
                "mean_logit_kl": h["mean_logit_kl_vs_baseline"]}

    return {
        "model": model, "repo": d["repo"], "K": len(targets),
        "baseline_nll": d["baseline_nll"], "baseline_perplexity": d["baseline_perplexity"],
        "control_layer": d["control_layer"], "control_rows": d["control_rows"],
        "control_median_abs_delta": ctrl_med,
        "controls": [{"row": c["row"], "delta_nll": c["delta_nll"],
                      "relative_pct_change": c["relative_pct_change"]} for c in controls],
        "target_rows": rows,
        "C1": c1, "C2": c2,
        "n_rows_clearing_control_threshold": n_clearing,
        "rank1_control_normalized": top_row["control_normalized"],
        "alpha_half_rank1": half,
        "decision": decision,
    }


def main():
    out = {"thresholds": {"C1": C1_THRESHOLD, "control_normalized": CONTROL_NORM_THRESHOLD,
                          "epsilon_small": EPS_SMALL},
           "prereg": "PREREG_E10_nlp_architecture_causal_v2.md", "models": {}}
    for m in MODELS:
        f = RES / f"e10_decoder_spectrum_{m}.json"
        if not f.exists():
            print(f"[skip] {m}: no responses yet")
            continue
        r = analyze(m)
        out["models"][m] = r
        print(f"\n{'='*66}\n{m}  (K={r['K']}, baseline NLL {r['baseline_nll']:.4f})\n{'='*66}")
        print(f"  control median |dNLL| = {r['control_median_abs_delta']:.6g}")
        for t in r["target_rows"]:
            print(f"  rank{t['structural_rank']} L{t['layer']}/r{t['row']}: "
                  f"dNLL={t['delta_nll']:+.6f} ({t['relative_pct_change']:+.2f}%)  "
                  f"ctrl_norm={t['control_normalized']:+.2f}"
                  f"{' *' if t['clears_control_threshold'] else ''}  "
                  f"KL={t['mean_logit_kl']:.5f}")
        print(f"  C1={r['C1']:.4f}  C2={r['C2']:.4f}  "
              f"rows clearing 3.0x: {r['n_rows_clearing_control_threshold']}/{r['K']}")
        print(f"  DECISION: {r['decision']}")

    p = RES / "e10_decoder_concentration.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
