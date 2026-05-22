# scripts/detection/run_hydra_test_evo2.py
"""
Hydra test for Evo2: iteratively ablate the largest-activation super weight
and measure whether perplexity eventually degrades.

Motivating observation
----------------------
Evo2 is completely robust to single super weight ablation (delta_pct < 0.01%)
despite harbouring very large activation outliers (out_max ~ 552 960 at
layer=29, similar in magnitude to GENERator's known super weight).

Hypothesis: "hydra effect" — when one outlier weight is ablated, another
takes its place and the model continues to function normally.  Only by
accumulating ablations can we eventually find a load-bearing weight.

Algorithm
---------
Round 0:
  1. Sweep activations on the intact model.
  2. Find the largest spike → candidate super weight (layer, row, col).
  3. Add (layer, row, col) to `ablated` list (col recorded for reference only).
  4. Measure perplexity with this ablation via superrow output hooks
     (needed because TE ignores in-place .weight.data writes; superrow
     zeros the entire output row so the effect is fully observable).
  5. Compute delta_pct from the original baseline.

Round k (k ≥ 1):
  - Re-run sweep with ALL previous superrow hooks active so the recorder
    sees the neuron as completely gone (single-scalar hooks only reduce
    output by ~1/d_inner, leaving the same row/col as top candidate).
  - Find the next biggest spike in the residual model.
  - Add it to `ablated` and repeat steps 4-5.

Stop when:
  - delta_pct ≥ --degradation_threshold  (model finally degraded → "hydra is dead")
  - --max_rounds reached                  (hydra survived every cut)
  - Same (layer, row, col) triplet re-detected (no new candidate)

Output
------
results/hydra_test_evo2.json — per-round record:
  {
    "baseline_ppl": float,
    "degradation_threshold_pct": float,
    "outcome": "degraded" | "max_rounds" | "redetection",
    "rounds": [
      {
        "round": int,
        "layer": int, "row": int, "col": int,
        "w_rc": float,
        "in_max": float, "out_max": float,
        "ppl": float,
        "delta_pct": float,
        "n_ablated": int
      },
      ...
    ]
  }
"""

import argparse
import json
import yaml
import torch
from pathlib import Path

from probes.dna_probes import get_probe
from detection.sweep import sweep
from detection.identify_spikes import find_spike_layer, extract_coords
from models import WRAPPER_MAP


# ──────────────────────────────────────────────────────────────────────────────
# Helper: sweep with ablation hooks active
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap(output):
    return (output[0], output[1:]) if isinstance(output, tuple) else (output, None)

def _rewrap(o, rest):
    return (o,) + rest if rest is not None else o


def sweep_with_ablations(wrapper, probe: str, ablated: list) -> dict:
    """
    Run activation sweep on the model with all ablation hooks active.

    Each entry in `ablated` is a dict: {layer: int, row: int, ...}

    Uses superrow ablation (zeros the entire output row) so that the
    ActivationRecorder sees the ablated neuron as completely gone and
    finds the next-highest row in subsequent rounds.  A single-scalar
    hook only removes ~1/11264 of the row's output, leaving the same
    row and column as the top candidate every time.

    Hooks are registered BEFORE ActivationRecorder hooks so the recorder
    captures the post-ablation output.
    """
    handles = []

    for entry in ablated:
        module = wrapper.get_target_module(entry["layer"])
        row = entry["row"]

        def make_hook(r):
            def _hook(mod, inp, output):
                o, rest = _unwrap(output)
                o = o.clone()
                o[..., r] = 0.0
                return _rewrap(o, rest)
            return _hook

        handles.append(module.register_forward_hook(make_hook(row)))

    try:
        records = sweep(wrapper, probe)
    finally:
        for h in handles:
            h.remove()

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Hydra loop
# ──────────────────────────────────────────────────────────────────────────────

def run_hydra_test(
    wrapper,
    probe: str,
    max_rounds: int = 10,
    degradation_threshold: float = 5.0,
) -> dict:
    """
    Iteratively ablate super weights until perplexity degrades or
    max_rounds is exhausted.

    Returns the full result dict ready for JSON serialisation.
    """
    print("=" * 70)
    print("HYDRA TEST — Evo2")
    print(f"  max_rounds={max_rounds}, degradation_threshold={degradation_threshold}%")
    print("=" * 70)

    # ── Baseline perplexity ───────────────────────────────────────────────────
    baseline_ppl = wrapper.compute_perplexity(probe)
    print(f"\nBaseline PPL : {baseline_ppl:.6f}\n")

    ablated: list = []          # accumulated ablations
    seen_rows:    set = set()   # (layer, row) — dedup guard
    rounds_log:   list = []
    outcome = "max_rounds"

    for round_idx in range(max_rounds):
        print(f"─── Round {round_idx + 1} / {max_rounds} ───")

        # 1. Sweep with current ablations active
        records = sweep_with_ablations(wrapper, probe, ablated)

        # 2. Find the spike layer and extract coords
        spike_layer = find_spike_layer(records)
        layer_idx, row, col = extract_coords(records, spike_layer)

        in_max  = records[spike_layer].get("in_max",  0.0)
        out_max = records[spike_layer].get("out_max", 0.0)

        print(
            f"  Candidate super-row: layer={layer_idx}, row={row}  "
            f"(dominant col={col})  in_max={in_max:.2f}  out_max={out_max:.2f}"
        )

        # 3. Guard: same (layer, row) already ablated?
        row_key = (layer_idx, row)
        if row_key in seen_rows:
            print(
                f"  [STOP] Same row re-detected (layer={layer_idx}, row={row}) — "
                "no new candidate found.  Hydra may have no more heads in range."
            )
            outcome = "redetection"
            break
        seen_rows.add(row_key)

        ablated.append({
            "layer": layer_idx,
            "row":   row,
            "col":   col,   # recorded for information; not used in hooks
        })

        # 4. Measure perplexity with ALL accumulated superrow ablations
        ppl = wrapper.compute_perplexity_ablated(probe, ablated, mode="superrow")
        delta_pct = (ppl - baseline_ppl) / abs(baseline_ppl) * 100

        print(
            f"  PPL after {len(ablated)} ablation(s): {ppl:.6f}  "
            f"delta={delta_pct:+.4f}%"
        )

        rounds_log.append({
            "round":     round_idx + 1,
            "layer":     layer_idx,
            "row":       row,
            "col":       col,
            "in_max":    in_max,
            "out_max":   out_max,
            "ppl":       ppl,
            "delta_pct": delta_pct,
            "n_ablated": len(ablated),
        })

        # 6. Termination check
        if delta_pct >= degradation_threshold:
            print(
                f"\n[STOP] delta_pct={delta_pct:+.4f}% ≥ threshold {degradation_threshold}%"
                " — model degraded.  The hydra is dead."
            )
            outcome = "degraded"
            break

        print(
            f"  Model still robust (delta={delta_pct:+.4f}% < {degradation_threshold}%)."
            "  Continuing...\n"
        )

    else:
        print(
            f"\n[DONE] Hydra survived all {max_rounds} rounds of ablation "
            f"with delta_pct={rounds_log[-1]['delta_pct']:+.4f}% "
            "— strong hydra effect confirmed."
        )

    result = {
        "baseline_ppl":               baseline_ppl,
        "degradation_threshold_pct":  degradation_threshold,
        "max_rounds":                 max_rounds,
        "outcome":                    outcome,
        "total_ablated":              len(ablated),
        "rounds":                     rounds_log,
    }

    # Print summary table
    print("\n" + "=" * 70)
    print(f"OUTCOME: {outcome.upper()}")
    print(f"{'Round':<7} {'Layer':>5} {'Row':>5} {'Col':>5} "
          f"{'out_max':>12} {'PPL':>12} {'delta%':>10}")
    print("-" * 70)
    for r in rounds_log:
        print(
            f"{r['round']:<7} {r['layer']:>5} {r['row']:>5} {r['col']:>5} "
            f"{r['out_max']:>12.2f} {r['ppl']:>12.6f} {r['delta_pct']:>+10.4f}"
        )
    print("=" * 70)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hydra test: iteratively ablate Evo2 super weights and track perplexity."
    )
    parser.add_argument(
        "--model", default="evo2", choices=list(WRAPPER_MAP.keys()),
        help="Model key (default: evo2)",
    )
    parser.add_argument(
        "--probe", default="actb_full",
        help="Probe sequence key (default: actb_full)",
    )
    parser.add_argument(
        "--max_rounds", type=int, default=10,
        help="Maximum number of ablation rounds (default: 10)",
    )
    parser.add_argument(
        "--degradation_threshold", type=float, default=5.0,
        help="Stop when delta_pct from baseline reaches this %% (default: 5.0)",
    )
    parser.add_argument(
        "--out", default="results/hydra_test_evo2.json",
        help="Output JSON path (default: results/hydra_test_evo2.json)",
    )
    args = parser.parse_args()

    config = yaml.safe_load(open(f"configs/{args.model}.yaml"))

    WrapperClass = WRAPPER_MAP[args.model]
    wrapper = WrapperClass(config)
    wrapper.load()

    probe_seq = get_probe(args.probe)

    result = run_hydra_test(
        wrapper,
        probe=probe_seq,
        max_rounds=args.max_rounds,
        degradation_threshold=args.degradation_threshold,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
