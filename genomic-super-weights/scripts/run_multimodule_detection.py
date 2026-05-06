# scripts/run_multimodule_detection.py
"""
Multi-module super-weight robustness check.

For every linear module type defined in a model's config (down_proj PLUS all
entries in extra_patterns), this script:

  1. Temporarily swaps the wrapper's target module to that pattern.
  2. Runs the same activation-spike sweep used by run_detection.py.
  3. Runs the same three-condition PPL/entropy ablation used by run_ablation.py:
       a. Baseline perplexity/entropy
       b. Prune the top-L1-norm row in the spike layer (single-row zeroing)
       c. Random-row control (mean over n_rand repeats, matched count)
  4. Reports Δ% for each module type.
  5. Saves results to results/multimodule_detection.json.

The goal: confirm whether the "no SW effect" finding for SSM/hybrid models is
robust across ALL module types, not just the down-projection.

Usage:
    python scripts/run_multimodule_detection.py --model evo2
    python scripts/run_multimodule_detection.py --model hybridna --probe actb_500
    python scripts/run_multimodule_detection.py --model dnabert2 --n_rand 5

    # dry-run: print which patterns will be scanned without running
    python scripts/run_multimodule_detection.py --model generator --list_patterns

Output JSON key: "<model>"
Each entry: dict keyed by pattern name with fields:
  pattern, num_layers_scanned,
  baseline, pruned_top_row, delta_pct, rand_mean, delta_rand_pct,
  spike_layer, spike_row, spike_out_max
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Repo root (one above scripts/) ───────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_INNER_ROOT  = _SCRIPTS_DIR.parent   # configs/, models/, detection/, …
_OUTER_ROOT  = _INNER_ROOT.parent    # results/, logs/, …

sys.path.insert(0, str(_INNER_ROOT))
sys.path.insert(0, str(_OUTER_ROOT))   # analysis/, probes/ live here

from detection.sweep import sweep
from detection.identify_spikes import find_spike_layer
from probes.dna_probes import get_probe
from models import WRAPPER_MAP
from analysis.ablation import causal_perplexity, masked_token_entropy


# ── Module resolution ─────────────────────────────────────────────────────────

def _resolve(model, pattern: str, layer_idx: int):
    """Resolve a dotted pattern like 'model.layers.{i}.mlp.down_proj'."""
    path = pattern.replace("{i}", str(layer_idx))
    obj  = model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def _pattern_exists(model, pattern: str, layer_idx: int) -> bool:
    try:
        _resolve(model, pattern, layer_idx)
        return True
    except AttributeError:
        return False


def _eval_with_row_zeroed(mod, row: int, metric_fn, probe: str) -> float:
    """
    Evaluate metric_fn with output row `row` of `mod` zeroed via a forward hook.
    Avoids any in-place weight modification — works with TE inference tensors.
    Zeroing output row i is equivalent to zeroing weight row i (W[i,:] @ x = 0).
    """
    def _hook(module, input, output):
        if isinstance(output, tuple):
            o = output[0].clone()
            o[..., row] = 0.0
            return (o,) + output[1:]
        o = output.clone()
        o[..., row] = 0.0
        return o

    handle = mod.register_forward_hook(_hook)
    try:
        result = metric_fn(probe)
    finally:
        handle.remove()
    return result


# ── Single-pattern ablation ───────────────────────────────────────────────────

def _ablate_pattern(
    wrapper,
    pattern: str,
    layer_indices: list[int],
    probe: str,
    n_rand: int,
    rng: random.Random,
    metric_fn,
) -> dict:
    """
    For a single module pattern:
      1. Sweep activations (still uses the wrapper's original target for hooks).
      2. Find the spike layer (restricted to layer_indices if given).
      3. Zero the top-L1-norm row of THAT pattern at THAT layer → eval.
      4. Random controls.

    The wrapper's forward hooks capture activations at the configured
    down_proj_pattern; we reuse that spike signal as the layer selector,
    then apply the zeroing to the pattern under test.  For patterns that
    ARE the down_proj (name == 'down_proj'), the spike and zeroing target
    are the same.  For other patterns (e.g. attn_out), we use the spike
    layer from the down_proj sweep as an anchor — then find the top-L1
    row in the test pattern at that same layer.
    """
    model = wrapper.model

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline = metric_fn(probe)

    # ── Find candidate layer via activation sweep ─────────────────────────────
    records   = sweep(wrapper, probe)
    spike_layer = find_spike_layer(records)

    # Restrict to layer_indices if the pattern only exists at a subset
    if layer_indices:
        # If spike_layer not in subset, pick the subset member with highest out_max
        if spike_layer not in layer_indices:
            spike_layer = max(layer_indices,
                              key=lambda li: records.get(li, {}).get("out_max", 0.0))

    spike_out_max = records.get(spike_layer, {}).get("out_max", float("nan"))

    # ── Identify top-L1 row in the target pattern at spike_layer ─────────────
    try:
        mod = _resolve(model, pattern, spike_layer)
    except AttributeError:
        return {
            "error": f"Pattern '{pattern}' not found at layer {spike_layer}",
            "pattern": pattern,
        }

    w       = mod.weight.data            # (out_features, in_features)
    l1      = w.abs().sum(dim=1)
    top_row = int(l1.argmax().item())

    # ── Prune top row → eval (hook-based: works with TE inference tensors) ────
    pruned = _eval_with_row_zeroed(mod, top_row, metric_fn, probe)

    delta_pct = (pruned - baseline) / max(abs(baseline), 1e-9) * 100

    # ── Random row controls ───────────────────────────────────────────────────
    # Sample random (layer, row) pairs from all valid positions in this pattern,
    # excluding the identified top row.
    all_coords: list[tuple[int, int]] = []
    for li in layer_indices:
        try:
            m     = _resolve(model, pattern, li)
            nrows = m.weight.data.shape[0]
            for ri in range(nrows):
                if not (li == spike_layer and ri == top_row):
                    all_coords.append((li, ri))
        except AttributeError:
            pass

    rand_vals = []
    if len(all_coords) >= 1:
        for _ in range(n_rand):
            li, ri = rng.choice(all_coords)
            mod_r  = _resolve(model, pattern, li)
            rv = _eval_with_row_zeroed(mod_r, ri, metric_fn, probe)
            rand_vals.append(rv)
    else:
        rand_vals = [baseline] * n_rand

    rand_mean     = float(np.mean(rand_vals))
    delta_rand_pct = (rand_mean - baseline) / max(abs(baseline), 1e-9) * 100

    print(
        f"    baseline={baseline:.4f}  pruned={pruned:.4f}  "
        f"SW_Δ%={delta_pct:+.2f}  rand_Δ%={delta_rand_pct:+.3f}"
        f"  spike_layer={spike_layer}  top_row={top_row}"
    )

    return {
        "pattern":            pattern,
        "num_layers_scanned": len(layer_indices),
        "baseline":           float(baseline),
        "pruned_top_row":     float(pruned),
        "delta_pct":          float(delta_pct),
        "rand_mean":          float(rand_mean),
        "delta_rand_pct":     float(delta_rand_pct),
        "spike_layer":        int(spike_layer),
        "spike_row":          int(top_row),
        "spike_out_max":      float(spike_out_max),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-module SW robustness check: scan all linear projections."
    )
    parser.add_argument("--model",  required=True, choices=list(WRAPPER_MAP.keys()))
    parser.add_argument("--probe",  default="actb_full",
                        help="Probe sequence key from probes/dna_probes.py")
    parser.add_argument("--n_rand", type=int, default=10,
                        help="Number of random-row control repeats per pattern.")
    parser.add_argument("--out",    default=None,
                        help="Output JSON (default: <outer_root>/results/multimodule_detection.json)")
    parser.add_argument("--list_patterns", action="store_true",
                        help="Print which patterns will be scanned and exit (dry-run).")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else _OUTER_ROOT / "results" / "multimodule_detection.json"

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = _INNER_ROOT / "configs" / f"{args.model}.yaml"
    config      = yaml.safe_load(config_path.read_text())
    num_layers  = config["num_layers"]

    # ── Build pattern list ────────────────────────────────────────────────────
    # Pattern spec: {"name": str, "pattern": str, "layer_indices": list[int] or None}
    all_layers = list(range(num_layers)) if num_layers else []

    patterns: list[dict] = [
        {
            "name":          "down_proj",
            "pattern":       config["down_proj_pattern"],
            "layer_indices": all_layers,
        }
    ]
    for ep in config.get("extra_patterns", []):
        li = ep.get("layer_indices")
        if li is None:
            li = all_layers
        patterns.append({
            "name":          ep["name"],
            "pattern":       ep["pattern"],
            "layer_indices": li,
        })

    if args.list_patterns:
        print(f"\nPatterns for {args.model}:")
        for p in patterns:
            print(f"  [{p['name']:12s}]  layers={len(p['layer_indices'])}  "
                  f"pattern={p['pattern']}")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.model} …")
    WrapperClass = WRAPPER_MAP[args.model]
    wrapper      = WrapperClass(config)
    wrapper.load()

    # Resolve num_layers if it was None (e.g. MegaDNA resolves at load time)
    if not all_layers:
        num_layers_actual = wrapper.num_layers
        all_layers = list(range(num_layers_actual))
        for p in patterns:
            if not p["layer_indices"]:
                p["layer_indices"] = all_layers

    probe = get_probe(args.probe)
    print(f"Probe: {args.probe}  ({len(probe)} chars)\n")

    # ── Build metric function (mirrors run_destruction_test logic) ────────────
    if hasattr(wrapper, "compute_perplexity"):
        _native = wrapper.compute_perplexity
        metric_fn = lambda seq: _native(seq)
    else:
        causal = config.get("causal", False)
        if causal:
            metric_fn = lambda seq: causal_perplexity(wrapper, seq)
        else:
            metric_fn = lambda seq: masked_token_entropy(wrapper, seq)

    rng     = random.Random(42)
    results = {}

    for spec in patterns:
        name    = spec["name"]
        pattern = spec["pattern"]
        layers  = spec["layer_indices"]

        # Verify pattern resolves on the actual model (skip if not present)
        first_li = layers[0] if layers else 0
        if not _pattern_exists(wrapper.model, pattern, first_li):
            print(f"  [{name}]  SKIP — pattern '{pattern}' not found at layer {first_li}")
            results[name] = {"error": f"pattern not found: {pattern}", "pattern": pattern}
            continue

        print(f"  [{name}]  {pattern}  ({len(layers)} layers) …")
        results[name] = _ablate_pattern(wrapper, pattern, layers, probe, args.n_rand, rng, metric_fn)
        results[name]["name"] = name

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  Multi-module SW scan — {args.model}")
    print(f"{'='*76}")
    header = f"  {'Module':<14}  {'base':>8}  {'pruned':>8}  {'SW_Δ%':>9}  {'rand_Δ%':>9}  {'spike_L':>7}"
    print(header)
    print("-" * 76)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name:<14}  ERROR: {r['error']}")
            continue
        print(
            f"  {name:<14}  {r['baseline']:>8.4f}  {r['pruned_top_row']:>8.4f}  "
            f"{r['delta_pct']:>+9.2f}  {r['delta_rand_pct']:>+9.3f}  "
            f"{r['spike_layer']:>7d}"
        )
    print(f"{'='*76}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    existing[args.model] = {
        "probe":    args.probe,
        "n_rand":   args.n_rand,
        "patterns": results,
    }
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
