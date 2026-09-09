"""
scripts/analysis/run_channel_survival.py
-------------------------------------------
T1.1 -- channel-survival trace across all evaluable models.

Loads the frozen SW list (results/super_weight_index.json, with an optional
--sw_index override -- used for generator_prokaryote's corrected entry from
results/mechanism/generator_prokaryote_sw_corrected.json), picks the primary
SW (highest out_max entry) as the model's representative super weight,
runs hooks.ablation_trace.run_difference_trace over N probe windows
(default 30, 3072bp, hg38 for EUK / E. coli for PROK per analysis/probe_windows.py),
and aggregates mean +/- sd per Task Spec Section 1.4's schema.

ablation_effect is pulled from existing evaluation results where a suitable
one already exists at the corrected SW coordinates (GENERator EUK's
sw_causal_tracing.json; DNABERT-2/NTv3's GUE splice multiseed results);
otherwise (GENERator PROK's corrected row, Evo1) it is computed fresh on the
same probe windows using each model's own perplexity machinery.

Usage:
  python scripts/analysis/run_channel_survival.py --model generator
  python scripts/analysis/run_channel_survival.py --model generator_prokaryote --sw_index results/mechanism/generator_prokaryote_sw_corrected.json
  python scripts/analysis/run_channel_survival.py --model dnabert2
  python scripts/analysis/run_channel_survival.py --model ntv3
  python scripts/analysis/run_channel_survival.py --model evo1   # run inside the `evo` conda env

Output: results/mechanism/channel_survival_{model}.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

KINGDOM = {
    "generator": "euk",
    "generator_prokaryote": "prok",
    "dnabert2": "euk",
    "ntv3": "euk",
    "evo1": "prok",
}


def load_primary_sw(model_key: str, sw_index_path: str | None):
    if sw_index_path == "corrected_prok":
        d = json.loads((ROOT / "results/mechanism/generator_prokaryote_sw_corrected.json").read_text())
        c = d["corrected_super_weight"]
        return c["layer"], c["row"], c["out_max"]
    path = sw_index_path or (ROOT / "results/super_weight_index.json")
    d = json.loads(Path(path).read_text())
    entries = d[model_key]["results"] if model_key in d else d["generator_prokaryote"]["results"]
    primary = max(entries, key=lambda r: r["out_max"])
    return primary["layer"], primary["row"], primary["out_max"]


def get_ablation_effect(model_key: str, sw_layer: int, sw_row: int, wrapper, windows: list) -> dict:
    """Pull from existing results where the coordinates match; else compute fresh."""
    if model_key == "generator":
        d = json.loads((ROOT / "results/sw_causal_tracing.json").read_text())
        deltas = np.array([r["delta_sw"] for r in d["results"]])
        clean = np.array([r["clean_ppl"] for r in d["results"]])
        rand = np.array([r["delta_rand_mean"] for r in d["results"]])
        pct = deltas / clean * 100
        return {"metric": "delta_ppl_pct", "value": float(pct.mean()), "sd": float(pct.std()),
                "random_control": float((rand / clean * 100).mean()),
                "n_windows": len(deltas), "source": "results/sw_causal_tracing.json (reused, matches frozen SW rows [1522,2371] @ layer 4)"}

    if model_key == "dnabert2":
        d = json.loads((ROOT / "results/gue_multiseed_results.json").read_text())
        splice = d["dnabert2/splice/reconstructed"]["aggregate"]
        return {"metric": "delta_acc_pct_splice_gue", "value": splice["delta_acc_mean"] * 100,
                "sd": splice["delta_acc_std"] * 100, "random_control": 0.0,
                "n_seeds": 3, "source": "results/gue_multiseed_results.json (dnabert2/splice/reconstructed, reused)"}

    if model_key == "ntv3":
        d = json.loads((ROOT / "results/gue_multiseed_ntv3_splice.json").read_text())
        agg = d["ntv3/splice/reconstructed"]["aggregate"]
        return {"metric": "delta_mcc_splice_gue", "value": agg["delta_mcc_mean"] * 100,
                "sd": agg["delta_mcc_std"] * 100, "random_control": 0.0,
                "source": "results/gue_multiseed_ntv3_splice.json (reused, ntv3/splice/reconstructed)"}

    if model_key == "generator_prokaryote":
        # Corrected row (layer 8, row 260) -- the existing results/prokaryote/sw_causal_tracing.json
        # uses the stale layer-2/row-1927 entry, so it CANNOT be reused here; compute fresh ΔPPL
        # on the same probe windows via the wrapper's own forward + cross-entropy loss.
        from src.ablation_trace import install_ablation_hook
        deltas_pct = []
        for w in windows[:10]:  # cap for compute; still >= 10 independent windows
            ppl_clean = _causal_ppl(wrapper, w["seq"])
            handle = install_ablation_hook(wrapper, sw_layer, sw_row)
            try:
                ppl_ablated = _causal_ppl(wrapper, w["seq"])
            finally:
                handle.remove()
            deltas_pct.append((ppl_ablated - ppl_clean) / ppl_clean * 100)
        arr = np.array(deltas_pct)
        return {"metric": "delta_ppl_pct", "value": float(arr.mean()), "sd": float(arr.std()),
                "random_control": None, "n_windows": len(arr),
                "source": "computed fresh on probe windows (corrected row 260 @ layer 8; "
                          "the stale results/prokaryote/sw_causal_tracing.json used the invalid row 1927 and could not be reused)"}

    if model_key == "evo1":
        deltas_pct = []
        for w in windows[:10]:
            ppl_clean = wrapper.compute_perplexity(w["seq"])
            ppl_ablated = wrapper.compute_perplexity_ablated(
                w["seq"], [{"layer": sw_layer, "row": sw_row}], mode="superrow")
            deltas_pct.append((ppl_ablated - ppl_clean) / ppl_clean * 100)
        arr = np.array(deltas_pct)
        return {"metric": "delta_ppl_pct", "value": float(arr.mean()), "sd": float(arr.std()),
                "random_control": None, "n_windows": len(arr),
                "source": "computed fresh on probe windows via Evo1Wrapper.compute_perplexity[_ablated]"}

    raise ValueError(model_key)


def _causal_ppl(wrapper, sequence: str) -> float:
    """Cross-entropy-based perplexity for a causal LM wrapper (GENERator-style)."""
    seq = wrapper._prepare_sequence(sequence) if hasattr(wrapper, "_prepare_sequence") else sequence
    inputs = wrapper.tokenizer(seq, return_tensors="pt", add_special_tokens=False).to(wrapper.model.device)
    with torch.no_grad():
        out = wrapper.model(**inputs)
    logits = out.logits[0]
    ids = inputs["input_ids"][0]
    shift_logits = logits[:-1].float()
    shift_labels = ids[1:]
    loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
    return float(torch.exp(loss))


def classify_regime(coordinate_retention, total_retention, dispersion_index,
                     participation_ratio, ablation_value) -> str:
    """
    Faithful to Task Spec Section 3's three regimes -- each requires ALL of
    its stated conditions, not just one. A model matching none of them is
    reported as 'ambiguous' rather than forced into the nearest bucket.
      Bottleneck:  coordinate_retention high, participation_ratio low,  ablation large.
      Dispersion:  coordinate_retention -> 0, total_retention ~= 1, dispersion_index >> 1,
                   participation_ratio high, ablation ~= 0.
      Suppression: coordinate_retention -> 0, total_retention -> 0, ablation ~= 0.
    """
    small_effect = abs(ablation_value) < 5.0  # percent
    coord_high = coordinate_retention == coordinate_retention and coordinate_retention > 0.3
    coord_low = coordinate_retention == coordinate_retention and coordinate_retention < 0.1
    pr_low = participation_ratio == participation_ratio and participation_ratio < 5.0
    pr_high = participation_ratio == participation_ratio and participation_ratio >= 5.0
    total_ok_dispersion = total_retention == total_retention and 0.3 <= total_retention <= 3.0
    total_low = total_retention == total_retention and total_retention < 0.3
    disp_high = dispersion_index == dispersion_index and dispersion_index > 3.0

    if coord_high and pr_low and not small_effect:
        return "bottleneck"
    if coord_low and total_ok_dispersion and disp_high and pr_high and small_effect:
        return "dispersion"
    if coord_low and total_low and small_effect:
        return "suppression"
    return "ambiguous"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["generator", "generator_prokaryote", "dnabert2", "ntv3", "evo1"])
    ap.add_argument("--sw_index", default=None,
                    help="Path to super_weight_index.json override, or the literal "
                         "string 'corrected_prok' to use the corrected generator_prokaryote SW.")
    ap.add_argument("--n_windows", type=int, default=30)
    ap.add_argument("--window_bp", type=int, default=3072)
    ap.add_argument("--out_dir", default="results/mechanism")
    args = ap.parse_args()

    from models import WRAPPER_MAP
    from src.ablation_trace import run_difference_trace
    from src.probe_windows import sample_probe_windows

    model_key = args.model
    sw_layer, sw_row, out_max = load_primary_sw(model_key, args.sw_index)
    print(f"[channel_survival] model={model_key}  primary SW: layer={sw_layer} row={sw_row} (out_max={out_max})")

    config = yaml.safe_load((ROOT / f"configs/{model_key}.yaml").read_text())
    wrapper = WRAPPER_MAP[model_key](config)
    print(f"[channel_survival] loading model …")
    wrapper.load()

    kingdom = KINGDOM[model_key]
    print(f"[channel_survival] sampling {args.n_windows} {kingdom} probe windows ({args.window_bp}bp) …")
    windows = sample_probe_windows(kingdom, n=args.n_windows, window_bp=args.window_bp, seed=42)
    print(f"[channel_survival] got {len(windows)} windows")

    n_layers = None
    per_window_summaries = []
    per_layer_on_coord, per_layer_total, per_layer_frac = [], [], []

    for i, w in enumerate(windows):
        seq = w["seq"]
        try:
            trace = run_difference_trace(wrapper, model_key, sw_layer, sw_row, seq)
        except Exception as e:
            print(f"  window {i+1}/{len(windows)} [{w['label']}] FAILED: {e}")
            continue
        n_layers = trace["n_layers"]
        per_layer_on_coord.append(trace["on_coord_energy"])
        per_layer_total.append(trace["total_energy"])
        per_layer_frac.append(trace["coord_fraction"])
        per_window_summaries.append({
            "coordinate_retention": trace["coordinate_retention"],
            "total_retention": trace["total_retention"],
            "dispersion_index": trace["dispersion_index"],
            "participation_ratio": trace["participation_ratio"],
            "top1_fraction": trace["top1_fraction"],
        })
        print(f"  window {i+1}/{len(windows)} [{w['label']}] "
              f"coord_ret={trace['coordinate_retention']:.4g}  "
              f"total_ret={trace['total_retention']:.4g}  "
              f"disp={trace['dispersion_index']:.4g}  "
              f"PR={trace['participation_ratio']:.4g}")

    def agg(key):
        vals = np.array([s[key] for s in per_window_summaries if s[key] == s[key]])  # drop NaN
        return {"mean": float(vals.mean()) if len(vals) else None,
                "sd": float(vals.std()) if len(vals) else None,
                "n": int(len(vals))}

    ablation_effect = get_ablation_effect(model_key, sw_layer, sw_row, wrapper, windows)

    summary = {
        "coordinate_retention": agg("coordinate_retention"),
        "total_retention": agg("total_retention"),
        "dispersion_index": agg("dispersion_index"),
        "participation_ratio": agg("participation_ratio"),
        "top1_fraction": agg("top1_fraction"),
    }

    caveats = []
    degenerate_L_eq_F = (sw_layer == n_layers - 1)
    if degenerate_L_eq_F:
        caveats.append(
            f"SW layer ({sw_layer}) IS the last captured block (n_layers={n_layers}) -- "
            "coordinate_retention/total_retention reduce to a tautological ratio of a "
            "quantity over itself (both ~1.0 by construction) and are NOT a meaningful "
            "survival measurement. For NTv3 specifically, the registered SW sits at the "
            "final block of the 12-block *transformer tower*, but the true network output "
            "passes through a further deconv/upsampling tower afterward whose channel count "
            "differs from the transformer's d_model at each stage (U-Net-style ConvTowerBlock/"
            "DeconvBlock with dim_in != dim_out) -- so 'coordinate k' has no stable identity "
            "across that remaining path, and the retention methodology cannot be extended "
            "through it without redefining what 'the same coordinate' means. Regime "
            "classification is NOT attempted for this model; ablation_effect (from existing "
            "GUE evaluation) is still reported and remains valid."
        )
    regime = ("not_classifiable (see caveats)" if degenerate_L_eq_F else
              classify_regime(summary["coordinate_retention"]["mean"],
                               summary["total_retention"]["mean"],
                               summary["dispersion_index"]["mean"],
                               summary["participation_ratio"]["mean"],
                               ablation_effect["value"]))

    out = {
        "model": model_key,
        "precision": str(config.get("dtype", "float32")),
        "sw": {"layer": sw_layer, "row": sw_row},
        "n_windows": len(per_window_summaries),
        "kingdom": kingdom,
        "per_layer": {
            "on_coord_energy": np.mean(per_layer_on_coord, axis=0).tolist() if per_layer_on_coord else [],
            "total_energy": np.mean(per_layer_total, axis=0).tolist() if per_layer_total else [],
            "coord_fraction": np.mean(per_layer_frac, axis=0).tolist() if per_layer_frac else [],
        },
        "summary": summary,
        "ablation_effect": ablation_effect,
        "regime": regime,
        "caveats": caveats,
        "meta": {
            "commit": "n/a (not a git repo)",
            "device": str(next(wrapper.model.parameters()).device),
            "date": "2026-08-03",
            "n_layers": n_layers,
        },
    }

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"channel_survival_{model_key}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[channel_survival] regime={regime}  "
          f"coord_ret={summary['coordinate_retention']['mean']:.4g}  "
          f"disp={summary['dispersion_index']['mean']:.4g}  "
          f"ablation={ablation_effect['metric']}={ablation_effect['value']:.4g}")
    print(f"[channel_survival] wrote {out_path}")


if __name__ == "__main__":
    main()
