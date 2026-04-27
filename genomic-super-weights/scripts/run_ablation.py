# scripts/run_ablation.py
"""
Usage:
  python scripts/run_ablation.py --model generator
  python scripts/run_ablation.py --model ntv3 --probe mixed_gc
"""
import argparse
import json
import yaml
from pathlib import Path

from probes.dna_probes import get_probe
from analysis.ablation import run_destruction_test
from models import WRAPPER_MAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(WRAPPER_MAP.keys()))
    parser.add_argument("--probe", default="actb_full")
    parser.add_argument("--sw_index", default="results/super_weight_index.json")
    parser.add_argument("--out",      default="results/ablation_results.json")
    args = parser.parse_args()

    config = yaml.safe_load(open(f"configs/{args.model}.yaml"))
    sw_index = json.loads(Path(args.sw_index).read_text())

    if args.model not in sw_index:
        print(f"No super weights found for {args.model} in index. Run run_detection.py first.")
        return

    # Support both old format (bare list) and new format ({"mode": ..., "results": [...]})
    entry = sw_index[args.model]
    sw_list = entry["results"] if isinstance(entry, dict) else entry

    if not sw_list:
        print(f"Empty super weight list for {args.model}.")
        return

    WrapperClass = WRAPPER_MAP[args.model]
    wrapper = WrapperClass(config)
    wrapper.load()

    detection_mode = entry.get("mode", "superrow") if isinstance(entry, dict) else "superrow"
    result = run_destruction_test(wrapper, sw_list, get_probe(args.probe), detection_mode=detection_mode)

    mode_label = "super weight" if detection_mode == "superweight" else "super row"
    print(f"\n{'Condition':<26} | {'Perplexity/Entropy':>18} | {'Delta':>10}")
    print("-" * 62)
    print(f"{'Original':<26} | {result['baseline']:>18.4f} | {'—':>10}")
    print(f"{f'Prune {mode_label}':<26} | {result['pruned']:>18.4f} | {result['delta_pct']:>+9.1f}%")
    print(f"{'Prune random (mean)':<26} | {result['pruned_rand_mean']:>18.4f} | {result['delta_rand_pct']:>+9.1f}%")

    ablation = {}
    out_path = Path(args.out)
    if out_path.exists():
        ablation = json.loads(out_path.read_text())
    ablation[args.model] = result
    out_path.write_text(json.dumps(ablation, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
