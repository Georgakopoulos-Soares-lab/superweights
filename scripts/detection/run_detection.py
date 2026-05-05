# scripts/run_detection.py
"""
Usage:
  python scripts/run_detection.py --model generator
  python scripts/run_detection.py --model dnabert2 --probe poly_a
  python scripts/run_detection.py --model evo2 --threshold 0.05

  --mode superweight   (default) Zero a single scalar W[row,col] — original Yu et al.
  --mode superrow                Zero the entire output row W[row,:] — use if scalar fails.
"""
import argparse
import json
import yaml
from pathlib import Path

from probes.dna_probes import get_probe
from detection.iterative_finder import find_all_super_weights
from detection.sweep import sweep
from analysis.visualize_activations import plot_activation_profile
from models import WRAPPER_MAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True, choices=list(WRAPPER_MAP.keys()))
    parser.add_argument("--probe",     default="actb_500")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max_iter",  type=int,   default=10)
    parser.add_argument("--out",       default="results/super_weight_index.json")
    parser.add_argument(
        "--mode",
        default="superweight",
        choices=["superweight", "superrow"],
        help="superweight: zero single scalar (Yu et al. original). "
             "superrow: zero entire output row (use if scalar zeroing fails to suppress spike).",
    )
    args = parser.parse_args()

    use_row_zeroing = args.mode == "superrow"

    config = yaml.safe_load(open(f"configs/{args.model}.yaml"))

    WrapperClass = WRAPPER_MAP[args.model]
    wrapper = WrapperClass(config)
    wrapper.load()

    probe = get_probe(args.probe)

    # Plot baseline activation profile
    records = sweep(wrapper, probe)
    try:
        plot_activation_profile(
            records, args.model,
            save_path=f"results/{args.model}_activation_profile.png"
        )
    except (ImportError, Exception) as e:
        print(f"[warn] Skipping activation plot: {e}")

    print(f"[detection mode] {'superrow (zero full row)' if use_row_zeroing else 'superweight (zero scalar)'}")

    # Run iterative detection
    super_weights = find_all_super_weights(
        wrapper,
        probe=probe,
        suppression_threshold=args.threshold,
        max_iterations=args.max_iter,
        use_row_zeroing=use_row_zeroing,
    )

    # Append to index
    index = {}
    index_path = Path(args.out)
    if index_path.exists():
        index = json.loads(index_path.read_text())
    index[args.model] = {"mode": args.mode, "results": super_weights}
    index_path.write_text(json.dumps(index, indent=2))

    print(f"\nResults saved to {args.out}")
    print(json.dumps(super_weights, indent=2))


if __name__ == "__main__":
    main()
