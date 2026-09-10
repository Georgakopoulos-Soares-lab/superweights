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

from src.dna_probes import get_probe
from src.iterative_finder import find_all_super_weights
from src.sweep import sweep
from src.visualize_activations import plot_activation_profile
from models import WRAPPER_MAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True, choices=list(WRAPPER_MAP.keys()))
    parser.add_argument("--probe",     default="actb_500")
    parser.add_argument(
        "--pad_to_multiple", type=int, default=0,
        help="Right-pad the probe with 'A' until its TOKEN length is a multiple of this. "
             "REQUIRED for NTv3: it is a conv/deconv U-Net with 8 stride-2 conv blocks "
             "(filter_list has 8 entries), so the deconv tower's skip connections only "
             "align when the token length is a multiple of 2**8 = 256. NTv3 tokenises at "
             "nucleotide level (vocab 11), so the canonical 504 bp probes give 504 tokens "
             "and 504 %% 256 = 248 -> 'The size of tensor a (6) must match tensor b (7)' "
             "inside modeling_ntv3_pretrained.py. Use --pad_to_multiple 256.")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max_iter",  type=int,   default=10)
    parser.add_argument("--out",       default="results/negative_results/super_weight_index.json")
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
    if args.pad_to_multiple > 0:
        n0 = len(wrapper.tokenizer(probe)["input_ids"])
        m = args.pad_to_multiple
        target = ((n0 + m - 1) // m) * m
        # nucleotide-level tokenisers map 1 bp -> 1 token, so pad in bp and re-check
        while len(wrapper.tokenizer(probe)["input_ids"]) < target:
            probe = probe + "A"
        n1 = len(wrapper.tokenizer(probe)["input_ids"])
        print(f"[probe] padded {n0} -> {n1} tokens (multiple of {m}); "
              f"{len(probe)} bp")

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
