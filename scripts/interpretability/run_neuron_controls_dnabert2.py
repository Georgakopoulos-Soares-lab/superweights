# scripts/interpretability/run_neuron_controls_dnabert2.py
"""
Control comparisons for the single-neuron pilot (task §4 in the pilot spec):

  - random neurons matched by (layer, activation magnitude)
  - random neurons matched by outgoing weight norm ||Wd[:,i]||
  - top-activation-only candidate (ignores gradient)
  - top-gradient-only candidate (ignores activation)
  - GC-matched and motif-containing (GT/AG-decoy) hard-negative subsets, sliced from the
    already-cached per-example raw predictions (no new forward passes)
  - task-specificity: same neuron coordinate, zero-ablated, evaluated on the unrelated
    EMP/H3K4me3 task (fine-tuned checkpoint fine-tuned separately) instead of splice

Requires: results/neuron_pilot_dnabert2_discovery.csv (full ranking),
          results/neuron_pilot_dnabert2_intervention.json (+ its raw/*.npz caches),
          the fine-tuned splice checkpoint, and (for the task-specificity control) the
          fine-tuned EMP/H3K4me3 checkpoint.

Usage:
    python scripts/interpretability/run_neuron_controls_dnabert2.py \
        --gue_root /work/11034/atzanakak/GUE/GUE --n_rand 10
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from neuron_pilot_common import (  # noqa: E402
    NeuronIntervention,
    auprc_numpy,
    auroc_numpy,
    flip_rates,
    get_wo_module,
)
from run_dnabert2_uk_audit import extract_mlp_weights  # noqa: E402
from run_gue_ablation import GUEDataset, _accuracy, _mcc  # noqa: E402
from run_neuron_causal_intervention_dnabert2 import (  # noqa: E402
    JunctionGUEDataset, evaluate_full, load_model as load_splice_model,
    MODEL_ID, REVISION, MAX_LENGTH,
)

D_FFN = 3072
DEFAULT_CKPT = ROOT / "results" / "gue_checkpoints" / "dnabert2_reconstructed" / "model_state.pt"
DEFAULT_H3K4ME3_CKPT = ROOT / "results" / "gue_checkpoints" / "dnabert2_H3K4me3" / "model_state.pt"


def load_discovery_full(csv_path: Path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def gc_fraction(seq: str) -> float:
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq) if seq else 0.0


def has_decoy_motif(seq: str, window=(190, 210)) -> bool:
    frag = seq[window[0]:window[1]].upper()
    return ("GT" in frag) or ("AG" in frag)


def matched_random_controls(pool_layer_records, exclude_indices, key, target_value, n, rng,
                             tolerance_frac=0.25):
    """Pick n neuron indices from pool_layer_records (list of dicts with 'neuron' and `key`)
    whose `key` value is within tolerance_frac of target_value, excluding exclude_indices."""
    target_abs = abs(target_value)
    lo, hi = target_abs * (1 - tolerance_frac), target_abs * (1 + tolerance_frac)
    candidates = [r for r in pool_layer_records
                  if int(r["neuron"]) not in exclude_indices and lo <= abs(float(r[key])) <= hi]
    if len(candidates) < n:
        # widen tolerance progressively rather than silently returning fewer than requested
        widened = tolerance_frac
        while len(candidates) < n and widened < 4.0:
            widened *= 2
            lo, hi = target_abs * (1 - widened), target_abs * (1 + widened)
            candidates = [r for r in pool_layer_records
                          if int(r["neuron"]) not in exclude_indices and lo <= abs(float(r[key])) <= hi]
    rng.shuffle(candidates)
    return [int(r["neuron"]) for r in candidates[:n]]


def run_intervention_condition(model, ds, device, layer, neuron, target, alpha, scope, baseline_preds_bin):
    spec = {"layer": layer, "neuron_idx": neuron, "target": target, "alpha": alpha, "scope": scope}
    metrics, raw = evaluate_full(model, ds, device, 64, intervention_spec=spec)
    flips = flip_rates(baseline_preds_bin, raw["preds_bin"])
    return metrics, raw, flips


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gue_root", required=True)
    p.add_argument("--n_rand", type=int, default=10)
    p.add_argument("--half_window", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--h3k4me3_ckpt", default=str(DEFAULT_H3K4ME3_CKPT))
    p.add_argument("--discovery_csv",
                    default=str(ROOT / "results" / "neuron_pilot_dnabert2_discovery.csv"))
    p.add_argument("--intervention_json",
                    default=str(ROOT / "results" / "neuron_pilot_dnabert2_intervention.json"))
    p.add_argument("--raw_dir", default=str(ROOT / "results" / "neuron_pilot_dnabert2_raw"))
    p.add_argument("--out", default=str(ROOT / "results" / "neuron_pilot_dnabert2_controls.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    t0 = time.time()
    rng = random.Random(args.seed)

    interv = json.loads(Path(args.intervention_json).read_text())
    strongest = interv["strongest_candidate"]
    layer, neuron = strongest["layer"], strongest["neuron"]
    shortlist_neurons = {c["neuron"] for c in interv["shortlist_candidates"] if c["layer"] == layer}
    full_ranking = load_discovery_full(Path(args.discovery_csv))
    layer_records = [r for r in full_ranking if int(r["layer"]) == layer]
    cand_record = next(r for r in layer_records if int(r["neuron"]) == neuron)

    model, tok = load_splice_model(Path(args.ckpt), args.device)
    test_csv = Path(args.gue_root) / "splice" / "reconstructed" / "test.csv"
    ds_junction = JunctionGUEDataset(test_csv, tok, MAX_LENGTH, half_window=args.half_window)

    baseline_npz = np.load(Path(args.raw_dir) / "baseline.npz")
    baseline_preds_bin = baseline_npz["preds_bin"]
    baseline_labels_bin = baseline_npz["labels_bin"]
    baseline_ppos = baseline_npz["ppos"]

    strongest_natural = interv["shortlist_candidates"][0]["conditions"]["neg_mean_suppress__all"]
    strongest_delta_mcc = strongest_natural["delta_mcc_3class"]
    neg_mean_target = interv["shortlist_candidates"][0]["neg_mean_allpos"]

    results = {"config": {"seed": args.seed, "n_rand": args.n_rand, "layer": layer, "neuron": neuron}}

    # ── 1. Random neurons matched by (layer, activation magnitude) ──────────
    act_matched_idx = matched_random_controls(
        layer_records, shortlist_neurons | {neuron}, "act_diff_junction",
        float(cand_record["act_diff_junction"]), args.n_rand, rng)
    act_matched_results = []
    for ni in act_matched_idx:
        rec = next(r for r in layer_records if int(r["neuron"]) == ni)
        target = float(rec["mean_act_neg_allpos"])
        metrics, _, flips = run_intervention_condition(
            model, ds_junction, args.device, layer, ni, target, 1.0, "all", baseline_preds_bin)
        act_matched_results.append({"neuron": ni, "delta_mcc_3class":
                                     metrics["mcc_3class"] - interv["baseline"]["mcc_3class"],
                                     "total_flip_rate": flips["total_flip_rate"]})
    results["random_matched_by_activation"] = {
        "neurons": act_matched_idx, "results": act_matched_results,
        "mean_delta_mcc": float(np.mean([r["delta_mcc_3class"] for r in act_matched_results])) if act_matched_results else None,
        "candidate_delta_mcc": strongest_delta_mcc,
    }
    print(f"[controls] activation-matched random (n={len(act_matched_idx)}): "
          f"mean Δmcc={results['random_matched_by_activation']['mean_delta_mcc']:+.4f} "
          f"vs candidate Δmcc={strongest_delta_mcc:+.4f}")

    # ── 2. Random neurons matched by outgoing weight norm ‖Wd[:,i]‖ ─────────
    Wg, Wu, Wd = extract_mlp_weights(model.bert, layer)
    wd_norms = np.linalg.norm(Wd, axis=0)  # (d_ffn,)
    cand_norm = float(wd_norms[neuron])
    lo, hi = cand_norm * 0.75, cand_norm * 1.25
    norm_pool = [i for i in range(D_FFN) if i not in shortlist_neurons and i != neuron
                and lo <= wd_norms[i] <= hi]
    widen = 0.25
    while len(norm_pool) < args.n_rand and widen < 4.0:
        widen *= 2
        lo, hi = cand_norm * (1 - widen), cand_norm * (1 + widen)
        norm_pool = [i for i in range(D_FFN) if i not in shortlist_neurons and i != neuron
                    and lo <= wd_norms[i] <= hi]
    rng.shuffle(norm_pool)
    norm_matched_idx = norm_pool[:args.n_rand]
    norm_matched_results = []
    for ni in norm_matched_idx:
        rec = next((r for r in layer_records if int(r["neuron"]) == ni), None)
        target = float(rec["mean_act_neg_allpos"]) if rec else 0.0
        metrics, _, flips = run_intervention_condition(
            model, ds_junction, args.device, layer, ni, target, 1.0, "all", baseline_preds_bin)
        norm_matched_results.append({"neuron": ni, "wd_norm": float(wd_norms[ni]),
                                      "delta_mcc_3class": metrics["mcc_3class"] - interv["baseline"]["mcc_3class"],
                                      "total_flip_rate": flips["total_flip_rate"]})
    results["random_matched_by_outgoing_norm"] = {
        "neurons": norm_matched_idx, "candidate_norm": cand_norm, "results": norm_matched_results,
        "mean_delta_mcc": float(np.mean([r["delta_mcc_3class"] for r in norm_matched_results])) if norm_matched_results else None,
        "candidate_delta_mcc": strongest_delta_mcc,
    }
    print(f"[controls] outgoing-norm-matched random (n={len(norm_matched_idx)}): "
          f"mean Δmcc={results['random_matched_by_outgoing_norm']['mean_delta_mcc']:+.4f}")

    # ── 3. Top-activation-only and top-gradient-only candidates ─────────────
    act_only_top = max(full_ranking, key=lambda r: abs(float(r["act_diff_junction"])))
    grad_only_top = max(full_ranking, key=lambda r: float(r["grad_abs_junction"]))
    for name, rec in [("top_activation_only", act_only_top), ("top_gradient_only", grad_only_top)]:
        li, ni = int(rec["layer"]), int(rec["neuron"])
        target = float(rec["mean_act_neg_allpos"])
        metrics, _, flips = run_intervention_condition(
            model, ds_junction, args.device, li, ni, target, 1.0, "all", baseline_preds_bin)
        results[name] = {
            "layer": li, "neuron": ni,
            "delta_mcc_3class": metrics["mcc_3class"] - interv["baseline"]["mcc_3class"],
            "total_flip_rate": flips["total_flip_rate"],
        }
        print(f"[controls] {name}: layer={li} neuron={ni} "
              f"Δmcc={results[name]['delta_mcc_3class']:+.4f}")

    # ── 4. GC-matched / motif-decoy hard negatives (sliced from cached raw preds) ──
    with open(test_csv) as f:
        test_rows = list(csv.reader(f))[1:]
    test_seqs = [r[0] for r in test_rows]
    gc_vals = np.array([gc_fraction(s) for s in test_seqs])
    decoy_flags = np.array([has_decoy_motif(s) for s in test_seqs])

    cand_npz_path = Path(args.raw_dir) / f"cand_L{layer}_N{neuron}_neg_mean_all.npz"
    if cand_npz_path.exists():
        cand_npz = np.load(cand_npz_path)
        cand_preds_bin = cand_npz["preds_bin"]

        neg_mask = baseline_labels_bin == 0
        gc_median = float(np.median(gc_vals[neg_mask]))
        pos_mean_gc = float(gc_vals[baseline_labels_bin == 1].mean())
        gc_matched_mask = neg_mask & (np.abs(gc_vals - pos_mean_gc) < 0.05)
        motif_decoy_mask = neg_mask & decoy_flags

        def subset_metrics(mask):
            if mask.sum() == 0:
                return {"n": 0}
            return {
                "n": int(mask.sum()),
                "baseline_acc": float((baseline_preds_bin[mask] == baseline_labels_bin[mask]).mean()),
                "intervened_acc": float((cand_preds_bin[mask] == baseline_labels_bin[mask]).mean()),
                "neg_to_pos_flip_rate": float((baseline_preds_bin[mask] != cand_preds_bin[mask]).mean()),
            }

        results["hard_negative_controls"] = {
            "gc_median_negatives": gc_median,
            "positive_class_mean_gc": pos_mean_gc,
            "gc_matched_negatives": subset_metrics(gc_matched_mask),
            "motif_decoy_negatives_GTorAG_in_window": subset_metrics(motif_decoy_mask),
            "all_negatives": subset_metrics(neg_mask),
        }
        print(f"[controls] GC-matched negatives (n={gc_matched_mask.sum()}): "
              f"{results['hard_negative_controls']['gc_matched_negatives']}")
        print(f"[controls] motif-decoy negatives (n={motif_decoy_mask.sum()}): "
              f"{results['hard_negative_controls']['motif_decoy_negatives_GTorAG_in_window']}")
    else:
        results["hard_negative_controls"] = None
        print(f"[controls][WARN] {cand_npz_path} not found — run the intervention script first "
              f"to cache raw per-example predictions for the strongest candidate.")

    # ── 5. Task specificity: same neuron on EMP/H3K4me3 ──────────────────────
    h3k_ckpt = Path(args.h3k4me3_ckpt)
    if h3k_ckpt.exists():
        del model
        torch.cuda.empty_cache()
        import transformers
        h3k_csv_dir = Path(args.gue_root) / "EMP" / "H3K4me3"
        with open(h3k_csv_dir / "train.csv") as f:
            h3k_labels = [int(r[1]) for r in list(csv.reader(f))[1:]]
        num_labels_h3k = len(set(h3k_labels))

        tok_h3k = transformers.AutoTokenizer.from_pretrained(
            MODEL_ID, trust_remote_code=True, revision=REVISION, model_max_length=128)
        model_h3k = transformers.AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=num_labels_h3k, trust_remote_code=True, revision=REVISION)
        state = torch.load(h3k_ckpt, map_location="cpu")
        model_h3k.load_state_dict(state, strict=False)
        model_h3k.to(args.device)
        model_h3k.eval()

        h3k_test_ds = GUEDataset(str(h3k_csv_dir / "test.csv"), tok_h3k, 128)

        def eval_h3k(intervention_spec=None):
            loader = torch.utils.data.DataLoader(h3k_test_ds, batch_size=64, shuffle=False,
                                                  collate_fn=lambda b: {k: torch.stack([x[k] for x in b]) for k in b[0]})
            preds, labels = [], []
            hook_handle = None
            if intervention_spec is not None:
                wo = get_wo_module(model_h3k.bert, intervention_spec["layer"])
                iv = NeuronIntervention(neuron_idx=intervention_spec["neuron_idx"], target=0.0, alpha=1.0)
                hook_handle = wo.register_forward_pre_hook(iv)
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(args.device) for k, v in batch.items()}
                    logits = model_h3k(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"]).logits
                    preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                    labels.extend(batch["labels"].cpu().numpy().tolist())
            if hook_handle is not None:
                hook_handle.remove()
            preds, labels = np.array(preds), np.array(labels)
            return {"accuracy": _accuracy(labels, preds), "mcc": _mcc(labels, preds)}

        h3k_baseline = eval_h3k(None)
        h3k_zero_ablated = eval_h3k({"layer": layer, "neuron_idx": neuron})
        results["task_specificity_H3K4me3"] = {
            "baseline": h3k_baseline, "zero_ablated_same_neuron": h3k_zero_ablated,
            "delta_acc": h3k_zero_ablated["accuracy"] - h3k_baseline["accuracy"],
            "delta_mcc": h3k_zero_ablated["mcc"] - h3k_baseline["mcc"],
            "splice_delta_mcc_for_comparison": strongest_delta_mcc,
        }
        print(f"[controls] H3K4me3 task-specificity: baseline mcc={h3k_baseline['mcc']:.4f} "
              f"zero-ablated mcc={h3k_zero_ablated['mcc']:.4f} "
              f"(Δmcc={results['task_specificity_H3K4me3']['delta_mcc']:+.4f}, "
              f"vs splice Δmcc={strongest_delta_mcc:+.4f})")
    else:
        results["task_specificity_H3K4me3"] = None
        print(f"[controls][WARN] H3K4me3 checkpoint not found at {h3k_ckpt} — "
              f"run run_gue_ablation.py --model dnabert2 --task EMP/H3K4me3 first. Skipping.")

    results["config"]["elapsed_sec"] = time.time() - t0
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"[controls] wrote {args.out}")
    print(f"[controls] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
