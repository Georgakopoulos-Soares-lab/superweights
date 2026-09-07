# scripts/interpretability/run_neuron_causal_intervention_dnabert2.py
"""
Causal intervention on shortlisted candidate neurons from run_neuron_discovery_dnabert2.py.

For each shortlisted candidate (layer, neuron), rewrites h[:, positions, neuron] toward a
target value (0 = zero-ablation, negative-class mean = suppression, positive-class mean =
amplification — see neuron_pilot_common.NeuronIntervention) at either all token positions or
only the junction window, and measures the effect on the held-out GUE splice/reconstructed
test set: 3-class accuracy/MCC (directly comparable to the documented ensemble numbers),
binary (splice-vs-not) accuracy/MCC/AUROC/AUPRC, mean log-odds shift, and positive<->negative
prediction flip rates.

Also reruns, through the *same* evaluation path, the existing full 10-super-row ensemble
ablation and the single best row (L3 r603) ablation — both as an apples-to-apples comparison
point ("fraction of the ensemble effect reproduced by one neuron") and as an internal
consistency check against the documented -25.5%/-1.45% numbers.

Requires the fine-tuned splice classifier checkpoint and
results/neuron_pilot_dnabert2_discovery.json (from run_neuron_discovery_dnabert2.py).

Usage:
    python scripts/interpretability/run_neuron_causal_intervention_dnabert2.py \
        --gue_root /work/11034/atzanakak/GUE/GUE --top_k 25 --sweep_top_n 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from neuron_pilot_common import (  # noqa: E402
    NeuronIntervention,
    auprc_numpy,
    auroc_numpy,
    flat_batch_offsets,
    flip_rates,
    get_wo_module,
    junction_window,
    splice_binary_logodds,
)
from run_gue_ablation import (  # noqa: E402
    GUEDataset,
    _accuracy,
    _f1_macro,
    _mcc,
    _save_row,
    _zero_row,
    _restore_row,
    evaluate as evaluate_baseline_pipeline,
)

MODEL_ID = "zhihan1996/DNABERT-2-117M"
REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
MAX_LENGTH = 80
DEFAULT_CKPT = ROOT / "results" / "gue_checkpoints" / "dnabert2_reconstructed" / "model_state.pt"
DOWN_PROJ_PATTERN = "bert.encoder.layer.{i}.mlp.wo"
NUM_LAYERS = 12
ALPHA_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


class JunctionGUEDataset(torch.utils.data.Dataset):
    """Like run_gue_ablation.GUEDataset but also stores the per-example junction token window
    (see neuron_pilot_common.junction_window). Uses the identical tokenizer call (same
    padding/truncation/max_length) as GUEDataset, so input_ids/attention_mask are byte-for-byte
    identical and baseline metrics computed via this dataset match evaluate_baseline_pipeline()
    on GUEDataset exactly."""

    def __init__(self, csv_path, tokenizer, max_length, half_window=4):
        with open(csv_path) as f:
            rows = list(csv.reader(f))[1:]
        texts = [r[0] for r in rows]
        labels3 = [int(r[1]) for r in rows]
        enc = tokenizer(texts, return_tensors="pt", padding="longest", max_length=max_length,
                        truncation=True, return_attention_mask=True, return_offsets_mapping=True)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        offsets = enc["offset_mapping"]
        self.labels3 = torch.tensor(labels3, dtype=torch.long)
        self.labels_bin = torch.tensor([1 if l in (0, 1) else 0 for l in labels3], dtype=torch.long)
        self.windows = [junction_window(offsets[i], half_window=half_window)
                        for i in range(len(labels3))]
        self.num_labels = 3

    def __len__(self):
        return len(self.labels3)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i],
            "labels": self.labels3[i], "labels_bin": self.labels_bin[i],
            "window": self.windows[i],
        }


def collate_junction(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in
            ("input_ids", "attention_mask", "labels", "labels_bin")}
    out["window"] = [b["window"] for b in batch]
    return out


def evaluate_full(model, dataset, device, batch_size=64, intervention_spec=None):
    """
    intervention_spec: None, or dict {layer, neuron_idx, target, alpha, scope} with
    scope in {"all", "junction"}. Returns (metrics_dict, raw_arrays_dict).
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=collate_junction)
    all_preds3, all_labels3 = [], []
    all_preds_bin, all_labels_bin = [], []
    all_logodds, all_ppos = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            hook_handle = None
            if intervention_spec is not None:
                spec = intervention_spec
                row_indices = None
                if spec["scope"] == "junction":
                    # DNABERT-2 unpads internally: mlp.wo sees a flat (total_nnz, D_FFN)
                    # tensor, so per-example junction windows must be converted to flat row
                    # indices via each example's own starting offset (see neuron_pilot_common
                    # module docstring / flat_batch_offsets).
                    row_offsets = flat_batch_offsets(attn.cpu())
                    row_indices = []
                    for b, win in enumerate(batch["window"]):
                        start = int(row_offsets[b].item())
                        row_indices.extend(start + p for p in win)
                iv = NeuronIntervention(neuron_idx=spec["neuron_idx"], target=spec["target"],
                                        alpha=spec["alpha"], row_indices=row_indices)
                wo = get_wo_module(model.bert, spec["layer"])
                hook_handle = wo.register_forward_pre_hook(iv)

            logits = model(input_ids=input_ids, attention_mask=attn).logits
            if hook_handle is not None:
                hook_handle.remove()

            preds3 = logits.argmax(dim=-1).cpu().numpy()
            log_odds, p_pos = splice_binary_logodds(logits)
            preds_bin = (log_odds > 0).long().cpu().numpy()

            all_preds3.extend(preds3.tolist())
            all_labels3.extend(batch["labels"].numpy().tolist())
            all_preds_bin.extend(preds_bin.tolist())
            all_labels_bin.extend(batch["labels_bin"].numpy().tolist())
            all_logodds.extend(log_odds.cpu().numpy().tolist())
            all_ppos.extend(p_pos.cpu().numpy().tolist())

    preds3 = np.array(all_preds3); labels3 = np.array(all_labels3)
    preds_bin = np.array(all_preds_bin); labels_bin = np.array(all_labels_bin)
    logodds = np.array(all_logodds); ppos = np.array(all_ppos)

    metrics = {
        "accuracy_3class": _accuracy(labels3, preds3),
        "mcc_3class": _mcc(labels3, preds3),
        "f1_3class": _f1_macro(labels3, preds3),
        "accuracy_binary": float((preds_bin == labels_bin).mean()),
        "mcc_binary": _mcc(labels_bin, preds_bin),
        "auroc_binary": auroc_numpy(labels_bin, ppos),
        "auprc_binary": auprc_numpy(labels_bin, ppos),
        "mean_logodds": float(logodds.mean()),
        "mean_logodds_pos": float(logodds[labels_bin == 1].mean()) if (labels_bin == 1).any() else float("nan"),
        "mean_logodds_neg": float(logodds[labels_bin == 0].mean()) if (labels_bin == 0).any() else float("nan"),
    }
    raw = {"preds3": preds3, "labels3": labels3, "preds_bin": preds_bin,
           "labels_bin": labels_bin, "logodds": logodds, "ppos": ppos}
    return metrics, raw


def load_model(ckpt_path: Path, device: str):
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, revision=REVISION, model_max_length=MAX_LENGTH)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=3, trust_remote_code=True, revision=REVISION)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gue_root", required=True)
    p.add_argument("--top_k", type=int, default=25)
    p.add_argument("--sweep_top_n", type=int, default=3)
    p.add_argument("--half_window", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--sw_index", default=str(ROOT / "results" / "super_weight_index.json"))
    p.add_argument("--discovery_json",
                    default=str(ROOT / "results" / "neuron_pilot_dnabert2_discovery.json"))
    p.add_argument("--out", default=str(ROOT / "results" / "neuron_pilot_dnabert2_intervention.json"))
    p.add_argument("--out_sweep_csv",
                    default=str(ROOT / "results" / "neuron_pilot_dnabert2_intervention_sweep.csv"))
    p.add_argument("--raw_out_dir", default=str(ROOT / "results" / "neuron_pilot_dnabert2_raw"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    t0 = time.time()
    raw_out_dir = Path(args.raw_out_dir)
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    disc = json.loads(Path(args.discovery_json).read_text())
    candidates = disc["top200"][:args.top_k]
    print(f"[intervention] {len(candidates)} shortlisted candidates from {args.discovery_json}")

    model, tok = load_model(Path(args.ckpt), args.device)
    test_csv = Path(args.gue_root) / "splice" / "reconstructed" / "test.csv"
    ds_junction = JunctionGUEDataset(test_csv, tok, MAX_LENGTH, half_window=args.half_window)
    ds_plain = GUEDataset(str(test_csv), tok, MAX_LENGTH)
    print(f"[intervention] test set: {len(ds_junction)} examples")

    # ── Baseline + internal consistency check ──────────────────────────────
    baseline_metrics, baseline_raw = evaluate_full(model, ds_junction, args.device,
                                                    args.batch_size, intervention_spec=None)
    baseline_orig = evaluate_baseline_pipeline(model, ds_plain, device=args.device)
    consistency_ok = (abs(baseline_metrics["accuracy_3class"] - baseline_orig["accuracy"]) < 1e-6
                       and abs(baseline_metrics["mcc_3class"] - baseline_orig["mcc"]) < 1e-6)
    print(f"[intervention] baseline (new pipeline): acc3={baseline_metrics['accuracy_3class']:.4f} "
          f"mcc3={baseline_metrics['mcc_3class']:.4f}")
    print(f"[intervention] baseline (original run_gue_ablation.evaluate): "
          f"acc={baseline_orig['accuracy']:.4f} mcc={baseline_orig['mcc']:.4f}")
    print(f"[intervention] internal consistency check: {'PASS' if consistency_ok else 'FAIL'}")
    if not consistency_ok:
        print("[intervention][WARN] new evaluation pipeline diverges from the original "
              "run_gue_ablation.evaluate() — investigate before trusting downstream numbers.")
    np.savez(raw_out_dir / "baseline.npz", **baseline_raw)

    # ── Existing full 10-row ensemble + single row L3 r603 ablation (comparison anchor) ──
    sw_index = json.loads(Path(args.sw_index).read_text())
    entry = sw_index.get("dnabert2", [])
    sw_list = entry.get("results", entry) if isinstance(entry, dict) else entry
    ensemble_result = None
    row603_result = None
    if sw_list:
        saves = [(sw["layer"], sw["row"], _save_row(model, DOWN_PROJ_PATTERN, sw["layer"], sw["row"]))
                 for sw in sw_list]
        for sw in sw_list:
            _zero_row(model, DOWN_PROJ_PATTERN, sw["layer"], sw["row"])
        ensemble_result = evaluate_baseline_pipeline(model, ds_plain, device=args.device)
        for layer, row, saved in saves:
            _restore_row(model, DOWN_PROJ_PATTERN, layer, row, saved)
        print(f"[intervention] existing 10-row ensemble ablation: acc={ensemble_result['accuracy']:.4f} "
              f"(Δacc={ensemble_result['accuracy']-baseline_orig['accuracy']:+.4f}) "
              f"mcc={ensemble_result['mcc']:.4f} "
              f"(Δmcc={ensemble_result['mcc']-baseline_orig['mcc']:+.4f})")

        row603_l3 = next((sw for sw in sw_list if sw["row"] == 603 and sw["layer"] == 3), None)
        if row603_l3 is not None:
            saved = _save_row(model, DOWN_PROJ_PATTERN, 3, 603)
            _zero_row(model, DOWN_PROJ_PATTERN, 3, 603)
            row603_result = evaluate_baseline_pipeline(model, ds_plain, device=args.device)
            _restore_row(model, DOWN_PROJ_PATTERN, 3, 603, saved)
            print(f"[intervention] existing single-row L3 r603 ablation: acc={row603_result['accuracy']:.4f} "
                  f"(Δacc={row603_result['accuracy']-baseline_orig['accuracy']:+.4f})")
    else:
        print("[intervention][WARN] no super rows found in sw_index for dnabert2 — "
              "skipping ensemble/row603 comparison anchors.")

    # ── Shortlisted candidates: natural-magnitude grid (alpha=1) ────────────
    conditions = [
        ("zero_suppress", 0.0),
        ("neg_mean_suppress", None),  # target filled per-candidate below
        ("pos_mean_amplify", None),
    ]
    scopes = ["all", "junction"]

    shortlist_results = []
    for ci, cand in enumerate(candidates):
        layer, neuron = cand["layer"], cand["neuron"]
        neg_mean = cand["mean_act_neg_allpos"]
        pos_mean = cand["mean_act_pos_allpos"]
        cand_out = {"layer": layer, "neuron": neuron, "discovery_rank": ci,
                    "discovery_combined_best": cand["combined_best"],
                    "neg_mean_allpos": neg_mean, "pos_mean_allpos": pos_mean, "conditions": {}}
        for scope in scopes:
            for cond_name, fixed_target in conditions:
                target = fixed_target if fixed_target is not None else (
                    neg_mean if cond_name == "neg_mean_suppress" else pos_mean)
                spec = {"layer": layer, "neuron_idx": neuron, "target": target,
                        "alpha": 1.0, "scope": scope}
                metrics, raw = evaluate_full(model, ds_junction, args.device, args.batch_size,
                                             intervention_spec=spec)
                flips = flip_rates(baseline_raw["preds_bin"], raw["preds_bin"])
                key = f"{cond_name}__{scope}"
                cand_out["conditions"][key] = {
                    **metrics,
                    "delta_mcc_3class": metrics["mcc_3class"] - baseline_metrics["mcc_3class"],
                    "delta_acc_3class": metrics["accuracy_3class"] - baseline_metrics["accuracy_3class"],
                    "delta_mcc_binary": metrics["mcc_binary"] - baseline_metrics["mcc_binary"],
                    "delta_logodds": metrics["mean_logodds"] - baseline_metrics["mean_logodds"],
                    **flips,
                }
                # cache raw per-example outputs for the representative natural-suppression
                # condition (neg_mean, all-positions) only, to bound storage
                if cond_name == "neg_mean_suppress" and scope == "all":
                    np.savez(raw_out_dir / f"cand_L{layer}_N{neuron}_neg_mean_all.npz", **raw)
        shortlist_results.append(cand_out)
        print(f"[intervention] [{ci+1}/{len(candidates)}] layer={layer} neuron={neuron} "
              f"Δmcc3(neg_mean,all)={cand_out['conditions']['neg_mean_suppress__all']['delta_mcc_3class']:+.4f} "
              f"Δmcc3(neg_mean,junction)={cand_out['conditions']['neg_mean_suppress__junction']['delta_mcc_3class']:+.4f}")

    # ── Rerank by causal effect (natural neg-mean, all-positions suppression, 3-class MCC) ──
    shortlist_results.sort(
        key=lambda c: c["conditions"]["neg_mean_suppress__all"]["delta_mcc_3class"])
    print("\n[intervention] causal rerank (most damaging suppression first):")
    for c in shortlist_results[:5]:
        print(f"  layer={c['layer']:2d} neuron={c['neuron']:4d} "
              f"Δmcc3={c['conditions']['neg_mean_suppress__all']['delta_mcc_3class']:+.4f} "
              f"(discovery_rank={c['discovery_rank']})")

    # ── Small magnitude sweep for the top-N reranked candidates ─────────────
    sweep_rows = []
    for c in shortlist_results[:args.sweep_top_n]:
        layer, neuron = c["layer"], c["neuron"]
        neg_mean, pos_mean = c["neg_mean_allpos"], c["pos_mean_allpos"]
        for target_name, target in [("neg_mean", neg_mean), ("pos_mean", pos_mean)]:
            for scope in scopes:
                for alpha in ALPHA_SWEEP:
                    spec = {"layer": layer, "neuron_idx": neuron, "target": target,
                            "alpha": alpha, "scope": scope}
                    metrics, raw = evaluate_full(model, ds_junction, args.device,
                                                 args.batch_size, intervention_spec=spec)
                    flips = flip_rates(baseline_raw["preds_bin"], raw["preds_bin"])
                    sweep_rows.append({
                        "layer": layer, "neuron": neuron, "target_name": target_name,
                        "target_value": target, "scope": scope, "alpha": alpha,
                        **metrics,
                        "delta_mcc_3class": metrics["mcc_3class"] - baseline_metrics["mcc_3class"],
                        "delta_acc_3class": metrics["accuracy_3class"] - baseline_metrics["accuracy_3class"],
                        "total_flip_rate": flips["total_flip_rate"],
                    })
        print(f"[intervention] sweep done for layer={layer} neuron={neuron}")

    # ── Save ──────────────────────────────────────────────────────────────
    out = {
        "config": {"top_k": args.top_k, "sweep_top_n": args.sweep_top_n,
                    "half_window": args.half_window, "ckpt": args.ckpt,
                    "elapsed_sec": time.time() - t0},
        "baseline": {**baseline_metrics, "original_pipeline": baseline_orig,
                     "internal_consistency_check_passed": consistency_ok},
        "ensemble_10row_ablation": ensemble_result,
        "row603_L3_ablation": row603_result,
        "shortlist_candidates": shortlist_results,
        "strongest_candidate": {"layer": shortlist_results[0]["layer"],
                                 "neuron": shortlist_results[0]["neuron"]},
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[intervention] wrote {args.out}")

    if sweep_rows:
        with open(args.out_sweep_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sweep_rows)
        print(f"[intervention] wrote sweep table ({len(sweep_rows)} rows) -> {args.out_sweep_csv}")

    print(f"[intervention] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
