# scripts/interpretability/run_neuron_structural_mapping_dnabert2.py
"""
Structural connection between causally-validated candidate neurons and the existing DNABERT-2
super-rows.

For each candidate (from run_neuron_causal_intervention_dnabert2.py's reranked shortlist):
  - inspects W_down[:, i] (mlp.wo.weight column i) to find its strongest outgoing row and
    checks whether that row is one of the 10 known super-rows (esp. row 603);
  - decomposes every known super-row's structural amplifier score
        ||U_k||_F^2 = sum_i C[k,i],  C[k,i] = Wd[k,i]^2 * ||Wg[i,:]||^2 * ||Wu[i,:]||^2
    (the exact formula already used in scripts/analysis/run_dnabert2_uk_audit.py — imported,
    not reimplemented) and reports the cumulative fraction of that score captured by the
    top 1/5/10/50/100 hidden neurons, to establish whether each row is a single dominant
    neuron or a distributed ensemble;
  - runs four explicit ablation comparisons on the fine-tuned splice classifier, all scored
    through the same evaluate() used everywhere else in this repo: (a) candidate-neuron-
    activation-only [pulled from the intervention JSON — already computed there], (b) the
    candidate's single largest outgoing scalar Wd[k*,i], (c) the candidate's full outgoing
    column Wd[:,i], (d) the existing full-row / 10-row-ensemble ablations [also pulled from
    the intervention JSON].

Requires: results/neuron_pilot_dnabert2_intervention.json (candidate list + existing row/
ensemble numbers) and the fine-tuned splice checkpoint.

Usage:
    python scripts/interpretability/run_neuron_structural_mapping_dnabert2.py \
        --gue_root /work/11034/atzanakak/GUE/GUE --top_n 5
"""
from __future__ import annotations

import argparse
import json
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

from neuron_pilot_common import get_wo_module, neuron_contribution_to_row  # noqa: E402
from run_dnabert2_uk_audit import extract_mlp_weights, frob_norm_uk  # noqa: E402
from run_gue_ablation import GUEDataset, evaluate as evaluate_pipeline  # noqa: E402

MODEL_ID = "zhihan1996/DNABERT-2-117M"
REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
MAX_LENGTH = 80
DEFAULT_CKPT = ROOT / "results" / "gue_checkpoints" / "dnabert2_reconstructed" / "model_state.pt"
CUTOFFS = (1, 5, 10, 50, 100)


def load_model(ckpt_path: Path, device: str):
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, revision=REVISION, model_max_length=MAX_LENGTH)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=3, trust_remote_code=True, revision=REVISION)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, tok


def cumulative_contribution(C_row: np.ndarray, cutoffs=CUTOFFS):
    order = np.argsort(-C_row)
    total = float(C_row.sum())
    cum = np.cumsum(C_row[order])
    table = {}
    for k in cutoffs:
        kk = min(k, len(C_row))
        table[str(k)] = float(cum[kk - 1] / total) if total > 0 else float("nan")
    return table, order[:100].tolist(), C_row[order[:5]].tolist()


def _save_col(wo, col):
    return wo.weight.data[:, col].clone()


def _zero_col(wo, col):
    with torch.no_grad():
        wo.weight.data[:, col] = 0.0


def _restore_col(wo, col, saved):
    with torch.no_grad():
        wo.weight.data[:, col] = saved


def _save_scalar(wo, row, col):
    return wo.weight.data[row, col].item()


def _zero_scalar(wo, row, col):
    with torch.no_grad():
        wo.weight.data[row, col] = 0.0


def _restore_scalar(wo, row, col, saved):
    with torch.no_grad():
        wo.weight.data[row, col] = saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gue_root", required=True)
    p.add_argument("--top_n", type=int, default=5,
                    help="number of top reranked candidates to structurally analyze")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--sw_index", default=str(ROOT / "results" / "super_weight_index.json"))
    p.add_argument("--intervention_json",
                    default=str(ROOT / "results" / "neuron_pilot_dnabert2_intervention.json"))
    p.add_argument("--out", default=str(ROOT / "results" / "neuron_pilot_dnabert2_structural.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    t0 = time.time()
    interv = json.loads(Path(args.intervention_json).read_text())
    candidates = interv["shortlist_candidates"][:args.top_n]
    baseline_orig = interv["baseline"]["original_pipeline"]

    sw_index = json.loads(Path(args.sw_index).read_text())
    entry = sw_index.get("dnabert2", [])
    sw_list = entry.get("results", entry) if isinstance(entry, dict) else entry
    sw_rows_by_layer: dict[int, list[int]] = {}
    for e in sw_list:
        sw_rows_by_layer.setdefault(int(e["layer"]), []).append(int(e["row"]))
    sw_coord_set = {(int(e["layer"]), int(e["row"])) for e in sw_list}

    model, tok = load_model(Path(args.ckpt), args.device)
    test_ds = GUEDataset(str(Path(args.gue_root) / "splice" / "reconstructed" / "test.csv"),
                          tok, MAX_LENGTH)

    # ── Cache per-layer weight extraction (only layers we actually need) ────
    needed_layers = sorted(set([c["layer"] for c in candidates]) | set(sw_rows_by_layer.keys()))
    weights_by_layer = {}
    for li in needed_layers:
        Wg, Wu, Wd = extract_mlp_weights(model.bert, li)
        weights_by_layer[li] = (Wg, Wu, Wd)
        print(f"[structural] extracted weights for layer {li}: Wg{Wg.shape} Wu{Wu.shape} Wd{Wd.shape}")

    # ── Known super-row decomposition (independent of candidates — characterizes the rows
    #    themselves: single neuron vs distributed ensemble) ─────────────────
    sw_row_decomposition = {}
    for li, rows in sw_rows_by_layer.items():
        Wg, Wu, Wd = weights_by_layer[li]
        for row in rows:
            C_row = neuron_contribution_to_row(Wd, Wg, Wu, row)
            table, top100_idx, top5_vals = cumulative_contribution(C_row)
            frob = float(np.sqrt(C_row.sum()))
            sw_row_decomposition[f"L{li}_r{row}"] = {
                "layer": li, "row": row, "frob_norm_uk": frob,
                "cumulative_fraction_top_k": table,
                "top5_neuron_indices": top100_idx[:5], "top5_neuron_C_values": top5_vals,
            }
            print(f"[structural] SW row L{li} r{row}: ||U_k||_F={frob:.2f}  "
                  f"top1={table['1']:.3f} top10={table['10']:.3f} top100={table['100']:.3f}")

    # ── Per-candidate structural mapping + explicit 4-way ablation comparison ──
    candidate_reports = []
    for cand in candidates:
        layer, neuron = cand["layer"], cand["neuron"]
        Wg, Wu, Wd = weights_by_layer[layer]
        col = Wd[:, neuron]  # outgoing weights from this neuron, shape (d_model,)
        strongest_row = int(np.argmax(np.abs(col)))
        strongest_val = float(col[strongest_row])
        is_known_sw_row = (layer, strongest_row) in sw_coord_set
        is_row603 = strongest_row == 603

        # this candidate's contribution to its own strongest row's structural score
        C_to_strongest = neuron_contribution_to_row(Wd, Wg, Wu, strongest_row)
        candidate_rank_in_strongest_row = int((C_to_strongest > C_to_strongest[neuron]).sum()) + 1
        candidate_frac_of_strongest_row = float(C_to_strongest[neuron] / C_to_strongest.sum()) \
            if C_to_strongest.sum() > 0 else float("nan")

        # if candidate's layer has a KNOWN sw row, report candidate's contribution to it too
        contribution_to_known_sw_rows = {}
        for row in sw_rows_by_layer.get(layer, []):
            C_row = neuron_contribution_to_row(Wd, Wg, Wu, row)
            rank = int((C_row > C_row[neuron]).sum()) + 1
            frac = float(C_row[neuron] / C_row.sum()) if C_row.sum() > 0 else float("nan")
            contribution_to_known_sw_rows[f"L{layer}_r{row}"] = {
                "neuron_rank_in_row": rank, "neuron_fraction_of_row_score": frac,
            }

        # ── four-way ablation comparison ────────────────────────────────────
        wo = get_wo_module(model.bert, layer)
        saved_scalar = _save_scalar(wo, strongest_row, neuron)
        _zero_scalar(wo, strongest_row, neuron)
        scalar_ablation = evaluate_pipeline(model, test_ds, device=args.device)
        _restore_scalar(wo, strongest_row, neuron, saved_scalar)

        saved_col = _save_col(wo, neuron)
        _zero_col(wo, neuron)
        column_ablation = evaluate_pipeline(model, test_ds, device=args.device)
        _restore_col(wo, neuron, saved_col)

        neuron_activation_only = cand["conditions"].get("zero_suppress__all")

        candidate_reports.append({
            "layer": layer, "neuron": neuron,
            "strongest_outgoing_row": strongest_row,
            "strongest_outgoing_value": strongest_val,
            "is_row603": is_row603,
            "is_known_super_row": is_known_sw_row,
            "candidate_rank_in_own_strongest_row": candidate_rank_in_strongest_row,
            "candidate_fraction_of_own_strongest_row_score": candidate_frac_of_strongest_row,
            "contribution_to_known_sw_rows_same_layer": contribution_to_known_sw_rows,
            "ablation_comparison": {
                "candidate_neuron_activation_only": {
                    "accuracy": neuron_activation_only["accuracy_3class"] if neuron_activation_only else None,
                    "mcc": neuron_activation_only["mcc_3class"] if neuron_activation_only else None,
                    "delta_acc": (neuron_activation_only["accuracy_3class"] - baseline_orig["accuracy"])
                                 if neuron_activation_only else None,
                    "delta_mcc": (neuron_activation_only["mcc_3class"] - baseline_orig["mcc"])
                                 if neuron_activation_only else None,
                },
                "largest_outgoing_scalar_only": {
                    "row": strongest_row, "col": neuron,
                    "accuracy": scalar_ablation["accuracy"], "mcc": scalar_ablation["mcc"],
                    "delta_acc": scalar_ablation["accuracy"] - baseline_orig["accuracy"],
                    "delta_mcc": scalar_ablation["mcc"] - baseline_orig["mcc"],
                },
                "full_outgoing_column": {
                    "accuracy": column_ablation["accuracy"], "mcc": column_ablation["mcc"],
                    "delta_acc": column_ablation["accuracy"] - baseline_orig["accuracy"],
                    "delta_mcc": column_ablation["mcc"] - baseline_orig["mcc"],
                },
                "existing_row603_L3_full_row": interv.get("row603_L3_ablation"),
                "existing_10row_ensemble": interv.get("ensemble_10row_ablation"),
            },
        })
        print(f"[structural] candidate L{layer} N{neuron}: strongest_row={strongest_row} "
              f"(is_row603={is_row603}, is_known_sw_row={is_known_sw_row}) "
              f"scalar_ablation_Δacc={scalar_ablation['accuracy']-baseline_orig['accuracy']:+.4f} "
              f"column_ablation_Δacc={column_ablation['accuracy']-baseline_orig['accuracy']:+.4f}")

    out = {
        "config": {"top_n": args.top_n, "ckpt": args.ckpt, "elapsed_sec": time.time() - t0},
        "baseline": baseline_orig,
        "known_sw_row_decomposition": sw_row_decomposition,
        "candidate_structural_reports": candidate_reports,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[structural] wrote {args.out}")
    print(f"[structural] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
