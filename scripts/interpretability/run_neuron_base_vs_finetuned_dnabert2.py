# scripts/interpretability/run_neuron_base_vs_finetuned_dnabert2.py
"""
Base (pretrained) vs fine-tuned comparison for the strongest candidate neuron.

Tests whether the neuron already separates splice from non-splice sequences before
fine-tuning (raw-activation AUROC, on both the base pretrained DNABERT-2 checkpoint — the
same checkpoint the original SW detection ran on — and the fine-tuned splice backbone), and
whether fine-tuning mainly changes its routing (||U_k||_F rank of row 603 / its own strongest
row, and cosine similarity of Wd[:,i] / Wg[i,:] / Wu[i,:] before vs after) or its raw
separability.

Requires: results/neuron_pilot_dnabert2_intervention.json (for the strongest candidate
coordinate) and the fine-tuned splice checkpoint.

Usage:
    python scripts/interpretability/run_neuron_base_vs_finetuned_dnabert2.py \
        --gue_root /work/11034/atzanakak/GUE/GUE
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

from neuron_pilot_common import (  # noqa: E402
    ActivationGradCapture,
    JUNCTION_CHAR_OFFSET,
    auroc_numpy,
    flat_batch_offsets,
    get_wo_module,
    junction_window,
)
from run_dnabert2_uk_audit import extract_mlp_weights, frob_norm_uk  # noqa: E402

MODEL_ID = "zhihan1996/DNABERT-2-117M"
REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
MAX_LENGTH = 80
DEFAULT_CKPT = ROOT / "results" / "gue_checkpoints" / "dnabert2_reconstructed" / "model_state.pt"


def sample_examples(csv_path: Path, n_pos, n_neg, seed):
    with open(csv_path) as f:
        rows = list(csv.reader(f))[1:]
    pos = [r[0] for r in rows if r[1] in ("0", "1")]
    neg = [r[0] for r in rows if r[1] == "2"]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:n_pos], neg[:n_neg]


def collect_activation(model_backbone, tok, seqs, layer, device, half_window):
    """Returns per-sequence junction-window-mean activation for `layer`, using the same
    forward-pre-hook primitive as discovery (grad tracking allowed but unused)."""
    vals = []
    for i in range(0, len(seqs), 16):
        batch = seqs[i:i + 16]
        enc = tok(batch, return_tensors="pt", padding="longest", truncation=True,
                  max_length=MAX_LENGTH, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        cap = ActivationGradCapture()
        handle = get_wo_module(model_backbone, layer).register_forward_pre_hook(cap)
        model_backbone(input_ids=input_ids, attention_mask=attn)
        handle.remove()

        # DNABERT-2 unpads internally: h is (total_nnz, D_FFN), not (B, L, D_FFN) — see
        # neuron_pilot_common module docstring. Each example's own rows are a contiguous
        # slice starting at flat_batch_offsets(attn)[b].
        h = cap.h.detach()
        row_offsets = flat_batch_offsets(attn.cpu())
        lengths = attn.sum(dim=1).cpu()
        for b in range(len(batch)):
            start = int(row_offsets[b].item())
            length = int(lengths[b].item())
            win = junction_window(offsets[b], half_window=half_window,
                                  char_offset=JUNCTION_CHAR_OFFSET)
            win = [p for p in win if p < length]
            if not win:
                win = [0]
            h_b = h[start:start + length, :]
            vals.append(h_b[win, :].mean(dim=0).cpu().numpy())
    return np.stack(vals)  # [N, D_FFN]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gue_root", required=True)
    p.add_argument("--n_pos", type=int, default=200)
    p.add_argument("--n_neg", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--half_window", type=int, default=4)
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--intervention_json",
                    default=str(ROOT / "results" / "neuron_pilot_dnabert2_intervention.json"))
    p.add_argument("--structural_json",
                    default=str(ROOT / "results" / "neuron_pilot_dnabert2_structural.json"))
    p.add_argument("--out", default=str(ROOT / "results" / "neuron_pilot_dnabert2_base_vs_finetuned.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    t0 = time.time()
    interv = json.loads(Path(args.intervention_json).read_text())
    layer, neuron = interv["strongest_candidate"]["layer"], interv["strongest_candidate"]["neuron"]
    print(f"[base_vs_ft] strongest candidate: layer={layer} neuron={neuron}")

    strongest_row = None
    if Path(args.structural_json).exists():
        struct = json.loads(Path(args.structural_json).read_text())
        rep = next((c for c in struct["candidate_structural_reports"]
                    if c["layer"] == layer and c["neuron"] == neuron), None)
        if rep is not None:
            strongest_row = rep["strongest_outgoing_row"]
    print(f"[base_vs_ft] candidate's strongest outgoing row (fine-tuned): {strongest_row}")

    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, revision=REVISION, model_max_length=MAX_LENGTH)

    print("[base_vs_ft] loading base pretrained model ...")
    base_model = transformers.AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, revision=REVISION, torch_dtype=torch.float32)
    base_model.to(args.device)
    base_model.eval()

    print("[base_vs_ft] loading fine-tuned splice classifier ...")
    ft_full = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=3, trust_remote_code=True, revision=REVISION)
    state = torch.load(Path(args.ckpt), map_location="cpu")
    ft_full.load_state_dict(state, strict=False)
    ft_full.to(args.device)
    ft_full.eval()
    ft_backbone = ft_full.bert

    test_csv = Path(args.gue_root) / "splice" / "reconstructed" / "test.csv"
    pos, neg = sample_examples(test_csv, args.n_pos, args.n_neg, args.seed)
    seqs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))
    print(f"[base_vs_ft] probing with {len(pos)} positive / {len(neg)} negative test sequences")

    act_base = collect_activation(base_model, tok, seqs, layer, args.device, args.half_window)
    act_ft = collect_activation(ft_backbone, tok, seqs, layer, args.device, args.half_window)

    auroc_base = auroc_numpy(labels, act_base[:, neuron])
    auroc_ft = auroc_numpy(labels, act_ft[:, neuron])
    # also report the |value| AUROC in case the neuron's separating direction flips sign
    auroc_base_abs = auroc_numpy(labels, np.abs(act_base[:, neuron]))
    auroc_ft_abs = auroc_numpy(labels, np.abs(act_ft[:, neuron]))
    print(f"[base_vs_ft] AUROC (raw activation, signed): base={auroc_base:.4f} fine-tuned={auroc_ft:.4f}")
    print(f"[base_vs_ft] AUROC (|activation|):            base={auroc_base_abs:.4f} fine-tuned={auroc_ft_abs:.4f}")

    Wg_base, Wu_base, Wd_base = extract_mlp_weights(base_model, layer)
    Wg_ft, Wu_ft, Wd_ft = extract_mlp_weights(ft_backbone, layer)

    frob_base = frob_norm_uk(Wg_base, Wu_base, Wd_base)
    frob_ft = frob_norm_uk(Wg_ft, Wu_ft, Wd_ft)

    def rank_of(frob, row):
        order = np.argsort(-frob)
        return int(np.where(order == row)[0][0]) + 1

    row603_rank_base = rank_of(frob_base, 603)
    row603_rank_ft = rank_of(frob_ft, 603)
    print(f"[base_vs_ft] row 603 ||U_k||_F rank at layer {layer}: "
          f"base={row603_rank_base}/768 (val={frob_base[603]:.2f}) "
          f"fine-tuned={row603_rank_ft}/768 (val={frob_ft[603]:.2f})")

    strongest_row_ranks = None
    if strongest_row is not None:
        strongest_row_ranks = {
            "row": strongest_row,
            "rank_base": rank_of(frob_base, strongest_row),
            "rank_finetuned": rank_of(frob_ft, strongest_row),
            "frob_base": float(frob_base[strongest_row]),
            "frob_finetuned": float(frob_ft[strongest_row]),
        }
        print(f"[base_vs_ft] candidate's strongest row {strongest_row} ||U_k||_F rank: "
              f"base={strongest_row_ranks['rank_base']}/768 "
              f"fine-tuned={strongest_row_ranks['rank_finetuned']}/768")

    def cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else float("nan")

    cos_wd = cosine(Wd_base[:, neuron], Wd_ft[:, neuron])
    cos_wg = cosine(Wg_base[neuron, :], Wg_ft[neuron, :])
    cos_wu = cosine(Wu_base[neuron, :], Wu_ft[neuron, :])
    print(f"[base_vs_ft] cosine similarity base vs fine-tuned: "
          f"Wd[:,i]={cos_wd:.4f}  Wg[i,:]={cos_wg:.4f}  Wu[i,:]={cos_wu:.4f}")

    out = {
        "config": {"layer": layer, "neuron": neuron, "n_pos": len(pos), "n_neg": len(neg),
                    "seed": args.seed, "elapsed_sec": time.time() - t0},
        "activation_auroc": {
            "base_signed": auroc_base, "finetuned_signed": auroc_ft,
            "base_abs": auroc_base_abs, "finetuned_abs": auroc_ft_abs,
        },
        "row603_uk_rank": {"base": row603_rank_base, "finetuned": row603_rank_ft,
                            "frob_base": float(frob_base[603]), "frob_finetuned": float(frob_ft[603]),
                            "total_rows": int(len(frob_base))},
        "candidate_strongest_row_uk_rank": strongest_row_ranks,
        "cosine_similarity_base_vs_finetuned": {"Wd_col": cos_wd, "Wg_row": cos_wg, "Wu_row": cos_wu},
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[base_vs_ft] wrote {args.out}")
    print(f"[base_vs_ft] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
