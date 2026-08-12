# scripts/interpretability/run_neuron_discovery_dnabert2.py
"""
Single-neuron discovery for DNABERT-2 splice classification (Apple-paper-style pilot).

Hooks the gated-MLP intermediate activation h = GELU(gate(x)) * up(x) immediately before
down-proj (mlp.wo), across all 12 layers simultaneously, on a sample of positive (donor +
acceptor, labels 1/0) and negative (label 2) splice examples from GUE splice/reconstructed
train.csv. For every (layer, neuron) computes a task-conditioned gradient x activation score,
evaluated both at the tokens covering the central splice junction (empirically confirmed at
character offset 200 in every 400bp example — see neuron_pilot_common.JUNCTION_CHAR_OFFSET)
and at a location-agnostic per-example arg-max over sequence positions.

Requires the fine-tuned splice classifier checkpoint (produced by the existing, unmodified
scripts/evaluation/run_gue_ablation.py --model dnabert2 --task splice/reconstructed).

Usage:
    python scripts/interpretability/run_neuron_discovery_dnabert2.py \
        --gue_root /work/11034/atzanakak/GUE/GUE --n_pos 300 --n_neg 300 --seed 42

Output:
    results/neuron_pilot_dnabert2_discovery.csv   (full 12*3072 ranked table)
    results/neuron_pilot_dnabert2_discovery.json  (top-200 + run config/metadata)
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

from neuron_pilot_common import (  # noqa: E402
    D_FFN,
    JUNCTION_CHAR_OFFSET,
    MultiLayerActivationGradCapture,
    flat_batch_offsets,
    junction_window,
    splice_binary_logodds,
)

N_LAYERS = 12
MAX_LENGTH = 80  # matches _MAX_LEN["reconstructed"] in run_gue_ablation.py — same as fine-tune/eval
MODEL_ID = "zhihan1996/DNABERT-2-117M"
REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
DEFAULT_CKPT = ROOT / "results" / "gue_checkpoints" / "dnabert2_reconstructed" / "model_state.pt"


def load_split(csv_path: Path):
    with open(csv_path) as f:
        rows = list(csv.reader(f))[1:]
    return [(r[0], int(r[1])) for r in rows]


def sample_examples(rows, n_pos, n_neg, seed):
    rng = random.Random(seed)
    pos = [r for r in rows if r[1] in (0, 1)]
    neg = [r for r in rows if r[1] == 2]
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos[:n_pos], neg[:n_neg]


def load_model_and_tokenizer(ckpt_path: Path, device: str):
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, revision=REVISION, model_max_length=MAX_LENGTH
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=3, trust_remote_code=True, revision=REVISION
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Fine-tuned splice checkpoint not found at {ckpt_path}. Run:\n"
            f"  python scripts/evaluation/run_gue_ablation.py --model dnabert2 "
            f"--task splice/reconstructed --gue_root <GUE_ROOT>\nfirst."
        )
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[discovery] loaded checkpoint {ckpt_path} "
          f"(missing={len(missing)} unexpected={len(unexpected)})")
    model.to(device)
    model.eval()
    return model, tok


def batched(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def run_discovery(model, tok, examples, device, half_window, batch_size=16):
    """
    examples: list of (sequence, label) tuples (already positive+negative combined, shuffled
    order irrelevant since stats are accumulated per class).

    Returns dict of numpy arrays, each shape [N_LAYERS, D_FFN]:
        act_junction_sum_pos/neg, act_maxloc_sum_pos/neg   (signed activation sums)
        gradabs_junction_sum, gradabs_maxloc_sum            (abs gradient sums, all examples)
        count_pos, count_neg (scalars), count_all (scalar)
    """
    acc = {
        "act_junction_sum_pos": np.zeros((N_LAYERS, D_FFN), dtype=np.float64),
        "act_junction_sum_neg": np.zeros((N_LAYERS, D_FFN), dtype=np.float64),
        "act_maxloc_sum_pos": np.zeros((N_LAYERS, D_FFN), dtype=np.float64),
        "act_maxloc_sum_neg": np.zeros((N_LAYERS, D_FFN), dtype=np.float64),
        "act_allpos_sum_pos": np.zeros((N_LAYERS, D_FFN), dtype=np.float64),
        "act_allpos_sum_neg": np.zeros((N_LAYERS, D_FFN), dtype=np.float64),
        "gradabs_junction_sum": np.zeros((N_LAYERS, D_FFN), dtype=np.float64),
        "gradabs_maxloc_sum": np.zeros((N_LAYERS, D_FFN), dtype=np.float64),
    }
    # all-positions token counts (denominator for act_allpos_sum_* — differs from
    # example counts because sequences have variable non-pad length)
    allpos_count_pos = 0
    allpos_count_neg = 0
    count_pos = 0
    count_neg = 0
    count_all = 0
    junction_token_positions_seen = []

    layer_indices = list(range(N_LAYERS))

    for batch in batched(examples, batch_size):
        seqs = [s for s, _ in batch]
        labels = [1 if lbl in (0, 1) else 0 for _, lbl in batch]  # binary pos/neg
        enc = tok(seqs, return_tensors="pt", padding="longest", truncation=True,
                  max_length=MAX_LENGTH, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")  # [B, L, 2], stays on CPU
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        model.zero_grad(set_to_none=True)
        cap = MultiLayerActivationGradCapture(model.bert, layer_indices)
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        log_odds, _ = splice_binary_logodds(out.logits)
        log_odds.sum().backward()

        # DNABERT-2 unpads internally: h/grad at mlp.wo have shape (total_nnz, D_FFN), a flat
        # concatenation of only the real tokens across the batch (see neuron_pilot_common
        # module docstring). flat_batch_offsets() gives each example's starting row.
        B, L = input_ids.shape
        row_offsets = flat_batch_offsets(attn_mask.cpu())  # [B], CPU is fine (bookkeeping only)
        lengths = attn_mask.sum(dim=1).cpu()

        for li in layer_indices:
            h = cap.h(li).detach()          # [total_nnz, D_FFN]
            g = cap.grad(li)
            g = torch.zeros_like(h) if g is None else g.detach()

            for b in range(B):
                start = int(row_offsets[b].item())
                length = int(lengths[b].item())
                h_b = h[start:start + length, :]   # [length, D_FFN] — this example's own rows
                g_b = g[start:start + length, :]

                offs_b = offsets[b]
                win = junction_window(offs_b, half_window=half_window,
                                       char_offset=JUNCTION_CHAR_OFFSET)
                win = [p for p in win if p < length]
                if li == 0:
                    junction_token_positions_seen.append(win[len(win) // 2] if win else -1)

                # location-agnostic: per-neuron arg over |h| within this example's own rows
                maxloc_absval, maxloc_idx = h_b.abs().max(dim=0)          # [D_FFN]
                h_at_max = torch.gather(h_b, 0, maxloc_idx.unsqueeze(0)).squeeze(0).cpu().numpy()
                g_at_max = torch.gather(g_b.abs(), 0, maxloc_idx.unsqueeze(0)).squeeze(0).cpu().numpy()

                h_allpos_sum = h_b.sum(dim=0).cpu().numpy()
                n_valid_b = length

                if win:
                    h_win = h_b[win, :].mean(dim=0).cpu().numpy()        # [D_FFN]
                    g_win = g_b[win, :].abs().mean(dim=0).cpu().numpy()  # [D_FFN]
                else:
                    h_win = np.zeros(D_FFN)
                    g_win = np.zeros(D_FFN)

                if labels[b] == 1:
                    if win:
                        acc["act_junction_sum_pos"][li] += h_win
                    acc["act_maxloc_sum_pos"][li] += h_at_max
                    acc["act_allpos_sum_pos"][li] += h_allpos_sum
                    if li == 0:
                        allpos_count_pos += n_valid_b
                else:
                    if win:
                        acc["act_junction_sum_neg"][li] += h_win
                    acc["act_maxloc_sum_neg"][li] += h_at_max
                    acc["act_allpos_sum_neg"][li] += h_allpos_sum
                    if li == 0:
                        allpos_count_neg += n_valid_b

                if win:
                    acc["gradabs_junction_sum"][li] += g_win
                acc["gradabs_maxloc_sum"][li] += g_at_max

        cap.remove()

        count_pos += sum(labels)
        count_neg += len(labels) - sum(labels)
        count_all += len(labels)

    acc["count_pos"] = count_pos
    acc["count_neg"] = count_neg
    acc["count_all"] = count_all
    acc["allpos_count_pos"] = allpos_count_pos
    acc["allpos_count_neg"] = allpos_count_neg
    acc["junction_token_positions_seen"] = junction_token_positions_seen
    return acc


def build_ranking(acc: dict) -> list[dict]:
    n_pos, n_neg, n_all = acc["count_pos"], acc["count_neg"], acc["count_all"]
    mean_junction_pos = acc["act_junction_sum_pos"] / max(n_pos, 1)
    mean_junction_neg = acc["act_junction_sum_neg"] / max(n_neg, 1)
    mean_maxloc_pos = acc["act_maxloc_sum_pos"] / max(n_pos, 1)
    mean_maxloc_neg = acc["act_maxloc_sum_neg"] / max(n_neg, 1)
    mean_allpos_pos = acc["act_allpos_sum_pos"] / max(acc["allpos_count_pos"], 1)
    mean_allpos_neg = acc["act_allpos_sum_neg"] / max(acc["allpos_count_neg"], 1)
    mean_gradabs_junction = acc["gradabs_junction_sum"] / max(n_all, 1)
    mean_gradabs_maxloc = acc["gradabs_maxloc_sum"] / max(n_all, 1)

    act_diff_junction = mean_junction_pos - mean_junction_neg
    act_diff_maxloc = mean_maxloc_pos - mean_maxloc_neg

    combined_junction = act_diff_junction * mean_gradabs_junction
    combined_maxloc = act_diff_maxloc * mean_gradabs_maxloc

    records = []
    for li in range(N_LAYERS):
        for ni in range(D_FFN):
            records.append({
                "layer": li,
                "neuron": ni,
                "act_diff_junction": float(act_diff_junction[li, ni]),
                "act_diff_maxloc": float(act_diff_maxloc[li, ni]),
                "grad_abs_junction": float(mean_gradabs_junction[li, ni]),
                "grad_abs_maxloc": float(mean_gradabs_maxloc[li, ni]),
                "combined_junction": float(combined_junction[li, ni]),
                "combined_maxloc": float(combined_maxloc[li, ni]),
                "combined_best": float(max(abs(combined_junction[li, ni]),
                                            abs(combined_maxloc[li, ni]))),
                "mean_act_pos_junction": float(mean_junction_pos[li, ni]),
                "mean_act_neg_junction": float(mean_junction_neg[li, ni]),
                "mean_act_pos_allpos": float(mean_allpos_pos[li, ni]),
                "mean_act_neg_allpos": float(mean_allpos_neg[li, ni]),
            })
    records.sort(key=lambda r: -r["combined_best"])
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gue_root", required=True)
    p.add_argument("--n_pos", type=int, default=300)
    p.add_argument("--n_neg", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--half_window", type=int, default=4,
                    help="±tokens around the junction token counted as 'junction window'")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--out_csv", default=str(ROOT / "results" / "neuron_pilot_dnabert2_discovery.csv"))
    p.add_argument("--out_json", default=str(ROOT / "results" / "neuron_pilot_dnabert2_discovery.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    t0 = time.time()
    rows = load_split(Path(args.gue_root) / "splice" / "reconstructed" / "train.csv")
    pos, neg = sample_examples(rows, args.n_pos, args.n_neg, args.seed)
    print(f"[discovery] sampled {len(pos)} positive / {len(neg)} negative from "
          f"{len(rows)} train examples (seed={args.seed})")
    examples = pos + neg

    model, tok = load_model_and_tokenizer(Path(args.ckpt), args.device)

    acc = run_discovery(model, tok, examples, args.device, args.half_window, args.batch_size)

    jt = np.array([p for p in acc["junction_token_positions_seen"] if p >= 0])
    print(f"[discovery] junction token index across examples: "
          f"mean={jt.mean():.1f} min={jt.min()} max={jt.max()} "
          f"(sequence max_length={MAX_LENGTH})")

    records = build_ranking(acc)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"[discovery] wrote full ranking ({len(records)} rows) -> {out_csv}")

    top200 = records[:200]
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps({
        "config": {
            "n_pos": len(pos), "n_neg": len(neg), "seed": args.seed,
            "half_window": args.half_window, "max_length": MAX_LENGTH,
            "ckpt": str(args.ckpt), "elapsed_sec": time.time() - t0,
            "junction_token_index_mean": float(jt.mean()),
            "junction_token_index_min": int(jt.min()),
            "junction_token_index_max": int(jt.max()),
        },
        "top200": top200,
    }, indent=2))
    print(f"[discovery] wrote top-200 + metadata -> {out_json}")
    print(f"[discovery] top-5 candidates:")
    for r in records[:5]:
        print(f"  layer={r['layer']:2d} neuron={r['neuron']:4d} "
              f"combined_best={r['combined_best']:.5g} "
              f"act_diff_junction={r['act_diff_junction']:.4g} "
              f"grad_abs_junction={r['grad_abs_junction']:.4g}")
    print(f"[discovery] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
