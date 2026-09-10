# scripts/run_gue_per_row_ablation.py
"""
Per-row GUE ablation: zero each detected superrow individually and measure
the impact on downstream task accuracy. Identifies which of the N superrows
is actually responsible for the performance drop.

Usage:
    python scripts/run_gue_per_row_ablation.py \\
        --model dnabert2 \\
        --task prom/prom_core_notata \\
        --gue_root /work/11034/atzanakak/GUE/GUE

    # all three tasks at once via shell loop:
    for task in prom/prom_core_notata EMP/H3K4me3 splice/reconstructed; do
        bash run.sh scripts/run_gue_per_row_ablation.py \\
            --model dnabert2 --task $task \\
            --gue_root /work/11034/atzanakak/GUE/GUE
    done

Output JSON key: "<model>/<task>"
Each entry contains:
  baseline        – fine-tuned model performance
  per_row         – list of {layer, row, accuracy, mcc, delta_acc, delta_mcc}
                    one entry per superrow, zeroed individually
  all_rows        – result of zeroing all superrows simultaneously (matches
                    the existing run_gue_ablation.py pruned_sw condition)
  rand_mean       – mean over 10 random-row controls (same count as superrows)
  rand_all        – individual random control results
"""

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import transformers
import yaml


# ── Re-use helpers from run_gue_ablation.py ──────────────────────────────────
# (copy-minimised — only what's needed)

_MAX_LEN = {
    "EMP":               128,
    "EPI":               128,
    "fungi":             512,
    "mouse":              30,
    "tf":                 30,
    "prom_core_all":      20,
    "prom_core_notata":   20,
    "prom_core_tata":     20,
    "prom_300_all":       70,
    "prom_300_notata":    70,
    "prom_300_tata":      70,
    "reconstructed":      80,
    "covid":             256,
    "species_40":        512,
    "species_20":        512,
}

_EPOCHS = {
    "EMP":              3,
    "EPI":              3,
    "fungi":            3,
    "mouse":            5,
    "tf":               3,
    "prom_core_all":    4,
    "prom_core_notata": 4,
    "prom_core_tata":  10,
    "prom_300_all":     4,
    "prom_300_notata":  4,
    "prom_300_tata":   10,
    "reconstructed":    5,
    "covid":            8,
    "species_40":       3,
    "species_20":       3,
}


def _task_key(task: str) -> str:
    leaf  = task.split("/")[-1]
    group = task.split("/")[0]
    return leaf if leaf in _MAX_LEN else group


# ── Dataset ──────────────────────────────────────────────────────────────────

class GUEDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int):
        with open(csv_path) as f:
            rows = list(csv.reader(f))[1:]

        if len(rows[0]) == 2:
            texts  = [r[0] for r in rows]
            labels = [int(r[1]) for r in rows]
        elif len(rows[0]) == 3:
            texts  = [[r[0], r[1]] for r in rows]
            labels = [int(r[2]) for r in rows]
        else:
            raise ValueError(f"Unexpected CSV columns: {len(rows[0])}")

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
        )
        self.input_ids      = enc["input_ids"]
        if "attention_mask" in enc:
            self.attention_mask = enc["attention_mask"]
        else:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            self.attention_mask = (self.input_ids != pad_id).long()
        self.labels     = torch.tensor(labels, dtype=torch.long)
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids":      self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels":         self.labels[i],
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── NTv3 wrapper ─────────────────────────────────────────────────────────────

class _NTv3Classifier(nn.Module):
    class _Output:
        def __init__(self, logits, loss=None):
            self.logits = logits
            self.loss   = loss

    def __init__(self, backbone, embed_dim: int, num_labels: int):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        nd      = getattr(self.backbone.config, "num_downsamples", 7)
        min_len = 2 ** (nd + 1)
        seq_len = input_ids.shape[1]
        if seq_len < min_len:
            pad    = min_len - seq_len
            pad_id = getattr(self.backbone.config, "pad_token_id", 1)
            input_ids      = torch.nn.functional.pad(input_ids,     (0, pad), value=pad_id)
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=0)

        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hidden = out.hidden_states[-1]
        elif hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        elif isinstance(out, dict) and "embeddings" in out:
            hidden = out["embeddings"]
        else:
            for v in (out.values() if isinstance(out, dict) else vars(out).values()):
                if isinstance(v, torch.Tensor) and v.dim() == 3:
                    hidden = v
                    break
            else:
                raise RuntimeError("Cannot find hidden states in NTv3 output")

        if attention_mask is not None:
            mask   = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)

        logits = self.classifier(pooled)
        loss   = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return self._Output(logits=logits, loss=loss)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataset, batch_size: int = 64, device: str = "cuda") -> dict:
    model.eval()
    loader     = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    all_preds, all_labels = [], []
    for batch in loader:
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].cpu().numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    return {
        "accuracy": float(sklearn.metrics.accuracy_score(labels, preds)),
        "mcc":      float(sklearn.metrics.matthews_corrcoef(labels, preds)),
        "f1":       float(sklearn.metrics.f1_score(labels, preds, average="macro", zero_division=0)),
    }


# ── Row zeroing helpers ───────────────────────────────────────────────────────

def _resolve_module(model, pattern: str, layer_idx: int):
    path = pattern.replace("{i}", str(layer_idx))
    obj  = model.backbone if isinstance(model, _NTv3Classifier) else model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def _save_row(model, pattern, layer_idx, row):
    return _resolve_module(model, pattern, layer_idx).weight.data[row, :].clone()


def _zero_row(model, pattern, layer_idx, row):
    with torch.no_grad():
        _resolve_module(model, pattern, layer_idx).weight.data[row, :] = 0.0


def _restore_row(model, pattern, layer_idx, row, saved):
    with torch.no_grad():
        _resolve_module(model, pattern, layer_idx).weight.data[row, :] = saved


# ── Per-row ablation ──────────────────────────────────────────────────────────

def run_per_row_ablation(
    model,
    sw_list: list,
    test_ds,
    pattern: str,
    num_layers: int,
    device: str,
    n_control: int = 10,
) -> dict:
    """
    For each superrow individually: zero it, evaluate, restore.
    Also runs the all-at-once condition and random controls.
    """
    baseline = evaluate(model, test_ds, device=device)
    print(f"  Baseline: acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")

    # ── Per-row: one at a time ────────────────────────────────────────────────
    print(f"\n  Per-row ablation ({len(sw_list)} superrows):")
    per_row_results = []
    for sw in sw_list:
        saved = _save_row(model, pattern, sw["layer"], sw["row"])
        _zero_row(model, pattern, sw["layer"], sw["row"])
        ev = evaluate(model, test_ds, device=device)
        _restore_row(model, pattern, sw["layer"], sw["row"], saved)

        delta_acc = ev["accuracy"] - baseline["accuracy"]
        delta_mcc = ev["mcc"]      - baseline["mcc"]
        print(f"    layer={sw['layer']:2d}  row={sw['row']:4d}  "
              f"acc={ev['accuracy']:.4f}  Δacc={delta_acc:+.4f}  Δmcc={delta_mcc:+.4f}  "
              f"out_max={sw.get('out_max', '?'):.1f}")
        per_row_results.append({
            "layer":     sw["layer"],
            "row":       sw["row"],
            "out_max":   sw.get("out_max"),
            "in_max":    sw.get("in_max"),
            "accuracy":  ev["accuracy"],
            "mcc":       ev["mcc"],
            "f1":        ev["f1"],
            "delta_acc": delta_acc,
            "delta_mcc": delta_mcc,
        })

    # Sort by most harmful (most negative delta_acc)
    per_row_results.sort(key=lambda x: x["delta_acc"])

    # ── All rows at once ──────────────────────────────────────────────────────
    print(f"\n  All {len(sw_list)} rows simultaneously:")
    saves = [(sw["layer"], sw["row"], _save_row(model, pattern, sw["layer"], sw["row"]))
             for sw in sw_list]
    for sw in sw_list:
        _zero_row(model, pattern, sw["layer"], sw["row"])
    all_ev = evaluate(model, test_ds, device=device)
    for layer, row, saved in saves:
        _restore_row(model, pattern, layer, row, saved)

    all_delta_acc = all_ev["accuracy"] - baseline["accuracy"]
    all_delta_mcc = all_ev["mcc"]      - baseline["mcc"]
    print(f"    acc={all_ev['accuracy']:.4f}  Δacc={all_delta_acc:+.4f}  Δmcc={all_delta_mcc:+.4f}")

    # ── Random controls (same count as superrows) ─────────────────────────────
    rng        = random.Random(42)
    sw_coords  = {(sw["layer"], sw["row"]) for sw in sw_list}
    n_prune    = len(sw_list)
    candidates = []
    for li in range(num_layers):
        m     = _resolve_module(model, pattern, li)
        nrows = m.weight.data.shape[0]
        for ri in range(nrows):
            if (li, ri) not in sw_coords:
                candidates.append((li, ri))

    print(f"\n  Random controls (n={n_control}, {n_prune} rows each):")
    rand_results = []
    for _ in range(n_control):
        chosen = rng.sample(candidates, n_prune)
        saves  = [(l, r, _save_row(model, pattern, l, r)) for l, r in chosen]
        for l, r in chosen:
            _zero_row(model, pattern, l, r)
        ev = evaluate(model, test_ds, device=device)
        rand_results.append(ev)
        for l, r, saved in saves:
            _restore_row(model, pattern, l, r, saved)

    rand_mean_acc = float(np.mean([m["accuracy"] for m in rand_results]))
    rand_mean_mcc = float(np.mean([m["mcc"]      for m in rand_results]))
    print(f"    mean acc={rand_mean_acc:.4f}  mean mcc={rand_mean_mcc:.4f}")

    return {
        "baseline":  baseline,
        "per_row":   per_row_results,
        "all_rows":  {**all_ev, "delta_acc": all_delta_acc, "delta_mcc": all_delta_mcc},
        "rand_mean": {"accuracy": rand_mean_acc, "mcc": rand_mean_mcc},
        "rand_all":  rand_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Per-row GUE ablation")
    parser.add_argument("--model",      required=True, choices=["dnabert2", "ntv3"])
    parser.add_argument("--task",       required=True)
    parser.add_argument("--gue_root",   required=True)
    parser.add_argument("--max_length", type=int,   default=None)
    parser.add_argument("--ckpt_dir",   default=None)
    parser.add_argument("--sw_index",   default="results/negative_results/super_weight_index.json")
    parser.add_argument("--out",        default="results/gue/gue_per_row_ablation.json")
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--n_rand",     type=int,   default=10)
    parser.add_argument("--hf_token",   default=None)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config     = yaml.safe_load(open(f"configs/{args.model}.yaml"))
    model_id   = config["model_id"]
    pattern    = config["down_proj_pattern"]
    num_layers = config["num_layers"]

    task_leaf  = args.task.split("/")[-1]
    tkey       = _task_key(args.task)
    max_length = args.max_length or _MAX_LEN.get(tkey, 512)
    ckpt_dir   = args.ckpt_dir   or f"results/gue_checkpoints/{args.model}_{task_leaf}"
    task_path  = os.path.join(args.gue_root, args.task)
    hf_token   = args.hf_token or os.environ.get("HF_TOKEN")

    print(f"\n{'='*64}")
    print(f"  Per-row ablation")
    print(f"  Model : {args.model}  ({model_id})")
    print(f"  Task  : {args.task}  (max_len={max_length})")
    print(f"  Ckpt  : {ckpt_dir}")
    print(f"{'='*64}\n")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tok_kwargs = {"trust_remote_code": True} if config.get("hf_trust_remote_code") else {}
    if "zhihan1996" in model_id:
        tok_kwargs["revision"] = "7bce263b15377fc15361f52cfab88f8b586abda0"
    if "InstaDeepAI" in model_id and "NTv3" in model_id:
        tok_kwargs["code_revision"] = "0ecff3637f0d3ba5b686d1095083218157c2ca34"
    if hf_token:
        tok_kwargs["token"] = hf_token

    print("Loading tokenizer ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, model_max_length=max_length, **tok_kwargs
    )
    if "InstaDeepAI" in model_id and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Test dataset ─────────────────────────────────────────────────────────
    print("Loading test dataset ...")
    test_ds = GUEDataset(os.path.join(task_path, "test.csv"), tokenizer, max_length)
    print(f"  test={len(test_ds)}  num_labels={test_ds.num_labels}")

    # ── Model + checkpoint ────────────────────────────────────────────────────
    print(f"Loading model from {model_id} ...")
    if "InstaDeepAI" in model_id and "NTv3" in model_id:
        backbone = transformers.AutoModelForMaskedLM.from_pretrained(model_id, **tok_kwargs)
        embed_dim = backbone.config.embed_dim
        model = _NTv3Classifier(backbone, embed_dim, test_ds.num_labels)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=test_ds.num_labels, **tok_kwargs
        )

    ckpt_path = Path(ckpt_dir) / "model_state.pt"
    if ckpt_path.exists():
        print(f"Loading fine-tuned weights from {ckpt_path} ...")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    else:
        print("[warn] No checkpoint found — evaluating base model (results will be meaningless).")

    model = model.to(args.device)

    # ── SW index ──────────────────────────────────────────────────────────────
    sw_index = json.loads(Path(args.sw_index).read_text())
    entry    = sw_index.get(args.model, [])
    sw_list  = entry.get("results", entry) if isinstance(entry, dict) else entry

    if not sw_list:
        print(f"[error] No superrows found for {args.model} in {args.sw_index}.")
        sys.exit(1)

    print(f"\nFound {len(sw_list)} superrow(s) for {args.model}:")
    for sw in sw_list:
        print(f"  layer={sw['layer']}  row={sw['row']}  out_max={sw.get('out_max', '?'):.1f}")

    # ── Run ───────────────────────────────────────────────────────────────────
    result = run_per_row_ablation(
        model, sw_list, test_ds, pattern, num_layers, args.device, n_control=args.n_rand
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Summary — {args.task}")
    print(f"  Baseline:  acc={result['baseline']['accuracy']:.4f}  mcc={result['baseline']['mcc']:.4f}")
    print(f"  All rows:  acc={result['all_rows']['accuracy']:.4f}  Δacc={result['all_rows']['delta_acc']:+.4f}")
    print(f"  Rand mean: acc={result['rand_mean']['accuracy']:.4f}")
    print(f"\n  Per-row (sorted by Δacc):")
    for r in result["per_row"]:
        print(f"    layer={r['layer']:2d}  row={r['row']:4d}  "
              f"Δacc={r['delta_acc']:+.4f}  Δmcc={r['delta_mcc']:+.4f}  "
              f"out_max={r['out_max']:.1f}")
    print(f"{'='*64}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    key = f"{args.model}/{args.task}"
    existing[key] = {
        "model":      args.model,
        "task":       args.task,
        "max_length": max_length,
        "ckpt_dir":   ckpt_dir,
        **result,
    }
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"Saved → {args.out}  (key: '{key}')")


if __name__ == "__main__":
    main()
