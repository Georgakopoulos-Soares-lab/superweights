# scripts/find_task_superweights.py
"""
Per-task superweight discovery for GUE benchmarks.

Instead of using PPL-based activation-spike detection (run_detection.py), this
script finds superweights empirically from task accuracy: it loads a fine-tuned
checkpoint, scans the top-K highest-L1-norm rows in every down-projection layer,
zeros each candidate row individually, and labels it a "task superweight" if the
resulting accuracy drop is statistically anomalous (z-score on Δacc).

Usage:
    # single task
    python scripts/find_task_superweights.py \\
        --model dnabert2 \\
        --task prom/prom_core_notata \\
        --gue_root /work/11034/atzanakak/GUE/GUE

    # scan ALL rows (slow but complete):
    python scripts/find_task_superweights.py \\
        --model dnabert2 --task EMP/H3K4me3 \\
        --gue_root /work/11034/atzanakak/GUE/GUE \\
        --scan_all

    # override thresholds:
    python scripts/find_task_superweights.py \\
        --model dnabert2 --task splice/reconstructed \\
        --gue_root /work/11034/atzanakak/GUE/GUE \\
        --topk 100 --z_thresh 2.5 --min_drop 0.005

Output JSON  : results/task_superweights.json
Key          : "<model>/<task>"
Each entry contains:
  baseline       – {accuracy, mcc, f1} of fine-tuned model on test split
  candidates     – all scanned rows: [{layer, row, l1_norm, accuracy, mcc, f1,
                   delta_acc, delta_mcc, z_score}]  (sorted by delta_acc)
  task_sw        – subset of candidates flagged as task superweights
  topk           – K used  (None if scan_all)
  z_thresh       – z-score threshold applied
  min_drop       – minimum |Δacc| threshold applied
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Directory structure:
#   <outer>/genomic-super-weights/            ← results/, logs/ live here
#       genomic-super-weights/                ← configs/, scripts/ live here
#           scripts/find_task_superweights.py ← this file
_SCRIPTS_DIR = Path(__file__).resolve().parent
_INNER_ROOT  = _SCRIPTS_DIR.parent            # configs/, stubs/, …
_OUTER_ROOT  = _INNER_ROOT.parent             # results/, logs/, …
_REPO_ROOT   = _OUTER_ROOT                    # alias used below for results

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import transformers
import yaml


# ── Shared constants (mirrors run_gue_ablation.py) ───────────────────────────

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


def _task_key(task: str) -> str:
    leaf  = task.split("/")[-1]
    group = task.split("/")[0]
    return leaf if leaf in _MAX_LEN else group


# ── Dataset ───────────────────────────────────────────────────────────────────

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


def _collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ── NTv3 wrapper (identical to run_gue_ablation.py) ──────────────────────────

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
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=_collate)
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


# ── Module / row helpers ──────────────────────────────────────────────────────

def _resolve_module(model, pattern: str, layer_idx: int):
    path = pattern.replace("{i}", str(layer_idx))
    obj  = model.backbone if isinstance(model, _NTv3Classifier) else model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def _save_row(model, pattern, layer_idx, row) -> torch.Tensor:
    return _resolve_module(model, pattern, layer_idx).weight.data[row, :].clone()


def _zero_row(model, pattern, layer_idx, row):
    with torch.no_grad():
        _resolve_module(model, pattern, layer_idx).weight.data[row, :] = 0.0


def _restore_row(model, pattern, layer_idx, row, saved: torch.Tensor):
    with torch.no_grad():
        _resolve_module(model, pattern, layer_idx).weight.data[row, :] = saved


# ── Task-SW scan ──────────────────────────────────────────────────────────────

def scan_task_superweights(
    model,
    test_ds,
    pattern: str,
    num_layers: int,
    device: str,
    topk: int | None = 50,
    z_thresh: float = 2.5,
    min_drop: float = 0.005,
    batch_size: int = 64,
) -> dict:
    """
    Scan rows of each down-projection layer for task-specific superweights.

    Strategy:
    1. For each layer, rank rows by L1 norm (largest = most "active").
    2. If topk is set, consider only the top-K rows; otherwise all rows.
    3. Zero each candidate row individually, measure Δacc = acc_zeroed - baseline.
    4. Compute z-scores of Δacc across all candidates.
    5. Flag a row as a task superweight if BOTH:
         z_score < -z_thresh   (anomalously large drop relative to other rows)
       AND
         delta_acc < -min_drop (drop is not trivially small in absolute terms).

    Returns dict with keys: baseline, candidates (all), task_sw (flagged subset).
    """
    baseline = evaluate(model, test_ds, batch_size=batch_size, device=device)
    print(f"  Baseline: acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")

    # Collect candidates
    candidates_raw: list[dict] = []
    for li in range(num_layers):
        mod   = _resolve_module(model, pattern, li)
        w     = mod.weight.data          # (out_features, in_features)
        l1    = w.abs().sum(dim=1)       # L1 norm per row

        if topk is not None and topk < w.shape[0]:
            row_indices = torch.topk(l1, topk).indices.tolist()
        else:
            row_indices = list(range(w.shape[0]))

        candidates_raw.append((li, row_indices, l1))

    total = sum(len(idxs) for _, idxs, _ in candidates_raw)
    print(f"  Scanning {total} candidate rows across {num_layers} layers "
          f"({'top-' + str(topk) if topk else 'all rows'}) …")

    results = []
    done    = 0
    for li, row_indices, l1 in candidates_raw:
        for ri in row_indices:
            saved = _save_row(model, pattern, li, ri)
            _zero_row(model, pattern, li, ri)
            ev    = evaluate(model, test_ds, batch_size=batch_size, device=device)
            _restore_row(model, pattern, li, ri, saved)

            delta_acc = ev["accuracy"] - baseline["accuracy"]
            delta_mcc = ev["mcc"]      - baseline["mcc"]
            results.append({
                "layer":     li,
                "row":       ri,
                "l1_norm":   float(l1[ri].item()),
                "accuracy":  ev["accuracy"],
                "mcc":       ev["mcc"],
                "f1":        ev["f1"],
                "delta_acc": delta_acc,
                "delta_mcc": delta_mcc,
                "z_score":   None,   # filled below
            })
            done += 1
            if done % 50 == 0 or done == total:
                print(f"    [{done}/{total}]  layer={li}  row={ri}  "
                      f"Δacc={delta_acc:+.4f}  Δmcc={delta_mcc:+.4f}")

    # Compute z-scores of delta_acc
    delta_accs = np.array([r["delta_acc"] for r in results])
    mean_d = float(delta_accs.mean())
    std_d  = float(delta_accs.std())
    for r in results:
        r["z_score"] = float((r["delta_acc"] - mean_d) / std_d) if std_d > 0 else 0.0

    # Sort by delta_acc (most harmful first)
    results.sort(key=lambda x: x["delta_acc"])

    # Identify task superweights
    task_sw = [
        r for r in results
        if r["z_score"] < -z_thresh and r["delta_acc"] < -min_drop
    ]

    print(f"\n  Found {len(task_sw)} task superweight(s) "
          f"(z < -{z_thresh}, Δacc < -{min_drop}):")
    for r in task_sw:
        print(f"    layer={r['layer']:2d}  row={r['row']:4d}  "
              f"Δacc={r['delta_acc']:+.5f}  z={r['z_score']:+.2f}  "
              f"L1={r['l1_norm']:.1f}")
    if not task_sw:
        # Fallback: report top-5 most impactful even if not above threshold
        print("  (Below threshold — showing top-5 most impactful for reference:)")
        for r in results[:5]:
            print(f"    layer={r['layer']:2d}  row={r['row']:4d}  "
                  f"Δacc={r['delta_acc']:+.5f}  z={r['z_score']:+.2f}")

    return {
        "baseline":   baseline,
        "candidates": results,
        "task_sw":    task_sw,
        "topk":       topk,
        "z_thresh":   z_thresh,
        "min_drop":   min_drop,
        "delta_acc_mean": mean_d,
        "delta_acc_std":  std_d,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect task-specific superweights for a single GUE task."
    )
    parser.add_argument("--model",      required=True, choices=["dnabert2", "ntv3"],
                        help="Model key — must match a configs/<model>.yaml")
    parser.add_argument("--task",       required=True,
                        help="GUE task path relative to gue_root, e.g. prom/prom_core_notata")
    parser.add_argument("--gue_root",   required=True,
                        help="Path to the GUE/ directory")
    parser.add_argument("--ckpt_dir",   default=None,
                        help="Fine-tuned checkpoint dir (default: <repo>/results/gue_checkpoints/<model>_<task_leaf>)")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--topk",       type=int, default=50,
                        help="Scan top-K rows per layer by L1 norm (default: 50). "
                             "Use 0 to scan ALL rows.")
    parser.add_argument("--z_thresh",   type=float, default=2.5,
                        help="Z-score threshold: rows with z < -z_thresh flagged as task SW.")
    parser.add_argument("--min_drop",   type=float, default=0.005,
                        help="Minimum absolute Δacc for a row to be flagged (default: 0.005).")
    parser.add_argument("--batch",      type=int, default=64)
    parser.add_argument("--out",        default=str(_REPO_ROOT / "results" / "task_superweights.json"),
                        help="Output JSON file (results merged by key).")
    parser.add_argument("--hf_token",   default=None)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    topk = args.topk if args.topk > 0 else None   # 0 → scan all

    # ── Config ──────────────────────────────────────────────────────────────
    config_path = _INNER_ROOT / "configs" / f"{args.model}.yaml"
    config     = yaml.safe_load(config_path.read_text())
    model_id   = config["model_id"]
    pattern    = config["down_proj_pattern"]
    num_layers = config["num_layers"]

    task_leaf  = args.task.split("/")[-1]
    tkey       = _task_key(args.task)
    max_length = args.max_length or _MAX_LEN.get(tkey, 512)
    ckpt_dir   = args.ckpt_dir   or str(
        _REPO_ROOT / "results" / "gue_checkpoints" / f"{args.model}_{task_leaf}"
    )
    task_path  = os.path.join(args.gue_root, args.task)
    hf_token   = args.hf_token or os.environ.get("HF_TOKEN")

    print(f"\n{'='*64}")
    print(f"  Task-SW scan")
    print(f"  Model  : {args.model}  ({model_id})")
    print(f"  Task   : {args.task}  (max_len={max_length})")
    print(f"  Ckpt   : {ckpt_dir}")
    print(f"  top-K  : {topk if topk else 'ALL'}"
          f"  z_thresh={args.z_thresh}  min_drop={args.min_drop}")
    print(f"{'='*64}\n")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tok_kwargs = {"trust_remote_code": True} if config.get("hf_trust_remote_code") else {}
    if "zhihan1996" in model_id:
        tok_kwargs["revision"] = "7bce263b15377fc15361f52cfab88f8b586abda0"
    if "InstaDeepAI" in model_id and "NTv3" in model_id:
        tok_kwargs["code_revision"] = "0ecff3637f0d3ba5b686d1095083218157c2ca34"
    if hf_token:
        tok_kwargs["token"] = hf_token

    print("Loading tokenizer …")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, model_max_length=max_length, **tok_kwargs
    )
    if "InstaDeepAI" in model_id and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Test dataset ─────────────────────────────────────────────────────────
    test_csv = os.path.join(task_path, "test.csv")
    if not os.path.exists(test_csv):
        print(f"[error] test.csv not found at {test_csv}", file=sys.stderr)
        sys.exit(1)

    print("Loading test dataset …")
    test_ds = GUEDataset(test_csv, tokenizer, max_length)
    print(f"  test={len(test_ds)}  num_labels={test_ds.num_labels}")

    # ── Model + checkpoint ────────────────────────────────────────────────────
    print(f"Loading model from {model_id} …")
    if "InstaDeepAI" in model_id and "NTv3" in model_id:
        backbone  = transformers.AutoModelForMaskedLM.from_pretrained(model_id, **tok_kwargs)
        embed_dim = backbone.config.embed_dim
        model     = _NTv3Classifier(backbone, embed_dim, test_ds.num_labels)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=test_ds.num_labels, **tok_kwargs
        )

    ckpt_path = Path(ckpt_dir) / "model_state.pt"
    if ckpt_path.exists():
        print(f"Loading fine-tuned weights from {ckpt_path} …")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        print(
            f"[warn] No checkpoint at {ckpt_path}.\n"
            "       Run run_gue_ablation.py first to fine-tune, or pass --ckpt_dir.\n"
            "       Evaluating base model — results will not reflect task learning.",
            file=sys.stderr,
        )

    model = model.to(args.device)

    # ── Scan ─────────────────────────────────────────────────────────────────
    result = scan_task_superweights(
        model,
        test_ds,
        pattern    = pattern,
        num_layers = num_layers,
        device     = args.device,
        topk       = topk,
        z_thresh   = args.z_thresh,
        min_drop   = args.min_drop,
        batch_size = args.batch,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    key = f"{args.model}/{args.task}"
    existing[key] = {
        "model":      args.model,
        "task":       args.task,
        "max_length": max_length,
        "ckpt_dir":   str(ckpt_dir),
        **result,
    }
    out_path.write_text(json.dumps(existing, indent=2))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Done — {args.task}")
    print(f"  Baseline:   acc={result['baseline']['accuracy']:.4f}  "
          f"mcc={result['baseline']['mcc']:.4f}")
    print(f"  Candidates: {len(result['candidates'])} scanned")
    print(f"  Task SWs:   {len(result['task_sw'])} found")
    for r in result["task_sw"]:
        print(f"    layer={r['layer']:2d}  row={r['row']:4d}  "
              f"Δacc={r['delta_acc']:+.5f}  z={r['z_score']:+.2f}")
    print(f"\nSaved → {args.out}  (key: '{key}')")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
