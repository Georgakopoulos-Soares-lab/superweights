"""
Fast linear-probe super-weight ablation.

Pipeline:
  1. Load backbone in bf16, frozen.
  2. Extract EOS-token embeddings for train + test (one forward pass, no grad).
  3. Train logistic regression on cached train embeddings.
  4. Evaluate on test → baseline accuracy / MCC.
  5. Zero the super-row weight in the down_proj layer.
  6. Re-extract test embeddings → re-evaluate → pruned accuracy.
  7. Repeat step 6 with 10 random rows → random-control mean.
  8. Save results to --ablation_out JSON.

Usage:
  python probe_ablation.py \\
      --model_name  GenerTeam/GENERator-v2-eukaryote-3b-base \\
      --dataset_name InstaDeepAI/nucleotide_transformer_downstream_tasks_revised \\
      --subset_name  H3K27ac \\
      --sw_index     results/super_weight_index.json \\
      --down_proj_pattern "model.layers.{i}.mlp.down_proj" \\
      --num_transformer_layers 30 \\
      --ablation_out results/generator_probe_ablation.json \\
      --max_length   512
"""

import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, accuracy_score
from transformers import AutoTokenizer, AutoModel


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--subset_name", default=None)
    p.add_argument("--sw_index", required=True)
    p.add_argument("--down_proj_pattern", default="model.layers.{i}.mlp.down_proj")
    p.add_argument("--num_transformer_layers", type=int, required=True)
    p.add_argument("--ablation_out", required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_rand_controls", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf_cache", default=None,
                   help="HF_HOME / TRANSFORMERS_CACHE directory")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _resolve_module(model, pattern, layer_idx):
    """Resolve dotted path like 'model.layers.{i}.mlp.down_proj'."""
    path = pattern.replace("{i}", str(layer_idx))
    m = model
    for attr in path.split("."):
        m = getattr(m, attr)
    return m


def _zero_rows(model, pattern, layer_idx, rows):
    """Zero specified output rows in a weight matrix. Returns saved values."""
    m = _resolve_module(model, pattern, layer_idx)
    saved = m.weight.data[rows, :].clone()
    m.weight.data[rows, :] = 0.0
    return saved


def _restore_rows(model, pattern, layer_idx, rows, saved):
    m = _resolve_module(model, pattern, layer_idx)
    m.weight.data[rows, :] = saved


# ── Embedding extraction ───────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, tokenizer, texts, max_length, batch_size, device):
    """Return (N, hidden_size) array of EOS-position embeddings."""
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        # Use last hidden state at EOS / last non-pad position
        hidden = out.hidden_states[-1]          # (B, L, H)
        # Find last non-padding token per sequence
        attention_mask = enc["attention_mask"]   # (B, L)
        seq_lens = attention_mask.sum(dim=1) - 1 # (B,)
        embs = hidden[torch.arange(hidden.size(0)), seq_lens]  # (B, H)
        all_embs.append(embs.float().cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"  extracted {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.concatenate(all_embs, axis=0)


# ── Sklearn probe ─────────────────────────────────────────────────────────────
def train_probe(X_train, y_train, seed):
    clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    clf.fit(X_train, y_train)
    return clf


def evaluate_probe(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    return {"accuracy": float(acc), "mcc": float(mcc)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dtype = torch.bfloat16 if (torch.cuda.is_available() and
                                     torch.cuda.get_device_capability()[0] >= 8) else torch.float32
    print(f"Device: {device}  dtype: {load_dtype}")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load backbone (no classification head) ────────────────────────────────
    print(f"Loading backbone: {args.model_name}")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        args.model_name, torch_dtype=load_dtype, trust_remote_code=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    print(f"Model loaded in {time.time()-t0:.1f}s  |  "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset_name}  subset={args.subset_name}")
    if args.subset_name:
        try:
            ds = load_dataset(args.dataset_name, args.subset_name)
        except ValueError:
            ds = load_dataset(args.dataset_name)
            if "task" in ds["train"].column_names:
                ds = ds.filter(lambda x: x["task"] == args.subset_name)
    else:
        ds = load_dataset(args.dataset_name)

    train_texts = ds["train"]["sequence"]
    train_labels = np.array(ds["train"]["label"], dtype=int)
    test_texts  = ds["test"]["sequence"]
    test_labels  = np.array(ds["test"]["label"], dtype=int)
    print(f"Train: {len(train_texts)}  Test: {len(test_texts)}")

    # ── Baseline: extract embeddings then train probe ─────────────────────────
    print("\n── Extracting train embeddings (baseline) ──")
    X_train = extract_embeddings(model, tokenizer, train_texts,
                                  args.max_length, args.batch_size, device)
    print("── Extracting test embeddings (baseline) ──")
    X_test_base = extract_embeddings(model, tokenizer, test_texts,
                                     args.max_length, args.batch_size, device)

    print("── Training logistic regression probe ──")
    clf = train_probe(X_train, train_labels, args.seed)
    baseline = evaluate_probe(clf, X_test_base, test_labels)
    print(f"  Baseline: {baseline}")

    # ── Load super-weight index ───────────────────────────────────────────────
    sw_index = json.loads(Path(args.sw_index).read_text())
    model_key = args.model_name.rstrip("/").split("/")[-1]
    entry = sw_index.get(model_key, sw_index.get(args.model_name, {}))
    sw_list = entry.get("results", entry) if isinstance(entry, dict) else entry

    # ── Condition 2: prune super row(s) ──────────────────────────────────────
    pruned_sw = None
    if sw_list:
        print("\n── Pruning super row(s) and re-extracting test embeddings ──")
        saved_vals = {}
        for sw in sw_list:
            li, row = sw["layer"], sw["row"]
            saved_vals[(li, row)] = _zero_rows(model, args.down_proj_pattern, li, [row])
        X_test_pruned = extract_embeddings(model, tokenizer, test_texts,
                                           args.max_length, args.batch_size, device)
        pruned_sw = evaluate_probe(clf, X_test_pruned, test_labels)
        print(f"  Pruned SW: {pruned_sw}")
        for (li, row), saved in saved_vals.items():
            _restore_rows(model, args.down_proj_pattern, li, [row], saved)
    else:
        print("  [skip] No super rows in index for this model.")

    # ── Condition 3: random controls ─────────────────────────────────────────
    print(f"\n── Random controls (n={args.n_rand_controls}) ──")
    n_rows_to_prune = max(len(sw_list), 1)
    rand_results = []
    for i in range(args.n_rand_controls):
        li = sw_list[0]["layer"] if sw_list else random.randint(0, args.num_transformer_layers - 1)
        m = _resolve_module(model, args.down_proj_pattern, li)
        n_rows = m.weight.data.shape[0]
        rows = random.sample(range(n_rows), min(n_rows_to_prune, n_rows))
        saved = _zero_rows(model, args.down_proj_pattern, li, rows)
        X_test_rand = extract_embeddings(model, tokenizer, test_texts,
                                         args.max_length, args.batch_size, device)
        r = evaluate_probe(clf, X_test_rand, test_labels)
        rand_results.append(r)
        _restore_rows(model, args.down_proj_pattern, li, rows, saved)
        print(f"  ctrl {i+1}/{args.n_rand_controls}: {r}")

    rand_mean = {
        k: float(np.mean([r[k] for r in rand_results]))
        for k in rand_results[0]
    }
    print(f"  Rand mean: {rand_mean}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Baseline MCC : {baseline['mcc']:.4f}")
    if pruned_sw:
        delta = pruned_sw['mcc'] - baseline['mcc']
        print(f"Pruned SW MCC: {pruned_sw['mcc']:.4f}  (Δ={delta:+.4f})")
    print(f"Rand ctrl MCC: {rand_mean['mcc']:.4f}")
    print("="*60)

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "model": args.model_name,
        "dataset": args.dataset_name,
        "subset": args.subset_name,
        "max_length": args.max_length,
        "baseline": baseline,
        "pruned_sw": pruned_sw,
        "pruned_rand_mean": rand_mean,
        "pruned_rand_all": rand_results,
        "sw_list": sw_list,
    }

    out_path = Path(args.ablation_out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    task_key = f"{model_key}__{args.subset_name or 'default'}"
    existing[task_key] = result
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
