"""
scripts/run_layer_drop_generator.py
-------------------------------------
Layer-importance profiling for GENERator via MLP down_proj zeroing.

Three experiments:
  1. Single-layer drop  — zero all rows in one layer at a time, measure ΔMCC.
                          Produces a per-layer importance bar chart.
  2. Cumulative drop    — drop layers one by one in order of ascending importance
                          (least-important first), measure cumulative ΔMCC.
                          Curves: (a) furthest-from-L4 first, (b) least-important first.
  3. SW-layer isolation — compare dropping L4 alone vs dropping every other layer
                          cumulatively until only L4 remains.

Run from repo root:
    python scripts/run_layer_drop_generator.py \\
        --subset_name H3K4me3 \\
        --hf_cache /work/11034/atzanakak/ls6/huggingface/transformers

All flags have defaults matching the GENERator 3B setup.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from transformers import AutoModel, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Module helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_module(model, pattern, layer_idx):
    path = pattern.replace("{i}", str(layer_idx))
    m = model
    for attr in path.split("."):
        m = getattr(m, attr)
    return m


def _zero_layer(model, pattern, layer_idx):
    """Zero all rows in one down_proj layer. Returns saved weight tensor."""
    m = _resolve_module(model, pattern, layer_idx)
    saved = m.weight.data.clone()
    m.weight.data.zero_()
    return saved


def _restore_layer(model, pattern, layer_idx, saved):
    m = _resolve_module(model, pattern, layer_idx)
    m.weight.data.copy_(saved)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, tokenizer, texts, max_length, batch_size, device,
                       verbose=False):
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(device)
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        seq_lens = enc["attention_mask"].sum(dim=1) - 1
        embs = hidden[torch.arange(hidden.size(0)), seq_lens]
        all_embs.append(embs.float().cpu().numpy())
        if verbose and (i // batch_size) % 5 == 0:
            print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    if verbose:
        print()
    return np.concatenate(all_embs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Probe
# ─────────────────────────────────────────────────────────────────────────────

def train_probe(X, y, seed):
    clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    clf.fit(X, y)
    return clf


def eval_probe(clf, X, y):
    preds = clf.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "mcc":      float(matthews_corrcoef(y, preds)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────────────────────────────────────

def single_layer_drop(model, tokenizer, pattern, num_layers,
                      test_texts, test_labels, clf, baseline_mcc,
                      max_length, batch_size, device):
    """Drop each layer independently; return list of ΔMCC per layer."""
    print("\n── Experiment 1: single-layer drop ──")
    results = []
    for li in range(num_layers):
        saved = _zero_layer(model, pattern, li)
        X = extract_embeddings(model, tokenizer, test_texts,
                               max_length, batch_size, device)
        m = eval_probe(clf, X, test_labels)
        delta = m["mcc"] - baseline_mcc
        results.append({"layer": li, "mcc": m["mcc"], "delta_mcc": delta,
                         "accuracy": m["accuracy"]})
        _restore_layer(model, pattern, li, saved)
        print(f"  L{li:2d}: mcc={m['mcc']:.4f}  Δmcc={delta:+.4f}")
    return results


def cumulative_drop(model, tokenizer, pattern, num_layers,
                    test_texts, test_labels, clf, baseline_mcc,
                    drop_order, order_name,
                    max_length, batch_size, device):
    """
    Drop layers in `drop_order` one by one; restore nothing between steps
    (layers stay zeroed). Returns list of cumulative results.
    """
    print(f"\n── Experiment 2: cumulative drop ({order_name}) ──")
    results = []
    saves = {}
    dropped = []
    for li in drop_order:
        saves[li] = _zero_layer(model, pattern, li)
        dropped.append(li)
        X = extract_embeddings(model, tokenizer, test_texts,
                               max_length, batch_size, device)
        m = eval_probe(clf, X, test_labels)
        delta = m["mcc"] - baseline_mcc
        results.append({"n_dropped": len(dropped), "layers_dropped": list(dropped),
                         "mcc": m["mcc"], "delta_mcc": delta,
                         "accuracy": m["accuracy"]})
        print(f"  n={len(dropped):2d}  last_dropped=L{li:2d}  "
              f"mcc={m['mcc']:.4f}  Δmcc={delta:+.4f}")
    # Restore all
    for li, saved in saves.items():
        _restore_layer(model, pattern, li, saved)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(single, cumul_far, cumul_imp, sw_layer,
                 baseline_mcc, out_png, task_tag):

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── Panel 1: per-layer importance bar chart ───────────────────────────────
    ax = axes[0]
    layers = [r["layer"] for r in single]
    deltas = [r["delta_mcc"] for r in single]
    colors = ["tomato" if li == sw_layer else
              ("steelblue" if d > -0.05 else "darkorange")
              for li, d in zip(layers, deltas)]
    bars = ax.bar(layers, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(sw_layer, color="tomato", lw=1.5, ls="--", alpha=0.7,
               label=f"SW layer (L{sw_layer})")
    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel("ΔMCC (single-layer drop)", fontsize=11)
    ax.set_title("Per-layer importance\n(lower bar = more critical)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: cumulative drop curves ──────────────────────────────────────
    ax = axes[1]
    ax.axhline(baseline_mcc, color="black", lw=1, ls="-", label="Baseline", zorder=1)

    xs_far = [r["n_dropped"] for r in cumul_far]
    ys_far = [r["mcc"]       for r in cumul_far]
    ax.plot(xs_far, ys_far, color="seagreen", lw=2, marker="o", ms=4,
            label="Furthest-from-L4 first")

    xs_imp = [r["n_dropped"] for r in cumul_imp]
    ys_imp = [r["mcc"]       for r in cumul_imp]
    ax.plot(xs_imp, ys_imp, color="steelblue", lw=2, marker="s", ms=4,
            label="Least-important first")

    # Mark where L4 falls in each order
    for res, color, name in [(cumul_far, "seagreen", "far"),
                              (cumul_imp, "steelblue", "imp")]:
        for r in res:
            if sw_layer in r["layers_dropped"] and sw_layer == r["layers_dropped"][-1]:
                ax.axvline(r["n_dropped"], color=color, lw=1.2, ls=":",
                           label=f"L4 added ({name})")
                break

    ax.set_xlabel("Number of layers dropped", fontsize=11)
    ax.set_ylabel("MCC", fontsize=11)
    ax.set_title("Cumulative layer drop\n(layers stay zeroed)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 3: L4 first vs L4 last ─────────────────────────────────────────
    ax = axes[2]
    ax.axhline(baseline_mcc, color="black", lw=1, ls="-", label="Baseline")

    # "L4 first" = cumul_imp with L4 as the first dropped (find that order)
    # We don't have a dedicated run for this — use single_layer L4 as 1-layer point
    # and mark it, then show cumul_imp (least-imp first) which drops L4 last.
    l4_single = next(r for r in single if r["layer"] == sw_layer)
    ax.scatter([1], [l4_single["mcc"]], color="tomato", zorder=5, s=80,
               label=f"Drop L4 alone: MCC={l4_single['mcc']:.3f}")

    xs_imp = [r["n_dropped"] for r in cumul_imp]
    ys_imp = [r["mcc"]       for r in cumul_imp]
    ax.plot(xs_imp, ys_imp, color="steelblue", lw=2, marker="s", ms=4,
            label="Others dropped first (L4 last)")

    # Horizontal line at L4-alone MCC
    ax.axhline(l4_single["mcc"], color="tomato", lw=1.2, ls="--", alpha=0.7)

    ax.set_xlabel("Number of layers dropped", fontsize=11)
    ax.set_ylabel("MCC", fontsize=11)
    ax.set_title(f"L4 isolation\nL4 alone vs L4 last", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"GENERator layer-drop profiling — {task_tag}\n"
        f"Baseline MCC={baseline_mcc:.4f}  |  SW layer=L{sw_layer}  |  "
        f"Red=SW layer, Orange=moderately critical, Blue=safe",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GENERator layer-drop importance profiling")
    p.add_argument("--model_name",   default="GenerTeam/GENERator-v2-eukaryote-3b-base")
    p.add_argument("--dataset_name", default="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    p.add_argument("--subset_name",  default="H3K4me3")
    p.add_argument("--sw_index",     default="results/super_weight_index.json")
    p.add_argument("--sw_key",       default="generator")
    p.add_argument("--down_proj_pattern", default="layers.{i}.mlp.down_proj")
    p.add_argument("--num_layers",   type=int, default=30)
    p.add_argument("--max_length",   type=int, default=512)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--hf_cache",     default=None)
    p.add_argument("--out",          default=None)
    p.add_argument("--plot",         default=None)
    return p.parse_args()


def main():
    args = parse_args()
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.hf_cache:
        os.environ["HF_HOME"]            = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    task_tag = args.subset_name or args.dataset_name.split("/")[-1]
    out_json = args.out  or f"results/layer_drop_generator_{task_tag}.json"
    out_png  = args.plot or f"results/layer_drop_generator_{task_tag}.png"

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_dtype = (torch.bfloat16
                  if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
                  else torch.float32)
    print(f"Device: {device}  dtype: {load_dtype}")

    # ── Tokenizer + model ─────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading backbone: {args.model_name}")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        args.model_name, dtype=load_dtype, trust_remote_code=True
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    print(f"  Loaded in {time.time()-t0:.1f}s  |  "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # ── Dataset ───────────────────────────────────────────────────────────────
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

    train_texts  = ds["train"]["sequence"]
    train_labels = np.array(ds["train"]["label"], dtype=int)
    test_texts   = ds["test"]["sequence"]
    test_labels  = np.array(ds["test"]["label"], dtype=int)
    print(f"  train={len(train_texts)}  test={len(test_texts)}")

    # ── SW layer ──────────────────────────────────────────────────────────────
    sw_index = json.loads(Path(args.sw_index).read_text())
    entry    = sw_index.get(args.sw_key, {})
    sw_list  = entry.get("results", []) if isinstance(entry, dict) else entry
    sw_layer = sw_list[0]["layer"] if sw_list else 4
    print(f"  SW layer: L{sw_layer}")

    # ── Baseline probe ────────────────────────────────────────────────────────
    print("\n── Extracting train embeddings (baseline) ──")
    X_train = extract_embeddings(model, tokenizer, train_texts,
                                 args.max_length, args.batch_size, device, verbose=True)
    print("── Training probe ──")
    clf = train_probe(X_train, train_labels, args.seed)

    print("── Extracting test embeddings (baseline) ──")
    X_test_base = extract_embeddings(model, tokenizer, test_texts,
                                     args.max_length, args.batch_size, device, verbose=True)
    baseline = eval_probe(clf, X_test_base, test_labels)
    print(f"  Baseline: acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")

    # ── Experiment 1: single-layer drop ───────────────────────────────────────
    single = single_layer_drop(
        model, tokenizer, args.down_proj_pattern, args.num_layers,
        test_texts, test_labels, clf, baseline["mcc"],
        args.max_length, args.batch_size, device,
    )

    # ── Experiment 2a: cumulative — furthest from L4 first ───────────────────
    order_far = sorted(range(args.num_layers), key=lambda li: -abs(li - sw_layer))
    cumul_far = cumulative_drop(
        model, tokenizer, args.down_proj_pattern, args.num_layers,
        test_texts, test_labels, clf, baseline["mcc"],
        order_far, "furthest-from-L4 first",
        args.max_length, args.batch_size, device,
    )

    # ── Experiment 2b: cumulative — least important (by single-drop ΔMCC) first
    order_imp = [r["layer"] for r in sorted(single, key=lambda r: r["delta_mcc"])]
    # Most negative ΔMCC = most critical → put those LAST
    order_imp = list(reversed(order_imp))  # least-important (closest to 0) first
    cumul_imp = cumulative_drop(
        model, tokenizer, args.down_proj_pattern, args.num_layers,
        test_texts, test_labels, clf, baseline["mcc"],
        order_imp, "least-important first",
        args.max_length, args.batch_size, device,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "model":       args.model_name,
        "dataset":     args.dataset_name,
        "subset":      args.subset_name,
        "max_length":  args.max_length,
        "sw_layer":    sw_layer,
        "baseline":    baseline,
        "single_drop": single,
        "cumul_far":   cumul_far,
        "cumul_imp":   cumul_imp,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(single, cumul_far, cumul_imp, sw_layer,
                 baseline["mcc"], out_png, task_tag)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Baseline MCC : {baseline['mcc']:.4f}")
    l4_res = next(r for r in single if r["layer"] == sw_layer)
    print(f"L{sw_layer} alone  MCC : {l4_res['mcc']:.4f}  "
          f"(Δ={l4_res['delta_mcc']:+.4f})")
    safest = max(single, key=lambda r: r["delta_mcc"])
    print(f"Safest layer : L{safest['layer']}  "
          f"(Δ={safest['delta_mcc']:+.4f})")
    most_critical = min(single, key=lambda r: r["delta_mcc"])
    print(f"Most critical: L{most_critical['layer']}  "
          f"(Δ={most_critical['delta_mcc']:+.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
