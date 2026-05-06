"""
scripts/run_compression_sweep_generator.py
-------------------------------------------
Progressive row-pruning compression sweep for GENERator (causal decoder).

Unlike the DNABERT-2 sweep (which has 10 distributed superrows and uses a
fine-tuned classification head), GENERator has exactly ONE dominant superrow
(L4, r2371) whose ablation collapses any downstream task to chance level.
This script asks: how much of the *rest* of the network can we safely prune
while keeping representation quality high?

Approach
--------
  1. Load backbone with AutoModel (returns bare LlamaModel).
  2. Extract train embeddings once → train a logistic regression probe.
  3. For each (criterion, fraction): zero the chosen rows in down_proj across
     all layers (the SW row is NEVER touched), re-extract test embeddings only,
     evaluate with the fixed probe, restore weights.

Criteria
--------
  l1_low       : ascending row-L1-norm — "least important by magnitude" first
  l1_high      : descending row-L1-norm — sanity / upper-bound damage check
  prox_near    : closest to SW in normalised (layer, row) 2-D space — prune
                 the SW neighbourhood first
  prox_far     : furthest from SW first — prune the peripheral "irrelevant" rows
  same_layer   : all other rows in the SW layer (L4) first, then remaining
                 layers by ascending distance from L4 — tests layer-level effects
  layer_dist   : rows in layers farthest from SW layer first, SW layer last
                 (row order within each layer: ascending L1) — tests whether
                 layer proximity to the SW matters
  random       : 10 seeds, mean ± std reported

Fractions swept (% of non-SW candidate pool):
  default: 0.5, 1, 2, 5, 10, 15, 20, 30

Run from repo root:
    python scripts/run_compression_sweep_generator.py \\
        --dataset_name "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised" \\
        --subset_name  H3K4me3 \\
        --out   results/compression_sweep_generator_H3K4me3.json \\
        --plot  results/compression_sweep_generator_H3K4me3.png

All CLI flags have sensible defaults matching the GENERator 3B setup.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from transformers import AutoModel, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Module helpers (mirror probe_ablation.py, no import dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_module(model, pattern, layer_idx):
    path = pattern.replace("{i}", str(layer_idx))
    m = model
    for attr in path.split("."):
        m = getattr(m, attr)
    return m


def _zero_rows(model, pattern, layer_idx, rows):
    m = _resolve_module(model, pattern, layer_idx)
    saved = m.weight.data[rows, :].clone()
    m.weight.data[rows, :] = 0.0
    return saved


def _restore_rows(model, pattern, layer_idx, rows, saved):
    m = _resolve_module(model, pattern, layer_idx)
    m.weight.data[rows, :] = saved


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, tokenizer, texts, max_length, batch_size, device,
                       verbose=True):
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
        hidden = out.hidden_states[-1]           # (B, L, H)
        attention_mask = enc["attention_mask"]   # (B, L)
        seq_lens = attention_mask.sum(dim=1) - 1 # (B,)
        embs = hidden[torch.arange(hidden.size(0)), seq_lens]  # (B, H)
        all_embs.append(embs.float().cpu().numpy())
        if verbose and (i // batch_size) % 5 == 0:
            print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    if verbose:
        print()
    return np.concatenate(all_embs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Probe
# ─────────────────────────────────────────────────────────────────────────────

def train_probe(X_train, y_train, seed):
    clf = LogisticRegression(max_iter=2000, random_state=seed, C=1.0)
    clf.fit(X_train, y_train)
    return clf


def eval_probe(clf, X, y):
    preds = clf.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "mcc":      float(matthews_corrcoef(y, preds)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Candidate pool
# ─────────────────────────────────────────────────────────────────────────────

def _all_candidates(model, pattern, num_layers, sw_coord):
    """All (layer, row) pairs except the single SW coord."""
    cands = []
    for li in range(num_layers):
        m = _resolve_module(model, pattern, li)
        nrows = m.weight.data.shape[0]
        for ri in range(nrows):
            if (li, ri) != sw_coord:
                cands.append((li, ri))
    return cands


# ─────────────────────────────────────────────────────────────────────────────
# Ranking criteria
# ─────────────────────────────────────────────────────────────────────────────

def rank_l1(model, pattern, candidates):
    """Ascending L1-norm (smallest first = least-magnitude)."""
    scored = []
    for li, ri in candidates:
        m = _resolve_module(model, pattern, li)
        score = float(m.weight.data[ri].abs().sum())
        scored.append((score, li, ri))
    scored.sort(key=lambda x: x[0])
    return [(li, ri) for _, li, ri in scored]


def rank_proximity(model, pattern, candidates, sw_layer, sw_row,
                   num_layers, num_rows, ascending=True):
    """
    Distance to the single SW in normalised (layer, row) space.
    ascending=True  → furthest first  (prox_far)
    ascending=False → closest first   (prox_near)
    """
    sw_pt = np.array([sw_layer / (num_layers - 1),
                      sw_row   / (num_rows   - 1)], dtype=float)
    scored = []
    for li, ri in candidates:
        pt   = np.array([li / (num_layers - 1), ri / (num_rows - 1)], dtype=float)
        dist = float(np.linalg.norm(sw_pt - pt))
        scored.append((dist, li, ri))
    scored.sort(key=lambda x: x[0], reverse=ascending)  # far-first → reverse=True
    return [(li, ri) for _, li, ri in scored]


def rank_same_layer(model, pattern, candidates, sw_layer):
    """
    SW layer rows first (ascending L1 within layer), then remaining layers
    ordered by ascending distance from sw_layer (ascending L1 within each).
    """
    by_layer = {}
    for li, ri in candidates:
        by_layer.setdefault(li, []).append(ri)

    def _layer_order(li):
        return (0 if li == sw_layer else abs(li - sw_layer))

    ordered = []
    for li in sorted(by_layer.keys(), key=_layer_order):
        rows = by_layer[li]
        m    = _resolve_module(model, pattern, li)
        rows_sorted = sorted(rows, key=lambda ri: float(m.weight.data[ri].abs().sum()))
        ordered.extend((li, ri) for ri in rows_sorted)
    return ordered


def rank_layer_dist(model, pattern, candidates, sw_layer):
    """
    Rows in layers farthest from SW layer first; within each layer, ascending L1.
    This is the complement of same_layer: periphery-first.
    """
    by_layer = {}
    for li, ri in candidates:
        by_layer.setdefault(li, []).append(ri)

    ordered = []
    for li in sorted(by_layer.keys(), key=lambda l: -abs(l - sw_layer)):
        rows = by_layer[li]
        m    = _resolve_module(model, pattern, li)
        rows_sorted = sorted(rows, key=lambda ri: float(m.weight.data[ri].abs().sum()))
        ordered.extend((li, ri) for ri in rows_sorted)
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation at a single fraction
# ─────────────────────────────────────────────────────────────────────────────

def eval_pruned(model, tokenizer, pattern, ranked_rows, n_prune,
                test_texts, test_labels, clf, max_length, batch_size, device):
    """Zero first n_prune rows, extract test embeddings, eval probe, restore."""
    chosen = ranked_rows[:n_prune]

    # Group by layer for efficient batch zeroing
    by_layer = {}
    for li, ri in chosen:
        by_layer.setdefault(li, []).append(ri)

    saves = {}
    for li, rows in by_layer.items():
        saves[li] = _zero_rows(model, pattern, li, rows)

    X_test = extract_embeddings(model, tokenizer, test_texts,
                                max_length, batch_size, device, verbose=False)
    result = eval_probe(clf, X_test, test_labels)

    for li, rows in by_layer.items():
        _restore_rows(model, pattern, li, rows, saves[li])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(model, tokenizer, sw, pattern, num_layers, num_rows,
              train_texts, train_labels, test_texts, test_labels,
              fracs, n_rand_seeds, max_length, batch_size, seed, device):

    sw_layer, sw_row = sw["layer"], sw["row"]
    sw_coord = (sw_layer, sw_row)

    # ── Baseline ─────────────────────────────────────────────────────────────
    print("\n── Extracting train embeddings ──")
    X_train = extract_embeddings(model, tokenizer, train_texts,
                                 max_length, batch_size, device)
    print("── Training probe ──")
    clf = train_probe(X_train, train_labels, seed)

    print("── Extracting test embeddings (baseline) ──")
    X_test_base = extract_embeddings(model, tokenizer, test_texts,
                                     max_length, batch_size, device)
    baseline = eval_probe(clf, X_test_base, test_labels)
    print(f"  Baseline: acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")

    # ── Candidate pool ────────────────────────────────────────────────────────
    candidates = _all_candidates(model, pattern, num_layers, sw_coord)
    n_total = len(candidates)
    print(f"  Candidate pool: {n_total} rows  "
          f"(total {num_layers * num_rows} − 1 SW)")

    # ── Pre-rank all deterministic criteria ───────────────────────────────────
    print("  Ranking by L1-norm ...", flush=True)
    ranked_l1_low  = rank_l1(model, pattern, candidates)
    ranked_l1_high = list(reversed(ranked_l1_low))

    print("  Ranking by proximity to SW ...", flush=True)
    ranked_prox_far  = rank_proximity(model, pattern, candidates,
                                      sw_layer, sw_row, num_layers, num_rows,
                                      ascending=True)
    ranked_prox_near = rank_proximity(model, pattern, candidates,
                                      sw_layer, sw_row, num_layers, num_rows,
                                      ascending=False)

    print("  Ranking by same-layer-first ...", flush=True)
    ranked_same_layer = rank_same_layer(model, pattern, candidates, sw_layer)

    print("  Ranking by layer-distance-from-SW ...", flush=True)
    ranked_layer_dist = rank_layer_dist(model, pattern, candidates, sw_layer)

    results = {
        "baseline":   baseline,
        "n_total":    n_total,
        "num_layers": num_layers,
        "num_rows":   num_rows,
        "sw":         sw,
        "fracs":      fracs,
        "curves":     {},
    }

    # ── Deterministic criteria ────────────────────────────────────────────────
    criteria = {
        "l1_low":      ranked_l1_low,
        "l1_high":     ranked_l1_high,
        "prox_far":    ranked_prox_far,
        "prox_near":   ranked_prox_near,
        "same_layer":  ranked_same_layer,
        "layer_dist":  ranked_layer_dist,
    }

    for crit, ranked in criteria.items():
        print(f"\n  Criterion: {crit}")
        curve = []
        for frac in fracs:
            n_prune = max(1, int(round(n_total * frac / 100)))
            m = eval_pruned(model, tokenizer, pattern, ranked, n_prune,
                            test_texts, test_labels, clf,
                            max_length, batch_size, device)
            da = (m["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100
            dm = (m["mcc"] - baseline["mcc"]) / max(abs(baseline["mcc"]), 1e-9) * 100
            print(f"    {frac:5.1f}%  n={n_prune:5d}  "
                  f"acc={m['accuracy']:.4f} ({da:+.2f}%)  "
                  f"mcc={m['mcc']:.4f} ({dm:+.2f}%)")
            curve.append({
                "frac": frac, "n_pruned": n_prune,
                "accuracy": m["accuracy"], "mcc": m["mcc"],
                "delta_acc_pct": da, "delta_mcc_pct": dm,
            })
        results["curves"][crit] = curve

    # ── Random criterion ──────────────────────────────────────────────────────
    print(f"\n  Criterion: random  ({n_rand_seeds} seeds)")
    rng = random.Random(seed)
    rand_curve = []
    for frac in fracs:
        n_prune  = max(1, int(round(n_total * frac / 100)))
        seed_acc, seed_mcc = [], []
        for _ in range(n_rand_seeds):
            ranked_rand = rng.sample(candidates, len(candidates))
            m = eval_pruned(model, tokenizer, pattern, ranked_rand, n_prune,
                            test_texts, test_labels, clf,
                            max_length, batch_size, device)
            seed_acc.append(m["accuracy"])
            seed_mcc.append(m["mcc"])
        mean_acc = float(np.mean(seed_acc))
        std_acc  = float(np.std(seed_acc))
        mean_mcc = float(np.mean(seed_mcc))
        std_mcc  = float(np.std(seed_mcc))
        da = (mean_acc - baseline["accuracy"]) / baseline["accuracy"] * 100
        dm = (mean_mcc - baseline["mcc"]) / max(abs(baseline["mcc"]), 1e-9) * 100
        print(f"    {frac:5.1f}%  n={n_prune:5d}  "
              f"acc={mean_acc:.4f}±{std_acc:.4f} ({da:+.2f}%)  "
              f"mcc={mean_mcc:.4f}±{std_mcc:.4f} ({dm:+.2f}%)")
        rand_curve.append({
            "frac": frac, "n_pruned": n_prune,
            "accuracy_mean": mean_acc, "accuracy_std": std_acc,
            "mcc_mean":      mean_mcc, "mcc_std":      std_mcc,
            "delta_acc_pct": da, "delta_mcc_pct": dm,
        })
    results["curves"]["random"] = rand_curve

    return results, clf


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

STYLE = {
    "l1_low":     dict(color="steelblue",    lw=2,   ls="-",  label="L1-low (least-magnitude first)"),
    "l1_high":    dict(color="royalblue",    lw=1.5, ls="--", label="L1-high (largest first)"),
    "prox_far":   dict(color="seagreen",     lw=2,   ls="-",  label="Prox-far (furthest from SW first)"),
    "prox_near":  dict(color="tomato",       lw=2,   ls="-",  label="Prox-near (closest to SW first)"),
    "same_layer": dict(color="darkorange",   lw=2,   ls="-",  label="Same-layer (L4 rows first)"),
    "layer_dist": dict(color="mediumpurple", lw=2,   ls="-",  label="Layer-dist (farthest layer first)"),
    "random":     dict(color="grey",         lw=2,   ls=":",  label="Random (mean ± 1 std)"),
}


def plot_sweep(results, out_png, task, model_name):
    curves  = results["curves"]
    fracs   = results["fracs"]
    bl_acc  = results["baseline"]["accuracy"]
    bl_mcc  = results["baseline"]["mcc"]
    sw      = results["sw"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=False)

    for metric, ax, bl_val, ylabel in [
        ("accuracy", axes[0], bl_acc, "Accuracy"),
        ("mcc",      axes[1], bl_mcc, "MCC"),
    ]:
        ax.axhline(bl_val, color="black", lw=1.2, ls="-", label="Baseline", zorder=1)

        for crit, style in STYLE.items():
            if crit not in curves:
                continue
            c = curves[crit]
            s = {k: v for k, v in style.items() if k != "label"}
            if crit == "random":
                means = [p[f"{metric}_mean"] for p in c]
                stds  = [p[f"{metric}_std"]  for p in c]
                xs    = [p["frac"] for p in c]
                ax.plot(xs, means, **s, label=style["label"], zorder=3)
                ax.fill_between(xs,
                                [m - s2 for m, s2 in zip(means, stds)],
                                [m + s2 for m, s2 in zip(means, stds)],
                                color=style["color"], alpha=0.15, zorder=2)
            else:
                xs = [p["frac"]  for p in c]
                ys = [p[metric]  for p in c]
                ax.plot(xs, ys, **s, label=style["label"], zorder=3)

        ax.set_xlabel("Rows pruned (% of non-SW candidate pool)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} vs pruning fraction\n{model_name} · {task}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(fracs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower left")

    plt.suptitle(
        f"GENERator compression sweep — {task}\n"
        f"Baseline acc={bl_acc:.4f}  mcc={bl_mcc:.4f}  |  "
        f"SW fixed at L{sw['layer']}/r{sw['row']}  |  "
        f"{results['n_total']} candidate rows ({results['num_layers']}L × {results['num_rows']}R − 1 SW)",
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
    p = argparse.ArgumentParser(description="GENERator compression sweep (linear-probe metric)")
    p.add_argument("--model_name",   default="GenerTeam/GENERator-v2-eukaryote-3b-base")
    p.add_argument("--dataset_name", default="InstaDeepAI/nucleotide_transformer_downstream_tasks_revised")
    p.add_argument("--subset_name",  default="H3K4me3")
    p.add_argument("--sw_index",     default="results/super_weight_index.json")
    p.add_argument("--sw_key",       default="generator",
                   help="Key in sw_index for this model")
    p.add_argument("--down_proj_pattern", default="layers.{i}.mlp.down_proj",
                   help="Dotted path to down-proj module (no leading 'model.' for AutoModel)")
    p.add_argument("--num_layers",   type=int, default=30)
    p.add_argument("--max_length",   type=int, default=512)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--fracs",        nargs="+", type=float,
                   default=[0.5, 1, 2, 5, 10, 15, 20, 30],
                   help="Pruning fractions in %% of candidate pool")
    p.add_argument("--n_rand_seeds", type=int, default=10)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--hf_cache",     default=None)
    p.add_argument("--out",          default=None)
    p.add_argument("--plot",         default=None)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.hf_cache:
        os.environ["HF_HOME"]           = args.hf_cache
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    task_tag = args.subset_name or args.dataset_name.split("/")[-1]
    out_json = args.out  or f"results/compression_sweep_generator_{task_tag}.json"
    out_png  = args.plot or f"results/compression_sweep_generator_{task_tag}.png"

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # ── SW entry ──────────────────────────────────────────────────────────────
    sw_index = json.loads(Path(args.sw_index).read_text())
    entry    = sw_index.get(args.sw_key, {})
    sw_list  = entry.get("results", []) if isinstance(entry, dict) else entry
    if not sw_list:
        print(f"[error] No SW found for key '{args.sw_key}' in {args.sw_index}")
        sys.exit(1)
    sw = sw_list[0]   # single superrow for GENERator
    print(f"  Superrow: layer={sw['layer']}  row={sw['row']}  "
          f"in_max={sw['in_max']:.1f}  out_max={sw['out_max']:.1f}")

    # ── Number of rows per layer ───────────────────────────────────────────────
    num_rows = _resolve_module(model, args.down_proj_pattern, 0).weight.data.shape[0]
    print(f"  down_proj rows per layer: {num_rows}")

    # ── Sweep ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Compression sweep: GENERator / {task_tag}")
    print(f"  Fractions: {args.fracs} %")
    print(f"{'='*64}")

    results, clf = run_sweep(
        model, tokenizer, sw,
        pattern=args.down_proj_pattern,
        num_layers=args.num_layers,
        num_rows=num_rows,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        fracs=args.fracs,
        n_rand_seeds=args.n_rand_seeds,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    payload = {
        "model":        args.model_name,
        "dataset":      args.dataset_name,
        "subset":       args.subset_name,
        "max_length":   args.max_length,
        "n_rand_seeds": args.n_rand_seeds,
        **results,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved → {out_json}")

    plot_sweep(results, out_png, task_tag, "GENERator-v2-eukaryote-3b")


if __name__ == "__main__":
    main()
