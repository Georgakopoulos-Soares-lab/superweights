"""
scripts/run_compression_sweep.py
---------------------------------
Progressive row-pruning sweep for DNABERT-2 mlp.wo (down-proj rows).

For several ranking criteria, removes an increasing fraction of rows and
measures accuracy + MCC on a GUE task test split.  Results form a
pruning sensitivity curve: how many rows can you remove before accuracy
degrades significantly?

Criteria
--------
  random    : uniform random order — 10 seeds, mean ± std reported
  l1_low    : ascending row-L1-norm (smallest = "least important" by magnitude)
  l1_high   : descending row-L1-norm (largest first — sanity / upper-bound check)
  prox_far  : furthest from any superrow first (remove the "irrelevant" part)
  prox_near : closest to any superrow first  (remove the SW neighbourhood)

Fractions swept (% of total rows, excluding the SW rows themselves):
  default: 0.5, 1, 2, 5, 10, 15, 20, 30

Total rows for DNABERT-2 mlp.wo: 12 layers × 768 = 9 216
  1 % ≈  92 rows   5 % ≈ 460 rows   15 % ≈ 1 382 rows

Run from repo root:
    python scripts/run_compression_sweep.py \\
        --model dnabert2 \\
        --task prom/prom_core_notata \\
        --gue_root /work/11034/atzanakak/GUE/GUE \\
        --ckpt_dir results/gue_checkpoints/dnabert2_prom_core_notata

Optional:
    --fracs 1 5 10 20        # override default sweep points (integers or floats, in %)
    --n_rand_seeds 10        # seeds for the random criterion (default 10)
    --sw_index results/super_weight_index.json
    --out    results/compression_sweep_dnabert2_prom_core_notata.json
    --plot   results/compression_sweep_dnabert2_prom_core_notata.png
"""

import argparse
import json
import random
import sys
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Shared helpers from run_gue_ablation ─────────────────────────────────────
# Import the pieces we need rather than re-implementing them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from run_gue_ablation import (
    GUEDataset,
    _NTv3Classifier,
    _HybriDNAClassifier,
    _GeneratorClassifier,
    _resolve_module,
    _save_row,
    _zero_row,
    _restore_row,
    evaluate,
    _task_key,
    _MAX_LEN,
    collate_fn,
)
# transformers imported lazily inside non-EVO2 code paths for evo2-container compat


# ── Ranking helpers ───────────────────────────────────────────────────────────

def _all_candidates(model, pattern, num_layers, sw_coords):
    """Return list of (layer, row) for every row except the SW rows."""
    cands = []
    for li in range(num_layers):
        m = _resolve_module(model, pattern, li)
        nrows = m.weight.data.shape[0]
        for ri in range(nrows):
            if (li, ri) not in sw_coords:
                cands.append((li, ri))
    return cands


def _rank_l1(model, pattern, candidates):
    """Rank candidates by ascending row L1-norm (smallest = prune first)."""
    scored = []
    for li, ri in candidates:
        m = _resolve_module(model, pattern, li)
        score = float(m.weight.data[ri].abs().sum().item())
        scored.append((score, li, ri))
    scored.sort(key=lambda x: x[0])
    return [(li, ri) for _, li, ri in scored]


def _rank_proximity(model, pattern, candidates, sw_list, num_layers, num_rows,
                    ascending=True):
    """
    Rank candidates by distance to the nearest superrow in normalised
    (layer, row) space.  ascending=True → furthest first (prune_far).
    ascending=False → closest first (prune_near).
    """
    sw_pts = np.array([[sw["layer"] / (num_layers - 1),
                        sw["row"]   / (num_rows   - 1)]
                       for sw in sw_list], dtype=float)

    scored = []
    for li, ri in candidates:
        pt = np.array([li / (num_layers - 1), ri / (num_rows - 1)], dtype=float)
        dist_to_nearest = float(np.min(np.linalg.norm(sw_pts - pt, axis=1)))
        scored.append((dist_to_nearest, li, ri))

    scored.sort(key=lambda x: x[0], reverse=ascending)  # far-first → reverse=True
    return [(li, ri) for _, li, ri in scored]


# ── Pruning evaluation at a single fraction ───────────────────────────────────

def _eval_pruned(model, pattern, ranked_rows, n_prune, test_ds, device):
    """Zero the first n_prune rows from ranked_rows, evaluate, restore."""
    chosen = ranked_rows[:n_prune]
    saves  = [(l, r, _save_row(model, pattern, l, r)) for l, r in chosen]
    for l, r in chosen:
        _zero_row(model, pattern, l, r)
    m = evaluate(model, test_ds, device=device)
    for l, r, saved in saves:
        _restore_row(model, pattern, l, r, saved)
    return m


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(model, sw_list, test_ds, pattern, num_layers, num_rows,
              fracs, n_rand_seeds, device):
    """
    Returns dict mapping criterion name → list of result dicts
    (one per fraction).
    """
    sw_coords  = {(sw["layer"], sw["row"]) for sw in sw_list}
    candidates = _all_candidates(model, pattern, num_layers, sw_coords)
    n_total    = len(candidates)

    baseline = evaluate(model, test_ds, device=device)
    print(f"  Baseline: acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")
    print(f"  Candidate pool: {n_total} rows  (total {num_layers*num_rows} - {len(sw_coords)} SW)")

    # Pre-rank for deterministic criteria
    print("  Ranking by L1-norm ...", flush=True)
    ranked_l1_low  = _rank_l1(model, pattern, candidates)
    ranked_l1_high = list(reversed(ranked_l1_low))

    print("  Ranking by proximity to superrows ...", flush=True)
    ranked_prox_far  = _rank_proximity(model, pattern, candidates, sw_list,
                                       num_layers, num_rows, ascending=True)
    ranked_prox_near = _rank_proximity(model, pattern, candidates, sw_list,
                                       num_layers, num_rows, ascending=False)

    results = {
        "baseline":   baseline,
        "n_total":    n_total,
        "num_layers": num_layers,
        "num_rows":   num_rows,
        "n_sw":       len(sw_list),
        "fracs":      fracs,
        "curves":     {},
    }

    # ── SW-only condition ─────────────────────────────────────────────────────
    # Remove ONLY the detected superweight rows — should cause a disproportionate
    # accuracy drop relative to the small number of rows removed.
    if sw_list:
        print(f"\n  Criterion: sw_only  (n={len(sw_list)} SW rows zeroed)")
        sw_saves = [(sw["layer"], sw["row"],
                     _save_row(model, pattern, sw["layer"], sw["row"]))
                    for sw in sw_list]
        for sw in sw_list:
            _zero_row(model, pattern, sw["layer"], sw["row"])
        sw_only_m = evaluate(model, test_ds, device=device)
        for layer, row, saved in sw_saves:
            _restore_row(model, pattern, layer, row, saved)
        sw_frac = len(sw_list) / n_total * 100 if n_total > 0 else 0.0
        delta_acc = (sw_only_m["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100
        delta_mcc = (sw_only_m["mcc"] - baseline["mcc"]) / max(abs(baseline["mcc"]), 1e-9) * 100
        print(f"    acc={sw_only_m['accuracy']:.4f} ({delta_acc:+.2f}%)  "
              f"mcc={sw_only_m['mcc']:.4f} ({delta_mcc:+.2f}%)  "
              f"frac={sw_frac:.3f}%")
        results["sw_only"] = {
            "accuracy": sw_only_m["accuracy"], "mcc": sw_only_m["mcc"],
            "delta_acc_pct": delta_acc, "delta_mcc_pct": delta_mcc,
            "n_sw": len(sw_list), "frac_pct": sw_frac,
        }
    else:
        print("\n  [skip] sw_only — no superweight rows defined.")
        results["sw_only"] = None

    criteria = {
        "l1_low":    ranked_l1_low,
        "l1_high":   ranked_l1_high,
        "prox_far":  ranked_prox_far,
        "prox_near": ranked_prox_near,
    }

    for crit, ranked in criteria.items():
        print(f"\n  Criterion: {crit}")
        curve = []
        for frac in fracs:
            n_prune = max(1, int(round(n_total * frac / 100)))
            m = _eval_pruned(model, pattern, ranked, n_prune, test_ds, device)
            delta_acc = (m["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100
            delta_mcc = (m["mcc"] - baseline["mcc"]) / max(abs(baseline["mcc"]), 1e-9) * 100
            print(f"    {frac:5.1f}%  n={n_prune:5d}  "
                  f"acc={m['accuracy']:.4f} ({delta_acc:+.2f}%)  "
                  f"mcc={m['mcc']:.4f} ({delta_mcc:+.2f}%)")
            curve.append({
                "frac": frac, "n_pruned": n_prune,
                "accuracy": m["accuracy"], "mcc": m["mcc"],
                "delta_acc_pct": delta_acc, "delta_mcc_pct": delta_mcc,
            })
        results["curves"][crit] = curve

    # Random criterion: multiple seeds
    print(f"\n  Criterion: random  ({n_rand_seeds} seeds)")
    rng = random.Random(42)
    rand_curve = []
    for frac in fracs:
        n_prune  = max(1, int(round(n_total * frac / 100)))
        seed_acc = []
        seed_mcc = []
        for _ in range(n_rand_seeds):
            ranked_rand = rng.sample(candidates, len(candidates))
            m = _eval_pruned(model, pattern, ranked_rand, n_prune, test_ds, device)
            seed_acc.append(m["accuracy"])
            seed_mcc.append(m["mcc"])
        mean_acc = float(np.mean(seed_acc))
        std_acc  = float(np.std(seed_acc))
        mean_mcc = float(np.mean(seed_mcc))
        std_mcc  = float(np.std(seed_mcc))
        delta_acc = (mean_acc - baseline["accuracy"]) / baseline["accuracy"] * 100
        delta_mcc = (mean_mcc - baseline["mcc"]) / max(abs(baseline["mcc"]), 1e-9) * 100
        print(f"    {frac:5.1f}%  n={n_prune:5d}  "
              f"acc={mean_acc:.4f}±{std_acc:.4f} ({delta_acc:+.2f}%)  "
              f"mcc={mean_mcc:.4f}±{std_mcc:.4f} ({delta_mcc:+.2f}%)")
        rand_curve.append({
            "frac": frac, "n_pruned": n_prune,
            "accuracy_mean": mean_acc, "accuracy_std": std_acc,
            "mcc_mean": mean_mcc,      "mcc_std":      std_mcc,
            "delta_acc_pct": delta_acc, "delta_mcc_pct": delta_mcc,
        })
    results["curves"]["random"] = rand_curve

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sweep(results, baseline, sw_list, out_png, task, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves  = results["curves"]
    fracs   = results["fracs"]
    bl_acc  = baseline["accuracy"]
    bl_mcc  = baseline["mcc"]

    STYLE = {
        "l1_low":    dict(color="steelblue",  lw=2,   ls="-",  label="L1-low (magnitude prune)"),
        "l1_high":   dict(color="royalblue",  lw=1.5, ls="--", label="L1-high (largest first)"),
        "prox_far":  dict(color="seagreen",   lw=2,   ls="-",  label="Prox-far (furthest from SW first)"),
        "prox_near": dict(color="tomato",     lw=2,   ls="-",  label="Prox-near (closest to SW first)"),
        "random":    dict(color="grey",       lw=2,   ls=":",  label="Random (mean ± 1 std)"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for metric, ax, bl_val, ylabel in [
        ("accuracy", axes[0], bl_acc, "Accuracy"),
        ("mcc",      axes[1], bl_mcc, "MCC"),
    ]:
        ax.axhline(bl_val, color="black", lw=1.2, ls="-", label="Baseline", zorder=1)

        for crit, style in STYLE.items():
            c = curves[crit]
            if crit == "random":
                means = [p[f"{metric}_mean"] for p in c]
                stds  = [p[f"{metric}_std"]  for p in c]
                xs    = [p["frac"] for p in c]
                ax.plot(xs, means, **{k: v for k, v in style.items() if k != "label"},
                        label=style["label"], zorder=3)
                ax.fill_between(xs,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                color=style["color"], alpha=0.15, zorder=2)
            else:
                xs = [p["frac"] for p in c]
                ys = [p[metric]  for p in c]
                ax.plot(xs, ys, **{k: v for k, v in style.items() if k != "label"},
                        label=style["label"], zorder=3)

        # SW-only marker: a special star showing the drop from removing ONLY the SW rows
        sw_only = results.get("sw_only")
        if sw_only is not None:
            sw_frac = sw_only.get("frac_pct", sw_only.get("n_sw", 0) / max(results["n_total"], 1) * 100)
            sw_val  = sw_only[metric]
            ax.scatter([sw_frac], [sw_val], marker="*", s=300, color="crimson",
                       zorder=5, label=f"SW-only (n={sw_only['n_sw']})", edgecolors="darkred", linewidths=0.8)
            ax.annotate(
                f"SW-only\n{sw_val:.3f}",
                xy=(sw_frac, sw_val),
                xytext=(sw_frac * 1.8, sw_val - (bl_val - sw_val) * 0.25),
                fontsize=8, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.0),
            )

        ax.set_xlabel("Rows pruned (% of non-SW pool)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} vs pruning fraction\n{model_name} · {task}", fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks(fracs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="lower left")

    plt.suptitle(
        f"Compression sweep — {model_name} / {task}\n"
        f"Baseline acc={bl_acc:.4f}  mcc={bl_mcc:.4f}  |  "
        f"{results['n_total']} candidate rows ({results['num_layers']}L × {results['num_rows']}R)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_png}")


# ── CLI ───────────────────────────────────────────────────────────────────────

_CAUSAL_MODELS = {"generator", "generator_prokaryote", "generator_prokaryote_1b", "hybridna"}
_EVO2_MODELS   = {"evo2"}

def _truncate_to_multiple(texts, factor: int):
    """Left-truncate each string so its length is divisible by `factor`."""
    out = []
    for t in texts:
        r = len(t) % factor
        out.append(t[r:] if r else t)
    return out


def _load_model_for_sweep(args, config, test_ds, ckpt_dir):
    """Load model (+ tokenizer) appropriate for the sweep, return (model, tokenizer)."""
    import os
    hf_token  = args.hf_token or os.environ.get("HF_TOKEN")
    tok_kwargs = {"trust_remote_code": True} if config.get("hf_trust_remote_code") else {}
    if hf_token:
        tok_kwargs["token"] = hf_token
    if "zhihan1996" in config["model_id"]:
        tok_kwargs["revision"] = "7bce263b15377fc15361f52cfab88f8b586abda0"

    model_id = config["model_id"]
    is_causal = args.model in _CAUSAL_MODELS
    is_evo2   = args.model in _EVO2_MODELS

    print(f"Loading tokenizer ...")
    if is_evo2:
        # EVO2 uses a custom package loader — try importing evo2 package
        try:
            from evo2 import Evo2
        except ImportError:
            raise ImportError(
                "The 'evo2' package is not installed in this environment. "
                "Activate the evo2 environment (e.g. source evo2_env/bin/activate) "
                "before running this script with --model evo2."
            )
        print("Loading EVO2 model ...")
        evo2_obj = Evo2(model_id)
        backbone = evo2_obj.model
        # Resolve EVO2 tokenizer
        tokenizer = None
        for attr in ("tokenizer", "tok", "token_encoder"):
            if hasattr(evo2_obj, attr):
                tokenizer = getattr(evo2_obj, attr)
                break
        if tokenizer is None:
            # Fallback: use char-level ord() encoding via a lambda
            class _OrdTokenizer:
                pad_token_id = 0
                def __call__(self, texts, return_tensors=None, padding=None,
                             max_length=None, truncation=None, return_attention_mask=None, **kw):
                    import torch as _torch
                    if isinstance(texts[0], str):
                        seqs = [t[:max_length] if max_length else t for t in texts]
                    else:
                        seqs = [t[:max_length] if max_length else t for t in texts]
                    ids = [[ord(c) for c in s] for s in seqs]
                    L = max(len(x) for x in ids)
                    padded = [x + [0] * (L - len(x)) for x in ids]
                    masks  = [[1] * len(x) + [0] * (L - len(x)) for x in ids]
                    return {
                        "input_ids":      _torch.tensor(padded, dtype=_torch.long),
                        "attention_mask": _torch.tensor(masks, dtype=_torch.long),
                    }
            tokenizer = _OrdTokenizer()
        hidden_size = getattr(backbone.config if hasattr(backbone, "config") else backbone,
                              "hidden_size", config.get("hidden_dim", 4096))
        ckpt_state = Path(ckpt_dir) / "model_state.pt"
        model = _HybriDNAClassifier(backbone, hidden_size, test_ds.num_labels)
        if ckpt_state.exists():
            print(f"Loading fine-tuned weights from {ckpt_state} ...")
            model.load_state_dict(torch.load(ckpt_state, map_location="cpu"), strict=False)
        else:
            print("[warn] No checkpoint found — evaluating base model weights.")
        return model, tokenizer

    elif is_causal:
        import transformers  # noqa: PLC0415
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, model_max_length=args.max_length or 512,
            padding_side="right", **tok_kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loading causal LM backbone from {model_id} ...")
        backbone = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            **tok_kwargs,
        )
        hidden_size = backbone.config.hidden_size
        ckpt_state  = Path(ckpt_dir) / "model_state.pt"
        model = _GeneratorClassifier(backbone, hidden_size, test_ds.num_labels)
        if ckpt_state.exists():
            print(f"Loading fine-tuned weights from {ckpt_state} ...")
            model.load_state_dict(torch.load(ckpt_state, map_location="cpu"), strict=False)
        else:
            print("[warn] No checkpoint found — evaluating base model weights.")
        return model, tokenizer

    else:
        # DNABERT-2, NTv3, HybriDNA (existing path)
        import transformers  # noqa: PLC0415
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, model_max_length=args.max_length or 512, **tok_kwargs
        )
        ckpt_state = Path(ckpt_dir) / "model_state.pt"
        print(f"Loading model from {model_id} ...")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=test_ds.num_labels, **tok_kwargs
        )
        if ckpt_state.exists():
            print(f"Loading fine-tuned weights from {ckpt_state} ...")
            model.load_state_dict(torch.load(ckpt_state, map_location="cpu"), strict=False)
        else:
            print("[warn] No checkpoint found — evaluating base model weights.")
        model = model.to(args.device)
        return model, tokenizer


def main():
    _ALL_MODELS = ["dnabert2", "ntv3", "hybridna",
                   "generator", "generator_prokaryote", "generator_prokaryote_1b",
                   "evo2"]
    parser = argparse.ArgumentParser(description="Progressive row-pruning sweep")
    parser.add_argument("--model",    required=True, choices=_ALL_MODELS)
    parser.add_argument("--task",     required=True)
    parser.add_argument("--gue_root", required=True)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--sw_index", default="results/super_weight_index.json")
    parser.add_argument("--evo2_sw_source", default=None,
                        help="For evo2: path to a JSON with candidate SW rows "
                             "(e.g. results/hydra_test_evo2.json). "
                             "Rounds are read as SW candidates in layer 29.")
    parser.add_argument("--out",      default=None,
                        help="Output JSON path (default: results/compression_sweep_<model>_<task_leaf>.json)")
    parser.add_argument("--plot",     default=None,
                        help="Output PNG path (default: results/compression_sweep_<model>_<task_leaf>.png)")
    parser.add_argument("--fracs",    nargs="+", type=float,
                        default=[0.5, 1, 2, 5, 10, 15, 20, 30],
                        help="Pruning fractions in %% of candidate pool (default: 0.5 1 2 5 10 15 20 30)")
    parser.add_argument("--n_rand_seeds", type=int, default=10)
    parser.add_argument("--max_length",   type=int, default=None)
    parser.add_argument("--hf_token",     default=None)
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    task_leaf = args.task.split("/")[-1]
    ckpt_dir  = args.ckpt_dir or f"results/gue_checkpoints/{args.model}_{task_leaf}"
    out_json  = args.out  or f"results/compression_sweep_{args.model}_{task_leaf}.json"
    out_png   = args.plot or f"results/compression_sweep_{args.model}_{task_leaf}.png"

    config     = yaml.safe_load(open(f"configs/{args.model}.yaml"))
    pattern    = config["down_proj_pattern"]
    num_layers = config["num_layers"]
    tkey       = _task_key(args.task)
    max_length = args.max_length or _MAX_LEN.get(tkey, 512)

    # For Generator models with 6-mer tokenizer, round max_length down to multiple of 6
    if args.model in _CAUSAL_MODELS and "generator" in args.model:
        max_length = (max_length // 6) * 6 or 6

    print(f"\n{'='*64}")
    print(f"  Compression sweep: {args.model} / {args.task}")
    print(f"  Fractions: {args.fracs} %")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"{'='*64}\n")

    # ── Test dataset (preliminary — we need the tokenizer first) ─────────────
    # For EVO2 and generator, we pre-load the tokenizer via _load_model_for_sweep.
    # Build a temporary dataset with a placeholder tokenizer to get num_labels.
    import os
    hf_token   = args.hf_token or os.environ.get("HF_TOKEN")
    tok_kwargs = {"trust_remote_code": True} if config.get("hf_trust_remote_code") else {}
    if hf_token:
        tok_kwargs["token"] = hf_token
    if "zhihan1996" in config["model_id"]:
        tok_kwargs["revision"] = "7bce263b15377fc15361f52cfab88f8b586abda0"

    if args.model not in _EVO2_MODELS:
        import transformers  # noqa: PLC0415 — lazy import for evo2-container compat
        print("Loading tokenizer ...")
        if args.model in _CAUSAL_MODELS:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                config["model_id"], model_max_length=max_length,
                padding_side="right", **tok_kwargs
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                config["model_id"], model_max_length=max_length, **tok_kwargs
            )
        print("Loading test split ...")
        test_ds = GUEDataset(f"{args.gue_root}/{args.task}/test.csv", tokenizer, max_length)
    else:
        # EVO2: defer tokenizer to _load_model_for_sweep; load dataset after model
        tokenizer = None
        test_ds   = None

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = _load_model_for_sweep(args, config, test_ds, ckpt_dir)

    if test_ds is None:
        # EVO2 tokenizer now available; build dataset
        print("Loading test split (EVO2 tokenizer) ...")
        test_ds = GUEDataset(f"{args.gue_root}/{args.task}/test.csv", tokenizer, max_length)

    if args.model not in _EVO2_MODELS and args.model not in _CAUSAL_MODELS:
        model = model.to(args.device)
    model.eval()
    print(f"  test={len(test_ds)}")

    # ── Number of rows per layer ──────────────────────────────────────────────
    num_rows = _resolve_module(model, pattern, 0).weight.data.shape[0]

    # ── SW list ───────────────────────────────────────────────────────────────
    sw_index = json.loads(Path(args.sw_index).read_text())
    entry    = sw_index.get(args.model, {})
    sw_list  = entry.get("results", []) if isinstance(entry, dict) else entry

    # For EVO2: optionally override SW list from the hydra/detection JSON
    if args.model in _EVO2_MODELS and args.evo2_sw_source:
        src = json.loads(Path(args.evo2_sw_source).read_text())
        # hydra_test_evo2.json has "rounds" list with layer/row/col dicts
        rounds  = src.get("rounds", [])
        sw_list = [{"layer": r["layer"], "row": r["row"]} for r in rounds]
        print(f"EVO2: using {len(sw_list)} potential SW rows from {args.evo2_sw_source}")
    elif args.model in _EVO2_MODELS and not sw_list:
        print("[warn] No EVO2 SW rows in index. Pass --evo2_sw_source to specify potential SWs.")

    print(f"\nSuperrows ({len(sw_list)}):")
    for sw in sw_list:
        print(f"  layer={sw['layer']}  row={sw['row']}")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    results = run_sweep(
        model, sw_list, test_ds, pattern, num_layers, num_rows,
        fracs=args.fracs, n_rand_seeds=args.n_rand_seeds, device=args.device,
    )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    payload = {
        "model": args.model, "task": args.task,
        "ckpt_dir": ckpt_dir, "max_length": max_length,
        **results,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(payload, indent=2))
    print(f"Results saved → {out_json}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_sweep(results, results["baseline"], sw_list, out_png, args.task, args.model)


if __name__ == "__main__":
    main()
