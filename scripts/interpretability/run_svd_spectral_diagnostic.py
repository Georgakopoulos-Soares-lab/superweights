"""scripts/interpretability/run_svd_spectral_diagnostic.py
----------------------------------------------------------
Tests the core hypothesis for the SVD-compression paper:

  "Rare biological signals (TF binding sites, rare motifs) reside in
   low-variance singular directions of DNABERT-2's MLP down-projection
   matrices and are disproportionately destroyed by naive SVD compression."

Three experiments, all using existing GUE data + pretrained/fine-tuned DNABERT-2:

  Exp 1 — Activation spectral profile (pretrained model, no labels needed)
    For each GUE task, hook the input to every `wo` (down-proj) layer.
    Compute per-sequence projection energy onto SVD singular directions.
    Metric: spectral centroid rank — positive vs negative sequences.
    Hypothesis: binding-positive seqs have higher centroid rank
    (energy in lower-σ tail) than non-binding seqs.

  Exp 2 — SVD compression sensitivity sweep (pretrained + probing)
    Apply rank-r truncated SVD to all down-proj layers simultaneously.
    Sweep r from 100% → 5%.  Evaluate accuracy on each GUE task.
    Hypothesis: TF binding tasks degrade first (lowest knee ratio).

  Exp 3 — Gradient spectral profile (fine-tuned model required)
    For sequences from a fine-tuned TF task, compute ∂CE/∂wo per layer.
    Project gradient onto SVD directions: g_i = u_i · (∂L/∂W) · v_i.
    Hypothesis: positives have more gradient energy in low-σ directions.

Theory background:
  - FASC (arXiv) shows factual knowledge in LLMs concentrates in
    low-variance but high-gradient-sensitivity SVD directions.
  - Rare TFBS → low pretraining frequency → low activation variance
    (these weight directions rarely activate strongly on average).
  - DNABERT interpretability (biorxiv) shows motif-specific heads are
    concentrated, not diffuse — concentrated signal + SVD truncation = danger.

Usage:
    # All experiments (use --ckpt_dir for Exp 3):
    python scripts/interpretability/run_svd_spectral_diagnostic.py \\
        --gue_root /home/nvidia/data/gue/GUE \\
        --out results/svd_spectral_diagnostic.json \\
        --plot_dir results/svd_spectral_plots

    # Exp 1+2 only (no fine-tuned checkpoint needed):
    python scripts/interpretability/run_svd_spectral_diagnostic.py \\
        --skip_exp3

    # Quick smoke test (small n_seqs):
    python scripts/interpretability/run_svd_spectral_diagnostic.py \\
        --n_seqs 50 --skip_exp3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent / "evaluation"))
sys.path.insert(0, str(_HERE.parent / "compression"))
sys.path.insert(0, str(_ROOT))

from run_gue_ablation import GUEDataset, evaluate, _task_key, _MAX_LEN, collate_fn

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DNABERT2_ID  = "zhihan1996/DNABERT-2-117M"
_DNABERT2_REV = "7bce263b15377fc15361f52cfab88f8b586abda0"
_DOWN_PROJ_PATTERN = "encoder.layer.{i}.mlp.wo"
_NUM_LAYERS        = 12
_HIDDEN            = 768
_INTERMEDIATE      = 3072

# GUE tasks to benchmark — TF tasks are the "rare motif" proxy
_GUE_TASKS = {
    # TF binding: rare / context-specific signals
    "tf/0": "TF-0",
    "tf/1": "TF-1",
    "tf/2": "TF-2",
    "tf/3": "TF-3",
    "tf/4": "TF-4",
    # Ubiquitous / well-represented tasks (control group)
    "splice/reconstructed": "Splice",
    "prom/prom_core_all":   "Prom-core",
    "EMP/H3":               "H3",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pretrained(device: str = "cuda"):
    import transformers
    transformers.logging.set_verbosity_error()
    tok = transformers.AutoTokenizer.from_pretrained(
        _DNABERT2_ID, revision=_DNABERT2_REV, trust_remote_code=True
    )
    model = transformers.AutoModel.from_pretrained(
        _DNABERT2_ID, revision=_DNABERT2_REV, trust_remote_code=True,
        device_map={"": "cpu"},
    ).to(device).eval()
    return model, tok


def load_finetuned(ckpt_dir: str, num_labels: int, device: str = "cuda"):
    import transformers
    transformers.logging.set_verbosity_error()
    tok = transformers.AutoTokenizer.from_pretrained(
        _DNABERT2_ID, revision=_DNABERT2_REV, trust_remote_code=True
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        _DNABERT2_ID, revision=_DNABERT2_REV, trust_remote_code=True,
        num_labels=num_labels, device_map={"": "cpu"},
    )
    state = torch.load(str(Path(ckpt_dir) / "model_state.pt"), map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), tok


# ─────────────────────────────────────────────────────────────────────────────
# SVD helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_wo(model, layer_idx: int):
    """Return the `wo` module for a given layer index."""
    obj = model
    for attr in _DOWN_PROJ_PATTERN.replace("{i}", str(layer_idx)).split("."):
        obj = getattr(obj, attr)
    return obj


def compute_all_svds(model) -> list[dict]:
    """SVD every down-proj (wo) matrix.  Returns list indexed by layer."""
    print("  Computing SVD for all 12 wo layers …", flush=True)
    svds = []
    for li in range(_NUM_LAYERS):
        W = _resolve_wo(model, li).weight.data.float().cpu()  # [768, 3072]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)   # Vh: [768, 3072]
        svds.append({"U": U, "S": S, "Vh": Vh, "layer": li})
        print(f"    layer {li}: S_max={S[0]:.3f}  S_min={S[-1]:.6f}  "
              f"S_ratio={S[0]/S[-1]:.1f}x  S_50%={S[int(len(S)*0.5)]:.4f}")
    return svds


def spectral_centroid_rank(energy: np.ndarray) -> float:
    """Weighted mean rank position in singular value spectrum (0=top, 1=bottom).

    energy: 1-D array of shape [num_sv], values are e_i = (Vh[i] · h)²
    Returns a scalar in [0, 1]: 0 = all energy in highest-σ direction,
                                 1 = all energy in lowest-σ direction.
    """
    total = energy.sum()
    if total < 1e-30:
        return 0.5
    ranks  = np.arange(len(energy), dtype=float) / (len(energy) - 1)
    return float((ranks * energy).sum() / total)


def spectral_entropy(energy: np.ndarray) -> float:
    """Normalised Shannon entropy of the energy distribution over singular directions."""
    p = energy / (energy.sum() + 1e-30)
    p = p[p > 1e-30]
    return float(-np.sum(p * np.log(p)) / np.log(len(energy)))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Activation spectral profile
# ─────────────────────────────────────────────────────────────────────────────

class _WoInputHook:
    """Stores the mean-pooled intermediate activation (input to wo) per forward call.

    DNABERT-2 uses flash attention which flattens sequences to [total_tokens, H]
    before passing through the MLP.  We therefore process one sequence per forward
    call (batch_size=1) and handle both 2-D [L, H] and 3-D [1, L, H] inputs.
    """
    def __init__(self):
        self.buffer: list[torch.Tensor] = []
        self._handle = None

    def attach(self, module):
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        h = input[0].detach().cpu().float()  # [L, H] or [1, L, H]
        if h.dim() == 3:
            h = h.squeeze(0)          # [L, H]
        # h is now [L, 3072] — mean-pool over token positions
        self.buffer.append(h.mean(dim=0))   # append [3072] vector

    def detach(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

    def collected(self) -> torch.Tensor:
        if not self.buffer:
            return torch.empty(0, _INTERMEDIATE)
        return torch.stack(self.buffer, dim=0)   # [N, 3072]

    def clear(self):
        self.buffer.clear()


def collect_activation_profiles(
    model, tok, sequences: list[str], svds: list[dict],
    device: str, batch_size: int = 1
) -> np.ndarray:
    """
    For each sequence return the per-layer mean spectral centroid rank
    averaged across all 12 layers.

    Processes one sequence at a time (batch_size=1) to correctly handle
    DNABERT-2's flash-attention flattened token layout.

    Returns: np.ndarray [N_seq], values in [0, 1].
    """
    _ = batch_size  # kept for API compat; always uses 1 internally

    # Attach hooks to all 12 wo modules
    hooks = []
    for li in range(_NUM_LAYERS):
        h = _WoInputHook()
        h.attach(_resolve_wo(model, li))
        hooks.append(h)

    # Run sequences ONE AT A TIME so each hook.buffer entry = one sequence
    model.eval()
    for seq in sequences:
        with torch.no_grad():
            enc = tok([seq], return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            model(**enc)

    # Compute spectral centroid for each sequence at each layer
    N = len(sequences)
    centroid_per_layer = np.zeros((N, _NUM_LAYERS), dtype=float)

    for li, hook in enumerate(hooks):
        acts = hook.collected().numpy()   # [N, 3072]
        Vh   = svds[li]["Vh"].numpy()     # [768, 3072]
        for si in range(N):
            c      = Vh @ acts[si]        # [768] projection coefficients
            energy = c ** 2
            centroid_per_layer[si, li] = spectral_centroid_rank(energy)
        hook.detach()
        hook.clear()

    return centroid_per_layer.mean(axis=1)   # [N], mean centroid rank across layers


def run_exp1_activation_spectral(
    model, tok, svds: list[dict], gue_root: str,
    device: str, n_seqs: int, batch_size: int
) -> dict:
    """Exp 1: measure spectral centroid rank for positive vs negative seqs
    in each GUE task.  Returns dict task_name → {pos_mean, neg_mean, delta}.
    """
    print("\n" + "=" * 70)
    print("EXP 1: Activation spectral profile")
    print("=" * 70)
    results = {}

    for task_path, task_name in _GUE_TASKS.items():
        csv_path = Path(gue_root) / task_path / "test.csv"
        if not csv_path.exists():
            print(f"  [{task_name}] SKIP — {csv_path} not found")
            continue

        rows = open(csv_path).readlines()[1:]  # skip header
        pos_seqs, neg_seqs = [], []
        for row in rows:
            parts = row.strip().split(",")
            seq, label = parts[0], parts[-1]
            if label == "1":
                pos_seqs.append(seq)
            else:
                neg_seqs.append(seq)

        # Subsample
        rng = np.random.default_rng(42)
        pos_seqs = list(rng.choice(pos_seqs, min(n_seqs, len(pos_seqs)), replace=False))
        neg_seqs = list(rng.choice(neg_seqs, min(n_seqs, len(neg_seqs)), replace=False))

        if not pos_seqs or not neg_seqs:
            continue

        print(f"  [{task_name}]  pos={len(pos_seqs)}  neg={len(neg_seqs)}", flush=True)

        pos_ranks = collect_activation_profiles(model, tok, pos_seqs, svds,
                                                device, batch_size)
        neg_ranks = collect_activation_profiles(model, tok, neg_seqs, svds,
                                                device, batch_size)

        pos_mean = float(pos_ranks.mean())
        neg_mean = float(neg_ranks.mean())
        delta    = pos_mean - neg_mean
        pos_std  = float(pos_ranks.std())
        neg_std  = float(neg_ranks.std())

        print(f"    centroid_rank: pos={pos_mean:.4f}±{pos_std:.4f}  "
              f"neg={neg_mean:.4f}±{neg_std:.4f}  Δ={delta:+.4f}")
        print(f"    → {'HIGHER rank in +' if delta > 0 else 'LOWER rank in +'} "
              f"(energy {'in low-σ tail' if delta > 0 else 'in high-σ head'} for binding seqs)")

        results[task_name] = {
            "task_path":   task_path,
            "pos_mean":    pos_mean,
            "pos_std":     pos_std,
            "neg_mean":    neg_mean,
            "neg_std":     neg_std,
            "delta":       delta,             # positive = pos seqs use lower-σ directions
            "n_pos":       len(pos_seqs),
            "n_neg":       len(neg_seqs),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: SVD compression sensitivity sweep
# ─────────────────────────────────────────────────────────────────────────────

def _apply_svd_compression(model, svds: list[dict], rank_ratio: float) -> list:
    """Replace all wo.weight with rank-r approximation.  Returns original weights."""
    originals = []
    k = max(1, int(round(rank_ratio * _HIDDEN)))
    for li, svd in enumerate(svds):
        wo    = _resolve_wo(model, li)
        W_orig = wo.weight.data.clone()
        originals.append(W_orig)
        U  = svd["U"].to(wo.weight.device)
        S  = svd["S"].to(wo.weight.device)
        Vh = svd["Vh"].to(wo.weight.device)
        W_approx = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
        wo.weight.data.copy_(W_approx.to(wo.weight.dtype))
    return originals


def _restore_svd(model, originals: list):
    for li, W_orig in enumerate(originals):
        wo = _resolve_wo(model, li)
        wo.weight.data.copy_(W_orig.to(wo.weight.device))


def run_exp2_compression_sweep(
    model, tok, svds: list[dict], gue_root: str,
    device: str, n_seqs: int, batch_size: int,
    ratios: list[float] | None = None
) -> dict:
    """Exp 2: apply rank-r SVD to all wo layers and measure accuracy degradation
    per GUE task.  Returns task_name → list of {ratio, accuracy} dicts.
    """
    if ratios is None:
        ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    print("\n" + "=" * 70)
    print("EXP 2: SVD compression sensitivity sweep")
    print("=" * 70)

    import csv as _csv
    import transformers
    transformers.logging.set_verbosity_error()

    # Build sequence classifiers for each task using the pretrained encoder
    # with a simple mean-pooling probe (fit on training set)
    # For simplicity: just use the raw encoder + majority-class accuracy
    # as a fast approximation.  For a full eval we'd need a probe classifier.
    # Here we embed + re-evaluate cosine similarity to class centroids.
    #
    # Simpler approach for paper: just measure how much the representation
    # CHANGES (L2 distance) for compressed vs uncompressed model.
    # But for accuracy vs ratio we need a probe.
    #
    # We'll use a nearest-centroid classifier trained on CLS embeddings.

    results = {}
    for task_path, task_name in _GUE_TASKS.items():
        train_csv = Path(gue_root) / task_path / "train.csv"
        test_csv  = Path(gue_root) / task_path / "test.csv"
        if not test_csv.exists():
            print(f"  [{task_name}] SKIP")
            continue

        print(f"\n  [{task_name}] building centroid probe …", flush=True)

        def _load_seqs(path, max_n=None):
            rows = open(path).readlines()[1:]
            seqs, lbls = [], []
            for r in rows:
                parts = r.strip().split(",")
                seqs.append(parts[0])
                lbls.append(int(parts[-1]))
            if max_n:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(seqs), min(max_n, len(seqs)), replace=False)
                seqs = [seqs[i] for i in idx]
                lbls = [lbls[i] for i in idx]
            return seqs, lbls

        def _embed(seqs, mod, n_batch=32):
            vecs = []
            mod.eval()
            with torch.no_grad():
                for s in range(0, len(seqs), n_batch):
                    batch = seqs[s:s + n_batch]
                    enc   = tok(batch, return_tensors="pt", padding=True,
                                truncation=True, max_length=512)
                    enc   = {k: v.to(device) for k, v in enc.items()}
                    out   = mod(**enc).last_hidden_state[:, 0, :].float().cpu()
                    vecs.append(out)
            return torch.cat(vecs, dim=0)

        train_seqs, train_lbls = _load_seqs(train_csv, max_n=min(n_seqs * 4, 500))
        test_seqs,  test_lbls  = _load_seqs(test_csv,  max_n=n_seqs)

        # Compute centroids at ratio=1.0 (uncompressed) — used for all ratios
        train_emb = _embed(train_seqs, model)
        labels_arr = np.array(train_lbls)
        unique_labels = sorted(set(train_lbls))
        centroids = {l: train_emb[labels_arr == l].mean(dim=0) for l in unique_labels}

        def _nearest_centroid_acc(embs):
            preds = []
            for e in embs:
                dists = {l: torch.dist(e, c).item() for l, c in centroids.items()}
                preds.append(min(dists, key=dists.get))
            return float(np.mean(np.array(preds) == np.array(test_lbls[:len(preds)])))

        ratio_results = []
        for ratio in ratios:
            originals = _apply_svd_compression(model, svds, ratio)
            test_emb  = _embed(test_seqs, model)
            acc       = _nearest_centroid_acc(test_emb)
            _restore_svd(model, originals)
            ratio_results.append({"ratio": ratio, "accuracy": round(acc, 4)})
            print(f"    ratio={ratio:.2f}  acc={acc:.4f}")

        results[task_name] = ratio_results
        # Find "knee": first ratio where accuracy drops > 5% of baseline
        baseline_acc = ratio_results[0]["accuracy"]
        knee = next((r["ratio"] for r in ratio_results
                     if r["accuracy"] < baseline_acc - 0.05), None)
        print(f"  → baseline={baseline_acc:.4f}  "
              f"knee ratio={knee if knee else 'not reached'}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Gradient spectral profile (fine-tuned model)
# ─────────────────────────────────────────────────────────────────────────────

def run_exp3_gradient_spectral(
    model, tok, svds: list[dict],
    sequences: list[str], labels: list[int],
    device: str, n_seqs: int, task_name: str
) -> dict:
    """Exp 3: compute ∂CE/∂wo per sequence, project onto SVD directions.
    Returns dict with pos_centroid_mean, neg_centroid_mean, delta per layer.
    """
    print(f"\n  [{task_name}] gradient spectral profile  n={min(len(sequences), n_seqs)}", flush=True)

    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(sequences))[:n_seqs]
    seqs_s = [sequences[i] for i in idx]
    lbls_s = [labels[i] for i in idx]

    pos_centroids = [[] for _ in range(_NUM_LAYERS)]  # per layer
    neg_centroids = [[] for _ in range(_NUM_LAYERS)]

    model.eval()
    for seq, lbl in zip(seqs_s, lbls_s):
        model.zero_grad()
        enc = tok([seq], return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        logits = out.logits if hasattr(out, "logits") else out.last_hidden_state[:, 0, :] @ torch.zeros(768, 2, device=device)
        if not hasattr(out, "logits"):
            break  # Exp 3 needs fine-tuned model with logits
        target = torch.tensor([lbl], device=device)
        loss   = F.cross_entropy(logits, target)
        loss.backward()

        for li in range(_NUM_LAYERS):
            wo = _resolve_wo(model, li)
            if wo.weight.grad is None:
                continue
            G  = wo.weight.grad.float().cpu()   # [768, 3072]
            U  = svds[li]["U"].cpu().numpy()    # [768, 768]
            Vh = svds[li]["Vh"].cpu().numpy()   # [768, 3072]
            # Scalar projection per singular direction: g_i = U[:,i] · G · Vh[i,:]
            g_per_sv = np.einsum("di,dj,ij->i", U, Vh, G.numpy())  # [768]
            energy   = g_per_sv ** 2
            centroid = spectral_centroid_rank(energy)
            if lbl == 1:
                pos_centroids[li].append(centroid)
            else:
                neg_centroids[li].append(centroid)

        model.zero_grad()

    per_layer = []
    for li in range(_NUM_LAYERS):
        pc = np.mean(pos_centroids[li]) if pos_centroids[li] else float("nan")
        nc = np.mean(neg_centroids[li]) if neg_centroids[li] else float("nan")
        delta = pc - nc if not np.isnan(pc) and not np.isnan(nc) else float("nan")
        per_layer.append({"layer": li, "pos_grad_centroid": pc,
                          "neg_grad_centroid": nc, "delta": delta})
        print(f"    layer {li:2d}  pos={pc:.4f}  neg={nc:.4f}  Δ={delta:+.4f}")

    mean_delta = float(np.nanmean([r["delta"] for r in per_layer]))
    print(f"  → mean gradient centroid Δ = {mean_delta:+.4f}  "
          f"({'pos uses lower-σ' if mean_delta > 0 else 'pos uses higher-σ'})")
    return {"per_layer": per_layer, "mean_delta": mean_delta, "task": task_name}


# ─────────────────────────────────────────────────────────────────────────────
# Singular value spectrum summary (printed once for context)
# ─────────────────────────────────────────────────────────────────────────────

def print_sv_spectrum_summary(svds: list[dict]):
    print("\n  Singular value spectrum summary (wo matrices):")
    print(f"  {'layer':>5}  {'S[0]':>8}  {'S[383]':>8}  {'S[767]':>8}  {'ratio':>8}")
    for li, svd in enumerate(svds):
        S = svd["S"]
        print(f"  {li:>5}  {S[0].item():>8.4f}  {S[383].item():>8.6f}  "
              f"{S[767].item():>8.6f}  {(S[0]/S[767]).item():>8.1f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_exp1(results: dict, plot_dir: str):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    tasks     = list(results.keys())
    pos_means = [results[t]["pos_mean"] for t in tasks]
    neg_means = [results[t]["neg_mean"] for t in tasks]
    deltas    = [results[t]["delta"]    for t in tasks]
    pos_stds  = [results[t]["pos_std"]  for t in tasks]
    neg_stds  = [results[t]["neg_std"]  for t in tasks]
    x = np.arange(len(tasks))

    # TF tasks vs others (different color)
    colors = ["steelblue" if t.startswith("TF") else "gray" for t in tasks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Exp 1: Activation Spectral Centroid Rank\n"
                 "Higher rank = energy in lower-σ tail of down-proj SVD spectrum",
                 fontsize=11)

    width = 0.35
    ax1.bar(x - width/2, pos_means, width, label="+ (binding/positive)", color="steelblue", alpha=0.8)
    ax1.bar(x + width/2, neg_means, width, label="− (non-binding/negative)", color="salmon", alpha=0.8)
    ax1.errorbar(x - width/2, pos_means, yerr=pos_stds, fmt="none",
                 ecolor="black", capsize=3, linewidth=0.8)
    ax1.errorbar(x + width/2, neg_means, yerr=neg_stds, fmt="none",
                 ecolor="black", capsize=3, linewidth=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(tasks, rotation=30, ha="right")
    ax1.set_ylabel("Mean spectral centroid rank [0=high-σ, 1=low-σ]")
    ax1.set_title("Spectral centroid: pos vs neg")
    ax1.legend(); ax1.grid(True, axis="y", alpha=0.3)
    ax1.axhline(0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.4)

    bar_colors = ["steelblue" if t.startswith("TF") else "gray" for t in tasks]
    bars = ax2.bar(x, deltas, color=bar_colors, alpha=0.85, edgecolor="black", lw=0.6)
    for bar, d in zip(bars, deltas):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 d + (0.001 if d >= 0 else -0.003),
                 f"{d:+.4f}", ha="center",
                 va="bottom" if d >= 0 else "top", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(tasks, rotation=30, ha="right")
    ax2.set_ylabel("Δ centroid rank (pos − neg)")
    ax2.set_title("Δ = positive: binding seqs use lower-σ directions\n(blue = TF tasks, gray = control tasks)")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = str(Path(plot_dir) / "exp1_activation_spectral.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out}")
    plt.close(fig)


def _plot_exp2(results: dict, plot_dir: str):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    tf_tasks    = [t for t in results if t.startswith("TF")]
    ctrl_tasks  = [t for t in results if not t.startswith("TF")]

    import matplotlib.cm as cm
    tf_colors   = cm.Blues(np.linspace(0.4, 0.9, max(len(tf_tasks), 1)))
    ctrl_colors = cm.Oranges(np.linspace(0.4, 0.9, max(len(ctrl_tasks), 1)))

    for tasks, colors, ls in [(tf_tasks, tf_colors, "-"), (ctrl_tasks, ctrl_colors, "--")]:
        for task, color in zip(tasks, colors):
            ratios = [r["ratio"]    for r in results[task]]
            accs   = [r["accuracy"] for r in results[task]]
            ax.plot(ratios, accs, label=task, color=color, linestyle=ls,
                    marker="o", markersize=4)

    ax.set_xlabel("Rank ratio r (fraction of singular values kept)")
    ax.set_ylabel("Nearest-centroid accuracy on test set")
    ax.set_title("Exp 2: SVD Compression Sensitivity by GUE Task\n"
                 "TF binding tasks (blue solid) vs control tasks (orange dashed)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = str(Path(plot_dir) / "exp2_compression_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out}")
    plt.close(fig)


def _plot_exp3(exp3_results: list, plot_dir: str):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for res in exp3_results:
        layers = [r["layer"]  for r in res["per_layer"]]
        deltas = [r["delta"]  for r in res["per_layer"]]
        ax.plot(layers, deltas, marker="o", markersize=4, label=res["task"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Δ gradient centroid rank (pos − neg)")
    ax.set_title("Exp 3: Gradient Spectral Profile per Layer\n"
                 "Positive = binding seqs have gradients concentrated in lower-σ directions")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = str(Path(plot_dir) / "exp3_gradient_spectral.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SVD spectral diagnostic for DNABERT-2")
    p.add_argument("--gue_root",    default="/home/nvidia/data/gue/GUE")
    p.add_argument("--configs_dir", default="configs")
    p.add_argument("--n_seqs",      type=int, default=200,
                   help="Max sequences per class per task. Default: 200")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--out",         default="results/svd_spectral_diagnostic.json")
    p.add_argument("--plot_dir",    default="results/svd_spectral_plots")
    p.add_argument("--skip_exp2",   action="store_true",
                   help="Skip SVD compression sweep (faster)")
    p.add_argument("--skip_exp3",   action="store_true",
                   help="Skip gradient spectral analysis (no fine-tuned ckpt needed)")
    p.add_argument("--ckpt_dir",    default=None,
                   help="Fine-tuned DNABERT-2 checkpoint dir for Exp 3 (optional)")
    p.add_argument("--exp3_task",   default="tf/0",
                   help="GUE task to use for Exp 3 gradient analysis")
    p.add_argument("--compression_ratios", nargs="+", type=float,
                   default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05])
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.perf_counter()

    print(f"\n{'='*70}")
    print("SVD Spectral Diagnostic: DNABERT-2 down-projection matrices")
    print(f"  n_seqs={args.n_seqs}  batch={args.batch_size}  device={args.device}")
    print(f"{'='*70}\n")

    print("  Loading pretrained DNABERT-2 …", flush=True)
    model, tok = load_pretrained(args.device)
    print("  Model loaded.")

    # ── Pre-compute SVD for all 12 layers ────────────────────────────────────
    svds = compute_all_svds(model)
    print_sv_spectrum_summary(svds)

    output = {
        "config": {
            "n_seqs": args.n_seqs,
            "batch_size": args.batch_size,
            "compression_ratios": args.compression_ratios,
        }
    }

    # ── Exp 1: Activation spectral profile ───────────────────────────────────
    exp1 = run_exp1_activation_spectral(
        model, tok, svds, args.gue_root,
        device=args.device, n_seqs=args.n_seqs, batch_size=args.batch_size
    )
    output["exp1_activation_spectral"] = exp1

    # Summary: TF tasks vs controls
    tf_deltas   = [exp1[k]["delta"] for k in exp1 if k.startswith("TF")]
    ctrl_deltas = [exp1[k]["delta"] for k in exp1 if not k.startswith("TF")]
    if tf_deltas:
        print(f"\n  Exp1 summary:  TF tasks mean Δ = {np.mean(tf_deltas):+.4f}  "
              f"control tasks mean Δ = {np.mean(ctrl_deltas) if ctrl_deltas else float('nan'):+.4f}")
        print(f"  {'✓' if np.mean(tf_deltas) > np.mean(ctrl_deltas) else '✗'} "
              f"Hypothesis: TF binding uses LOWER-σ directions more than control tasks")
        output["exp1_summary"] = {
            "tf_mean_delta":   float(np.mean(tf_deltas)),
            "ctrl_mean_delta": float(np.mean(ctrl_deltas)) if ctrl_deltas else None,
            "hypothesis_supported": bool(np.mean(tf_deltas) > (np.mean(ctrl_deltas) if ctrl_deltas else 0)),
        }

    # ── Exp 2: SVD compression sweep ─────────────────────────────────────────
    if not args.skip_exp2:
        exp2 = run_exp2_compression_sweep(
            model, tok, svds, args.gue_root,
            device=args.device, n_seqs=args.n_seqs, batch_size=args.batch_size,
            ratios=args.compression_ratios,
        )
        output["exp2_compression_sweep"] = exp2
    else:
        print("\n  [Exp 2 skipped]")
        output["exp2_compression_sweep"] = None

    # ── Exp 3: Gradient spectral profile ─────────────────────────────────────
    if not args.skip_exp3:
        if args.ckpt_dir is None:
            # Try to auto-detect a TF checkpoint
            task_leaf = args.exp3_task.replace("/", "_")
            auto = _ROOT / f"results/gue_checkpoints/dnabert2_{task_leaf}"
            if not auto.exists():
                print(f"\n  [Exp 3 skipped] no checkpoint at {auto}  "
                      f"— run with --ckpt_dir or skip with --skip_exp3")
                output["exp3_gradient_spectral"] = None
            else:
                args.ckpt_dir = str(auto)

        if args.ckpt_dir is not None:
            import csv as _csv
            rows = open(Path(args.gue_root) / args.exp3_task / "test.csv").readlines()[1:]
            seqs = [r.strip().split(",")[0]  for r in rows]
            lbls = [int(r.strip().split(",")[-1]) for r in rows]
            num_labels = len(set(lbls))
            print(f"\n  Loading fine-tuned DNABERT-2 from {args.ckpt_dir} …")
            ft_model, ft_tok = load_finetuned(args.ckpt_dir, num_labels, args.device)
            ft_svds = compute_all_svds(ft_model)
            exp3 = run_exp3_gradient_spectral(
                ft_model, ft_tok, ft_svds, seqs, lbls,
                device=args.device, n_seqs=args.n_seqs, task_name=args.exp3_task
            )
            output["exp3_gradient_spectral"] = exp3
    else:
        print("\n  [Exp 3 skipped]")
        output["exp3_gradient_spectral"] = None

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_exp1(exp1, args.plot_dir)
    if output["exp2_compression_sweep"]:
        _plot_exp2(output["exp2_compression_sweep"], args.plot_dir)
    if output["exp3_gradient_spectral"]:
        _plot_exp3([output["exp3_gradient_spectral"]], args.plot_dir)

    total = time.perf_counter() - t_start
    print(f"\n  Total runtime: {total:.1f}s")


if __name__ == "__main__":
    main()
