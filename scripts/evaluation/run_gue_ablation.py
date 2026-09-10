# scripts/run_gue_ablation.py
"""
Three-condition GUE task ablation for masked-LM genomic models (DNABERT-2, NTv3).

Usage (from glm_super_weight/genomic-super-weights/):

    python scripts/run_gue_ablation.py \\
        --model dnabert2 \\
        --task prom/prom_core_notata \\
        --gue_root /work/11034/atzanakak/GUE/GUE

    # with layer sweep (finds task-specific super rows missed by PPL-based detection)
    python scripts/run_gue_ablation.py \\
        --model dnabert2 --task EMP/H3K4me3 \\
        --gue_root /work/11034/atzanakak/GUE/GUE \\
        --sweep

    # skip fine-tuning if checkpoint already exists
    python scripts/run_gue_ablation.py \\
        --model dnabert2 --task prom/prom_core_notata \\
        --gue_root /work/11034/atzanakak/GUE/GUE \\
        --ckpt_dir results/gue_checkpoints/dnabert2_prom_core_notata

Three conditions (same structure as run_ablation.py):
    1. Baseline      – evaluate fine-tuned model on test split.
    2. Prune SW      – zero rows from sw_index for this model → re-evaluate.
    3. Random ctrl   – zero N random rows → mean accuracy / MCC over 10 repeats.
                       Rows are matched in count to the pruned-SW condition.

Optional --sweep:
    Layer sweep      – for each layer, zero its highest-L1-norm row, measure Δacc.
                       Surfaces task-specific super rows missed by PPL-based detection.

Results are merged into --out JSON under key "<model>/<task>".
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
# sklearn replaced by numpy implementations below (_accuracy, _mcc, _f1_macro)
import torch
import torch.nn as nn
# transformers imported lazily inside run_task() to allow evo2-container usage
import yaml


# ── Max-length lookup (from official run_dnabert2.sh) ───────────────────────
_MAX_LEN = {
    "EMP":           128,
    "EPI":           128,
    "fungi":         512,
    "mouse":          30,
    "tf":             30,
    "prom_core_all":  20,
    "prom_core_notata": 20,
    "prom_core_tata": 20,
    "prom_300_all":   70,
    "prom_300_notata": 70,
    "prom_300_tata":  70,
    "reconstructed": 80,  # splice
    "covid":         256,
    "species_40":    512,
    "species_20":    512,  # fungi
}

# ── Default fine-tune epochs per task ────────────────────────────────────────
_EPOCHS = {
    "EMP":            3,
    "EPI":            3,
    "fungi":          3,
    "mouse":          5,
    "tf":             3,
    "prom_core_all":  4,
    "prom_core_notata": 4,
    "prom_core_tata": 10,
    "prom_300_all":   4,
    "prom_300_notata": 4,
    "prom_300_tata":  10,
    "reconstructed":  5,
    "covid":          8,
    "species_40":     3,
    "species_20":     3,
}


def _task_key(task: str) -> str:
    """Map task path (e.g. 'prom/prom_core_notata') to a key for lookup dicts."""
    leaf = task.split("/")[-1]
    group = task.split("/")[0]
    # leaf wins if known, otherwise fall back to group
    return leaf if leaf in _MAX_LEN else group


# ── Dataset ──────────────────────────────────────────────────────────────────

class GUEDataset(torch.utils.data.Dataset):
    """Reads GUE CSVs (sequence,label or sequence1,sequence2,label)."""

    def __init__(self, csv_path: str, tokenizer, max_length: int):
        with open(csv_path) as f:
            rows = list(csv.reader(f))[1:]  # skip header

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
        self.input_ids = enc["input_ids"]
        # Some tokenizers (e.g. NTv3) don't return attention_mask — synthesize it
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


# ── HybriDNA classification wrapper ─────────────────────────────────────────

class _HybriDNAClassifier(nn.Module):
    """Wraps a HybriDNA-7B causal-LM backbone with a mean-pool + linear head."""

    class _Output:
        def __init__(self, logits, loss=None):
            self.logits = logits
            self.loss   = loss

    def __init__(self, backbone, hidden_size: int, num_labels: int):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hidden = out.hidden_states[-1]
        elif hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        else:
            raise RuntimeError("Cannot find hidden states in HybriDNA output")

        if attention_mask is not None:
            mask   = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)

        # Lazily move the classifier head to match the backbone output device/dtype.
        # With device_map="auto" the head is created on CPU; hidden lands on cuda:0.
        if (self.classifier.weight.device != pooled.device
                or self.classifier.weight.dtype != pooled.dtype):
            self.classifier = self.classifier.to(
                device=pooled.device, dtype=pooled.dtype
            )

        logits = self.classifier(pooled)
        loss   = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return self._Output(logits=logits, loss=loss)


# ── GENERator classification wrapper ─────────────────────────────────────────

class _GeneratorClassifier(nn.Module):
    """
    Wraps a GENERator (LlamaForCausalLM) backbone with a mean-pool + linear head.
    Identical pattern to _HybriDNAClassifier; handles device/dtype mismatch lazily.
    """

    class _Output:
        def __init__(self, logits, loss=None):
            self.logits = logits
            self.loss   = loss

    def __init__(self, backbone, hidden_size: int, num_labels: int):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hidden = out.hidden_states[-1]
        elif hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        else:
            raise RuntimeError("Cannot find hidden states in GENERator output")

        if attention_mask is not None:
            mask   = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)

        if (self.classifier.weight.device != pooled.device
                or self.classifier.weight.dtype != pooled.dtype):
            self.classifier = self.classifier.to(
                device=pooled.device, dtype=pooled.dtype
            )

        logits = self.classifier(pooled)
        loss   = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return self._Output(logits=logits, loss=loss)


# ── NTv3 classification wrapper ───────────────────────────────────────────────

class _NTv3Classifier(nn.Module):
    """
    Wraps an NTv3 masked-LM backbone with a mean-pool + linear classification head.
    NTv3 only registers AutoModelForMaskedLM, so we build the classifier manually.
    Exposes `.logits` in the output so the rest of the training/eval code is unchanged.
    Also exposes `.loss` when labels are passed (cross-entropy).
    """

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
        # NTv3 conv tower halves the sequence num_downsamples times.
        # Minimum safe input length = 2^(num_downsamples+1) to avoid avg_pool1d
        # producing output size 0 when sequences are short (e.g. GUE prom ~249bp).
        # NTv3 is a conv/deconv U-Net: the conv tower halves the sequence once per
        # block and the deconv tower's skip connections (`y = y + r`) only align when
        # the length is an exact MULTIPLE of 2**n_conv_blocks. Padding merely up to a
        # minimum is not enough -- a 400-token splice input is >min_len but
        # 400 % 256 = 144, which raises
        #   "The size of tensor a (24) must match the size of tensor b (25)".
        # Pad up to the next multiple instead.
        nd = getattr(self.backbone.config, "num_downsamples", 7)
        min_len = 2 ** (nd + 1)
        seq_len = input_ids.shape[1]
        target = ((seq_len + min_len - 1) // min_len) * min_len
        if seq_len < target:
            pad = target - seq_len
            pad_id = getattr(self.backbone.config, "pad_token_id", 1)
            input_ids     = torch.nn.functional.pad(input_ids,     (0, pad), value=pad_id)
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=0)

        # NTv3 backbone returns a dict; embeddings are under 'embeddings' key.
        # Fall back to hidden_states or last_hidden_state depending on output shape.
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Prefer last encoder hidden state; NTv3 may return a dict with 'embeddings'
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hidden = out.hidden_states[-1]          # (B, L, D)
        elif hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        elif isinstance(out, dict) and "embeddings" in out:
            hidden = out["embeddings"]
        else:
            # last resort: first tensor in output that is 3-D
            for v in (out.values() if isinstance(out, dict) else vars(out).values()):
                if isinstance(v, torch.Tensor) and v.dim() == 3:
                    hidden = v
                    break
            else:
                raise RuntimeError("Cannot find hidden states in NTv3 output")

        # Mean pool over sequence dimension (masked positions excluded)
        if attention_mask is not None:
            mask  = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(1)

        logits = self.classifier(pooled)
        loss   = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return self._Output(logits=logits, loss=loss)


# ── Metric helpers (numpy-only, no sklearn dependency) ───────────────────────

def _accuracy(labels, preds):
    return float(np.mean(labels == preds))

def _mcc(labels, preds):
    classes = np.unique(np.concatenate([labels, preds]))
    if len(classes) == 2:
        tp = int(np.sum((preds == classes[1]) & (labels == classes[1])))
        tn = int(np.sum((preds == classes[0]) & (labels == classes[0])))
        fp = int(np.sum((preds == classes[1]) & (labels == classes[0])))
        fn = int(np.sum((preds == classes[0]) & (labels == classes[1])))
        denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
        return float((tp*tn - fp*fn) / denom) if denom > 0 else 0.0
    # multiclass via confusion matrix
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    C = np.zeros((n, n), dtype=np.float64)
    for lt, lp in zip(labels, preds):
        C[idx[lt], idx[lp]] += 1
    s = C.sum()
    t_k = C.sum(axis=1)
    p_k = C.sum(axis=0)
    cov_ytyp = np.trace(C) * s - np.dot(t_k, p_k)
    cov_ytyt = s**2 - np.dot(t_k, t_k)
    cov_ypyp = s**2 - np.dot(p_k, p_k)
    denom = (cov_ytyt * cov_ypyp) ** 0.5
    return float(cov_ytyp / denom) if denom > 0 else 0.0

def _f1_macro(labels, preds):
    f1s = []
    for c in np.unique(labels):
        tp = float(np.sum((preds == c) & (labels == c)))
        fp = float(np.sum((preds == c) & (labels != c)))
        fn = float(np.sum((preds != c) & (labels == c)))
        denom = 2*tp + fp + fn
        f1s.append(2*tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0

# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataset, batch_size: int = 64, device: str = "cuda") -> dict:
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn
    )
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
        "accuracy": _accuracy(labels, preds),
        "mcc":      _mcc(labels, preds),
        "f1":       _f1_macro(labels, preds),
    }


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def fine_tune(
    model,
    train_ds,
    val_ds,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    ckpt_dir: str,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Linear warmup (10% of steps) + linear decay to 0
    total_steps  = epochs * len(loader)
    warmup_steps = max(1, total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_mcc, best_ckpt = -1.0, None

    for ep in range(epochs):
        total_loss = 0.0
        model.train()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out   = model(**batch)
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        val_m = evaluate(model, val_ds, device=device)
        torch.cuda.empty_cache()
        print(
            f"  Epoch {ep+1}/{epochs}  loss={total_loss/len(loader):.4f}"
            f"  val_acc={val_m['accuracy']:.4f}  val_mcc={val_m['mcc']:.4f}"
        )
        if val_m["mcc"] > best_mcc:
            best_mcc  = val_m["mcc"]
            best_ckpt = os.path.join(ckpt_dir, "model_state.pt")
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ↑ New best val_mcc={best_mcc:.4f} — checkpoint saved.")

    # Restore best weights into the live model
    if best_ckpt and os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location="cpu"), strict=False)
    print(f"  Saved best checkpoint (val_mcc={best_mcc:.4f}) → {best_ckpt}")


# ── Module resolution (same logic as BaseGenomicWrapper._resolve_module) ─────

def _resolve_module(model, pattern: str, layer_idx: int):
    path = pattern.replace("{i}", str(layer_idx))
    # Unwrap classifier wrappers so the pattern navigates the actual backbone.
    obj = model.backbone if isinstance(model, (_NTv3Classifier, _HybriDNAClassifier, _GeneratorClassifier)) else model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


# ── Row zeroing helpers ───────────────────────────────────────────────────────

def _save_row(model, pattern: str, layer_idx: int, row: int) -> torch.Tensor:
    m = _resolve_module(model, pattern, layer_idx)
    return m.weight.data[row, :].clone()


def _zero_row(model, pattern: str, layer_idx: int, row: int):
    m = _resolve_module(model, pattern, layer_idx)
    with torch.no_grad():
        m.weight.data[row, :] = 0.0


def _restore_row(model, pattern: str, layer_idx: int, row: int, saved: torch.Tensor):
    m = _resolve_module(model, pattern, layer_idx)
    with torch.no_grad():
        m.weight.data[row, :] = saved


# ── Structured-random sampler ─────────────────────────────────────────────────

def _structured_rand_sample(sw_list: list, num_rows_per_layer: int, rng) -> list:
    """
    Returns a list of (layer, row) pairs that mirror the *structure* of sw_list:
      - same layer distribution (how many rows come from each layer)
      - same row-repetition pattern (if SW row X appears in 4 layers, the randomly
        chosen substitute row also appears in those exact same 4 layers)

    Algorithm
    ---------
    1. Group sw_list entries by their row index → each group records which layers
       that row appears in.
    2. Sort groups by size descending to form a "template"
       e.g. [[3,5,6,7], [3,5], [3], [3], [9], [9]]
    3. Sample len(groups) distinct random row indices (avoiding actual SW rows).
    4. Assign each random row to the same layer list as its template group.
    """
    from collections import defaultdict
    row_to_layers: dict = defaultdict(list)
    for sw in sw_list:
        row_to_layers[sw["row"]].append(sw["layer"])

    # Template: sorted by repetition count desc so sampling is deterministic in structure
    groups = sorted(row_to_layers.values(), key=len, reverse=True)
    n_unique = len(groups)

    sw_rows = {sw["row"] for sw in sw_list}
    pool = [r for r in range(num_rows_per_layer) if r not in sw_rows]
    chosen_rows = rng.sample(pool, n_unique)

    result = []
    for row_idx, layer_list in zip(chosen_rows, groups):
        for layer in layer_list:
            result.append((layer, row_idx))
    return result


# ── Three-condition ablation ──────────────────────────────────────────────────

def run_ablation(
    model,
    sw_list: list,
    test_ds,
    pattern: str,
    num_layers: int,
    device: str,
    n_control: int = 10,
    n_structured_control: int = 0,
) -> dict:
    """
    Mirrors run_destruction_test() from analysis/ablation.py but uses
    task accuracy + MCC instead of perplexity.

    sw_list: list of {"layer": int, "row": int} dicts (from super_weight_index.json).
             May be empty — in that case condition 2 is skipped.

    n_structured_control: if > 0, run an additional structured-random condition
             that samples rows with the same layer distribution and row-repetition
             pattern as the detected super-rows (see _structured_rand_sample).
    """
    baseline = evaluate(model, test_ds, device=device)
    print(f"  Baseline: acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")
    result = {"baseline": baseline}

    # ── Condition 2: prune detected SW row(s) ──────────────────────────────
    if sw_list:
        saves = [(sw["layer"], sw["row"],
                  _save_row(model, pattern, sw["layer"], sw["row"]))
                 for sw in sw_list]
        for sw in sw_list:
            _zero_row(model, pattern, sw["layer"], sw["row"])
        sw_m = evaluate(model, test_ds, device=device)
        print(f"  Pruned SW: acc={sw_m['accuracy']:.4f}  mcc={sw_m['mcc']:.4f}")
        for layer, row, saved in saves:
            _restore_row(model, pattern, layer, row, saved)

        result["pruned_sw"] = sw_m
        result["delta_acc_sw_pct"] = (
            (sw_m["accuracy"] - baseline["accuracy"])
            / max(baseline["accuracy"], 1e-9) * 100
        )
        result["delta_mcc_sw_pct"] = (
            (sw_m["mcc"] - baseline["mcc"])
            / max(abs(baseline["mcc"]), 1e-9) * 100
        )
    else:
        print("  [skip] No super rows in index for this model — skipping prune-SW condition.")
        result["pruned_sw"] = None

    # ── Condition 3: random row controls ──────────────────────────────────
    rng        = random.Random(42)
    sw_coords  = {(sw["layer"], sw["row"]) for sw in sw_list}
    n_prune    = max(len(sw_list), 1)  # match count; at least 1

    # Build candidate list: all (layer, row) excluding SW coords
    candidates = []
    for li in range(num_layers):
        m     = _resolve_module(model, pattern, li)
        nrows = m.weight.data.shape[0]
        for ri in range(nrows):
            if (li, ri) not in sw_coords:
                candidates.append((li, ri))

    rand_results = []
    for _ in range(n_control):
        chosen = rng.sample(candidates, n_prune)
        saves  = [(l, r, _save_row(model, pattern, l, r)) for l, r in chosen]
        for l, r in chosen:
            _zero_row(model, pattern, l, r)
        m = evaluate(model, test_ds, device=device)
        rand_results.append(m)
        for l, r, saved in saves:
            _restore_row(model, pattern, l, r, saved)

    rand_mean_acc = float(np.mean([m["accuracy"] for m in rand_results]))
    rand_mean_mcc = float(np.mean([m["mcc"]      for m in rand_results]))
    print(f"  Rand ctrl (n={n_control} mean): acc={rand_mean_acc:.4f}  mcc={rand_mean_mcc:.4f}")

    result["pruned_rand_mean"] = {"accuracy": rand_mean_acc, "mcc": rand_mean_mcc}
    result["pruned_rand_all"]  = rand_results

    # ── Condition 4: structured-random control ────────────────────────────
    if n_structured_control > 0 and sw_list:
        # Determine number of rows in the target layer (same for all layers in mlp.wo)
        num_rows_per_layer = _resolve_module(model, pattern, 0).weight.data.shape[0]

        struct_rng     = random.Random(42)
        struct_results = []
        for rep in range(n_structured_control):
            chosen = _structured_rand_sample(sw_list, num_rows_per_layer, struct_rng)
            # Filter out any accidental SW coord collisions
            chosen = [(l, r) for l, r in chosen if (l, r) not in sw_coords]
            saves  = [(l, r, _save_row(model, pattern, l, r)) for l, r in chosen]
            for l, r in chosen:
                _zero_row(model, pattern, l, r)
            m = evaluate(model, test_ds, device=device)
            struct_results.append(m)
            for l, r, saved in saves:
                _restore_row(model, pattern, l, r, saved)

        struct_mean_acc = float(np.mean([m["accuracy"] for m in struct_results]))
        struct_mean_mcc = float(np.mean([m["mcc"]      for m in struct_results]))
        print(f"  Struct rand ctrl (n={n_structured_control} mean): "
              f"acc={struct_mean_acc:.4f}  mcc={struct_mean_mcc:.4f}")
        result["pruned_struct_rand_mean"] = {"accuracy": struct_mean_acc, "mcc": struct_mean_mcc}
        result["pruned_struct_rand_all"]  = struct_results

    return result


# ── Layer sweep ───────────────────────────────────────────────────────────────

def layer_sweep(
    model,
    test_ds,
    pattern: str,
    num_layers: int,
    device: str,
) -> list:
    """
    For each layer: zero the row with the highest L1 norm (most 'active' row
    in the weight matrix), evaluate accuracy + MCC, restore.

    Returns list of dicts sorted by delta_acc (most negative = most critical).
    This surfaces task-specific super rows that are invisible to PPL-based detection.
    """
    baseline = evaluate(model, test_ds, device=device)
    print(f"\n  [sweep] Baseline: acc={baseline['accuracy']:.4f}  mcc={baseline['mcc']:.4f}")

    records = []
    for li in range(num_layers):
        m       = _resolve_module(model, pattern, li)
        w       = m.weight.data
        top_row = int(w.abs().sum(dim=1).argmax().item())
        saved   = _save_row(model, pattern, li, top_row)
        _zero_row(model, pattern, li, top_row)
        ev = evaluate(model, test_ds, device=device)
        _restore_row(model, pattern, li, top_row, saved)

        delta_acc = ev["accuracy"] - baseline["accuracy"]
        delta_mcc = ev["mcc"]      - baseline["mcc"]
        print(
            f"  [sweep] Layer {li:2d}  top_row={top_row:4d}"
            f"  acc={ev['accuracy']:.4f}  Δacc={delta_acc:+.4f}  Δmcc={delta_mcc:+.4f}"
        )
        records.append({
            "layer":     li,
            "row":       top_row,
            "accuracy":  ev["accuracy"],
            "mcc":       ev["mcc"],
            "delta_acc": delta_acc,
            "delta_mcc": delta_mcc,
        })

    records.sort(key=lambda x: x["delta_acc"])  # most negative first
    print("\n  [sweep] Top-5 most critical layers (by Δacc):")
    for r in records[:5]:
        print(f"    Layer {r['layer']:2d}  row={r['row']:4d}  Δacc={r['delta_acc']:+.4f}"
              f"  Δmcc={r['delta_mcc']:+.4f}")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GUE task ablation for genomic masked LMs")
    parser.add_argument("--model",    required=True,
                        choices=["dnabert2", "ntv3", "hybridna",
                                 "generator", "generator_prokaryote", "generator_prokaryote_1b"],
                        help="Model key — must match a configs/<model>.yaml")
    parser.add_argument("--task",     required=True,
                        help="GUE task path relative to gue_root, e.g. prom/prom_core_notata")
    parser.add_argument("--gue_root", required=True,
                        help="Path to the GUE/ directory (contains EMP, prom, splice, ...)")
    parser.add_argument("--epochs",   type=int,   default=None,
                        help="Fine-tune epochs (default: task-specific from official script)")
    parser.add_argument("--lr",       type=float, default=3e-5)
    parser.add_argument("--batch",    type=int,   default=8)
    parser.add_argument("--max_length", type=int, default=None,
                        help="Tokenizer max_length (default: task-specific)")
    parser.add_argument("--ckpt_dir", default=None,
                        help="Where to save/load the fine-tune checkpoint "
                             "(default: results/gue_checkpoints/<model>_<task_leaf>)")
    parser.add_argument("--sw_index", default="results/negative_results/super_weight_index.json")
    parser.add_argument("--out",      default="results/experiments/gue/gue_ablation_results.json")
    parser.add_argument("--sweep",    action="store_true",
                        help="Run layer sweep to find task-specific super rows")
    parser.add_argument("--structured_rand", type=int, default=0, metavar="N",
                        help="Run N structured-random controls matching the superrow "
                             "layer distribution and row-repetition pattern (default: 0 = off)")
    parser.add_argument("--no_finetune", action="store_true",
                        help="Skip fine-tuning (requires existing checkpoint at --ckpt_dir)")
    parser.add_argument("--hf_token",  default=None,
                        help="HuggingFace token for gated/private repos (or set HF_TOKEN env var)")
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── Config ──────────────────────────────────────────────────────────────
    config = yaml.safe_load(open(f"configs/{args.model}.yaml"))
    model_id     = config["model_id"]
    pattern      = config["down_proj_pattern"]
    num_layers   = config["num_layers"]

    task_leaf    = args.task.split("/")[-1]
    tkey         = _task_key(args.task)
    max_length   = args.max_length or _MAX_LEN.get(tkey, 512)
    epochs       = args.epochs     or _EPOCHS.get(tkey, 3)
    ckpt_dir     = args.ckpt_dir   or f"results/gue_checkpoints/{args.model}_{task_leaf}"
    task_path    = os.path.join(args.gue_root, args.task)

    # HF token: CLI arg > env var
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    print(f"\n{'='*64}")
    print(f"  Model    : {args.model}  ({model_id})")
    print(f"  Task     : {args.task}  (max_len={max_length}, epochs={epochs})")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"{'='*64}\n")

    # ── Data splits ─────────────────────────────────────────────────────────
    # Check which dev split file exists
    dev_csv = os.path.join(task_path, "dev.csv")
    if not os.path.exists(dev_csv):
        dev_csv = os.path.join(task_path, "valid.csv")
    if not os.path.exists(dev_csv):
        # Some GUE tasks have only train+test; use test for monitoring
        dev_csv = os.path.join(task_path, "test.csv")
        print("[warn] No dev.csv found — using test.csv for validation monitoring.")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    import transformers  # noqa: PLC0415 — lazy import for evo2-container compat
    print("Loading tokenizer ...")
    tok_kwargs = {"trust_remote_code": True} if config.get("hf_trust_remote_code") else {}
    # Pin revisions so HF doesn't re-download and overwrite reviewed code files.
    # For DNABERT-2: pin the model repo revision (flash-attn patch lives there).
    # For NTv3: use code_revision to pin the ntv3_base_model *code* repo separately
    #   from the NTv3_650M_pre *weights* repo — they are different repos with
    #   different commit hashes.
    if "zhihan1996" in model_id:
        tok_kwargs["revision"] = "7bce263b15377fc15361f52cfab88f8b586abda0"
    if "InstaDeepAI" in model_id and "NTv3" in model_id:
        tok_kwargs["code_revision"] = "0ecff3637f0d3ba5b686d1095083218157c2ca34"
    if hf_token:
        tok_kwargs["token"] = hf_token
    is_generator = args.model in ("generator", "generator_prokaryote", "generator_prokaryote_1b")
    is_hybridna  = "Mishamq" in model_id or args.model == "hybridna"

    # For Generator 6-mer tokenizer: sequences must be multiples of 6 bp.
    # Round max_length down to the nearest multiple of 6 so truncation is clean.
    if is_generator:
        max_length = (max_length // 6) * 6 or 6

    if is_hybridna:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, model_max_length=max_length, padding_side="left", **tok_kwargs
        )
    elif is_generator:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, model_max_length=max_length, padding_side="right", **tok_kwargs
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, model_max_length=max_length, **tok_kwargs
        )
    if "InstaDeepAI" in model_id:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    if (is_hybridna or is_generator) and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("Loading datasets ...")
    train_ds = GUEDataset(os.path.join(task_path, "train.csv"), tokenizer, max_length)
    val_ds   = GUEDataset(dev_csv,                              tokenizer, max_length)
    test_ds  = GUEDataset(os.path.join(task_path, "test.csv"), tokenizer, max_length)
    num_labels = train_ds.num_labels
    print(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
          f"  num_labels={num_labels}")

    # ── Model ────────────────────────────────────────────────────────────────
    ckpt_state_path = Path(ckpt_dir) / "model_state.pt"
    ckpt_exists     = ckpt_state_path.exists()

    print(f"Loading base model from {model_id} ...")
    # NTv3 only registers AutoModelForMaskedLM (not AutoModelForSequenceClassification).
    # Load the backbone and attach a linear classification head manually.
    # HybriDNA/GENERator are causal LMs — load via AutoModelForCausalLM + mean-pool head.
    if is_hybridna:
        _repo_root = Path(__file__).resolve().parents[1]
        _stubs = str(_repo_root / "stubs")
        if _stubs not in sys.path:
            sys.path.insert(0, _stubs)
        backbone = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            use_mamba_kernels=False,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **tok_kwargs,
        )
        hidden_size = backbone.config.hidden_size
        model = _HybriDNAClassifier(backbone, hidden_size, num_labels)
    elif is_generator:
        backbone = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="auto",
            **tok_kwargs,
        )
        hidden_size = backbone.config.hidden_size
        model = _GeneratorClassifier(backbone, hidden_size, num_labels)
    elif "InstaDeepAI" in model_id and "NTv3" in model_id:
        backbone = transformers.AutoModelForMaskedLM.from_pretrained(
            model_id, **tok_kwargs
        )
        embed_dim = backbone.config.embed_dim
        model = _NTv3Classifier(backbone, embed_dim, num_labels)
    else:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels, **tok_kwargs
        )

    if ckpt_exists:
        print(f"Loading fine-tuned weights from {ckpt_state_path} ...")
        # strict=False: rotary embedding cos/sin caches are buffers that get
        # recomputed at runtime and may appear in older checkpoints as unexpected keys.
        model.load_state_dict(
            torch.load(ckpt_state_path, map_location="cpu"), strict=False
        )

    if not is_hybridna and not is_generator:  # device_map="auto" already placed these across devices
        model = model.to(args.device)

    # ── Fine-tuning ───────────────────────────────────────────────────────────
    if not ckpt_exists and not args.no_finetune:
        print(f"\nFine-tuning for {epochs} epochs ...")
        fine_tune(model, train_ds, val_ds, epochs, args.lr, args.batch,
                  args.device, ckpt_dir)
    elif args.no_finetune and not ckpt_exists:
        print("[warn] --no_finetune set but no checkpoint found — evaluating base model.")

    # ── SW index ─────────────────────────────────────────────────────────────
    sw_index = json.loads(Path(args.sw_index).read_text())
    entry    = sw_index.get(args.model, [])
    sw_list  = entry.get("results", entry) if isinstance(entry, dict) else entry
    if sw_list:
        print(f"\nFound {len(sw_list)} super row(s) for {args.model} in index:")
        for sw in sw_list:
            print(f"  layer={sw['layer']}  row={sw['row']}")
    else:
        print(f"\n[info] No super rows in index for {args.model}. "
              f"Run run_detection.py first, or rely on --sweep for task-based discovery.")

    # ── Three-condition ablation ──────────────────────────────────────────────
    print("\n── Three-condition ablation ──────────────────────────────────────")
    ablation_result = run_ablation(
        model, sw_list, test_ds, pattern, num_layers, args.device,
        n_structured_control=args.structured_rand,
    )

    # ── Layer sweep (optional) ────────────────────────────────────────────────
    sweep_result = None
    if args.sweep:
        print("\n── Layer sweep (task-specific super row discovery) ───────────────")
        sweep_result = layer_sweep(model, test_ds, pattern, num_layers, args.device)
        ablation_result["layer_sweep"] = sweep_result

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Task: {args.task}")
    bl = ablation_result["baseline"]
    print(f"  Baseline    acc={bl['accuracy']:.4f}  mcc={bl['mcc']:.4f}")
    if ablation_result["pruned_sw"] is not None:
        sw = ablation_result["pruned_sw"]
        print(f"  Pruned SW   acc={sw['accuracy']:.4f}  mcc={sw['mcc']:.4f}"
              f"  Δacc={ablation_result['delta_acc_sw_pct']:+.2f}%"
              f"  Δmcc={ablation_result['delta_mcc_sw_pct']:+.2f}%")
    rm = ablation_result["pruned_rand_mean"]
    print(f"  Rand ctrl   acc={rm['accuracy']:.4f}  mcc={rm['mcc']:.4f}")
    if "pruned_struct_rand_mean" in ablation_result:
        sr = ablation_result["pruned_struct_rand_mean"]
        print(f"  Struct rand acc={sr['accuracy']:.4f}  mcc={sr['mcc']:.4f}"
              f"  (layer+row-pattern matched, n={args.structured_rand})")
    if sweep_result:
        top = sweep_result[0]
        print(f"  Worst sweep layer={top['layer']}  row={top['row']}"
              f"  Δacc={top['delta_acc']:+.4f}  Δmcc={top['delta_mcc']:+.4f}")
    print(f"{'='*64}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}
    key = f"{args.model}/{args.task}"
    existing[key] = {
        "model":       args.model,
        "task":        args.task,
        "max_length":  max_length,
        "epochs":      epochs,
        "ckpt_dir":    ckpt_dir,
        **ablation_result,
    }
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"Results saved to {args.out}  (key: '{key}')")


if __name__ == "__main__":
    main()
