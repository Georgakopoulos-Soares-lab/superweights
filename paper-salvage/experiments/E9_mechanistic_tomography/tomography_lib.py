"""
Shared intervention/measurement engine for E9 mechanistic tomography.

Reuses existing, already-validated pieces rather than reinventing them:
  - models.WRAPPER_MAP (model loading, pinned DNABERT-2 revision, down_proj resolution)
  - scripts/mechanism/run_ensemble_encoding.read_windows/gc_frac/dinuc_freqs
  - scripts/evaluation/run_gue_ablation._save_row/_restore_row (row save/restore idiom)

New in this file (E9-specific, not previously implemented anywhere in the repo):
  - _scale_rows: generalizes zero-ablation to arbitrary alpha = 1 - epsilon*a_i per row,
    applied to a whole mask (list of (layer,row) coordinates) at once.
  - DNABERT-2 fixed-batch MLM loss under an arbitrary multi-row mask (generalizes
    run_pretrained_epistasis.py's single/pair-only ablation to arbitrary subset masks).
  - GENERator dose-response generation under an arbitrary alpha for a single row
    (thin wrapper around the run_sw_steering.py generation loop, parameterized by alpha
    instead of raw scale so both models share the same alpha_i = 1 - epsilon*a_i framing).

Intervention semantics (row identity, point in forward pass, before/after
activation/residual/LayerNorm) are documented in INTERVENTION_BASIS.md — this file
implements exactly that ontology and nothing else.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "mechanism"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from run_ensemble_encoding import read_windows, gc_frac, dinuc_freqs  # noqa: E402
from run_gue_ablation import _resolve_module, _save_row, _restore_row  # noqa: E402

SEED = 42
DNABERT2_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
HG38_FASTA = str(ROOT / "data/reference/hg38/hg38.fa")
HG38_BED = str(ROOT / "data/regions/hg38/random_262kb.bed")


# ── generic multi-row scaling (the new E9 primitive) ─────────────────────────

def _scale_rows(model, pattern: str, coords: Sequence[tuple[int, int]],
                 alphas: Sequence[float]) -> list[torch.Tensor]:
    """Set weight.data[row,:] = saved_row * alpha for each (layer,row) in coords.

    Returns the list of saved (pre-intervention) rows, same order as coords, so the
    caller can restore with _restore_rows. alpha=1.0 is untouched, alpha=0.0 is the
    existing zero-ablation convention, alpha=0.5 is E9's new partial-suppression point.
    """
    saved = [_save_row(model, pattern, l, r) for l, r in coords]
    with torch.no_grad():
        for (l, r), s, a in zip(coords, saved, alphas):
            m = _resolve_module(model, pattern, l)
            m.weight.data[r, :] = s * a
    return saved


def _restore_rows(model, pattern: str, coords: Sequence[tuple[int, int]],
                   saved: Sequence[torch.Tensor]) -> None:
    for (l, r), s in zip(coords, saved):
        _restore_row(model, pattern, l, r, s)


def with_mask(model, pattern: str, coords: Sequence[tuple[int, int]],
              alphas: Sequence[float], fn):
    """Apply a multi-row alpha mask, run fn(), restore, return fn()'s result."""
    saved = _scale_rows(model, pattern, coords, alphas)
    try:
        return fn()
    finally:
        _restore_rows(model, pattern, coords, saved)


def alphas_for_mask(a: Sequence[int], epsilon: float) -> list[float]:
    """alpha_i = 1 - epsilon * a_i, the frozen E9 parameterization."""
    return [1.0 - epsilon * ai for ai in a]


# ── DNABERT-2: pretrained MLM loading + fixed-batch loss ─────────────────────

def load_dnabert2_pretrained(device: str = "cuda"):
    """Load DNABERT-2 via the canonical wrapper (pinned revision, transformers>=5
    config compatibility fixes already applied there), then apply the same
    Triton flash-attn workaround run_pretrained_epistasis.py documents (MLM has zero
    attention dropout and reaches the broken Triton kernel; fine-tuned paths don't).
    """
    from models import WRAPPER_MAP
    cfg = yaml.safe_load((ROOT / "configs/dnabert2.yaml").read_text())
    assert cfg["model_id"] == "zhihan1996/DNABERT-2-117M"
    w = WRAPPER_MAP["dnabert2"](cfg)
    w.load()
    model = w.model
    mod = sys.modules[type(model).__module__]
    patched = getattr(mod, "flash_attn_qkvpacked_func", None) is not None
    if patched:
        mod.flash_attn_qkvpacked_func = None
    return model.eval(), w.tokenizer, cfg, patched


def read_fasta_windows_mlm(fasta: str, bed: str, n_windows: int, win_bp: int,
                            rng: random.Random) -> list[str]:
    """Same logic as run_pretrained_epistasis._read_fasta_windows."""
    from pyfaidx import Fasta
    fa = Fasta(fasta, as_raw=True)
    regions = []
    with open(bed) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                regions.append((p[0], int(p[1]), int(p[2])))
    rng.shuffle(regions)
    seqs = []
    for chrom, s, e in regions:
        if len(seqs) >= n_windows:
            break
        if e - s < win_bp:
            continue
        st = rng.randrange(s, e - win_bp)
        seq = str(fa[chrom][st:st + win_bp]).upper()
        if seq.count("N") / max(len(seq), 1) < 0.01:
            seqs.append(seq)
    return seqs


def build_fixed_batches(tok, seqs, device, max_len, batch_size, mask_prob, seed=SEED):
    """Same logic as run_pretrained_epistasis._build_fixed_batches — one fixed mask
    realization reused across every condition so deltas are paired."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for i in range(0, len(seqs), batch_size):
        enc = tok(seqs[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)
        ids = enc["input_ids"]
        attn = enc.get("attention_mask")
        if attn is None:
            attn = torch.ones_like(ids)
        special = torch.zeros_like(ids, dtype=torch.bool)
        for sid in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id):
            if sid is not None:
                special |= ids == sid
        sel = (torch.rand(ids.shape, generator=g) < mask_prob) & (attn == 1) & ~special
        if sel.sum() == 0:
            continue
        inp = ids.clone()
        inp[sel] = tok.mask_token_id
        lab = ids.clone()
        lab[~sel] = -100
        batches.append({"input_ids": inp.to(device),
                        "attention_mask": attn.to(device),
                        "labels": lab.to(device),
                        "n_masked": int(sel.sum())})
    return batches


@torch.no_grad()
def mlm_loss_per_batch(model, batches) -> list[tuple[float, int]]:
    """(sum_loss_over_tokens, n_masked) per batch -- lets a caller recompute the
    weighted-mean loss over any resampled subset of batches without rerunning the model
    (used for the batch-level bootstrap declared in the E9 prereg)."""
    out = []
    for b in batches:
        o = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                  labels=b["labels"])
        out.append((float(o.loss) * b["n_masked"], b["n_masked"]))
    return out


def aggregate_per_batch(per_batch: list[tuple[float, int]]) -> float:
    tot = sum(s for s, _ in per_batch)
    n = sum(k for _, k in per_batch)
    return tot / max(n, 1)


def mlm_loss(model, batches) -> float:
    return aggregate_per_batch(mlm_loss_per_batch(model, batches))


def dnabert2_response(model, pattern, coords, a, epsilon, batches) -> float:
    """MLM loss under mask `a` (list of 0/1, same length/order as coords) at scale
    epsilon, relative caller must subtract baseline separately."""
    active = [(c, ai) for c, ai in zip(coords, a) if ai]
    if not active:
        return mlm_loss(model, batches)
    ac, aa = zip(*active)
    alphas = alphas_for_mask(aa, epsilon)
    return with_mask(model, pattern, list(ac), alphas, lambda: mlm_loss(model, batches))


class ChannelNormHook:
    """Forward hook on DNABERT-2's layer-9 encoder output, capturing mean |h| for a fixed
    set of channels (H6's residual-norm secondary endpoint; explicit-layer-9 convention,
    see PROVENANCE_AND_BASELINES.md). Accumulates across whatever forward passes run while
    armed, then `read()` returns the running mean and resets."""

    def __init__(self, model, channels: Sequence[int], layer: int = 9):
        m = _resolve_module(model, "bert.encoder.layer.{i}", layer)
        self.channels = list(channels)
        self._sum = None
        self._n = 0
        self._handle = m.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        vals = h[..., self.channels].detach().abs().mean(dim=(0, 1))
        if self._sum is None:
            self._sum = vals.clone()
        else:
            self._sum += vals
        self._n += 1

    def read(self) -> list[float]:
        if self._n == 0:
            return [float("nan")] * len(self.channels)
        vals = (self._sum / self._n).tolist()
        self._sum, self._n = None, 0
        return vals

    def remove(self):
        self._handle.remove()


def dnabert2_response_per_batch(model, pattern, coords, a, epsilon, batches):
    active = [(c, ai) for c, ai in zip(coords, a) if ai]
    if not active:
        return mlm_loss_per_batch(model, batches)
    ac, aa = zip(*active)
    alphas = alphas_for_mask(aa, epsilon)
    return with_mask(model, pattern, list(ac), alphas,
                     lambda: mlm_loss_per_batch(model, batches))


# ── GENERator: dose-response generation ───────────────────────────────────────

def load_generator():
    from models import WRAPPER_MAP
    cfg = yaml.safe_load((ROOT / "configs/generator.yaml").read_text())
    w = WRAPPER_MAP["generator"](cfg)
    w.load()
    return w.model, w.tokenizer, cfg


def generator_gc_response(model, tok, pattern, layer, row, alpha, prompts,
                           max_new, seed=SEED) -> tuple[list[float], list[str]]:
    """Single-row alpha scaling + generation + GC readout — same generation loop as
    run_sw_steering.py's `generate()`, parameterized by alpha instead of raw scale
    (identical semantics: alpha IS the raw multiplicative scale in that script)."""
    dev = next(model.parameters()).device
    saved = _save_row(model, pattern, layer, row)
    m = _resolve_module(model, pattern, layer)
    with torch.no_grad():
        m.weight.data[row, :] = saved * alpha
    gcs, seqs = [], []
    try:
        for i, p in enumerate(prompts):
            enc = tok(p, return_tensors="pt")
            ids = enc["input_ids"].to(dev)
            torch.manual_seed(seed + i)
            with torch.no_grad():
                o = model.generate(input_ids=ids, max_new_tokens=max_new,
                                   do_sample=True, top_k=50, temperature=1.0,
                                   pad_token_id=getattr(tok, "pad_token_id", None) or 0)
            new = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
            new = "".join(c for c in new.upper() if c in "ACGT")
            if len(new) >= 10:
                gcs.append(gc_frac(new))
                seqs.append(new)
    finally:
        _restore_row(model, pattern, layer, row, saved)
    return gcs, seqs
