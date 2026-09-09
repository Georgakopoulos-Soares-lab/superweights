"""
Shared measurement engine for E10 (both arms).

Reuses, unmodified:
  - scripts/evaluation/run_gue_ablation._resolve_module/_save_row/_restore_row
  - E9's alpha_i = 1 - epsilon*a_i row-scale ontology (INTERVENTION_BASIS.md)

New here (not previously implemented): causal-LM NLL/KL/entropy under a row mask, and a
WikiText-2 window builder shared by both arms. Everything else is E9 machinery.

Governing protocol: PREREG_E10_nlp_architecture_causal_v2.md (locked
3e4b991d71b38cfe15550a57c10c73323cb3b751ba3c768d6c95fc4989833a8a).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from run_gue_ablation import _resolve_module, _save_row, _restore_row  # noqa: E402

SEED = 42
WIKITEXT = ("wikitext", "wikitext-2-raw-v1", "test")


# ── row-scale intervention (E9 ontology, generalized to any model) ───────────

def scale_rows(model, pattern: str, coords: Sequence[tuple[int, int]],
               alphas: Sequence[float]) -> list[torch.Tensor]:
    saved = [_save_row(model, pattern, l, r) for l, r in coords]
    with torch.no_grad():
        for (l, r), s, a in zip(coords, saved, alphas):
            _resolve_module(model, pattern, l).weight.data[r, :] = s * a
    return saved


def restore_rows(model, pattern: str, coords, saved) -> None:
    for (l, r), s in zip(coords, saved):
        _restore_row(model, pattern, l, r, s)


class masked:
    """Context manager: apply alpha mask on enter, restore on exit (always)."""

    def __init__(self, model, pattern, coords, alphas):
        self.model, self.pattern, self.coords, self.alphas = model, pattern, coords, alphas

    def __enter__(self):
        self.saved = scale_rows(self.model, self.pattern, self.coords, self.alphas)
        return self.model

    def __exit__(self, *exc):
        restore_rows(self.model, self.pattern, self.coords, self.saved)
        return False


def alphas_for_mask(a: Sequence[int], epsilon: float) -> list[float]:
    """alpha_i = 1 - epsilon * a_i -- the frozen E9/E10 parameterization."""
    return [1.0 - epsilon * ai for ai in a]


# ── frozen WikiText-2 windows (ENDPOINTS.md) ─────────────────────────────────

def build_windows(tokenizer, n_windows: int, max_tokens: int = 512, seed: int = SEED):
    """N windows of exactly max_tokens tokens, from shuffled non-empty WikiText-2 test
    lines concatenated in order until the window fills. Text selection is seeded
    identically for every model; tokenization is each model's own (ENDPOINTS.md)."""
    import random
    from datasets import load_dataset

    ds = load_dataset(WIKITEXT[0], WIKITEXT[1], split=WIKITEXT[2])
    lines = [t.strip() for t in ds["text"] if t.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)

    windows, buf, i = [], [], 0
    while len(windows) < n_windows and i < len(lines):
        buf.append(lines[i])
        i += 1
        ids = tokenizer("\n".join(buf), return_tensors=None)["input_ids"]
        if len(ids) >= max_tokens:
            windows.append(torch.tensor(ids[:max_tokens], dtype=torch.long))
            buf = []
    if len(windows) < n_windows:
        raise RuntimeError(f"only built {len(windows)}/{n_windows} windows")
    return windows


def batch_windows(windows, batch_size):
    return [torch.stack(windows[i:i + batch_size])
            for i in range(0, len(windows), batch_size)]


# ── causal-LM endpoint (ARM A) ───────────────────────────────────────────────

@torch.no_grad()
def causal_lm_batch(model, ids: torch.Tensor, baseline_logprobs: torch.Tensor | None = None):
    """Returns (sum_nll, n_tokens, logprobs, mean_kl_vs_baseline, mean_entropy).

    sum_nll/n_tokens are kept separately (not a batch mean) so a context-level
    bootstrap can reweight batches exactly, matching E9's per-batch convention.
    KL is KL(baseline || intervened), mean over predicted positions.
    """
    logits = model(input_ids=ids).logits.float()
    shift_logits = logits[:, :-1, :]
    shift_labels = ids[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)

    nll = F.nll_loss(logprobs.reshape(-1, logprobs.shape[-1]),
                     shift_labels.reshape(-1), reduction="sum")
    n_tok = shift_labels.numel()
    entropy = float((-(logprobs.exp() * logprobs).sum(-1)).mean())

    kl = float("nan")
    if baseline_logprobs is not None:
        kl = float((baseline_logprobs.exp() *
                    (baseline_logprobs - logprobs)).sum(-1).mean())
    return float(nll), int(n_tok), logprobs, kl, entropy


# ── MLM endpoint (ARM B) -- E9's build_fixed_batches logic, text instead of DNA ──

def build_fixed_mlm_batches(tok, windows, device, batch_size, mask_prob=0.15, seed=SEED):
    """One fixed mask realization reused across every condition so deltas are paired.
    Same logic as E9 tomography_lib.build_fixed_batches (which is domain-agnostic);
    inputs here are pre-tokenized text windows rather than FASTA strings."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for i in range(0, len(windows), batch_size):
        ids = torch.stack(windows[i:i + batch_size])
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
        batches.append({"input_ids": inp.to(device), "attention_mask": attn.to(device),
                        "labels": lab.to(device), "n_masked": int(sel.sum())})
    return batches


@torch.no_grad()
def mlm_loss_per_batch(model, batches):
    """(sum_loss_over_masked_tokens, n_masked) per batch -- E9's convention verbatim,
    so a batch-level bootstrap can reweight without rerunning the model."""
    out = []
    for b in batches:
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        lab = b["labels"]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               lab.reshape(-1), ignore_index=-100, reduction="sum")
        out.append((float(loss), int((lab != -100).sum())))
    return out


def weighted_mean(per_batch):
    s = sum(x[0] for x in per_batch)
    n = sum(x[1] for x in per_batch)
    return s / max(n, 1)
