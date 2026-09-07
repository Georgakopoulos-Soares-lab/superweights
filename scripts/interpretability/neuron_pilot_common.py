# scripts/interpretability/neuron_pilot_common.py
"""
Shared primitives for the DNABERT-2 single-neuron pilot (Apple-paper-style single-neuron
causal analysis: "A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language
Models"). Reused across discovery, causal-intervention, structural-mapping, controls,
base-vs-finetuned and broadcast-impulse scripts in this directory.

DNABERT-2's gated MLP (BertGatedLinearUnitMLP, mlp.gated_layers / mlp.act / mlp.wo):
    hidden      = gated_layers(x)            # (.., 2*d_ffn), no bias
    gate, up    = hidden[..., :d_ffn], hidden[..., d_ffn:]
    h           = GELU(gate) * up            # (.., d_ffn)   <- the neuron activation we probe
    out         = wo(h)                       # (.., d_model) = down-projection

We hook `mlp.wo` with a forward_pre_hook: its single positional argument *is* h (dropout is
the identity in eval mode). This matches the down_proj_pattern already used elsewhere in this
repo ("bert.encoder.layer.{i}.mlp.wo"), so no new module-resolution logic is needed.

IMPORTANT — DNABERT-2 (Mosaic BERT) unpads internally: `BertEncoder.forward` calls
bert_padding.unpad_input() ONCE, right after the embeddings, converting the [B, L, D] hidden
states into a FLAT [total_nnz, D] tensor (total_nnz = number of real, non-padding tokens
across the whole batch) that stays flat through every transformer block — including mlp.wo —
and is only re-padded back to [B, L, D] once, after the final block. So `h` observed at our
mlp.wo hook has shape (total_nnz, d_ffn), NOT (B, L, d_ffn). Row j's (batch, position) is
recoverable via flat_batch_offsets(): batch b's rows occupy
[offsets[b], offsets[b] + attention_mask[b].sum()) in ascending position order (see
bert_padding.unpad_input's `indices = nonzero(attention_mask.flatten())`, which is exactly
this cumulative-offset layout for standard right-padding). All hooks/utilities below operate
on this flat 2D convention.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

D_FFN = 3072
D_MODEL = 768
# Exon/intron boundary in GUE splice/reconstructed (400bp windows), empirically confirmed:
# label 0 (acceptor) has "AG" at chars 198-199, label 1 (donor) has "GT" at chars 200-201,
# in every example — the junction is at character offset 200 for all three classes.
JUNCTION_CHAR_OFFSET = 200


def get_wo_module(backbone, layer_idx: int):
    """backbone = a BertModel (model.bert), or anything with .encoder.layer[i].mlp.wo"""
    return backbone.encoder.layer[layer_idx].mlp.wo


def get_gated_layers_module(backbone, layer_idx: int):
    return backbone.encoder.layer[layer_idx].mlp.gated_layers


# ─────────────────────────────────────────────────────────────────────────────
# Junction localization (per-example, via tokenizer offset mapping)
# ─────────────────────────────────────────────────────────────────────────────

def junction_token_index(offsets: torch.Tensor, char_offset: int = JUNCTION_CHAR_OFFSET) -> int:
    """
    offsets: LongTensor [seq_len, 2] from tokenizer(..., return_offsets_mapping=True) for ONE
    example (special tokens have (0,0)). Returns the token index whose [start,end) span
    contains char_offset; falls back to the nearest real (non-special) token if the offset
    lands exactly on a boundary/gap.
    """
    starts = offsets[:, 0]
    ends = offsets[:, 1]
    hit = ((starts <= char_offset) & (ends > char_offset)).nonzero(as_tuple=True)[0]
    if len(hit) > 0:
        return int(hit[0].item())
    real = (ends > starts).nonzero(as_tuple=True)[0]
    if len(real) == 0:
        return 0
    dist = torch.where(
        starts[real] > char_offset,
        starts[real] - char_offset,
        torch.where(ends[real] <= char_offset, char_offset - ends[real] + 1,
                    torch.zeros_like(starts[real])),
    )
    return int(real[dist.argmin()].item())


def junction_window(offsets: torch.Tensor, half_window: int = 4,
                     char_offset: int = JUNCTION_CHAR_OFFSET) -> List[int]:
    center = junction_token_index(offsets, char_offset)
    seq_len = offsets.shape[0]
    return [p for p in range(center - half_window, center + half_window + 1) if 0 <= p < seq_len]


def flat_batch_offsets(attention_mask: torch.Tensor) -> torch.Tensor:
    """Given a [B, L] attention mask, returns a [B] LongTensor of each example's starting row
    index within the flat (total_nnz, D) unpadded representation DNABERT-2 uses internally
    (see module docstring). Batch b's valid (local) position p maps to flat row
    offsets[b] + p, valid for p < attention_mask[b].sum()."""
    lens = attention_mask.sum(dim=1)
    return torch.cumsum(lens, dim=0) - lens


# ─────────────────────────────────────────────────────────────────────────────
# Discovery: activation + gradient capture
# ─────────────────────────────────────────────────────────────────────────────

class ActivationGradCapture:
    """forward_pre_hook: retains h and its gradient for one layer. Non-destructive (returns
    None so the input is unmodified). Safe to use both under normal grad tracking (e.g.
    discovery's forward+backward pass, where .grad is populated after backward()) and inside
    torch.no_grad() (e.g. the broadcast-impulse trace, which only needs the activation value
    itself) — retain_grad() is skipped when h.requires_grad is False rather than raising."""

    def __init__(self):
        self.h = None

    def __call__(self, module, args):
        h = args[0]
        if h.requires_grad:
            h.retain_grad()
        self.h = h
        return None


class MultiLayerActivationGradCapture:
    """Registers ActivationGradCapture on mlp.wo across a set of layers simultaneously."""

    def __init__(self, backbone, layer_indices):
        self.layer_indices = list(layer_indices)
        self.captures = {li: ActivationGradCapture() for li in self.layer_indices}
        self._handles = [
            get_wo_module(backbone, li).register_forward_pre_hook(self.captures[li])
            for li in self.layer_indices
        ]

    def h(self, layer_idx: int):
        return self.captures[layer_idx].h

    def grad(self, layer_idx: int):
        t = self.captures[layer_idx].h
        return None if t is None else t.grad

    def remove(self):
        for hd in self._handles:
            hd.remove()
        self._handles = []


# ─────────────────────────────────────────────────────────────────────────────
# Causal intervention: alpha-interpolation hook
# ─────────────────────────────────────────────────────────────────────────────

class NeuronIntervention:
    """
    forward_pre_hook on mlp.wo implementing, for selected FLAT rows of h (see module
    docstring — h has shape (total_nnz, d_ffn), one row per real token across the whole
    batch, no padding rows exist):

        h'[row, neuron_idx] = h[row, neuron_idx] + alpha * (target - h[row, neuron_idx])
                               for row in row_indices

    target : float   (0.0 = zero-ablation; class-mean activation for suppress/amplify)
    alpha  : float   interpolation coefficient (0 = clean/no-op, 1 = fully at target,
                      >1 extrapolates past the target)
    row_indices : list[int] | None. Flat row indices into h's first dimension. None means
                  "every row" — this is safe and correct as "all positions" because every row
                  in the flat representation is already a real (non-padding) token; there is
                  no padding to accidentally touch. For a junction-only (or any other
                  position-restricted) scope, the caller must convert per-example local token
                  positions to flat rows via flat_batch_offsets() before constructing this
                  hook (see run_neuron_causal_intervention_dnabert2.evaluate_full).

    Records self.n_modified (rows actually touched) so tests and callers can sanity-check
    scope without inspecting the tensor directly.
    """

    def __init__(self, neuron_idx: int, target: float, alpha: float,
                 row_indices: Optional[List[int]] = None):
        self.neuron_idx = neuron_idx
        self.target = float(target)
        self.alpha = float(alpha)
        self.row_indices = row_indices
        self.n_modified = 0

    def __call__(self, module, args):
        h = args[0]
        if self.alpha == 0.0 or (self.row_indices is not None and len(self.row_indices) == 0):
            self.n_modified = 0
            return None
        h = h.clone()
        idx = slice(None) if self.row_indices is None else self.row_indices
        clean = h[idx, self.neuron_idx]
        h[idx, self.neuron_idx] = clean + self.alpha * (self.target - clean)
        self.n_modified = h.shape[0] if self.row_indices is None else len(self.row_indices)
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Binary splice-vs-non-splice framing (positive = donor(1) U acceptor(0), negative = 2)
# ─────────────────────────────────────────────────────────────────────────────

def splice_binary_logodds(logits: torch.Tensor):
    """
    logits: [B, 3] raw classifier logits (GUE splice/reconstructed label order:
            0=acceptor, 1=donor, 2=negative).
    Returns (log_odds, p_pos): log_odds = log P(pos) - log P(neg) ("splice class log-odds"),
    p_pos = P(label in {0,1}).
    """
    logp = torch.log_softmax(logits, dim=-1)
    log_p_pos = torch.logsumexp(logp[:, [0, 1]], dim=-1)
    log_p_neg = logp[:, 2]
    log_odds = log_p_pos - log_p_neg
    p_pos = log_p_pos.exp()
    return log_odds, p_pos


# ─────────────────────────────────────────────────────────────────────────────
# numpy-only AUROC / AUPRC (no sklearn dependency — matches this repo's existing precedent
# in scripts/evaluation/run_gue_ablation.py: "sklearn replaced by numpy implementations")
# ─────────────────────────────────────────────────────────────────────────────

def auroc_numpy(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUROC via the Mann-Whitney U / rank-sum identity."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks across ties
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sum_rank_per_val = np.zeros(len(uniq))
    np.add.at(sum_rank_per_val, inv, ranks)
    avg_rank = (sum_rank_per_val / counts)[inv]
    sum_ranks_pos = avg_rank[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def auprc_numpy(labels: np.ndarray, scores: np.ndarray) -> float:
    """Binary AUPRC as the standard step-function average precision
    AP = sum_n (R_n - R_{n-1}) * P_n (matches sklearn.metrics.average_precision_score
    exactly — verified against it; a naive trapezoidal rule is NOT used here because it
    is systematically optimistic relative to this standard definition)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels_sorted = labels[order]
    scores_sorted = scores[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    # collapse to one point per distinct score threshold (last occurrence of each tied
    # block) — matches sklearn's precision_recall_curve tie-handling exactly.
    distinct = np.r_[np.where(np.diff(scores_sorted) != 0)[0], len(scores_sorted) - 1]
    tp = tp[distinct]
    fp = fp[distinct]
    precision = tp / (tp + fp)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def flip_rates(pred_before: np.ndarray, pred_after: np.ndarray) -> dict:
    """pred_* are binary (0/1) splice-vs-not predictions (same length, paired by example)."""
    pred_before = np.asarray(pred_before)
    pred_after = np.asarray(pred_after)
    pos_mask = pred_before == 1
    neg_mask = pred_before == 0
    p2n = int(((pred_before == 1) & (pred_after == 0)).sum())
    n2p = int(((pred_before == 0) & (pred_after == 1)).sum())
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    return {
        "pos_to_neg_flips": p2n,
        "pos_to_neg_rate": p2n / n_pos if n_pos else float("nan"),
        "neg_to_pos_flips": n2p,
        "neg_to_pos_rate": n2p / n_neg if n_neg else float("nan"),
        "total_flip_rate": (p2n + n2p) / len(pred_before),
        "n_pos_before": n_pos,
        "n_neg_before": n_neg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ||U_k||_F structural decomposition (imports the canonical implementation — see
# scripts/analysis/run_dnabert2_uk_audit.py for the source of truth; duplicated as a
# thin re-export point is avoided by importing directly where needed)
# ─────────────────────────────────────────────────────────────────────────────

def neuron_contribution_to_row(Wd: np.ndarray, Wg: np.ndarray, Wu: np.ndarray, row: int) -> np.ndarray:
    """
    C[row, i] = Wd[row,i]^2 * ||Wg[i,:]||^2 * ||Wu[i,:]||^2   for all i in 0..d_ffn-1.
    Sum over i equals ||U_row||_F^2 (see run_dnabert2_uk_audit.frob_norm_uk).
    Wg, Wu: (d_ffn, d_model); Wd: (d_model, d_ffn).
    """
    norm_g2 = (Wg ** 2).sum(axis=1)
    norm_u2 = (Wu ** 2).sum(axis=1)
    weight = norm_g2 * norm_u2
    return (Wd[row, :] ** 2) * weight
