"""
hooks/ablation_trace.py
------------------------
Tier 1 core primitive (Agent Task Spec Section 1): the ablation difference
trace. Implemented once here and reused by Tiers 1-3.

For a model wrapper, a super-weight layer L and output coordinate k, this
runs matched clean / ablated forward passes on one probe sequence, captures
the residual-stream hidden state emitted by *every* block, and computes:

  - per-layer on_coord_energy / total_energy / coord_fraction
  - summary scalars: coordinate_retention, total_retention, dispersion_index,
    participation_ratio, top1_fraction

Architecture-specific glue (how to enumerate "blocks", how to unwrap a
block's forward-hook output into a plain (seq, d_model) tensor) is
centralized in get_blocks() / unwrap_block_output() -- add a new model there
and everything else (the trace math) is unchanged.

The ablation hook itself patches the *output* of the model wrapper's
registered down_proj module (Pattern B from the codebase survey: the same
technique used in models/evo1_wrapper.py::compute_perplexity_ablated and
scripts/interpretability/run_sw_causal_tracing.py), not an in-place weight
edit -- this works uniformly across architectures without permanently
mutating weights.
"""
from __future__ import annotations

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Architecture-specific glue
# ─────────────────────────────────────────────────────────────────────────────

def get_blocks(model_key: str, wrapper):
    """Return the ordered list of block modules whose output is the
    residual-stream hidden state passed to the next block."""
    model = wrapper.model
    if model_key in ("generator", "generator_prokaryote", "generator_prokaryote_1b"):
        return list(model.model.layers)
    if model_key == "dnabert2":
        bert = model.bert if hasattr(model, "bert") else model
        return list(bert.encoder.layer)
    if model_key == "ntv3":
        return list(model.core.transformer_blocks)
    if model_key == "evo1":
        return list(model.blocks)
    if model_key == "caduceus":
        return list(model.backbone.layers)
    raise ValueError(f"No block resolver registered for model_key={model_key!r}")


def unwrap_block_output(model_key: str, output) -> torch.Tensor:
    """Return a detached, cloned float32 (seq, d_model) tensor for one block's
    forward-hook output, regardless of whether the block returns a plain
    tensor, a tuple (Llama/StripedHyena-style), a dict (NTv3 returns
    {"embeddings": ..., "attention_weights": ...}), or Caduceus's
    (hidden_states, residual) fused-add-norm tuple -- mamba_ssm's Block
    class defers the residual add for kernel-fusion reasons, so the true
    residual-stream value at a Caduceus block boundary is the ELEMENT-WISE
    SUM of the two tuple entries, not just the first one."""
    if isinstance(output, dict):
        h = output["embeddings"]
    elif model_key == "caduceus" and isinstance(output, tuple):
        h = output[0] + output[1]
    elif isinstance(output, tuple):
        h = output[0]
    else:
        h = output
    if h.dim() == 3:
        h = h[0]
    # .clone() is required even after .detach().float(): if h is already
    # float32, .float() is a no-op view onto the same storage, and a later
    # in-place op inside the model (e.g. the next block's residual add) would
    # silently corrupt an uncloned capture.
    return h.detach().float().clone()


def install_ablation_hook(wrapper, layer: int, row: int, model_key: str = None):
    """Zero output coordinate `row` of the down_proj at `layer` on every
    forward pass, via forward hook (not in-place weight edit).

    Caduceus has no down_proj: its block returns (hidden_states, residual)
    where `hidden_states` is that block's OWN fresh mixer contribution
    (pre-accumulation) and `residual` is the carried history from all prior
    blocks. Ablating "this block's contribution to coordinate row" therefore
    means zeroing row of `hidden_states` only, leaving `residual` (history)
    untouched -- the semantic analog of zeroing a down_proj output before
    its residual add.
    """
    if model_key == "caduceus":
        block = wrapper.model.backbone.layers[layer]

        def _hook(mod, inp, output):
            hs, res = output
            hs = hs.clone()
            hs[..., row] = 0.0
            return (hs, res)

        return block.register_forward_hook(_hook)

    module = wrapper.get_target_module(layer)

    def _hook(mod, inp, output):
        if isinstance(output, tuple):
            o = output[0].clone()
            o[..., row] = 0.0
            return (o,) + tuple(output[1:])
        else:
            o = output.clone()
            o[..., row] = 0.0
            return o

    return module.register_forward_hook(_hook)


def install_ablation_hooks_multi(wrapper, pairs, model_key: str = None):
    """Ablate a SET of (layer, row) super-weight coordinates simultaneously.

    Generalization of install_ablation_hook() to the multi-row / multi-layer
    case needed by Phase A1: DNABERT-2's super weight is a 10-row ensemble
    spanning layers {3,5,6,7,9}, and Evo1's index likewise holds 10 rows at
    layer 11. Rows sharing a layer are zeroed by a single hook on that layer.

    Returns a list of handles; caller must remove all of them.
    """
    from collections import defaultdict

    by_layer = defaultdict(list)
    for layer, row in pairs:
        by_layer[layer].append(row)

    handles = []
    for layer, rows in by_layer.items():
        rows_t = sorted(set(rows))

        if model_key == "caduceus":
            block = wrapper.model.backbone.layers[layer]

            def _make(rows_t=rows_t):
                def _hook(mod, inp, output):
                    hs, res = output
                    hs = hs.clone()
                    hs[..., rows_t] = 0.0
                    return (hs, res)
                return _hook

            handles.append(block.register_forward_hook(_make()))
        else:
            module = wrapper.get_target_module(layer)

            def _make(rows_t=rows_t):
                def _hook(mod, inp, output):
                    if isinstance(output, tuple):
                        o = output[0].clone()
                        o[..., rows_t] = 0.0
                        return (o,) + tuple(output[1:])
                    o = output.clone()
                    o[..., rows_t] = 0.0
                    return o
                return _hook

            handles.append(module.register_forward_hook(_make()))

    return handles


class BlockRecorder:
    """Context manager: hooks every block, captures its output each forward pass."""

    def __init__(self, model_key: str, blocks: list):
        self.model_key = model_key
        self.blocks = blocks
        self.outputs = [None] * len(blocks)
        self._handles = []

    def _make_hook(self, i):
        def hook(module, inp, output):
            self.outputs[i] = unwrap_block_output(self.model_key, output)
        return hook

    def __enter__(self):
        self.outputs = [None] * len(self.blocks)
        self._handles = [blk.register_forward_hook(self._make_hook(i))
                          for i, blk in enumerate(self.blocks)]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Core primitive
# ─────────────────────────────────────────────────────────────────────────────

def run_clean_trace(wrapper, model_key: str, sequence: str) -> list:
    """Run one clean forward pass, return per-block hidden states (no ablation)."""
    blocks = get_blocks(model_key, wrapper)
    with BlockRecorder(model_key, blocks) as rec:
        wrapper.forward(sequence)
        return list(rec.outputs)


def run_difference_trace(wrapper, model_key: str, sw_layer: int, sw_row: int,
                          sequence: str) -> dict:
    """
    Run matched clean/ablated forward passes on `sequence`, ablating output
    coordinate `sw_row` of the down_proj at `sw_layer`, and compute the
    per-layer difference-trace scalars defined in Task Spec Section 1.2.

    Returns a dict with keys: on_coord_energy, total_energy, coord_fraction
    (each a length-n_layers list), plus the summary scalars
    coordinate_retention, total_retention, dispersion_index,
    participation_ratio, top1_fraction, p_star.
    """
    blocks = get_blocks(model_key, wrapper)
    n_layers = len(blocks)

    clean = run_clean_trace(wrapper, model_key, sequence)

    handle = install_ablation_hook(wrapper, sw_layer, sw_row, model_key=model_key)
    try:
        with BlockRecorder(model_key, blocks) as rec:
            wrapper.forward(sequence)
            ablated = list(rec.outputs)
    finally:
        handle.remove()

    on_coord_energy = np.zeros(n_layers)
    total_energy = np.zeros(n_layers)
    coord_fraction = np.zeros(n_layers)
    deltas = [None] * n_layers

    for i in range(n_layers):
        if clean[i] is None or ablated[i] is None:
            continue
        c, a = clean[i], ablated[i]
        seq_len = min(c.shape[0], a.shape[0])
        delta = c[:seq_len] - a[:seq_len]
        deltas[i] = delta
        on_coord_energy[i] = float((delta[:, sw_row] ** 2).sum())
        total_energy[i] = float((delta ** 2).sum())
        coord_fraction[i] = (on_coord_energy[i] / total_energy[i]
                              if total_energy[i] > 0 else 0.0)

    L, F = sw_layer, n_layers - 1
    coordinate_retention = (on_coord_energy[F] / on_coord_energy[L]
                             if on_coord_energy[L] > 0 else float("nan"))
    total_retention = (total_energy[F] / total_energy[L]
                        if total_energy[L] > 0 else float("nan"))
    dispersion_index = (total_retention / coordinate_retention
                         if coordinate_retention and coordinate_retention == coordinate_retention
                         and coordinate_retention != 0 else float("nan"))

    deltaF = deltas[F]
    if deltaF is not None and deltaF.numel() > 0:
        pos_norms = (deltaF ** 2).sum(dim=1)
        p_star = int(pos_norms.argmax())
        vec = deltaF[p_star]
        sq = vec ** 2
        sq_sum = float(sq.sum())
        sq4_sum = float((sq ** 2).sum())
        participation_ratio = (sq_sum ** 2 / sq4_sum) if sq4_sum > 0 else float("nan")
        vec_norm = float(vec.norm())
        top1_fraction = (float(vec.abs().max()) / vec_norm) if vec_norm > 0 else float("nan")
    else:
        p_star = -1
        participation_ratio = float("nan")
        top1_fraction = float("nan")

    return {
        "n_layers": n_layers,
        "sw_layer": sw_layer,
        "sw_row": sw_row,
        "on_coord_energy": on_coord_energy.tolist(),
        "total_energy": total_energy.tolist(),
        "coord_fraction": coord_fraction.tolist(),
        "coordinate_retention": coordinate_retention,
        "total_retention": total_retention,
        "dispersion_index": dispersion_index,
        "participation_ratio": participation_ratio,
        "top1_fraction": top1_fraction,
        "p_star": p_star,
    }
