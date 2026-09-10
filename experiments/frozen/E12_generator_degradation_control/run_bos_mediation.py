#!/usr/bin/env python3
"""Priority-3 causal mediation: does row L4/r2371's functional damage in GENERator-EUK
route disproportionately through its BOS-position (position 0) contribution?

INTERVENTION TENSOR (documented precisely, per task requirement):
  `down_proj` (pattern "model.layers.{layer}.mlp.down_proj") is an nn.Linear mapping the
  d_ffn-dimensional gated-intermediate activation to one d_model-dimensional vector that is
  ADDED INTO THE RESIDUAL STREAM (standard pre-norm gated-FFN block: h += down_proj(act)).
  Row `row` of down_proj.weight (shape [d_model, d_ffn]) is the row whose dot product with
  the full intermediate vector produces exactly ONE scalar component of that per-token
  contribution -- i.e. down_proj's OUTPUT channel `row`. Every existing intervention in this
  repo (tomography_lib.generator_gc_response, e12_lib.row_scaled) edits the WEIGHT
  (down_proj.weight[row, :] *= alpha), which necessarily applies the same scale at every
  token position, because a static weight cannot depend on token position.

  This experiment intervenes one step later in the SAME forward pass: a forward hook on the
  SAME down_proj module that reads/writes its OUTPUT tensor, output[..., row], indexed by
  token position. This is the cleanest available position-specific intervention point:
  downstream of the row's own gating/nonlinearity (already baked into the intermediate
  activation the weight multiplies), upstream of the residual add -- so the hook's edit is
  EXACTLY what reaches the residual stream at that position, nothing more, nothing less.

  Two mechanisms, composed per condition:
    - ZERO: output[:, positions, row] = 0   (removes row 2371's contribution there)
    - INJECT: output[:, positions, row] = v  (overwrites with a captured value `v`,
      normally row 2371's own INTACT contribution at that position, captured from a separate
      clean forward pass over the identical input)
  Conditions B/E/F additionally hold down_proj.weight[row,:] = 0 for the whole forward pass
  (the existing, validated full-ablation mechanism, `e12_lib.row_scaled(..., "alpha", 0.0)`)
  so that every position *not* explicitly injected is correctly ablated, including at
  positions beyond the hook's own bookkeeping (relevant for the KV-cache generation
  endpoint -- see PositionRowHook below).

CONDITIONS:
  A  intact                                            -- no weight edit, no hook
  B  full row-2371 ablation                             -- weight zeroed, no hook
  C  ablate row-2371 ONLY at BOS (position 0)            -- weight intact, hook ZEROs pos 0 only
  D  ablate row-2371 everywhere EXCEPT BOS               -- weight intact, hook ZEROs all but pos 0
  E  full ablation + restore intact contribution at BOS  -- weight zeroed, hook INJECTs captured
                                                             intact value at pos 0 only
  F  full ablation + restore at a matched non-BOS position -- weight zeroed, hook INJECTs
                                                             captured intact value at position
                                                             floor(L/2) of the prompt window
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SALVAGE = HERE.parents[1]
ROOT = SALVAGE.parent
E9_DIR = SALVAGE / "experiments" / "E9_mechanistic_tomography"
sys.path[:0] = [str(HERE), str(E9_DIR)]
import e12_lib as e12  # noqa: E402
import tomography_lib as tl  # noqa: E402
from run_gue_ablation import _resolve_module, _save_row, _restore_row  # noqa: E402

OUT_DIR = ROOT / "results" / "E_BOS_MEDIATION"
LAYER = e12.GENERATOR_LAYER  # 4
ROW = e12.GENERATOR_ROW_PRIMARY  # 2371
PATTERN = "model.layers.{i}.mlp.down_proj"
REVISION = "7dc01bccce5b65e15141170538afdc2ff09d8dde"  # resolved commit found in the local
                                                          # HF cache (BASELINE_REGRESSION.md,
                                                          # 2026-08-22); pinned explicitly here
                                                          # per the checkpoint-provenance gap
                                                          # flagged in the Priority-3 cost audit.
HIST_DAMAGE_INTACT = 6.385381    # results/experiments/E12/run_e12_full.log:23, damage[row2371](1.0)
HIST_DAMAGE_ABLATED = 8.754320   # results/experiments/E12/run_e12_full.log:27, damage[row2371](0.0)
REPRO_RTOL = 1e-3


# ── model loading ──────────────────────────────────────────────────────────────

def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "GenerTeam/GENERator-v2-eukaryote-3b-base", revision=REVISION,
        trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        "GenerTeam/GENERator-v2-eukaryote-3b-base", revision=REVISION,
        trust_remote_code=True, local_files_only=True, dtype=torch.float32)
    model.eval().cuda()
    resolved = getattr(model.config, "_commit_hash", None)
    return model, tok, resolved


# ── position hook ──────────────────────────────────────────────────────────────

class PositionRowHook:
    """One hook instance implements ZERO and/or INJECT on down_proj's output channel `row`.

    zero_fn(L, is_prefill) -> 1D bool array length L (True = zero this position)
    inject_fn(L, is_prefill) -> (idx list, value tensor) or (None, None)
    capture: if not None, a list this hook appends the RAW (pre-edit) output[..., row]
      tensor to on every call -- used to record the intact contribution for later reuse
      as an injection value, and for the smoke-test tensor-identity checks.
    KV-cache note (smoke test 6): `generate()` feeds the whole prompt in ONE forward call
    (L == prompt length > 1, "prefill"), then one new token per call thereafter (L == 1,
    "decode"). Position 0 of the ORIGINAL sequence only ever appears as local index 0 of
    the prefill call -- it is never re-computed during decode. zero_fn/inject_fn receive
    `is_prefill` so a condition can distinguish "this call's local position 0" (only
    meaningful during prefill) from "every position after the prompt" (every decode call).
    """

    def __init__(self, row: int, zero_fn=None, inject_fn=None, capture: list | None = None):
        self.row = row
        self.zero_fn = zero_fn
        self.inject_fn = inject_fn
        self.capture = capture
        self.last_zeroed_positions: list[list[int]] = []  # for smoke test 6 bookkeeping

    def __call__(self, _module, _inputs, output):
        was_tuple = isinstance(output, tuple)
        y = output[0] if was_tuple else output
        L = y.shape[1]
        is_prefill = L > 1
        if self.zero_fn is not None:
            mask = self.zero_fn(L, is_prefill)
            idxs = [i for i, m in enumerate(mask) if m]
            self.last_zeroed_positions.append(idxs)
            if idxs:
                y[:, idxs, self.row] = 0.0
        if self.inject_fn is not None:
            idxs, vals = self.inject_fn(L, is_prefill)
            if idxs:
                y[:, idxs, self.row] = vals.to(y.dtype).to(y.device)
        # Captured AFTER any zero/inject edit above, so a capture registered alongside an
        # active intervention records what actually reached the residual stream -- not the
        # pre-edit value (which a capture-only hook, e.g. _forced_capture, still gets
        # correctly since it never sets zero_fn/inject_fn in the first place).
        if self.capture is not None:
            self.capture.append(y[..., self.row].detach().clone())
        return y if was_tuple is False else (y,) + output[1:]


def register(model, fn) -> torch.utils.hooks.RemovableHandle:
    module = _resolve_module(model, PATTERN, LAYER)
    return module.register_forward_hook(fn)


# ── per-window native-NLL under one condition ───────────────────────────────────

def _tokenize(tok, seq: str):
    prep = e12._prepare_sequence(tok, seq)
    enc = tok(prep, return_tensors="pt", add_special_tokens=False).to("cuda")
    return enc["input_ids"]


def _forced_capture(model, tok, seq) -> torch.Tensor | None:
    """Intact forward pass (no weight edit, no zero/inject) capturing down_proj output
    channel `row` at every position. Returns None for degenerate (<2 token) windows."""
    ids = _tokenize(tok, seq)
    if ids.shape[1] < 2:
        return None
    cap: list[torch.Tensor] = []
    h = register(model, PositionRowHook(ROW, capture=cap))
    with torch.no_grad():
        model(input_ids=ids)
    h.remove()
    return cap[0][0]  # [L] -- batch size is always 1 here


@torch.no_grad()
def nll_for_condition(model, tok, seq: str, condition: str,
                       intact_cache: dict | None = None) -> tuple[float, int, dict]:
    """Returns (sum_nll_over_tokens, n_tokens, debug) for one window under one condition.
    `intact_cache` (keyed by seq) memoizes the captured intact row-output vector so E/F/smoke
    tests over the same window don't re-run the intact pass redundantly."""
    ids = _tokenize(tok, seq)
    if ids.shape[1] < 2:
        return 0.0, 0, {}
    L = ids.shape[1]
    debug: dict = {}

    def need_intact():
        if intact_cache is not None and seq in intact_cache:
            return intact_cache[seq]
        v = _forced_capture(model, tok, seq)
        if intact_cache is not None:
            intact_cache[seq] = v
        return v

    if condition == "A_intact":
        h = None
        loss = _run_loss(model, ids)

    elif condition == "B_full_ablation_weight":
        saved = e12.set_row_alpha(model, PATTERN, LAYER, ROW, 0.0)
        try:
            loss = _run_loss(model, ids)
        finally:
            e12.restore_row(model, PATTERN, LAYER, ROW, saved)

    elif condition == "B_full_ablation_hook":
        # Smoke-test-2 cross-check: hook-based all-positions zero, weight untouched.
        h = register(model, PositionRowHook(ROW, zero_fn=lambda L_, pf: [True] * L_))
        try:
            loss = _run_loss(model, ids)
        finally:
            h.remove()

    elif condition == "C_ablate_bos_only":
        h = register(model, PositionRowHook(
            ROW, zero_fn=lambda L_, pf: [i == 0 for i in range(L_)]))
        try:
            loss = _run_loss(model, ids)
        finally:
            h.remove()

    elif condition == "D_ablate_all_except_bos":
        h = register(model, PositionRowHook(
            ROW, zero_fn=lambda L_, pf: [i != 0 for i in range(L_)]))
        try:
            loss = _run_loss(model, ids)
        finally:
            h.remove()

    elif condition == "E_restore_at_bos":
        intact_row = need_intact()
        saved = e12.set_row_alpha(model, PATTERN, LAYER, ROW, 0.0)
        target_pos = 0
        inj_fn = lambda L_, pf, tp=target_pos: ([tp], intact_row[tp:tp + 1])
        h = register(model, PositionRowHook(ROW, inject_fn=inj_fn))
        try:
            loss = _run_loss(model, ids)
        finally:
            h.remove()
            e12.restore_row(model, PATTERN, LAYER, ROW, saved)
        debug["restore_position"] = target_pos

    elif condition == "F_restore_at_matched_nonbos":
        intact_row = need_intact()
        target_pos = L // 2
        saved = e12.set_row_alpha(model, PATTERN, LAYER, ROW, 0.0)
        inj_fn = lambda L_, pf, tp=target_pos: ([tp], intact_row[tp:tp + 1])
        h = register(model, PositionRowHook(ROW, inject_fn=inj_fn))
        try:
            loss = _run_loss(model, ids)
        finally:
            h.remove()
            e12.restore_row(model, PATTERN, LAYER, ROW, saved)
        debug["restore_position"] = target_pos

    elif condition == "SMOKE_restore_everywhere":
        intact_row = need_intact()
        saved = e12.set_row_alpha(model, PATTERN, LAYER, ROW, 0.0)
        inj_fn = lambda L_, pf, iv=intact_row: (list(range(L_)), iv[:L_])
        h = register(model, PositionRowHook(ROW, inject_fn=inj_fn))
        try:
            loss = _run_loss(model, ids)
        finally:
            h.remove()
            e12.restore_row(model, PATTERN, LAYER, ROW, saved)

    else:
        raise ValueError(condition)

    n_tok = L - 1
    return loss * n_tok, n_tok, debug


def _run_loss(model, ids) -> float:
    out = model(input_ids=ids, labels=ids)
    return float(out.loss)


def damage(model, tok, windows, condition: str, intact_cache: dict | None = None) -> float:
    tot, n = 0.0, 0
    for _c, _s, seq in windows:
        s, k, _ = nll_for_condition(model, tok, seq, condition, intact_cache)
        tot += s
        n += k
    return tot / max(n, 1)


def damage_per_window(model, tok, windows, condition: str, intact_cache=None) -> list[float]:
    """Per-window mean NLL (not pooled) -- needed for paired bootstrap over windows."""
    out = []
    for _c, _s, seq in windows:
        s, k, _ = nll_for_condition(model, tok, seq, condition, intact_cache)
        out.append(s / k if k else float("nan"))
    return out


# ── bootstrap ────────────────────────────────────────────────────────────────

def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot=5000, seed=42) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    n = len(a)
    point = float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[r] = np.mean(a[idx]) - np.mean(b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": point, "ci95": [float(lo), float(hi)], "n": int(n)}


def rescue_fraction_bootstrap(cond_num: np.ndarray, cond_ablation: np.ndarray,
                               cond_intact: np.ndarray, n_boot=5000, seed=42) -> dict:
    """rescue_fraction = (effect_ablation - effect_condition) / (effect_ablation - effect_intact)
    computed per-window-mean arrays (paired: same window order in all three), matching this
    project's convention of pairing the SAME resample draw across every array in one statistic."""
    a = np.asarray(cond_num, float)
    b = np.asarray(cond_ablation, float)
    c = np.asarray(cond_intact, float)
    valid = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    a, b, c = a[valid], b[valid], c[valid]
    n = len(a)

    def stat(x, y, z):
        denom = np.mean(y) - np.mean(z)
        if abs(denom) < 1e-8:
            return float("nan")
        return float((np.mean(y) - np.mean(x)) / denom)

    point = stat(a, b, c)
    rng = np.random.default_rng(seed)
    draws = []
    for r in range(n_boot):
        idx = rng.integers(0, n, n)
        v = stat(a[idx], b[idx], c[idx])
        if np.isfinite(v):
            draws.append(v)
    ci = list(np.percentile(draws, [2.5, 97.5])) if draws else [None, None]
    return {"point": point, "ci95": ci, "n": int(n), "n_boot_finite": len(draws)}


# ── generation / GC endpoint (secondary; KV-cache prefill-vs-decode aware) ──────

BASES = "ACGT"


def _gc_frac(seq: str) -> float:
    gc = sum(1 for c in seq if c in "GC")
    at = sum(1 for c in seq if c in "AT")
    return gc / (gc + at) if (gc + at) else float("nan")


def _forced_capture_prompt(model, tok, prompt_ids: torch.Tensor) -> torch.Tensor:
    """Same as _forced_capture but operating on already-tokenized prompt ids (for the
    generation endpoint, so the SAME prompt tokenization is reused for capture and gen)."""
    cap: list[torch.Tensor] = []
    h = register(model, PositionRowHook(ROW, capture=cap))
    with torch.no_grad():
        model(input_ids=prompt_ids)
    h.remove()
    return cap[0][0]


def _condition_hook_for_generation(condition: str, intact_row: torch.Tensor | None):
    """Returns a PositionRowHook (or None for A) implementing `condition` for use inside
    model.generate()'s KV-cache loop -- see PositionRowHook's docstring for the
    prefill-vs-decode distinction this depends on."""
    if condition == "A_intact":
        return None
    if condition == "B_full_ablation_weight":
        return None  # handled by weight edit alone
    if condition == "C_ablate_bos_only":
        return PositionRowHook(ROW, zero_fn=lambda L, pf: ([True] + [False] * (L - 1)) if pf
                                else [False] * L)
    if condition == "D_ablate_all_except_bos":
        return PositionRowHook(ROW, zero_fn=lambda L, pf: ([False] + [True] * (L - 1)) if pf
                                else [True] * L)
    if condition == "E_restore_at_bos":
        return PositionRowHook(ROW, inject_fn=lambda L, pf, iv=intact_row: (
            ([0], iv[0:1]) if pf else ([], None)))
    if condition == "F_restore_at_matched_nonbos":
        return PositionRowHook(ROW, inject_fn=lambda L, pf, iv=intact_row: (
            ([L // 2], iv[L // 2:L // 2 + 1]) if pf else ([], None)))
    raise ValueError(condition)


NEEDS_WEIGHT_ABLATION = {"B_full_ablation_weight", "E_restore_at_bos", "F_restore_at_matched_nonbos"}


@torch.no_grad()
def generate_for_condition(model, tok, prompts: list[str], condition: str,
                            max_new: int = 64, seed: int = e12.BASE_SEED) -> list[dict]:
    records = []
    saved = None
    if condition in NEEDS_WEIGHT_ABLATION:
        saved = e12.set_row_alpha(model, PATTERN, LAYER, ROW, 0.0)
    try:
        for i, p in enumerate(prompts):
            enc = tok(p, return_tensors="pt")
            ids = enc["input_ids"].cuda()
            intact_row = None
            if condition in ("E_restore_at_bos", "F_restore_at_matched_nonbos"):
                # capture BEFORE the weight is ablated -- but weight is already ablated at
                # this point in the loop (set once for the whole prompt list above), so
                # capture from a temporary restore instead of re-deriving saved/ablated state.
                e12.restore_row(model, PATTERN, LAYER, ROW, saved)
                intact_row = _forced_capture_prompt(model, tok, ids)
                m = _resolve_module(model, PATTERN, LAYER)
                with torch.no_grad():
                    m.weight.data[ROW, :] = saved * 0.0
            hook = _condition_hook_for_generation(condition, intact_row)
            handle = register(model, hook) if hook is not None else None
            torch.manual_seed(seed + i)
            try:
                out = model.generate(
                    input_ids=ids, max_new_tokens=max_new,
                    pad_token_id=getattr(tok, "pad_token_id", None) or 0,
                    do_sample=True, top_k=50, temperature=1.0,
                )
            finally:
                if handle is not None:
                    handle.remove()
            new_ids = out[0][ids.shape[1]:]
            raw = tok.decode(new_ids, skip_special_tokens=True)
            filt = "".join(c for c in raw.upper() if c in BASES)
            records.append({
                "prompt_idx": i, "gc": _gc_frac(filt) if len(filt) >= 10 else float("nan"),
                "n_filtered": len(filt),
            })
    finally:
        if saved is not None:
            e12.restore_row(model, PATTERN, LAYER, ROW, saved)
    return records


if __name__ == "__main__":
    print("This module is imported by run_bos_mediation_main.py -- see that file for the "
          "provenance check / smoke tests / full run entry point.")
