"""
Shared measurement engine for E12 -- degradation-matched control for the GENERator GC
result. Reuses E9's tomography_lib.py (row save/restore, generation loop, windowing,
gc_frac) unmodified per the locked prereg, and adds only the instrumentation/metrics that
are genuinely new to this experiment:

  - build_corpora: disjoint prompt-pool / damage-pool windowing from a partitioned BED file.
  - generator_response_unified: same generation loop as tomography_lib's
    generator_gc_response (do_sample, top_k=50, temperature=1.0, per-prompt seed offset,
    same silent ACGT filter) but also captures per-token predictive entropy (via a
    passthrough LogitsProcessor inserted ahead of GENERator's own bp-collapse step -- see
    _ScoreRecorder), the raw pre-filter decoded string, and the raw generated token ids
    (for self-NLL) -- none of that existed before this experiment.
  - damage: teacher-forced mean per-token causal-LM NLL over the damage pool, under the same
    row-scaling mechanism tomography_lib already uses.
  - self_nll: teacher-forced NLL of the (row-scaled) model on its own generated continuation.
  - quality metrics: distinct-n, homopolymer runs, top-kmer share, non-ACGT filtering rate.
  - bootstrap_gc_diff: new paired, per-prompt bootstrap (nothing existing resamples GC per
    prompt; run_fit_observers.bootstrap_mae_diff resamples batches, not prompts).
  - random-direction control: replace row 2371's weight with c * unit_random_direction *
    ||original_row||, per the prereg's step B3.3.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import LogitsProcessor

E9_DIR = Path(__file__).resolve().parents[1] / "E9_mechanistic_tomography"
sys.path.insert(0, str(E9_DIR))
import tomography_lib as tl  # noqa: E402
from run_ensemble_encoding import gc_frac, read_windows  # noqa: E402
from run_gue_ablation import _resolve_module, _save_row, _restore_row  # noqa: E402

ROOT = tl.ROOT
HG38_FASTA = tl.HG38_FASTA
HG38_BED = tl.HG38_BED
BASES = "ACGT"

GENERATOR_ROW_PRIMARY = 2371
GENERATOR_ROW_SECONDARY = 1522
GENERATOR_LAYER = 4

BASE_SEED = 42


# ── corpus: two disjoint hg38 window pools ────────────────────────────────────

def _read_bed_regions(bed: str) -> list[tuple[str, int, int]]:
    regs = []
    with open(bed) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                regs.append((p[0], int(p[1]), int(p[2])))
    return regs


def build_corpora(seed: int = BASE_SEED, n_prompt: int = 96, prompt_win_bp: int = 170,
                   n_damage: int = 100, damage_win_bp: int = 512,
                   max_repartition_tries: int = 5):
    """Partition the BED region list into two disjoint halves (fixed-seed shuffle then
    split) BEFORE sampling either pool, sample each pool only from its own half, then
    assert zero chromosome+coordinate overlap between the sampled windows. Re-partitions
    with a new fixed seed (seed+1, seed+2, ...) if the assertion fails, recording that this
    happened. Returns (prompt_windows, damage_windows, meta)."""
    regions = _read_bed_regions(HG38_BED)
    meta = {"base_seed": seed, "repartition_tries": 0, "n_regions_total": len(regions)}

    for attempt in range(max_repartition_tries):
        part_seed = seed + attempt
        rng_part = random.Random(part_seed)
        regs = list(regions)
        rng_part.shuffle(regs)
        half = len(regs) // 2
        half_a, half_b = regs[:half], regs[half:]

        def _sample(regs_half, n, win_bp, rng):
            from pyfaidx import Fasta
            fa = Fasta(HG38_FASTA, as_raw=True)
            out = []
            pool = list(regs_half)
            rng.shuffle(pool)
            for c, s, e in pool:
                if len(out) >= n:
                    break
                if e - s < win_bp:
                    continue
                st = rng.randrange(s, e - win_bp)
                sq = str(fa[c][st:st + win_bp]).upper()
                if sq.count("N") / max(len(sq), 1) < 0.01:
                    out.append((c, st, sq))
            return out

        rng_prompt = random.Random(seed + 1000 + attempt)
        rng_damage = random.Random(seed + 2000 + attempt)
        prompt_wins = _sample(half_a, n_prompt, prompt_win_bp, rng_prompt)
        damage_wins = _sample(half_b, n_damage, damage_win_bp, rng_damage)

        def _span(c, st, sq):
            return (c, st, st + len(sq))

        prompt_spans = {_span(*w) for w in prompt_wins}

        def overlaps(a, b):
            if a[0] != b[0]:
                return False
            return a[1] < b[2] and b[1] < a[2]

        conflict = False
        for w in damage_wins:
            dspan = _span(*w)
            for pspan in prompt_spans:
                if overlaps(dspan, pspan):
                    conflict = True
                    break
            if conflict:
                break

        meta["repartition_tries"] = attempt
        if not conflict and len(prompt_wins) >= n_prompt and len(damage_wins) >= n_damage:
            meta["partition_seed_used"] = part_seed
            meta["n_prompt"] = len(prompt_wins)
            meta["n_damage"] = len(damage_wins)
            return prompt_wins, damage_wins, meta
        meta.setdefault("repartition_events", []).append(
            {"attempt": attempt, "conflict": conflict,
             "n_prompt_found": len(prompt_wins), "n_damage_found": len(damage_wins)})

    raise RuntimeError(f"could not build disjoint corpora after {max_repartition_tries} "
                        f"tries: {meta}")


# ── GENERator sequence prep (mirrors GeneratorWrapper._prepare_sequence) ─────

def _prepare_sequence(tok, sequence: str) -> str:
    remainder = len(sequence) % 6
    if remainder:
        sequence = sequence[remainder:]
    bos = tok.bos_token or ""
    return bos + sequence


# ── row / direction scaling (thin wrappers around tomography_lib's primitives) ─

def set_row_alpha(model, pattern, layer, row, alpha):
    saved = _save_row(model, pattern, layer, row)
    m = _resolve_module(model, pattern, layer)
    with torch.no_grad():
        m.weight.data[row, :] = saved * alpha
    return saved


def restore_row(model, pattern, layer, row, saved):
    _restore_row(model, pattern, layer, row, saved)


def set_row_random_direction(model, pattern, layer, row, unit_dir: torch.Tensor,
                              orig_row: torch.Tensor, c: float):
    """Replace `row`'s weight with c * unit_dir * ||orig_row||_2 (prereg B3.3)."""
    m = _resolve_module(model, pattern, layer)
    norm = orig_row.norm().item()
    new_row = (unit_dir * (c * norm)).to(dtype=m.weight.dtype, device=m.weight.device)
    with torch.no_grad():
        m.weight.data[row, :] = new_row


def make_unit_random_direction(intermediate_size: int, row: int, dtype, device,
                                base_seed: int = 20260823) -> torch.Tensor:
    rng = np.random.default_rng(base_seed + row)
    v = rng.normal(size=intermediate_size).astype(np.float64)
    v = v / np.linalg.norm(v)
    return torch.tensor(v, dtype=dtype, device=device)


# ── unified row-scaling context: "alpha" (raw multiplicative scale) or
#    "direction" (replace row with c * unit_random_direction * ||orig_row||) ────

@contextmanager
def row_scaled(model, pattern, layer, row, kind: str, value: float,
                unit_dir: torch.Tensor | None = None):
    """kind='alpha': scale the row by raw multiplicative `value` (tomography_lib
    convention). kind='direction': replace the row with value * unit_dir * ||orig_row||
    (prereg B3.3; unit_dir must be given). Saves/restores the original row regardless."""
    saved = _save_row(model, pattern, layer, row)
    m = _resolve_module(model, pattern, layer)
    try:
        with torch.no_grad():
            if kind == "alpha":
                m.weight.data[row, :] = saved * value
            elif kind == "direction":
                assert unit_dir is not None
                norm = saved.norm().item()
                new_row = (unit_dir * (value * norm)).to(dtype=m.weight.dtype,
                                                           device=m.weight.device)
                m.weight.data[row, :] = new_row
            else:
                raise ValueError(f"unknown kind {kind!r}")
        yield saved
    finally:
        _restore_row(model, pattern, layer, row, saved)


@torch.no_grad()
def self_nll_batch(model, tok, pattern, layer, row, kind, value, records,
                    unit_dir: torch.Tensor | None = None) -> list[float]:
    """Teacher-forced mean per-token NLL of the (row-scaled) model on its own generated
    continuation, for every record in `records`, scaling the row ONCE for the whole batch
    (cheaper than re-scaling per record) rather than per-call save/restore."""
    dev = next(model.parameters()).device
    out = []
    with row_scaled(model, pattern, layer, row, kind, value, unit_dir):
        for rec in records:
            prompt_ids = rec["prompt_ids"]
            new_ids = rec["new_ids"]
            if len(new_ids) < 2:
                out.append(float("nan"))
                continue
            full = torch.tensor([prompt_ids + new_ids], device=dev)
            labels = full.clone()
            labels[0, :len(prompt_ids)] = -100
            o = model(input_ids=full, labels=labels)
            out.append(float(o.loss))
    return out


class _ScoreRecorder(LogitsProcessor):
    """Passthrough LogitsProcessor that records the logits it sees and does not modify
    them. GENERator's own generate() (modeling_generator.py) appends a `_BPLogitsProcessor`
    LAST in the processor chain that decomposes each k-mer token into an independent
    per-base-pair sample and then force-collapses the returned `scores` to one-hot over
    the single selected token -- so `out.scores` from generate() is NOT usable for entropy
    (it is always a one-hot vector by construction, regardless of how uncertain the model
    actually was). HF appends user-supplied `logits_processor` entries after the built-in
    temperature/top_k/top_p warpers but the GENERator wrapper always appends its own
    `_BPLogitsProcessor` after whatever the caller supplies, so a recorder passed in via
    `logits_processor=[...]` sees the fully temperature/top_k/top_p-processed distribution
    over the whole k-mer vocabulary immediately BEFORE the bp-collapse -- exactly "the
    model's softmax over the full vocabulary at each sampling step," from the same forward
    pass, no extra model call. Confirmed empirically (n_finite == vocab-adjacent counts
    instead of 1, and entropy ~8 nats rather than identically 0) before adopting this."""

    def __init__(self):
        self.captured: list[torch.Tensor] = []

    def __call__(self, input_ids, scores):
        self.captured.append(scores.detach())
        return scores


@torch.no_grad()
def generator_response_unified(model, tok, pattern, layer, row, kind, value, prompts,
                                max_new, seed=BASE_SEED, do_sample=True,
                                unit_dir: torch.Tensor | None = None):
    """Generation + full B1 instrumentation for one condition, row scaled via row_scaled
    (alpha or direction) so control-row and random-direction conditions share one code
    path. Predictive entropy is computed from the pre-bp-collapse distribution captured by
    _ScoreRecorder (see its docstring) -- not from generate()'s own returned `scores`."""
    dev = next(model.parameters()).device
    records = []
    with row_scaled(model, pattern, layer, row, kind, value, unit_dir):
        for i, p in enumerate(prompts):
            enc = tok(p, return_tensors="pt")
            ids = enc["input_ids"].to(dev)
            torch.manual_seed(seed + i)
            recorder = _ScoreRecorder()
            gen_kwargs = dict(
                input_ids=ids, max_new_tokens=max_new,
                pad_token_id=getattr(tok, "pad_token_id", None) or 0,
                output_scores=True, return_dict_in_generate=True,
                logits_processor=[recorder],
            )
            if do_sample:
                gen_kwargs.update(do_sample=True, top_k=50, temperature=1.0)
            else:
                gen_kwargs.update(do_sample=False)
            out = model.generate(**gen_kwargs)
            seq = out.sequences[0]
            new_ids = seq[ids.shape[1]:]
            raw = tok.decode(new_ids, skip_special_tokens=True)
            filt = "".join(c for c in raw.upper() if c in BASES)

            ent = float("nan")
            if recorder.captured:
                ents = []
                for step_logits in recorder.captured:
                    logp = torch.log_softmax(step_logits[0].float(), dim=-1)
                    p_ = logp.exp()
                    term = p_ * logp
                    # entries filtered out by top_k/top_p carry logp=-inf, p_=0; their
                    # 0*(-inf) product is nan under IEEE arithmetic but the true
                    # contribution to Shannon entropy of a zero-probability outcome is 0.
                    term = torch.where(p_ > 0, term, torch.zeros_like(term))
                    h = -term.sum().item()
                    ents.append(h)
                ent = float(np.mean(ents)) if ents else float("nan")

            records.append({
                "prompt_idx": i,
                "raw": raw,
                "filtered": filt,
                "gc": gc_frac(filt) if len(filt) >= 10 else float("nan"),
                "n_filtered": len(filt),
                "n_raw": len(raw),
                "entropy": ent,
                "new_ids": new_ids.detach().cpu().tolist(),
                "prompt_ids": ids[0].detach().cpu().tolist(),
            })
    return records


@torch.no_grad()
def damage_unified(model, tok, pattern, layer, row, kind, value, windows,
                    unit_dir: torch.Tensor | None = None) -> float:
    with row_scaled(model, pattern, layer, row, kind, value, unit_dir):
        return damage(model, tok, windows)


# ── damage metric: teacher-forced NLL over the damage pool ───────────────────

@torch.no_grad()
def window_nll(model, tok, seq: str) -> tuple[float, int]:
    """(sum_nll_over_tokens, n_tokens) for one window, teacher-forced, standard shifted
    causal-LM loss (transformers' internal label-shift), float32."""
    dev = next(model.parameters()).device
    prep = _prepare_sequence(tok, seq)
    enc = tok(prep, return_tensors="pt", add_special_tokens=False).to(dev)
    ids = enc["input_ids"]
    if ids.shape[1] < 2:
        return 0.0, 0
    out = model(input_ids=ids, labels=ids)
    n_tok = ids.shape[1] - 1
    return float(out.loss) * n_tok, n_tok


@torch.no_grad()
def damage(model, tok, windows: list[tuple[str, int, str]]) -> float:
    tot, n = 0.0, 0
    for _c, _s, seq in windows:
        s, k = window_nll(model, tok, seq)
        tot += s
        n += k
    return tot / max(n, 1)


def damage_under_row_alpha(model, tok, pattern, layer, row, alpha, windows) -> float:
    saved = _save_row(model, pattern, layer, row)
    m = _resolve_module(model, pattern, layer)
    with torch.no_grad():
        m.weight.data[row, :] = saved * alpha
    try:
        return damage(model, tok, windows)
    finally:
        _restore_row(model, pattern, layer, row, saved)


def damage_under_row_direction(model, tok, pattern, layer, row, unit_dir, orig_row, c,
                                windows) -> float:
    saved = _save_row(model, pattern, layer, row)
    try:
        set_row_random_direction(model, pattern, layer, row, unit_dir, orig_row, c)
        return damage(model, tok, windows)
    finally:
        _restore_row(model, pattern, layer, row, saved)


def find_matching_scale(damage_fn, target: float, grid: Sequence[float],
                         max_refine: int = 3):
    """Evaluate damage_fn(scale) over `grid`, find the scale whose damage is closest to
    `target`; if the target lies strictly between two adjacent (by-scale-sorted) grid
    points' damage values, bisect within that bracketing interval up to `max_refine` extra
    evaluations. Returns a dict with all evaluated points, the best match, and whether the
    target was reachable on the grid at all."""
    evals = [(s, damage_fn(s)) for s in grid]
    evals.sort(key=lambda t: t[0])
    scales = [s for s, _ in evals]
    ds = [d for _, d in evals]

    reachable = min(ds) <= target <= max(ds)
    bracket = None
    for j in range(len(evals) - 1):
        lo, hi = ds[j], ds[j + 1]
        if (lo - target) * (hi - target) <= 0 and lo != hi:
            bracket = (scales[j], scales[j + 1])
            break

    refine_evals = []
    if bracket is not None:
        lo_s, hi_s = bracket
        for _ in range(max_refine):
            mid_s = (lo_s + hi_s) / 2.0
            mid_d = damage_fn(mid_s)
            refine_evals.append((mid_s, mid_d))
            evals.append((mid_s, mid_d))
            lo_d = [d for s, d in evals if s == lo_s][0]
            if (lo_d - target) * (mid_d - target) <= 0:
                hi_s = mid_s
            else:
                lo_s = mid_s

    best_scale, best_damage = min(evals, key=lambda t: abs(t[1] - target))
    if not reachable:
        best_scale, best_damage = max(evals, key=lambda t: t[1])

    return {
        "grid_evals": [{"scale": s, "damage": d} for s, d in sorted(evals)],
        "refine_evals": [{"scale": s, "damage": d} for s, d in refine_evals],
        "target": target,
        "reachable": bool(reachable),
        "matched_scale": best_scale,
        "matched_damage": best_damage,
        "fallback_used": not reachable,
    }


# ── self-NLL: model's own NLL on its own generated continuation ─────────────

@torch.no_grad()
def self_nll_for_record(model, tok, pattern, layer, row, alpha, rec) -> float:
    dev = next(model.parameters()).device
    saved = _save_row(model, pattern, layer, row)
    m = _resolve_module(model, pattern, layer)
    with torch.no_grad():
        m.weight.data[row, :] = saved * alpha
    try:
        prompt_ids = rec["prompt_ids"]
        new_ids = rec["new_ids"]
        if len(new_ids) < 2:
            return float("nan")
        full = torch.tensor([prompt_ids + new_ids], device=dev)
        labels = full.clone()
        labels[0, :len(prompt_ids)] = -100
        out = model(input_ids=full, labels=labels)
        return float(out.loss)
    finally:
        _restore_row(model, pattern, layer, row, saved)


# ── quality metrics (B1, all new) ─────────────────────────────────────────────

def distinct_n(seq: str, n: int) -> float:
    if len(seq) < n:
        return float("nan")
    grams = [seq[i:i + n] for i in range(len(seq) - n + 1)]
    if not grams:
        return float("nan")
    return len(set(grams)) / len(grams)


def homopolymer_stats(seq: str) -> tuple[int, float]:
    if not seq:
        return 0, float("nan")
    runs = []
    cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return max(runs), float(np.mean(runs))


def longest_homopolymer_span(seq: str) -> tuple[int, int]:
    if not seq:
        return 0, 0
    best_len, best_start = 1, 0
    cur_len, cur_start = 1, 0
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
            cur_start, cur_len = i, 1
    if cur_len > best_len:
        best_len, best_start = cur_len, cur_start
    return best_start, best_start + best_len


def top_kmer_share(seq: str, k: int) -> float:
    if len(seq) < k:
        return float("nan")
    grams = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    if not grams:
        return float("nan")
    c = Counter(grams)
    return c.most_common(1)[0][1] / len(grams)


def kmer_counts(seq: str, k: int) -> Counter:
    if len(seq) < k:
        return Counter()
    return Counter(seq[i:i + k] for i in range(len(seq) - k + 1))


def non_acgt_filter_rate(raw: str, filtered: str) -> float:
    if len(raw) == 0:
        return float("nan")
    return 1.0 - (len(filtered) / len(raw))


def quality_metrics_for_record(rec: dict) -> dict:
    filt = rec["filtered"]
    longest, mean_run = homopolymer_stats(filt) if len(filt) >= 1 else (0, float("nan"))
    return {
        "distinct2": distinct_n(filt, 2),
        "distinct3": distinct_n(filt, 3),
        "distinct4": distinct_n(filt, 4),
        "longest_homopolymer": longest,
        "mean_homopolymer": mean_run,
        "top3mer_share": top_kmer_share(filt, 3),
        "top6mer_share": top_kmer_share(filt, 6),
        "non_acgt_filter_rate": non_acgt_filter_rate(rec["raw"], rec["filtered"]),
        "entropy": rec["entropy"],
    }


# ── bootstrap (new: per-prompt paired resampling) ─────────────────────────────

def bootstrap_gc_diff(gc_a: np.ndarray, gc_b: np.ndarray, n_boot: int = 5000,
                       seed: int = BASE_SEED) -> dict:
    """Paired bootstrap over prompts: the same resampled prompt-index draw is applied to
    both arrays on every resample. gc_a/gc_b must be the same length and indexed by the
    same prompt order; prompt indices that are NaN in either array are excluded from the
    resampling universe for this comparison (pairwise complete)."""
    a = np.asarray(gc_a, dtype=float)
    b = np.asarray(gc_b, dtype=float)
    assert len(a) == len(b), "bootstrap_gc_diff requires aligned per-prompt arrays"
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    n = len(a)
    point = float(np.mean(a) - np.mean(b)) if n else float("nan")
    if n == 0:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[r] = np.mean(a[idx]) - np.mean(b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n": int(n)}
