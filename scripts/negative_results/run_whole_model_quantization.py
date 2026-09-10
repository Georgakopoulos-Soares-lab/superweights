"""
scripts/negative_results/run_whole_model_quantization.py
---------------------------------------------------
Full-model grid-search quantization ablation for GENERator eukaryote
and prokaryote (LlamaForCausalLM backbone, 30 layers).

Extends the down-proj-only script to ALL 7 linear projection types:
  MLP:      down_proj | gate_proj | up_proj
  Attention: q_proj  | k_proj   | v_proj  | o_proj

Architecture note (LlamaForCausalLM)
-------------------------------------
  gate_proj / up_proj : Linear(hidden_size  → intermediate_size)
                        weight shape (intermediate_size, hidden_size)
                        row index = intermediate feature dimension
  down_proj           : Linear(intermediate_size → hidden_size)
                        weight shape (hidden_size, intermediate_size)
                        row index = output hidden dimension  ← SW lives here
  q_proj / o_proj     : (num_heads*head_dim, hidden_size) / (hidden_size, ...)
  k_proj / v_proj     : (num_kv_heads*head_dim, hidden_size)

Super-weight (SW) exclusion
-----------------------------
  down_proj  — exclude (sw_layer, sw_row)   : the actual super-row
  gate_proj  — exclude (sw_layer, sw_col)   : intermediate feature that
  up_proj      feeds most strongly into the SW row (down_proj.weight[sw_row, sw_col]
               is the dominant weight)
  q/k/v/o    — no exclusion (no known SW in attention layers)

Scope groups
------------
  down_proj_only  — baseline, matches run_quantization_ablation.py
  mlp             — all 3 MLP projections (down + gate + up)
  attn            — all 4 attention projections (q + k + v + o)
  full            — all 7 projections

Row-selection criteria  (SW rows always exempt)
------------------------------------------------
  l1_low    — ascending global L1 norm across all candidate rows
              (smallest = least important = quantise first)
  proximity — ascending normalised (layer/num_layers, row/module_nrows)
              L2 distance to nearest SW coordinate; rows closest to the
              SW neighbourhood are quantised first (Yu et al. hypothesis)
  random    — uniform random, averaged over --n_rand_seeds seeds

Grid dimensions
---------------
  --fracs       percent of TOTAL eligible rows in the chosen scope
                default: 1 5 10 20 30 50 75 100
  --bits        8 (INT8, default) or 4 (INT4)
  --scopes      space-separated scope names  (default: full)
  --criteria    space-separated criterion names (default: l1_low proximity random)
  --n_rand_seeds seeds for random criterion (default: 5)

Special single-shot conditions (always run per scope)
  yu_all        quantise every eligible row → upper-bound PPL cost
  sw_fragility  quantise only the SW rows in MLP modules → sensitivity check

Usage
-----
  # Eukaryote GENERator
  python scripts/negative_results/run_whole_model_quantization.py \\
      --model generator \\
      --out   results/whole_model_quant_generator.json

  # Prokaryote GENERator
  python scripts/negative_results/run_whole_model_quantization.py \\
      --model generator_prokaryote \\
      --out   results/whole_model_quant_generator_prokaryote.json

  # Only MLP scope with INT4, coarser grid
  python scripts/negative_results/run_whole_model_quantization.py \\
      --model generator \\
      --scopes mlp \\
      --bits 4 \\
      --fracs 5 10 25 50 100 \\
      --out   results/whole_model_quant_generator_mlp_int4.json
"""

import argparse
import json
import random as _random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Module patterns (LlamaForCausalLM / GENERator) ───────────────────────────
LLAMA_PATTERNS = {
    "down_proj": "model.layers.{i}.mlp.down_proj",
    "gate_proj": "model.layers.{i}.mlp.gate_proj",
    "up_proj":   "model.layers.{i}.mlp.up_proj",
    "q_proj":    "model.layers.{i}.self_attn.q_proj",
    "k_proj":    "model.layers.{i}.self_attn.k_proj",
    "v_proj":    "model.layers.{i}.self_attn.v_proj",
    "o_proj":    "model.layers.{i}.self_attn.o_proj",
}

# Module groups for scoping the quantization pool
SCOPE_MODULES = {
    "down_proj_only": ["down_proj"],
    "mlp":  ["down_proj", "gate_proj", "up_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "full": ["down_proj", "gate_proj", "up_proj",
             "q_proj", "k_proj", "v_proj", "o_proj"],
}

# MLP projections that share the intermediate-feature index with the SW
_MLP_PROJ_TYPES = {"down_proj", "gate_proj", "up_proj"}

# Probe sequences for perplexity evaluation.
# Generated with a fixed seed so they are reproducible but diverse.
# 6 sequences × 1200 nt = 7200 nt / 6-mer tokenizer = 1200 tokens total,
# well above the 4 kbp / ~667 token threshold requested.
# GC biases cover low-GC eukaryote (40%), neutral (50%), and high-GC
# prokaryote (60–65%) ranges so both model variants are challenged equally.
def _make_probe_seqs(n_seqs: int = 6, seq_len_nt: int = 1200, seed: int = 42) -> list:
    import random as _r
    rng    = _r.Random(seed)
    bases  = "ACGT"
    gc_biases = [0.40, 0.50, 0.60, 0.65, 0.45, 0.55]
    seqs   = []
    for i in range(n_seqs):
        gc = gc_biases[i % len(gc_biases)]
        at = 1.0 - gc
        w  = [at / 2, gc / 2, gc / 2, at / 2]   # A, C, G, T
        seq = "".join(rng.choices(bases, weights=w, k=seq_len_nt))
        seqs.append(seq)
    return seqs

PROBE_SEQS = _make_probe_seqs()   # 6 × 1200 nt = 7200 nt ≈ 1200 tokens


# ─────────────────────────────────────────────────────────────────────────────
# Module resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(model, pattern: str, layer_idx: int):
    """Navigate the module hierarchy to the nn.Module at pattern.format(i=layer_idx)."""
    path = pattern.replace("{i}", str(layer_idx))
    obj = model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Super-weight exclusion
# ─────────────────────────────────────────────────────────────────────────────

def _build_sw_exclusion(sw_list: list, scope_modules: list) -> dict:
    """Return per-module-type set of (layer, row) coordinates to exclude.

    down_proj : exclude SW row index  (sw["row"]  — output-hidden dimension)
    gate_proj : exclude SW col index  (sw["col"]  — the intermediate feature
    up_proj     that feeds most strongly into the SW row)
    attn mods : no exclusion (no known SW in attention)
    """
    down_sw  = {(sw["layer"], sw["row"]) for sw in sw_list}
    # Use sw["col"] when available; fall back gracefully if absent
    gate_sw  = {(sw["layer"], sw["col"])
                for sw in sw_list if sw.get("col") is not None}

    exclusion = {}
    for mod_key in scope_modules:
        if mod_key == "down_proj":
            exclusion[mod_key] = down_sw
        elif mod_key in ("gate_proj", "up_proj"):
            exclusion[mod_key] = gate_sw
        else:
            exclusion[mod_key] = set()   # q/k/v/o: no known SW
    return exclusion


# ─────────────────────────────────────────────────────────────────────────────
# Candidate pool
# ─────────────────────────────────────────────────────────────────────────────

def _build_candidates(model, scope_modules: list, num_layers: int,
                      sw_exclusion: dict) -> list:
    """Return flat list of (mod_key, layer, row) for every eligible weight row.

    A row is eligible if it is NOT in the SW exclusion set for its module type.
    """
    candidates = []
    for mod_key in scope_modules:
        pattern = LLAMA_PATTERNS[mod_key]
        ex_set  = sw_exclusion.get(mod_key, set())
        for li in range(num_layers):
            m     = _resolve(model, pattern, li)
            nrows = m.weight.data.shape[0]
            for ri in range(nrows):
                if (li, ri) not in ex_set:
                    candidates.append((mod_key, li, ri))
    return candidates


def _count_sw_excluded(sw_exclusion: dict) -> int:
    """Total number of (module, layer, row) triples excluded across all modules."""
    return sum(len(v) for v in sw_exclusion.values())


# ─────────────────────────────────────────────────────────────────────────────
# Ranking criteria
# ─────────────────────────────────────────────────────────────────────────────

def _rank_l1_low(model, candidates: list) -> list:
    """Sort candidates globally by ascending L1 norm (smallest = quantise first).

    Batches all rows within the same (mod_key, layer) into one GPU call to
    avoid per-row kernel overhead.  L1 scores are brought to CPU.
    """
    from collections import defaultdict
    print("    Computing L1 norms ...", flush=True, end=" ")

    # Group candidate indices by (mod_key, layer)
    groups: dict = defaultdict(list)
    for idx, (mod_key, li, ri) in enumerate(candidates):
        groups[(mod_key, li)].append((idx, ri))

    scores = [0.0] * len(candidates)
    for (mod_key, li), items in groups.items():
        m   = _resolve(model, LLAMA_PATTERNS[mod_key], li)
        row_indices = [ri for _, ri in items]
        idx_t = torch.tensor(row_indices, dtype=torch.long, device=m.weight.device)
        l1s   = m.weight.data[idx_t].float().abs().sum(dim=1).cpu().tolist()
        for (orig_idx, _), l1 in zip(items, l1s):
            scores[orig_idx] = l1

    order  = sorted(range(len(candidates)), key=lambda i: scores[i])
    result = [candidates[i] for i in order]
    print(f"done. Range [{scores[order[0]]:.2f}, {scores[order[-1]]:.2f}]")
    return result


def _rank_proximity(model, candidates: list, sw_list: list,
                    num_layers: int) -> list:
    """Sort candidates by ascending normalised Euclidean distance to nearest SW.

    Proximity is computed in the unit square [0,1]^2 using:
        (layer / (num_layers - 1),  row / (module_nrows - 1))

    The SW reference point is placed at:
        (sw_layer / (num_layers - 1),  sw_row / down_proj_nrows - 1)

    All modules are mapped to the same normalised grid, with rows divided by
    their own module's nrows. This makes close-to-SW rows in ANY module type
    the first to be quantised, consistent with the Yu et al. hypothesis that
    the SW can compensate for precision loss in its neighbourhood.

    ascending distance → closest-to-SW rows quantised first (near_sw order).
    """
    print("    Computing proximity scores ...", flush=True, end=" ")

    # Cache nrows to avoid redundant attribute access
    nrows_cache: dict = {}
    def _nrows(mk: str, li: int) -> int:
        key = (mk, li)
        if key not in nrows_cache:
            nrows_cache[key] = _resolve(model, LLAMA_PATTERNS[mk], li).weight.data.shape[0]
        return nrows_cache[key]

    # SW reference points in normalised space
    # Row is normalised using down_proj nrows (the space in which SW was detected)
    dp_nrows_l0 = _nrows("down_proj", 0)
    sw_pts = np.array([
        [sw["layer"] / max(num_layers - 1, 1),
         sw["row"]   / max(dp_nrows_l0 - 1, 1)]
        for sw in sw_list
    ], dtype=np.float32)                              # (n_sw, 2)

    scored = []
    for mod_key, li, ri in candidates:
        mod_nrows = _nrows(mod_key, li)
        pt = np.array([
            li / max(num_layers - 1, 1),
            ri / max(mod_nrows - 1, 1),
        ], dtype=np.float32)
        dist = float(np.min(np.linalg.norm(sw_pts - pt, axis=1)))
        scored.append((dist, mod_key, li, ri))

    scored.sort(key=lambda x: x[0])   # ascending: closest first
    print(f"done. Dist range [{scored[0][0]:.4f}, {scored[-1][0]:.4f}]")
    return [(mk, li, ri) for _, mk, li, ri in scored]


# ─────────────────────────────────────────────────────────────────────────────
# Quantisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quantize_rows_cpu(rows: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Vectorised per-row RTN quantisation on a CPU float32 tensor.

    rows : shape (N, D), CPU, float32
    Returns a new CPU float32 tensor of the same shape.
    """
    maxval = 2 ** (bits - 1) - 1
    amax   = rows.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # (N,1)
    scale  = amax / maxval
    q      = torch.round(rows / scale).clamp_(-maxval, maxval)
    return q * scale


def _apply_quantization(model, rows_to_quantize: list, bits: int = 8) -> dict:
    """Quantise selected (mod_key, layer, row) triples in-place.

    Batches all selected rows within the same (mod_key, layer) into a single
    GPU→CPU transfer, performs RTN quantisation on CPU, then writes back.
    Saves are stored as CPU clones of FULL module weight matrices (one entry
    per touched module) so restoration is a simple weight matrix copy and
    the saves dict never accumulates per-row GPU tensors.

    Returns
    -------
    saves : dict mapping (mod_key, layer_idx) → CPU float32 weight clone
    """
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for mod_key, li, ri in rows_to_quantize:
        groups[(mod_key, li)].append(ri)

    saves: dict = {}
    with torch.no_grad():
        for (mod_key, li), row_indices in groups.items():
            m     = _resolve(model, LLAMA_PATTERNS[mod_key], li)
            dtype = m.weight.dtype
            dev   = m.weight.device

            # Save the full weight matrix on CPU before touching it
            if (mod_key, li) not in saves:
                saves[(mod_key, li)] = m.weight.data.cpu()

            # Pull selected rows to CPU float32, quantise, write back
            idx_t = torch.tensor(row_indices, dtype=torch.long, device=dev)
            rows_cpu  = m.weight.data[idx_t].float().cpu()   # (n_sel, dim)
            rows_q    = _quantize_rows_cpu(rows_cpu, bits=bits)
            m.weight.data[idx_t] = rows_q.to(dtype).to(dev)

    return saves


def _restore_quantization(model, saves: dict):
    """Restore module weights saved by _apply_quantization.

    saves is a dict (mod_key, layer_idx) → CPU weight tensor.
    """
    with torch.no_grad():
        for (mod_key, li), weight_cpu in saves.items():
            m = _resolve(model, LLAMA_PATTERNS[mod_key], li)
            m.weight.data.copy_(weight_cpu.to(m.weight.dtype).to(m.weight.device))


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _generator_perplexity(model, tokenizer, sequences: list) -> float:
    """Mean per-token cross-entropy loss over all sequences."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for seq in sequences:
            remainder = len(seq) % 6
            if remainder:
                seq = seq[remainder:]
            if not seq:
                continue
            bos = tokenizer.bos_token or ""
            seq = bos + seq
            inputs = tokenizer(seq, return_tensors="pt",
                               add_special_tokens=False).to(device)
            ids = inputs["input_ids"]
            if ids.shape[1] < 2:
                continue
            out          = model(input_ids=ids, labels=ids)
            n_tok         = ids.shape[1] - 1
            total_loss   += out.loss.item() * n_tok
            total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main grid search
# ─────────────────────────────────────────────────────────────────────────────

def run_scope_grid(
    model,
    tokenizer,
    sw_list:      list,
    scope:        str,
    criteria:     list,
    fracs:        list,
    n_rand_seeds: int,
    sequences:    list,
    bits:         int  = 8,
    num_layers:   int  = 30,
) -> dict:
    """Run the full grid search for a single scope.

    Returns a dict with baseline, yu_all, sw_fragility, and per-criterion
    sweep curves, suitable for JSON serialisation.
    """
    scope_modules = SCOPE_MODULES[scope]
    sw_exclusion  = _build_sw_exclusion(sw_list, scope_modules)
    candidates    = _build_candidates(model, scope_modules, num_layers, sw_exclusion)
    n_total       = len(candidates)
    n_excluded    = _count_sw_excluded(sw_exclusion)

    print(f"\n{'='*60}")
    print(f"  Scope : {scope}  |  modules: {scope_modules}")
    print(f"  Rows  : {n_total:,} eligible  ({n_excluded} SW-excluded across MLP)")
    print(f"  Bits  : INT{bits}")

    baseline_ppl = _generator_perplexity(model, tokenizer, sequences)
    print(f"  Baseline PPL: {baseline_ppl:.6f}")

    scope_results = {
        "scope":         scope,
        "modules":       scope_modules,
        "n_candidates":  n_total,
        "n_sw_excluded": n_excluded,
        "baseline_ppl":  baseline_ppl,
        "criteria":      {},
    }

    # ── yu_all: quantise EVERY eligible row (SW excluded) ────────────────────
    print("  [yu_all] Quantising all eligible rows (SW exempt) ...", flush=True)
    saves_all  = _apply_quantization(model, candidates, bits=bits)
    yu_all_ppl = _generator_perplexity(model, tokenizer, sequences)
    _restore_quantization(model, saves_all)
    yu_delta   = yu_all_ppl - baseline_ppl
    print(f"    yu_all PPL: {yu_all_ppl:.6f}  Δ={yu_delta:+.6f}")
    scope_results["yu_all_ppl"]   = yu_all_ppl
    scope_results["yu_all_delta"] = yu_delta

    # ── yu_all_including_sw: quantise EVERYTHING including SW rows ────────────
    # Build SW row list for this scope (same logic as sw_fragility below)
    _sw_rows_in_scope = []
    for mod_key in scope_modules:
        if mod_key == "down_proj":
            for sw in sw_list:
                _sw_rows_in_scope.append(("down_proj", sw["layer"], sw["row"]))
        elif mod_key in ("gate_proj", "up_proj"):
            for sw in sw_list:
                if sw.get("col") is not None:
                    _sw_rows_in_scope.append((mod_key, sw["layer"], sw["col"]))

    all_including_sw = candidates + _sw_rows_in_scope
    n_total_with_sw  = len(all_including_sw)
    print(f"  [yu_all_including_sw] Quantising all {n_total_with_sw:,} rows "
          f"(+{len(_sw_rows_in_scope)} SW) ...", flush=True)
    saves_full = _apply_quantization(model, all_including_sw, bits=bits)
    yu_all_sw_ppl = _generator_perplexity(model, tokenizer, sequences)
    _restore_quantization(model, saves_full)
    yu_sw_delta  = yu_all_sw_ppl - baseline_ppl
    sw_cost      = yu_all_sw_ppl - yu_all_ppl   # marginal cost of including SW rows
    print(f"    yu_all_including_sw PPL: {yu_all_sw_ppl:.6f}  Δ={yu_sw_delta:+.6f}  "
          f"SW marginal cost: {sw_cost:+.6f}")
    scope_results["yu_all_including_sw_ppl"]   = yu_all_sw_ppl
    scope_results["yu_all_including_sw_delta"] = yu_sw_delta
    scope_results["sw_marginal_cost"]          = sw_cost
    scope_results["n_sw_rows_in_scope"]        = len(_sw_rows_in_scope)

    # ── sw_fragility: quantise only the SW rows present in this scope ─────────
    sw_rows_to_q = []
    for mod_key in scope_modules:
        if mod_key == "down_proj":
            for sw in sw_list:
                sw_rows_to_q.append(("down_proj", sw["layer"], sw["row"]))
        elif mod_key in ("gate_proj", "up_proj"):
            for sw in sw_list:
                if sw.get("col") is not None:
                    sw_rows_to_q.append((mod_key, sw["layer"], sw["col"]))

    if sw_rows_to_q:
        print(f"  [sw_fragility] Quantising {len(sw_rows_to_q)} SW rows ...", flush=True)
        saves_sw = _apply_quantization(model, sw_rows_to_q, bits=bits)
        ppl_sw   = _generator_perplexity(model, tokenizer, sequences)
        _restore_quantization(model, saves_sw)
        sw_delta = ppl_sw - baseline_ppl
        print(f"    sw_fragility PPL: {ppl_sw:.6f}  Δ={sw_delta:+.6f}")
        scope_results["sw_fragility"] = {
            "n_sw_rows": len(sw_rows_to_q),
            "ppl":       ppl_sw,
            "delta_ppl": sw_delta,
        }
    else:
        scope_results["sw_fragility"] = None

    # ── Pre-rank for deterministic criteria ───────────────────────────────────
    ranked: dict = {}
    if "l1_low" in criteria:
        print("  [ranking] l1_low ...", flush=True)
        ranked["l1_low"] = _rank_l1_low(model, candidates)
    if "proximity" in criteria:
        print("  [ranking] proximity ...", flush=True)
        ranked["proximity"] = _rank_proximity(model, candidates, sw_list, num_layers)

    # ── Sweep fracs ──────────────────────────────────────────────────────────
    for criterion in criteria:
        print(f"\n  === Criterion: {criterion} ===")
        crit_results = []

        if criterion in ("l1_low", "proximity"):
            ranked_list = ranked[criterion]
            for frac in fracs:
                n_q    = max(1, int(round(n_total * frac / 100.0)))
                chosen = ranked_list[:n_q]
                saves  = _apply_quantization(model, chosen, bits=bits)
                ppl    = _generator_perplexity(model, tokenizer, sequences)
                _restore_quantization(model, saves)
                delta  = ppl - baseline_ppl
                print(f"    frac={frac:5.1f}%  n={n_q:7,}  "
                      f"PPL={ppl:.6f}  Δ={delta:+.6f}")
                crit_results.append({
                    "frac":      frac,
                    "n_rows":    n_q,
                    "ppl":       ppl,
                    "delta_ppl": delta,
                })

        elif criterion == "random":
            for frac in fracs:
                n_q  = max(1, int(round(n_total * frac / 100.0)))
                ppls = []
                for seed in range(n_rand_seeds):
                    chosen = _random.Random(seed).sample(candidates, n_q)
                    saves  = _apply_quantization(model, chosen, bits=bits)
                    ppl_r  = _generator_perplexity(model, tokenizer, sequences)
                    _restore_quantization(model, saves)
                    ppls.append(ppl_r)
                mean_ppl = float(np.mean(ppls))
                std_ppl  = float(np.std(ppls))
                delta    = mean_ppl - baseline_ppl
                print(f"    frac={frac:5.1f}%  n={n_q:7,}  "
                      f"PPL={mean_ppl:.6f}±{std_ppl:.6f}  Δ={delta:+.6f}")
                crit_results.append({
                    "frac":          frac,
                    "n_rows":        n_q,
                    "ppl_mean":      mean_ppl,
                    "ppl_std":       std_ppl,
                    "delta_ppl_mean": delta,
                    "seeds":         n_rand_seeds,
                })

        scope_results["criteria"][criterion] = crit_results

    return scope_results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_results(all_results: dict, plot_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available — skipping.")
        return

    model_name = all_results.get("model", "generator")
    bits       = all_results.get("bits", 8)
    scopes     = list(all_results.get("scopes", {}).keys())
    if not scopes:
        return

    # One column per scope, one row (PPL delta)
    ncols = len(scopes)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)

    colours = {
        "l1_low":    "steelblue",
        "proximity": "darkorange",
        "random":    "gray",
    }
    markers = {"l1_low": "o", "proximity": "s", "random": "^"}

    for col_idx, scope in enumerate(scopes):
        ax   = axes[0][col_idx]
        data = all_results["scopes"][scope]

        baseline = data.get("baseline_ppl", 0.0)
        yu_delta = data.get("yu_all_delta", None)
        sw_frag  = data.get("sw_fragility")

        # yu_all reference line
        if yu_delta is not None:
            ax.axhline(yu_delta, color="black", linewidth=1.2, linestyle=":",
                       label=f"yu_all INT{bits} Δ={yu_delta:+.4f}")

        # sw_fragility reference line
        if sw_frag and sw_frag.get("delta_ppl") is not None:
            sw_d = sw_frag["delta_ppl"]
            ax.axhline(sw_d, color="crimson", linewidth=1.5, linestyle="-.",
                       label=f"SW-only Δ={sw_d:+.4f} (n={sw_frag['n_sw_rows']})")

        # Per-criterion curves
        for crit, crit_data in data.get("criteria", {}).items():
            if not crit_data:
                continue
            fracs = [r["frac"] for r in crit_data]
            if crit == "random":
                deltas = [r["delta_ppl_mean"] for r in crit_data]
                stds   = [r["ppl_std"]        for r in crit_data]
                ax.errorbar(fracs, deltas, yerr=stds,
                            fmt=markers.get(crit, "x") + "--",
                            color=colours.get(crit, "purple"),
                            label=f"random (mean±std, n={crit_data[0]['seeds']})",
                            capsize=3)
            else:
                deltas = [r["delta_ppl"] for r in crit_data]
                ax.plot(fracs, deltas,
                        marker=markers.get(crit, "o"),
                        color=colours.get(crit, "purple"),
                        label=crit)

        ax.axhline(0, color="lightgray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"% of eligible rows quantised (INT{bits})")
        ax.set_ylabel("Δ PPL  (↑ worse)")
        ax.set_title(f"{model_name} / scope={scope}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {plot_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_generator(model_name: str, cfg: dict, device: str):
    """Load a GENERator model via its wrapper and return (model, tokenizer)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(repo_root))
    from models.generator_wrapper import GeneratorWrapper

    wrapper = GeneratorWrapper(cfg)
    wrapper.load()
    model     = wrapper.model
    tokenizer = wrapper.tokenizer
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Full-model INT8/INT4 quantisation grid search for GENERator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",
                   default="generator",
                   choices=["generator", "generator_prokaryote",
                            "generator_prokaryote_1b"],
                   help="Which GENERator variant to load.")
    p.add_argument("--scopes",
                   nargs="+",
                   default=["full"],
                   choices=list(SCOPE_MODULES.keys()),
                   help="Which module scope(s) to sweep.")
    p.add_argument("--criteria",
                   nargs="*",
                   default=["l1_low", "proximity", "random"],
                   choices=["l1_low", "proximity", "random"],
                   help="Row-selection criteria to sweep. Pass empty to skip the "
                        "sweep and only run baseline/yu_all/yu_all_including_sw/"
                        "sw_fragility conditions.")
    p.add_argument("--fracs",
                   nargs="+",
                   type=float,
                   default=[1.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0],
                   help="Percent of eligible rows to quantise at each grid point.")
    p.add_argument("--bits",
                   type=int,
                   default=8,
                   choices=[4, 8],
                   help="Quantisation bit width.")
    p.add_argument("--n_rand_seeds",
                   type=int,
                   default=5,
                   help="Seeds for the random criterion.")
    p.add_argument("--sw_index",
                   default="results/negative_results/super_weight_index.json",
                   help="Path to super_weight_index.json.")
    p.add_argument("--configs_dir",
                   default="configs",
                   help="Directory with model YAML configs.")
    p.add_argument("--out",
                   default=None,
                   help="Output JSON path (auto-derived from model name if omitted).")
    p.add_argument("--plot",
                   default=None,
                   help="Output plot PNG path (auto-derived from --out if omitted).")
    p.add_argument("--device",
                   default="cuda",
                   help="Torch device.")
    p.add_argument("--verify_quant",
                   action="store_true",
                   help="Before the main sweep, run a quick sanity check that "
                        "prints per-row quantisation error stats for one module. "
                        "Useful to confirm quantisation is actually being applied.")
    p.add_argument("--n_probe_seqs",
                   type=int, default=6,
                   help="Number of probe sequences for perplexity evaluation "
                        "(default 6 → 7,200 nt; use 100+ for powered eval).")
    p.add_argument("--probe_seq_len",
                   type=int, default=1200,
                   help="Length (nt) of each probe sequence.")
    p.add_argument("--probe_seed",
                   type=int, default=42,
                   help="Seed for probe sequence generation.")
    return p.parse_args()


def main():
    args   = parse_args()

    # ── Build probe sequences (configurable evaluation size) ──────────────────
    probe_seqs = _make_probe_seqs(
        n_seqs=args.n_probe_seqs,
        seq_len_nt=args.probe_seq_len,
        seed=args.probe_seed,
    )
    print(f"[probe] {len(probe_seqs)} seqs × {args.probe_seq_len} nt = "
          f"{len(probe_seqs) * args.probe_seq_len:,} nt "
          f"({sum(len(s) // 6 for s in probe_seqs):,} tokens ≈6-mer)")

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = Path(args.configs_dir) / f"{args.model}.yaml"
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ── Load SW index ─────────────────────────────────────────────────────────
    sw_list = []
    sw_index_path = Path(args.sw_index)
    if sw_index_path.exists():
        with open(sw_index_path) as f:
            sw_data = json.load(f)
        if args.model in sw_data:
            sw_list = sw_data[args.model]["results"]
            print(f"Loaded {len(sw_list)} SW entries for '{args.model}':")
            for sw in sw_list:
                print(f"  layer={sw['layer']}  row={sw['row']}  "
                      f"col={sw.get('col', 'N/A')}  out_max={sw.get('out_max', '?'):.2e}")
        else:
            print(f"Warning: '{args.model}' not found in SW index. "
                  f"SW-exclusion will be empty.")
    else:
        print(f"Warning: SW index not found at '{args.sw_index}'. "
              f"Proceeding without SW-exclusion.")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {args.model} ...", flush=True)
    model, tokenizer = _load_generator(args.model, cfg, args.device)
    num_layers       = cfg["num_layers"]
    print(f"  num_layers={num_layers}  device={next(model.parameters()).device}")

    # Sanity: print actual module shapes for the first layer
    print("  Module shapes (layer 0):")
    for mod_key in SCOPE_MODULES["full"]:
        try:
            m = _resolve(model, LLAMA_PATTERNS[mod_key], 0)
            print(f"    {mod_key:12s}: {tuple(m.weight.data.shape)}")
        except AttributeError:
            print(f"    {mod_key:12s}: NOT FOUND")

    # ── Optional quantisation sanity check ───────────────────────────────────
    if args.verify_quant:
        print(f"\n=== Quantisation sanity check (down_proj layer 0, INT{args.bits}) ===")
        m_check   = _resolve(model, LLAMA_PATTERNS["down_proj"], 0)
        w_orig    = m_check.weight.data.float().cpu()          # (nrows, dim)
        from collections import defaultdict
        saves_chk = _apply_quantization(
            model, [("down_proj", 0, ri) for ri in range(w_orig.shape[0])],
            bits=args.bits
        )
        w_quant   = m_check.weight.data.float().cpu()
        _restore_quantization(model, saves_chk)
        diff      = (w_quant - w_orig)
        rel_err   = diff.abs() / (w_orig.abs().amax(dim=1, keepdim=True).clamp(min=1e-12))
        print(f"  Weight change  max_abs={diff.abs().max():.6f}  "
              f"mean_abs={diff.abs().mean():.6f}  "
              f"rms={diff.pow(2).mean().sqrt():.6f}")
        print(f"  Relative error  max={rel_err.max():.6f}  mean={rel_err.mean():.6f}")
        print(f"  Rows with zero change: "
              f"{(diff.abs().sum(dim=1) == 0).sum().item()} / {w_orig.shape[0]}")
        print(f"  Restored correctly: "
              f"{torch.allclose(m_check.weight.data.float().cpu(), w_orig, atol=1e-5)}")
        print(f"  Eval token count: "
              f"{sum(len(s) // 6 for s in probe_seqs)} tokens across {len(probe_seqs)} seqs")

    # ── Grid search ───────────────────────────────────────────────────────────
    all_results = {
        "model":        args.model,
        "bits":         args.bits,
        "fracs":        args.fracs,
        "n_rand_seeds": args.n_rand_seeds,
        "criteria":     args.criteria,
        "sw_list":      sw_list,
        "scopes":       {},
        "probe": {
            "n_seqs": args.n_probe_seqs,
            "seq_len_nt": args.probe_seq_len,
            "total_nt": args.n_probe_seqs * args.probe_seq_len,
            "approx_tokens": sum(len(s) // 6 for s in probe_seqs),
            "seed": args.probe_seed,
        },
    }

    for scope in args.scopes:
        scope_res = run_scope_grid(
            model        = model,
            tokenizer    = tokenizer,
            sw_list      = sw_list,
            scope        = scope,
            criteria     = args.criteria,
            fracs        = args.fracs,
            n_rand_seeds = args.n_rand_seeds,
            sequences    = probe_seqs,
            bits         = args.bits,
            num_layers   = num_layers,
        )
        all_results["scopes"][scope] = scope_res

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path  = args.out  or f"results/whole_model_quant_{args.model}.json"
    plot_path = args.plot or out_path.replace(".json", ".png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    _plot_results(all_results, plot_path)


if __name__ == "__main__":
    main()
