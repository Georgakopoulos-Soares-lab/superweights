"""
scripts/analysis/run_subspace_retention.py
---------------------------------------------
PHASE A1 -- subspace retention: generalize the retention metric from 1-D
(a single coordinate) to k-D (the super-weight SUBSPACE).

Motivation (Task Spec v2 A1): `coord_fraction` measures survival of ONE
coordinate. DNABERT-2's super weight is a 10-row redundant ensemble writing
to ~6 unique down-projection output coordinates -- measuring a single
coordinate is the wrong dimensionality for an inherently distributed
representation. This asks whether the SW *subspace* survives to readout even
when no individual coordinate does.

Math (S = set of unique SW output coordinates, P_S = projection onto
span{e_k : k in S}; since S is axis-aligned this is just a column gather):

    subspace_energy(i)   = || P_S . dh_i ||_F^2
    total_energy(i)      = || dh_i ||_F^2
    subspace_fraction(i) = subspace_energy(i) / total_energy(i)

Primary statistic: subspace_fraction at the FINAL block, restricted to the
task readout position (CLS for encoders; all prediction positions for causal
LMs). For |S| == 1 this reduces exactly to T1.1's coord_fraction, so the
metric is a strict generalization that puts every model on one axis.

CONTROL: a random |S|-dimensional coordinate subspace, seed 42, 10 draws.
Each draw preserves the SW set's exact layer structure -- each unique SW row
is mapped to a unique random row, and that random row is ablated at every
layer where its SW counterpart appeared -- so the control differs from the
real measurement ONLY in which coordinates were chosen.

Usage (one model per invocation; see run_phase_a1_all.sh for the sweep):
    python scripts/analysis/run_subspace_retention.py --model dnabert2

Output: results/mechanism/subspace_retention_{model}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models import WRAPPER_MAP  # noqa: E402
from src.ablation_trace import (  # noqa: E402
    get_blocks, run_clean_trace, BlockRecorder, install_ablation_hooks_multi,
)
from src.probe_windows import sample_probe_windows  # noqa: E402

SEED = 42

KINGDOM = {
    "generator": "euk",
    "generator_prokaryote": "prok",
    "dnabert2": "euk",
    "ntv3": "euk",
    "evo1": "prok",
}

# "encoder" -> readout at the pooled/CLS position (index 0);
# "causal"  -> readout over all prediction positions.
MODEL_TYPE = {
    "generator": "causal",
    "generator_prokaryote": "causal",
    "dnabert2": "encoder",
    "ntv3": "encoder",
    "evo1": "causal",
}


# ─────────────────────────────────────────────────────────────────────────────
# Super-weight sets
# ─────────────────────────────────────────────────────────────────────────────

def load_sw_pairs(model_key: str, variant: str) -> list[tuple[int, int]]:
    """Return the list of (layer, row) SW coordinates to ablate.

    variant="spec": the set named explicitly in Task Spec v2 A1.
    variant="index": every entry in results/super_weight_index.json.

    Both are run because they disagree for two models, and the disagreement
    is itself decision-relevant: the spec calls Evo1 a single row {3776},
    but the frozen index actually holds a TEN-row ensemble at layer 11 --
    so "DNABERT-2 is an ensemble, Evo1 is not" cannot be the thing that
    distinguishes them. Reporting both makes the A1 verdict robust to that
    choice instead of hiding it.
    """
    if model_key == "generator_prokaryote":
        # Canonical index entry (L2/r1927) is eukaryotic-probe contamination
        # (prior session, CHECKPOINT_REPORT.md). Use the corrected detection.
        corrected = json.loads(
            (ROOT / "results/mechanism/generator_prokaryote_sw_corrected.json").read_text())
        c = corrected["corrected_super_weight"]
        if variant == "spec":
            return [(c["layer"], c["row"])]
        # "index" variant: primary + the secondary row found at the same layer
        pairs = [(c["layer"], c["row"])]
        sec = c.get("secondary_row")
        if sec:
            pairs.append((sec.get("layer", c["layer"]), sec["row"]))
        return pairs

    index = json.loads((ROOT / "results/super_weight_index.json").read_text())
    entries = index[model_key]["results"]
    all_pairs = [(e["layer"], e["row"]) for e in entries]

    if variant == "index":
        return all_pairs

    # variant == "spec"
    if model_key == "generator":
        return [(4, 2371)]          # spec: "the single coordinate {2371}"
    if model_key == "evo1":
        return [(11, 3776)]         # spec: "{3776}"
    if model_key == "dnabert2":
        return all_pairs            # spec: the full 10-row ensemble
    return all_pairs


def subspace_of(pairs) -> list[int]:
    """S = unique output coordinates written by the SW rows."""
    return sorted({row for _, row in pairs})


def random_pairs_like(pairs, d_model: int, rng) -> list[tuple[int, int]]:
    """A random SW set with the SAME layer structure and |S|.

    Each unique real SW row is mapped to a distinct random coordinate, and
    that random coordinate is placed at exactly the layers where its real
    counterpart appeared. So the control matches the real ablation in
    dimensionality, in row-count, and in per-layer structure.
    """
    real_rows = subspace_of(pairs)
    choices = rng.choice(d_model, size=len(real_rows), replace=False)
    mapping = {r: int(c) for r, c in zip(real_rows, choices)}
    return [(layer, mapping[row]) for layer, row in pairs]


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement
# ─────────────────────────────────────────────────────────────────────────────

def ablated_trace(wrapper, model_key: str, blocks, pairs, sequence: str) -> list:
    handles = install_ablation_hooks_multi(wrapper, pairs, model_key=model_key)
    try:
        with BlockRecorder(model_key, blocks) as rec:
            wrapper.forward(sequence)
            return list(rec.outputs)
    finally:
        for h in handles:
            h.remove()


def subspace_profile(clean, ablated, coords: list[int], readout: str) -> dict:
    """Per-layer subspace/total energy of the ablation difference trace.

    Returns per-layer lists plus the final-layer statistics, computed both
    over all positions and restricted to the task readout position.
    """
    n_layers = len(clean)
    sub_e = np.zeros(n_layers)
    tot_e = np.zeros(n_layers)
    frac = np.zeros(n_layers)
    frac_task = np.zeros(n_layers)

    for i in range(n_layers):
        if clean[i] is None or ablated[i] is None:
            continue
        c, a = clean[i], ablated[i]
        seq_len = min(c.shape[0], a.shape[0])
        delta = (c[:seq_len] - a[:seq_len]).float()

        sub_e[i] = float((delta[:, coords] ** 2).sum())
        tot_e[i] = float((delta ** 2).sum())
        frac[i] = sub_e[i] / tot_e[i] if tot_e[i] > 0 else 0.0

        # task-position readout: CLS (index 0) for encoders, all positions
        # for causal LMs (every position is a prediction position)
        d_task = delta[0:1] if readout == "encoder" else delta
        st = float((d_task[:, coords] ** 2).sum())
        tt = float((d_task ** 2).sum())
        frac_task[i] = st / tt if tt > 0 else 0.0

    return {
        "subspace_energy": sub_e.tolist(),
        "total_energy": tot_e.tolist(),
        "subspace_fraction": frac.tolist(),
        "subspace_fraction_taskpos": frac_task.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(KINGDOM.keys()))
    ap.add_argument("--n_windows", type=int, default=10)
    ap.add_argument("--window_bp", type=int, default=3072)
    ap.add_argument("--n_random_draws", type=int, default=10)
    ap.add_argument("--n_random_windows", type=int, default=3,
                    help="windows used for the random-subspace control "
                         "(draws x windows measurements total)")
    args = ap.parse_args()

    model_key = args.model

    # NTv3 is excluded by construction, not by omission -- record why.
    if model_key == "ntv3":
        out = {
            "model": "ntv3",
            "computable": False,
            "reason": (
                "NTv3's registered super weight (L11, row 1472) sits at the LAST block of "
                "its 12-block transformer tower, but the network's true output passes "
                "through a further U-Net deconv/upsampling tower whose ConvTowerBlock/"
                "DeconvBlock stages change the channel count (dim_in != dim_out). A "
                "coordinate index therefore has no stable identity between the SW layer "
                "and the readout, so span{e_k} is undefined downstream and "
                "subspace_fraction_final cannot be computed. Additionally L == F for the "
                "traceable portion, making any retention ratio tautological. This is an "
                "architectural finding, NOT evidence for or against the retention thesis."
            ),
            "dtype": "float32",
            "seed": SEED,
        }
        p = ROOT / "results/mechanism/subspace_retention_ntv3.json"
        p.write_text(json.dumps(out, indent=2))
        print(f"[A1] ntv3 -- not computable by construction; wrote {p}")
        return

    print(f"[A1] {model_key}: loading model ...")
    config = yaml.safe_load((ROOT / f"configs/{model_key}.yaml").read_text())
    wrapper = WRAPPER_MAP[model_key](config)
    wrapper.load()
    blocks = get_blocks(model_key, wrapper)
    n_layers = len(blocks)
    readout = MODEL_TYPE[model_key]

    windows = sample_probe_windows(KINGDOM[model_key], n=args.n_windows,
                                    window_bp=args.window_bp, seed=SEED)
    print(f"[A1] {model_key}: {n_layers} blocks, {len(windows)} probe windows")

    variants = {}
    for variant in ("spec", "index"):
        pairs = load_sw_pairs(model_key, variant)
        coords = subspace_of(pairs)
        print(f"\n[A1] {model_key} / variant={variant}: "
              f"{len(pairs)} SW rows -> |S|={len(coords)} coords {coords} "
              f"at layers {sorted({l for l, _ in pairs})}")

        per_window_frac, per_window_frac_task = [], []
        per_window_sub_e, per_window_tot_e = [], []
        d_model = None

        for wi, w in enumerate(windows):
            clean = run_clean_trace(wrapper, model_key, w["seq"])
            d_model = clean[0].shape[-1]
            abl = ablated_trace(wrapper, model_key, blocks, pairs, w["seq"])
            prof = subspace_profile(clean, abl, coords, readout)
            per_window_frac.append(prof["subspace_fraction"])
            per_window_frac_task.append(prof["subspace_fraction_taskpos"])
            per_window_sub_e.append(prof["subspace_energy"])
            per_window_tot_e.append(prof["total_energy"])
            print(f"    window {wi+1}/{len(windows)} [{w['label']}] "
                  f"subspace_fraction_final={prof['subspace_fraction'][-1]:.5f} "
                  f"(taskpos {prof['subspace_fraction_taskpos'][-1]:.5f})")

        mean_frac = np.mean(per_window_frac, axis=0)
        mean_frac_task = np.mean(per_window_frac_task, axis=0)
        mean_sub_e = np.mean(per_window_sub_e, axis=0)

        F = n_layers - 1
        L = min(l for l, _ in pairs)
        finals = np.array([f[F] for f in per_window_frac])
        finals_task = np.array([f[F] for f in per_window_frac_task])

        # absolute (blow-up-confounded) form, reported only alongside the fraction
        subspace_retention_abs = (float(mean_sub_e[F] / mean_sub_e[L])
                                   if mean_sub_e[L] > 0 else float("nan"))

        # ---- random-subspace control ----
        print(f"[A1] {model_key} / {variant}: random control "
              f"({args.n_random_draws} draws x {args.n_random_windows} windows) ...")
        rand_finals, rand_finals_task = [], []
        for draw in range(args.n_random_draws):
            rng = np.random.RandomState(SEED + draw)
            rpairs = random_pairs_like(pairs, d_model, rng)
            rcoords = subspace_of(rpairs)
            for w in windows[: args.n_random_windows]:
                clean = run_clean_trace(wrapper, model_key, w["seq"])
                abl = ablated_trace(wrapper, model_key, blocks, rpairs, w["seq"])
                prof = subspace_profile(clean, abl, rcoords, readout)
                rand_finals.append(prof["subspace_fraction"][F])
                rand_finals_task.append(prof["subspace_fraction_taskpos"][F])
        rand_finals = np.array(rand_finals)
        rand_finals_task = np.array(rand_finals_task)

        # separation vs the random floor, in sd units
        z = ((finals.mean() - rand_finals.mean()) / rand_finals.std()
             if rand_finals.std() > 0 else float("inf"))
        z_task = ((finals_task.mean() - rand_finals_task.mean()) / rand_finals_task.std()
                  if rand_finals_task.std() > 0 else float("inf"))

        variants[variant] = {
            "sw_pairs": [[l, r] for l, r in pairs],
            "subspace_coords": coords,
            "subspace_dim": len(coords),
            "sw_layers": sorted({l for l, _ in pairs}),
            "source_layer_L": L,
            "final_layer_F": F,
            "per_layer_mean_subspace_fraction": mean_frac.tolist(),
            "per_layer_mean_subspace_fraction_taskpos": mean_frac_task.tolist(),
            "subspace_fraction_final": {
                "mean": float(finals.mean()), "sd": float(finals.std()),
                "n": int(finals.size)},
            "subspace_fraction_final_taskpos": {
                "mean": float(finals_task.mean()), "sd": float(finals_task.std()),
                "n": int(finals_task.size)},
            "subspace_retention_absolute": subspace_retention_abs,
            "subspace_retention_absolute_caveat": (
                "Absolute ratio subspace_energy(F)/subspace_energy(L). Confounded when "
                "total perturbation energy itself blows up between L and F (as it does "
                "~700-1000x in Evo1). Never interpret without subspace_fraction_final."),
            "random_control": {
                "n_draws": args.n_random_draws,
                "n_windows": args.n_random_windows,
                "final_mean": float(rand_finals.mean()),
                "final_sd": float(rand_finals.std()),
                "final_taskpos_mean": float(rand_finals_task.mean()),
                "final_taskpos_sd": float(rand_finals_task.std()),
            },
            "separation_z_vs_random": float(z),
            "separation_z_vs_random_taskpos": float(z_task),
        }
        print(f"[A1] {model_key} / {variant}: "
              f"subspace_fraction_final={finals.mean():.5f}+-{finals.std():.5f}  "
              f"random={rand_finals.mean():.5f}+-{rand_finals.std():.5f}  z={z:.2f}")

    out = {
        "model": model_key,
        "computable": True,
        "n_layers": n_layers,
        "n_windows": len(windows),
        "window_bp": args.window_bp,
        "kingdom": KINGDOM[model_key],
        "readout": readout,
        "dtype": "float32",
        "seed": SEED,
        "variants": variants,
        "note_prok": ("generator_prokaryote uses the CORRECTED super weight (L8/r260) from "
                       "results/mechanism/generator_prokaryote_sw_corrected.json, not the "
                       "contaminated canonical index entry L2/r1927."
                       if model_key == "generator_prokaryote" else None),
    }
    p = ROOT / f"results/mechanism/subspace_retention_{model_key}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n[A1] wrote {p}")


if __name__ == "__main__":
    main()
