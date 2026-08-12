"""
scripts/interpretability/evo1_input_sensitivity_gate.py
-------------------------------------------------------
STEP 2 GATE for E2 (see docs/prereg/PREREG_evo1_broadcast.md v2, §Precondition).

The Evo1 residual was reported "bit-for-bit identical from layer 13 onward". If that
holds of the *hidden state* rather than of a summary statistic, Evo1's output is
input-independent from L13, which is incompatible with its reported PPL of 3.11 — and
every measurement taken through this trace, including the residual-attribution numbers
already in the manuscript, would be uninterpretable.

Test: forward clearly different input sequences and diff the layer-20 hidden state.

  differ    -> the median statistic is stable while per-coordinate values vary.
               An AC component exists. Proceed to the amended (AC-relative ε) protocol.
  identical -> the trace is broken. STOP.

Model loading matches run_sw_broadcast_impulse.py exactly (bf16 everywhere except
poles/residues; forward under torch.autocast bf16), so the gate tests the same numerical
path the assay runs on — not a cleaner one.

Secondary, free from the same passes and reported as secondary:
  - layer-to-layer diff (h_13 vs h_20, same input): distinguishes "frozen across layers"
    from "frozen across inputs". These are different failure modes and the phrase
    "bit-for-bit identical from layer 13" is ambiguous between them.
  - DC/AC magnitudes per layer: median |h|, mean(h), std(h - mean(h)). This is the
    quantity the STEP 3 ε formula consumes; it is NOT the AC/DC decomposition Branch C
    would require (that one is about downstream *dependence* on the AC component).

Usage (inside evo2.sif):
  apptainer exec --nv evo2.sif python scripts/interpretability/evo1_input_sensitivity_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

GATE_LAYER = 20          # the layer the prereg names
FREEZE_LAYER = 13        # first layer of the reported "freeze"
SW_LAYER = 11
SW_ROW = 3776
REPORT_LAYERS = [SW_LAYER, 12, FREEZE_LAYER, 15, GATE_LAYER, 25, 31]


def load_evo1(device: str = "cuda"):
    """Identical to _load_model('evo1', ...) in run_sw_broadcast_impulse.py."""
    from evo import Evo
    evo = Evo("evo-1-8k-base", device=device)
    for n, p in evo.model.named_parameters():
        if any(k in n for k in ("poles", "residues")):
            continue
        p.data = p.data.to(torch.bfloat16)
    return evo.model, evo.tokenizer


def capture_blocks(model, tokenizer, seq: str, device: str = "cuda") -> dict[int, torch.Tensor]:
    """Return {block_index: (T, D) hidden state}, captured exactly as the assay does."""
    blocks = list(model.blocks)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(li: int):
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[li] = (hs[0] if hs.dim() == 3 else hs).detach().cpu()
        return hook

    for li, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(li)))

    ids = torch.tensor(list(tokenizer.tokenize(seq)), dtype=torch.long,
                       device=device).unsqueeze(0)
    try:
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(ids)
    finally:
        for h in handles:
            h.remove()
    return captured


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Coordinate-wise difference stats between two (T, D) hidden states."""
    T = min(a.shape[0], b.shape[0])
    x, y = a[:T].float(), b[:T].float()
    d = (x - y).abs()
    n = d.numel()
    return {
        "shape": [T, int(a.shape[-1])],
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "median_abs_diff": float(d.median()),
        "frac_coords_differ": float((d > 0).sum().item() / n),
        "n_coords_differ": int((d > 0).sum().item()),
        "n_coords_total": int(n),
        "bitwise_identical": bool(torch.equal(x, y)),
        # scale-relative: is the difference meaningful next to the values themselves?
        "max_abs_diff_over_median_abs_h": float(
            d.max() / (x.abs().median() + 1e-30)),
        # at the SW coordinate specifically
        "sw_row_max_abs_diff": float(d[:, SW_ROW].max()),
        "sw_row_frac_differ": float((d[:, SW_ROW] > 0).sum().item() / T),
    }


def dc_ac(h: torch.Tensor) -> dict:
    """DC offset vs AC (input-dependent) component of one (T, D) hidden state."""
    x = h.float()
    mean_per_token = x.mean(dim=-1, keepdim=True)       # DC across the 4096 dims
    ac = x - mean_per_token
    return {
        "median_abs_h": float(x.abs().median()),
        "max_abs_h": float(x.abs().max()),
        "mean_h": float(x.mean()),
        "std_ac": float(ac.std()),
        "eps_alpha_0.01": float(0.01 * ac.std()),
        "sw_row_median_abs_h": float(x[:, SW_ROW].abs().median()),
        "sw_row_max_abs_h": float(x[:, SW_ROW].abs().max()),
    }


def main() -> int:
    from probes.dna_probes import get_probe

    device = "cuda"
    # Same length (504 nt) so the diff is coordinate-aligned end to end.
    seqs = {
        "actb_500": get_probe("actb_500"),                    # assay probe, human ACTB CDS
        "poly_a": get_probe("poly_a"),                        # maximally different
        "pseudomonadota_504": get_probe("pseudomonadota")[:504],  # unrelated real sequence
    }
    for k, v in seqs.items():
        print(f"  {k:20s} len={len(v)}  {v[:40]}")
    assert len({len(v) for v in seqs.values()}) == 1, "sequences must be equal length"

    print("\nloading Evo1 (bf16 except poles/residues) ...", flush=True)
    model, tokenizer = load_evo1(device)

    caps: dict[str, dict[int, torch.Tensor]] = {}
    for name, seq in seqs.items():
        print(f"forward: {name} ...", flush=True)
        caps[name] = capture_blocks(model, tokenizer, seq, device)

    report: dict = {
        "gate_layer": GATE_LAYER,
        "sw_layer": SW_LAYER,
        "sw_row": SW_ROW,
        "sequences": {k: {"len": len(v), "head": v[:60]} for k, v in seqs.items()},
        "pairs": {},
        "layer_to_layer_same_input": {},
        "dc_ac_by_layer": {},
    }

    # ── PRIMARY: input-dependence at the gate layer ──────────────────────────
    pairs = [("actb_500", "poly_a"),
             ("actb_500", "pseudomonadota_504"),
             ("poly_a", "pseudomonadota_504")]
    print(f"\n=== PRIMARY: layer-{GATE_LAYER} hidden state, input vs input ===")
    for a, b in pairs:
        st = diff_stats(caps[a][GATE_LAYER], caps[b][GATE_LAYER])
        report["pairs"][f"{a}__vs__{b}"] = st
        print(f"\n  {a} vs {b}   shape={st['shape']}")
        print(f"    max  |Δ|            = {st['max_abs_diff']:.6e}")
        print(f"    mean |Δ|            = {st['mean_abs_diff']:.6e}")
        print(f"    median |Δ|          = {st['median_abs_diff']:.6e}")
        print(f"    coords differing    = {st['n_coords_differ']}/{st['n_coords_total']}"
              f"  ({100*st['frac_coords_differ']:.4f}%)")
        print(f"    bitwise identical   = {st['bitwise_identical']}")
        print(f"    max|Δ| / median|h|  = {st['max_abs_diff_over_median_abs_h']:.4e}")
        print(f"    row {SW_ROW}: max|Δ|={st['sw_row_max_abs_diff']:.6e}  "
              f"frac tokens differing={st['sw_row_frac_differ']:.4f}")

    # also sweep the other reported layers, so "from L13 onward" is tested as stated
    print(f"\n=== input-dependence across layers (actb_500 vs poly_a) ===")
    per_layer = {}
    for li in sorted(caps["actb_500"].keys()):
        st = diff_stats(caps["actb_500"][li], caps["poly_a"][li])
        per_layer[str(li)] = {k: st[k] for k in
                              ("max_abs_diff", "mean_abs_diff", "frac_coords_differ",
                               "bitwise_identical")}
        if li in REPORT_LAYERS or li >= FREEZE_LAYER:
            print(f"  L{li:2d}  max|Δ|={st['max_abs_diff']:.4e}  "
                  f"mean|Δ|={st['mean_abs_diff']:.4e}  "
                  f"frac={st['frac_coords_differ']:.4f}  "
                  f"identical={st['bitwise_identical']}")
    report["per_layer_actb_vs_polya"] = per_layer

    # ── SECONDARY: is the state frozen *across layers* for one input? ────────
    print(f"\n=== SECONDARY: same input (actb_500), layer {FREEZE_LAYER} vs later ===")
    for li in [15, GATE_LAYER, 25, 31]:
        st = diff_stats(caps["actb_500"][FREEZE_LAYER], caps["actb_500"][li])
        report["layer_to_layer_same_input"][f"L{FREEZE_LAYER}_vs_L{li}"] = st
        print(f"  L{FREEZE_LAYER} vs L{li:2d}  max|Δ|={st['max_abs_diff']:.4e}  "
              f"mean|Δ|={st['mean_abs_diff']:.4e}  "
              f"frac={st['frac_coords_differ']:.4f}  identical={st['bitwise_identical']}")

    # ── SECONDARY: DC/AC magnitudes (input to the STEP 3 ε formula) ──────────
    print(f"\n=== SECONDARY: DC/AC magnitude by layer (actb_500) ===")
    print(f"  {'L':>3}  {'median|h|':>12}  {'max|h|':>12}  {'mean h (DC)':>13}  "
          f"{'std AC':>12}  {'eps@a=0.01':>12}")
    for li in sorted(caps["actb_500"].keys()):
        s = dc_ac(caps["actb_500"][li])
        report["dc_ac_by_layer"][str(li)] = s
        if li in REPORT_LAYERS:
            print(f"  {li:3d}  {s['median_abs_h']:12.4e}  {s['max_abs_h']:12.4e}  "
                  f"{s['mean_h']:13.4e}  {s['std_ac']:12.4e}  {s['eps_alpha_0.01']:12.4e}")

    # ── verdict ─────────────────────────────────────────────────────────────
    gate_pairs = [report["pairs"][f"{a}__vs__{b}"] for a, b in pairs]
    all_differ = all(not p["bitwise_identical"] for p in gate_pairs)
    report["verdict"] = "DIFFER (AC component exists)" if all_differ else "IDENTICAL (trace broken)"
    print(f"\n=== VERDICT: {report['verdict']} ===")

    out = ROOT / "results" / "evo1_input_sensitivity_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"  saved -> {out}")
    return 0 if all_differ else 2


if __name__ == "__main__":
    sys.exit(main())
