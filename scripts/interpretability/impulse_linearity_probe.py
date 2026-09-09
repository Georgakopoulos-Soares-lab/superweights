"""
scripts/interpretability/impulse_linearity_probe.py
----------------------------------------------------
Instrument calibration for the broadcast assay. Not a new experiment: it measures whether
T is a property of the model or of the ε that was chosen.

T_m = ‖Δh_m‖ / ε is only comparable across models if the response is linear in ε, i.e. if
‖Δh‖ ∝ ε so that the ratio is ε-invariant. The first AC-relative run on DNABERT-2 showed
‖Δh‖ ≈ 4.5 at BOTH ε = 1.0 and ε = 0.0037 — a 270× change in ε with no change in response.
If that holds, T is dominated by its own denominator and every cross-model T comparison in
R3 is comparing ε choices rather than routing behaviour.

Sweeps ε over several decades at the SW row and reports ‖Δh‖ at the first downstream layer,
T, and output KL. A linear regime shows ‖Δh‖ ∝ ε (T flat); saturation shows ‖Δh‖ flat
(T ∝ 1/ε).

Usage:
  python scripts/interpretability/impulse_linearity_probe.py --model dnabert2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.interpretability.run_sw_broadcast_impulse import (  # noqa: E402
    SW_TARGETS, _get_blocks, _get_logits, _load_model, _tokenize,
    ac_epsilon, kl_divergence, run_pass,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2", choices=list(SW_TARGETS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tgt = SW_TARGETS[args.model]
    sw_layer, sw_row, arch, is_causal = (
        tgt["sw_layer"], tgt["sw_row"], tgt["arch"], tgt["causal"])

    from src.dna_probes import get_probe
    seq = get_probe(tgt["probe"])

    model, tok = _load_model(args.model, args.device)
    blocks = _get_blocks(model, arch)
    inp = _tokenize(tok, seq, arch, args.device)

    clean_hs, clean_out = run_pass(model, inp, blocks, arch)
    logits_clean = _get_logits(clean_out, arch)
    src = clean_hs[sw_layer]
    sw_pos = int(src[:, sw_row].abs().argmax().item())

    eps_ac, meta = ac_epsilon(src)
    print(f"[{args.model}] sw_layer={sw_layer} sw_row={sw_row} sw_pos={sw_pos}")
    print(f"[{args.model}] std_AC={meta['std_ac']:.4e}  eps_AC(alpha=0.01)={eps_ac:.4e}")
    print(f"[{args.model}] median|h|={meta['median_abs_h']:.4e}  max|h|={meta['max_abs_h']:.4e}")

    first_down = sw_layer + 1
    eps_list = sorted({eps_ac, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0})

    print(f"\n{'eps':>12}  {'||dh||@L%d' % first_down:>12}  {'T=||dh||/eps':>14}  {'KL':>12}")
    rows = []
    for eps in eps_list:
        ph, po = run_pass(model, inp, blocks, arch, inject_layer=sw_layer,
                          inject_pos=sw_pos, inject_row=sw_row, epsilon=eps)
        c, p = clean_hs[first_down], ph[first_down]
        L = min(c.shape[0], p.shape[0])
        delta = p[:L].float() - c[:L].float()
        dnorm = delta.norm(dim=-1)
        dh = float(dnorm.mean())
        kl = kl_divergence(logits_clean, _get_logits(po, arch), sw_pos, is_causal)
        tag = "  <- AC eps" if abs(eps - eps_ac) < 1e-12 else ""
        print(f"{eps:12.4e}  {dh:12.4e}  {dh/eps:14.4e}  {kl:12.4e}{tag}")
        rows.append({"eps": eps, "dh_mean": dh, "T": dh / eps, "kl": kl})

    # linearity verdict: over the decades spanned, does ||dh|| track eps?
    lo, hi = rows[0], rows[-1]
    eps_ratio = hi["eps"] / lo["eps"]
    dh_ratio = hi["dh_mean"] / lo["dh_mean"] if lo["dh_mean"] > 0 else float("inf")
    print(f"\n  eps spans {eps_ratio:.3g}x;  ||dh|| spans {dh_ratio:.3g}x")
    if dh_ratio < 0.1 * eps_ratio:
        verdict = ("SATURATED / eps-INDEPENDENT — T tracks 1/eps, not the model. "
                   "Cross-model T is not comparable at differing eps.")
    elif 0.5 * eps_ratio <= dh_ratio <= 2.0 * eps_ratio:
        verdict = "LINEAR — T is eps-invariant and cross-model comparison is sound."
    else:
        verdict = "PARTIALLY LINEAR — T is eps-dependent; report the eps alongside T."
    print(f"  VERDICT: {verdict}")

    out = Path(args.out or ROOT / "results" / f"impulse_linearity_{args.model}.json")
    out.write_text(json.dumps(
        {"model": args.model, "sw_layer": sw_layer, "sw_row": sw_row, "sw_pos": sw_pos,
         "first_downstream_layer": first_down, "eps_ac": eps_ac, "ac_meta": meta,
         "sweep": rows, "eps_span": eps_ratio, "dh_span": dh_ratio,
         "verdict": verdict}, indent=2))
    print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
