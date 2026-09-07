"""
scripts/interpretability/impulse_determinism_check.py
------------------------------------------------------
Noise-floor check for the broadcast assay.

The assay computes Δh between a clean and a perturbed forward pass. That is only a
measurement of the perturbation if the forward pass is deterministic. This script runs the
SAME input twice with NO injection and reports ‖Δh‖ and KL between the two — the noise
floor — then runs the actual SW injection and reports the same quantities as signal.

If signal ≲ noise, every T/C/KL number from that model is measuring run-to-run
nondeterminism rather than causal broadcast.

Found on DNABERT-2, where two identical passes differ by KL = 0.53 while the published SW
impulse KL is 0.31.

Usage:
  python scripts/interpretability/impulse_determinism_check.py --model dnabert2
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


def dh(a: torch.Tensor, b: torch.Tensor) -> float:
    L = min(a.shape[0], b.shape[0])
    return float((b[:L].float() - a[:L].float()).norm(dim=-1).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2", choices=list(SW_TARGETS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    t = SW_TARGETS[args.model]
    sl, sr, arch, causal = t["sw_layer"], t["sw_row"], t["arch"], t["causal"]

    from probes.dna_probes import get_probe
    model, tok = _load_model(args.model, args.device)
    blocks = _get_blocks(model, arch)
    inp = _tokenize(tok, get_probe(t["probe"]), arch, args.device)

    ref, ref_out = run_pass(model, inp, blocks, arch)
    logits_ref = _get_logits(ref_out, arch)
    src = ref[sl]
    pos = int(src[:, sr].abs().argmax().item())
    eps, meta = ac_epsilon(src)

    downstream = [li for li in sorted(ref) if li > sl]
    print(f"[{args.model}] sw_layer={sl} sw_row={sr} sw_pos={pos} "
          f"dtype={next(model.parameters()).dtype}")
    print(f"[{args.model}] eps_AC={eps:.4e}  std_AC={meta['std_ac']:.4e}  "
          f"median|h|={meta['median_abs_h']:.4e}")

    # ── noise floor: identical calls, no injection ──────────────────────────
    noise = {li: [] for li in downstream}
    noise_kl = []
    for _ in range(args.repeats):
        rep, rep_out = run_pass(model, inp, blocks, arch)
        for li in downstream:
            noise[li].append(dh(ref[li], rep[li]))
        noise_kl.append(kl_divergence(logits_ref, _get_logits(rep_out, arch), pos, causal))

    # ── signal: the actual SW injection ─────────────────────────────────────
    ph, po = run_pass(model, inp, blocks, arch, inject_layer=sl, inject_pos=pos,
                      inject_row=sr, epsilon=eps)
    sig = {li: dh(ref[li], ph[li]) for li in downstream}
    sig_kl = kl_divergence(logits_ref, _get_logits(po, arch), pos, causal)

    print(f"\n  {'L':>3}  {'noise ||dh||':>14}  {'signal ||dh||':>14}  {'SNR':>8}")
    rows = []
    for li in downstream:
        n = sum(noise[li]) / len(noise[li])
        s = sig[li]
        snr = s / n if n > 0 else float("inf")
        rows.append({"layer": li, "noise_dh": n, "signal_dh": s, "snr": snr})
        print(f"  {li:3d}  {n:14.4e}  {s:14.4e}  {snr:8.3f}")

    n_kl = sum(noise_kl) / len(noise_kl)
    kl_snr = sig_kl / n_kl if n_kl > 0 else float("inf")
    print(f"\n  KL noise floor (mean of {args.repeats}) = {n_kl:.4e}")
    print(f"  KL signal (SW injection)              = {sig_kl:.4e}")
    print(f"  KL SNR                                = {kl_snr:.3f}")

    deterministic = n_kl == 0.0 and all(r["noise_dh"] == 0.0 for r in rows)
    worst = min((r["snr"] for r in rows), default=float("inf"))
    if deterministic:
        verdict = "DETERMINISTIC — noise floor is exactly zero; T/C/KL are measurements."
    elif kl_snr < 2.0 or worst < 2.0:
        verdict = ("NOISE-DOMINATED — signal is not separable from run-to-run variation. "
                   "T/C/KL from this model are not interpretable.")
    else:
        verdict = ("NONDETERMINISTIC BUT SEPARABLE — report the noise floor alongside "
                   "every T/C/KL value.")
    print(f"\n  VERDICT: {verdict}")

    out = ROOT / "results" / f"impulse_determinism_{args.model}.json"
    out.write_text(json.dumps({
        "model": args.model, "dtype": str(next(model.parameters()).dtype),
        "sw_layer": sl, "sw_row": sr, "sw_pos": pos, "eps_ac": eps, "ac_meta": meta,
        "repeats": args.repeats, "per_layer": rows,
        "kl_noise_floor": n_kl, "kl_noise_samples": noise_kl, "kl_signal": sig_kl,
        "kl_snr": kl_snr, "deterministic": deterministic, "verdict": verdict,
    }, indent=2))
    print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
