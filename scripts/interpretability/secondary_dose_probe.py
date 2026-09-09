"""
scripts/interpretability/secondary_dose_probe.py
-------------------------------------------------
STEP 3d — does the secondary (AC-matched, alpha = 1.0) dose produce a measurable output KL?

At the primary dose (alpha = 0.01) output KL is ~0 for every model: GENERator EUK and PROK
give exactly 0.0, eager DNABERT-2 gives 1.8e-8. If KL is also ~0 at alpha = 1.0 then KL is
the wrong functional metric for these models at any small-signal dose, and the answer is a
different metric — NOT a larger perturbation.

Reports KL at both doses, plus two candidate replacement metrics measured alongside so the
proposal (if needed) is grounded rather than speculative:
  - KL restricted to the SW token position, instead of the sequence mean
  - top-k rank displacement at the SW token

Reports only. Does not choose a metric, does not escalate the dose.

Usage:
  python scripts/interpretability/secondary_dose_probe.py --model generator_prokaryote
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
    ALPHA, ALPHA_KL, SW_TARGETS, _get_blocks, _get_logits, _load_model, _tokenize,
    ac_epsilon, kl_divergence, run_pass,
)


def kl_at_position(lc: torch.Tensor, lp: torch.Tensor, pos: int) -> float:
    p = torch.softmax(lc[0, pos].float(), dim=-1)
    q = torch.softmax(lp[0, pos].float(), dim=-1)
    return float((p * ((p + 1e-12).log() - (q + 1e-12).log())).sum())


def topk_displacement(lc: torch.Tensor, lp: torch.Tensor, pos: int, k: int = 10) -> dict:
    """How far the top-k tokens move in rank when the perturbation is applied."""
    c = lc[0, pos].float()
    p = lp[0, pos].float()
    top_c = torch.topk(c, k).indices
    rank_p = torch.argsort(torch.argsort(p, descending=True))
    rank_c = torch.argsort(torch.argsort(c, descending=True))
    disp = (rank_p[top_c].float() - rank_c[top_c].float()).abs()
    return {
        "mean_abs_rank_shift": float(disp.mean()),
        "max_abs_rank_shift": float(disp.max()),
        "top1_changed": bool(int(torch.argmax(c)) != int(torch.argmax(p))),
        "topk_set_overlap": float(
            len(set(top_c.tolist()) & set(torch.topk(p, k).indices.tolist())) / k),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="generator_prokaryote", choices=list(SW_TARGETS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t = SW_TARGETS[args.model]
    sl, sr, arch, causal = t["sw_layer"], t["sw_row"], t["arch"], t["causal"]

    from src.dna_probes import get_probe
    model, tok = _load_model(args.model, args.device)
    blocks = _get_blocks(model, arch)
    inp = _tokenize(tok, get_probe(t["probe"]), arch, args.device)

    clean, cout = run_pass(model, inp, blocks, arch)
    lc = _get_logits(cout, arch)
    src = clean[sl]
    pos = int(src[:, sr].abs().argmax().item())
    kl_pos = min(pos + 1, lc.shape[1] - 1) if causal else pos

    print(f"[{args.model}] sw_layer={sl} sw_row={sr} sw_pos={pos} "
          f"dtype={next(model.parameters()).dtype}")

    rows = []
    for alpha, label in [(ALPHA, "primary"), (ALPHA_KL, "secondary")]:
        eps, meta = ac_epsilon(src, alpha)
        ph, po = run_pass(model, inp, blocks, arch, inject_layer=sl, inject_pos=pos,
                          inject_row=sr, epsilon=eps)
        lp = _get_logits(po, arch)
        kl_seq = kl_divergence(lc, lp, pos, causal)
        kl_tok = kl_at_position(lc, lp, kl_pos)
        disp = topk_displacement(lc, lp, kl_pos)
        logit_delta = float((lp[0, kl_pos].float() - lc[0, kl_pos].float()).abs().max())
        rows.append({
            "alpha": alpha, "label": label, "epsilon": eps, "std_ac": meta["std_ac"],
            "kl_protocol": kl_seq, "kl_at_sw_token": kl_tok,
            "max_abs_logit_delta": logit_delta, **disp,
        })
        print(f"\n  alpha={alpha:<5g} ({label})  eps={eps:.4e}")
        print(f"    KL (protocol defn)        = {kl_seq:.6e}")
        print(f"    KL @ SW token position    = {kl_tok:.6e}")
        print(f"    max |Δ logit| @ that pos  = {logit_delta:.6e}")
        print(f"    top-10 mean rank shift    = {disp['mean_abs_rank_shift']:.3f}  "
              f"max {disp['max_abs_rank_shift']:.0f}")
        print(f"    top-1 changed             = {disp['top1_changed']}   "
              f"top-10 overlap = {disp['topk_set_overlap']:.2f}")

    sec = rows[-1]
    measurable = sec["kl_protocol"] > 1e-6
    verdict = ("KL MEASURABLE at alpha=1.0 — dual dose stands, lock as specified"
               if measurable else
               "KL ~0 AT alpha=1.0 TOO — KL is the wrong functional metric at any "
               "small-signal dose. Do NOT escalate. Report and stop.")
    print(f"\n  VERDICT: {verdict}")

    out = ROOT / "results" / f"secondary_dose_probe_{args.model}.json"
    out.write_text(json.dumps({"model": args.model, "sw_pos": pos, "kl_pos": kl_pos,
                               "doses": rows, "kl_measurable_at_secondary": measurable,
                               "verdict": verdict}, indent=2))
    print(f"  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
