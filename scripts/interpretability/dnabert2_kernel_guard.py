"""
scripts/interpretability/dnabert2_kernel_guard.py
--------------------------------------------------
Guard for the DNABERT-2 determinism fix (STEP 3a).

Forcing the eager PyTorch attention path changes the model's computation: a different
kernel, and no silent fp16 round-trip (the Triton path casts qkv to fp16 because the kernel
supports only fp16/bf16, then casts back). A fix that alters the model is not a fix.

Requirement: masked-LM perplexity under the new kernel must match the original to within
1%. If it does not, report that the fix altered the model rather than adopting it.

Both arms are run in the SAME process on the SAME inputs, differing only in the kernel, so
any perplexity gap is attributable to the kernel alone.

Usage:
  python scripts/interpretability/dnabert2_kernel_guard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

TOL_PCT = 1.0          # the stated guard
MASK_FRAC = 0.15
SEED = 0


def _bert_layers_module():
    import sys as _sys
    for name, mod in list(_sys.modules.items()):
        if name.endswith("bert_layers") and hasattr(mod, "flash_attn_qkvpacked_func"):
            return mod
    return None


def mlm_perplexity(model, tok, seqs: list[str], device: str) -> tuple[float, int]:
    """Mean masked-LM perplexity over a fixed, seeded masking pattern."""
    g = torch.Generator().manual_seed(SEED)
    total_nll, total_n = 0.0, 0
    mask_id = tok.mask_token_id
    for s in seqs:
        enc = tok(s, return_tensors="pt", truncation=True, max_length=512)
        ids = enc["input_ids"].to(device)
        inp = {k: v.to(device) for k, v in enc.items()}

        # deterministic mask over non-special positions
        special = torch.tensor(
            tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True),
            device=device, dtype=torch.bool)
        cand = (~special).nonzero(as_tuple=True)[0]
        k = max(1, int(MASK_FRAC * cand.numel()))
        pick = cand[torch.randperm(cand.numel(), generator=g)[:k].to(device)]

        labels = ids.clone()
        masked = ids.clone()
        masked[0, pick] = mask_id
        inp["input_ids"] = masked

        with torch.no_grad():
            out = model(**inp)
        logits = out.logits if hasattr(out, "logits") else out[0]
        lp = torch.log_softmax(logits[0, pick].float(), dim=-1)
        nll = -lp[torch.arange(pick.numel(), device=device), labels[0, pick]]
        total_nll += float(nll.sum())
        total_n += pick.numel()
    return float(torch.exp(torch.tensor(total_nll / total_n))), total_n


def main() -> int:
    from probes.dna_probes import PROBES
    from scripts.interpretability.run_sw_broadcast_impulse import (
        _dnabert2_force_eager_attention, _load_model,
    )

    device = "cuda"
    seqs = [v[:504] for v in PROBES.values()]
    print(f"guard: {len(seqs)} sequences, {int(MASK_FRAC*100)}% masked, seed {SEED}")

    # _load_model already forces eager; load first, then restore the Triton kernel to get
    # the ORIGINAL arm, so both arms share one model instance and one set of weights.
    model, tok = _load_model("dnabert2", device)
    mod = _bert_layers_module()
    if mod is None:
        print("FAIL: could not locate bert_layers module")
        return 2

    import importlib
    orig_fn = None
    try:
        fa = importlib.import_module(mod.__name__.rsplit(".", 1)[0] + ".flash_attn_triton")
        orig_fn = fa.flash_attn_qkvpacked_func
    except Exception as e:
        print(f"  note: could not re-import Triton kernel ({e})")

    results = {}

    # The Triton path is nondeterministic, so a single PPL from it is not a fixed quantity.
    # Repeat it to establish its own spread before comparing anything to it.
    if orig_fn is not None:
        mod.flash_attn_qkvpacked_func = orig_fn
        reps = [mlm_perplexity(model, tok, seqs, device)[0] for _ in range(3)]
        ppl_orig = sum(reps) / len(reps)
        results["triton_fp16_repeats"] = reps
        results["triton_fp16"] = ppl_orig
        spread = (max(reps) - min(reps)) / ppl_orig * 100.0
        results["triton_spread_pct"] = spread
        print(f"  ORIGINAL (Triton, fp16 attn): PPL = {ppl_orig:.4f}   "
              f"repeats = {[round(r, 4) for r in reps]}   spread = {spread:.3f}%")
    else:
        ppl_orig = None
        print("  ORIGINAL arm unavailable")

    mod.flash_attn_qkvpacked_func = None
    reps_e = [mlm_perplexity(model, tok, seqs, device)[0] for _ in range(3)]
    ppl_eager = sum(reps_e) / len(reps_e)
    results["eager_fp32_repeats"] = reps_e
    results["eager_fp32"] = ppl_eager
    spread_e = (max(reps_e) - min(reps_e)) / ppl_eager * 100.0
    results["eager_spread_pct"] = spread_e
    print(f"  FIXED    (eager, fp32 attn):  PPL = {ppl_eager:.4f}   "
          f"repeats = {[round(r, 4) for r in reps_e]}   spread = {spread_e:.3f}%")

    if ppl_orig is None:
        print("\n  VERDICT: INCONCLUSIVE — no original arm to compare against")
        verdict, passed = "inconclusive", False
    else:
        delta = 100.0 * (ppl_eager - ppl_orig) / ppl_orig
        results["delta_pct"] = delta
        print(f"\n  Δ = {delta:+.4f}%   (tolerance ±{TOL_PCT}%)")
        passed = abs(delta) <= TOL_PCT
        verdict = ("PASS — kernel change does not alter the model beyond tolerance"
                   if passed else
                   "FAIL — the fix altered the model; report this instead of adopting it")
        print(f"  VERDICT: {verdict}")

    # determinism of the original arm, for the record
    results["verdict"] = verdict
    results["passed"] = bool(passed)
    results["tolerance_pct"] = TOL_PCT
    out = ROOT / "results" / "dnabert2_kernel_guard.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"  saved -> {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
