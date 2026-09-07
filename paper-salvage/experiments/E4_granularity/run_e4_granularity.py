"""
E4 — c_{k,i} granularity decomposition across the genomic gated FFNs.

For each model's empirical SW row at its SW layer:
  top-1 share            max_i c / sum_i c
  participation ratio    (sum c)^2 / sum c^2  -- effective number of contributing units
  rank profile           the leading contributors

Weights only. No forward pass.

Shape verification is mandatory before any number is trusted: the canonical convention is
gate/up [d_ffn, d_model] and down [d_model, d_ffn], and a silent transpose produces a
plausible-looking ranking that is wrong. NTv3 and Evo1 are the two whose module paths were
never verified (the uk_frobenius adapter registry flags ntv3 as "verify"). If a shape check
fails the model is REPORTED AND SKIPPED, not guessed at.

Reports association only. Per PHASE_1_BLOCKING §E4: do not claim that a low participation
ratio *causes* ensemble behaviour.

Usage:
  python run_e4_granularity.py                 # all models
  python run_e4_granularity.py --model evo1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SALVAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SALVAGE / "src"))

from uk_frobenius import layer_report, row_granularity  # noqa: E402
from scripts.interpretability.run_sw_broadcast_impulse import (  # noqa: E402
    SW_TARGETS, _get_blocks, _load_model,
)


def mlp_tensors(model, arch: str, layer: int):
    """(W_gate, W_up, W_down) in canonical convention, or (None, reason)."""
    blk = _get_blocks(model, arch)[layer]
    if arch == "llama":
        m = blk.mlp
        return m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight
    if arch == "bert":
        m = blk.mlp
        packed = m.gated_layers.weight          # [2 * d_ffn, d_model]
        d_ffn = packed.shape[0] // 2
        return packed[:d_ffn], packed[d_ffn:], m.wo.weight
    if arch == "evo1":
        m = blk.mlp
        return m.l1.weight, m.l2.weight, m.l3.weight
    if arch == "ntv3":
        # NTv3's SelfAttentionBlock carries the FFN inline as fc1/fc2 with a SiLU, and fc1
        # is PACKED gate+up on adjacent row blocks -- the DNABERT-2 layout, not the Llama
        # one. Verified by shape at L11: fc1 (12288, 1536) = [2 * d_ffn, d_model] with
        # d_ffn = 6144, fc2 (1536, 6144) = [d_model, d_ffn]. There is no `mlp` module;
        # `blk.mlp` resolves to an unrelated bound method.
        #
        # NOTE: this contradicts the uk_frobenius ADAPTERS registry, which maps "ntv3" to
        # adapter_llama_swiglu (and flags it "verify -- confirm module path"). That entry is
        # wrong for this checkpoint; it would raise on model.model.layers.
        #
        # Which half is gate and which is up cannot be read off the shapes -- but it does
        # not matter here: c_{k,i} = W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2 is
        # symmetric under swapping the two, so every E4 quantity is invariant to the choice.
        if not (hasattr(blk, "fc1") and hasattr(blk, "fc2")):
            raise AttributeError(
                f"no fc1/fc2 on {type(blk).__name__}; "
                f"children={[n for n, _ in blk.named_children()]}")
        packed = blk.fc1.weight                  # [2 * d_ffn, d_model]
        down = blk.fc2.weight                    # [d_model, d_ffn]
        d_ffn = down.shape[1]
        if packed.shape[0] != 2 * d_ffn:
            raise AttributeError(
                f"fc1 is {tuple(packed.shape)}, expected [2*d_ffn={2*d_ffn}, d_model]; "
                "not a packed gate/up layout -- refusing to guess")
        return packed[:d_ffn], packed[d_ffn:], down
    raise ValueError(arch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    models = list(SW_TARGETS) if args.model == "all" else [args.model]
    out_rows = []

    for name in models:
        t = SW_TARGETS[name]
        layer, k, arch = t["sw_layer"], t["sw_row"], t["arch"]
        print(f"\n=== {name}  arch={arch}  SW layer {layer}, row {k} ===")
        try:
            model, _ = _load_model(name, args.device)
        except Exception as exc:
            print(f"    LOAD FAILED: {type(exc).__name__}: {exc}")
            out_rows.append({"model": name, "status": "LOAD_FAILED", "error": str(exc)})
            continue

        try:
            W_gate, W_up, W_down = mlp_tensors(model, arch, layer)
        except Exception as exc:
            print(f"    ADAPTER FAILED: {type(exc).__name__}: {exc}")
            out_rows.append({"model": name, "status": "ADAPTER_FAILED", "error": str(exc)})
            del model
            torch.cuda.empty_cache()
            continue

        print(f"    shapes: gate {tuple(W_gate.shape)}  up {tuple(W_up.shape)}  "
              f"down {tuple(W_down.shape)}")

        # ---- mandatory shape verification -------------------------------------
        ok = True
        reasons = []
        if W_gate.shape != W_up.shape:
            ok = False
            reasons.append(f"gate {tuple(W_gate.shape)} != up {tuple(W_up.shape)}")
        else:
            d_ffn, d_model = W_gate.shape
            if W_down.shape != (d_model, d_ffn):
                ok = False
                reasons.append(f"down is {tuple(W_down.shape)}, expected {(d_model, d_ffn)}")
            elif not (0 <= k < d_model):
                ok = False
                reasons.append(f"SW row {k} outside d_model={d_model}")
        if not ok:
            print(f"    SHAPE CHECK FAILED: {'; '.join(reasons)}")
            print("    -> reported and skipped, not guessed")
            out_rows.append({"model": name, "status": "SHAPE_CHECK_FAILED",
                             "reasons": reasons,
                             "shapes": {"gate": list(W_gate.shape), "up": list(W_up.shape),
                                        "down": list(W_down.shape)}})
            del model, W_gate, W_up, W_down
            torch.cuda.empty_cache()
            continue
        print(f"    shape check OK  (d_model={d_model}, d_ffn={d_ffn})")

        rep, contrib = layer_report(W_gate.cpu(), W_up.cpu(), W_down.cpu(),
                                    layer=layer, query_rows=[k])
        gran = row_granularity(contrib, k)
        qr = rep.query_ranks[k]
        order = torch.argsort(contrib[k], descending=True)[:10]
        total = float(contrib[k].sum())

        print(f"    row rank            {qr['rank']} / {rep.d_model}   "
              f"max/median {rep.max_over_median:.1f}")
        print(f"    top1_share          {gran.top1_share:.4f}   (unit {gran.top1_index})")
        print(f"    top5_share          {gran.top5_share:.4f}")
        print(f"    participation ratio {gran.participation_ratio:.2f} of {rep.d_ffn}")
        print(f"    regime              {gran.regime}")

        out_rows.append({
            "model": name, "status": "ok", "arch": arch, "sw_layer": layer, "sw_row": k,
            "d_model": rep.d_model, "d_ffn": rep.d_ffn,
            "row_rank": qr["rank"], "max_over_median": rep.max_over_median,
            "top1_index": gran.top1_index, "top1_share": gran.top1_share,
            "top5_share": gran.top5_share,
            "participation_ratio": gran.participation_ratio,
            "pr_fraction_of_dffn": gran.participation_ratio / rep.d_ffn,
            "regime": gran.regime,
            "top10_contributors": [
                {"i": int(idx), "share": float(contrib[k][idx] / total)} for idx in order],
        })
        del model, W_gate, W_up, W_down, contrib
        torch.cuda.empty_cache()

    # Merge, so per-model invocations accumulate instead of overwriting each other.
    out = ROOT / "results" / "e4_granularity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}
    if out.exists():
        for row in json.loads(out.read_text()).get("models", []):
            merged[row["model"]] = row
    for row in out_rows:
        merged[row["model"]] = row
    order = list(SW_TARGETS)
    out.write_text(json.dumps(
        {"note": "association only; a low PR is not claimed to cause "
                 "ensemble behaviour (PHASE_1_BLOCKING E4)",
         "models": [merged[m] for m in order if m in merged]}, indent=2))
    print(f"\n  saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
