"""
experiments/E7_exact_dimensionality/run_phase1_detection_evo2.py

Phase 1 frozen prospective detection for Evo 2 7B (PREREG_exact_operator_dimensionality.md).
Must run inside evo2.sif with --cleanenv --env PYTHONNOUSERSITE=1 (see MODEL_PANEL.md --
the host-side ~/.local transformer_engine is ABI-incompatible with the container's own).

  /opt/apps/tacc-apptainer/1.1.8/bin/apptainer exec --nv --cleanenv \
    --bind /work,/tmp --env PYTHONNOUSERSITE=1 --env HF_HOME=... --env HF_HUB_CACHE=... \
    /work/11034/atzanakak/ls6/containers/evo2.sif \
    python3 experiments/E7_exact_dimensionality/run_phase1_detection_evo2.py

Same statistic/threshold/tie-break/null rule as run_phase1_detection.py's genomic branch --
duplicated rather than imported because this script runs in a completely separate Python
environment (container site-packages only, no repo conda env).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch

SALVAGE = Path("/work/11034/atzanakak/glm_super_weight/genomic-super-weights/manuscript")
ROOT = SALVAGE  # repo root; harness paths are written relative to it
PREREG = SALVAGE / "docs" / "prereg" / "PREREG_exact_operator_dimensionality.md"
RATIO_THRESHOLD = 5.0
sys.path.insert(0, str(SALVAGE / "experiments" / "E7_exact_dimensionality"))

_ACTB_CDS = (
    "ATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAAGCCGGCTTCGCGGGCGACGATGCCCCGAGGGCC"
    "GTCTTCCCCTCCATCGTGGGGCGCCCCAGGCACCAGGGCGTGATGGTGGGCATGGGTCAGAAGGATTCCTATGTGGGCGACGAGGCC"
    "CAGAGCAAGAGAGGCATCCTCACCCTGAAGTACCCCATCGAGCACGGCATCGTCACCAACTGGGACGACATGGAGAAAATCTGGCAC"
    "CACACCTTCTACAATGAGCTGCGTGTGGCTCCCGAGGAGCACCCCGTGCTGCTCACCGAGGCCCCCCTGAACCCGAAGGCCAACCGC"
    "GAGAAGATGACCCAGATCATGTTTGAGACCTTCAATACCCCCGCCATGTACGTTGCTATCCAGGCTGTGCTATCCCTGTACGCCTCT"
    "GGCCGTACCACTGGCATCGTGATGGACTCCGGTGACGGGGTCACCCACACTGTGCCCATCTACGAGGGGTATGCCCTCCCCCATGCC"
    "ATCCTGCGTCTGGACCTGGCTGGCCGGGACCTGACTGACTACCTCATGAAGATCCTCACCGAGCGCGGCTACAGCTTCACCACCACG"
    "GCCGAGCGGGAAATCGTGCGTGACATTAAGGAGAAGCTGTGCTACGTCGCCCTGGACTTCGAGCAAGAGATGGCCACGGCTGCTTCC"
    "AGCTCCTCCCTGGAGAAGAGCTACGAGCTGCCTGACGGCCAGGTCATCACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTC"
    "TTCCAGCCTTCCTTCCTGGGCATGGAGTCCTGTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGC"
    "AAAGACCTGTACGCCAACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCATTGCCGACAGGATGCAGAAGGAGATCACTGCC"
    "CTGGCACCCAGCACAATGAAGATCAAGATCATTGCTCCTCCTGAGCGCAAGTACTCCGTGTGGATCGGCGGCTCCATCCTGGCCTCG"
    "CTGTCCACCTTCCAGCAGATGTGGATCAGCAAGCAGGAGTATGACGAGTCCGGCCCCTCCATCGTCCACCGCAAATGCTTCTAA"
)
assert len(_ACTB_CDS) == 1128
_ACTB_500 = _ACTB_CDS[:504]
assert len(_ACTB_500) == 504


def verify_lock() -> None:
    r = subprocess.run(
        [sys.executable, str(SALVAGE / "src" / "prereg_lock.py"), "verify", str(PREREG)],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0 or "OK" not in r.stdout:
        print(r.stderr, file=sys.stderr)
        raise SystemExit("prereg_lock verify FAILED -- refusing to run any forward pass.")


class DownProjRecorder:
    def __init__(self):
        self.out_max = {}
        self.out_channel = {}
        self.median_channel_max = {}
        self._handles = []

    def register(self, modules: dict):
        for layer_idx, module in modules.items():
            self._handles.append(module.register_forward_hook(self._hook(layer_idx)))

    def _hook(self, layer_idx):
        def fn(module, args, output):
            y = output[0] if isinstance(output, tuple) else output
            flat = y.reshape(-1, y.shape[-1]).abs().float()
            channel_max = flat.max(dim=0).values
            self.out_max[layer_idx] = float(channel_max.max())
            self.out_channel[layer_idx] = int(channel_max.argmax())
            self.median_channel_max[layer_idx] = float(channel_max.median())
        return fn

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def main() -> None:
    print("Verifying E7 prereg lock before any forward pass ...")
    verify_lock()

    # A100 is compute capability 8.0; FP8 requires 8.9+. Monkeypatch transformer_engine's
    # fp8_autocast to a no-op before loading -- identical fix already established in this
    # repo's own models/evo2_wrapper.py.
    try:
        from contextlib import contextmanager
        import transformer_engine.pytorch as te

        @contextmanager
        def _noop_fp8(*args, **kwargs):
            yield

        te.fp8_autocast = _noop_fp8
    except ImportError:
        pass

    from evo2 import Evo2
    t0 = time.time()
    print("loading arcinstitute/evo2_7b ...")
    evo = Evo2("evo2_7b")
    model = evo.model
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Matches models/evo1_wrapper.py's established tokenization convention exactly: no BOS
    # prepended, encode-or-tokenize fallback, raw sequence passed straight through.
    tok = evo.tokenizer
    encode = getattr(tok, "encode", None) or getattr(tok, "tokenize", None)
    tokens = encode(_ACTB_500)
    device = next(model.parameters()).device
    if isinstance(tokens, torch.Tensor):
        input_ids = tokens.view(1, -1).long().to(device)
    else:
        input_ids = torch.tensor([list(tokens)], dtype=torch.long, device=device)
    print(f"  input token count: {input_ids.shape[1]}")

    blocks = model.blocks
    n_layers = len(blocks)
    modules = {i: blocks[i].mlp.l3 for i in range(n_layers)}
    rec = DownProjRecorder()
    rec.register(modules)
    with torch.no_grad():
        model(input_ids)
    rec.remove()

    per_layer = []
    for i in range(n_layers):
        if i not in rec.out_max:
            continue
        med = rec.median_channel_max[i]
        ratio = rec.out_max[i] / med if med > 0 else float("inf")
        per_layer.append(dict(layer=i, out_max=rec.out_max[i], out_channel=rec.out_channel[i],
                               median_channel_max=med, ratio=ratio))

    best = max(per_layer, key=lambda r: r["out_max"])
    passes = best["ratio"] >= RATIO_THRESHOLD

    result = dict(
        model="Evo2-7B", group="genomic", repo="arcinstitute/evo2_7b", resolved_revision=None,
        ratio_threshold=RATIO_THRESHOLD,
        candidate=dict(layer=best["layer"], row=best["out_channel"], out_max=best["out_max"],
                       ratio=best["ratio"]) if passes else None,
        null_outcome=(not passes),
        null_reason=None if passes else f"global-max ratio {best['ratio']:.2f} < {RATIO_THRESHOLD}",
        per_layer=per_layer,
    )
    print(f"\nEvo2-7B: candidate layer={best['layer']} row={best['out_channel']} "
          f"out_max={best['out_max']:.4g} ratio={best['ratio']:.2f}  "
          f"{'ACCEPTED' if passes else 'NULL (below threshold)'}")

    if passes:
        from spectral_lib import row_spectral_metrics
        layer, row = best["layer"], best["out_channel"]
        blk = blocks[layer].mlp
        Wg, Wu, Wd = blk.l1.weight.detach().cpu(), blk.l2.weight.detach().cpu(), blk.l3.weight.detach().cpu()
        print(f"  spectral: gate{tuple(Wg.shape)} up{tuple(Wu.shape)} down{tuple(Wd.shape)}")
        metrics, sigmas = row_spectral_metrics(Wg, Wu, Wd[row], device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"  q1={metrics.q1:.6f} PR_spec={metrics.pr_spec:.4f} frob={metrics.frob_norm:.6g} "
              f"n_sv={metrics.n_singular_values}")
        result["spectral"] = dict(q1=metrics.q1, pr_spec=metrics.pr_spec,
                                   stable_rank=metrics.stable_rank, frob_norm=metrics.frob_norm,
                                   n_singular_values=metrics.n_singular_values)

    out_path = ROOT / "results" / "e7_phase1_detection_evo2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
