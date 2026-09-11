"""
experiments/E7_exact_dimensionality/run_confirmatory_spectral_lite.py

Given a completed Phase-1 detection JSON (results/experiments/E7/e7_phase1_detection_{model}.json), fetches
ONLY the safetensors shard(s) holding the detected layer's mlp.{gate,up,down}_proj tensors
(weight-only, no full-model reload -- both Qwen2.5-7B and GenomeOcean-4B use the standard,
unpacked Llama/Mistral-style MLP naming, confirmed in MODEL_PANEL.md) and computes q1/PR_spec
on the exact detected (layer, row) coordinate.

Usage:
  python run_confirmatory_spectral_lite.py --model qwen25 --repo Qwen/Qwen2.5-7B
  python run_confirmatory_spectral_lite.py --model genomeocean --repo DOEJGI/GenomeOcean-4B
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE  # repo root; harness paths are written relative to it
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spectral_lib import row_spectral_metrics  # noqa: E402

PREREG = SALVAGE / "docs" / "prereg" / "PREREG_exact_operator_dimensionality.md"


def verify_lock() -> None:
    r = subprocess.run(
        [sys.executable, str(SALVAGE / "src" / "prereg_lock.py"), "verify", str(PREREG)],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0 or "OK" not in r.stdout:
        print(r.stderr, file=sys.stderr)
        raise SystemExit("prereg_lock verify FAILED.")


def fetch_layer_tensors(repo: str, layer: int):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    names = {
        "gate": f"model.layers.{layer}.mlp.gate_proj.weight",
        "up": f"model.layers.{layer}.mlp.up_proj.weight",
        "down": f"model.layers.{layer}.mlp.down_proj.weight",
    }
    idx_path = hf_hub_download(repo, "model.safetensors.index.json")
    weight_map = json.loads(Path(idx_path).read_text())["weight_map"]
    missing = [n for n in names.values() if n not in weight_map]
    if missing:
        raise RuntimeError(f"tensor names not in index: {missing}")
    shards = sorted({weight_map[n] for n in names.values()})
    print(f"  layer {layer}: shards needed: {shards}")
    loaded = {}
    for shard in shards:
        p = hf_hub_download(repo, shard)
        data = load_file(p)
        for key, tname in names.items():
            if tname in data:
                loaded[key] = data[tname]
        del data
    return loaded["gate"], loaded["up"], loaded["down"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--repo", required=True)
    args = p.parse_args()

    print("Verifying E7 prereg lock ...")
    verify_lock()

    det_path = ROOT / "results" / f"e7_phase1_detection_{args.model}.json"
    det = json.loads(det_path.read_text())
    if det["null_outcome"]:
        print(f"{args.model}: Phase-1 detection was NULL -- nothing to compute. "
              f"Reason: {det['null_reason']}")
        return

    layer = det["candidate"]["layer"]
    row = det["candidate"]["row"]
    print(f"{args.model}: candidate layer={layer} row={row} (from {det_path})")

    gate, up, down = fetch_layer_tensors(args.repo, layer)
    print(f"  gate{tuple(gate.shape)} up{tuple(up.shape)} down{tuple(down.shape)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics, sigmas = row_spectral_metrics(gate, up, down[row], device=device)
    print(f"  q1={metrics.q1:.6f} PR_spec={metrics.pr_spec:.4f} frob={metrics.frob_norm:.6g} "
          f"n_sv={metrics.n_singular_values}")

    det["spectral"] = dict(q1=metrics.q1, pr_spec=metrics.pr_spec,
                            stable_rank=metrics.stable_rank, frob_norm=metrics.frob_norm,
                            n_singular_values=metrics.n_singular_values)
    det_path.write_text(json.dumps(det, indent=2))
    print(f"updated {det_path}")


if __name__ == "__main__":
    main()
