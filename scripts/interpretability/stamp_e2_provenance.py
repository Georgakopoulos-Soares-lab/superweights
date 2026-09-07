"""
scripts/interpretability/stamp_e2_provenance.py
------------------------------------------------
Attach run provenance to the E2 broadcast results.

Additive only: it writes a `provenance` block per model entry and never touches a measured
value. Run AFTER the five-model run, so the harness is not modified mid-run (which would
make earlier and later models record different things).

Fields required by the E2 run spec that the harness does not itself record:
  exact checkpoint, dtype, attention implementation, seed, git commit, environment /
  container identity, prereg lock hash, and the model's measured noise floor.

The noise floor is read from results/impulse_determinism_<model>.json, produced by
scripts/interpretability/impulse_determinism_check.py.

Usage:
  python scripts/interpretability/stamp_e2_provenance.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

LOCK_SHA = "3d7515d0b7889f65038acd8479e85976248e7dbd0e5a701303de3a1b37dbfdc3"
LOCK_UTC = "2026-08-13T02:59:20+00:00"

CHECKPOINT = {
    "generator": "GenerTeam/GENERator-v2-eukaryote-3b-base",
    "generator_prokaryote": "GenerTeam/GENERator-v2-prokaryote-3b-base",
    "dnabert2": "zhihan1996/DNABERT-2-117M @ 7bce263b15377fc15361f52cfab88f8b586abda0",
    "ntv3": "InstaDeepAI/NTv3_650M_pre @ code_revision 0ecff3637f0d3ba5b686d1095083218157c2ca34",
    "evo1": "togethercomputer/evo-1-8k-base @ revision 1.1_fix",
}

ATTENTION = {
    "generator": "default (HF Llama SDPA)",
    "generator_prokaryote": "default (HF Llama SDPA)",
    "dnabert2": "EAGER PyTorch attention, Triton flash-attn disabled (D-014); "
                "avoids the kernel's fp16 cast of qkv/bias and its nondeterminism",
    "ntv3": "default",
    "evo1": "use_flash_attn=False set BEFORE StripedHyena(config); 3 of 32 blocks change "
            "kernel (attn_layer_idxs [8, 16, 24]), the other 29 are Hyena",
}

ENVIRONMENT = {
    "evo1": {
        "container": "/work/11034/atzanakak/ls6/containers/evo2.sif",
        "container_sha256":
            "ecb001192eeee416239da526ceaf9a73685fc9609aad404a23a36ac0bc161706",
        "apptainer": "tacc-apptainer/1.1.8",
        "note": "evo-model 0.5 and stripedhyena 0.2.2 resolve from ~/.local, OUTSIDE the "
                "image, so the image hash alone does not pin the environment",
    },
    "_default": {
        "conda_env": "/work/11034/atzanakak/work/11034/atzanakak/miniconda3/envs/grlm",
        "note": "LD_LIBRARY_PATH must be prepended with $ENV/lib or PIL fails on GLIBCXX",
    },
}


def git(*a: str) -> str | None:
    try:
        return subprocess.check_output(["git", *a], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def noise_floor(model: str) -> dict:
    p = ROOT / "results" / f"impulse_determinism_{model}.json"
    if not p.exists():
        return {"status": "NOT MEASURED", "path": str(p)}
    d = json.loads(p.read_text())
    return {
        "status": "measured",
        "deterministic": d.get("deterministic"),
        "kl_noise_floor": d.get("kl_noise_floor"),
        "kl_signal_at_primary": d.get("kl_signal"),
        "per_layer_noise_dh": {str(r["layer"]): r["noise_dh"] for r in d.get("per_layer", [])},
        "verdict": d.get("verdict"),
        "source": str(p.relative_to(ROOT)),
    }


def main() -> int:
    path = ROOT / "results" / "sw_broadcast_impulse.json"
    data = json.loads(path.read_text())
    commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))

    stamped = 0
    for key, entry in data.items():
        if not isinstance(entry, dict) or "sw_metrics" not in entry:
            continue
        if key.endswith("_SUPERSEDED"):
            continue
        base = entry.get("model", key.split("__")[0])
        entry["provenance"] = {
            "prereg_lock_sha256": LOCK_SHA,
            "prereg_lock_utc": LOCK_UTC,
            "git_commit": commit,
            "git_dirty": dirty,
            "seed": 42,
            "n_controls": 5,
            "checkpoint": CHECKPOINT.get(base, "UNKNOWN"),
            "param_dtype": entry.get("precision", {}).get("param_dtype"),
            "attention_implementation": ATTENTION.get(base, "UNKNOWN"),
            "environment": ENVIRONMENT.get(base, ENVIRONMENT["_default"]),
            "noise_floor": noise_floor(base),
            "decisions": ["D-011", "D-013", "D-014", "D-015"],
            "null_rule": "a result is a null only if headroom >= 4x at that layer AND the "
                         "model's noise floor is below the observed signal",
        }
        stamped += 1
        print(f"  stamped {key}")

    path.write_text(json.dumps(data, indent=2))
    print(f"\n  {stamped} entries stamped -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
