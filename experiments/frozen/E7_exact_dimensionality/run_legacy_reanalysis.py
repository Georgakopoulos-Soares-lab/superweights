"""
experiments/E7_exact_dimensionality/run_legacy_reanalysis.py

Phase 6 -- ONLY run after the Phase-5 new-model confirmatory decision is frozen (per
PREREG_exact_operator_dimensionality.md's binding ordering rule: legacy spectra must not be
inspected before the new-model branch is recorded). Applies the identical, unmodified
spectral_lib.py code to E5's six discovery rows (Llama-7B, Mistral-7B, OLMo-7B, GENERator
EUK, DNABERT-2, NTv3). Supporting/generalization data only -- never re-enters the Phase-5
primary group decision.

Weight-only. Reuses the exact loaders already validated in E5/E6 (fetch_layer_tensors for
NLP, ADAPTERS for genomic).

Usage:
  python run_legacy_reanalysis.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE.parent
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E1_nlp_validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uk_frobenius import ADAPTERS  # noqa: E402
from run_e1_retrospective import fetch_layer_tensors  # noqa: E402
from spectral_lib import row_spectral_metrics  # noqa: E402

PREREG = SALVAGE / "docs" / "prereg" / "PREREG_exact_operator_dimensionality.md"

LEGACY_PANEL = [
    dict(name="Llama-7B", group="nlp", loader="nlp_shard", repo="huggyllama/llama-7b",
         layer=2, row=3968),
    dict(name="Mistral-7B", group="nlp", loader="nlp_shard", repo="mistralai/Mistral-7B-v0.1",
         layer=1, row=2070),
    dict(name="OLMo-7B", group="nlp", loader="nlp_shard", repo="allenai/OLMo-7B-0724-hf",
         layer=1, row=269),
    dict(name="GENERator EUK", group="genomic", loader="generator",
         repo="GenerTeam/GENERator-v2-eukaryote-3b-base", layer=4, row=2371,
         revision="7dc01bccce5b65e15141170538afdc2ff09d8dde"),
    dict(name="DNABERT-2", group="genomic", loader="dnabert2", repo="zhihan1996/DNABERT-2-117M",
         layer=5, row=603, revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(name="NTv3", group="genomic", loader="ntv3", repo="InstaDeepAI/NTv3_650M_pre",
         layer=11, row=1472, code_revision="0ecff3637f0d3ba5b686d1095083218157c2ca34"),
]

# f_cross and diagonal PR already published in E5/E6 for exact cross-reference
LEGACY_PUBLISHED = {
    "Llama-7B": dict(f_cross=0.1928631494935161, diag_pr=1.2442913096024062),
    "Mistral-7B": dict(f_cross=0.06701384889903268, diag_pr=1.0235319953623476),
    "OLMo-7B": dict(f_cross=0.09261719663848049, diag_pr=1.0944),
    "GENERator EUK": dict(f_cross=0.8290834232540367, diag_pr=4.559937375295324),
    "DNABERT-2": dict(f_cross=0.5074395677014457, diag_pr=3.642180751507628),
    "NTv3": dict(f_cross=0.2073229606178682, diag_pr=22.71939144675925),
}


def verify_lock() -> None:
    r = subprocess.run(
        [sys.executable, str(SALVAGE / "src" / "prereg_lock.py"), "verify", str(PREREG)],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0 or "OK" not in r.stdout:
        print(r.stderr, file=sys.stderr)
        raise SystemExit("prereg_lock verify FAILED.")


def load_weights(spec: dict):
    loader = spec["loader"]
    if loader == "nlp_shard":
        t = fetch_layer_tensors(spec["repo"], spec["layer"])
        return t["gate"], t["up"], t["down"]
    if loader == "generator":
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(spec["repo"], revision=spec["revision"],
                                                   trust_remote_code=True, torch_dtype=torch.float32)
        g, u, d = ADAPTERS["llama_swiglu"](m, spec["layer"])
        g, u, d = g.detach().clone(), u.detach().clone(), d.detach().clone()
        del m
        return g, u, d
    if loader == "dnabert2":
        from transformers import AutoModelForMaskedLM
        m = AutoModelForMaskedLM.from_pretrained(spec["repo"], revision=spec["revision"],
                                                    trust_remote_code=True, torch_dtype=torch.float32)
        g, u, d = ADAPTERS["dnabert2"](m, spec["layer"])
        g, u, d = g.detach().clone(), u.detach().clone(), d.detach().clone()
        del m
        return g, u, d
    if loader == "ntv3":
        from transformers import AutoModelForMaskedLM
        m = AutoModelForMaskedLM.from_pretrained(spec["repo"], trust_remote_code=True,
                                                    code_revision=spec["code_revision"],
                                                    torch_dtype=torch.float32)
        g, u, d = ADAPTERS["ntv3"](m, spec["layer"])
        g, u, d = g.detach().clone(), u.detach().clone(), d.detach().clone()
        del m
        return g, u, d
    raise ValueError(loader)


def main() -> None:
    print("Verifying E7 prereg lock ...")
    verify_lock()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for spec in LEGACY_PANEL:
        name = spec["name"]
        print(f"=== {name} L{spec['layer']} row {spec['row']} ===")
        Wg, Wu, Wd = load_weights(spec)
        metrics, sigmas = row_spectral_metrics(Wg, Wu, Wd[spec["row"]], device=device)
        pub = LEGACY_PUBLISHED[name]
        print(f"  q1={metrics.q1:.6f} PR_spec={metrics.pr_spec:.4f} "
              f"(vs. published f_cross={pub['f_cross']:.4f}, diagonal PR={pub['diag_pr']:.4f})")
        results[name] = dict(
            group=spec["group"], layer=spec["layer"], row=spec["row"],
            q1=metrics.q1, pr_spec=metrics.pr_spec, stable_rank=metrics.stable_rank,
            frob_norm=metrics.frob_norm, n_singular_values=metrics.n_singular_values,
            published_f_cross=pub["f_cross"], published_diagonal_pr=pub["diag_pr"],
        )
        del Wg, Wu, Wd

    out_path = ROOT / "results" / "e7_legacy_reanalysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
