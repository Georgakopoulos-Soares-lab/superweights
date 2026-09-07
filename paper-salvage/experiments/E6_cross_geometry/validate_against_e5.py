"""
Pre-lock regression check: reproduce E5's own six already-published f_cross values using
cross_geometry_lib.py's independent implementation. This uses E5's SIX DISCOVERY ROWS ONLY
-- it is a regression check on the new code, not a new measurement, and per
PREREG_cross_geometry_stageA.md's disclosure, these six rows do not count as E6 evidence and
are not part of the confirmatory panel. Weight-only. Run before locking the E6 prereg.

Expected f_cross (from experiments/E5_dimensionality/GATE0_RESULTS.md):
    Llama-7B        0.1929
    Mistral-7B      0.0670
    OLMo-7B         0.0926
    GENERator EUK   0.8291
    DNABERT-2       0.5074
    NTv3            0.2073

Usage:
  python experiments/E6_cross_geometry/validate_against_e5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

SALVAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E1_nlp_validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uk_frobenius import ADAPTERS  # noqa: E402
from run_e1_retrospective import fetch_layer_tensors  # noqa: E402
from cross_geometry_lib import pair_matrix, row_cross_geometry  # noqa: E402

EXPECTED = {
    "Llama-7B": 0.1928631494935161,
    "Mistral-7B": 0.06701384889903268,
    "OLMo-7B": 0.09261719663848049,
    "GENERator EUK": 0.8290834232540367,
    "DNABERT-2": 0.5074395677014457,
    "NTv3": 0.2073229606178682,
}

E5_ROWS = [
    dict(name="Llama-7B", loader="nlp_shard", repo="huggyllama/llama-7b", layer=2, row=3968),
    dict(name="Mistral-7B", loader="nlp_shard", repo="mistralai/Mistral-7B-v0.1", layer=1, row=2070),
    dict(name="OLMo-7B", loader="nlp_shard", repo="allenai/OLMo-7B-0724-hf", layer=1, row=269),
    dict(name="GENERator EUK", loader="generator", repo="GenerTeam/GENERator-v2-eukaryote-3b-base",
         layer=4, row=2371, revision="7dc01bccce5b65e15141170538afdc2ff09d8dde"),
    dict(name="DNABERT-2", loader="dnabert2", repo="zhihan1996/DNABERT-2-117M", layer=5, row=603,
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(name="NTv3", loader="ntv3", repo="InstaDeepAI/NTv3_650M_pre", layer=11, row=1472,
         code_revision="0ecff3637f0d3ba5b686d1095083218157c2ca34"),
]


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
    all_ok = True
    for spec in E5_ROWS:
        name = spec["name"]
        Wg, Wu, Wd = load_weights(spec)
        K = pair_matrix(Wg, Wu)
        geo = row_cross_geometry(K, Wd[spec["row"]])
        expected = EXPECTED[name]
        rel_err = abs(geo.f_cross - expected) / abs(expected)
        ok = rel_err < 1e-6
        all_ok &= ok
        print(f"{name:16s} f_cross={geo.f_cross:.10f}  expected={expected:.10f}  "
              f"rel_err={rel_err:.2e}  {'OK' if ok else 'MISMATCH'}")
        del Wg, Wu, Wd, K

    print()
    if all_ok:
        print("ALL SIX E5 REGRESSION VALUES REPRODUCED -- cross_geometry_lib.py is a correct, "
              "independent reimplementation of the same exact/diagonal quantities E5 measured.")
    else:
        raise SystemExit("REGRESSION FAILURE -- cross_geometry_lib.py disagrees with E5's "
                          "published numbers. Do not lock the E6 prereg until this is resolved.")


if __name__ == "__main__":
    main()
