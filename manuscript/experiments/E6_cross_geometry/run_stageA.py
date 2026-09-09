"""
experiments/E6_cross_geometry/run_stageA.py

Confirmatory E6 Stage-A run. Verifies the prereg lock BEFORE loading any confirmation-panel
weight. Weight-only: no forward pass, no activation collection, no ablation.

For each of the 13 frozen panel candidates (see CONFIRMATION_PANEL.md /
PREREG_cross_geometry_stageA.md), computes:
  - S_exact, S_diag, f_cross (signed)
  - x_pos, x_neg, x_pos_norm, x_neg_norm, gross_cross_norm, kappa
  - candidate's percentile + robust-z within its own layer's full f_cross distribution
  - exact row rank (secondary, near-free byproduct)

Candidates sharing a (model, layer) key reuse the same loaded weights and the same
pair_matrix / layer_fcross_distribution computation -- no weight is loaded twice.

Applies the Step 1/2/3 mechanical decision tree from the locked prereg and assigns a branch.

Writes: genomic-super-weights/results/e6_stageA.json

Usage:
  python experiments/E6_cross_geometry/run_stageA.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE.parent
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E1_nlp_validation"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E5_dimensionality"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uk_frobenius import ADAPTERS  # noqa: E402
from run_e1_retrospective import fetch_layer_tensors  # noqa: E402
from dimensionality_lib import rank_of, exact_ranksum_pvalue  # noqa: E402  (E5 reuse)
from cross_geometry_lib import (  # noqa: E402
    pair_matrix, row_cross_geometry, layer_fcross_distribution, candidate_layer_position,
)

PREREG = SALVAGE / "docs" / "prereg" / "PREREG_cross_geometry_stageA.md"

PANEL = [
    dict(idx=0, group="nlp", model="OLMo-7B", loader="nlp_shard",
         repo="allenai/OLMo-7B-0724-hf", layer=2, row=269),
    dict(idx=1, group="nlp", model="OLMo-7B", loader="nlp_shard",
         repo="allenai/OLMo-7B-0724-hf", layer=7, row=269),
    dict(idx=2, group="nlp", model="OLMo-7B", loader="nlp_shard",
         repo="allenai/OLMo-7B-0724-hf", layer=24, row=269),
    dict(idx=3, group="genomic", model="DNABERT-2", loader="dnabert2",
         repo="zhihan1996/DNABERT-2-117M", layer=3, row=86,
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(idx=4, group="genomic", model="DNABERT-2", loader="dnabert2",
         repo="zhihan1996/DNABERT-2-117M", layer=3, row=399,
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(idx=5, group="genomic", model="DNABERT-2", loader="dnabert2",
         repo="zhihan1996/DNABERT-2-117M", layer=3, row=603,
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(idx=6, group="genomic", model="DNABERT-2", loader="dnabert2",
         repo="zhihan1996/DNABERT-2-117M", layer=3, row=641,
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(idx=7, group="genomic", model="DNABERT-2", loader="dnabert2",
         repo="zhihan1996/DNABERT-2-117M", layer=5, row=86,
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(idx=8, group="genomic", model="DNABERT-2", loader="dnabert2",
         repo="zhihan1996/DNABERT-2-117M", layer=6, row=603,
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(idx=9, group="genomic", model="DNABERT-2", loader="dnabert2",
         repo="zhihan1996/DNABERT-2-117M", layer=7, row=603,
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(idx=10, group="genomic", model="DNABERT-2", loader="dnabert2",
         repo="zhihan1996/DNABERT-2-117M", layer=9, row=264,
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(idx=11, group="genomic", model="DNABERT-2", loader="dnabert2",
         repo="zhihan1996/DNABERT-2-117M", layer=9, row=294,
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(idx=12, group="genomic", model="GENERator EUK", loader="generator",
         repo="GenerTeam/GENERator-v2-eukaryote-3b-base", layer=4, row=1522,
         revision="7dc01bccce5b65e15141170538afdc2ff09d8dde"),
]


def verify_lock() -> None:
    r = subprocess.run(
        [sys.executable, str(SALVAGE / "src" / "prereg_lock.py"), "verify", str(PREREG)],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0 or "OK" not in r.stdout:
        print(r.stderr, file=sys.stderr)
        raise SystemExit("prereg_lock verify FAILED -- refusing to load real weights.")


def load_weights(spec: dict):
    loader = spec["loader"]
    if loader == "nlp_shard":
        t = fetch_layer_tensors(spec["repo"], spec["layer"])
        if t is None:
            raise RuntimeError(f"{spec['model']} L{spec['layer']}: shard fetch failed")
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
    raise ValueError(loader)


def geo_dict(geo) -> dict:
    return dict(s_exact=geo.s_exact, s_diag=geo.s_diag, f_cross=geo.f_cross,
                x_pos=geo.x_pos, x_neg=geo.x_neg, x_pos_norm=geo.x_pos_norm,
                x_neg_norm=geo.x_neg_norm, gross_cross_norm=geo.gross_cross_norm,
                kappa=geo.kappa, d_ffn=geo.d_ffn)


def main() -> None:
    print("Verifying E6 Stage-A prereg lock before touching any confirmation-panel weight ...")
    verify_lock()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    # group candidates by (model, layer) so shared weights load / K-matrix build only once
    groups = defaultdict(list)
    for spec in PANEL:
        groups[(spec["model"], spec["layer"])].append(spec)

    results = {}
    for (model, layer), specs in groups.items():
        loader_spec = specs[0]
        print(f"=== {model} L{layer} ({len(specs)} candidate row(s): "
              f"{[s['row'] for s in specs]}) ===")
        t0 = time.time()
        Wg, Wu, Wd = load_weights(loader_spec)
        d_ffn, d_model = Wg.shape
        print(f"  loaded in {time.time()-t0:.1f}s  gate{tuple(Wg.shape)} up{tuple(Wu.shape)} down{tuple(Wd.shape)}")

        t0 = time.time()
        K = pair_matrix(Wg, Wu)
        print(f"  pair_matrix built in {time.time()-t0:.1f}s")

        t0 = time.time()
        dist = layer_fcross_distribution(Wg, Wu, Wd, device=device)
        print(f"  layer_fcross_distribution ({d_model} rows) in {time.time()-t0:.1f}s  "
              f"median={dist.median:.4f} iqr={dist.iqr:.4f}")

        for spec in specs:
            row = spec["row"]
            geo = row_cross_geometry(K, Wd[row])
            pos = candidate_layer_position(geo.f_cross, dist)
            exact_rank = rank_of(torch.from_numpy(dist.s_exact), row)
            print(f"  row {row}: f_cross={geo.f_cross:.4f}  kappa="
                  f"{geo.kappa if isinstance(geo.kappa, str) else f'{geo.kappa:.4f}'}  "
                  f"layer_pct={pos['percentile']:.1f}  exact_rank={exact_rank}/{d_model}")

            results[f"{model}_L{layer}_r{row}"] = dict(
                group=spec["group"], model=model, repo=spec["repo"], layer=layer, row=row,
                d_model=d_model, d_ffn=d_ffn,
                geometry=geo_dict(geo),
                layer_position=dict(percentile=pos["percentile"], robust_z=pos["robust_z"],
                                     layer_median=pos["layer_median"], layer_iqr=pos["layer_iqr"],
                                     d_model=pos["d_model"]),
                exact_rank=exact_rank,
            )
        del Wg, Wu, Wd, K, dist
        if device == "cuda":
            torch.cuda.empty_cache()
        print()

    # ---- mechanical decision tree ----
    nlp_fcross = [v["geometry"]["f_cross"] for v in results.values() if v["group"] == "nlp"]
    gen_fcross = [v["geometry"]["f_cross"] for v in results.values() if v["group"] == "genomic"]
    nlp_max = max(nlp_fcross)
    gen_min = min(gen_fcross)

    complete_sep = gen_min > nlp_max
    if nlp_max > 0:
        margin_pass = gen_min >= 1.5 * nlp_max
        margin_rule = "relative (1.5x)"
    else:
        margin_pass = gen_min >= nlp_max + 0.10
        margin_rule = "absolute (+0.10)"
    q1_pass = complete_sep and margin_pass

    # exact_ranksum_pvalue tests "group_a tends to exceed group_b"; call with genomic as
    # group_a to report the informative direction (genomic > nlp).
    ranksum = exact_ranksum_pvalue(gen_fcross, nlp_fcross)

    branch = None
    q2_median_pct = None
    q3_median_kappa = None

    if not q1_pass:
        branch = "C"
    else:
        gen_pcts = [v["layer_position"]["percentile"] for v in results.values() if v["group"] == "genomic"]
        q2_median_pct = float(np.median(gen_pcts))
        q2_pass = q2_median_pct >= 75.0
        if not q2_pass:
            branch = "B"
        else:
            gen_kappas = [v["geometry"]["kappa"] for v in results.values() if v["group"] == "genomic"]
            numeric_kappas = [k for k in gen_kappas if isinstance(k, (int, float))]
            q3_median_kappa = float(np.median(numeric_kappas)) if numeric_kappas else None
            q3_pass = (q3_median_kappa is not None) and (q3_median_kappa >= 0.3)
            branch = "A" if q3_pass else "D"

    decision = dict(
        nlp_max_f_cross=nlp_max, gen_min_f_cross=gen_min,
        complete_separation=complete_sep, margin_rule=margin_rule, margin_pass=margin_pass,
        q1_pass=q1_pass,
        exact_ranksum=ranksum,
        q2_median_genomic_layer_percentile=q2_median_pct,
        q3_median_genomic_kappa=q3_median_kappa,
        branch=branch,
    )

    print("=" * 70)
    print(f"E6 STAGE A MECHANICAL DECISION: BRANCH {branch}")
    print(f"  NLP f_cross:     {nlp_fcross}")
    print(f"  Genomic f_cross: {gen_fcross}")
    print(f"  Q1 (replication): complete_sep={complete_sep} margin_pass={margin_pass} ({margin_rule}) -> {q1_pass}")
    if q2_median_pct is not None:
        print(f"  Q2 (specificity): median genomic layer percentile = {q2_median_pct:.1f} (need >= 75.0)")
    if q3_median_kappa is not None:
        print(f"  Q3 (coherence):   median genomic kappa = {q3_median_kappa:.4f} (need >= 0.3)")
    print(f"  exact rank-sum (genomic vs nlp): p = {ranksum['p_value_one_sided']:.4f}  "
          f"complete_separation={ranksum['is_complete_separation']}")
    print("=" * 70)

    out = dict(prereg=str(PREREG), panel_order=[s["model"] + f"_L{s['layer']}_r{s['row']}" for s in PANEL],
               results=results, decision=decision)
    out_path = ROOT / "results" / "e6_stageA.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
