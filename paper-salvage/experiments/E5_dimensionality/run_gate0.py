"""
experiments/E5_dimensionality/run_gate0.py

Confirmatory Gate-0 run. Verifies the prereg lock BEFORE loading any real target-row weight
(per PREREG_dimensionality_gate0.md). Computes, for the six-model primary panel:

  0B  exactness robustness check (cross-term fraction, all-rows exact rank)
      -- gates whether 0C/0D proceed at all (stop-the-study trigger)
  0C  D / A / C factor decomposition + cross-factor winner ranks
  0D  D x A permutation null, alignment-excess effect sizes, exact rank-sum decision

Writes the single source-of-truth JSON to
  genomic-super-weights/results/e5_gate0.json
GATE0_RESULTS.md is written separately, by hand, from this JSON -- so the JSON is never
silently out of sync with the prose.

Usage:
  python experiments/E5_dimensionality/run_gate0.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE.parent
sys.path.insert(0, str(SALVAGE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SALVAGE / "experiments" / "E1_nlp_validation"))

from uk_frobenius import ADAPTERS  # noqa: E402
from run_e1_retrospective import fetch_layer_tensors  # noqa: E402
from dimensionality_lib import (  # noqa: E402
    factor_D, factor_A, factor_C, vector_profile, rank_of,
    exact_uk_all_rows, cross_term_fraction,
    permutation_null, alignment_excess_max, alignment_excess_top1share, alignment_excess_pr,
    exact_ranksum_pvalue, complete_separation,
)

PREREG = SALVAGE / "docs" / "prereg" / "PREREG_dimensionality_gate0.md"
N_PERM = 20_000
MASTER_SEED = 42

PANEL = [
    dict(idx=0, group="nlp", name="Llama-7B", repo="huggyllama/llama-7b",
         layer=2, row=3968, loader="nlp_shard"),
    dict(idx=1, group="nlp", name="Mistral-7B", repo="mistralai/Mistral-7B-v0.1",
         layer=1, row=2070, loader="nlp_shard"),
    dict(idx=2, group="nlp", name="OLMo-7B", repo="allenai/OLMo-7B-0724-hf",
         layer=1, row=269, loader="nlp_shard"),
    dict(idx=3, group="genomic", name="GENERator EUK",
         repo="GenerTeam/GENERator-v2-eukaryote-3b-base", layer=4, row=2371,
         loader="generator", revision="7dc01bccce5b65e15141170538afdc2ff09d8dde"),
    dict(idx=4, group="genomic", name="DNABERT-2", repo="zhihan1996/DNABERT-2-117M",
         layer=5, row=603, loader="dnabert2",
         revision="7bce263b15377fc15361f52cfab88f8b586abda0"),
    dict(idx=5, group="genomic", name="NTv3", repo="InstaDeepAI/NTv3_650M_pre",
         layer=11, row=1472, loader="ntv3",
         code_revision="0ecff3637f0d3ba5b686d1095083218157c2ca34"),
]


def verify_lock() -> None:
    r = subprocess.run(
        [sys.executable, str(SALVAGE / "src" / "prereg_lock.py"), "verify", str(PREREG)],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0 or "OK" not in r.stdout:
        print(r.stderr, file=sys.stderr)
        raise SystemExit("prereg_lock verify FAILED -- refusing to load real weights. "
                          "The Gate-0 prereg must verify unchanged before any confirmatory "
                          "number is generated.")


def load_weights(spec: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (W_gate, W_up, W_down) in canonical convention, float32 as loaded (promoted
    to float64 by the caller before any arithmetic)."""
    loader = spec["loader"]
    if loader == "nlp_shard":
        tensors = fetch_layer_tensors(spec["repo"], spec["layer"])
        if tensors is None:
            raise RuntimeError(f"{spec['name']}: shard fetch failed")
        return tensors["gate"], tensors["up"], tensors["down"]

    if loader == "generator":
        from transformers import AutoModelForCausalLM
        # Deviation from the prereg's literal "AutoModel.from_pretrained" wording: GENERator
        # is a LlamaForCausalLM-family remote-code model whose base AutoModel class does not
        # expose the .model.layers path the adapter_llama_swiglu docstring requires; the
        # ForCausalLM wrapper does (matches the existing tested pattern in
        # scripts/analysis/run_sw_mechanistic.py::extract_mlp_weights and
        # scripts/interpretability/run_sw_broadcast_impulse.py's "llama" arch branch). This
        # is a loader-mechanics correction, not a change to which tensors, model, layer, or
        # row are used -- disclosed in GATE0_RESULTS.md's deviations section, prereg not
        # re-locked over it.
        m = AutoModelForCausalLM.from_pretrained(
            spec["repo"], revision=spec.get("revision"), trust_remote_code=True,
            torch_dtype=torch.float32)
        g, u, d = ADAPTERS["llama_swiglu"](m, spec["layer"])
        g, u, d = g.detach().clone(), u.detach().clone(), d.detach().clone()
        del m
        return g, u, d

    if loader == "dnabert2":
        from transformers import AutoModelForMaskedLM
        # Same class of deviation as above: adapter_dnabert2 requires model.bert.encoder...,
        # which only the *ForMaskedLM wrapper exposes (bare AutoModel returns a BertModel
        # with no .bert attribute). Matches neuron_pilot_common.get_wo_module's and
        # run_sw_broadcast_impulse.py's own tested "bert" arch loading.
        m = AutoModelForMaskedLM.from_pretrained(
            spec["repo"], revision=spec["revision"], trust_remote_code=True,
            torch_dtype=torch.float32)
        g, u, d = ADAPTERS["dnabert2"](m, spec["layer"])
        g, u, d = g.detach().clone(), u.detach().clone(), d.detach().clone()
        del m
        return g, u, d

    if loader == "ntv3":
        from transformers import AutoModelForMaskedLM
        m = AutoModelForMaskedLM.from_pretrained(
            spec["repo"], trust_remote_code=True, code_revision=spec["code_revision"],
            torch_dtype=torch.float32)
        g, u, d = ADAPTERS["ntv3"](m, spec["layer"])
        g, u, d = g.detach().clone(), u.detach().clone(), d.detach().clone()
        del m
        return g, u, d

    raise ValueError(f"unknown loader {loader}")


def profile_dict(p) -> dict:
    return dict(top1_index=p.top1_index, top1_share=p.top1_share, top5_share=p.top5_share,
                top10_share=p.top10_share, participation_ratio=p.participation_ratio,
                d_ffn=p.d_ffn)


def ae_dict(ae) -> dict:
    return dict(observed=ae.observed, null_median=ae.null_median, null_mad=ae.null_mad,
                ae=ae.ae, percentile=ae.percentile)


def main() -> None:
    print("Verifying prereg lock before touching any real weight ...")
    verify_lock()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"0B device: {device}\n")

    seed_children = np.random.SeedSequence(MASTER_SEED).spawn(len(PANEL))

    # ---- Phase 1: load weights, compute 0B for every model before any 0C/0D decision ----
    phase1 = {}
    for spec in PANEL:
        name = spec["name"]
        print(f"=== [{spec['idx']}] {name} ({spec['group']}) L{spec['layer']} row {spec['row']} ===")
        t0 = time.time()
        Wg, Wu, Wd = load_weights(spec)
        d_ffn, d_model = Wg.shape
        assert Wd.shape == (d_model, d_ffn), f"{name}: shape guard failed {tuple(Wd.shape)}"
        assert 0 <= spec["row"] < d_model, f"{name}: row {spec['row']} out of range for d_model={d_model}"
        print(f"  loaded in {time.time()-t0:.1f}s  gate{tuple(Wg.shape)} up{tuple(Wu.shape)} down{tuple(Wd.shape)}")

        D = factor_D(Wd[spec["row"]])
        A = factor_A(Wg, Wu)
        C = factor_C(D, A)

        t0 = time.time()
        exact_all = exact_uk_all_rows(Wg, Wu, Wd, device=device)
        s_exact = float(exact_all[spec["row"]])
        s_diag = float(math.sqrt(float(C.sum())))
        f_cross = cross_term_fraction(s_exact, s_diag)
        exact_rank = rank_of(exact_all, spec["row"])
        exact_pct = 100.0 * (1.0 - (exact_rank - 1) / d_model)
        print(f"  0B: exact {time.time()-t0:.1f}s  S_exact={s_exact:.6g} S_diag={s_diag:.6g} "
              f"f_cross={f_cross:.4f}  exact_rank={exact_rank}/{d_model} (pct {exact_pct:.3f})")

        stop = (abs(f_cross) > 0.20) or (exact_pct < 99.0)
        if stop:
            print(f"  !!! STOP-TRIGGER FIRED for {name}: |f_cross|={abs(f_cross):.4f} "
                  f"(>0.20?) exact_pct={exact_pct:.3f} (<99.0?)")

        phase1[name] = dict(
            spec=spec, d_model=d_model, d_ffn=d_ffn,
            D=D, A=A, C=C, seed_seq=seed_children[spec["idx"]],
            b0=dict(s_exact=s_exact, s_diag=s_diag, f_cross=f_cross,
                    exact_rank=exact_rank, exact_percentile=exact_pct, device=device,
                    stop_triggered=stop),
        )
        del Wg, Wu, Wd, exact_all
        print()

    any_stop = any(v["b0"]["stop_triggered"] for v in phase1.values())
    print("=" * 70)
    print(f"0B stop-trigger check across all {len(PANEL)} models: "
          f"{'FIRED -- ABORTING' if any_stop else 'clear, proceeding to 0C/0D'}")
    print("=" * 70)

    out = {
        "prereg": str(PREREG), "n_perm": N_PERM, "master_seed": MASTER_SEED,
        "panel_order": [s["name"] for s in PANEL],
        "stop_trigger_fired": any_stop,
        "models": {},
    }

    if any_stop:
        for name, v in phase1.items():
            out["models"][name] = dict(group=v["spec"]["group"], b0=v["b0"])
        write_out(out)
        print("\nGate 0 ABORTED at the 0B stop trigger -- 0C/0D were NOT run on any model, "
              "per the preregistered rule. See GATE0_RESULTS.md for the required write-up.")
        return

    # ---- Phase 2: 0C factor decomposition + 0D permutation test ----
    for name, v in phase1.items():
        spec, D, A, C = v["spec"], v["D"], v["A"], v["C"]
        prof_D, prof_A, prof_C = vector_profile(D), vector_profile(A), vector_profile(C)
        i_star = prof_C.top1_index
        rank_i_star_under_D = rank_of(D, i_star)
        rank_i_star_under_A = rank_of(A, i_star)
        print(f"[{name}] 0C: C top1_index={i_star} top1_share={prof_C.top1_share:.4f} "
              f"PR={prof_C.participation_ratio:.2f}  "
              f"(that index ranks #{rank_i_star_under_D} under D, #{rank_i_star_under_A} under A)")

        t0 = time.time()
        null = permutation_null(D, A, N_PERM, v["seed_seq"])
        ae_max = alignment_excess_max(float(C.max()), null)
        ae_top1 = alignment_excess_top1share(prof_C.top1_share, null)
        ae_pr = alignment_excess_pr(prof_C.participation_ratio, null)
        print(f"  0D: {N_PERM} permutations in {time.time()-t0:.1f}s   "
              f"AE_top1share={ae_top1.ae if isinstance(ae_top1.ae, str) else f'{ae_top1.ae:.3f}'} "
              f"(pct {ae_top1.percentile:.2f})   "
              f"AE_max={ae_max.ae if isinstance(ae_max.ae, str) else f'{ae_max.ae:.3f}'}   "
              f"AE_PR={ae_pr.ae if isinstance(ae_pr.ae, str) else f'{ae_pr.ae:.3f}'}")

        out["models"][name] = dict(
            group=spec["group"], layer=spec["layer"], row=spec["row"],
            repo=spec["repo"], d_model=v["d_model"], d_ffn=v["d_ffn"],
            b0=v["b0"],
            c0=dict(
                D=profile_dict(prof_D), A=profile_dict(prof_A), C=profile_dict(prof_C),
                c_winner_index=i_star,
                c_winner_rank_under_D=rank_i_star_under_D,
                c_winner_rank_under_A=rank_i_star_under_A,
            ),
            d0=dict(ae_max=ae_dict(ae_max), ae_top1share=ae_dict(ae_top1), ae_pr=ae_dict(ae_pr)),
        )
        print()

    # ---- Gate-0 mechanical decision ----
    def group_values(metric_key: str, group: str) -> list:
        vals = []
        for name, m in out["models"].items():
            if m["group"] != group:
                continue
            v = m["d0"][metric_key]["ae"]
            vals.append(v)
        return vals

    nlp_top1 = group_values("ae_top1share", "nlp")
    gen_top1 = group_values("ae_top1share", "genomic")
    nlp_max = group_values("ae_max", "nlp")
    gen_max = group_values("ae_max", "genomic")
    nlp_pr = group_values("ae_pr", "nlp")
    gen_pr = group_values("ae_pr", "genomic")

    primary_pass = complete_separation(nlp_top1, gen_top1)
    primary_test = exact_ranksum_pvalue(
        [v for v in nlp_top1 if isinstance(v, (int, float))],
        [v for v in gen_top1 if isinstance(v, (int, float))],
    ) if all(isinstance(v, (int, float)) for v in nlp_top1 + gen_top1) else None

    secondary_max_pass = complete_separation(nlp_max, gen_max)
    secondary_pr_pass = complete_separation(nlp_pr, gen_pr)

    out["decision"] = dict(
        primary_metric="AE_top1share",
        nlp_values=dict(zip([n for n, m in out["models"].items() if m["group"] == "nlp"], nlp_top1)),
        genomic_values=dict(zip([n for n, m in out["models"].items() if m["group"] == "genomic"], gen_top1)),
        primary_pass=primary_pass,
        exact_ranksum=primary_test,
        secondary_ae_max_complete_separation=secondary_max_pass,
        secondary_ae_pr_complete_separation=secondary_pr_pass,
        gate0_result="PASS" if primary_pass else "FAIL",
    )

    print("=" * 70)
    print(f"GATE 0 PRIMARY DECISION: {'PASS' if primary_pass else 'FAIL'}")
    print(f"  NLP AE_top1share:     {out['decision']['nlp_values']}")
    print(f"  Genomic AE_top1share: {out['decision']['genomic_values']}")
    if primary_test:
        print(f"  exact rank-sum p = {primary_test['p_value_one_sided']:.4f}  "
              f"(complete separation: {primary_test['is_complete_separation']})")
    print(f"  secondary AE_max complete separation:  {secondary_max_pass}")
    print(f"  secondary AE_PR  complete separation:  {secondary_pr_pass}")
    print("=" * 70)

    write_out(out)


def write_out(out: dict) -> None:
    out_path = ROOT / "results" / "e5_gate0.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.random.SeedSequence):
            return {"entropy": o.entropy}
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        raise TypeError(type(o))

    # strip the large in-memory tensors before serializing (spec/seed_seq only in phase1,
    # not in `out`, so this is just a defensive default for anything unexpected)
    out_path.write_text(json.dumps(out, indent=2, default=_default))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
