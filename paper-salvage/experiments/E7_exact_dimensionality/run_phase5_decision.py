"""
experiments/E7_exact_dimensionality/run_phase5_decision.py

Phase 5 final step: assemble the four confirmatory-panel results (Phi-3's published-coordinate
aggregate, Qwen2.5/GenomeOcean/Evo2's Phase-1-detected-row spectra) and apply the mechanical
Branch A/B/C/D decision tree exactly as PREREG_exact_operator_dimensionality.md specifies.

Usage:
  python run_phase5_decision.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE.parent
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


def main() -> None:
    print("Verifying E7 prereg lock ...")
    verify_lock()

    phi3 = json.loads((ROOT / "results" / "e7_phi3_spectral.json").read_text())
    qwen25 = json.loads((ROOT / "results" / "e7_phase1_detection_qwen25.json").read_text())
    genomeocean = json.loads((ROOT / "results" / "e7_phase1_detection_genomeocean.json").read_text())
    evo2 = json.loads((ROOT / "results" / "e7_phase1_detection_evo2.json").read_text())

    models = {}
    models["Phi-3-mini-4k-instruct"] = dict(
        group="nlp", null=False,
        q1=phi3["model_level"]["q1_median"], pr_spec=phi3["model_level"]["pr_spec_median"],
        n_rows_aggregated=len(phi3["rows"]),
    )
    for name, det in [("Qwen2.5-7B", qwen25), ("GenomeOcean-4B", genomeocean), ("Evo2-7B", evo2)]:
        if det["null_outcome"]:
            models[name] = dict(group=det["group"], null=True, q1=None, pr_spec=None)
        else:
            models[name] = dict(group=det["group"], null=False,
                                 q1=det["spectral"]["q1"], pr_spec=det["spectral"]["pr_spec"],
                                 candidate=det["candidate"])

    nlp_models = {k: v for k, v in models.items() if v["group"] == "nlp"}
    genomic_models = {k: v for k, v in models.items() if v["group"] == "genomic"}
    nlp_surviving = {k: v for k, v in nlp_models.items() if not v["null"]}
    genomic_surviving = {k: v for k, v in genomic_models.items() if not v["null"]}

    print(f"\nNLP models: {list(nlp_models.keys())} -- surviving: {list(nlp_surviving.keys())}")
    print(f"Genomic models: {list(genomic_models.keys())} -- surviving: {list(genomic_surviving.keys())}")

    decision = dict(models={k: {kk: vv for kk, vv in v.items() if kk != "candidate"} for k, v in models.items()})

    # Step 1: completeness check
    if len(nlp_surviving) == 0 or len(genomic_surviving) == 0:
        decision["branch"] = "D"
        decision["reason"] = "one or both groups have zero surviving candidates"
        print(f"\nBRANCH D: {decision['reason']}")
        write_and_exit(decision)
        return

    degraded = (len(nlp_surviving) < 2) or (len(genomic_surviving) < 2)
    if degraded:
        decision["degraded_to_exploratory"] = True
        print("\nNOTE: one group has only 1 surviving model -- this run is an EXPLORATORY "
              "EXTENSION, not eligible for Branch A, per the prereg.")

    nlp_q1 = {k: v["q1"] for k, v in nlp_surviving.items()}
    gen_q1 = {k: v["q1"] for k, v in genomic_surviving.items()}
    nlp_pr = {k: v["pr_spec"] for k, v in nlp_surviving.items()}
    gen_pr = {k: v["pr_spec"] for k, v in genomic_surviving.items()}

    print(f"\nNLP q1: {nlp_q1}")
    print(f"Genomic q1: {gen_q1}")
    print(f"NLP PR_spec: {nlp_pr}")
    print(f"Genomic PR_spec: {gen_pr}")

    decision["nlp_q1"] = nlp_q1
    decision["genomic_q1"] = gen_q1
    decision["nlp_pr_spec"] = nlp_pr
    decision["genomic_pr_spec"] = gen_pr

    if degraded:
        decision["branch"] = "EXPLORATORY (not A/B/C/D -- insufficient models for a group decision)"
        write_and_exit(decision)
        return

    min_nlp_q1 = min(nlp_q1.values())
    max_gen_q1 = max(gen_q1.values())
    complete_sep = min_nlp_q1 > max_gen_q1
    within_nlp = max(nlp_q1.values()) - min(nlp_q1.values())
    within_gen = max(gen_q1.values()) - min(gen_q1.values())
    gap = min_nlp_q1 - max_gen_q1

    print(f"\nStep 2: min(NLP q1)={min_nlp_q1:.6f} max(genomic q1)={max_gen_q1:.6f} "
          f"gap={gap:.6f} within_nlp={within_nlp:.6f} within_genomic={within_gen:.6f}")

    branch_a = complete_sep and (gap > within_nlp) and (gap > within_gen)
    decision["step2"] = dict(complete_separation=complete_sep, gap=gap,
                              within_nlp_range=within_nlp, within_genomic_range=within_gen,
                              branch_a_pass=branch_a)

    if branch_a:
        # secondary consistency check on PR_spec
        min_nlp_pr = min(nlp_pr.values())
        max_gen_pr = max(gen_pr.values())
        pr_complete_sep = max_gen_pr > min_nlp_pr  # genomic > nlp orientation
        decision["step2_pr_spec_secondary"] = dict(
            min_nlp_pr=min_nlp_pr, max_gen_pr=max_gen_pr,
            genomic_exceeds_nlp=pr_complete_sep)
        decision["branch"] = "A"
        print(f"\nBRANCH A: complete separation AND gap exceeds both within-group ranges.")
        write_and_exit(decision)
        return

    # Step 3: Branch B check -- specifically GenomeOcean crossing while Evo2 stays separated
    go_q1 = gen_q1.get("GenomeOcean-4B")
    evo2_q1 = gen_q1.get("Evo2-7B")
    branch_b = False
    if go_q1 is not None and evo2_q1 is not None:
        go_crosses = go_q1 >= min_nlp_q1
        evo2_separated = evo2_q1 < min_nlp_q1
        evo2_gap = min_nlp_q1 - evo2_q1
        evo2_gap_exceeds_within_nlp = evo2_gap > within_nlp
        branch_b = go_crosses and evo2_separated and evo2_gap_exceeds_within_nlp
        decision["step3"] = dict(
            genomeocean_q1=go_q1, evo2_q1=evo2_q1, genomeocean_crosses_into_nlp_range=go_crosses,
            evo2_separated_from_nlp=evo2_separated, evo2_gap=evo2_gap,
            evo2_gap_exceeds_within_nlp_range=evo2_gap_exceeds_within_nlp, branch_b_pass=branch_b)

    if branch_b:
        decision["branch"] = "B"
        print("\nBRANCH B: GenomeOcean-4B crosses into/above the NLP q1 range while Evo2-7B "
              "remains clearly, individually separated from NLP.")
        write_and_exit(decision)
        return

    decision["branch"] = "C"
    print("\nBRANCH C: separation fails and does not match the specific Branch-B pattern -- "
          "no robust model-level difference found under this panel.")
    write_and_exit(decision)


def write_and_exit(decision: dict) -> None:
    out_path = ROOT / "results" / "e7_phase5_decision.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
