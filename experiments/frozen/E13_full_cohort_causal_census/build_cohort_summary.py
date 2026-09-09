#!/usr/bin/env python3
"""Build the preregistered one-row-per-model E13 cohort summary."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R = ROOT / "results"
E13 = R / "E13"


def read_csv(path):
    with path.open() as f: return list(csv.DictReader(f))


def f(v):
    if v in (None, "", "nan"): return None
    return float(v)


def structural_gap(model, source):
    """Primary candidate q1 minus median structural-control q1, only where measured."""
    path = ROOT / source
    if not path.exists(): return None
    d = json.loads(path.read_text())
    controls = d.get("control_rows", [])
    q = [float(x["q1"]) for x in controls if x.get("q1") is not None]
    if not q: return None
    if "spectral" in d:
        cq = float(d["spectral"]["q1"])
    elif "candidate" in d and d["candidate"].get("q1") is not None:
        cq = float(d["candidate"]["q1"])
    else: return None
    return cq - median(q)


def main():
    manifest=json.loads((E13/"candidate_manifest.json").read_text())
    cand=read_csv(E13/"candidate_effects.csv")
    ctrl=read_csv(E13/"control_effects.csv")
    tomo=read_csv(E13/"tomography_results.csv")
    structural={r["model"]:r for r in read_csv(R/"E11"/"scale_ladder_backfilled.csv")}
    out=[]
    for model in manifest["panel_order"]:
        sr=structural[model]
        primary=next(c for c in manifest["candidates"] if c["model"]==model and c["primary"])
        row={"model":model,"architecture":sr["architecture"],"domain":sr["domain"],
             "total_params":int(sr["total_params"]),
             "n_structural_candidates":manifest["candidate_counts"][model],
             "primary_candidate_q1":primary["q1"],
             "candidate_control_structural_q1_gap":structural_gap(model,primary["source"])}
        for eps,label in ((0.5,"eps0p5"),(1.0,"eps1p0")):
            cr=[x for x in cand if x["model"]==model and float(x["epsilon"])==eps]
            strongest=max(cr,key=lambda x:float(x["relative_loss_change"]))
            controls=[float(x["relative_loss_change"]) for x in ctrl
                      if x["model"]==model and int(x["layer"])==int(strongest["layer"])
                      and float(x["epsilon"])==eps]
            row[f"strongest_singleton_relative_loss_change_{label}"]=float(strongest["relative_loss_change"])
            row[f"strongest_singleton_layer_{label}"]=int(strongest["layer"])
            row[f"strongest_singleton_row_{label}"]=int(strongest["row"])
            row[f"median_same_layer_control_effect_{label}"]=median(controls)
            row[f"candidate_minus_median_control_effect_{label}"]=float(strongest["relative_loss_change"])-median(controls)
            tr=next(x for x in tomo if x["model"]==model and float(x["epsilon"])==eps)
            if eps==0.5:
                row["interaction_identifiable"]=tr["interaction_identifiable"].lower()=="true"
            for k in ("F2_heldout_mae","F2_heldout_normalized_mae","F3_heldout_mae",
                      "F3_heldout_normalized_mae","F2_to_F3_relative_mae_improvement",
                      "bootstrap_ci95_low","bootstrap_ci95_high"):
                row[f"{k}_{label}"]=f(tr[k])
            row[f"pair_terms_materially_improved_{label}"]=(
                None if tr["pair_terms_materially_improved"]=="" else
                tr["pair_terms_materially_improved"].lower()=="true")
            row[f"second_order_observer_adequate_{label}"]=(
                None if tr["F3_adequate"]=="" else tr["F3_adequate"].lower()=="true")
            row[f"interaction_decision_{label}"]=tr["decision"]
        out.append(row)
    csv_path=E13/"cohort_summary.csv"
    with csv_path.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    payload={"n_models":len(out),"panel_order":manifest["panel_order"],
             "missing_models":[],"rows":out}
    (E13/"cohort_summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(f"wrote {csv_path} and cohort_summary.json ({len(out)} models)")


if __name__=="__main__": main()
