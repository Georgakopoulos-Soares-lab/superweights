#!/usr/bin/env python3
"""Compare Stage-0 measurements to locked E9/E10 references and apply the frozen gate."""
from __future__ import annotations

import csv,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
R=ROOT/"results"
E13=R/"E13"
MODELS=["llama","mistral","olmo","phi3","modernbert-base","dnabert2"]


def decoder_old(model,layer,row,eps):
    d=json.loads((R/f"e10_decoder_spectrum_{model}.json").read_text())
    alpha=1-eps
    c=next((x for x in d["conditions"] if x.get("layer")==layer and x.get("row")==row
            and x.get("alpha")==alpha),None)
    return None if c is None else {"baseline":d["baseline_nll"],"loss":c["nll"],
                                    "relative":c["relative_pct_change"]/100}


def singleton_old(path,index,eps,baseline_key):
    d=json.loads(path.read_text()); c=d["responses"]["pools"]["singletons"][index]
    return {"baseline":d[baseline_key],"loss":c[f"loss_eps{eps}"],
            "relative":c[f"dloss_eps{eps}"]/d[baseline_key]}


def judgment(new,old):
    if old is None:return "NO_MATCHED_OLD_EPSILON_REFERENCE"
    if new==0 or old["relative"]==0:
        return "REPRODUCES_SMALL_NOISY" if abs(new-old["relative"])<.01 else "MATERIAL_DISCREPANCY"
    if (new>0)!=(old["relative"]>0):return "MATERIAL_DISCREPANCY"
    ratio=abs(new/old["relative"])
    if .5<=ratio<=2:return "REPRODUCES"
    # Protocol allows both effects being small/noisy in the same direction.
    if abs(new)<.02 and abs(old["relative"])<.02:return "REPRODUCES_SMALL_NOISY"
    return "MATERIAL_DISCREPANCY"


def main():
    raw=json.loads((E13/"stage0_raw_responses.json").read_text())
    missing=[m for m in MODELS if m not in raw]
    if missing:raise SystemExit(f"Stage 0 incomplete: {missing}")
    rows=[]
    for model in MODELS:
        d=raw[model]
        for c in d["conditions"]:
            if c["kind"]!="candidate":continue
            eps=c["epsilon"]
            if model in ("llama","mistral","olmo","phi3"):
                old=decoder_old(model,c["layer"],c["row"],eps)
            elif model=="modernbert-base":
                old=singleton_old(R/"e10_encoder_responses_modernbert.json",0,eps,
                                  "baseline_mlm_loss")
            else:
                old=singleton_old(ROOT/"manuscript"/"experiments"/
                                  "E9_mechanistic_tomography"/"dnabert2_mask_responses.json",
                                  0,eps,"baseline_mlm_loss")
            j=judgment(c["relative_loss_change"],old)
            rows.append({"model":model,"layer":c["layer"],"row":c["row"],"epsilon":eps,
                         "new_baseline_loss":d["baseline_loss"],
                         "new_perturbed_loss":c.get("nll",c.get("loss")),
                         "new_relative_loss_change":c["relative_loss_change"],
                         "old_baseline_loss":"" if old is None else old["baseline"],
                         "old_perturbed_loss":"" if old is None else old["loss"],
                         "old_relative_loss_change":"" if old is None else old["relative"],
                         "new_to_old_effect_ratio":"" if old is None or old["relative"]==0 else
                         c["relative_loss_change"]/old["relative"],"judgment":j})
    with (E13/"stage0_reproduction_gate.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    bad=[r for r in rows if r["judgment"]=="MATERIAL_DISCREPANCY"]
    gate="STOP_MATERIAL_DISCREPANCY" if bad else "PASS"
    lines=["# E13 Stage-0 reproduction gate","",f"**Gate decision: {gate}.**","",
           "| Model | ε | New relative Δloss | Old relative Δloss | Ratio | Decision |",
           "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        old="—" if r["old_relative_loss_change"]=="" else f"{100*r['old_relative_loss_change']:+.3f}%"
        ratio="—" if r["new_to_old_effect_ratio"]=="" else f"{r['new_to_old_effect_ratio']:.3f}×"
        lines.append(f"| {r['model']} | {r['epsilon']} | {100*r['new_relative_loss_change']:+.3f}% | "
                     f"{old} | {ratio} | {r['judgment']} |")
    lines += ["","The gate uses the preregistered same-sign and approximate-magnitude rule. "
              "A missing historical epsilon cell is reported rather than fabricated; the available "
              "matched cell still governs reproduction for that model.","",
              "See `STAGE1_RESOURCE_ESTIMATE.md` for the pre-launch compute/disk/loading estimate."]
    (HERE/"STAGE0_REPORT.md").write_text("\n".join(lines)+"\n")
    (E13/"stage0_gate_decision.json").write_text(json.dumps({"decision":gate,
        "material_discrepancies":bad,"n_rows":len(rows)},indent=2)+"\n")
    print(gate)
    if bad:raise SystemExit(2)


if __name__=="__main__":main()
