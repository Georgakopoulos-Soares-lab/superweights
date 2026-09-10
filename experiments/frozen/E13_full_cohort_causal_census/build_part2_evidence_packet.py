#!/usr/bin/env python3
"""Build the Part 2 evidence tables and replacement Figure 2 from final E13 artifacts."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
E13 = ROOT / "results" / "E13"
AUDIT2 = ROOT / "audit" / "round2"
FIGDIR = ROOT / "manuscript" / "figures" / "main"
OUT_MD = ROOT / "PART2_EVIDENCE_PACKET.md"
EXCLUDED = "Phi-3-mini-4k-instruct"
N_BOOT = 5000

# round-2 panel D: display-name / sign-flip model set for the top-norm-control comparison
# (audit/rederivations/final_check.md Section 2; audit/rederivations/scripts/fig2_panel_topnorm.py)
DISPLAY_D = {
    "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B": "Qwen2.5-3B", "HuggingFaceTB/SmolLM2-135M": "SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M": "SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B": "SmolLM2-1.7B",
    "GenerTeam/GENERator-v2-prokaryote-1.2b-base": "GEN-PROK-1.2B",
    "GenerTeam/GENERator-v2-prokaryote-3b-base": "GEN-PROK-3B",
    "EuroBERT/EuroBERT-210m": "EuroBERT-210M", "EuroBERT/EuroBERT-610m": "EuroBERT-610M",
    "EuroBERT/EuroBERT-2.1B": "EuroBERT-2.1B", "answerdotai/ModernBERT-large": "ModernBERT-large",
    "ModernBERT-base": "ModernBERT-base", "DNABERT-2": "DNABERT-2",
    "GENERator-EUK-3B": "GEN-EUK-3B", "Llama-7B": "Llama-7B", "Mistral-7B": "Mistral-7B",
    "OLMo-7B-0724-hf": "OLMo-7B", "MosaicBERT": "MosaicBERT", "GenomeOcean-4B": "GenomeOcean-4B",
    "Qwen2.5-7B": "Qwen2.5-7B", "NTv3": "NTv3",
}
SIGN_FLIP_MODELS_D = {"MosaicBERT", "Qwen2.5-7B", "Qwen/Qwen2.5-0.5B", "NTv3"}


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def pct(x: float, digits=2) -> str:
    return f"{100*x:+.{digits}f}%"


def ci_median(values: list[float], seed: int) -> tuple[float, float]:
    a = np.asarray(values, float); rng = np.random.default_rng(seed)
    b = np.asarray([np.median(rng.choice(a, len(a), replace=True)) for _ in range(N_BOOT)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def display_name(model: str) -> str:
    return (model.replace("HuggingFaceTB/", "").replace("Qwen/", "")
            .replace("GenerTeam/GENERATOR-v2-", "GENERATOR-")
            .replace("GenerTeam/GENERator-v2-", "GENERATOR-")
            .replace("answerdotai/", "").replace("EuroBERT/", ""))


def main() -> None:
    manifest = json.loads((E13 / "candidate_manifest.json").read_text())
    meta = {r["model"]: r for r in read_csv(ROOT / "results" / "E11" / "scale_ladder_backfilled.csv")}
    prim = {c["model"]: c for c in manifest["candidates"] if c["primary"] and c["model"] != EXCLUDED}
    controls = {c["model"]: c for c in manifest["control_sets"] if c["model"] != EXCLUDED}
    panel = [m for m in manifest["panel_order"] if m != EXCLUDED]
    candidate_final = {
        (r["model"], float(r["epsilon"])): r
        for r in read_csv(E13 / "part1_22_candidate_effects.csv")
    }
    control_final = {
        (r["model"], int(r["layer"]), int(r["row"]), float(r["epsilon"])): r
        for r in read_csv(E13 / "part1_22_control_effects.csv")
    }
    raw = {}
    for path in (E13 / "raw").glob("*.json"):
        d = json.loads(path.read_text()); raw[d["model"]] = d

    rows, long_rows = [], []
    for model in panel:
        d, m, c, cs = raw[model], meta[model], prim[model], controls[model]
        candidate_conditions = [x for x in d["conditions"] if x["kind"] == "candidate"]
        assert {(x["layer"], x["row"]) for x in candidate_conditions} == {(c["layer"], c["row"])}
        assert sorted({x["row"] for x in d["conditions"] if x["kind"] == "control"}) == cs["control_rows"]
        row = {
            "model": model, "display_model": display_name(model), "repo": d["repo"],
            "requested_revision": d["requested_revision"], "resolved_revision": d.get("resolved_revision"),
            "architecture": d["architecture"], "domain": d["domain"], "family": m["family"],
            "total_params": int(m["total_params"]), "non_embed_params": int(m["non_embed_params"]),
            "candidate_layer": c["layer"], "candidate_row": c["row"],
            "candidate_source": c["source"], "q1": c["q1"], "pr_spec": float(m["pr_spec"]),
            "frob_norm": c["frob_norm"], "layer_median_frob_norm": c["layer_median_frob_norm"],
            "frob_ratio_to_layer_median": c["frob_ratio_to_layer_median"],
            "control_rows": ";".join(str(x) for x in cs["control_rows"]),
            "baseline_loss": d["baseline_loss"], "endpoint_json": json.dumps(d["endpoint"], sort_keys=True),
            "raw_artifact": f"results/experiments/E13/raw/{d['slug']}.json",
        }
        for eps, tag in ((0.5, "eps0p5"), (1.0, "eps1p0")):
            cand = next(x for x in candidate_conditions if x["epsilon"] == eps)
            cand_stored = candidate_final[(model, eps)]
            assert np.isclose(float(cand_stored["relative_loss_change"]), cand["relative_loss_change"])
            ctrls = sorted([x for x in d["conditions"] if x["kind"] == "control" and x["epsilon"] == eps], key=lambda x:x["row"])
            rels = [float(x["relative_loss_change"]) for x in ctrls]
            abss = [float(x["absolute_delta_loss"]) for x in ctrls]
            row.update({
                f"candidate_perturbed_loss_{tag}": cand["perturbed_loss"],
                f"candidate_absolute_delta_loss_{tag}": cand["absolute_delta_loss"],
                f"candidate_relative_loss_change_{tag}": cand["relative_loss_change"],
                f"candidate_relative_ci95_low_{tag}": cand_stored["relative_ci95_low"],
                f"candidate_relative_ci95_high_{tag}": cand_stored["relative_ci95_high"],
                f"median_control_absolute_delta_loss_{tag}": statistics.median(abss),
                f"median_control_relative_loss_change_{tag}": statistics.median(rels),
                f"candidate_minus_median_control_absolute_{tag}": cand["absolute_delta_loss"] - statistics.median(abss),
                f"candidate_minus_median_control_relative_{tag}": cand["relative_loss_change"] - statistics.median(rels),
            })
            for i, x in enumerate(ctrls, 1):
                ctrl_stored = control_final[(model, x["layer"], x["row"], eps)]
                assert np.isclose(float(ctrl_stored["relative_loss_change"]), x["relative_loss_change"])
                row[f"control{i}_layer"] = x["layer"]; row[f"control{i}_row"] = x["row"]
                row[f"control{i}_perturbed_loss_{tag}"] = x["perturbed_loss"]
                row[f"control{i}_absolute_delta_loss_{tag}"] = x["absolute_delta_loss"]
                row[f"control{i}_relative_loss_change_{tag}"] = x["relative_loss_change"]
                row[f"control{i}_relative_ci95_low_{tag}"] = ctrl_stored["relative_ci95_low"]
                row[f"control{i}_relative_ci95_high_{tag}"] = ctrl_stored["relative_ci95_high"]
                long_rows.append({"model":model,"display_model":display_name(model),"architecture":d["architecture"],
                                  "domain":d["domain"],"epsilon":eps,"kind":"control","control_index":i,
                                  "layer":x["layer"],"row":x["row"],"baseline_loss":d["baseline_loss"],
                                  "perturbed_loss":x["perturbed_loss"],"absolute_delta_loss":x["absolute_delta_loss"],
                                  "relative_loss_change":x["relative_loss_change"]})
            long_rows.append({"model":model,"display_model":display_name(model),"architecture":d["architecture"],
                              "domain":d["domain"],"epsilon":eps,"kind":"candidate","control_index":"",
                              "layer":cand["layer"],"row":cand["row"],"baseline_loss":d["baseline_loss"],
                              "perturbed_loss":cand["perturbed_loss"],"absolute_delta_loss":cand["absolute_delta_loss"],
                              "relative_loss_change":cand["relative_loss_change"]})
        rows.append(row)

    write_csv(E13 / "part2_22_model_results.csv", rows)
    write_csv(E13 / "figure2_candidate_control_data.csv", long_rows)
    sf = [{"model":r["model"],"display_model":r["display_model"],"architecture":r["architecture"],
           "domain":r["domain"],"q1":r["q1"],"frob_ratio_to_layer_median":r["frob_ratio_to_layer_median"],
           "full_ablation_relative_loss_change":r["candidate_relative_loss_change_eps1p0"]} for r in rows]
    write_csv(E13 / "figure2_structure_function_data.csv", sf)
    stats = cohort_stats(rows)
    (E13 / "part2_recomputed_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    make_figure(rows, long_rows, stats)
    write_packet(rows, manifest, stats)
    print(f"WROTE {OUT_MD}, {len(rows)} models, {len(long_rows)} plotting rows")


def cohort_stats(rows: list[dict]) -> dict:
    out = {"n_models":len(rows),"n_boot":N_BOOT,"cohort_ci_method":"percentile model bootstrap of median"}
    for tag, seed in (("eps0p5",47),("eps1p0",52)):
        cand=[float(r[f"candidate_relative_loss_change_{tag}"]) for r in rows]
        ctrl=[float(r[f"median_control_relative_loss_change_{tag}"]) for r in rows]
        gaps=[float(r[f"candidate_minus_median_control_relative_{tag}"]) for r in rows]
        out[tag]={"candidate_gt_median_control":sum(x>0 for x in gaps),"median_candidate":statistics.median(cand),
                  "median_control":statistics.median(ctrl),"median_candidate_minus_control":statistics.median(gaps),
                  "candidate_minus_control_ci95":ci_median(gaps,seed),"seed":seed,
                  "sorted_candidate_effects":[{"model":r["model"],"effect":float(r[f"candidate_relative_loss_change_{tag}"])}
                                              for r in sorted(rows,key=lambda z:float(z[f"candidate_relative_loss_change_{tag}"]))]}
    xq=np.asarray([float(r["q1"]) for r in rows]); xf=np.asarray([float(r["frob_ratio_to_layer_median"]) for r in rows]);
    y=np.asarray([float(r["candidate_relative_loss_change_eps1p0"]) for r in rows])
    corr=json.loads((E13/"part1_22_structure_function_correlations.json").read_text())["analyses"]
    out["correlations"]={a["x"]:a for a in corr}
    assert np.isclose(spearmanr(xq,y)[0],out["correlations"]["q1"]["spearman_rho"])
    assert np.isclose(spearmanr(xf,y)[0],out["correlations"]["frob_ratio_to_layer_median"]["spearman_rho"])
    groups=defaultdict(list)
    for r in rows: groups[f"{r['domain']}/{r['architecture']}"].append(float(r["candidate_relative_loss_change_eps1p0"]))
    out["descriptive_subgroup_medians"]={k:{"n":len(v),"median":statistics.median(v)} for k,v in groups.items()}

    # round-2 panel D: candidate vs. top-norm-control causal gap (audit/rederivations/structural_vs_causal_gap.csv)
    topnorm = read_csv(AUDIT2 / "structural_vs_causal_gap.csv")
    assert len(topnorm) == 44, f"expected 44 rows (22 models x 2 eps), got {len(topnorm)}"
    by_model_eps_d = {(r["model"], r["epsilon"]): r for r in topnorm}
    assert {r["model"] for r in topnorm} == {r["model"] for r in rows}
    d_stats = {}
    for tag, eps in (("eps0p5", "0.5"), ("eps1p0", "1.0")):
        gaps = [float(by_model_eps_d[(r["model"], eps)]["causal_topk_gap"]) for r in rows]
        med_ctrl = [float(by_model_eps_d[(r["model"], eps)]["median_topk_control_relative_loss_change"]) for r in rows]
        d_stats[tag] = {
            "n_negative": sum(g < 0 for g in gaps),
            "median_gap": statistics.median(gaps),
            "median_topnorm_control_relative_loss_change": statistics.median(med_ctrl),
        }
    out["panel_d_topnorm_control"] = d_stats
    return out


def make_figure(rows: list[dict], long_rows: list[dict], stats: dict) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    # round-3: larger default type across the whole figure (labels, ticks, legends) --
    # explicit fontsize= calls below are bumped individually where they need to differ
    # from these defaults (e.g. panel D's small negative-gap annotations).
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })
    order=[r["model"] for r in sorted(rows,key=lambda z:float(z["candidate_relative_loss_change_eps1p0"]))]
    labels={r["model"]:r["display_model"] for r in rows}; x=np.arange(len(order))
    fig=plt.figure(figsize=(17,16),constrained_layout=True); gs=fig.add_gridspec(4,1,height_ratios=[1,1,1.05,1.15])
    # extra vertical padding between rows -- with the larger round-3 label fonts, the
    # default constrained-layout padding let adjacent rotated y-axis labels touch/overlap
    # at row boundaries (panel C's and panel D's ylabels in particular).
    fig.set_constrained_layout_pads(hspace=0.045, h_pad=0.06)
    for ax,eps,tag,title in [(fig.add_subplot(gs[0]),.5,"eps0p5","A  Partial suppression (ε=0.5; row scale α=0.5)"),
                             (fig.add_subplot(gs[1]),1.,"eps1p0","B  Full ablation (ε=1.0; row scale α=0)")]:
        for i,m in enumerate(order):
            vals=[100*float(r["relative_loss_change"]) for r in long_rows if r["model"]==m and float(r["epsilon"])==eps and r["kind"]=="control"]
            jitter=np.linspace(-.14,.14,len(vals)); ax.scatter(i+jitter,vals,s=17,c="#999999",alpha=.72,zorder=2)
            med=np.median(vals); ax.plot([i-.18,i+.18],[med,med],color="#444444",lw=1.2,zorder=3)
            cand=100*float(next(r for r in long_rows if r["model"]==m and float(r["epsilon"])==eps and r["kind"]=="candidate")["relative_loss_change"])
            ax.scatter(i,cand,s=48,marker="D",c="#CC3311",edgecolor="white",linewidth=.5,zorder=4)
        s=stats[tag]; lo,hi=s["candidate_minus_control_ci95"]
        ax.text(.01,.96,f"{s['candidate_gt_median_control']}/22 candidate > median control\nmedian gap {pct(s['median_candidate_minus_control'])} (95% CI {pct(lo)} to {pct(hi)})",
                transform=ax.transAxes,va="top",fontsize=11,bbox=dict(facecolor="white",alpha=.88,edgecolor="#cccccc"))
        ax.axhline(0,color="black",lw=.8); ax.set_yscale("symlog",linthresh=.1,linscale=.8)
        ax.set_ylabel("Relative loss change (%)"); ax.set_title(title,loc="left",fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in order],rotation=55,ha="right",fontsize=9)
        ax.margins(y=.10)
        ax.grid(axis="y",which="both",alpha=.18)
    ax=fig.add_subplot(gs[2]); colors={"text":"#4477AA","genomic":"#228833"}; markers={"decoder":"o","encoder":"s"}
    for r in rows:
        ax.scatter(float(r["q1"]),100*float(r["candidate_relative_loss_change_eps1p0"]),s=62,
                   c=colors[r["domain"]],marker=markers[r["architecture"]],edgecolor="white",linewidth=.6)
    cq=stats["correlations"]["q1"]; lo,hi=cq["bootstrap_ci95"]
    ax.text(.02,.96,f"Spearman ρ={cq['spearman_rho']:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]\np={cq['asymptotic_p']:.3f}; n=22",
            transform=ax.transAxes,va="top",fontsize=12,bbox=dict(facecolor="white",alpha=.9,edgecolor="#cccccc"))
    ax.axhline(0,color="black",lw=.8); ax.set_yscale("symlog",linthresh=.1,linscale=.8)
    ax.set_xlabel("Candidate spectral concentration q₁"); ax.set_ylabel("Full-ablation loss change (%)")
    ax.set_title("C  Spectral concentration does not predict causal-effect magnitude",loc="left",fontweight="bold")
    from matplotlib.lines import Line2D
    legend=[Line2D([0],[0],marker='o',color='w',label='Decoder',markerfacecolor='#666',markersize=7),
            Line2D([0],[0],marker='s',color='w',label='Encoder',markerfacecolor='#666',markersize=7),
            Line2D([0],[0],marker='o',color='w',label='Text',markerfacecolor=colors['text'],markersize=7),
            Line2D([0],[0],marker='o',color='w',label='Genomic',markerfacecolor=colors['genomic'],markersize=7)]
    ax.legend(handles=legend,ncol=4,loc="lower right",fontsize=10); ax.grid(alpha=.18,which="both")
    ax.margins(y=.10)

    # --- Panel D (round-2): candidate vs. top-norm-control causal gap, both epsilons ---
    topnorm = read_csv(AUDIT2 / "structural_vs_causal_gap.csv")
    by_model_eps_d = {(r["model"], r["epsilon"]): r for r in topnorm}
    ax_d = fig.add_subplot(gs[3])
    eps_style = {"0.5": dict(fill=True, s=52, offset=-0.14), "1.0": dict(fill=False, s=68, offset=0.14)}
    dom_marker = {"decoder": "o", "encoder": "s"}
    neg_offsets_d = [-13, -35, -57]
    for i, m in enumerate(order):
        r = next(z for z in rows if z["model"] == m)
        arch, dom = r["architecture"], r["domain"]
        neg_eps_count = 0
        for eps in ("0.5", "1.0"):
            dr = by_model_eps_d[(m, eps)]
            gap = float(dr["causal_topk_gap"])
            st = eps_style[eps]
            kw = dict(marker=dom_marker[arch], s=st["s"], zorder=4)
            if st["fill"]:
                kw.update(color=colors[dom], edgecolor="black", linewidths=0.9)
            else:
                kw.update(facecolors="none", edgecolors=colors[dom], linewidths=1.2)
            ax_d.scatter(i + st["offset"], gap, **kw)
            if m in SIGN_FLIP_MODELS_D and gap < 0:
                dy = neg_offsets_d[neg_eps_count % len(neg_offsets_d)]
                ax_d.annotate(f"{DISPLAY_D[m]}, " + r"$\epsilon$=" + eps, (i + st["offset"], gap),
                              xytext=(0, dy), textcoords="offset points", ha="center", va="top",
                              fontsize=7, color="0.15",
                              arrowprops=dict(arrowstyle="-", lw=0.4, color="0.5"))
                neg_eps_count += 1
    ax_d.axhline(0, color="black", lw=0.8, zorder=2)
    ax_d.set_yscale("symlog", linthresh=1e-4)
    ax_d.set_ylim(-4, 15)
    ax_d.set_ylabel("Candidate $-$ control (%)")
    ax_d.set_xticks(x); ax_d.set_xticklabels([labels[m] for m in order], rotation=55, ha="right", fontsize=9)
    ax_d.set_xlim(-0.7, len(order) - 0.3)
    ax_d.grid(axis="y", which="both", alpha=0.18)
    d0, d1 = stats["panel_d_topnorm_control"]["eps0p5"], stats["panel_d_topnorm_control"]["eps1p0"]
    n_pos_05 = len(order) - d0["n_negative"]; n_pos_10 = len(order) - d1["n_negative"]
    ax_d.text(.01, .97,
              f"candidate > median top-norm control: {n_pos_05}/22 (ε=0.5), {n_pos_10}/22 (ε=1.0)\n"
              f"top-norm controls themselves near-inert: median effect "
              f"{pct(d0['median_topnorm_control_relative_loss_change'], 3)} (ε=0.5), "
              f"{pct(d1['median_topnorm_control_relative_loss_change'], 3)} (ε=1.0)",
              transform=ax_d.transAxes, va="top", fontsize=10,
              bbox=dict(facecolor="white", alpha=.88, edgecolor="#cccccc"))
    d_legend = [Line2D([0],[0],marker='o',color='w',label=r'$\epsilon$=0.5 (filled)',markerfacecolor='#666',markeredgecolor='black',markersize=7),
                Line2D([0],[0],marker='o',color='w',label=r'$\epsilon$=1.0 (open)',markerfacecolor='none',markeredgecolor='#666',markersize=7),
                Line2D([0],[0],marker='o',color='w',label='Decoder',markerfacecolor='#666',markersize=7),
                Line2D([0],[0],marker='s',color='w',label='Encoder',markerfacecolor='#666',markersize=7)]
    ax_d.legend(handles=d_legend, ncol=4, loc="upper right", fontsize=9.5)
    ax_d.set_title("D  Candidate vs. top-norm same-layer controls, both intervention strengths",loc="left",fontweight="bold")

    # round-3: suptitle removed -- the figure's headline lives in the manuscript caption,
    # not embedded in the image (matches the same change made to Fig. 1).
    fig.savefig(FIGDIR/"fig2_part2_functional_criticality.png",dpi=240)
    fig.savefig(FIGDIR/"fig2_part2_functional_criticality.pdf")
    plt.close(fig)


def write_packet(rows: list[dict], manifest: dict, stats: dict) -> None:
    lines=["# Part 2 functional-criticality evidence packet","",
           "Scope: evidence assembly for Results, Methods, and Figure 2; not polished manuscript prose. No new model response, causal intervention, or tomography was run.","",
           "## 1. Experiment and frozen cohort","",
           "The completed experiment contains 22 models, one frozen primary candidate per model, five independently scored same-layer controls, and ε∈{0.5,1.0}. Phi-3 is outside this 22-model analysis by explicit scope clarification; it is not missing. It is also the only manifest entry with a six-row pre-existing basis, whereas every included model has one primary row. Its prior tomography remains case-study material.","",
           "All 22 raw candidate coordinates match the corresponding primary coordinate in `candidate_manifest.json` exactly; all five raw control coordinates match the frozen control set exactly.","",
           "| Model | Checkpoint / requested revision | Resolved revision | Type | Candidate | Candidate provenance | Five controls |",
           "|---|---|---|---|---:|---|---|" ]
    for r in rows:
        lines.append(f"| {r['model']} | `{r['repo']}` @ `{r['requested_revision']}` | `{r['resolved_revision']}` | {r['domain']}/{r['architecture']} | L{r['candidate_layer']}/r{r['candidate_row']} | `{r['candidate_source']}` | L{r['candidate_layer']}: {r['control_rows']} |")
    lines += ["","Control selection: `SeedSequence(42).spawn(23)[panel_index]`; sample five distinct rows without replacement from `0..d_model-1`, excluding every candidate in that model/layer, then sort. One deterministic stream per model; Phi-3 consumes its stream across layers 2 then 4. Source: `build_candidate_manifest.py` and `candidate_manifest.json`.","",
              "### DNABERT-2 selection versus evaluation provenance","",
              "The canonical discovery path for L5/r603 used the 504-bp ACTB probe with tokenizer default special tokens and exactly reproduces historical `out_max=944.5556030273438` (ratio 152.734548, activation rank 1). The later no-special-token structural census is not used to invalidate it. The Part 2 causal endpoint is separate: 256 seed-42 hg38 windows, fixed masked-token batches. Candidate-selection ACTB preprocessing and causal-evaluation hg38 preprocessing are not the same assay.","",
              "## 2. Exact intervention and endpoints","",
              "For one candidate or one control at a time, clone down-projection row `w`, set `w_ε=(1-ε)w`, evaluate, and restore the cloned row in a context-manager/finally path. Thus ε=0.5 means 50% row scaling and ε=1 means a zero row. Models are in eval mode and loaded/evaluated in float32; no autocast is declared. Candidate and controls are never masked jointly.","",
              "| Group | Evaluation units and preprocessing | Endpoint | Exact implemented batch/unit structure |",
              "|---|---|---|---|",
              "| Text decoders (n=10) | WikiText-2-raw-v1 test; nonempty lines shuffled with `random.Random(42)`, concatenated and each model tokenized into 100 exactly-512-token windows. | Teacher-forced shifted-label mean token NLL: logits `[:-1]`, labels `[1:]`, summed NLL divided by predicted-token count. | Batch 4 for loaded models >3B parameters (25 bootstrap units); otherwise batch 8 (13 units, final partial batch). |",
              "| Text encoders (n=6) | Same WikiText construction, 256 × 512-token windows; batch 16. One fixed mask realization, seed 42, probability .15; CLS/SEP/PAD excluded; selected tokens replaced by MASK and all other labels set to -100. | Summed cross-entropy over masked tokens divided by masked-token count. | 16 fixed MLM batches; identical inputs/masks reused for baseline and every condition. MosaicBERT uses `bert-base-uncased` tokenizer; others use checkpoint tokenizers. |",
              "| Genomic decoders (n=4) | hg38 `random_262kb.bed`; seed-42 disjoint partition; 100 damage windows of 512 bp, <1% N. GENERator trims the left `len%6` bases, prepends BOS, then tokenizes with specials disabled. GenomeOcean does no 6-bp trim or forced BOS and tokenizes with specials disabled. | Teacher-forced shifted causal-LM mean token NLL (`labels=input_ids`; internal label shift), weighted over tokens. | 100 individual-window bootstrap units. The separately constructed 96-window prompt pool is not used in this endpoint. |",
              "| Genomic encoders (n=2) | hg38 FASTA + `random_262kb.bed`; regions shuffled and starts sampled with `random.Random(42)`; 256 windows of 600 bp, <1% N; tokenizer padding/truncation to 256 tokens; batch 16. Fixed .15 masks with seed 42; special/PAD excluded. | Masked-nucleotide summed loss divided by masked-token count. | 16 fixed MLM batches. DNABERT-2 uses the pinned pretrained MLM/eager compatibility loader; NTv3 uses its pinned remote code revision. |","",
              "Raw `endpoint` metadata and actual unit counts are preserved model-by-model in `part2_22_model_results.csv` and `results/experiments/E13/raw/*.json`.","",
              "Checkpoint provenance note: NTv3's weight revision remains recorded as unpinned (`E5/E6 original`); only its required remote-code loader is pinned to commit `0ecff3637f0d3ba5b686d1095083218157c2ca34` (matches `NTV3_CODE_REVISION` in `genomic_encoder_lib.py`). That is a provenance limitation, not an exact weight revision. For models requested without a revision, the resolved Hub commit recorded by the completed run is reported in the table above.","",
              "## 3. Effect definitions","",
              "For evaluation unit j, raw records store `(S_j,N_j)`: summed loss and contributing-token count. `L0=ΣS0j/ΣN0j`; `Lε=ΣSεj/ΣNεj`; absolute change `ΔL=Lε-L0`; signed relative change `R=(Lε-L0)/L0`; percent change is `100R`. Candidate effect is R for the frozen candidate. Each control effect is its independently measured R. Same-layer median control is the median of five signed control R values. Candidate-minus-control is `G=Rcandidate-median(Rcontrol,1..5)`.","",
              "The 18/22 and 20/22 counts use `G>0`. Cohort +0.60%/+0.88% summaries are medians of G across 22 models. The q1 correlation and Figure 2C y-axis use signed full-ablation candidate R—not G and not an absolute value. Figure 2A/B show signed candidate and individual-control R.","",
              "## 4. Recomputed cohort results","" ]
    for tag,label in [("eps0p5","ε=0.5"),("eps1p0","ε=1.0")]:
        s=stats[tag]; lo,hi=s['candidate_minus_control_ci95']
        lines += [f"### {label}","",f"- Candidate > median control: **{s['candidate_gt_median_control']}/22**.",f"- Median candidate effect: **{pct(s['median_candidate'])}**.",f"- Median median-control effect: **{pct(s['median_control'])}**.",f"- Median candidate-minus-control: **{pct(s['median_candidate_minus_control'])}**.",f"- 95% model-bootstrap percentile CI for median gap: **{pct(lo)} to {pct(hi)}** (5,000 draws; seed {s['seed']}).","",
                  "Sorted candidate effects (weakest to strongest):","", "| Model | Signed effect |","|---|---:|"]
        for x in s['sorted_candidate_effects']: lines.append(f"| {x['model']} | {pct(x['effect'])} |")
        lines.append("")
    cq=stats['correlations']['q1']; cf=stats['correlations']['frob_ratio_to_layer_median']
    lines += ["## 5. Structure-function associations","",
              f"q1 versus signed full-ablation candidate R: Spearman ρ={cq['spearman_rho']:.9f}, asymptotic two-sided p={cq['asymptotic_p']:.9f}, 5,000-model-bootstrap percentile CI [{cq['bootstrap_ci95'][0]:.9f},{cq['bootstrap_ci95'][1]:.9f}], seed {cq['seed']}, n=22.","",
              f"Layer-relative Frobenius magnitude: ρ={cf['spearman_rho']:.9f}, p={cf['asymptotic_p']:.9f}, CI [{cf['bootstrap_ci95'][0]:.9f},{cf['bootstrap_ci95'][1]:.9f}], seed {cf['seed']}, n=22.","",
              "Repository search found no completed leave-one-model-out, leave-one-family-out, within-group, covariate-adjusted, or influence diagnostic for the Frobenius association. Therefore it is not promoted to Figure 2; it remains a fixed-panel secondary association requiring robustness work before headline use.","",
              "## 6. Exact uncertainty procedures","",
              "Per-condition CI: 5,000 percentile draws. A single resampled index vector selects paired baseline and perturbed units jointly, then each loss is recomputed as total summed loss / total contributing tokens and converted to R. Units are actual stored batches for text decoders and both encoder groups, and individual windows for genomic decoders. Condition RNG streams come sequentially from `SeedSequence(42).spawn(1000)` in frozen panel/condition order.","",
              "Cohort gap CI: resample the 22 model-level G values with replacement, take the median per draw; 5,000 percentile draws, seeds 47 (.5) and 52 (1.0). Correlation: SciPy `spearmanr` on 22 model rows; default asymptotic two-sided p. CI resamples 22 model indices with replacement, recomputes rho, drops only nonfinite bootstrap replicates, and takes percentiles; seeds 43 (q1) and 44 (Frobenius). Negative and zero effects are retained unchanged.","",
              "## 7. Descriptive subgroup medians","" ]
    for k,v in stats['descriptive_subgroup_medians'].items(): lines.append(f"- {k}: n={v['n']}, median signed full-ablation candidate effect **{pct(v['median'])}**.")
    lines += ["","These are descriptive only. Objectives, architectures, domains, tokenizers, and panel composition are confounded; no architecture/domain determination claim is licensed.","",
              "## 8. Figure 2 deliverables","",
              "Panels A/B use a signed symmetric-log axis (`linthresh=0.1 percentage points`), retaining negative, sub-percent, and catastrophic effects. Diamonds are candidates; gray points are all five controls; short bars are within-model control medians. Models share one order based on full-ablation candidate effect. Panel C uses q1 and the exact signed full-ablation candidate R used in the correlation. No Frobenius panel is included because robustness analyses do not exist.","",
              "- `results/experiments/E13/figure2_candidate_control_data.csv`","- `results/experiments/E13/figure2_structure_function_data.csv`","- `experiments/figures/main/fig2_part2_functional_criticality.png`","- `experiments/figures/main/fig2_part2_functional_criticality.pdf`","",
              "## Claims directly supported by the completed experiment","",
              "- Frozen candidates exceed same-layer median controls in 18/22 models at ε=.5 and 20/22 at ε=1.","- The cohort-median signed candidate-minus-control gap is positive at both strengths with model-bootstrap CIs above zero.","- Full-ablation candidate effects span negative/sub-percent values through approximately +699%.","- q1 does not predict full-ablation causal magnitude in this 22-model panel; its CI spans substantial negative and positive correlations.","- The measured candidate effect is heterogeneous across models and native objectives.","",
              "## Claims not supported / caveats","",
              "- No universal effect: NTv3 and EuroBERT-2.1B have nonpositive full-ablation candidate-minus-control gaps.","- No claim that architecture or domain causes the descriptive subgroup differences.","- Native-objective relative losses are useful within each model but objectives/tokenizers differ across groups.","- No downstream-task mechanism, interaction order, or cohort-wide tomography conclusion follows from singleton effects.","- The Frobenius association lacks existing influence/covariate/leave-group robustness analyses and is not a headline result.","- DNABERT-2 discovery is input-boundary dependent; its ACTB discovery preprocessing must not be conflated with its hg38 causal endpoint.","",
              "## Remaining manuscript issues","",
              "- Decide how briefly to disclose that the original DNABERT-2 wrapper recorded revision `main`; the pinned state reproduces its stored activation exactly, but the original resolved commit/environment were not recorded.","- Replace the old five-decoder/tomography Figure 2 references and captions with this 22-model singleton census; retain tomography only as mechanistic case-study material.","- Keep the Frobenius result secondary unless a separately authorized robustness analysis is completed.","- Universal Part 2B tomography remains stopped.","",
              "## Primary provenance paths","","- `results/experiments/E13/candidate_manifest.json`","- `results/experiments/E11/scale_ladder_backfilled.csv`","- `results/experiments/E13/raw/*.json`","- `results/experiments/E13/part1_22_candidate_effects.csv`","- `results/experiments/E13/part1_22_control_effects.csv`","- `results/experiments/E13/part1_22_cohort_summary.csv`","- `results/experiments/E13/part1_22_structure_function_correlations.json`","- `experiments/frozen/E13_full_cohort_causal_census/run_singleton_census.py`","- `experiments/frozen/E13_full_cohort_causal_census/write_part1_22_report.py`","- `experiments/frozen/E10_nlp_architecture_causal/e10_lib.py`","- `experiments/frozen/E9_mechanistic_tomography/tomography_lib.py`","- `experiments/frozen/E12_generator_degradation_control/e12_lib.py`","- `results/experiments/E13_dnabert2_reproducibility/DNABERT2_EXECUTION_PATH_DIAGNOSTIC.md`",""]
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__": main()
