#!/usr/bin/env python3
"""Publication-quality E13 Figures A-C from compiled, frozen tables."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
E13=ROOT/"results"/"E13"
FIG=E13/"figures"
COLORS={("text","decoder"):"#4477AA",("text","encoder"):"#EE6677",
        ("genomic","decoder"):"#228833",("genomic","encoder"):"#CCBB44"}


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight")
    fig.savefig(FIG/f"{name}.png",dpi=300,bbox_inches="tight")
    plt.close(fig)


def figure_a(candidates):
    rows=[r for r in candidates if float(r["epsilon"])==1.0 and r["primary"]=="True"]
    fig,axs=plt.subplots(1,2,figsize=(10.2,4.2),constrained_layout=True)
    for ax,xkey,xlabel,logx in [(axs[0],"q1",r"Structural concentration $q_1$",False),
                                (axs[1],"frob_ratio_to_layer_median",
                                 r"Layer-relative $\|U_k\|_F$",True)]:
        for r in rows:
            key=(r["domain"],r["architecture"])
            ax.scatter(float(r[xkey]),100*float(r["relative_loss_change"]),s=45,
                       color=COLORS[key],edgecolor="white",linewidth=.5,zorder=3)
        if logx:ax.set_xscale("log")
        ax.axhline(0,color="0.6",lw=.8)
        ax.set_xlabel(xlabel); ax.set_ylabel("Full-ablation relative loss change (%)")
        ax.grid(alpha=.18)
    handles=[]
    for key,color in COLORS.items():
        handles.append(plt.Line2D([],[],marker="o",ls="",color=color,label=f"{key[0]} {key[1]}"))
    axs[0].legend(handles=handles,frameon=False,fontsize=8)
    save(fig,"figure_A_structure_vs_causal")


def figure_b(summary):
    n=len(summary); y=np.arange(n)
    fig,ax=plt.subplots(figsize=(8.3,9.4),constrained_layout=True)
    for eps,label,marker,off in [(0.5,"ε=0.5","o",-.13),(1.0,"ε=1.0","s",.13)]:
        tag="eps0p5" if eps==.5 else "eps1p0"
        x=[100*float(r[f"candidate_minus_median_control_effect_{tag}"]) for r in summary]
        ax.scatter(x,y+off,label=label,marker=marker,s=32,zorder=3)
    ax.axvline(0,color="0.45",lw=.9)
    ax.set_yticks(y,labels=[r["model"] for r in summary],fontsize=7.5)
    ax.invert_yaxis(); ax.set_xlabel("Strongest candidate effect − median same-layer control (percentage points)")
    ax.grid(axis="x",alpha=.2); ax.legend(frameon=False)
    save(fig,"figure_B_full_cohort_census")


def figure_c(summary):
    n=len(summary); y=np.arange(n)
    fig,ax=plt.subplots(figsize=(8.3,9.4),constrained_layout=True)
    for i,r in enumerate(summary):
        if r["interaction_identifiable"] != "True":
            ax.text(-4,i,"N/I",ha="center",va="center",fontsize=6.5,color="0.55")
            continue
        for tag,marker,off,label in [("eps0p5","o",-.13,"ε=0.5"),("eps1p0","s",.13,"ε=1.0")]:
            x=100*float(r[f"F2_to_F3_relative_mae_improvement_{tag}"])
            lo=100*float(r[f"bootstrap_ci95_low_{tag}"]); hi=100*float(r[f"bootstrap_ci95_high_{tag}"])
            ax.errorbar(x,i+off,xerr=[[x-lo],[hi-x]],fmt=marker,capsize=3,label=label if i==3 else None)
    ax.axvline(0,color="0.45",lw=.9)
    ax.set_yticks(y,labels=[r["model"] for r in summary],fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("Held-out F2→F3 relative MAE improvement (%)")
    ax.text(.99,.01,"N/I = not identifiable from one-component basis",transform=ax.transAxes,
            ha="right",va="bottom",fontsize=7,color="0.4")
    ax.grid(axis="x",alpha=.2); ax.legend(frameon=False)
    save(fig,"figure_C_interaction_complexity")


def main():
    candidates=read(E13/"candidate_effects.csv")
    summary=read(E13/"cohort_summary.csv")
    assert len(summary)==23
    figure_a(candidates); figure_b(summary); figure_c(summary)
    print(f"wrote PDF+PNG Figures A-C to {FIG}")


if __name__=="__main__":main()
