#!/usr/bin/env/env python3
"""Revised structural Figure 1, sourced only from final artifacts.

Panel C was extended in round-2 (audit/round2/final_check.md Section 2) from the original
12-model E11 panel to the full 22-model cohort, and split into two side-by-side sub-panels
(random same-layer controls / top-norm same-layer controls) sharing one "C" label -- see
audit/round2/scripts/fig1_full22_two_panel.py for the standalone version this was folded
back into the unified Fig. 1. Panels A, B, D are unchanged (12-model E11 panel; D was not
in scope for the round-2 update).
"""
from pathlib import Path
import csv, json, sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as ml

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent.parent; RESULTS=ROOT/'results'; AUDIT2=ROOT/'audit'/'round2'
sys.path.insert(0,str(ROOT/'scripts'/'analysis')); sys.path.insert(0,str(HERE))
from _figstyle import apply_style, panel_label
from _paper_encoding import DOMAIN_COLOR, ARCH_MARKER

ARCH={m:'decoder' for m in ['Qwen2.5-0.5B','Qwen2.5-1.5B','Qwen2.5-3B','SmolLM2-135M','SmolLM2-360M','SmolLM2-1.7B','GENERator-PROK-1.2B','GENERator-PROK-3B']}
ARCH.update({m:'encoder' for m in ['EuroBERT-210M','EuroBERT-610M','EuroBERT-2.1B','ModernBERT-large']})
DOMAIN={m:('genomic' if 'PROK' in m else 'text') for m in ARCH}

def key(name):
    return {'Qwen/Qwen2.5-0.5B':'Qwen2.5-0.5B','Qwen/Qwen2.5-1.5B':'Qwen2.5-1.5B','Qwen/Qwen2.5-3B':'Qwen2.5-3B',
            'HuggingFaceTB/SmolLM2-135M':'SmolLM2-135M','HuggingFaceTB/SmolLM2-360M':'SmolLM2-360M','HuggingFaceTB/SmolLM2-1.7B':'SmolLM2-1.7B',
            'GenerTeam/GENERator-v2-prokaryote-1.2b-base':'GENERator-PROK-1.2B','GenerTeam/GENERator-v2-prokaryote-3b-base':'GENERator-PROK-3B',
            'EuroBERT/EuroBERT-210m':'EuroBERT-210M','EuroBERT/EuroBERT-610m':'EuroBERT-610M','EuroBERT/EuroBERT-2.1B':'EuroBERT-2.1B',
            'answerdotai/ModernBERT-large':'ModernBERT-large'}.get(name,name)

def display(name):
    return {'GENERator-EUK-3B':'GEN-EUK-3B','GenomeOcean-4B':'GenomeOcean-4B',
            'DNABERT-2':'DNABERT-2','NTv3':'NTv3','Phi-3-mini-4k-instruct':'Phi-3-mini',
            'Qwen/Qwen2.5-7B':'Qwen2.5-7B'}.get(name,key(name))

def rows(name):
    with (RESULTS/'E11'/name).open() as f: return list(csv.DictReader(f))

# ---- full-22 display-name map for panel C (round-2 extension) --------------------------
DISPLAY22 = {
    "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B", "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
    "HuggingFaceTB/SmolLM2-135M": "SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M": "SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B": "SmolLM2-1.7B",
    "GenerTeam/GENERator-v2-prokaryote-1.2b-base": "GEN-PROK-1.2B", "GenerTeam/GENERator-v2-prokaryote-3b-base": "GEN-PROK-3B",
    "EuroBERT/EuroBERT-210m": "EuroBERT-210M", "EuroBERT/EuroBERT-610m": "EuroBERT-610M", "EuroBERT/EuroBERT-2.1B": "EuroBERT-2.1B",
    "answerdotai/ModernBERT-large": "ModernBERT-large", "ModernBERT-base": "ModernBERT-base", "DNABERT-2": "DNABERT-2",
    "GENERator-EUK-3B": "GEN-EUK-3B", "Llama-7B": "Llama-7B", "Mistral-7B": "Mistral-7B", "OLMo-7B-0724-hf": "OLMo-7B",
    "MosaicBERT": "MosaicBERT", "GenomeOcean-4B": "GenomeOcean-4B", "Qwen2.5-7B": "Qwen2.5-7B", "NTv3": "NTv3",
}
NEGATIVE_TOPNORM_MODELS = {
    "HuggingFaceTB/SmolLM2-360M", "EuroBERT/EuroBERT-210m", "EuroBERT/EuroBERT-2.1B",
    "answerdotai/ModernBERT-large", "DNABERT-2", "GENERator-EUK-3B", "GenomeOcean-4B",
}

def load_panel_c22():
    summary = list(csv.DictReader(open(AUDIT2/'fig1c_random_vs_topk_gaps_full22.csv')))
    assert len(summary) == 22
    meta = {r['model']: r for r in summary}
    random_ctrl = defaultdict(list)
    for r in csv.DictReader(open(RESULTS/'E11'/'scale_ladder_controls.csv')):
        random_ctrl[r['model']].append(float(r['q1']))
    for r in csv.DictReader(open(AUDIT2/'section2_random_control_q1_batch2.csv')):
        random_ctrl[r['model']].append(float(r['q1']))
    topnorm_ctrl = defaultdict(list)
    for r in csv.DictReader(open(AUDIT2/'topk_norm_q1_summary.csv')):
        if r['coord_role'] != 'candidate':
            topnorm_ctrl[r['model']].append(float(r['q1']))
    order = sorted(meta, key=lambda m: -float(meta[m]['random_control_gap']))
    return meta, random_ctrl, topnorm_ctrl, order

def main():
    apply_style(); all_rows=rows('scale_ladder.csv'); cohort=[r for r in all_rows if r['source_type']=='measured']; controls=rows('scale_ladder_controls.csv')
    full=[r for r in all_rows if r.get('q1','').lower() not in {'','nan','none'}]
    for r in full: r['q1']=float(r['q1']); r['display']=display(r['model'])
    for r in cohort: r['model']=key(r['model'])
    by={r['model']:[] for r in cohort}
    for r in controls: by[key(r['model'])].append(float(r['q1']))
    for r in cohort: r['q1']=float(r['q1']); r['gap']=r['q1']-np.mean(by[r['model']])
    cohort.sort(key=lambda r:r['q1'],reverse=True); full.sort(key=lambda r:r['q1'],reverse=True)

    fig = plt.figure(figsize=(12.4, 10.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.15, 1.0], hspace=0.55, wspace=0.28)
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
    cL = fig.add_subplot(gs[1, 0]); cR = fig.add_subplot(gs[1, 1], sharey=cL)
    d = fig.add_subplot(gs[2, :])

    # A: retrospective diagonal calibration only.
    old=json.loads((RESULTS/'e1_nlp_retrospective.json').read_text())['models']; old=[m for m in old if m['name'] in {'Llama-7B','Mistral-7B','OLMo-7B'}]; y=np.arange(3)
    a.barh(y,[m['level2']['top1_share'] for m in old],color=DOMAIN_COLOR['text'],edgecolor='black',height=.52); a.set_yticks(y); a.set_yticklabels([m['name'] for m in old]); a.set_xlim(0,1.08)
    for yy,m in zip(y,old): a.text(m['level2']['top1_share']+.015,yy,f"rank {m['level1']['row_rank']}",va='center',fontsize=7)
    a.set_xlabel('published scalar top-1 share\n(diagonal calibration only)'); panel_label(a,'A')

    # B: exact q1 cohort, without a threshold.
    xs=np.arange(len(full))
    for x,r in zip(xs,full): b.scatter(x,r['q1'],s=42,marker=ARCH_MARKER[r['architecture']],color=DOMAIN_COLOR[r['domain']],edgecolor='black',zorder=3)
    labels=[r['display'] for r in full]; b.set_xticks(xs); b.set_xticklabels(labels,rotation=58,ha='right',fontsize=6.4); b.set_ylabel(r'exact $q_1$'); b.set_ylim(.35,1.03); panel_label(b,'B')
    b.legend(handles=[ml.Line2D([],[],marker='o',color='.35',linestyle='',label='decoder'),ml.Line2D([],[],marker='s',color='.35',linestyle='',label='encoder'),ml.Line2D([],[],marker='o',color=DOMAIN_COLOR['text'],linestyle='',label='text'),ml.Line2D([],[],marker='o',color=DOMAIN_COLOR['genomic'],linestyle='',label='genomic')],frameon=False,fontsize=6,ncol=2,loc='lower left')
    b.text(.02,.72,'Evo2-7B: no candidate\n(ratio 2.22 < 5.0)',transform=b.transAxes,ha='left',va='bottom',fontsize=6,color='.35')

    # C (round-2, full 22-model cohort): candidate vs. random controls | vs. top-norm controls.
    meta, random_ctrl, topnorm_ctrl, order = load_panel_c22()
    xs_c = np.arange(len(order))
    neg_offsets = [7, 20, 33]  # stagger heights so adjacent negative-gap labels don't collide
    neg_i = 0
    for x, m in zip(xs_c, order):
        r = meta[m]; arch, dom = r['architecture'], r['domain']; cand_q1 = float(r['candidate_q1'])
        cL.scatter(np.full(5,x)+np.linspace(-.16,.16,5), random_ctrl[m], s=20, facecolors='none', edgecolors=DOMAIN_COLOR[dom], linewidths=.7, zorder=3)
        cL.scatter(x, cand_q1, s=50, marker=ARCH_MARKER[arch], color=DOMAIN_COLOR[dom], edgecolor='black', zorder=4)
        cR.scatter(np.full(5,x)+np.linspace(-.16,.16,5), topnorm_ctrl[m], s=20, facecolors='none', edgecolors=DOMAIN_COLOR[dom], linewidths=.7, zorder=3)
        cR.scatter(x, cand_q1, s=50, marker=ARCH_MARKER[arch], color=DOMAIN_COLOR[dom], edgecolor='black', zorder=4)
        if m in NEGATIVE_TOPNORM_MODELS:
            cR.annotate(DISPLAY22[m], (x, cand_q1), xytext=(0, neg_offsets[neg_i % len(neg_offsets)]), textcoords='offset points',
                        ha='center', va='bottom', fontsize=4.8, color='0.2',
                        arrowprops=dict(arrowstyle='-', lw=0.4, color='0.4'))
            neg_i += 1
    c_labels = [DISPLAY22[m] for m in order]
    for ax in (cL, cR):
        ax.set_xticks(xs_c); ax.set_xticklabels(c_labels, rotation=62, ha='right', fontsize=5.2)
        ax.set_xlim(-0.7, len(order)-0.3)
    cL.set_ylim(0, 1.18); cL.set_yticks([0,.2,.4,.6,.8,1.0]); cL.set_ylabel(r'$q_1$ (candidate and same-layer controls)')
    plt.setp(cR.get_yticklabels(), visible=False)
    cL.set_title('random same-layer controls', fontsize=8, fontweight='normal', loc='left', pad=3)
    cR.set_title('top-norm same-layer controls', fontsize=8, fontweight='normal', loc='left', pad=3)
    med_rc = np.median([float(meta[m]['random_control_gap']) for m in order])
    med_tk = np.median([float(meta[m]['topk_by_norm_gap']) for m in order])
    cL.text(.02,.99, f'median gap = {med_rc:.3f}\nn = 22', transform=cL.transAxes, fontsize=6.5, va='top', ha='left')
    cR.text(.02,.99, f'median gap = {med_tk:.3f}\nn = 22, 7 negative', transform=cR.transAxes, fontsize=6.5, va='top', ha='left')
    cL.legend(handles=[ml.Line2D([],[],marker='o',color='black',linestyle='',label='candidate',markerfacecolor='.35'),ml.Line2D([],[],marker='o',color='.35',linestyle='',label='5 controls',markerfacecolor='none')],frameon=False,fontsize=6,loc='lower left')
    panel_label(cL, 'C')  # single spanning label for the C1/C2 pair, per journal convention

    # D: local gap versus scale (unchanged, 12-model E11 panel; not in round-2 scope).
    offsets=[(4,5),(4,-10),(4,5),(4,-10),(4,5),(4,-10),(4,5),(4,-10),(4,5),(4,-10),(4,5),(4,-10)]
    for i,r in enumerate(cohort):
        d.scatter(float(r['non_embed_params']),r['gap'],s=62,marker=ARCH_MARKER[ARCH[r['model']]],color=DOMAIN_COLOR[DOMAIN[r['model']]],edgecolor='black')
        d.annotate(r['model'].replace('GENERator-','GEN-'),(float(r['non_embed_params']),r['gap']),xytext=offsets[i],textcoords='offset points',fontsize=5.5)
    d.set_xscale('log'); d.set_xlabel('non-embedding parameters'); d.set_ylabel(r'candidate $q_1$ − mean control $q_1$'); d.set_ylim(0,1.02); panel_label(d,'D')

    fig.suptitle('High-gain gated-FFN rows are locally spectrally exceptional',fontsize=11)
    out=HERE/'main'/'fig1_structural'; fig.savefig(out.with_suffix('.pdf')); fig.savefig(out.with_suffix('.png'),dpi=300)

    src=HERE/'source_data'; src.mkdir(exist_ok=True)
    with (src/'fig1_structural_panel_a.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','rank','published_top1_share']); w.writeheader(); w.writerows({'model':m['name'],'rank':m['level1']['row_rank'],'published_top1_share':m['level2']['top1_share']} for m in old)
    with (src/'fig1_structural_panel_b.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','display_name','architecture','domain','q1']); w.writeheader(); w.writerows({'model':r['model'],'display_name':r['display'],'architecture':r['architecture'],'domain':r['domain'],'q1':r['q1']} for r in full)
    with (src/'fig1_structural_panel_bcd.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','architecture','domain','non_embedding_params','candidate_q1','control_q1_values','mean_control_q1','candidate_control_gap']); w.writeheader()
        for r in cohort: w.writerow({'model':r['model'],'architecture':ARCH[r['model']],'domain':DOMAIN[r['model']],'non_embedding_params':r['non_embed_params'],'candidate_q1':r['q1'],'control_q1_values':';'.join(map(str,by[r['model']])), 'mean_control_q1':np.mean(by[r['model']]),'candidate_control_gap':r['gap']})
    # panel C source data already lives at paper-salvage/figures/source_data/fig1_panel_c_full22_source.csv
    # (written by audit/round2/scripts/fig1_full22_two_panel.py); not rewritten here.

    print(f'saved -> {out}.png / .pdf')

if __name__=='__main__': main()
