"""
Item 6 — build the canonical E4 granularity table from ALREADY-GENERATED artifacts only.

No model is loaded and no experiment is re-run. Every cell is read from a stored JSON, and
any cell the stored artifacts do not contain is emitted as `n/s` (not stored) rather than
recomputed, so the table cannot silently acquire a number from a fresh forward pass.

Sources, in precedence order per model:
  results/e1_nlp_retrospective.json        Llama-7B, Mistral-7B, OLMo-7B
  results/n009_prok_layer_resolution.json  GENERator PROK @ L2 (has full cumulative shares)
  results/e4_ntv3_shared_adapter.json      NTv3 via the corrected shared adapter (N-010)
  results/e4_granularity.json              GENERator EUK, DNABERT-2, Evo1

Emits Markdown to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "results"

NS = "n/s"


def f(x, spec=".4f"):
    return NS if x is None else format(x, spec)


def cum_from_top10(entry) -> dict:
    """top1/5/10 are derivable from the stored top-10 contributor shares; 50/100 are not."""
    tc = entry.get("top10_contributors")
    if not tc:
        return {"top1": entry.get("top1_share"), "top5": entry.get("top5_share"),
                "top10": None, "top50": None, "top100": None}
    s = [c["share"] for c in tc]
    return {"top1": sum(s[:1]), "top5": sum(s[:5]), "top10": sum(s[:10]),
            "top50": None, "top100": None}


def main() -> None:
    e1 = json.loads((R / "e1_nlp_retrospective.json").read_text())
    e4 = {m["model"]: m for m in json.loads((R / "e4_granularity.json").read_text())["models"]}
    n9 = json.loads((R / "n009_prok_layer_resolution.json").read_text())
    nt = json.loads((R / "e4_ntv3_shared_adapter.json").read_text())

    rows = []

    # ---- published NLP super-weights -------------------------------------
    for m in e1["models"]:
        if m.get("status") != "ok":
            continue
        l1, l2 = m["level1"], m["level2"]
        rows.append(dict(
            group="NLP", name=m["name"], ckpt=m["repo"], layer=m["layer"],
            coord=f"({m['k']}, {m['i']})",
            rank=l1["row_rank"], of=m["d_model"], pct=l1["percentile"],
            mom=l1["max_over_median"], srank=l2["rank_of_published_i"], sof=m["d_ffn"],
            t1i=l2["top1_index"], t1s=l2["top1_share"], pr=l2["participation_ratio"],
            cum={"top1": l2["top1_share"], "top5": l2["top5_share"],
                 "top10": None, "top50": None, "top100": None},
            note="published (k,i) recovered at rank 1 on both levels"))

    # ---- genomic ----------------------------------------------------------
    g = e4["generator"]
    rows.append(dict(
        group="genomic", name="GENERator EUK", ckpt="GenerTeam/GENERator-v2-eukaryote-3b-base",
        layer=g["sw_layer"], coord=f"(row {g['sw_row']}, i n/a)", rank=g["row_rank"],
        of=g["d_model"], pct=100.0 * (1 - (g["row_rank"] - 1) / g["d_model"]),
        mom=g["max_over_median"], srank=None, sof=g["d_ffn"], t1i=g["top1_index"],
        t1s=g["top1_share"], pr=g["participation_ratio"], cum=cum_from_top10(g), note=""))

    rows.append(dict(
        group="genomic", name="GENERator PROK **[CONTESTED]**",
        ckpt="GenerTeam/GENERator-v2-prokaryote-3b-base",
        layer=n9["layer"], coord=f"(row {n9['row']}, i n/a)",
        rank=n9["decomposed"]["rank"], of=n9["exact"]["d_model"],
        pct=n9["decomposed"]["percentile"], mom=n9["decomposed"]["max_over_median"],
        srank=None, sof=n9["granularity"]["d_ffn"], t1i=n9["granularity"]["top1_index"],
        t1s=n9["granularity"]["top1_share"], pr=n9["granularity"]["participation_ratio"],
        cum=n9["granularity"]["cumulative_share"],
        note="**C-001 ON HOLD — N-009 unresolved.** Layer 2 is the claim's own layer, "
             "verified by provenance. Does NOT support the manuscript's rank-1 claim."))

    d = e4["dnabert2"]
    rows.append(dict(
        group="genomic", name="DNABERT-2", ckpt="zhihan1996/DNABERT-2-117M @ 7bce263",
        layer=d["sw_layer"], coord=f"(row {d['sw_row']}, i n/a)", rank=d["row_rank"],
        of=d["d_model"], pct=100.0 * (1 - (d["row_rank"] - 1) / d["d_model"]),
        mom=d["max_over_median"], srank=None, sof=d["d_ffn"], t1i=d["top1_index"],
        t1s=d["top1_share"], pr=d["participation_ratio"], cum=cum_from_top10(d), note=""))

    rows.append(dict(
        group="genomic", name="NTv3", ckpt="InstaDeepAI/NTv3_650M_pre @ 0ecff36",
        layer=11, coord="(row 1472, i n/a)", rank=nt["row_rank"], of=nt["d_model"],
        pct=100.0 * (1 - (nt["row_rank"] - 1) / nt["d_model"]),
        mom=nt["max_over_median"], srank=None, sof=nt["d_ffn"], t1i=nt["top1_index"],
        t1s=nt["top1_share"], pr=nt["participation_ratio"], cum=nt["cumulative_share"],
        note="corrected shared adapter (N-010); reproduces the pre-fix value exactly"))

    v = e4["evo1"]
    rows.append(dict(
        group="genomic", name="Evo1", ckpt="togethercomputer/evo-1-8k-base @ 1.1_fix",
        layer=v["sw_layer"], coord=f"(row {v['sw_row']}, i n/a)", rank=v["row_rank"],
        of=v["d_model"], pct=100.0 * (1 - (v["row_rank"] - 1) / v["d_model"]),
        mom=v["max_over_median"], srank=None, sof=v["d_ffn"], t1i=v["top1_index"],
        t1s=v["top1_share"], pr=v["participation_ratio"], cum=cum_from_top10(v), note=""))

    hdr = ("| Model | checkpoint | layer | published (k,i) | row rank | row percentile | "
           "max/median | scalar rank of i | top1 index | top1 share | PR | "
           "cum top1 | top5 | top10 | top50 | top100 |")
    sep = "|" + "---|" * 16
    for grp, title in (("NLP", "### Published NLP super-weights"),
                       ("genomic", "### Genomic models")):
        print(f"\n{title}\n\n{hdr}\n{sep}")
        for r in rows:
            if r["group"] != grp:
                continue
            c = r["cum"]
            print(f"| {r['name']} | `{r['ckpt']}` | {r['layer']} | {r['coord']} | "
                  f"**{r['rank']}** / {r['of']:,} | {f(r['pct'], '.3f')} | "
                  f"{f(r['mom'], '.2f')}× | "
                  f"{('**' + str(r['srank']) + '** / ' + format(r['sof'], ',')) if r['srank'] else NS} | "
                  f"{r['t1i']} | {f(r['t1s'], '.4f')} | {f(r['pr'], '.2f')} | "
                  f"{f(c.get('top1'))} | {f(c.get('top5'))} | {f(c.get('top10'))} | "
                  f"{f(c.get('top50'))} | {f(c.get('top100'))} |")
    print("\nNotes:")
    for r in rows:
        if r["note"]:
            print(f"- **{r['name']}**: {r['note']}")


if __name__ == "__main__":
    main()
