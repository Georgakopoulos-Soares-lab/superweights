"""
E10 pre-measurement protocol audit: was the intervention basis selected by q1 (WRONG --
q1 is internal spectral concentration / rank-1ness, an annotation) or by the exact operator
magnitude ||U_k||_F (CORRECT -- the pre-existing high-gain criterion, uk_frobenius.py)?

Re-ranks both encoders from the ALREADY-COMPUTED full ranking JSONs (which recorded
frob_norm = ||U_k||_F for every row, per spectral_lib.SpectralMetrics), under:

  (a) global ||U_k||_F  -- raw magnitude across all layers
  (b) within-layer ||U_k||_F rank + value/median ratio -- the convention
      uk_frobenius.layer_report actually uses (E1's Level-1 row recovery)
  (c) q1                -- what E10 wrongly used for selection

and reports where each model's E8 canonical detector row lands under each.
No model is loaded; no SVD is recomputed; this reads existing artifacts only.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = {"mosaicbert": (9, 287), "modernbert": (15, 251)}


def audit(name: str):
    p = ROOT / "results" / f"e10_encoder_row_ranking_{name}.json"
    d = json.loads(p.read_text())
    rows = d["full_ranking"]
    canon_layer, canon_row = CANONICAL[name]

    def find(rs, layer, row):
        for i, r in enumerate(rs):
            if r["layer"] == layer and r["row"] == row:
                return i, r
        raise KeyError

    # (a) global ||U_k||_F
    by_frob = sorted(rows, key=lambda r: r["frob_norm"], reverse=True)
    ci_frob, cr = find(by_frob, canon_layer, canon_row)

    # (b) within-layer ||U_k||_F ratio to that layer's median (layer_report convention)
    by_layer: dict[int, list] = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r)
    enriched = []
    for layer, rs in by_layer.items():
        med = statistics.median([x["frob_norm"] for x in rs])
        for x in rs:
            y = dict(x)
            y["layer_median_frob"] = med
            y["frob_over_layer_median"] = x["frob_norm"] / med if med > 0 else float("inf")
            y["within_layer_rank"] = None
            enriched.append(y)
    for layer, rs in by_layer.items():
        ordered = sorted([e for e in enriched if e["layer"] == layer],
                         key=lambda r: r["frob_norm"], reverse=True)
        for i, e in enumerate(ordered):
            e["within_layer_rank"] = i
    by_ratio = sorted(enriched, key=lambda r: r["frob_over_layer_median"], reverse=True)
    ci_ratio, cr2 = find(by_ratio, canon_layer, canon_row)

    # (c) q1 (what was wrongly used)
    by_q1 = sorted(rows, key=lambda r: r["q1"], reverse=True)
    ci_q1, _ = find(by_q1, canon_layer, canon_row)

    print(f"\n{'='*72}\n{name}  (canonical detector row L{canon_layer}/r{canon_row})\n{'='*72}")
    print(f"canonical rank under GLOBAL ||U_k||_F        : {ci_frob} of {len(rows)}")
    print(f"canonical rank under WITHIN-LAYER ratio      : {ci_ratio} of {len(rows)}"
          f"  (within its own layer: #{cr2['within_layer_rank']}, "
          f"{cr2['frob_over_layer_median']:.1f}x layer median)")
    print(f"canonical rank under q1 (WRONGLY USED)       : {ci_q1} of {len(rows)}")

    print(f"\n-- top-10 by GLOBAL ||U_k||_F (correct magnitude criterion) --")
    for i, r in enumerate(by_frob[:10]):
        mark = "  <-- E8 CANONICAL" if (r["layer"], r["row"]) == (canon_layer, canon_row) else ""
        print(f"  {i:2d}  L{r['layer']:>2}/r{r['row']:<4}  ||U_k||_F={r['frob_norm']:>10.3f}  "
              f"q1={r['q1']:.4f}{mark}")

    print(f"\n-- top-10 by WITHIN-LAYER ||U_k||_F / layer median (layer_report convention) --")
    for i, r in enumerate(by_ratio[:10]):
        mark = "  <-- E8 CANONICAL" if (r["layer"], r["row"]) == (canon_layer, canon_row) else ""
        print(f"  {i:2d}  L{r['layer']:>2}/r{r['row']:<4}  ratio={r['frob_over_layer_median']:>8.1f}x  "
              f"||U_k||_F={r['frob_norm']:>10.3f}  q1={r['q1']:.4f}{mark}")

    print(f"\n-- top-10 by q1 (what E10 WRONGLY froze) --")
    for i, r in enumerate(by_q1[:10]):
        mark = "  <-- E8 CANONICAL" if (r["layer"], r["row"]) == (canon_layer, canon_row) else ""
        print(f"  {i:2d}  L{r['layer']:>2}/r{r['row']:<4}  q1={r['q1']:.4f}  "
              f"||U_k||_F={r['frob_norm']:>10.3f}{mark}")

    # overlap between the wrong basis and the correct ones
    setq = {(r["layer"], r["row"]) for r in by_q1[:10]}
    setf = {(r["layer"], r["row"]) for r in by_frob[:10]}
    setr = {(r["layer"], r["row"]) for r in by_ratio[:10]}
    print(f"\noverlap q1-top10 vs global-frob-top10 : {len(setq & setf)}/10")
    print(f"overlap q1-top10 vs within-layer-top10: {len(setq & setr)}/10")

    out = dict(model=name, canonical=dict(layer=canon_layer, row=canon_row),
               canonical_rank_global_frob=ci_frob, canonical_rank_within_layer_ratio=ci_ratio,
               canonical_rank_q1=ci_q1,
               top10_global_frob=[{k: r[k] for k in ("layer", "row", "frob_norm", "q1")}
                                   for r in by_frob[:10]],
               top10_within_layer_ratio=[{k: r[k] for k in
                                           ("layer", "row", "frob_norm", "q1",
                                            "frob_over_layer_median", "within_layer_rank")}
                                          for r in by_ratio[:10]],
               top10_q1=[{k: r[k] for k in ("layer", "row", "frob_norm", "q1")}
                          for r in by_q1[:10]],
               overlap_q1_vs_global_frob=len(setq & setf),
               overlap_q1_vs_within_layer=len(setq & setr))
    (ROOT / "results" / f"e10_selection_audit_{name}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    for m in ("mosaicbert", "modernbert"):
        audit(m)
