#!/usr/bin/env python
"""Round-2 follow-up (CORRECTED): row-index recurrence across layers, with provenance
and circularity determined by checking DECODER_INTERVENTION_FREEZE.md and
BASIS_FREEZE.md before treating any set as independent evidence.

CRITICAL CORRECTION from the first pass: OLMo's and Phi-3's multi-layer row sets were
NOT independently discovered by this project's detectors -- they are direct citations of
Yu et al.'s published Table 2 (see experiments/E10_nlp_architecture_causal/
DECODER_INTERVENTION_FREEZE.md). Phi-3's set is therefore fully circular and excluded from
the chance-magnitude analysis. OLMo's set is *mostly* circular (Yu et al. published row 269
at layers {1,2,7,24}); only this session's independent detector finding of L30/r269
(activation-argmax, not in Yu et al.) is genuinely new evidence. Llama-7B's L30/r3968
(activation-argmax) IS fully independent -- Yu et al. published only ONE row for Llama
(L2/r3968), so the L2/L30 collision is a real, non-circular, independently-discovered pair.
DNABERT-2's basis is internally-sourced (this project's own prior activation-magnitude
ranking, BASIS_FREEZE.md / super_weight_index.json) -- not literature-circular, but not
an independent statistical test of the Sun/Yu phenomenon either; it is best read as a
same-domain-as-detector-design instance, and the only one with real k (10 coordinates).

The i.i.d.-uniform-row-index null is retained ONLY as a labeled, explicitly-flagged-wrong
reference point, per author instruction -- Sun et al. 2402.17762 and Yu et al.'s own
published tables already establish that massive-activation/super-weight rows concentrate
at a small, fixed set of dimensions, not uniformly. The fold-over-chance numbers are
computed for completeness but should not be cited as evidence of non-randomness.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "audit/rederivations/row_index_recurrence.csv"

# (model, d_model, coords, provenance, circular, note)
RECORDS = [
    ("DNABERT-2 (E9 10-row basis)", 768,
     [(5, 603), (3, 86), (3, 399), (9, 264), (9, 294), (3, 603), (3, 641), (7, 603), (6, 603), (5, 86)],
     "internal prior work: super_weight_index.json, top-10 by global out_max (activation "
     "magnitude) across ALL layers combined -- BASIS_FREEZE.md. NOT from Yu et al. or Sun et al.",
     False,
     "Only entry with real k. Selection method (flat top-10-by-activation across all layers) "
     "is exactly the kind of ranking that would surface a persistently-massive row multiple "
     "times if Sun et al.'s phenomenon holds -- informative as a genomic replication of that "
     "phenomenon, not as an independent statistical surprise."),
    ("Phi-3-mini -- EXCLUDED, fully circular", 3072,
     [(2, 525), (2, 1693), (2, 1113), (4, 525), (4, 1113), (4, 1693)],
     "Yu et al. Table 2, published pre-existing set, cited verbatim in "
     "DECODER_INTERVENTION_FREEZE.md Section 4 ('pre-existing SET of 6 published rows'). "
     "This project performed zero independent row search for Phi-3.",
     True,
     "100% recurrence is a property of Yu et al.'s published table, not a finding of this "
     "audit or this project. Do not use as evidence of a real phenomenon here -- it is a "
     "citation, not a discovery. Still consistent with Yu et al.'s own claim, but that is "
     "Yu et al.'s claim to make, not ours."),
    ("OLMo-7B -- mostly circular, ONE new instance", 4096,
     [(1, 269), (2, 269), (7, 269), (24, 269), (30, 269)],
     "Layers {1,2,7,24}: Yu et al. Table 2 published set (DECODER_INTERVENTION_FREEZE.md "
     "Section 3). Layer 30: THIS SESSION's independent activation-argmax finding "
     "(audit/rederivations/section3_text_decoder_calibration.json) -- not in Yu et al.",
     True,
     "4/5 coordinates are a citation; only L30 is new. The genuinely independent finding is "
     "a single new pair (any-published-layer, L30) -- k=2 worth of new evidence, not k=5. "
     "Report L30 as an independent replication of Yu et al.'s row-269 family, not as this "
     "audit discovering a 5-layer recurrence from scratch."),
    ("Llama-7B -- fully independent (no external multi-row citation)", 4096,
     [(2, 3968), (30, 3968)],
     "Layer 2: Yu et al. Table 2, sole published coordinate for Llama-7B (K=1, "
     "'no ranking performed', DECODER_INTERVENTION_FREEZE.md Section 1). Layer 30: THIS "
     "SESSION's independent activation-argmax finding, not in any prior publication used "
     "by this project.",
     False,
     "The cleanest non-circular data point: Yu et al. published only ONE row for Llama-7B, "
     "so this session finding a SECOND, different-layer instance of the same row index via "
     "an independent method (activation-argmax on WikiText-2) is genuinely new. Still k=2 "
     "(one pair) -- demoted per instruction on statistical weight, but not for circularity."),
    ("NTv3 -- fully independent, genomic (untested by Sun et al. or Yu et al.)", 1536,
     [(11, 1472), (6, 1472)],
     "Both layers found by this session's own detector runs (audit/rederivations/"
     "section3_ntv3_detector_recheck.json). NTv3 does not appear in Yu et al. or Sun et al. "
     "at all -- neither paper studied genomic models.",
     False,
     "Fully independent and outside prior literature's scope (genomic encoder). Still k=2 "
     "(one pair) -- demoted per instruction on statistical weight, not on circularity."),
    ("Mistral-7B (single coordinate, no recurrence found)", 4096, [(1, 2070)],
     "Yu et al. Table 2, sole published coordinate (K=1). This session's ratio-argmax and "
     "activation-argmax both landed on the same coordinate -- no second instance found.",
     False, "k=1, no recurrence check possible; also not circular since no citation-driven set was used."),
    ("GENERator-EUK-3B (single coordinate, no recurrence found)", 3072, [(4, 2371)],
     "This session's ratio-argmax and activation-argmax both landed on the frozen candidate "
     "-- no second instance found.",
     False, "k=1, no recurrence check possible."),
]


def main():
    rows_out = []
    for model, d_model, coords, provenance, circular, note in RECORDS:
        k = len(coords)
        rows_by_index = {}
        for layer, row in coords:
            rows_by_index.setdefault(row, set()).add(layer)
        n_pairs_total = k * (k - 1) // 2
        n_colliding_pairs = sum(len(layers) * (len(layers) - 1) // 2 for layers in rows_by_index.values())
        expected_pairs = n_pairs_total / d_model if d_model else float("nan")
        fold = (n_colliding_pairs / expected_pairs) if expected_pairs > 0 else float("nan")
        rows_out.append(dict(
            model=model, d_model=d_model, n_coordinates=k,
            n_pairs_total=n_pairs_total, n_colliding_pairs_observed=n_colliding_pairs,
            expected_colliding_pairs_under_iid_uniform_null=round(expected_pairs, 6) if expected_pairs == expected_pairs else "",
            fold_over_iid_null=round(fold, 1) if fold == fold else "",
            null_known_false="TRUE -- Sun et al. 2402.17762 and Yu et al. Table 2 both establish "
                              "massive-activation/super-weight rows concentrate at a small fixed "
                              "set of dims, not i.i.d. uniform; fold numbers are NOT evidence, "
                              "reference only",
            circular=circular, provenance=provenance,
            coordinates="; ".join(f"L{l}/r{r}" for l, r in coords), note=note,
        ))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {OUT} ({len(rows_out)} rows)")
    for r in rows_out:
        tag = " [CIRCULAR -- EXCLUDE FROM EVIDENCE]" if r["circular"] else ""
        print(f"  {r['model']}{tag}")
        print(f"    k={r['n_coordinates']}  observed_pairs={r['n_colliding_pairs_observed']}/{r['n_pairs_total']}  "
              f"fold_over_null={r['fold_over_iid_null']} (null known false, reference only)")


if __name__ == "__main__":
    main()
