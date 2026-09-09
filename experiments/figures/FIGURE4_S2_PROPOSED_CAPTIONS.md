# Proposed captions — new Figure 4 and candidate Supplementary Figure S2

Not yet inserted into `actual_manuscript.md`. Every number below was checked directly against
`results/E_BOS_MEDIATION/bos_mediation_results.json` (see the assert block in
`fig4_generator.py::main()`, which fails loudly if the artifact ever drifts from these values).

---

## Figure 4

**Figure 4. A BOS-localized causal mechanism at GENERator EUK's high-gain pathway (L4/r2371).**
**(A)** Exact q₁ of the primary high-gain row (see Fig. 1B). **(B)** BOS-centered
attention/activation phenotype (unchanged from the previous version of this figure; single
run, not seed-replicated): mean incoming attention mass at position 0 (37.9%, 33.0× the
uniform expectation), fraction of layer-head observations with attention argmax at position 0
(78.8%), and a co-localized activation maximum. Association only — in a causal decoder,
position 0 can only attend to itself, so part of this co-occurrence is architecturally
guaranteed; this panel does not establish that row 2371 causes the sink. **(C)** Native NLL
under six interventions on row 2371's FFN/down-projection contribution (n=100 damage windows;
provenance gate and all 6 required smoke/invariance tests passed —
`results/E_BOS_MEDIATION/{provenance_check,smoke_tests}.json`): intact (6.385), full ablation
(8.754), ablating the row's contribution at the BOS position only (8.722; paired-bootstrap gap
vs. intact 2.336, 95% CI [2.20, 2.49] — statistically indistinguishable from full ablation),
preserving the row's contribution at BOS only while ablating it everywhere else (6.386; gap
vs. intact 0.0009, CI [−0.004, 0.005] — statistically indistinguishable from intact),
restoring only the intact BOS contribution after full ablation (6.386, rescue fraction ~100%,
CI [99.8, 100.2]%), and restoring an equally large contribution at a matched non-BOS position
after full ablation (8.754, rescue fraction ~0%, CI [−0.1, 0.2]%). Loss of the row's BOS
contribution alone reproduces essentially all of the full-ablation damage, and
preserving/restoring only that BOS contribution rescues essentially all of it; an equally
large restoration elsewhere does not — the effect is position-specific, not a generic response
to any large activation injected at this location. **(D)** Generated GC fraction (n=24
prompts) under the same six conditions. Preserving/restoring BOS rescues most (~93%) of the GC
drop caused by full ablation, and restoring at the matched non-BOS position does not (rescue
fraction ~−6%, i.e. no better than no intervention) — consistent with the BOS-specific pattern
in (C). One result is plotted as measured, not explained: BOS-only ablation lowers GC (0.292)
slightly *below* the full-ablation level (0.309) rather than matching it — NLL and GC
dissociate at this one condition.

---

## Supplementary Figure S2

**Supplementary Figure S2. Random-direction / damage-tracking control at GENERator EUK
L4/r2371.** Retained essentially unchanged from the previous main-text Figure 4C; superseded
there by the BOS-mediation result (new Figure 4C/D), which answers a different, position-
resolved question and does not make this control redundant. **(a)** Generated GC fraction
against the scale *c* of a fixed random unit direction substituted for row 2371's weight
vector (*c*=0 is equivalent to full ablation), against the full-ablation (GC=0.3065) and
untouched-baseline (GC=0.4204) levels; error bars are SEM. **(b)** Generated GC fraction
against native-loss damage (NLL) on shared axes for row 2371's own α-sweep (α ∈ {0, 0.5,
1.0}), the random-direction grid (*c* ∈ {0.0125, 0.5, 1.0, 3.0, 8.0}), and 5 inert same-layer
control rows. Composition tracks damage magnitude rather than which intervention produced it:
at matched NLL damage, a random direction reproduces the low-GC phenotype seen under ablation,
and only at the largest tested scale (*c*=8.0, which still retains 84% of the ablation damage)
does GC recover part of the way (29.6%) toward baseline. This argues against a simple
interpretation in which preserving the row's specific learned direction is required for the
phenotype, but does not establish full direction-independence: the tested random directions
remain functionally close to removal even at their largest scale, and the control rows never
reached comparably large native-loss damage, so these data do not exclude an equally
catastrophic perturbation elsewhere producing a similar compositional shift.
