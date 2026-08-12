# PHASE 3 — Assembly and submission

**Entry condition:** Phase 1 complete, Phase 2 drafted, R3 thesis sentence resolved.

---

## 3.1 Consistency pass

This is the pass that would have caught what sank v15. Do it mechanically, not by reading.

- [ ] **Abstract vs. Results.** Extract every quantitative claim in the abstract; verify
      each against its Results sentence and its ledger row. The v15 failure (pruning
      tolerance) was exactly this.
- [ ] **No forbidden phrases.** Grep for: `shadow redundancy`, `eight models`,
      `extremelly`, and any sentence asserting that C is necessary for criticality.
- [ ] **Every figure panel exists.** Grep captions for `pending`, `omitted`, `TBD`.
- [ ] **Every claim in the ledger has a resolvable evidence path.**
- [ ] **Every number in main text matches its source file.** Not from memory.
- [ ] **All preregistrations verify:** `python src/prereg_lock.py verify --all`
- [ ] **Hedging calibration.** Method claims assertive; mechanism claims hedged. Read R1
      and R3 back to back and check the register actually differs.

## 3.2 Statistics audit

- [ ] Every n stated, including seed counts
- [ ] Every post-hoc analytic choice either removed or disclosed at the point of use
- [ ] NTv3 metric resolution applied per PHASE_0 §0.4
- [ ] Multiple-comparison correction stated where used
- [ ] Random-control arms reported for every ablation claim

## 3.3 Reproducibility package

- [ ] `src/uk_frobenius.py` released — it is the paper's main artifact and costs nothing
      to run. This is the EmmaEmb move: ship the thing, not just the finding.
- [ ] Impulse protocol script released
- [ ] Steering script released
- [ ] Environment pinned; model versions and revisions recorded
- [ ] Zenodo DOI for code and derived results
- [ ] README with a one-command reproduction of Figure 1

## 3.4 Reviewer-anticipation pass

Write internal answers to the objections you can already predict:

1. "The predictor fails on Evo1." → It does not fail; it correctly identifies structural
   concentration. Concentration and criticality are separate properties, which is the point
   of the paper.
2. "Only n=1 architecture transition supports the routing decomposition." → Acknowledged
   in Limitations; the NLP validation supplies a second, independent axis of generality.
3. "Predictor rank 706 at DNABERT-2 L7." → Source vs. propagator; residual attribution
   resolves it and this is now in main text.
4. "The biology is just GC content." → The claim is not that the model discovered new
   biology; it is that structural origin generalizes while functional recruitment does not.
   Steering (if it lands) makes the compositional channel causal, not merely correlated.
5. "Model coverage is uneven." → Stated in a table in R2, not buried.

## 3.5 Submission

- [ ] Venue chosen and formatted (Genome Biology / Nature Methods / NAR GaB / Patterns)
- [ ] Cover letter naming the three contributions: closed-form predictor with external
      NLP validation, concentration/criticality dissociation, bidirectional steering
- [ ] Preprint posted
- [ ] Data/code availability statement matching what was actually deposited
