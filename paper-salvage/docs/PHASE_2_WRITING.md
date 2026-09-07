# PHASE 2 — Writing (runs in parallel with Phase 1)

**Goal:** draft everything that does not depend on the Evo1 result. That is R1, R2, R4,
R6, R7, plus Intro, Discussion and Limitations.

R3 gets written to the point of its thesis-sentence slot and then waits on E2.

---

## Drafting order

Deliberately not front-to-back. Write from the most-settled evidence outward, so that the
sections with the least interpretive freedom anchor the ones with the most.

1. **Methods** — first. It forces the coverage table to be honest before any prose has a
   chance to oversell.
2. **R2** — the panel and ablations. Mostly exists in v15; needs the coverage table, the
   rename, and the L7/r603 promotion to main text.
3. **R1** — derivation and predictor. Genomic ranks are already in hand; NLP validation
   numbers drop in from E1.
4. **R6** — functional recruitment. Compress the 35-task table to four representative rows
   plus a heatmap; full table to supplement. Resolve NTv3 metric per PHASE_0 §0.4.
5. **R4** — what it computes. Hexamer + compact shuffles. Steering paragraph drops in
   from E3.
6. **R7** — one paragraph.
7. **Intro** — after the results are written, not before. Paragraph 2 is the "why genomics"
   text in `PAPER_OUTLINE.md`, verbatim or close.
8. **Abstract** — last. It is the thing v15 got most wrong.
9. **Discussion + Limitations** — port the v15 Limitations block and update against the
   new structure. Add: the routing decomposition rests on a limited set of architecture
   transitions; the corpus contrast is n=2 within one model family.
10. **R3** — thesis sentence, once E2 lands.

## Section-by-section notes

### Methods
Coverage table first. Then: detect-then-ablate protocol (unchanged from v15), quadratic
amplifier audit, residual-stream attribution, impulse protocol (**new — needs writing from
scratch**, no v15 text exists), causal ablation, shuffles, hexamer scan, fine-tuning,
steering (new), compression (compressed).

### R1
The derivation is short; do not pad it. The interesting content is the two-level NLP
validation and the c_{k,i} decomposition. Wording rule from CLAUDE.md applies.

### R2
Rename to "High-gain channels exhibit divergent functional criticality across genomic
architectures." Lead with the coverage table. The L7/r603 outlier is not a caveat — it is
a demonstration that the predictor measures local de-novo generation, and correctly fails
to fire on a propagator layer. Write it that way.

### R4
Pair the shuffle result with the routing result explicitly (D-007). One sentence plus a
footnote on the PROK sign convention — v15 spends roughly three paragraphs on this and it
reads as defensiveness.

### R6
Four representative tasks in main text: fungal species (−82.5%), one chromatin/composition
task, splice (−0.05%), one local promoter/TF task (≈0%). Plus a compact distribution across
all 35 and the fact that all 35 random-row controls fell within ±0.1 pp — that control is
strong and currently undersold.

## Ledger discipline

Every claim added to a draft section gets a row in `CLAIMS_LEDGER.md` in the same commit.
If you cannot fill in the evidence path, the claim is not ready to be written.

## Exit criteria

- All sections except R3's thesis sentence drafted
- Claims ledger complete for every drafted claim
- Figures 1, 2, 4, 5 rendered (Figure 3 waits on E2)
- Supplement assembled per `CUT_LIST.md`
