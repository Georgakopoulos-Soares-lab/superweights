# C-032 retirement recommendation — made now, independent of E8's outcome

**This recommendation does not wait for E8.** It is based entirely on E7's already-final
results (`experiments/E7_exact_dimensionality/RESULTS.md`), which this document does not
extend or reinterpret.

## The claim, as it currently reads in `docs/CLAIMS_LEDGER.md`

> C-032 | R2/R3/R6 | Granularity of the causal object differs across models: PR 3.64
> (DNABERT-2) / 4.56 (EUK) / 22.7 (NTv3) / 122 (Evo1) / 2192 (PROK, contested) vs 1.02–1.24
> for published NLP SWs | ... | supported | E4 canonical table done — association only

## Why RETIRE, not HOLD or SUPERSEDE-in-place

E5 already flagged this claim's measurement validity (the diagonal decomposition it is built
from captures the exact operator far less reliably for genomic rows than for NLP rows). E6's
row-level near-miss left that flag unresolved either way. **E7 resolves it, decisively, in the
direction of retirement:**

- **GENERator EUK's exact `PR_spec` (1.0653) sits squarely inside the published-NLP range
  (1.02–1.24)** — not near the genomic pole the old claim assigns it (diagonal PR 4.56). The
  claim's own headline contrast fails for one of its four genomic data points.
- Every genomic model's diagonal PR overstates its exact `PR_spec` by a large, one-directional
  factor: EUK 4.3x, DNABERT-2 2.4x, NTv3 3.5x. The bias is systematic and always in the same
  direction (diagonal PR inflates apparent genomic dimensionality), which is exactly what E5's
  independent `f_cross` measurement already predicted.
- This is not a borderline or contested result the way E6's was — it required no threshold,
  no margin, no near-miss judgment call. GENERator EUK's exact value is simply, directly
  inside the NLP range.

**Recommend: RETIRE.** The claim's evidence path (`experiments/E4_granularity/
CANONICAL_TABLE.md`) and its historical PR values are preserved exactly as recorded — nothing
about the ledger's history is rewritten. A new claim (proposed C-034 in E7's `RESULTS.md`)
describes what the exact metric actually shows, and is offered as a separate row, not a
mutation of C-032's numbers.

## What this recommendation does not do

It does not edit `docs/CLAIMS_LEDGER.md`. Per this project's standing discipline (every prior
claim-status recommendation in E5/E6/E7 was recorded as a recommendation, not a self-executed
ledger edit), the actual retirement — moving C-032 to the "Retired claims" table with a
one-line reason, exactly as X-001 through X-006 are already recorded there — is left for
author action. This document makes the call explicit and decisive so that action does not
have to wait for E8, which tests a different, follow-on hypothesis and could not change this
conclusion either way.
