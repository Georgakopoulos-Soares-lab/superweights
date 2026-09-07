# paper-salvage

Restructure of the super-weights-in-genomic-LMs manuscript. Drop this folder inside the
existing repo (it references the old tree but never writes to it).

**Read `CLAUDE.md` first.** It contains the hard constraints on claims — those are not
style preferences, they encode the specific errors that made v15 unpublishable.

## Thesis

> Gated MLPs create structurally predictable high-gain channels, but architectures differ
> fundamentally in how these signals are routed and recruited for biological computation.

`structural amplifier → architectural routing → encoded information → causal steering → downstream recruitment`

## Layout

```
CLAUDE.md                     operating rules — read first
docs/
  PROJECT_STATUS.md           live state; update at the end of every session
  PAPER_OUTLINE.md            frozen skeleton, figures, model coverage table
  DECISIONS.md                append-only decision log (D-001 … D-013)
  CLAIMS_LEDGER.md            every claim → evidence → strength → status, plus N-notes
  CUT_LIST.md                 keep / demote-to-supplement / removed as a claim
  ENVIRONMENT.md              container hashes + package versions for Methods
  PHASE_0_TRIAGE.md           inventory, migration, submission blockers
  PHASE_1_BLOCKING.md         E1–E4 + the binding stop rule
  PHASE_2_WRITING.md          Evo1-independent drafting, runs in parallel with Phase 1
  PHASE_3_ASSEMBLY.md         consistency pass, stats audit, reviewer anticipation
  prereg/                     locked predictions + LOCKS.jsonl ledger
    archive/                  superseded prereg versions (see the notice in each)
src/
  uk_frobenius.py             the predictor: ‖U_k‖_F, c_{k,i}, granularity, layer ranks
  test_uk_frobenius.py        self-test — run before trusting any audit
  prereg_lock.py              hash + commit + timestamp locking
experiments/E1…E5/            one folder per experiment, spec in its README
results/keep/                 migrated artifacts, each with PROVENANCE.md
figures/                      one script per figure
```

`prereg_lock.py` resolves its ledger relative to its own location, so it only writes to
`docs/prereg/LOCKS.jsonl` when it sits in `src/`. Keep that layout.

## First session

```bash
python src/test_uk_frobenius.py          # verify the predictor before anything else
cat CLAUDE.md docs/PAPER_OUTLINE.md
# then work PHASE_0_TRIAGE.md top to bottom
```

Priority in Phase 0: **locate the GENERator JSONs.** Figure 2 panels A–D cannot be rendered
without them and the v15 caption currently admits they are missing.

## Status at a glance

- Phases 1 and 2 run **in parallel**. R1, R2, R4, R6, R7 do not depend on Evo1.
- The only thing blocked on Evo1 is R3's thesis sentence, and both variants are pre-drafted.
- The stop rule in `PHASE_1_BLOCKING.md` is binding. Four experiments, then re-assess.

## Requirements

`torch`, `transformers`, `numpy`. The predictor needs no GPU beyond loading weights — it
is pure weight algebra and runs in float64. It is also the artifact worth releasing: a
reviewer can reproduce Figure 1 in minutes on a laptop for the small models.
