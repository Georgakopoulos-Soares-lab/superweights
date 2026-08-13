# CLAUDE.md — operating rules for this folder

You are working on the salvage and restructure of a manuscript on super-weights in
genomic language models. The previous draft (`v15.pdf`) is being rebuilt around a new
thesis. Read `docs/PAPER_OUTLINE.md` before doing anything substantive.

## The thesis (frozen — do not re-litigate)

**Gated MLPs create structurally predictable high-gain channels, but architectures
differ fundamentally in how these signals are routed and recruited for biological
computation.**

Causal progression of the paper:
`structural amplifier → architectural routing → encoded information → causal steering → downstream recruitment`

## The five questions

Every result in the main text must answer one of these. If it doesn't, it goes to
supplement or is cut. No exceptions, no "but it's interesting."

1. Can we identify the amplifier from weights alone?
2. At what granularity is it concentrated (scalar / row / ensemble)?
3. How does architecture route its signal?
4. What information does it carry?
5. What does the model use that information for?

## Hard constraints on claims

These are not style preferences. Violating them reintroduces the errors that made v15
unpublishable.

- **C (coordinate preservation) is NOT necessary for criticality.** _(C value re-measured
  and confirmed on the clean eager kernel — N-011. The second clause below still rests on
  C-027/C-028, which have not been audited for the Triton/eager question.)_ DNABERT-2 has
  C ≈ 0 and the strongest encoder phenotype. Never write that (‖U_k‖_F, C) jointly
  predict criticality. C describes *routing geometry*.

  History: the constraint was originally derived from an impulse C measured through a
  noise-dominated kernel. The locked-protocol re-measurement on eager attention gives
  DNABERT-2 C = +0.0331 → −0.0221 (peak +0.0625), still ≈ 0, so the constraint survives and
  X-003 stays retired (N-011). What did **not** survive is the companion claim that
  DNABERT-2 has the largest impulse KL — that was the noise floor and is retired as X-006.

  Unaffected either way: DNABERT-2's *ablation* phenotype (C-027 splice −25.5 pp, C-028
  ensemble behaviour) is a different experiment. Only the routing number is in question.
- **Do not claim T predicts criticality** until the Evo1 broadcast run lands. The
  R3 thesis sentence is a slot with two drafted variants — see `docs/PHASE_1_BLOCKING.md`.
- **"Recovers the published super-weight output rows"** unless the scalar index *i*
  also ranks highly, in which case you may say scalar recovery. Be pedantic here.
- **Never write "shadow redundancy."** The pruning data contradict it.
- **Never write "eight models"** as though they form a uniform benchmark. Use the
  coverage table in `docs/PAPER_OUTLINE.md`.
- **Do not assert causal arrows between routing and context-sensitivity.** EUK and PROK
  "differ jointly in input-context sensitivity and downstream routing." Not "diffusion
  causes contextual integration."
- Assertive about the **method**. Hedged about the **mechanism**. That calibration is
  deliberate.

## Working discipline

- Every claim that will appear in the manuscript gets a row in `docs/CLAIMS_LEDGER.md`
  with its evidence file and status. If you add a claim to a draft section, add the row
  in the same commit.
- Every decision that changes scope gets an entry in `docs/DECISIONS.md`. Append only;
  never rewrite history.
- Update `docs/PROJECT_STATUS.md` at the end of any session that changes state.
- Predictions must be locked *before* the confirming run: `python src/prereg_lock.py lock <file>`.
- Results we are keeping get copied into `results/keep/` with provenance recorded. Do not
  reference paths that live only in the old tree.

## What not to do

- Do not add new analyses that aren't in the current phase doc. This project failed once
  by scattering. The stop rule is in `docs/PHASE_1_BLOCKING.md` and it is binding.
- Do not silently drop a negative result. Null steering, null Evo1 — both get reported.
  Pre-committed reporting language is in the preregistrations.
- Do not delete anything from the old tree. Cuts are *demotions* to supplement; see
  `docs/CUT_LIST.md`.
