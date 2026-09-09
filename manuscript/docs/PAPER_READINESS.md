# PAPER_READINESS.md

**Date:** 2026-08-13. Sufficiency call for the six proposed main-result sections
(`PAPER_OUTLINE.md` v2), against `NEW_DIRECTION_EVIDENCE_AUDIT.md` and `CLAIMS_LEDGER.md`.
**Default answer to "requires another experiment" is NO**, per the governing instructions —
overridden only where an essential main-text claim genuinely cannot be made without it,
disclosure of the limitation is insufficient, and the missing result would actually change
whether the section survives. Curiosity, completeness, and reviewer-proofing are explicitly
not admissible reasons.

---

## R1 — Where NLP super-weight criteria transfer, and where they fail

**Evidence complete?** Partial. The calibration core is solid and unaffected by anything in
this session: NLP retrospective recovery (C-004, C-005 — rank 1/4,096 row and scalar, three
models), genomic cold-weight ranks for GENERator EUK, DNABERT-2, and NTv3 (C-002, C-003), the
DNABERT-2 706/768 leaky-signature example (C-010, already resolved as source-vs-propagator),
and Evo1's structural-fires/functional-null dissociation (C-009). GENERator **PROK**'s
cold-weight rank is not: it is the exact artifact N-009 already found non-reproducible, and
C-001 has been on hold since the prior session.

**Main unresolved vulnerability:** the PROK half of the cross-architecture confirmation rests
on a claim (C-001) that does not reproduce from the current checkpoint at its own layer, for
a cause not yet determined.

**Requires another experiment?** **NO.** Three independent architectures (GENERator EUK
decoder, DNABERT-2 encoder, NTv3 encoder) already give a real, reproducible, three-for-three
cold-weight confirmation without PROK. Disclosing PROK as on-hold (which the ledger already
does) is sufficient; dropping the PROK number from the headline claim does not change whether
R1 survives.

## R2/R3 — DNABERT-2 redundant pair / pretrained pair / joint norm carriage

**Evidence complete?** No. Zero artifacts of any kind exist for a 2-row pair ablation, its
base-model superadditivity, or layer-9 joint residual-norm carriage
(`NEW_DIRECTION_EVIDENCE_AUDIT.md` items 1–4).

**Main unresolved vulnerability:** the section as specifically envisioned (a redundant pair)
cannot be written at all — there is nothing to disclose a limitation *of*, because there is no
result.

**Requires another experiment?** **NO**, for a specific reason distinct from R1 and R6: this
is not a case of "the result exists but is thin" where disclosure could carry it — the result
simply does not exist. But the manuscript does not need this *specific* finding to remain
sound. The already-real, already-established DNABERT-2 ensemble result (C-027, C-028: no
single row exceeds −1.45%, all 10 rows together give −25.5%) already demonstrates that
DNABERT-2's causal object is distributed rather than a single scalar — a genuinely different
and still-interesting granularity finding relative to GENERator's single dominant row. R2/R3
as pair-specific sections are simply not drafted this round; the paper's soundness does not
depend on them. (If a future session runs the pair-ablation experiment and it confirms the
claim, R2/R3 can be written then — that is a decision for a session explicitly instructed to
run it, not a gap this pass should paper over.)

## R4 — GENERator canonical decoder super-activation / BOS attention-sink phenotype

**Evidence complete?** No. No script anywhere computes an attention weight, an argmax over
attention, or a shuffle-insensitivity test of attention for any GENERator model
(`NEW_DIRECTION_EVIDENCE_AUDIT.md` item 5).

**Main unresolved vulnerability:** the attention-sink-specific framing (token-0 ratio, argmax
rate) has no measurement to report, positive or null.

**Requires another experiment?** **NO.** The paper's high-level message needs a "genomic
decoder shows a canonical super-activation phenotype" data point, and that is already
supplied — independent of attention patterns — by the existing detect-then-ablate result
(GENERator EUK/PROK: single dominant row, out_max in the hundreds of thousands, +23,000%+ PPL
on ablation; C-007, C-008). The attention-to-BOS-token measurement would be a genuine
*addition* (and Yu et al.'s own finding that SW and attention sinks can dissociate makes it a
non-trivial one), but it is not required to support the claim the outline currently makes for
this section, and its absence does not remove the decoder-phenotype evidence the paper
already has.

## R5 — GENERator EUK causal GC steering

**Evidence complete?** No. E3 (`prereg/PREREG_steering.md`) specifies almost exactly this
design but was never locked or run; C-025 stays `pending`. No steering effect size — "38.6×"
or otherwise — exists anywhere in the repository (`NEW_DIRECTION_EVIDENCE_AUDIT.md` items
6–7).

**Main unresolved vulnerability:** the paper currently has only destructive (ablation)
causal evidence for the EUK/PROK channels; it has no positive/sufficiency demonstration that
scaling the channel *produces* a controlled compositional shift without comparable
degradation.

**Requires another experiment?** **NO**, under the governing criteria, even though this is
the closest call of the six. The manuscript's core transfer-question thesis does not require
a positive-steering result — extensive destructive-ablation causal evidence already exists
(C-007, C-008, C-026, C-027/C-028) and is sufficient to support every claim currently in
`PAPER_OUTLINE.md` v2. A steering result would *strengthen* the paper (exactly the kind of
"not just destructive ablation" evidence the new framing values) but its absence does not
make any currently-planned claim false or unwritable, and disclosing "steering was
preregistered but not run" is an honest, sufficient statement. **If the author independently
decides this section is wanted before submission**, E3 is the single best-specified,
already-preregistered candidate — see the final report's experiment-verdict section for the
distinction between "not required" and "highest value if one is ever authorized."

## R6 — Quantization granularity determines whether explicit SW protection is meaningful

**Evidence complete?** Yes, for the section as scoped. The empirical null is real and already
in the manuscript (whole-model INT4, with vs. without SW exemption, differs by
−0.0008/+0.0004 PPL, below noise floor — C-031, `paper/main.tex:787-800`). The mathematical
mechanism explaining it is directly verifiable from code already in the repository, with no
run required (C-033: per-row RTN scale = max\|row\|/maxval maps the row's own max element to
the quantizer endpoint exactly).

**Main unresolved vulnerability:** the link between C-033 (the mechanism) and C-031 (the
observed null) is a disclosed *inference* ("consistent with"), not a proof for the specific
runs C-031 draws on — C-031 itself has no recorded evidence path in the ledger
(`results/keep/UNMIGRATED.md`), so it has not been traced to a specific script/config/seed the
way C-033's mechanism has been traced to its code.

**Requires another experiment?** **NO.** Tracing C-031 to its exact generating run (to confirm
the SW row was in fact the max-magnitude element of its row in that run) is a documentation/
provenance task — find the existing JSON and script, not run anything new — and does not rise
to the level of a new experiment. The section is draftable now with the inference disclosed
as such.

---

## Overall verdict

**No section requires a new experiment to be written as scoped in `PAPER_OUTLINE.md` v2.**
R1 and R6 are draftable now on existing evidence. R2/R3, R4, and R5 are not draftable in their
specifically-envisioned forms — not because a disclosed limitation would be insufficient, but
because there is nothing to disclose *about a result*; there is no result. The correct action
for those three is to not draft them this round (R2/R3 falls back to the real, established
ensemble finding; R4 is covered at the ablation level without the attention-specific claim;
R5 is simply absent), not to run the missing experiments under this pass's constraints.
