# MANUSCRIPT_MIGRATION_MAP.md

**Date:** 2026-08-13. **Scope:** section-by-section disposition of `paper/main.tex` (the v6
manuscript — title: "A Structural Predictor of Super-Weights Across Genomic Language Model
Architectures", 1,086 lines) against `PAPER_OUTLINE.md` v2. **No manuscript prose was
rewritten in this pass** — this is a map for a future writing session, not the rewrite
itself. Every disposition below cites the exact line range in `paper/main.tex` as it stands
today, its supporting claim ID(s), and its new destination.

**Legend:** KEEP (usable as-is or with only cosmetic edits) · REWRITE (content is
real/usable, framing must change) · MOVE TO SUPPLEMENT (real, but not main-text under the new
outline) · DELETE (claim retired, no replacement) · BLOCKED BY PROVENANCE (cannot be used
in its current form — either the section rests on a `contested`/`on hold` artifact, or the
retirement instruction covering it is itself unaudited; flagged for the author rather than
resolved).

---

## Front matter

| Lines | Content | Disposition | Why | New destination | Claims |
|---|---|---|---|---|---|
| 1–4, 45–47 | File header + title ("A Structural Predictor...") | **REWRITE** | Title leads with U_k, now demoted (D-016/D-017) | New title should foreground the transfer-question thesis | — |
| 70–113 | Abstract | **REWRITE** | Contains: "eight of the best genomic language models" (forbidden framing, D-001/CUT_LIST — never fixed, still present); "SW neighbourhood is extremely tolerant to pruning and INT4 quantization... suggesting refined SW-aware compression schemes are possible" (**verbatim retired claim X-001**, `PROJECT_STATUS.md` already flagged this as an unfixed blocker); U_k framed as first/lead contribution; PROK SAE cited as supporting evidence (line 95–97, now cut, N-013) | Rewritten against R1–R6 once R2–R5 have evidence or are explicitly scoped out | X-001 (retired), N-013 |

## Introduction / "Super-weight existence across architectures"

| Lines | Content | Disposition | Why | New destination | Claims |
|---|---|---|---|---|---|
| 114–139 | Intro framing, "why genomics" | **KEEP** | The "why genomics" paragraph is retained verbatim in `PAPER_OUTLINE.md` v2 | Intro, unchanged | — |
| 140, 171, 176, 238 | "eight genomic language models" / "eight models" (4 occurrences: abstract-adjacent intro line, results-opening line, protocol line, Figure 1 caption) | **REWRITE** | Forbidden framing (D-001, `CUT_LIST.md`, `CLAUDE.md`); coverage table already exists in `PAPER_OUTLINE.md` and is not used here | Replace every instance with the coverage table reference | — |
| 141–164 | "Our central result is a closed-form, weight-only predictor..." through "...shadow redundancy" | **REWRITE + DELETE (last clause)** | Leads with U_k as the central result (D-016/D-017); the final clause ("unusually tolerant to pruning and INT4 quantization... shadow redundancy") is the **verbatim retired X-001 claim** appearing a second time | R1 (reframed) for the predictor material; the shadow-redundancy clause is deleted outright, no replacement | X-001 |
| 166–233 | Detect-then-ablate results (GENERator EUK/PROK, DNABERT-2, NTv3, Evo1, HybridNA/MegaDNA/Caduceus) | **KEEP** | Real, established, per-model ablation numbers (C-007, C-008, C-009, C-027/8 background, plus the null SSM/hybrid results) are unaffected by any new-direction finding | R1 background / coverage table support | C-007, C-008, C-009 |
| 217–233 | Evo1 detection-stage description | **KEEP, with a caution** | The existing account (structural candidate fires, ablation null, attributed to mixer redistribution) stands; do **not** append the unaudited 2^24-saturation/rescue narrative here (D-019) | R1, as the "detector alone is insufficient" example | C-009 |
| 235–249 | Figure 1 + caption | **REWRITE (caption only)** | Contains "eight genomic language models" (see above); panel content itself is fine | R1 | — |

## "A weight-only structural predictor..." (old R1 core)

| Lines | Content | Disposition | Why | New destination | Claims |
|---|---|---|---|---|---|
| 251–270 | Frobenius-bound derivation | **KEEP** | Mathematically unaffected; still a legitimate confirmatory signature, just no longer the headline | R1, reframed per D-017 as calibration/confirmatory, not novel | — |
| 272–281 | GENERator EUK/PROK cold-weight ranks | **KEEP, with a flag on the PROK number** | EUK reproduces exactly (N-009); PROK's stored value (2,648/rank 1) does **not** reproduce against current weights at the same layer (recomputed rank 1277/1289) — this is not new to this session, but must not be silently presented as clean in a rewrite | R1; PROK number needs an explicit "on hold, N-009" caveat wherever it appears | C-001 (on hold) |
| 283–303 | DNABERT-2 706/768 outlier + residual-attribution resolution | **KEEP** | This is exactly the fact the new framing wants foregrounded as evidence that structural detection alone is insufficient (task item A) — already in the manuscript, already resolved | R1, promoted from "outlier explained away" to "evidence the signature is leaky" | C-010 |
| 305–312 | NTv3 cold-weight rank | **KEEP** | Weights-only claim (C-003), unaffected by NTv3's contested *functional* result (C-029) — these are different claims about different things and must not be conflated in the rewrite | R1 | C-003 |
| 314–360 | Evo1 predictor + residual-stream attribution + "predictor is necessary, not sufficient" | **KEEP, with the same caution as line 217–233** | Real, audited account; do not append the 2^24 narrative | R1 | C-009 |
| 362–390 | Figure 2 (E–F) + caption | **KEEP, panels A–D remain BLOCKED BY PROVENANCE** | Caption already accurately states panels A–D are pending (though the stated *reason*, "pending GENERator JSONs," is stale — the JSONs exist; the real blocker is N-009, per `REFERENCE_AUDIT.md`, unchanged this session) | R1 | N-009 |

## "Super-weights encode nucleotide composition..." (old R4 — PROK/EUK mechanism)

This is the section most affected by the PROK provenance conflict (N-015). Every EUK-only
paragraph is unaffected; every PROK-bearing paragraph is now **BLOCKED BY PROVENANCE** because
it is keyed to the same layer-2/row-1927 artifact N-009 already found non-reproducible, and
the two `contested` ledger rows (C-020, C-021 PROK half) sit directly inside it.

| Lines | Content | Disposition | Why | New destination | Claims |
|---|---|---|---|---|---|
| 392–406 | Section intro, "SW-activation/ablation-cost correlation reversed sign between the two kingdoms" | **BLOCKED BY PROVENANCE** | States the sign-reversal framing as settled fact in the section's own topic sentence | Must be rewritten once N-015 resolves either direction | N-015, C-020 |
| 407–419 | Causal necessity, context-uniform (EUK + PROK numbers) | EUK: **KEEP**. PROK: **BLOCKED BY PROVENANCE** | EUK's row (2,371/1,522 at L4) is unaffected; PROK's row (1,927 at L2) is the contested artifact | EUK half → R1/background; PROK half held pending N-015 | C-001 |
| 420–431 | Shuffle controls, "Shuffle controls separate the two kingdoms" | EUK: **KEEP**. PROK: **BLOCKED BY PROVENANCE** | Same split; this is exactly ledger row C-021, already split in this session's reconciliation | EUK half → supplement (per original `CUT_LIST` "compact shuffle" note); PROK half held | C-021 |
| 433–459 | "Ablation cost scales with super-weight write magnitude" — **contains the r=−0.710/+0.710 sign-convention discussion** | EUK: **KEEP**. PROK + sign-convention paragraph: **BLOCKED BY PROVENANCE** | This is the exact passage the task brief calls "old PROK sign reversal" / "old r≈−0.710 interpretation" — it is currently presented as a **resolved, established** finding (C-020) in the manuscript. The retraction instruction has no local artifact (N-015); the manuscript must not keep asserting it as settled, but it also must not be silently flipped to the opposite (invalid) story | EUK half (r=+0.437) → R1/background, unaffected; PROK half held pending N-015 | C-019 (EUK, unaffected), C-020 (PROK, contested) |
| 461–481 | PROK sparse autoencoder | **DELETE** | Matches task item C/D exactly ("old PROK SAE result is invalid," "old PROK composition story is invalid"); independently, `CUT_LIST.md` already demoted this and N-013 found a real, disclosable methods weakness (fp16 clamp at ±60,000 vs. the PROK SW's own out_max of 506,014) | Cut entirely; if kept anywhere, only as a one-line methods-limitations note citing the clamp, not as supporting evidence for anything | N-013 |
| 483–493 | Motif enrichment (Fisher + BH) | EUK: **MOVE TO SUPPLEMENT** (already `CUT_LIST` policy). PROK: **BLOCKED BY PROVENANCE** | EUK homopolymer result stands on its own SW row; PROK half keyed to the contested row | EUK → supplement; PROK held | C-023 |
| 495–503 | OLS regression on composition features | EUK: **MOVE TO SUPPLEMENT**. PROK: **BLOCKED BY PROVENANCE** | Same split, already a `CUT_LIST` demotion for EUK | EUK → supplement; PROK held | — (no ledger row; unaffected) |
| 505–510 | "Super-weights are not kingdom-specific detectors" (kingdom-specificity test) | **DELETE** | This is the literal "old kingdom-level interpretation" the task instructs retiring (item C) — even though both directions were already null (p=0.538, p=0.644) and it was never used to assert a positive kingdom-specific story, the framing itself (testing and reporting on a "kingdom" axis for a contested PROK row) should not survive | Cut; if the null result is wanted anywhere, it needs re-derivation once/if the real PROK SW is identified with a real artifact | — |
| 512–530 | Figure 3 + caption (panels A–D, EUK/PROK hexamer scatter, shuffle bars) | EUK panels: **KEEP**. PROK panels (C, and the PROK half of D): **BLOCKED BY PROVENANCE** | Panel-level split matching the sections above; a future rendering pass should consider an EUK-only version of this figure, mirroring the existing Figure-2 EUK-only option already flagged in `REFERENCE_AUDIT.md` | EUK → R1/background figure; PROK held | N-009, N-015 |

## Functional consequences (old R6)

| Lines | Content | Disposition | Why | New destination | Claims |
|---|---|---|---|---|---|
| 540–560 | DNABERT-2 GUE fine-tune + SW-ensemble ablation (splice −25.5pp, histone n.s., promoter inconclusive) | **KEEP** | Real, established, uncontested — the strongest uncontested functional-encoder example (matches the new framing's own assessment) | R1/background; anchors "DNABERT-2 is the strongest uncontested functional encoder example" | C-027 |
| 561–600 | NTv3 five-seed splice table + discussion | **BLOCKED BY PROVENANCE** | C-029 downgraded to `contested` this session (N-014) — must not be presented as a confirmed positive replication until N-014 resolves | Held; if kept, must carry an explicit "contested, see N-014" caveat rather than being presented as settled | C-029 |
| 602–608 | "The splice/histone dissociation... is consistent with... a local-hexamer detector role" | **KEEP** | Already correctly hedged as "reliability and magnitude rather than a clean dissociation" — matches the new state's own instruction (item E) without needing any edit | R1/background | — |
| 610–621 | Per-row DNABERT-2 ablation (max single-row −1.45%, full ensemble −25.5%) | **KEEP** | This is the real, established "distributed causal object" finding — the correct anchor for any future R2/R3 material, explicitly distinct from the missing "redundant pair" claim | Background for a future R2/R3 if/when pair evidence exists; usable now as "DNABERT-2's causal object is an ensemble, not a single row" | C-027, C-028 |
| 623–698 | GENERator 35-task table + bimodal discussion | **KEEP** | Unaffected — uses the EUK row (2,371), not the contested PROK row | R1/background, functional-recruitment contrast | C-026 |
| 700–715 | Figure 4 + caption | **KEEP, with the same NTv3-panel caveat as the table above** | Panel B is the NTv3 splice-per-seed figure | R1/background; panel B caption needs the contested flag | C-029 |

## Compression (old R7 → new R6)

| Lines | Content | Disposition | Why | New destination | Claims |
|---|---|---|---|---|---|
| 717–728 | Section intro, "SW proximity stratifies rows by compression sensitivity in a more graded manner" | **REWRITE** | Framing needs to lead with the new R6 derivation (C-033), not the old pruning-distance framing | R6 | C-033 |
| 730–747 | Pruning sweep — far-SW fragility (5.9× worse than near-SW at 20%) | **BLOCKED BY PROVENANCE (flagged conflict, not unilaterally resolved)** | This is a real, `X-001`-adjacent result already corrected once (from "near-SW tolerance/shadow redundancy" to "far-SW fragility" — see `PHASE_0_TRIAGE.md` B2). The new instruction (task item G) explicitly asks to retire *this* corrected version too, on the strength of an alleged layer-depth control that has **no artifact anywhere in this repo** (`NEW_DIRECTION_EVIDENCE_AUDIT.md` cross-cutting notes). The narrower, explicit "must not claim" guardrail list only forbids "SW proximity predicts pruning tolerance" and "shadow redundancy," which is arguably satisfied by the *already-corrected* far-SW-fragility framing — but the fuller instruction text is broader. This is a genuine, disclosed judgment call, not silently resolved either way. | Held for author decision: keep as the already-corrected, real result, or cut entirely pending a real layer-depth control | X-001, cross-cutting note in evidence audit |
| 749–760 | GENERator INT4 quantization-sensitivity result (near-SW tolerates INT4 better than random) | **MOVE TO SUPPLEMENT** | Real result, but the new R6 headline is the scale-endpoint derivation (C-033) + the whole-model exemption null (C-031), not this narrower per-fraction result; keeping it in main text would re-create the "many analyses" problem D-010 already warned against | Supplement | — |
| 762–785 | U_k-guided INT4 precision allocation on DNABERT-2 GUE tasks | **MOVE TO SUPPLEMENT, reframe if kept** | The "advantage" framing here (low-U_k rows tolerate INT4 better on SW-independent tasks) is a *different* claim from C-033/C-031 and is not itself contradicted, but it is exactly the kind of "SW-aware precision-allocation advantage" language task item G asks to retire the spirit of — supplement, and if retained, framed as a minor, task-specific observation, not a "principled criterion" | Supplement | — |
| 787–800 | **Whole-model INT4 with SW exemption** (+0.0879 EUK / +0.4664 PROK non-exempt; −0.0008/+0.0004 change with exemption; "protecting the SW row alone confers no measurable benefit") | **KEEP — this is the anchor for the new R6** | This is the real, already-in-manuscript empirical result behind C-031, and it is exactly what the new C-033 mathematical derivation now explains. No rewrite of the numbers needed — only the surrounding interpretation, which should shift from "contra Yu et al., graded picture" to "expected, given what per-row quantization's own math does" | **R6, primary anchor** | C-031, C-033 |
| 802–821 | Figure 5 + caption | Panel A: **held per the pruning conflict above**. Panel B: **MOVE TO SUPPLEMENT**. Panel C: **KEEP** | Matches the three items above | Panel C → R6; A/B per their own entries | — |

## Discussion

| Lines | Content | Disposition | Why | New destination | Claims |
|---|---|---|---|---|---|
| 823–841 | "The main mechanistic contribution of this work is the identification of ‖U_k‖_F..." | **REWRITE** | States U_k as "the main mechanistic contribution" — directly contradicts D-016/D-017 | Discussion, reframed around the transfer thesis | — |
| 842–854 | PROK sign-convention discussion, "a matched EUK SAE is in progress" | **BLOCKED BY PROVENANCE (PROK) / stale (EUK SAE status)** | Same PROK issue as lines 433–459; also, "in progress" is a stale status claim this session did not verify one way or the other | Held; EUK SAE status needs a fresh check before any rewrite states it as still in progress | N-015 |
| 856–867 | Splice/histone discussion | **KEEP** | Same as lines 602–608, already correctly hedged | R1/background | — |
| 869–881 | Compression discussion (far-SW fragility restated, INT4 precision-allocation "principled criterion") | **Same disposition as the corresponding Results paragraphs above** | Discussion restates, doesn't add new claims | R6 (INT4 exemption part) / held (pruning part) | X-001, C-033 |
| 883–end (Limitations) | Limitations block | **KEEP, REWRITE content** | Structurally worth keeping (already flagged as well-written in `CUT_LIST.md`); content needs updating to name the new BLOCKED sections (R2–R5) and the three `contested` claims as limitations of *this manuscript's current evidence*, not permanent facts | Discussion/Limitations | all `contested`/`BLOCKED` items above |

## Methods and Bibliography

| Lines | Content | Disposition | Why | New destination | Claims |
|---|---|---|---|---|---|
| ~900–1010 | Methods (model specs, protocols, precision disclosures) | **KEEP** | Factual protocol descriptions; precision disclosures for Evo1 bf16 and DNABERT-2's Triton fp16 attention were already added in the prior session (D-012/D-014/N-008) | Methods, unchanged | — |
| 1060–1086 | Bibliography | **KEEP, one entry flagged** | Ref [11] (Sun et al., `arXiv:2603.05498`) already flagged as unresolved/uncited in `REFERENCE_AUDIT.md`, unchanged this session — see `NEW_DIRECTION_EVIDENCE_AUDIT.md` Task 8 notes below | Bibliography | — |

---

## Summary — what a writing session can safely touch first

**Immediately draftable, no blockers:** R1 (reframed intro + predictor sections, lines
114–390 minus the "eight models"/shadow-redundancy phrases) and R6 (lines 787–800 as the
anchor, plus the new C-033 derivation).

**Needs one author decision before drafting:** the entire old-R4 PROK/EUK mechanism section
(lines 392–530) and the Discussion's PROK paragraph (842–854) — N-015. The NTv3 splice
material (lines 561–600, 707–710) — N-014. The pruning-sweep framing (lines 730–747,
802–811) — the far-SW-fragility conflict above.

**Cut outright, no further discussion needed:** the PROK SAE section (461–481), the
kingdom-specificity test (505–510), the "eight models" phrasing (4 sites), and the verbatim
shadow-redundancy clauses in the abstract and intro (111–113, 164).
