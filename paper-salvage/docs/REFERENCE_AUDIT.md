# REFERENCE_AUDIT.md

## Ref [11] — Sun et al. — **FLAGGED, needs external lookup. Not corrected.**

**Status: cannot be resolved from local sources. Do not guess.**

### What the manuscript says

`paper/main.tex:1071` (also `manuscript.txt:420`, `manuscript_prism.txt:795`,
`previous_main.tex:792`, and `paper/main_old.tex:898` as a URL):

> Sun, P.\ et al. The Spike, the Sparse and the Sink: Anatomy of Massive Activations and
> Attention Sinks. Preprint arXiv:2603.05498 (2026).

**The entry is never cited in the body.** `grep` for `Sun` across `paper/main.tex` returns
only the bibliography line. There is no in-text use to disambiguate what it was meant to
support.

### What the local authoritative source says

`docs/superweight_paper.txt` is the local copy of Yu et al. (2024), which is the paper that
introduces the super-weight framing this manuscript builds on. Its reference list, line 634:

> Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large
> language models. In ICLR 2024 Workshop on Mathematical and Empirical Understanding of
> Foundation Models, 2024. URL https://openreview.net/forum?id=1ayU4fMqme

### Why this is not a one-character fix

The triage entry (PHASE_0 §B5) assumes `2603.05498` should be `2402.17762`. Local evidence
does not support making that substitution:

1. **Different author.** The manuscript says *Sun, **P.***; the massive-activations paper is
   *Sun, **Mingjie***.
2. **Different title.** *"The Spike, the Sparse and the Sink: Anatomy of Massive Activations
   and Attention Sinks"* vs *"Massive activations in large language models"*.
3. **Different year.** 2026 vs 2024.
4. **The local bibliography gives no arXiv ID for Sun et al. 2024** — it cites an OpenReview
   forum ID, `1ayU4fMqme`. So `2402.17762` is asserted only in this project's own triage
   notes and is not corroborated by any local source.
5. `2603.05498` is a plausible March-2026 identifier and may well be a real, different paper.

So there are two distinct possibilities and local material cannot separate them: either the
entry is a corrupted reference to Sun M. et al. 2024, or it is a correct reference to a
genuine 2026 paper that simply is not cited anywhere in the text.

### What is needed

An external arXiv lookup of `2603.05498`, and of `2402.17762`, to establish which (if either)
matches the entry. **Flagged for the author. No change made.**

### Recommendation, not applied

Whichever way the lookup resolves, an uncited bibliography entry should be either cited or
removed.

---

## Figure 2 panels A–D — source data **exist**, but rendering is blocked on N-009

`paper/main.tex:368` states panels A–D "are pending GENERator JSONs and are omitted from
this render." The panels are *GENERator EUK/PROK activation lifecycle and step-up-layer
‖U_k‖_F bars*.

**The JSONs are present in the repository:**

| Panel content | File | Present |
|---|---|---|
| EUK activation lifecycle | `results/activation_lifecycle_generator.json` | ✓ |
| PROK activation lifecycle | `results/activation_lifecycle_generator_prokaryote.json` | ✓ |
| EUK step-up ‖U_k‖_F | `results/sw_mechanistic_generator.json` | ✓ |
| PROK step-up ‖U_k‖_F | `results/sw_mechanistic_generator_prokaryote.json` | ✓ |

So the caption's stated reason for omission is **out of date**.

**Not rendered, deliberately.** The PROK ‖U_k‖_F panel would be drawn from
`sw_mechanistic_generator_prokaryote.json` — the exact artifact that item 1 of this session
found **does not reproduce** from the current checkpoint at its own layer (stored value
2648.4773 / rank 1 vs recomputed 5.5106 / rank 1277; see
`experiments/E4_granularity/N009_RESOLUTION.md`). Publishing a figure panel from an
unreproducible artifact is a scientific call, not a rendering task.

EUK's artifact *does* reproduce (max/median 12.24 both, rank 1), so an EUK-only render is
defensible — but splitting a four-panel figure is also an authorial decision.

**Flagged for the author.** No panels rendered, no caption edited beyond leaving it as is.
Resolve N-009 first; the plotting inputs are ready and waiting.

---

## Addendum, 2026-08-13 — reference check for the new-direction framing (Task 8, no broad
literature search performed)

Per `DECISIONS.md` D-016/D-017, checked whether the manuscript's bibliography already carries
the specific citations the new framing needs. Repo-local sources only; nothing external was
looked up.

- **Yu et al. (super-weight paper)** — `arXiv:2411.07191`, already cited correctly and
  extensively (`paper/main.tex:1077-1078`, and as the local full-text copy
  `docs/superweight_paper.txt`). No issue.
- **Sun et al. (massive activations / amplifier prior art)** — unresolved, exactly as
  documented above (ref [11], `arXiv:2603.05498`, author-initial/title/year mismatch against
  the one local candidate, never cited in the body). This is now more consequential than
  before: D-017 explicitly reframes `‖U_k‖_F` as "substantially related to prior work by Sun
  et al.," so this citation moving from "flagged, uncited, low-stakes" to "flagged, uncited,
  load-bearing for the paper's own positioning claim" — **still not resolved by this pass**,
  still needs the same external arXiv lookup (`2603.05498` vs. `2402.17762`) recommended
  above. Do not guess.
- **Attention-sink prior art (Xiao et al., StreamingLLM-type work)** — **not cited anywhere**
  in `paper/main.tex`'s bibliography (`grep -n "Xiao\|attention sink\|StreamingLLM"` returns
  zero hits in the manuscript; the only local occurrences of "attention sink" language are
  inside the reproduced Yu et al. full text, `docs/superweight_paper.txt:790-794`, which cites
  "Xiao et al., 2024" and "Son et al., 2024" for attention sinks and separately reports that
  Yu et al. themselves found attention sinks persist even with super-weights removed — i.e.
  Yu et al.'s own paper already establishes the SW/attention-sink dissociation the new
  framing's guardrail relies on ("do not claim the SW causes the attention sink")). **Not
  needed immediately**: R4 (the section that would need this citation) is currently
  `BLOCKED, no artifact` per `PAPER_OUTLINE.md` v2 and is not being drafted this pass. If a
  future session obtains real GENERator attention-pattern data and drafts R4, it will need to
  add a proper citation for Xiao et al.'s original work — flagged here so it isn't forgotten,
  not resolved now (would require an external lookup this pass is not authorized to guess at).
