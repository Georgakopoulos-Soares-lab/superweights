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
