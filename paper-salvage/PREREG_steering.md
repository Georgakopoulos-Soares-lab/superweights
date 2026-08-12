# PREREG — E3, bidirectional steering of generated sequence composition

**Lock before running:** `python src/prereg_lock.py lock docs/prereg/PREREG_steering.md`

---

## Question

Does scaling the super-weight channel bidirectionally steer the nucleotide composition of
generated sequences, without degrading generation quality by a comparable amount?

## Why this matters

Every causal claim in v15 is destructive: remove the row, perplexity explodes. That is the
weakest form of causal evidence. A positive, dose-dependent, bidirectional intervention
would establish that the channel *controls* composition rather than merely correlating
with it.

## Design

**Models and targets**
- GENERator EUK: layer 4, row 2371 (dominant); optionally row 1522 (secondary)
- GENERator PROK: layer 2, row 1927
- PROK is expected to **mirror** EUK because its write is exclusively negative.

**Interventions** — run both:
- activation steering of the SW channel
- row scaling of W_down[k, :]

**Dose levels:** ≥ 5 conditions spanning both directions, e.g. {0.25, 0.5, 1.0, 2.0, 4.0}.

**Controls:** ≥ 10 matched random rows, at every dose level. Same seed structure as the
SW condition.

**Sampling:** ≥ 1,000 sequences per condition, fixed length, fixed sampling parameters,
fixed seeds across conditions.

## Measurements (per condition)

| Metric | Role |
|---|---|
| GC% | primary effect |
| 1-mer / 2-mer / 6-mer distributions | compositional vs. k-mer-specific |
| sequence complexity, repeat fraction | degeneration check |
| **generation entropy** | **primary discriminator** |
| perplexity / loss vs. baseline | degeneration check |
| KL vs. baseline output distribution | intervention magnitude |

## The discriminator

- **Steering:** GC shifts monotonically with dose; entropy stays flat; random rows do not
  move GC.
- **Degeneration:** entropy climbs alongside the GC shift → composition is being read off
  degraded output, not steered.

Monotonic + bidirectional (opposite signs for up- vs. down-scaling) + entropy-flat +
random-null is very difficult to explain as damage.

## Predictions (state before running)

- Direction of GC shift under up-scaling, EUK: ______
- Direction of GC shift under up-scaling, PROK: ______
- Expected entropy behaviour: ______
- _(Confidence: ___ / 5)_

## Pre-committed reporting (D-008)

**If the effect holds** → R5 headline:
> A single amplifier pathway bidirectionally steers generated sequence composition without
> comparably degrading generation quality.

**If GC only shifts once generation degenerates, or the effect is non-monotonic, or random
rows move GC comparably** → **the result is still reported**, in two sentences within R4:
> Scaling the SW channel altered generated composition only in conjunction with a
> comparable loss of generation quality [entropy / PPL numbers], so we do not interpret the
> channel as a controllable composition setpoint. Its activation is predictive of
> composition; we find no evidence that it is sufficient to set it.

This is not a file-drawer case. "Activation correlates with composition" and "the channel
controls composition" are different claims, and a null here says the first holds without
the second — which makes R4 more credible, not less.

---

_Locked: (filled by prereg_lock.py)_
