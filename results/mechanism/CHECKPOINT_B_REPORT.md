# ⏸ CHECKPOINT B — Phases B & C (Task Spec v2)

Session date: 2026-08-07. Follows `CHECKPOINT_A_REPORT.md` (A1 KILL, A2 real-but-dispersed).
Phase D not started (requires explicit opt-in).

---

## PHASE B — splice / GC confound: **TWO FINDINGS**

The splice collapse is **not** a composition side-effect. Three independent lines of
evidence, all pointing the same way.

**Pipeline validation first.** My initial run resolved `max_length` to 512 by looking up the
full task path in `_MAX_LEN`; the correct key is the task *leaf* (`"reconstructed"` → **80**),
which is what the checkpoints were fine-tuned at. Fixed. The corrected run reproduces the prior
result **exactly** — baseline **92.74%** (recorded: 0.9274), Δ **−25.54 pp** (recorded: −0.2554)
— so everything below is measured on a validated pipeline.

### B1 — Can composition alone separate splice classes? **No.**

| | value |
|---|---|
| Composition-only classifier (GC + 1/2/3-mer freqs, multinomial logistic regression) | **56.23%** |
| Majority-class baseline | 56.58% |
| MCC | **0.059** |

The composition-only model performs **below the majority-class baseline**, with MCC ≈ 0.
Nucleotide composition carries essentially **no** class information for this dataset. The
confound's premise — that splice classes are compositionally distinctive — is false here, which
already makes the confound implausible before B2/B3 are run.

### B2 — Is the collapse concentrated in GC outliers? **No.**

| GC quartile | range | n | baseline | Δacc |
|---|---|---|---|---|
| Q1 | 0.235–0.375 | 1113 | 95.15% | **−24.83 pp** |
| Q2 | 0.378–0.458 | 1155 | 93.54% | **−21.15 pp** |
| Q3 | 0.460–0.557 | 1146 | 92.18% | **−23.76 pp** |
| Q4 | 0.560–0.812 | 1148 | 90.16% | **−32.43 pp** |

The effect is large in **every** quartile (−21 to −32 pp) — there is no GC region where it
vanishes. A composition confound would predict the collapse concentrating at the GC extremes and
weakening in the middle; instead the middle quartiles collapse just as hard.

**Honest caveat:** this is not perfectly flat. Q4 (highest GC) is ~8–11 pp stronger than
Q1–Q3, and the 11.28 pp spread only clears my "uniform" threshold (12.77 pp) narrowly. The
correct reading is *"large and present across the whole GC range, somewhat stronger at high GC"* —
not *"perfectly uniform."* It does not change the verdict, because no quartile shows the
attenuation a confound requires.

### B3 — Does it survive composition matching? **Yes — it gets stronger.** ★ decisive

GC-matched subset built by fine-binning (20 quantile bins) and taking equal numbers per class per
bin, seed 42:

| | value |
|---|---|
| Balanced subset | n = 2,688 (896 per class) |
| Per-class mean GC after matching | 0.4714 / 0.4724 / 0.4718 (spread **0.001**) |
| Composition classifier on this subset | **34.38%** vs 33.33% majority — matching worked, composition is now uninformative |
| Baseline accuracy | 92.40% |
| **Δacc** | **−39.05 ± 1.44 pp** |
| **Fraction of full-set effect retained** | **152.9%** |

With composition information removed by construction, ablating the SW ensemble hurts **more**
(−39.05 pp) than on the unmatched set (−25.54 pp). The SW is not proxying for GC — if anything,
composition was *masking* part of its contribution.

### B4 — Decoder cross-check (supporting, non-decisive)

GENERator, the composition-carrying decoder, shows ≈ −0.05% on splice where DNABERT-2 collapses.
If the collapse were pure composition, the composition-carrying model's SW should also hurt splice.
Reported as **suggestive evidence against the confound only** — encoder/decoder differences
(architecture, objective, tokenizer, and the fact that GENERator was never fine-tuned on splice)
make it non-decisive.

### Verdict

**TWO FINDINGS.** The paper may claim both that the SW encodes nucleotide composition **and**
that the SW ensemble is functionally required for motif-dependent splice recognition. These are
distinct results, not one result described twice.

Deliverables: `splice_gc_confound.{json,csv}`.

---

## PHASE C — contamination audit (mandatory) ✅

### C1 index correction — done
- `results/super_weight_index.json` → `generator_prokaryote` now **L8/r260** (+ secondary r1325),
  with a full `provenance` block (old entry, why invalid, why the new one is right, references).
- Old entry archived → `results/super_weight_index_generator_prokaryote_ORIGINAL_SUSPECT.json`.
- **The layer moves too: L2 → L8.** Anything hard-coding `layer 2` for PROK is wrong independently
  of the row index.

### ⚠️ New finding — the 1B sibling is also suspect
`generator_prokaryote_1b` (L2/r1397) came from the **same `detection_run_sbatch` pipeline** with the
**same layer-2 signature** as the entry proven contaminated, and has never been re-detected with a
prokaryotic probe. Flagged in the index as `SUSPECT - UNVERIFIED`. It does not appear in the current
manuscript, so this is cheap to fix if ever needed — but it must not be cited as-is.

### Blast radius — full catalogue in `prok_contamination_audit.md`
- **9 result files** in `results/prokaryote/` (6 directly recording L2/r1927, 3 derived) → all must be rerun.
- **8 scripts** hard-code the contaminated coordinates.
- **README** (6 locations) and **`paper/main.tex`** (9 locations).

### The two manuscript claims that are actually in danger
1. **`r = −0.710` kingdom sign-reversal** (`main.tex:95`, `:434`) — derived from
   `sw_hexamer_causal.json`, computed on the contaminated row. The whole kingdom argument rests on it.
2. **PROK SAE** (`main.tex:447–465`, `:752–756`, `:925`) — `main.tex:925` states it was trained
   "on GENERator PROK at **layer 2**", i.e. on the contaminated layer's residual stream. Worse:
   **no SAE checkpoint exists anywhere under `results/`**, so the artifact is not reproducible from
   this repo regardless.

**Both are invalid until a full PROK rerun on L8/r260 (Task Spec D1) exists.** If D1 is not run,
they must be **cut, not softened** — there is no version of these claims that survives on the
current evidence.

One correction to the task spec's C1 list: it asks me to update the *"L4-vs-L2 layer confound"*
line in Limitations. No such sentence exists — the six current limitations don't mention the layer
comparison at all. So this is an item to **add** (now L4 vs L8), not edit.

---

## Where the paper stands (§7 decision tree, resolved)

```
A1 subspace retention ──► KILL              ✗ Paper A eliminated
A2 Evo1 fp64 ──────────► real-but-dispersed   (hypothesis level, no causal claim)
B  splice/GC ──────────► TWO FINDINGS       ✓ Paper B AVAILABLE
C  contamination ──────► done; kingdom + SAE claims quarantined
```

### ✅ Paper B is the defensible paper

> *Super-weights in genomic gated-FFN transformers encode nucleotide composition **and** the
> super-weight ensemble is functionally required for motif-dependent splice recognition
> (DNABERT-2 + NTv3), with characterized detection failure modes.*

Supported by: the composition result (EUK); the splice dissociation surviving three
composition-confound tests; NTv3 replication (5-seed, p = 0.008); and the failure-mode
methodology.

**What must NOT appear**, on this session's evidence:
- Any structural/systemic decomposition or retention-based mechanism claim (**A1 KILL** — retention
  fails at every dimensionality, and the metric ranks the *null* model above the *load-bearing* one).
- Any causal-gating claim for Evo1 (**A2** — rescue failed; hypothesis level only).
- The kingdom sign-reversal and the PROK SAE (**C** — contaminated; require D1 or removal).

### Failure modes for the methodology section (all three now quantified)
1. **Probe contamination** — PROK detected with a eukaryotic probe; wrong layer *and* wrong row.
2. **Retention ≠ load-bearing** — A1, including the inversion where the null model outranks the
   load-bearing one.
3. **Detection magnitude ≠ causal importance** — Evo1: a genuine ~1.25e6 activation (fp64-confirmed,
   mixer-amplified 24×) that does nothing when ablated.

---

## Optional next step

**Phase D1** — the full PROK biology rerun on L8/r260 (hexamer KL, four shuffle controls, motif
enrichment, GC correlation, retrained layer-8 SAE) is the only route to keeping the kingdom/SAE
material. Worth noting the likely outcome: if L8/r260 encodes composition the way EUK does, the
kingdom sign-reversal is **dead** (both kingdoms doing the same thing) — so D1 may well confirm a
cut rather than rescue the claim. Requires explicit opt-in.
