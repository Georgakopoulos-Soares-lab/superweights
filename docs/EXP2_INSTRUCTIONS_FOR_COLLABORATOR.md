# EXP2 — uniform detector + causal re-evaluation on the three 7B text decoders

**For:** the collaborator who already has `llama-7b`, `Mistral-7B-v0.1` and `OLMo-7B-0724-hf`
weights on disk.
**From:** the paper-closing session. Companion item (the within-model graded sweep) is already
done — see [`results/paper_closing/within_model_slope_comparison.json`](../results/paper_closing/within_model_slope_comparison.json).
**Effort:** three commands, ~15 min of GPU each, one GPU, no training, no downloads if the
weights are cached.
**Everything you need is in this repo.** You should not have to read any other document.

---

## 1. Why this needs doing

`audit/detector_provenance.csv` records how each of the 22 census candidates was selected.
**16/22 used the current rule** — global argmax of the *layer-relative activation ratio*

```
ratio(l,r) = max_t |a[l,r,t]| / median_r' ( max_t |a[l,r',t]| )
```

**6/22 used a legacy rule** — global argmax of the *absolute* activation, from a
pre-ratio-rule protocol. Of those six:

| model | frozen coord | legacy-vs-ratio check | outcome |
|---|---|---|---|
| GENERator-EUK-3B | L4 / r2371 | done | ✅ agree (ratio rank 1) |
| NTv3 | L11 / r1472 | done | ❌ **disagree** — frozen is ratio-rank 3; argmax is L6/r1472 |
| DNABERT-2 | L5 / r603 | done | ❌ **disagree** — argmax is L8/r603 (ratio 302 vs 153) |
| **Llama-7B** | **L2 / r3968** | **never run** | ❓ ← you |
| **Mistral-7B** | **L1 / r2070** | **never run** | ❓ ← you |
| **OLMo-7B-0724-hf** | **L1 / r269** | **never run** | ❓ ← you |

Two of the three checked legacy models turned out **not** to be the ratio-argmax. So we
cannot currently write "the census applied one uniform selection rule" — it is unverified for
3/22 rows, and every cohort-level statement conditioned on the selection rule inherits that
gap. This experiment closes it.

**Both outcomes are fine and neither is a crisis.** Please do not tune anything to get one.

* **frozen == ratio-argmax** → legacy and current rules agree here; provenance closes; the
  census row stands as published.
* **frozen != ratio-argmax** → the DNABERT-2 / NTv3 pattern again. The script then causally
  evaluates *both* coordinates on the *same* eval pool, which tells us whether picking the
  "wrong" coordinate changed any reported number. For DNABERT-2 and NTv3 it did not.

---

## 2. Prerequisites

**GPU.** One device, **≥ 48 GB** recommended (weights are loaded in fp32: ~28 GB for a 7B,
plus activations and full-vocabulary logits). A 40 GB A100 will probably fit; an 80 GB card
certainly does. No multi-GPU needed.

**Weights.** Pinned revisions — please use these exactly, they are what the census resolved:

| model | HF repo | revision |
|---|---|---|
| Llama-7B | `huggyllama/llama-7b` | `4782ad278652c7c71b72204d462d6d01eaaf7549` |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` |
| OLMo-7B-0724-hf | `allenai/OLMo-7B-0724-hf` | `1ee306df318ee15bfe4a76ebd5c002b0105b1ab6` |

If your cache has a *different* revision, run it anyway and **tell us** — the reproduction
gates in §4 will fail loudly and that failure is itself informative. Do not silently
substitute a revision.

**Python.** Anything that loads these three models. Verified here on
`torch 2.11.0+cu130`, `transformers 5.5.0`, plus `numpy`, `scipy`, `pyarrow`.
`datasets` is optional (see §3).

**Eval corpus.** WikiText-2 raw, test split. The script tries
`datasets.load_dataset("wikitext","wikitext-2-raw-v1","test")` first. If you have no network
or no `datasets`, it falls back to a local parquet and reproduces
`e10_lib.build_windows` line-for-line (same non-empty filter, same `random.Random(42)`
shuffle, same fill-to-512 concatenation):

```
frozen_inputs/wikitext_repo/wikitext-2-raw-v1/test-00000-of-00001.parquet
sha256 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91
```

That file is committed to the repo. Both paths give identical windows — this is the same
file the census used, not a substitute corpus. Override with `--wikitext_parquet` if you keep
it elsewhere.

---

## 3. What to run

```bash
cd /path/to/superweights

# 0. HARNESS CHECK FIRST -- ~45 s, 1.7B model, downloads ~3.2 GB if not cached.
#    SmolLM2-1.7B was selected by the CURRENT rule, so this MUST print
#    "uniform rule AGREES" and pass all four gates. If it does not, stop and tell us:
#    the harness is broken and the 7B numbers would be meaningless.
python scripts/paper_closing/run_uniform_detector_text.py --model smollm2-1.7b

# 1-3. the three real targets, one per invocation, any order
python scripts/paper_closing/run_uniform_detector_text.py --model llama
python scripts/paper_closing/run_uniform_detector_text.py --model mistral
python scripts/paper_closing/run_uniform_detector_text.py --model olmo
```

Add `--local_files_only` if the machine is offline and the weights are already cached.

Each run writes `results/paper_closing/uniform_detector_text_<model>.json`. Nothing else is
touched — **the frozen census is never modified**, and no existing result file is overwritten.

### What the script actually does

1. Loads the model in **fp32** at the pinned revision (fp32 because the census endpoint is
   fp32; fp16 would move the loss in the 4th decimal and break the gates).
2. Builds the census's **frozen 100-window WikiText-2 eval pool** (512 tokens each) and takes
   the first 24 windows as the detector's inputs.
3. Hooks **every** layer's `model.layers.{i}.mlp.down_proj` output and records per-row
   `max_t |a|`, then computes the layer-relative ratio and takes the **global argmax over all
   layers × rows** — the uniform rule, at full scope. Reported under **three** tokenization
   conventions (census ids verbatim, `add_special_tokens=True`, `add_special_tokens=False`)
   because the rule requires declaring and reporting both.
4. Reports the **rank and ratio of the frozen coordinate**, so "is it the argmax?" is
   answerable straight from the JSON without anyone re-running this.
5. Causally evaluates the frozen coordinate at ε=0.5 and ε=1.0 (row scaled by α=1−ε, so
   ε=1.0 is full ablation), weighted-mean causal-LM NLL over the 100 windows.
6. **Only if the rules disagree**, evaluates the uniform coordinate at both ε *and* 5 seeded
   controls in its layer, on the same pool, so the comparison is paired and internally valid.

Row selection is by activation ratio **only** — never by causal outcome.

---

## 4. The four reproduction gates (read this if a run aborts)

The endpoint, eval pool, batch size, ablation parameterization and control-draw RNG are all
the census's own, so four **stored** quantities must reproduce before any new number is
trusted. The script checks them automatically and **aborts** on mismatch:

| gate | source |
|---|---|
| `baseline_loss` | `audit/census_master.csv` |
| `R_cand_eps1.0` at the frozen coordinate | `audit/census_master.csv` |
| `R_cand_eps0.5` at the frozen coordinate | `audit/census_master.csv` |
| the 5 stored control rows | redrawn from `SeedSequence(42).spawn(23)[panel_index]` |

Expected values, for reference:

| model | baseline_loss | R_cand ε=1.0 | R_cand ε=0.5 | control rows |
|---|---|---|---|---|
| Llama-7B | 2.2452528551553326 | 3.007945593585156 | 0.03855088724231734 | 892, 2034, 2379, 3729, 3751 |
| Mistral-7B | 2.207224828193799 | 2.871492377308394 | 3.013789457796915 | 190, 305, 746, 1912, 2687 |
| OLMo-7B | 2.4953157010610325 | 1.1177580569022323 | 0.006288051768048048 | 292, 437, 2877, 2908, 3160 |
| SmolLM2-1.7B *(check)* | 2.6145340592893835 | 6.988738472947954 | 0.03905575617362741 | 317, 582, 1422, 1547, 2043 |

Tolerance is 2 % relative. On this machine the 1.7B check reproduced to `1.5e-08`,
`1.6e-07`, `1.5e-08` and exactly, so a real pass is *far* inside tolerance — a marginal pass
is a red flag, not a pass.

**A gate failure means the harness is wrong, not that the science changed.** Likely causes,
in order: wrong revision; fp16/bf16 instead of fp32; a different WikiText copy; a tokenizer
that adds/omits BOS differently from the census. Please send us the log rather than working
around it. `--no_gate` records the misses instead of aborting — **triage only**, do not
report numbers produced under it.

---

## 5. What to send back

1. The three JSONs from `results/paper_closing/`.
2. The console logs (they contain the gate table and the detector lines).
3. `nvidia-smi`, `python -c "import torch,transformers;print(torch.__version__,transformers.__version__)"`,
   and how you obtained the weights (repo + revision actually resolved).
4. Anything you had to change, however small.

The single most important line per model is this one:

```
[census_ids] uniform argmax = L<l>/r<r> ratio <x> | frozen L<L>/r<R> ratio <y>
             rank <k>/<n> | frozen_is_argmax=<True|False> | argmax stable in <p>% of inputs
```

For reference, the verified 1.7B positive control printed:

```
[census_ids] uniform argmax = L7/r227 ratio 3181.70 | frozen L7/r227 ratio 3181.70
             rank 1/49152 | frozen_is_argmax=True | argmax stable in 100% of inputs
```

---

## 6. Please don't

* Don't change `--n_act_windows`, `--n_eval_windows`, the batch sizes, or the seed. They are
  the census's; changing them breaks the gates and makes the result incomparable.
* Don't edit `audit/census_master.csv` or `audit/detector_provenance.csv`. We update those
  from your JSONs once the gates pass.
* Don't re-run with a different threshold to "get" agreement. The 5.0 accept threshold is the
  detector's own, fixed since E11.
* Don't drop a result because it disagrees. Disagreement is the more interesting outcome and
  it is already the majority pattern among the checked legacy models (2 of 3).

---

## 7. Context: what the companion experiment found

The within-model graded sweep (the other half of this pair, already run) asked whether the
activation ratio *grades* causal severity or merely *detects* super rows. Same design in a
genomic decoder (GENERator-EUK-3B, L4/r2371) and a text decoder (SmolLM2-1.7B, L7/r227):
36 rows log-spaced by ratio **rank** inside one layer, each ablated, native LM NLL.

| | GENERator-EUK-3B (genomic) | SmolLM2-1.7B (text) |
|---|---|---|
| ρ, all rows | +0.454 (p=5.4e-3, n=36) | +0.625 (p=4.6e-5, n=36) |
| ρ, excl. frozen candidate | +0.406 (p=1.6e-2, n=35) | +0.592 (p=1.8e-4, n=35) |
| ρ, ratio < 5 only | +0.040 (p=0.86, n=21) | **−0.584** (p=5.4e-3, n=21) |
| ρ, ratio ≥ 5 only *(post hoc)* | +0.639 (p=1.0e-2, n=15) | **+0.975** (p=7.1e-10, n=15) |
| candidate ÷ median sub-threshold damage | 6.1e4 × | 5.4e5 × |

**Two-regime, replicated across domains:** below the detector's accept threshold the ratio
carries no positive graded signal (flat in genomic, mildly *inverted* in text); above it the
ratio grades severity strongly, and survives removing the frozen candidate (+0.556 / +0.969).
The mediocre pooled ρ we reported earlier is a dilution artifact of averaging a noise regime
with a graded one. The sub-/supra-threshold split is flagged **post hoc** everywhere it
appears — the split point is the detector's own pre-existing rule and was not fitted, but the
decision to decompose came after seeing the scatter.

Figure: `results/paper_closing/fig_within_model_slope_2panel.{png,pdf}`.

### One more thing the sweep turned up, which bears on your run

Sweeping 36 rows instead of the census's 1 found a **second** independently catastrophic row
in the same layer of SmolLM2-1.7B, and an interaction:

| row | ratio | ratio rank | rel. NLL increase, α=0 |
|---|---|---|---|
| 227 (frozen census candidate) | 3181.70 | 1 | **+6.9887** |
| **161** (in no census artifact) | 63.54 | **4** | **+2.8718** |
| 749 | 358.56 | **2** | +0.0221 |

The row ranked **4th** by activation ratio is **130× more damaging** than the row ranked 2nd.
And ablating row 749 — harmless on its own — **abolishes** row 161's catastrophe
(joint 161+749 = +0.0207, i.e. 0.7 % of the sum of singles). Re-verified by an independent
implementation, weight drift 0.00e+00. Geometrically these are the layer's extremes:
cos(161,749) = +0.3952 (z = +17.6, the maximum of all 19,900 pairs among the 200
highest-norm rows), cos(227,749) = −0.3980 (the minimum). No mechanism is claimed yet.

**Why this matters for what you're running:** it is a concrete case where the ratio-argmax and
the *most causally important* coordinate come apart within a single layer. So if your run
reports `frozen_is_argmax=False` for one of the three models, that on its own does **not**
mean the census picked a functionally wrong row — which is exactly why the script also
causally evaluates both coordinates on the same pool. Please report the ratio result and the
causal result separately and let us reconcile them; don't collapse them into a verdict.

Artifacts if you want to look: `results/paper_closing/smollm2_second_row_epistasis.json`,
`results/paper_closing/smollm2_161_749_geometry.json`, and §2c of
[`results/paper_closing/PAPER_CLOSING_REPORT.md`](../results/paper_closing/PAPER_CLOSING_REPORT.md).
