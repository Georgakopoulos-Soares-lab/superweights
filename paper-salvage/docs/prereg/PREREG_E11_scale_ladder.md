# PREREG — E11: decoupling encoder/decoder architecture from parameter scale

**Lock before any measurement:**
`python3 src/prereg_lock.py lock docs/prereg/PREREG_E11_scale_ladder.md`

**Verify before and after measurement:**
`python3 src/prereg_lock.py verify --all`

## Why this experiment exists, and exactly what claim it tests

C-034 (`CLAIMS_LEDGER.md`: `q1` tracks encoder/decoder organization more than text-vs-genomic domain, 7 decoders `q1` 0.90-0.99, 4 encoders `q1` 0.39-0.90) is confounded with scale as currently measured: every decoder in the panel is ≥3B params, every encoder ≤650M. A reviewer can attribute the entire pattern to "spectral concentration emerges with scale," consistent with the massive-activation literature this manuscript already cites.

**This experiment does not test the manuscript's headline claim.** Per D-027 (2026-08-23), the headline architecture-level finding is now causal-response complexity (C-046/C-047: single-component-dominant decoders vs. pair-terms-required encoders, established via E9/E10/E10b's causal tomography), not `q1` magnitude. C-034 was already reported hedged at birth (2026-08-14: "not a clean universal domain split") and is confirmatory/calibration material under D-017's standing rule, not a novel predictor. **E11 tests whether that secondary, already-qualified claim is worth retaining in the paper at all**, not whether a strong claim survives a challenge. Accordingly, the decision rule below (see "Decision rules") includes outright deletion as a legitimate, expected outcome — this is not a claim that gets a rescue paragraph if it weakens.

## Critical constraint: `U_k` is only defined for gated FFNs

**Gated-FFN classification procedure (frozen, run before any model is added to the ladder):**

1. Fetch the candidate's `config.json` (not the weights) from the pinned revision.
2. Inspect for a gated FFN signature: two projections from `hidden_size` to `intermediate_size` that are combined multiplicatively before the down-projection — either as separate `gate_proj`/`up_proj` weights (Llama/Mistral/Qwen/GENERator-style `SwiGLU`), or a single packed `gate_up_proj` of shape `[2*intermediate_size, hidden_size]` (Phi-3-style), or the encoder-side GLU/GeGLU equivalent (MosaicBERT/ModernBERT/DNABERT-2/NTv3-style, module names vary — inspect `named_modules()` directly, not just the class name).
3. Reject (do not approximate with a fallback) if the FFN is a classic two-matrix MLP: one up-projection, a pointwise nonlinearity (gelu/relu), one down-projection, with no elementwise gating between two projections.
4. Record the classification (`gated` / `non-gated`, plus the exact module-name pattern used) as a column in `results/E11/scale_ladder.csv` for every candidate considered, not only accepted ones.
5. Do not infer gating from the model family name or from prior knowledge of a sibling model — verify each checkpoint's own config, since gating can change across a family's releases (e.g., an architecture revision between sizes).

## Model selection

**Model list, HF repo IDs, and pinned revisions are the result of a live Hugging Face Hub search performed as part of this same protocol-drafting pass** (per the working instruction not to assume from memory), with gating verified from each model's actual source/`modeling_*.py` where the config's `hidden_act`/intermediate-size ratio alone would be ambiguous or misleading (flagged per-row below where this mattered). One model (GENERator PROK, both sizes) has its gating inferred from a Llama-superset config schema rather than a directly-fetched `modeling_generator.py` — this is resolved for real at measurement time by this protocol's own classification procedure (step 1-2 above), not assumed permanently from this research pass.

### Decoder ladders

**Qwen2.5** (first ladder; 7B already in panel, no change to it):

| Repo | Revision (pin) | hidden_act | hidden_size | intermediate_size | layers | total params | non-embed params (est.) |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | silu | 896 | 4864 | 24 | 494,032,768 | ~358M |
| `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` | silu | 1536 | 8960 | 28 | 1,543,714,304 | ~1.31B |
| `Qwen/Qwen2.5-3B` | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | silu | 2048 | 11008 | 36 | 3,085,938,688 | ~2.77B |
| `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` (existing panel pin, unchanged) | silu | 3584 | 18944 | 28 | 7,615,616,512 | ~6.53B |

Note: 7B has untied embeddings; 0.5B/1.5B/3B are tied — a real architectural discontinuity within this family, recorded as a column, not smoothed over.

**OLMo-2** (second required family — confirmed architecturally and lineage-distinct from the existing panel's OLMo-7B v1 (`allenai/OLMo-7B-0724-hf`, `model_type: olmo`, `vocab_size: 50304`, `rope_theta: 10000`, has `clip_qkv`): OLMo-2 is `model_type: olmo2`, `vocab_size: 100352`, `rope_theta: 500000`, no `clip_qkv`, GQA at 32B — different tokenizer and attention design, not the same checkpoint lineage, so it stands as its own independent ladder, not an extension of the existing OLMo-7B panel entry):

| Repo | Revision (pin) | hidden_act | hidden_size | intermediate_size | layers | total params |
|---|---|---|---|---|---|---|
| `allenai/OLMo-2-0425-1B` | `a1847dff35000b4271fa70afc5db10fd29fedbdf` | silu | 2048 | 8192 | 16 | 1,484,916,736 |
| `allenai/OLMo-2-1124-7B` | `7df9a82518afdecae4e8c026b27adccc8c1f0032` | silu | 4096 | 11008 | 32 | 7,298,617,344 |
| `allenai/OLMo-2-1124-13B` | `3fefddc1bf18a30e1d9b91000271630718f2aa8b` | silu | 5120 | 13824 | 40 | 13,716,198,400 |

**Capped at 13B, 32B excluded — author decision, 2026-08-23, cost.** `allenai/OLMo-2-0325-32B` was identified and config-verified during model selection (silu, `hidden_size=5120`, `intermediate_size=27648`, 64 layers, 32,234,279,936 params, same recipe lineage per arXiv:2501.00656) but is not included in this ladder: it crosses the cost-control threshold by a wide margin for one rung, and 1B/7B/13B already satisfies the ≥3-sizes requirement. If a reviewer specifically presses on the top of the decoder scale range, 32B remains available as a documented, not-yet-measured extension, not a gap this pass failed to find.

Recipe-sharing is confirmed for 7B/13B directly from the OLMo-2 paper (arXiv:2501.00656: identical two-stage OLMo-Mix-1124/Dolmino-1124 corpus, with 13B/32B both described as scaling the same recipe used for 7B). The 1B (`0425`) is a later release presented as the same family/architecture lineage but its data-mixture identity with the 7B/13B trio was **not** directly confirmed — recorded as a soft spot in the "shared recipe" premise for this specific rung, not for the family as a whole.

**SmolLM2** (third, bonus decoder family — included specifically to fill the small-decoder range that mirrors the existing encoders' own scale (110-650M), which no other decoder ladder reaches; without it, the within-family slope check only ever operates among already-large decoders and cannot speak to whether *architecture* still separates from *domain* at the scale where encoders actually live):

| Repo | Revision (pin) | hidden_act | hidden_size | intermediate_size | layers | total params |
|---|---|---|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M` | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | silu | 576 | 1536 | 30 | 134,515,008 |
| `HuggingFaceTB/SmolLM2-360M` | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | silu | 960 | 2560 | 32 | 361,821,120 |
| `HuggingFaceTB/SmolLM2-1.7B` | `effd688a12921b4cc83e3312b6feb579f70f9c71` | silu | 2048 | 8192 | 24 | 1,711,376,384 |

Same tokenizer/vocab (49152) across all three, confirming shared recipe; `rope_theta` differs slightly (100000 for 135M/360M vs. 130000 for 1.7B) — recorded, not treated as disqualifying.

Rejected decoder candidates: **StableLM-2** (only `1.6b`/`12b` base sizes exist in the org — confirmed via full org listing scan; a real gated SwiGLU family, but moot at 2 sizes). **Gemma-2** (`google/gemma-2-2b/config.json` → HTTP 401, still license-gated, no valid HF token in this environment — excluded for access reasons, not architecture, consistent with this repo's existing note).

### Encoders at larger scale

This search materially changes the experiment's expected outcome space: a genuine gated bidirectional encoder family **above 1B was found** (EuroBERT), so the "if you cannot find a gated encoder above ~1B, say so explicitly" contingency is not triggered — it is retained as a decision-rule fallback only, not the expected result.

| Repo | Revision (pin) | Domain | hidden_act | hidden_size | intermediate_size | layers | total params | Gating evidence |
|---|---|---|---|---|---|---|---|---|
| `EuroBERT/EuroBERT-210m` | `39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e` | text | silu | 768 | 3072 | 12 | 310,266,624 | source: `down_proj(act_fn(gate_proj(x)) * up_proj(x))`, `is_decoder: false` |
| `EuroBERT/EuroBERT-610m` | `d9af784ed20db6c2096e335ec6a67dd4a219924c` | text | silu | 1152 | 4096 | 26 | 755,625,600 | same source class, confirmed bidirectional |
| `EuroBERT/EuroBERT-2.1B` | `81245a4d71f43452badf5e04458e4ddb831ff109` | text | silu | 2304 | 6144 | 32 | 2,403,092,736 | same architecture class per config (shared `EuroBertForMaskedLM`) |
| `answerdotai/ModernBERT-large` | `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | text | gelu (GeGLU) | 1024 | 2624 | 28 | 395,881,664 | source: `input, gate = self.Wi(x).chunk(2, dim=-1); Wo(act(input) * gate)` |
| `Alibaba-NLP/gte-large-en-v1.5` | `104333d6af6f97649377c2afbde10a7704870c7b` | text | gelu (misleading — see note) | — | 4096 | — | 434,139,136 | source: unconditional `NewGatedMLP`, packed `up_gate_proj` split in two, gated |

**Note on gte-large-en-v1.5**: its config reports `hidden_act: gelu` with `intermediate_size` = 4× hidden_size, the classic *non-gated* ratio — a naive config-only heuristic would wrongly reject it. Its actual `modeling.py` shows an unconditional `NewGatedMLP` (packed `up_gate_proj`, split into gate/up, gated before `down_proj`). **This is exactly why step 2 of the classification procedure requires inspecting `named_modules()`/source, not `hidden_act` alone** — this model is the concrete case that validates that requirement, not a hypothetical.

EuroBERT's three sizes give it its own directly-measurable within-family slope, exactly like a decoder ladder — this is the strongest single addition this search produced. GTE-large is included as an added encoder data point (different codebase/training than EuroBERT/ModernBERT) purely for regression leverage; it does not form its own ladder (only one size checked).

NTv3: confirmed **no larger sibling exists** anywhere in InstaDeepAI's org (full org listing scanned, 77 repos; the NTv3 line tops out at 650M across context-length variants). Stays at its existing 650M panel entry.

Rejected encoder candidates (config- and, where noted, source-verified):

| Candidate | Why rejected |
|---|---|
| `BAAI/bge-large-en-v1.5` | Standard `BertModel`, `hidden_act: gelu`, intermediate=4× hidden — confirmed non-gated `BertIntermediate`/`BertOutput`. |
| `BAAI/bge-m3` | `XLMRobertaModel`, `hidden_act: gelu`, intermediate=4× hidden — confirmed non-gated. (Param count ≈568M is a file-size estimate, not tensor-verified — moot since rejected on architecture.) |
| `facebook/xlm-roberta-xl` | Confirmed non-gated (`hidden_act: gelu`, 4× intermediate ratio). Notable: this is a genuinely large encoder (3.48B params, confirmed) — it illustrates the paper's scarcity point directly: large encoders exist, gated ones are what's rare. |
| `chandar-lab/NeoBERT` | Confirmed **gated** (source: `xformers.ops.SwiGLU`, compute-rescaled intermediate size 2048) but only 245M total params — below this protocol's 300M floor. Logged as a near-miss, not included; if the floor is ever relaxed this is the first candidate to add. |

### Genomic ladder

`GenerTeam/GENERator-v2-prokaryote-1.2b-base` (rev `8b2f768b0d293953518ff91d34600f9322ef1f94`, `hidden_size=2048, intermediate_size=5632, layers=26, silu`, 1,162,061,824 params) and `GenerTeam/GENERator-v2-prokaryote-3b-base` (rev `b18ac86df77359d894d7bc050cea78e2d0713021`, `hidden_size=3072, intermediate_size=8448, layers=30, silu`, 2,998,262,784 params) — both already locally configured, config schema is a strict Llama-config superset (`model_type: llama`, custom `GENERatorForCausalLM` remote-code class). **Gating is confirmed from config schema only in this research pass, not from a directly-fetched `modeling_generator.py`** — this protocol's own step 1-2 classification procedure re-verifies this from source before either checkpoint is used, per this document's own rule against inferring gating from family name/schema alone.

Full org scan (`GenerTeam`, 10 repos) confirms no additional GENERator EUK/PROK size exists below 1.2B or above 3B (the "v1" non-"v2" 1.2B/3B eukaryote checkpoints are older releases of the same sizes, not new rungs, and are not used).

**Bonus genomic encoder** (not a ladder — single size per domain, included as an added encoder data point since the existing panel has zero genomic encoders besides NTv3): `GenerTeam/GENERanno-prokaryote-0.5b-base` and `GenerTeam/GENERanno-eukaryote-0.5b-base` (rev `d02db0f24f2c62fa1efde760217cdf75771b0228` / `1f2d71462e07078391b0af4442c26ddce9f2287e`, `hidden_size=1280, intermediate_size=3520, layers=28, silu`, 493,395,200 params each). Confirmed bidirectional (`is_causal` defaults false) and gated (source: `GenerannoMLP` with `gate_proj`/`up_proj`/`down_proj`) — a different model class (`GenerannoForMaskedLM`, MLM objective) from GENERator, not a size variant of it.

### Cost-control flag for the measurement phase (not resolved here)

This ladder's largest rungs — `Qwen2.5-7B` (already in panel), `OLMo-2-1124-7B`, `OLMo-2-1124-13B`, `Qwen2.5-3B`, `GENERator-v2-prokaryote-3b-base` — exceed the ~3B cost-control threshold in this document's own "Cost control" section (`EuroBERT-2.1B` is just under it). These are already known and were surfaced to the author before measurement began (author instruction 2026-08-23: proceed through this known set without a fresh per-model confirmation; check in again only if a model or resource need outside this already-flagged set comes up, or if a prereg lock fails verification).

Every new model gets a pinned revision SHA. Existing panel models already lacking a pin (per E10's `PANEL` dict: Llama-7B, Mistral-7B, OLMo-7B-0724-hf all have `revision=None`) are **not** retroactively archaeology-hunted for the exact untracked commit that produced their original numbers. If any of these is reused as a ladder anchor (e.g. if OLMo is used as a second decoder family), it is pinned **now**, to the current default-branch commit, with an explicit note in `results/E11/scale_ladder.csv` and `E11_summary.md` that this pin postdates, and may not be bit-identical to, whatever untracked checkpoint state produced the existing OLMo panel numbers this ladder is being compared against.

## Per-model measurement (reused pipeline, unchanged)

1. **Detection**: `DownProjRecorder` + ratio/threshold pattern from `run_phase1_detection.py` (`RATIO_THRESHOLD=5.0`, global max output activation vs. layer-median channel max). Record the ratio even on failure; failures are detection nulls, not silent drops.
2. **Spectral metrics**: `spectral_lib.row_spectral_metrics(W_gate, W_up, w_down_row)` → `SpectralMetrics(q1, pr_spec, stable_rank, sigma1, frob_norm, n_singular_values)`, imported unmodified from `experiments/E7_exact_dimensionality/spectral_lib.py`. `frob_norm` is `‖U_k‖_F` exactly (asserted equal in the source module).
3. **Control rows**: extend the `numpy.random.SeedSequence(42).spawn(N)`-then-`rng.choice(pool, size=5, replace=False)` idiom (`E8_encoder_decoder/run_detection_and_spectral.py`, `E10_nlp_architecture_causal/generate_decoder_controls.py`), with `N = len(full E11 ladder)` and a single fixed panel ordering declared in `results/E11/scale_ladder.csv`'s own header before any control row is drawn (the spawn index is the model's position in that fixed ordering — the same convention already used for panel sizes of 2 and 5).
4. **Record per model**: total params, non-embedding params, `d_model`, FFN intermediate size, number of layers, candidate layer index, relative depth (candidate layer / total layers), layer median `‖U_k‖_F`, candidate's ratio to it, gated-FFN classification (from the procedure above).

## Analysis (frozen)

1. **Primary plot**: `q1` vs. `log10(non-embedding params)`, colored by architecture class (decoder/encoder), shaped by domain (text/genomic), with within-family lines connecting sizes in the same ladder.
2. **Within-family slope**: Δq1 per decade of log10(params), per ladder, reported with direction and magnitude — not pooled across families into one number.
3. **Architecture term after conditioning on scale**: OLS `q1 ~ log10(non_embedding_params) + is_decoder`, report coefficients, standard errors, R². Report the partial correlation of `is_decoder` with `q1` controlling for `log10(params)`. Explicitly caveated as descriptive (n on the order of 15-25, models not exchangeable), matching the manuscript's existing statistical stance for C-034/C-046.
4. **Both magnitude rankings**: candidate `‖U_k‖_F` as absolute value and as ratio-to-layer-median; note any rank disagreement between the two (feeds section C below).
5. **Depth control**: `q1` vs. relative depth; flag if large models' candidates sit at systematically different relative depths than small models' (a second, independent confound if so).
6. **Candidate-vs-control gap by scale**: candidate `q1` minus mean control `q1`, at every scale point; check whether this gap is scale-dependent or stable. A stable gap alongside scale-dependent absolute `q1` is reported as the most defensible remaining version of an architecture-linked (as opposed to purely scale-linked) effect, since it does not depend on the disputed OLS specification.

## Decision rules (frozen, three branches — not two)

Because C-034 is calibration material, not the headline, the branches below include outright removal, not only "soften the language":

- **Branch 1 — keep, cite E11 as the check**: the architecture coefficient (`is_decoder`) survives conditioning — `|coefficient| / SE > 2` — **and** the within-family slopes are small enough that the predicted `q1` change across each family's own observed parameter range is less than half the raw architecture gap (median decoder `q1` − median encoder `q1`) in the unconditioned data. Report C-034 as calibration material with E11 cited as the scale-confound check that it survived.
- **Branch 2 — cut C-034 from the paper**: `|coefficient| / SE < 1`, or the raw architecture gap shrinks by more than 50% once conditioned on scale (i.e. scale explains most of what looked like an architecture effect). Do **not** rewrite C-034 into a softer claim — remove it. A secondary claim that needs a rescue paragraph is not worth the page space; the paper already has its architecture-level headline (C-046/C-047) and does not need q1-by-architecture to carry any weight.
- **Branch 3 — demote to a one-line supplementary footnote**: neither of the above thresholds is met cleanly (`1 ≤ |coefficient|/SE ≤ 2`), or the sample is too thin to trust either direction (e.g. zero eligible gated encoders found above ~1B, leaving the encoder side of the regression dominated by 1-2 points). The footnote states plainly that the scale confound was checked and found inconclusive, points to `results/E11/`, and does not appear in the main text.

If encoders cannot be sampled above ~1B at all, this is reported as an explicit range restriction regardless of which branch the regression itself lands in — it constrains what any of the three branches can claim, and Branch 3 is the likely default in that case unless the within-available-range (e.g. ModernBERT-base → ModernBERT-large) within-family slope is itself decisively flat or decisively steep.

## Cost control

Every model in the final ladder that exceeds ~3B parameters (i.e. GENERator PROK 3B, and Qwen2.5-3B/7B already partly measured) triggers the existing cost-control rule: print the planned model list, estimated memory, and estimated wall-clock, and get explicit confirmation before loading, before any such model is loaded in this experiment.

## Deliverables

- `results/E11/scale_ladder.csv` — one row per model considered (including rejected/non-gated candidates, per the classification procedure), all fields from "Per-model measurement."
- `results/E11/scale_ladder_controls.csv` — control rows, all models.
- `figures/E11/q1_vs_scale.pdf`
- `results/E11/regression_summary.json` — OLS coefficients/SEs/R², partial correlation, per-family slopes.
- `results/E11/E11_summary.md` — 3-4 paragraphs, drop-in ready, stating which of the three decision-rule branches occurred and why, plus the exact sentence for Limitations if the encoder range is restricted. Explicitly states that this experiment tests C-034 (calibration material), not the paper's architecture-level headline (C-046/C-047), so a reader does not conflate the two.

## Pre-committed reporting

Whichever branch occurs — keep, cut, or footnote — is reported with the same care; a null/cut result on non-headline calibration material is a legitimate, useful outcome under this repo's existing "negative results are deliverables" rule, and is not held to a lower documentation standard just because it is not headline-bearing.
