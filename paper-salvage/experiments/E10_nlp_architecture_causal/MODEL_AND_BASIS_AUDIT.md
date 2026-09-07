# E10 Phase 0 — model and basis audit (pre-E10 artifacts only)

**Purpose.** For each of the 7 models E10 will run causal-intervention protocols on, this
document collects — from artifacts that already existed before E10 started, and only from
those — the checkpoint identity, module path, previously-established high-gain row(s), the
`U_k`/`Ux` structural artifact, the `q1`/`PR_spec` value and its source, any prior
activation-detection artifact, the corpus/probe used, and whether a SET of high-gain rows or
only a single canonical row is on record. No number in this document was computed by this
audit — every value is a citation to an existing file. Where a fact is not on record anywhere
found, it is marked **NOT FOUND in pre-E10 artifacts**.

All file paths are relative to the repository root
(`/work/11034/atzanakak/glm_super_weight/genomic-super-weights/`); `paper-salvage/results/`
and the top-level `results/` are two different, both-real directories — raw JSON outputs
referenced below live in the top-level `results/`, not inside `paper-salvage/`.

---

## 1. Llama (Llama-7B)

- **Checkpoint / HF revision:** `huggyllama/llama-7b`. **No pinned revision recorded anywhere**
  — explicitly disclosed as a gap (`paper-salvage/experiments/E5_dimensionality/GATE0_RESULTS.md`,
  line 144: "`huggyllama/llama-7b` (no pinned revision — same gap noted in the earlier
  feasibility audit)").
- **Architecture / module path:** `model.model.layers[layer].mlp.{gate_proj, up_proj,
  down_proj}` (standard SwiGLU, `LlamaMLP`). Adapter: `adapter_llama_swiglu` in
  `paper-salvage/src/uk_frobenius.py`. The E1 script fetches the same three tensor names
  directly by string (`paper-salvage/experiments/E1_nlp_validation/run_e1_retrospective.py`,
  `names = {"gate": "model.layers.{layer}.mlp.gate_proj.weight", ...}`).
- **Established high-gain row:** **L2 / row 3968** (output row `k`), scalar `i=7003`
  (`paper-salvage/experiments/E1_nlp_validation/RESULTS.md`, "Level 1 — row recovery" table;
  also `paper-salvage/experiments/E7_exact_dimensionality/RESULTS.md`, legacy discovery panel
  table: "Llama-7B | NLP | L2/r3968"). Sourced originally from Yu et al. (2024, arXiv:2411.07191)
  Table 2, read from `paper-salvage/docs/superweight_paper.txt`
  (`run_e1_retrospective.py` docstring/comment, "for Llama-7B ... layers[2].mlp.down_proj.weight[3968, 7003]").
- **U_k / Ux artifact:** `paper-salvage/src/uk_frobenius.py` (`uk_frobenius()` /
  `uk_contributions()`, cold-weight closed form `‖U_k‖_F = sqrt(Σ_i W_down[k,i]^2 · ‖W_gate[i,:]‖^2 · ‖W_up[i,:]‖^2)`).
  Recorded value for row 3968, layer 2: `uk_norm = 123.6356847200251`
  (`results/e1_nlp_retrospective.json`, `models[0].level1.uk_norm`; layer median
  `4.205097956825915`, max/median `29.4×`).
- **q1 / PR_spec:** **q1 = 0.9888**, `PR_spec = 1.0227`
  (`paper-salvage/experiments/E7_exact_dimensionality/RESULTS.md`, "Legacy discovery panel
  (Phase 6)" table, row "Llama-7B | NLP | L2/r3968"). Matches the prompt's stated
  approximate value (~0.989) exactly to 3 decimal places. No contradicting value found in
  `CLAIMS_LEDGER.md` or elsewhere.
- **Prior activation-detection artifact:** **NOT FOUND in pre-E10 artifacts.** Llama's row was
  never located by this project's own forward-pass spike detector — it is a cold-weight
  recovery of a coordinate already published by Yu et al.; E1 explicitly states "Weights
  only. No forward pass, no GPU" (`E1_nlp_validation/RESULTS.md`, line 10).
- **Corpus / probe (E7/E8):** **N/A** — Llama's q1/PR_spec (E7 Phase 6, "legacy reanalysis")
  was computed weight-only with zero forward passes, per E7's own provenance table
  (`E7_exact_dimensionality/RESULTS.md`, Provenance section: "Forward passes | ... zero for
  Phi-3 and the legacy panel (weight-only)").
- **Single row vs. SET:** **Single canonical row only.** Explicitly stated:
  "Llama-7B, Mistral-7B | Table 2 lists only one coordinate each; both are E5 primaries, no
  independent second row exists to select." (`paper-salvage/experiments/E6_cross_geometry/CONFIRMATION_PANEL.md`,
  "Not included, with reasons" table). Yu et al. also lists coordinates for Llama-13B/30B and
  Llama2-7B/13B, but those are **different checkpoints**, never fetched by this project
  (same CONFIRMATION_PANEL.md row, and `E7_exact_dimensionality/MODEL_PANEL.md`, "not pursued
  this session").

---

## 2. Mistral (Mistral-7B-v0.1)

- **Checkpoint / HF revision:** `mistralai/Mistral-7B-v0.1`. **No pinned revision recorded**
  (`paper-salvage/experiments/E5_dimensionality/GATE0_RESULTS.md`, line 144).
- **Architecture / module path:** `model.model.layers[layer].mlp.{gate_proj, up_proj,
  down_proj}` — identical `MistralMLP` SwiGLU layout to Llama, same `adapter_llama_swiglu`
  (`paper-salvage/src/uk_frobenius.py`, comment: "Llama / Mistral / GENERator: gate_proj,
  up_proj, down_proj").
- **Established high-gain row:** **L1 / row 2070**, scalar `i=7310`
  (`paper-salvage/experiments/E1_nlp_validation/RESULTS.md`, Level-1 table; also E7 legacy
  panel table "Mistral-7B | NLP | L1/r2070"). Source: Yu et al. Table 2.
- **U_k / Ux artifact:** same `uk_frobenius.py` machinery. Row 2070, layer 1:
  `uk_norm = 0.37510928518138187` (`results/e1_nlp_retrospective.json`, `models[1].level1.uk_norm`;
  layer median `0.01397999823680717`, max/median `26.8×`).
- **q1 / PR_spec:** **q1 = 0.9922**, `PR_spec = 1.0158`
  (`E7_exact_dimensionality/RESULTS.md`, legacy panel table, "Mistral-7B | NLP | L1/r2070").
  Matches the prompt's ~0.992 exactly. No contradiction found elsewhere.
- **Prior activation-detection artifact:** **NOT FOUND in pre-E10 artifacts** — same reasoning
  as Llama; cold-weight recovery of a published coordinate, no forward pass run
  (`E1_nlp_validation/RESULTS.md`, line 10).
- **Corpus / probe:** **N/A** — weight-only (E7 provenance table, same citation as Llama above).
- **Single row vs. SET:** **Single canonical row only** — same CONFIRMATION_PANEL.md citation
  as Llama: "Llama-7B, Mistral-7B | Table 2 lists only one coordinate each."

---

## 3. OLMo (OLMo-7B-0724-hf)

- **Checkpoint / HF revision:** `allenai/OLMo-7B-0724-hf`. **No pinned revision recorded**
  (`paper-salvage/experiments/E6_cross_geometry/RESULTS.md`, provenance table: "`allenai/OLMo-7B-0724-hf`
  (no pinned revision, same gap already disclosed for E5)").
- **Architecture / module path:** `model.layers.{layer}.mlp.{gate_proj, up_proj,
  down_proj}.weight` — same tensor-name convention as Llama/Mistral, confirmed directly by
  the fetch code in `run_e1_retrospective.py` (`names` dict, identical string template used
  for all three of Llama/Mistral/OLMo). No OLMo-specific adapter exists in
  `uk_frobenius.py`'s `ADAPTERS` dict (only `llama_swiglu`/`generator`/`mistral`/`ntv3`/
  `dnabert2`/`evo1` are registered) — OLMo was always accessed via the generic tensor-name
  path in E1, not via the `ADAPTERS` registry.
- **Established high-gain row:** **canonical primary: L1 / row 269**, scalar `i=7467`
  (`E1_nlp_validation/RESULTS.md`, Level-1 table; E7 legacy panel table "OLMo-7B | NLP |
  L1/r269"). Source: Yu et al. Table 2.
- **U_k / Ux artifact:** `uk_frobenius.py`. Row 269, layer 1: `uk_norm = 0.867859727565239`
  (`results/e1_nlp_retrospective.json`, `models[2].level1.uk_norm`; layer median
  `0.022964561329679993`, max/median `37.8×`).
- **q1 / PR_spec:** **q1 = 0.9646**, `PR_spec = 1.0747`
  (`E7_exact_dimensionality/RESULTS.md`, legacy panel table, "OLMo-7B | NLP | L1/r269").
  Matches the prompt's ~0.965 exactly. No contradiction found.
- **Prior activation-detection artifact:** **NOT FOUND in pre-E10 artifacts** — cold-weight
  recovery only, no forward pass (same E1 discipline as Llama/Mistral).
- **Corpus / probe:** **N/A** — weight-only (E7 provenance table).
- **Single row vs. SET:** **A genuine SET exists for OLMo, beyond the single canonical row**,
  unlike Llama/Mistral. Yu et al. Table 2 lists **output row 269 recurring at four layers**
  (1, 2, 7, 24), each with a different scalar `i`
  (`E6_cross_geometry/CONFIRMATION_PANEL.md`, "NLP candidates" table: OLMo-7B L2/r269/i=8275,
  L7/r269/i=453, L24/r269/i=2300; text notes "the recurring output row (269) across all four of
  OLMo-7B's listed layers (1, 2, 7, 24) with a different scalar `i` each time"). **Only L1/r269
  is used as the primary/canonical detection object in E1/E5/E7**; L2/L7/L24 were used only as
  an independent confirmation panel in E6, and no q1/PR_spec was computed for them in any
  artifact found (E6 measures `f_cross`/diagonal PR under the old approximation, not the exact
  E7 spectral metric, for this panel — check `E6_cross_geometry/RESULTS.md` directly before
  assuming otherwise).

---

## 4. Phi-3 (Phi-3-mini-4k-instruct)

- **Checkpoint / HF revision:** `microsoft/Phi-3-mini-4k-instruct`, resolved HF commit
  **`f39ac1d28e925b323eae81227eaba4464caced4e`**
  (`paper-salvage/experiments/E7_exact_dimensionality/PROVENANCE_ADDENDUM.md`, "Phi-3-mini-4k-instruct"
  section). Shard used: `model-00001-of-00002.safetensors`, SHA256
  `b7492726c01287bf6e13c3d74c65ade3d436d50da1cf5bb6925bc962419d6610` (same file).
- **Architecture / module path:** packed `Phi3MLP` — `model.model.layers[i].mlp.gate_up_proj`
  (packed, first half = gate, second half = up, via `.chunk(2, dim=-1)`) and
  `model.model.layers[i].mlp.down_proj`, verified from installed `transformers` source
  (`E7_exact_dimensionality/MODEL_PANEL.md`, "Phi-3-mini-4k-instruct" section, quoting
  `transformers/models/phi3/modeling_phi3.py::Phi3MLP` directly). A dedicated
  `adapter_phi3` was written for E7 ("in E7's own code," not added to `uk_frobenius.py` —
  same MODEL_PANEL.md section); exact file location of `adapter_phi3` was **not directly
  located in this audit's file listing** — flagged below under Open questions.
- **Established high-gain rows:** **six published rows** (all `mlp.down_proj`), from Yu et al.
  Table 2: `L2/r525(i=808)`, `L2/r1693(i=808)`, `L2/r1113(i=808)`, `L4/r525(i=2723)`,
  `L4/r1113(i=2723)`, `L4/r1693(i=2723)` (`E7_exact_dimensionality/MODEL_PANEL.md`,
  "Phi-3-mini-4k-instruct" section; raw values also in `results/e7_phi3_spectral.json`,
  `rows[]`). Treated by E7 as **"one model-level unit"** — all six rows aggregate to one
  value (median) before entering any group comparison, though each is reported individually
  (`MODEL_PANEL.md`, same section).
- **U_k / Ux artifact:** computed via the E7-local `adapter_phi3` feeding
  `spectral_lib.row_spectral_metrics` (weight-only; see Corpus/probe below). Raw per-row
  numbers in `results/e7_phi3_spectral.json` (e.g. L2/r525: `q1=0.9528710903808127`,
  `pr_spec=1.1010014183004648`).
- **q1 / PR_spec:** **model-level median q1 = 0.9028**, median `PR_spec = 1.2249**
  (`E7_exact_dimensionality/RESULTS.md`, "New confirmatory panel (Phase 5)" table, and
  `results/e7_phi3_spectral.json`, `model_level.q1_median` = `0.9028371971796194`,
  `pr_spec_median` = `1.224901693263825`). Individual rows range from `q1=0.5584` (L4/r1113) to
  `q1=0.9529` (L2/r525) — `RESULTS.md` line 27-28. **Matches the prompt's stated ~0.903**
  exactly on the median. No contradicting value found.
- **Prior activation-detection artifact:** **NOT FOUND in pre-E10 artifacts** — all six rows
  are published coordinates recovered structurally; E7's own provenance table states "zero
  [forward passes] for Phi-3" (`E7_exact_dimensionality/RESULTS.md`, Provenance section).
- **Corpus / probe:** **N/A** — weight-only, no forward pass, no tokenizer/corpus used
  (same provenance-table citation).
- **Single row vs. SET:** **A pre-existing SET of six rows exists** (L2×3, L4×3), explicitly
  documented and individually reported, though collapsed to one model-level median for any
  group-level statistic (`MODEL_PANEL.md` and `RESULTS.md`, both cited above).

---

## 5. Qwen2.5 (Qwen2.5-7B)

- **Checkpoint / HF revision:** `Qwen/Qwen2.5-7B`, resolved HF commit
  **`d149729398750b98c0af14eb82c78cfe92750796`**
  (`E7_exact_dimensionality/PROVENANCE_ADDENDUM.md`, "Qwen2.5-7B" section; independently
  confirmed identical in the raw detection JSON, `results/e7_phase1_detection_qwen25.json`,
  key `resolved_revision`). Shard used for the spectral re-fetch:
  `model-00004-of-00004.safetensors`, SHA256
  `b5a2298dddcf228129975a9a271912a9f8dc817957deecde523e3154481ec3fb`.
- **Architecture / module path:** standard unpacked `Qwen2MLP` — `model.model.layers[i].mlp.
  {gate_proj, up_proj, down_proj}`, same family as Llama/Mistral, uses
  `adapter_llama_swiglu` unmodified (`E7_exact_dimensionality/MODEL_PANEL.md`, "Qwen2.5-7B"
  section: "confirmed by Qwen2.5 sharing `transformers`' generic `LlamaMLP`-pattern
  implementation").
- **Established high-gain row:** **L26 / row 458** (no published coordinate existed before
  this project; found via this project's own frozen prospective detection protocol),
  detection ratio 639.2× (`E7_exact_dimensionality/RESULTS.md`, Phase 5 table: "Qwen2.5-7B |
  NLP | L26/r458 | ELIGIBLE-PROSPECTIVE, ratio 639.2"; raw detection record
  `results/e7_phase1_detection_qwen25.json`, `candidate: {layer: 26, row: 458, out_max:
  9038.07, ratio: 639.17}`).
- **U_k / Ux artifact:** `spectral_lib.row_spectral_metrics` on the row-26/458 weight slice
  via `adapter_llama_swiglu`; raw values in `results/e7_phase1_detection_qwen25.json`,
  `spectral: {q1: 0.9529104185966117, pr_spec: 1.0995461100252455, stable_rank:
  1.049416587839116, frob_norm: 29.68742128221388, n_singular_values: 3584}`.
- **q1 / PR_spec:** **q1 = 0.9529**, `PR_spec = 1.0995`
  (`E7_exact_dimensionality/RESULTS.md`, Phase 5 table, and the raw JSON above — both agree
  exactly). **Matches the prompt's stated ~0.953** exactly. No contradiction found.
- **Prior activation-detection artifact:** **Found** — `results/e7_phase1_detection_qwen25.json`
  is exactly this artifact: a full forward-pass sweep hooking every `down_proj` module,
  recording per-layer `out_max`/`out_channel`/`median_channel_max`/`ratio`; layer 26 was the
  global-max layer with ratio 639.17 vs. the 5.0× acceptance threshold
  (`E7_exact_dimensionality/run_phase1_detection.py`, `RATIO_THRESHOLD = 5.0`).
- **Corpus / probe:** **WikiText-2-raw-v1, `test` split**, first 20 non-empty lines
  concatenated with newlines, tokenized and truncated to **512 tokens**
  (`E7_exact_dimensionality/run_phase1_detection.py`, `build_wikitext_input()`, called from
  `run_qwen25()`). This is the detection-forward-pass input only; the q1/PR_spec computation
  itself is weight-only on the already-located row (no forward pass needed for the spectral
  step).
- **Single row vs. SET:** **Single canonical row only** — Qwen2.5-7B had no pre-existing
  published coordinate; the one row found by this project's own prospective detector
  (L26/r458) is the sole row on record (`MODEL_PANEL.md`, "Qwen2.5-7B" section: "No published
  coordinate exists — requires the frozen prospective detection protocol").

---

## 6. MosaicBERT (mosaic-bert-base)

- **Checkpoint / HF revision:** `mosaicml/mosaic-bert-base`, resolved HF commit
  **`c89bbadc24278928f22bcdd7de6b61a5a2d08553`** (found only in the raw detection JSON,
  `results/e8_detection_mosaicbert.json`, key `resolved_revision` — **not** documented in any
  E8 markdown, which calls the checkpoint "unpinned"; see Open questions). Tokenizer:
  `bert-base-uncased` (also unpinned) — MosaicBERT ships no tokenizer of its own
  (`E8_encoder_decoder/RESULTS.md`, "Deviation from the locked prereg text" section).
- **Architecture / module path:** custom remote-code `BertForMaskedLM`
  (`bert_layers.BertForMaskedLM`), bidirectional attention + ALiBi. FFN module path:
  `model.bert.encoder.layer[i].mlp.{gated_layers, wo}` — `gated_layers` packs gate+up on
  adjacent row blocks (first half = gate/activated, second half = up), `wo` is the
  down-projection (biased). **Identical `BertGatedLinearUnitMLP` class DNABERT-2 uses**
  (`E8_encoder_decoder/MODEL_AUDIT.md`, section 1). Adapter used: `adapter_mosaicbert` in
  `E8_encoder_decoder/run_detection_and_spectral.py` (equivalent to
  `uk_frobenius.adapter_dnabert2` — MODEL_AUDIT.md states the existing DNABERT-2 adapter
  "applies to MosaicBERT unmodified," though E8's own script defines its own
  `adapter_mosaicbert` rather than importing the DNABERT-2 one).
- **Established high-gain row:** **L9 / row 287** — no published coordinate; found via E8's
  own frozen prospective detection protocol, detection ratio **288.65×**
  (`E8_encoder_decoder/RESULTS.md`, Detection table; raw record
  `results/e8_detection_mosaicbert.json`, `candidate: {layer: 9, row: 287, out_max:
  253.305, ratio: 288.65}`).
- **U_k / Ux artifact:** `spectral_lib.row_spectral_metrics` via `adapter_mosaicbert`; raw
  values `results/e8_detection_mosaicbert.json`, `spectral: {q1: 0.4765682244123624,
  pr_spec: 4.115864702588391, stable_rank: 2.0983354507805485, frob_norm:
  19.997746047192482, n_singular_values: 768}`.
- **q1 / PR_spec:** **q1 = 0.4766**, `PR_spec = 4.1159`
  (`E8_encoder_decoder/RESULTS.md`, Spectral table; matches raw JSON above exactly).
  **Matches the prompt's stated ~0.477** exactly. Also cross-checked against
  `CLAIMS_LEDGER.md` C-035 ("MosaicBERT candidate `q1`=0.477 ... `PR_spec` 4.1") — consistent,
  no contradiction.
- **Prior activation-detection artifact:** **Found** — `results/e8_detection_mosaicbert.json`,
  full forward-pass sweep over every layer's `mlp.wo` module, same hook/threshold mechanism
  as Qwen2.5's detector (`E8_encoder_decoder/run_detection_and_spectral.py`,
  `DownProjRecorder`, `RATIO_THRESHOLD = 5.0`).
- **Corpus / probe:** **WikiText-2-raw-v1, `test` split**, same construction as Qwen2.5/E7
  (first 20 non-empty lines, newline-joined, truncated to 512 tokens) — identical
  `build_wikitext_input()` function duplicated in `E8_encoder_decoder/run_detection_and_spectral.py`.
  Additionally, E8 measured **5 seeded control rows** at the same layer (layer 9), seed
  `numpy.random.SeedSequence(42)` (`E8_encoder_decoder/RESULTS.md`, "Same-layer control
  rows" table and Provenance section: "Seed | control-row sampling only,
  `numpy.random.SeedSequence(42).spawn(2)`, fixed panel order (MosaicBERT=0, ModernBERT=1)").
- **Single row vs. SET:** **Only ONE canonical high-gain row** (L9/r287). The 5 "control
  rows" at the same layer are explicitly *not* additional high-gain candidates — they are a
  random background sample used to show the candidate is locally exceptional
  (`E8_encoder_decoder/RESULTS.md`, "Same-layer control rows" section: control-row q1 sits
  around 0.03–0.05, i.e. far from high-gain). **No pre-existing SET of multiple high-gain
  rows exists for MosaicBERT** in any artifact found. This determines the encoder Case
  distinction noted in the summary below.

---

## 7. ModernBERT (ModernBERT-base)

- **Checkpoint / HF revision:** `answerdotai/ModernBERT-base`, resolved HF commit
  **`8949b909ec900327062f0ebf497f51aef5e6f0c8`** (found only in the raw detection JSON,
  `results/e8_detection_modernbert.json`, key `resolved_revision` — again not documented in
  any E8 markdown; see Open questions).
- **Architecture / module path:** native `transformers` `ModernBertForMaskedLM`
  (`model_type: modernbert`, no `trust_remote_code` needed), bidirectional, alternating
  local/global attention with RoPE. FFN module path: `model.model.layers[i].mlp.{Wi, Wo}` —
  `Wi` packs gate+up (chunked in half; despite the local variable naming, activation is
  applied to the **first** half), `Wo` is the down-projection, **no bias anywhere**
  (`E8_encoder_decoder/MODEL_AUDIT.md`, section 2, quoting
  `modeling_modernbert.py::ModernBertMLP` directly). Adapter: `adapter_modernbert` in
  `E8_encoder_decoder/run_detection_and_spectral.py` (new adapter, not added to
  `uk_frobenius.py`).
- **Established high-gain row:** **L15 / row 251** — no published coordinate; found via E8's
  own detection protocol, detection ratio **561.12×**
  (`E8_encoder_decoder/RESULTS.md`, Detection table; raw record
  `results/e8_detection_modernbert.json`, `candidate: {layer: 15, row: 251, out_max:
  34279.30, ratio: 561.12}`).
- **U_k / Ux artifact:** `spectral_lib.row_spectral_metrics` via `adapter_modernbert`; raw
  values `results/e8_detection_modernbert.json`, `spectral: {q1: 0.8969889514776379,
  pr_spec: 1.2336584717613133, stable_rank: 1.1148409334949654, frob_norm:
  836.046538392924, n_singular_values: 768}`.
- **q1 / PR_spec:** **q1 = 0.8970**, `PR_spec = 1.2337**
  (`E8_encoder_decoder/RESULTS.md`, Spectral table; matches raw JSON exactly). **Matches the
  prompt's stated ~0.897** exactly. Cross-checked against `CLAIMS_LEDGER.md` C-035
  ("ModernBERT candidate `q1`=0.897 ... `PR_spec` 1.2") — consistent, no contradiction. Note
  E8's own RESULTS.md repeatedly flags this margin as **"thin"** relative to the decoder
  floor (0.21% relative margin on q1) — this is a documented caveat, not a contradiction,
  but is directly relevant to how confidently E10 can treat ModernBERT as "encoder-side."
- **Prior activation-detection artifact:** **Found** — `results/e8_detection_modernbert.json`,
  full forward-pass sweep over every layer's `mlp.Wo` module, same mechanism as MosaicBERT.
- **Corpus / probe:** **WikiText-2-raw-v1, `test` split**, identical construction to
  MosaicBERT/Qwen2.5 (512-token truncation) — same `build_wikitext_input()` function, same
  script. 5 seeded control rows at layer 15, same seed scheme as MosaicBERT
  (`E8_encoder_decoder/RESULTS.md`, Provenance section, panel index "ModernBERT=1").
- **Single row vs. SET:** **Only ONE canonical high-gain row** (L15/r251), same status as
  MosaicBERT — the 5 control rows are a random background sample, not additional candidates
  (`E8_encoder_decoder/RESULTS.md`, "Same-layer control rows" table: control-row q1 around
  0.02–0.05). **No pre-existing SET of multiple high-gain rows exists for ModernBERT** in
  any artifact found.

---

## Summary table

| Model | Checkpoint (revision) | Row(s) | q1 (matches prompt?) | Activation-detection artifact | Corpus | Single row vs. SET |
|---|---|---|---:|---|---|---|
| Llama-7B | `huggyllama/llama-7b` (unpinned) | L2/r3968 | 0.9888 (yes, ~0.989) | NOT FOUND | N/A (weight-only) | single |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` (unpinned) | L1/r2070 | 0.9922 (yes, ~0.992) | NOT FOUND | N/A (weight-only) | single |
| OLMo-7B | `allenai/OLMo-7B-0724-hf` (unpinned) | L1/r269 (canonical); L2/L7/L24 r269 also on record | 0.9646 (yes, ~0.965) | NOT FOUND | N/A (weight-only) | SET exists (4 layers, same output row, only L1 canonical) |
| Phi-3-mini-4k-instruct | `microsoft/Phi-3-mini-4k-instruct` @ `f39ac1d2...` | 6 rows (L2×3, L4×3) | 0.9028 median (yes, ~0.903) | NOT FOUND | N/A (weight-only) | SET of 6, aggregated to 1 model-level unit |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` @ `d1497293...` | L26/r458 | 0.9529 (yes, ~0.953) | `results/e7_phase1_detection_qwen25.json` | WikiText-2-raw-v1 test, 512 tok | single |
| MosaicBERT | `mosaicml/mosaic-bert-base` @ `c89bbadc...` | L9/r287 | 0.4766 (yes, ~0.477) | `results/e8_detection_mosaicbert.json` | WikiText-2-raw-v1 test, 512 tok | single (no SET) |
| ModernBERT | `answerdotai/ModernBERT-base` @ `8949b909...` | L15/r251 | 0.8970 (yes, ~0.897; thin margin flagged in RESULTS.md) | `results/e8_detection_modernbert.json` | WikiText-2-raw-v1 test, 512 tok | single (no SET) |

---

## Open questions / gaps

The following are things Phase 1/2 of E10 will need to resolve before freezing intervention
objects — none of them are guessed at above; they are named here precisely because they are
**NOT FOUND** or otherwise incomplete in pre-E10 artifacts.

1. **No pinned HF revision exists for Llama-7B, Mistral-7B, or OLMo-7B anywhere in this
   repository.** Every E1/E5/E6/E7 measurement on these three checkpoints was run against
   whatever revision `main` resolved to at fetch time, with no commit hash captured. If E10's
   causal intervention needs bit-identical reproducibility, these three revisions must be
   resolved and pinned fresh before any run (this is a new lookup, not something this audit
   can supply from existing artifacts).
2. **MosaicBERT's and ModernBERT's resolved HF commits were never written into any E8
   markdown** — `MODEL_AUDIT.md` and `RESULTS.md` both call both checkpoints "unpinned." The
   exact commits (`c89bbadc24278928f22bcdd7de6b61a5a2d08553` for MosaicBERT,
   `8949b909ec900327062f0ebf497f51aef5e6f0c8` for ModernBERT) exist only inside the raw
   `resolved_revision` field of `results/e8_detection_mosaicbert.json` /
   `results/e8_detection_modernbert.json`. Confirm these are still the current `main` HEAD
   commits for both repos before treating them as a stable pin (a repo's `main` can move).
3. **No `q1`/`PR_spec` (exact spectral metric) was computed for OLMo-7B's non-canonical rows**
   (L2/r269, L7/r269, L24/r269) — they exist only under the older diagonal-PR/`f_cross`
   metric in E6, if at all; check `E6_cross_geometry/RESULTS.md` directly rather than
   assuming a value exists. If E10 wants to treat OLMo as having a genuine multi-row SET
   under the *exact* metric (not just the diagonal one), this measurement does not yet exist.
4. **The exact file/line location of E7's `adapter_phi3` was not confirmed in this audit.**
   `MODEL_PANEL.md` states one was written "in E7's own code," but the file list surfaced by
   this audit's search did not include a script literally named or grepped to contain
   `adapter_phi3`'s definition body — only `run_phi3_spectral.py` was found by filename,
   which is presumably where it lives, but its exact contents were not read here.
5. **No causal-intervention tooling exists yet for any of the 7 models.**
   `E8_encoder_decoder/CAUSAL_FOLLOWUP_FEASIBILITY.md` states plainly that ablating a
   singular vector (as opposed to a diagonal scalar `c_{k,i}`) requires "a genuinely new
   activation-space (not weight-space) perturbation tool" that "this repository does not
   have" — this applies to all 7 models here, decoder and encoder alike, not just the
   encoders. E10 Phase 1/2 needs to build this before any intervention can run.
6. **No native causal/behavioral endpoint was located in this audit for the 5 decoders**
   (Llama, Mistral, OLMo, Phi-3, Qwen2.5) — `CAUSAL_FOLLOWUP_FEASIBILITY.md` only documents
   endpoints for the 2 encoders (masked-token loss, both `*ForMaskedLM`). What loss/metric
   E10 will use as the decoder-side causal endpoint is **NOT FOUND in pre-E10 artifacts** and
   will need to be decided fresh.
7. **Qwen2.5-7B and both encoders' high-gain rows were found by this project's own detector
   on a single WikiText-2 512-token window** — no robustness check across multiple windows
   or corpora exists for these three rows in any artifact found. If E10's protocol requires a
   frozen, previously-validated probe *set* (plural inputs) rather than one fixed window,
   that does not yet exist for any of these three models.
8. **DNABERT-2's mature single-neuron causal-intervention pipeline**
   (`scripts/interpretability/run_neuron_causal_intervention_dnabert2.py`, referenced in
   `CAUSAL_FOLLOWUP_FEASIBILITY.md`) was named as the closest existing precedent for how an
   NLP intervention tool might be built, but this audit did not read that script's contents —
   a future session designing E10's intervention tooling should read it directly rather than
   inferring its interface from this document.
