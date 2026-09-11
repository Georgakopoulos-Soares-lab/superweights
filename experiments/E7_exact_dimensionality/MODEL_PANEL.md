# E7 — model compatibility audit and panel freeze

**No SVD, spectral share, `f_cross`, or any E7/E5/E6 measurement was computed before this file
was frozen.** Everything below comes from config/source inspection and cache/availability
checks only — no target-layer weight was passed through any dimensionality metric.

## A. Genomic candidates

### 1. Gene42 — **UNAVAILABLE**

The paper (arXiv:2503.16565) states models are "publicly available at huggingface.co/
inceptionai" and confirms the required architecture (LLaMA-style decoder, SwiGLU, RMSNorm,
RoPE — Gene42-L: hidden 2048, 24 layers, FFN dim 5440). **No such checkpoint exists on
HuggingFace as of this audit.** Checked directly:
- `huggingface.co/inceptionai`'s full model listing (36 repos, all Jais/Sherkala family — no
  Gene42/Prot42/Chem42/Omics42 entry).
- HF-wide search for `gene42` (15 hits, all under `madog/Gene42-*`) and `omics42` (0 hits).
- Every `madog/Gene42-*` repo is an **empty placeholder** (verified via the HF API: each
  contains only a `.gitattributes` file, zero actual weights or config — these are GUE-task
  finetune repo names with no content, not the base model).

No alternative org name found. Excluded — not for an architectural reason, for
non-availability. If a future session finds the checkpoint (e.g. released later, or under a
name not surfaced by this search), Gene42 remains the single best candidate for the
bridge-model role described below.

### 2. Evo 2 7B — **ELIGIBLE-PROSPECTIVE**

Initial check failed: `import evo2` inside `evo2.sif` raised
`undefined symbol: _ZN3c105ErrorC2E...` from `~/.local`'s `transformer_engine_torch` — an ABI
mismatch between the host-side, Evo1-oriented `~/.local` package tree and whatever torch
build the `evo2` package needs. **The user corrected this**: the existing repository
convention (`scripts/detection/run_detection_evo2.sbatch`) already invokes the container with
`--cleanenv --env PYTHONNOUSERSITE=1`, which excludes `~/.local` and uses the container's own
`/usr/local/lib/python3.11/dist-packages/{transformer_engine,evo2}` instead. Re-run with the
correct flags: **`import evo2` succeeds cleanly.**

Architecture, verified from actual source inside the container (`vortex.model.layers`, the
package `evo2`'s vendored StripedHyena2 implementation — not from the paper alone):

```python
class ParallelGatedMLP(nn.Module):
    self.l1 = nn.Linear(hidden_size, inner_size, bias=False)   # gate
    self.l2 = nn.Linear(hidden_size, inner_size, bias=False)   # up
    self.l3 = nn.Linear(inner_size, hidden_size, bias=False)   # down
    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        y = self.l3(self.act(z1) * z2)
        return y
```

Structurally identical to Evo1's `l1/l2/l3` (already validated compatible with the exact
`U_k` construction). Confirmed present on **every** block — hyena/conv blocks and the five
attention-only blocks alike (`attn_layer_idxs = [3, 10, 17, 24, 31]`, matching
`configs/evo2_7b.yaml`'s existing comment in this repo).

**Notable methods detail, worth stating precisely rather than glossing over:**
`evo2-7b-8k.yml`'s actual config has `evo2_style_activations: true`, and the source
conditionally replaces the activation with `nn.Identity()` for every layer **except layer 0**:
`if self.layer_idx > 0 and evo2_style_activations: self.act = nn.Identity()`. This means for
any Evo2 target layer other than 0, the block computes `y = l3(z1 * z2)` with **no
elementwise nonlinearity at all** on the gate branch. `U_k` is normally described (correctly)
as a bilinear *surrogate* that drops the real SiLU/GELU nonlinearity — for Evo2 layers > 0,
there is no nonlinearity to drop: `U_k` is the literal bilinear operator the block computes,
not an approximation of a nonlinear one. This is disclosed as a methods-relevant fact, not
used to claim anything stronger about causal dimensionality (the working hypothesis's second
arrow, structure → causal unit, is explicitly out of scope for E7 regardless).

Config confirmed directly (`evo2-7b-8k.yml`): `hidden_size=4096`, `inner_mlp_size=11008`,
`num_layers=32` — matching `configs/evo2_7b.yaml`'s existing `down_proj_pattern:
"blocks.{i}.mlp.l3"` exactly. `evo2.models.MODEL_NAMES` includes `evo2_7b` →
`arcinstitute/evo2_7b` (official Arc Institute checkpoint). **Not locally cached** — a fresh
download is required (see Phase 5 provenance for exact size/revision once fetched).

No published SW/high-gain coordinate exists for Evo2 anywhere (this repository or the
literature) — it requires the frozen prospective detection protocol (Phase 1).

### 3. GenomeOcean-4B — **ELIGIBLE-PROSPECTIVE** (selected as the one-replacement candidate)

Selected because both of the two named high-priority genomic candidates could not both serve
(Gene42 unavailable; Evo2 turned out eligible, satisfying one slot, but the instructions still
call for "Gene42 or equivalent... plus Evo 2 or another genuinely independent genomic family,"
and Gene42's specific *diagnostic* role — LLaMA-style architecture on genomic data, a bridge
against "GLMs differ only because architecture is unlike NLP" — has a close substitute below.

Architecture, verified from the actual released `config.json` (not from this repository's
existing, unverified comment in `configs/genomeocean.yaml`):

```json
{"architectures": ["MistralForCausalLM"], "hidden_act": "silu",
 "hidden_size": 3072, "intermediate_size": 16384, "num_hidden_layers": 24}
```

Standard `MistralMLP` (`gate_proj`/`up_proj`/`down_proj`), already validated compatible via
E5/E6's own `adapter_llama_swiglu`. Trained on metagenomic assembly data by a different team
(DOE Joint Genome Institute; HF org moved from `pGenomeOcean` to `DOEJGI`, redirect
confirmed), independent of GENERator/DNABERT-2/NTv3/Evo1. This repository already has unused
scaffolding for it (`models/genomeocean_wrapper.py`, `configs/genomeocean.yaml`) but **no
E5/E6 (or any prior) measurement of any kind exists** — it does not appear in
`docs/PAPER_OUTLINE.md`'s model coverage table at all.

**Serves Gene42's specific diagnostic role.** Gene42's value was "genomic training data +
autoregressive decoder + LLaMA-style architecture," proposed as a bridge: if it patterns with
the genomic group, that is evidence architecture alone does not explain the split (since
architecture is NLP-like); if it patterns with NLP, architecture becomes a live explanatory
factor. GenomeOcean-4B is, architecturally, the **same family** as this study's own Mistral-7B
NLP model (`MistralForCausalLM`) — trained on a genomic domain instead. It can carry exactly
this diagnostic, and is disclosed here as doing so deliberately, not as a like-for-like
Gene42 substitute in every other respect (different scale, different specific corpus).

Not locally cached; a fresh download is required. No published coordinate exists — requires
the frozen prospective detection protocol.

### NTv3 / DNABERT-2 / GENERator EUK — not re-audited here

Already fully characterized in E5/E6; contribute to the **legacy discovery panel** (Phase 6)
only, per the governing instruction that E5/E6 models are discovery/supporting data for this
newly-formulated hypothesis, not new confirmatory units.

## B. NLP candidates

### 1. Phi-3-mini-4k-instruct — **ELIGIBLE-PUBLISHED**

Yu et al. Table 2 publishes six coordinates (all `mlp.down_proj`):
`L2/[525,808]`, `L2/[1693,808]`, `L2/[1113,808]`, `L4/[525,2723]`, `L4/[1113,2723]`,
`L4/[1693,2723]`. Config confirmed directly (`microsoft/Phi-3-mini-4k-instruct/config.json`):
`hidden_size=3072`, `intermediate_size=8192`, `hidden_act=silu`, `num_hidden_layers=32`,
`attention_bias=false`. MLP structure confirmed from the actual installed `transformers`
source (`transformers/models/phi3/modeling_phi3.py::Phi3MLP`), **not assumed**:

```python
self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)   # packed
self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
up_states = self.gate_up_proj(hidden_states)
gate, up_states = up_states.chunk(2, dim=-1)          # first half = gate, second = up
up_states = up_states * self.activation_fn(gate)
```

Packed gate+up, first-half-gate/second-half-up — the DNABERT-2/NTv3 layout, not the Llama
one. Needs its own adapter (`adapter_phi3` in E7's own code, following the same
shape-guard-or-raise discipline as `uk_frobenius.py`'s existing packed adapters; `uk_frobenius.py`
itself is not modified). No bias anywhere in the MLP. Not locally cached; single-shard-only
fetch is sufficient (six published coordinates span layers 2 and 4 only — two shards at most).

**One model-level unit**, per the governing instruction — all six rows aggregate to one
value (median) before entering any group comparison; each row is still reported individually.

### 2. Qwen2.5-7B — **ELIGIBLE-PROSPECTIVE**

Config confirmed directly (`Qwen/Qwen2.5-7B/config.json`): `architectures:
["Qwen2ForCausalLM"]`, `hidden_act=silu`, `hidden_size=3584`, `intermediate_size=18944`,
`num_hidden_layers=28`. `Qwen2MLP` is a standard, unpacked `gate_proj`/`up_proj`/`down_proj`
SwiGLU module (same family as Llama/Mistral/GENERator, already validated via
`adapter_llama_swiglu`) — confirmed by Qwen2.5 sharing `transformers`' generic
`LlamaMLP`-pattern implementation. No published coordinate exists — requires the frozen
prospective detection protocol. Not locally cached; needs full forward-capable weights
(~15GB) since detection requires an actual forward pass, unlike Phi-3's shard-only fetch.

### 3. Gemma-2 (9B and 2B both checked) — **UNAVAILABLE**

`google/gemma-2-9b` and `google/gemma-2-2b` are both **gated** (`"gated": "manual"` per the
HF API). The cached credential file at `/work/11034/atzanakak/ls6/huggingface/token` is
**empty** (0 bytes) — both variants return HTTP 401 even with that file's contents attached
as a bearer token. No valid HF authentication is available in this environment. Excluded for
infrastructure reasons, not architecture — Gemma 2's `Gemma2MLP` is a standard SwiGLU module
and would very likely be eligible if access were available. If a future session obtains a
valid, license-accepted HF token, Gemma 2 is the natural next NLP addition.

### Llama-13B / Llama-30B / Llama2-7B / Llama2-13B / OLMo-1B — not pursued this session

Yu et al. publishes coordinates for all of these (secondary robustness tier, priority 3-4 per
the governing instruction). None is locally cached; obtaining even single-layer shards for
five additional checkpoints, on top of the two new NLP and two new genomic models already
authorized, was judged disproportionate to this bounded session — consistent with the
instruction to prefer "genuinely different new NLP family/families" (already satisfied by
Qwen2.5) over repeated same-family scale checks. Left as provenanced, ready candidates for a
future robustness-tier extension, not run here.

## Classification summary

| Model | Class | Reason |
|---|---|---|
| Evo 2 7B | **ELIGIBLE-PROSPECTIVE** | Compatible `ParallelGatedMLP`, confirmed via container source; needs fresh checkpoint + frozen detection |
| GenomeOcean-4B | **ELIGIBLE-PROSPECTIVE** | Compatible `MistralMLP`; Gene42's replacement, serving its bridge-diagnostic role |
| Gene42 | **UNAVAILABLE** | No public checkpoint exists despite the paper's claim (verified directly) |
| Phi-3-mini-4k-instruct | **ELIGIBLE-PUBLISHED** | Compatible packed `Phi3MLP`; 6 Yu-table coordinates, aggregated to 1 model-level unit |
| Qwen2.5-7B | **ELIGIBLE-PROSPECTIVE** | Compatible `Qwen2MLP`; needs frozen detection |
| Gemma 2 (9B, 2B) | **UNAVAILABLE** | Gated; no valid HF token in this environment |
| Llama-13B/30B, Llama2-7B/13B, OLMo-1B | not pursued | Provenanced, deferred — scope discipline, see above |
| Llama-7B, Mistral-7B, OLMo-7B, GENERator EUK, DNABERT-2, NTv3 | **DISCOVERY (E5/E6)** | Legacy panel, Phase 6 only — not new confirmatory units |

## Model-level replication adequacy

**New confirmatory panel: 2 new NLP model units (Phi-3, Qwen2.5) + 2 new genomic model units
(Evo 2, GenomeOcean-4B).** This meets the instruction's stated ideal ("≥2 new NLP model units
and ≥2 new genomic model units") — the panel is adequate for **model-level, not row-level,
group comparison**, directly addressing E6's pseudoreplication problem (this time, no group
has more than 2 models, and no single checkpoint can dominate a group the way OLMo/DNABERT-2
did in E6).

**This is still a small-n, exploratory-scale confirmatory test (n=2 vs n=2 new models,
combinable with the n=6 legacy models only as supporting/generalization evidence, never
pooled into the same primary group comparison).** It is not being labeled a large, powered
independent replication — it is labeled exactly what the governing instruction permits: a
model-level confirmatory panel adequate to move past E6's specific row-level pseudoreplication
flaw, not adequate to make a sweeping architecture-general claim.
