# E8 — text-encoder compatibility audit

**No activation spike, `q1`, `PR_spec`, or any other E8 measurement was computed before this
file was frozen.** Everything below comes from config/source inspection only.

## 1. MosaicBERT (`mosaicml/mosaic-bert-base`) — **ELIGIBLE**

| | |
|---|---|
| Architecture | `BertForMaskedLM` (custom remote code, `bert_layers.BertForMaskedLM`) |
| Attention | **Bidirectional** (standard BERT self-attention + ALiBi, no causal mask) |
| `hidden_size` / `intermediate_size` / layers | 768 / 3072 / 12 |
| Activation | GELU (`nn.GELU(approximate='none')`) |

Gated-FFN class, verified from the actual remote-code source
(`bert_layers.py::BertGatedLinearUnitMLP`), **not from the paper or repo comment**:

```python
self.gated_layers = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
self.wo = nn.Linear(intermediate_size, hidden_size)     # bias=True (default)
hidden_states = self.gated_layers(hidden_states)
gated = hidden_states[:, :intermediate_size]              # first half -> activated (our "gate")
non_gated = hidden_states[:, intermediate_size:]           # second half -> multiplier (our "up")
hidden_states = self.act(gated) * non_gated
hidden_states = self.wo(hidden_states)                     # + residual + LayerNorm outside this class
```

**This is the identical `BertGatedLinearUnitMLP` class DNABERT-2 uses** — DNABERT-2 is itself
built on the Mosaic BERT codebase (already documented in this repo,
`scripts/interpretability/neuron_pilot_common.py`'s own module docstring: "DNABERT-2 (Mosaic
BERT) unpads internally"). Same tensor paths (`model.bert.encoder.layer[i].mlp.gated_layers`,
`.wo`), same packed layout, same bias convention (packed layer unbiased, `wo` biased). **E7/E5's
existing `uk_frobenius.adapter_dnabert2` applies to MosaicBERT unmodified** — the strongest
possible architecture bridge to DNABERT-2 this audit could have found, exactly as intended:
same FFN code, natural-language domain instead of DNA.

`U_k` compatibility: valid without modification. `gate_up` packing convention matches the
adapter's existing assumption (first half activated).

## 2. ModernBERT (`answerdotai/ModernBERT-base`) — **ELIGIBLE**

| | |
|---|---|
| Architecture | `ModernBertForMaskedLM` (native `transformers`, `model_type: modernbert`, no `trust_remote_code` needed — confirmed present in this environment's installed `transformers` 4.57.6) |
| Attention | **Bidirectional**, alternating local (window 128, every layer) / global (every 3rd layer) — no causal mask anywhere |
| `hidden_size` / `intermediate_size` / layers | 768 / 1152 / 22 |
| Activation | GELU (`config.hidden_activation`) |

Gated-FFN class, verified from the installed `transformers` source
(`modeling_modernbert.py::ModernBertMLP`), independent of MosaicBERT's implementation (a
different codebase, Answer.AI/LightOn, not a MosaicML derivative):

```python
self.Wi = nn.Linear(hidden_size, intermediate_size * 2, bias=config.mlp_bias)   # mlp_bias=False
self.Wo = nn.Linear(intermediate_size, hidden_size, bias=config.mlp_bias)        # mlp_bias=False
input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
return self.Wo(self.drop(self.act(input) * gate))
```

Despite the local variable being named `gate` for the *second* half, the activation is
applied to the **first** half (`input`) — the same "first half activated" orientation as
MosaicBERT/DNABERT-2/NTv3, just with different internal naming. No bias anywhere in the MLP
(`mlp_bias: false`). Module path: `model.model.layers[i].mlp.{Wi,Wo}` (`ModernBertForMaskedLM`
→ `.model` (`ModernBertModel`) → `.layers[i]` (`ModernBertEncoderLayer`) → `.mlp`
(`ModernBertMLP`), verified directly from the class definitions, not assumed).

**New adapter required** (`adapter_modernbert` in E8's own code, not added to
`uk_frobenius.py`, matching this project's "prefer self-contained experiment code" precedent)
— packed at `Wi`, split via `.chunk(2)`, no bias, distinct module names from every existing
adapter in this repo.

`U_k` compatibility: valid without modification.

## Independence from MosaicBERT

ModernBERT is not a MosaicBERT derivative: different codebase (Answer.AI/LightOn vs.
MosaicML), different attention mechanism (alternating local/global with RoPE vs. ALiBi
full-attention), different bias convention (fully unbiased MLP vs. biased down-projection),
different normalization placement, different tokenizer/vocabulary. The two share only the
GLU-family FFN pattern and bidirectional attention — exactly the two properties this audit
needs to hold constant while varying everything else, and exactly why both were named as the
primary candidates.

## Hard gate

**Two independent bidirectional text-encoder families are ELIGIBLE.** No replacement model is
needed. E8 proceeds to Phase 2 with both. `E8_BLOCKED.md` is not written.

## Not pursued

No optional replacement was sought — the gate is satisfied by MosaicBERT and ModernBERT
alone, and the governing instruction caps this at "at most ONE other" only if one of the two
primaries were unavailable or incompatible. Neither was.
