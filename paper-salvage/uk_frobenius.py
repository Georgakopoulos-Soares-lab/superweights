"""
Closed-form structural predictor of high-gain gated-FFN output channels.

    ||U_k||_F = sqrt( sum_i  W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2 )

with per-hidden-unit contributions

    c_{k,i} = W_down[k,i]^2 * ||W_gate[i,:]||^2 * ||W_up[i,:]||^2

Weights only. No forward pass, no training data, no tokenizer.

The c_{k,i} decomposition also answers the granularity question directly: top-1 share and
participation ratio say whether the amplifier is a scalar, a small set, or a distributed
row.

Conventions
-----------
    W_gate : [d_ffn,   d_model]
    W_up   : [d_ffn,   d_model]
    W_down : [d_model, d_ffn]

Adapters below map each architecture's naming onto that. DNABERT-2 packs gate and up on
adjacent row blocks of a single `gated_layers` matrix; Evo1 names them l1/l2/l3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Callable

import torch


# --------------------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------------------

@torch.no_grad()
def uk_contributions(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Per-(k, i) contributions c_{k,i}. Returns [d_model, d_ffn].

    Computed in float64 by default: the quantity is a product of three squared norms and
    underflows/overflows readily in bf16 for 7B-scale weights.
    """
    W_gate = W_gate.to(dtype)
    W_up = W_up.to(dtype)
    W_down = W_down.to(dtype)

    if W_gate.shape != W_up.shape:
        raise ValueError(f"gate {tuple(W_gate.shape)} != up {tuple(W_up.shape)}")
    d_ffn, _ = W_gate.shape
    if W_down.shape[1] != d_ffn:
        raise ValueError(
            f"W_down has d_ffn={W_down.shape[1]}, gate/up have d_ffn={d_ffn}. "
            "Check the adapter for transposed conventions."
        )

    gate_sq = (W_gate * W_gate).sum(dim=1)          # [d_ffn]
    up_sq = (W_up * W_up).sum(dim=1)                # [d_ffn]
    return (W_down * W_down) * (gate_sq * up_sq)[None, :]   # [d_model, d_ffn]


@torch.no_grad()
def uk_frobenius(W_gate, W_up, W_down, dtype=torch.float64) -> torch.Tensor:
    """||U_k||_F per output coordinate k. Returns [d_model]."""
    return uk_contributions(W_gate, W_up, W_down, dtype).sum(dim=1).sqrt()


# --------------------------------------------------------------------------------------
# Granularity (E4)
# --------------------------------------------------------------------------------------

@dataclass
class RowGranularity:
    row: int
    uk_norm: float
    top1_index: int
    top1_share: float          # max_i c / sum_i c
    top5_share: float
    participation_ratio: float  # (sum c)^2 / sum c^2 -- effective n contributors
    d_ffn: int

    @property
    def regime(self) -> str:
        """Coarse label. Descriptive only -- do not use as a causal claim."""
        if self.top1_share > 0.5:
            return "scalar-dominated"
        if self.participation_ratio < 0.01 * self.d_ffn:
            return "sparse"
        return "distributed"


@torch.no_grad()
def row_granularity(contributions: torch.Tensor, row: int) -> RowGranularity:
    c = contributions[row]
    total = c.sum()
    if total <= 0:
        raise ValueError(f"row {row} has zero total contribution")
    order = torch.argsort(c, descending=True)
    return RowGranularity(
        row=row,
        uk_norm=float(total.sqrt()),
        top1_index=int(order[0]),
        top1_share=float(c[order[0]] / total),
        top5_share=float(c[order[:5]].sum() / total),
        participation_ratio=float(total.pow(2) / (c * c).sum()),
        d_ffn=int(c.numel()),
    )


@torch.no_grad()
def scalar_rank(contributions: torch.Tensor, row: int, index: int) -> int:
    """1-based rank of hidden unit `index` within row `row`. For NLP Level-2 validation."""
    c = contributions[row]
    return int((c > c[index]).sum()) + 1


# --------------------------------------------------------------------------------------
# Layer report
# --------------------------------------------------------------------------------------

@dataclass
class LayerReport:
    layer: int
    d_model: int
    d_ffn: int
    median: float
    max: float
    max_over_median: float
    top_rows: list           # [(row, ||U_k||_F), ...]
    query_ranks: dict        # {row: {"rank": int, "percentile": float, "value": float}}


@torch.no_grad()
def layer_report(
    W_gate, W_up, W_down, layer: int, query_rows=(), top_n: int = 10
) -> tuple[LayerReport, torch.Tensor]:
    contrib = uk_contributions(W_gate, W_up, W_down)
    uk = contrib.sum(dim=1).sqrt()
    order = torch.argsort(uk, descending=True)
    d_model = int(uk.numel())

    ranks = {}
    for r in query_rows:
        rank = int((uk > uk[r]).sum()) + 1
        ranks[int(r)] = {
            "rank": rank,
            "percentile": 100.0 * (1.0 - (rank - 1) / d_model),
            "value": float(uk[r]),
        }

    median = float(uk.median())
    report = LayerReport(
        layer=layer,
        d_model=d_model,
        d_ffn=int(contrib.shape[1]),
        median=median,
        max=float(uk.max()),
        max_over_median=float(uk.max() / median) if median > 0 else float("inf"),
        top_rows=[(int(i), float(uk[i])) for i in order[:top_n]],
        query_ranks=ranks,
    )
    return report, contrib


# --------------------------------------------------------------------------------------
# Architecture adapters
# --------------------------------------------------------------------------------------
# Each returns (W_gate, W_up, W_down) for one layer, in the canonical convention above.
# Verify shapes on a new architecture before trusting the numbers -- a silent transpose
# produces a plausible-looking ranking that is wrong.

def adapter_llama_swiglu(model, layer: int):
    """Llama / Mistral / GENERator: gate_proj, up_proj, down_proj."""
    mlp = model.model.layers[layer].mlp
    return mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight


def adapter_dnabert2(model, layer: int):
    """DNABERT-2 BertGatedLinearUnitMLP: gate and up packed on adjacent row blocks."""
    mlp = model.bert.encoder.layer[layer].mlp
    packed = mlp.gated_layers.weight        # [2 * d_ffn, d_model]
    d_ffn = packed.shape[0] // 2
    return packed[:d_ffn], packed[d_ffn:], mlp.wo.weight


def adapter_evo1(model, block: int):
    """Evo1 ParallelGatedConvBlock: l1 = gate, l2 = up, l3 = down."""
    mlp = model.blocks[block].mlp
    return mlp.l1.weight, mlp.l2.weight, mlp.l3.weight


ADAPTERS: dict[str, Callable] = {
    "llama_swiglu": adapter_llama_swiglu,
    "generator": adapter_llama_swiglu,
    "mistral": adapter_llama_swiglu,
    "ntv3": adapter_llama_swiglu,      # verify -- Mistral-style, confirm module path
    "dnabert2": adapter_dnabert2,
    "evo1": adapter_evo1,
}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def audit(model, adapter: str, layers, query=None, top_n: int = 10) -> dict:
    """query: {layer: [rows...]} of empirically known SW rows, for rank reporting."""
    get = ADAPTERS[adapter]
    query = query or {}
    out = {"adapter": adapter, "layers": {}}
    for L in layers:
        g, u, d = get(model, L)
        rep, contrib = layer_report(g, u, d, L, query_rows=query.get(L, ()), top_n=top_n)
        entry = asdict(rep)
        entry["granularity"] = {
            int(r): asdict(row_granularity(contrib, int(r))) for r in query.get(L, ())
        }
        out["layers"][L] = entry
        del contrib
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Cold-weight ||U_k||_F audit.")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--revision", default=None, help="pin the HF revision")
    p.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
    p.add_argument("--layers", required=True, help="e.g. 0-29 or 2,4,11")
    p.add_argument("--query", default=None,
                   help='known SW rows as JSON, e.g. \'{"4": [2371, 1522]}\'')
    p.add_argument("--out", required=True)
    p.add_argument("--top-n", type=int, default=10)
    a = p.parse_args()

    if "-" in a.layers:
        lo, hi = a.layers.split("-")
        layers = list(range(int(lo), int(hi) + 1))
    else:
        layers = [int(x) for x in a.layers.split(",")]

    query = {int(k): v for k, v in json.loads(a.query).items()} if a.query else {}

    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        a.model, revision=a.revision, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()

    result = audit(model, a.adapter, layers, query, a.top_n)
    result["model"] = a.model
    result["revision"] = a.revision

    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {a.out}")
