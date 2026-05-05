"""
stubs/mamba_ssm/ops/triton/layernorm_gated.py

Stub for mamba_ssm.ops.triton.layernorm_gated.

modeling_hybridna.py has a hard top-level import:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

RMSNormGated is imported but never called anywhere in the file — it is a dead
import left over from development. This stub provides a no-op RMSNorm class so
the import resolves without requiring the real mamba_ssm / triton kernels.
"""
import torch.nn as nn


class RMSNorm(nn.Module):
    """No-op stub for mamba_ssm RMSNorm (never invoked in HybriDNA forward pass)."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x
