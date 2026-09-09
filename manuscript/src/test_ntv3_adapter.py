"""Tests for the NTv3 adapter (N-010).

Every test here FAILS under the previous registry entry `ADAPTERS["ntv3"] =
adapter_llama_swiglu`, which reached for `model.model.layers[...].mlp.gate_proj` — NTv3 has
no such path.

    python src/test_ntv3_adapter.py

Uses a stub with NTv3's real module layout and real shapes, so no checkpoint download and
no GPU are needed.
"""

import torch
import torch.nn as nn

from uk_frobenius import ADAPTERS, adapter_llama_swiglu, adapter_ntv3, uk_contributions

D_MODEL, D_FFN = 1536, 6144          # NTv3_650M_pre, verified at L11


class _Block(nn.Module):
    """NTv3 SelfAttentionBlock layout: FFN inline as fc1 (packed gate+up) + fc2."""

    def __init__(self, d_model=D_MODEL, d_ffn=D_FFN):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 2 * d_ffn, bias=False)   # weight [2*d_ffn, d_model]
        self.fc2 = nn.Linear(d_ffn, d_model, bias=False)       # weight [d_model, d_ffn]
        self._ffn_activation_fn = nn.SiLU()

    # NTv3 really does expose an unrelated bound method named `mlp`; reproduce that, since
    # it is what made the naive `getattr(blk, "mlp")` resolution fail.
    def mlp(self, x):
        return x


class _Core(nn.Module):
    def __init__(self, n=12):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(_Block() for _ in range(n))


class _NTv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _Core()


def test_resolves_and_shapes():
    m = _NTv3()
    g, u, d = adapter_ntv3(m, 11)
    assert g.shape == (D_FFN, D_MODEL), g.shape
    assert u.shape == (D_FFN, D_MODEL), u.shape
    assert d.shape == (D_MODEL, D_FFN), d.shape
    # canonical convention the rest of the module assumes
    assert g.shape == u.shape
    assert d.shape == (g.shape[1], g.shape[0])
    print("  ok  resolves fc1/fc2 and returns canonical shapes")


def test_registry_wired():
    assert ADAPTERS["ntv3"] is adapter_ntv3, "registry still points at the wrong adapter"
    assert ADAPTERS["ntv3"] is not adapter_llama_swiglu
    print("  ok  ADAPTERS['ntv3'] is the NTv3 adapter, not the Llama one")


def test_old_adapter_would_fail():
    """The regression this guards: the Llama adapter cannot address an NTv3 model."""
    m = _NTv3()
    try:
        adapter_llama_swiglu(m, 11)
    except AttributeError:
        print("  ok  adapter_llama_swiglu raises AttributeError on an NTv3 model, as it must")
        return
    raise AssertionError("adapter_llama_swiglu unexpectedly succeeded on an NTv3 model")


def test_halves_are_the_two_packed_blocks():
    m = _NTv3()
    packed = m.core.transformer_blocks[11].fc1.weight
    g, u, _ = adapter_ntv3(m, 11)
    assert torch.equal(g, packed[:D_FFN])
    assert torch.equal(u, packed[D_FFN:])
    print("  ok  gate/up are the two adjacent row blocks of fc1")


def test_gate_up_swap_invariance():
    """c_{k,i} is symmetric under gate<->up, so the unresolvable ordering cannot matter."""
    m = _NTv3()
    g, u, d = adapter_ntv3(m, 11)
    a = uk_contributions(g, u, d)
    b = uk_contributions(u, g, d)
    assert torch.allclose(a, b, rtol=0, atol=0), "c_{k,i} is not swap-symmetric"
    print("  ok  c_{k,i} identical under gate/up swap")


def test_rejects_mismatched_layout():
    """A block whose fc1 is not 2*d_ffn must raise, not be silently transposed."""
    m = _NTv3()
    blk = m.core.transformer_blocks[11]
    blk.fc1 = nn.Linear(D_MODEL, 3 * D_FFN, bias=False)
    try:
        adapter_ntv3(m, 11)
    except ValueError as e:
        assert "refusing to guess" in str(e)
        print("  ok  rejects a non-packed fc1 instead of guessing")
        return
    raise AssertionError("adapter_ntv3 accepted a layout it cannot interpret")


def test_rejects_missing_fc():
    m = _NTv3()
    blk = m.core.transformer_blocks[11]
    del blk.fc1
    try:
        adapter_ntv3(m, 11)
    except ValueError as e:
        assert "no fc1/fc2" in str(e)
        print("  ok  rejects a block without fc1/fc2")
        return
    raise AssertionError("adapter_ntv3 accepted a block with no fc1")


if __name__ == "__main__":
    torch.manual_seed(0)
    for fn in (test_resolves_and_shapes, test_registry_wired, test_old_adapter_would_fail,
               test_halves_are_the_two_packed_blocks, test_gate_up_swap_invariance,
               test_rejects_mismatched_layout, test_rejects_missing_fc):
        fn()
    print("all NTv3 adapter checks passed")
