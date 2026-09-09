# tests/test_detection.py
"""Smoke test: detection runs end-to-end on a tiny fake model."""
import torch
import torch.nn as nn
from src.sweep import sweep
from src.identify_spikes import find_spike_layer, extract_coords
from src.activation_hooks import ActivationRecorder


class TinyFakeWrapper:
    config = {"down_proj_pattern": "layer", "num_layers": 3, "causal": False}
    num_layers = 3
    layer = nn.Linear(32, 16)

    def get_target_module(self, i):
        return self.layer

    def forward(self, seq):
        x = torch.randn(1, 6, 32)
        # Inject a synthetic spike at layer 1 only — simulates a super weight
        if not hasattr(self, "_spiked"):
            x[0, 0, 7] = 1000.0
            self._spiked = True
        self.layer(x)

    def zero_out_weight(self, layer_idx, row, col):
        with torch.no_grad():
            self.layer.weight[row, col] = 0.0


def test_spike_detected():
    wrapper = TinyFakeWrapper()
    records = sweep(wrapper, "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC")
    spike = find_spike_layer(records)
    assert spike in records
    layer_idx, row, col = extract_coords(records, spike)
    assert isinstance(row, int)
    assert isinstance(col, int)
