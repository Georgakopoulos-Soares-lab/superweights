# tests/test_hooks.py
"""Smoke test: hooks attach, record, and detach cleanly."""
import torch
import torch.nn as nn
from hooks.activation_hooks import ActivationRecorder


class FakeWrapper:
    config = {"down_proj_pattern": "layer", "num_layers": 2}
    num_layers = 2
    layer = nn.Linear(16, 8)

    def get_target_module(self, i):
        return self.layer

    def forward(self, seq):
        x = torch.randn(1, 4, 16)
        self.layer(x)


def test_recorder_fills_records():
    wrapper = FakeWrapper()
    recorder = ActivationRecorder()
    recorder.register(wrapper, [0, 1])
    wrapper.forward("ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC")
    recorder.remove_all()
    assert 0 in recorder.records
    assert "in_max" in recorder.records[0]
    assert "out_channel" in recorder.records[0]
