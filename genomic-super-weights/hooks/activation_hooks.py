# hooks/activation_hooks.py
"""
Registers forward hooks on each target projection layer to record, per layer:
  - in_max:      max absolute value of the INPUT tensor  → identifies SW column
  - in_channel:  which hidden dimension holds that max   → SW col index
  - out_max:     max absolute value of the OUTPUT tensor → identifies SW row
  - out_channel: which hidden dimension holds that max   → SW row index

This directly implements Figure 3 of Yu et al. (2024):
  SW row = out_channel of the spike layer
  SW col = in_channel  of the spike layer
"""
import torch
from typing import Dict, List


class ActivationRecorder:

    def __init__(self):
        self.records: Dict[int, dict] = {}
        self._handles = []

    def register(self, model_wrapper, layer_indices: List[int]):
        self.records = {}
        self._handles = []
        for i in layer_indices:
            module = model_wrapper.get_target_module(i)
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(i)))
            self._handles.append(module.register_forward_hook(self._post_hook(i)))

    def _pre_hook(self, layer_idx: int):
        def hook(module, args):
            x = args[0]
            flat = x.reshape(-1, x.shape[-1]).abs()       # [B*L, H]
            self.records.setdefault(layer_idx, {})
            self.records[layer_idx]["in_max"]     = flat.max().item()
            self.records[layer_idx]["in_channel"] = flat.max(dim=0).values.argmax().item()
        return hook

    def _post_hook(self, layer_idx: int):
        def hook(module, args, output):
            y = output[0] if isinstance(output, tuple) else output
            flat = y.reshape(-1, y.shape[-1]).abs()
            self.records.setdefault(layer_idx, {})
            self.records[layer_idx]["out_max"]     = flat.max().item()
            self.records[layer_idx]["out_channel"] = flat.max(dim=0).values.argmax().item()
        return hook

    def remove_all(self):
        for h in self._handles:
            h.remove()
        self._handles = []
