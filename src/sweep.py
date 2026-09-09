# detection/sweep.py
"""
Run one forward pass through the model and collect activation stats from all target layers.
Requires only a single input sequence — no dataset needed (Yu et al. 2024, Section 3.1).
"""
from src.activation_hooks import ActivationRecorder
from src.dna_probes import DEFAULT_PROBE


def sweep(model_wrapper, probe: str = DEFAULT_PROBE) -> dict:
    """
    Returns dict: { layer_idx: {in_max, in_channel, out_max, out_channel} }
    """
    recorder = ActivationRecorder()
    recorder.register(model_wrapper, list(range(model_wrapper.num_layers)))
    model_wrapper.forward(probe)
    recorder.remove_all()
    return recorder.records
