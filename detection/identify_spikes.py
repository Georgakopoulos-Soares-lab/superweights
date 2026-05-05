# detection/identify_spikes.py
"""
Super weight location strategy (Yu et al. 2024, Figure 3):

  - The super weight CREATES the spike in the OUTPUT of its layer.
  - That spike then propagates via skip connections into the INPUT of all later layers.
  - Therefore: find the EARLIEST layer where out_max is anomalously large.
    That is the layer containing the super weight.

  - SW row = out_channel at that layer  (output channel index)
  - SW col = in_channel  at that layer  (input channel index — from the INPUT hook)

Do NOT use the layer with the largest in_max — that is always a downstream layer
receiving the already-propagated super activation.
"""
import numpy as np


def find_spike_layer(records: dict) -> int:
    """
    Find the EARLIEST layer with an anomalously large out_max.
    Uses a z-score threshold to define 'anomalous'.
    """
    layers    = sorted(records.keys())
    out_maxes = np.array([records[i].get("out_max", 0.0) for i in layers])

    mean = out_maxes.mean()
    std  = out_maxes.std()

    if std == 0:
        return layers[int(np.argmax(out_maxes))]

    z_scores = (out_maxes - mean) / std

    # Find earliest layer with z-score above threshold
    THRESHOLD = 3.0
    anomalous = [layers[i] for i, z in enumerate(z_scores) if z > THRESHOLD]

    if anomalous:
        return anomalous[0]   # earliest anomalous layer = where SW lives

    # Fallback: layer with largest out_max
    return layers[int(np.argmax(out_maxes))]


def extract_coords(records: dict, spike_layer: int) -> tuple:
    """
    Returns (layer_idx, row, col).
    row = out_channel at spike layer  (row of the weight matrix W)
    col = in_channel  at spike layer  (col of the weight matrix W)
    """
    row = records[spike_layer]["out_channel"]
    col = records[spike_layer]["in_channel"]
    return (spike_layer, row, col)
