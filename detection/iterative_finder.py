# detection/iterative_finder.py
"""
Full iterative super weight / super row detection loop.

Row-level mode (default, use_row_zeroing=True):
  Zeros the entire identified output row and checks whether the activation spike
  collapses.  This is appropriate for models like GENERator where the super entity
  is a uniformly scaled output neuron rather than a single scalar.

Scalar mode (use_row_zeroing=False):
  Original Yu et al. behaviour — zeros the single (row, col) scalar.

Algorithm (both modes):
  1. Run sweep → find spike layer → extract row (and col)
  2. Zero out the row (or scalar)
  3. Re-run sweep
  4. Repeat until spike suppressed below threshold × initial_spike OR max_iterations.
"""
from detection.sweep import sweep
from detection.identify_spikes import find_spike_layer, extract_coords
from probes.dna_probes import DEFAULT_PROBE


def find_all_super_weights(
    model_wrapper,
    probe: str = DEFAULT_PROBE,
    suppression_threshold: float = 0.1,
    max_iterations: int = 10,
    use_row_zeroing: bool = True,
) -> list:
    """
    Returns list of dicts:
      [{"layer": int, "row": int, "col": int, "in_max": float, "out_max": float}, ...]

    use_row_zeroing: if True, zero the entire identified row (super row mode).
                     if False, zero only the single scalar (original Yu et al. mode).
    """
    super_weights = []
    found_rows    = set()   # (layer, row) — dedup for row mode
    found_coords  = set()   # (layer, row, col) — dedup for scalar mode

    # Sanity check: confirm same object is returned each call
    m1 = model_wrapper.get_target_module(0)
    m2 = model_wrapper.get_target_module(0)
    assert m1 is m2, "get_target_module() returns different objects — caching issue!"

    records = sweep(model_wrapper, probe)

    initial_max = max(v.get("in_max", 0.0) for v in records.values())
    if initial_max != initial_max:  # NaN check
        raise RuntimeError(
            "Initial sweep returned NaN activations. Check model dtype — "
            "try dtype: float32 in the config."
        )

    initial_spike = records[find_spike_layer(records)]["out_max"]

    for iteration in range(max_iterations):
        spike_layer = find_spike_layer(records)
        current_spike = records[spike_layer].get("out_max", 0.0)

        # Termination: output spike suppressed below threshold
        if current_spike < suppression_threshold * initial_spike:
            print(f"[iter {iteration}] Output spike suppressed. Detection complete.")
            break

        layer_idx, row, col = extract_coords(records, spike_layer)

        if use_row_zeroing:
            key = (layer_idx, row)
            if key in found_rows:
                print(
                    f"[iter {iteration}] Same row re-detected (layer={layer_idx}, row={row}) — "
                    "no further super rows found. Detection complete."
                )
                break
            found_rows.add(key)
            print(
                f"[iter {iteration}] Super row at layer={layer_idx}, row={row}  "
                f"in_max={records[spike_layer]['in_max']:.2f}, "
                f"out_max={records[spike_layer]['out_max']:.2f}"
            )
            super_weights.append({
                "layer":   layer_idx,
                "row":     row,
                "col":     col,
                "in_max":  records[spike_layer]["in_max"],
                "out_max": records[spike_layer]["out_max"],
            })
            model_wrapper.zero_out_row(layer_idx, row)

        else:
            coords = (layer_idx, row, col)
            if coords in found_coords:
                print(
                    f"[iter {iteration}] Same coordinates re-detected — "
                    "no further super weights found. Detection complete."
                )
                break
            found_coords.add(coords)
            print(
                f"[iter {iteration}] SW at layer={layer_idx}, "
                f"row={row}, col={col}, "
                f"in_max={records[spike_layer]['in_max']:.2f}, "
                f"out_max={records[spike_layer]['out_max']:.2f}"
            )
            super_weights.append({
                "layer":   layer_idx,
                "row":     row,
                "col":     col,
                "in_max":  records[spike_layer]["in_max"],
                "out_max": records[spike_layer]["out_max"],
            })
            model_wrapper.zero_out_weight(layer_idx, row, col)

        records = sweep(model_wrapper, probe)

    return super_weights
