"""
experiments/E10_nlp_architecture_causal/generate_decoder_controls.py

E10 Step D1: freeze 5 same-layer random-control rows per decoder, BEFORE any causal
measurement. Same seeding convention as E8's control-row draw
(numpy.random.SeedSequence(42).spawn(N), panel-indexed, excluding target rows from the pool)
generalized from 2 encoders to 5 decoders.

Control layer per model = the layer of that model's highest-q1 (top-ranked) structural row in
its Step-D1 frozen set (DECODER_INTERVENTION_FREEZE.md), matching E8's "same layer as the
candidate" convention.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

# (model, d_model, control_layer, excluded_rows_at_that_layer)
PANEL = [
    ("llama",   4096, 2,  [3968]),
    ("mistral", 4096, 1,  [2070]),
    ("olmo",    4096, 24, [269]),
    ("phi3",    3072, 2,  [525, 1693, 1113]),
    ("qwen25",  3584, 26, [458]),
]

N_CONTROL_ROWS = 5

results = {}
seed_seqs = np.random.SeedSequence(42).spawn(len(PANEL))
for panel_idx, (name, d_model, layer, excluded) in enumerate(PANEL):
    rng = np.random.default_rng(seed_seqs[panel_idx])
    pool = [r for r in range(d_model) if r not in excluded]
    control_rows = sorted(rng.choice(pool, size=N_CONTROL_ROWS, replace=False).tolist())
    results[name] = dict(control_layer=layer, control_rows=control_rows, d_model=d_model,
                          excluded_rows=excluded, panel_index=panel_idx)
    print(f"{name}: layer {layer}, controls {control_rows}")

out_path = ROOT / "results" / "e10_decoder_control_rows.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2))
print(f"\nwrote {out_path}")
