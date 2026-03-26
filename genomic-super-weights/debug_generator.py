# debug_generator.py
import torch
import yaml
from models.generator_wrapper import GeneratorWrapper
from hooks.activation_hooks import ActivationRecorder

config = yaml.safe_load(open("configs/generator.yaml"))
wrapper = GeneratorWrapper(config)
wrapper.load()

from probes.dna_probes import DEFAULT_PROBE
probe = DEFAULT_PROBE

# ── 1. Print ALL layer out_max values (not just the spike layer) ──────────────
recorder = ActivationRecorder()
recorder.register(wrapper, list(range(wrapper.num_layers)))
wrapper.forward(probe)
recorder.remove_all()

print("=== FULL ACTIVATION PROFILE (all layers) ===")
for i in sorted(recorder.records.keys()):
    r = recorder.records[i]
    print(f"  layer {i:>2}  in_max={r.get('in_max',0):>12.2f}  out_max={r.get('out_max',0):>12.2f}  in_ch={r.get('in_channel')}  out_ch={r.get('out_channel')}")

# ── 2. Check the actual value at (4, 2371, 2536) before and after zeroing ─────
module = wrapper.get_target_module(4)
print(f"\n=== WEIGHT at [4][2371,2536] BEFORE zero: {module.weight.data[2371, 2536].item():.6f}")
wrapper.zero_out_weight(4, 2371, 2536)
print(f"=== WEIGHT at [4][2371,2536] AFTER  zero: {module.weight.data[2371, 2536].item():.6f}")

# ── 3. Re-run sweep AFTER zeroing and print full profile again ────────────────
recorder2 = ActivationRecorder()
recorder2.register(wrapper, list(range(wrapper.num_layers)))
wrapper.forward(probe)
recorder2.remove_all()

print("\n=== FULL ACTIVATION PROFILE AFTER ZEROING [4][2371,2536] ===")
for i in sorted(recorder2.records.keys()):
    r = recorder2.records[i]
    print(f"  layer {i:>2}  in_max={r.get('in_max',0):>12.2f}  out_max={r.get('out_max',0):>12.2f}  in_ch={r.get('in_channel')}  out_ch={r.get('out_channel')}")

# ── 4. Print the module path and confirm it resolves to a real Linear layer ───
print(f"\n=== MODULE TYPE at layer 4: {type(module)}")
print(f"=== WEIGHT SHAPE: {module.weight.shape}")
print(f"=== WEIGHT DEVICE: {module.weight.device}")
print(f"=== WEIGHT DTYPE: {module.weight.dtype}")

# ── 5. Check if the spike in_max at layer 4 is the same as out_max at layer 3 ─
r3 = recorder.records.get(3, {})
r4 = recorder.records.get(4, {})
print(f"\n=== PROPAGATION CHECK ===")
print(f"  layer 3 out_max = {r3.get('out_max', 'N/A')}")
print(f"  layer 4 in_max  = {r4.get('in_max',  'N/A')}")
print(f"  Are they equal? {abs(r3.get('out_max',0) - r4.get('in_max',0)) < 1.0}")
