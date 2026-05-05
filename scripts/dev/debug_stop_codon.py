"""
debug_stop_codon.py — Does removing the super row inflate stop-codon probabilities?

Yu et al. Fig 5: removing the super weight in Llama causes stopword probabilities
(the, a, ., ,) to be amplified ~2–10×.  The genomic analogue: does removing
GENERator's super row (layer 4, row 2371) inflate stop-codon token probabilities?

GENERator uses a 6-mer tokenizer.  Stop codons (TAA, TAG, TGA) appear embedded
inside 6-mer tokens.  We identify all vocabulary tokens that *contain* a stop
codon at position 0 (i.e., the 6-mer starts with a stop codon), which would be
the next predicted codon immediately following the current sequence.

Method:
  1. Identify all vocabulary tokens containing each stop codon.
  2. For each of several input positions in ACTB_CDS, take the logits of the
     LAST token position (next-token prediction).
  3. Compute softmax probabilities and sum stop-codon token probabilities.
  4. Compare: intact model vs. super row zeroed.

Usage:
  python debug_stop_codon.py
"""
import torch
import torch.nn.functional as F
import yaml
from models import WRAPPER_MAP
from probes.dna_probes import PROBES

SUPER_ROW    = 2371
LAYER        = 4
PROBE_KEY    = "actb_full"
# Evaluate next-token predictions at these positions (bp offsets, must be multiples of 6)
EVAL_OFFSETS = [0, 60, 120, 180, 240, 300, 360, 420]  # 8 different contexts
CONTEXT_LEN  = 120   # bp of context for each prediction (20 tokens)

STOP_CODONS  = ["TAA", "TAG", "TGA"]
N_RANDOM     = 10    # number of random rows to test as control
INTERMEDIATE = 8448  # GENERator down_proj input dimension

config  = yaml.safe_load(open("configs/generator.yaml"))
wrapper = WRAPPER_MAP["generator"](config)
wrapper.load()

tokenizer = wrapper.tokenizer
model     = wrapper.model
probe_seq = PROBES[PROBE_KEY]

# ── Step 1: find stop-codon tokens in vocabulary ──────────────────────────────
vocab = tokenizer.get_vocab()   # {token_str: id}

stop_token_ids   = {sc: [] for sc in STOP_CODONS}
stop_token_ids["any"] = []

for token_str, token_id in vocab.items():
    upper = token_str.upper()
    for sc in STOP_CODONS:
        # Token starts with the stop codon (codon is at position 0 of this 6-mer)
        if upper.startswith(sc):
            stop_token_ids[sc].append(token_id)
            if token_id not in stop_token_ids["any"]:
                stop_token_ids["any"].append(token_id)

print("Stop codon token counts in vocabulary:")
for sc in STOP_CODONS:
    print(f"  {sc}: {len(stop_token_ids[sc])} tokens")
print(f"  any: {len(stop_token_ids['any'])} tokens total")

if not stop_token_ids["any"]:
    print("\nWARNING: No stop-codon tokens found in vocabulary.")
    print("The 6-mer vocabulary may encode both strands or use different conventions.")
    print("Printing top-20 vocabulary tokens for manual inspection:")
    samples = sorted(vocab.items(), key=lambda x: x[1])[:40]
    for t, i in samples:
        print(f"  id={i:5d}  '{t}'")
    raise SystemExit(1)


# ── Step 2: helper — get next-token probabilities at last position ────────────
def get_next_token_probs(sequence: str):
    seq    = wrapper._prepare_sequence(sequence)
    inputs = tokenizer(seq, return_tensors="pt",
                       add_special_tokens=False).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits   # [1, seq_len, vocab]
    # Last token position = next-token prediction
    last_logits = logits[0, -1, :]        # [vocab]
    probs       = F.softmax(last_logits.float(), dim=-1)
    return probs


def stop_codon_prob(probs):
    ids = stop_token_ids["any"]
    return probs[ids].sum().item() if ids else 0.0

def top_k_tokens(probs, k=10):
    topk = torch.topk(probs, k)
    results = []
    id_to_token = {v: k for k, v in vocab.items()}
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        results.append((id_to_token.get(idx, f"[{idx}]"), score))
    return results


# ── Step 3: run on multiple contexts, intact vs zeroed ───────────────────────
sr_module = wrapper.get_target_module(LAYER)
sr_saved  = sr_module.weight.data[SUPER_ROW, :].clone()

print()
print("=" * 72)
print("Stop-codon probability: intact model vs. super row zeroed")
print(f"{'Context offset':>16}  {'intact':>10}  {'zeroed':>10}  {'ratio':>8}")
print("-" * 72)

results_intact = []
results_zeroed = []

for offset in EVAL_OFFSETS:
    end   = offset + CONTEXT_LEN
    if end > len(probe_seq):
        break
    # Ensure slice is divisible by 6
    chunk = probe_seq[offset:end]
    if len(chunk) % 6 != 0:
        chunk = chunk[:len(chunk) - len(chunk) % 6]
    if len(chunk) < 12:
        continue

    # Intact
    p_intact = get_next_token_probs(chunk)
    sc_intact = stop_codon_prob(p_intact)
    results_intact.append(sc_intact)

    # Zeroed
    with torch.no_grad():
        sr_module.weight.data[SUPER_ROW, :] = 0.0
    p_zeroed = get_next_token_probs(chunk)
    sc_zeroed = stop_codon_prob(p_zeroed)
    results_zeroed.append(sc_zeroed)

    with torch.no_grad():
        sr_module.weight.data[SUPER_ROW, :] = sr_saved

    ratio = sc_zeroed / sc_intact if sc_intact > 0 else float("nan")
    print(f"  bp {offset:>4}–{end:<4}         {sc_intact:>10.4f}  {sc_zeroed:>10.4f}  {ratio:>8.2f}×")

# Restore just in case
with torch.no_grad():
    sr_module.weight.data[SUPER_ROW, :] = sr_saved

print()
mean_i = sum(results_intact) / len(results_intact) if results_intact else 0
mean_z = sum(results_zeroed) / len(results_zeroed) if results_zeroed else 0
overall_ratio = mean_z / mean_i if mean_i > 0 else float("nan")
print(f"  Mean stop-codon prob: intact={mean_i:.4f}  zeroed={mean_z:.4f}  "
      f"ratio={overall_ratio:.2f}×")

# ── Step 3b: random-row control ──────────────────────────────────────────────
import random
random.seed(42)
torch.manual_seed(42)

n_rows = sr_module.weight.data.shape[0]   # = hidden_size (3072)
rand_rows = random.sample([r for r in range(n_rows) if r != SUPER_ROW], N_RANDOM)

print()
print("=" * 72)
print(f"Random-row control ({N_RANDOM} rows): mean stop-codon ratio vs. super row")
print(f"  (each row zeroed individually; ratio = zeroed / intact, mean over contexts)")
print("-" * 72)

rand_ratios = []
for rr in rand_rows:
    rr_saved = sr_module.weight.data[rr, :].clone()
    row_sc_zeroed = []
    row_sc_intact = []
    for offset in EVAL_OFFSETS:
        end   = offset + CONTEXT_LEN
        if end > len(probe_seq):
            break
        chunk = probe_seq[offset:end]
        if len(chunk) % 6 != 0:
            chunk = chunk[:len(chunk) - len(chunk) % 6]
        if len(chunk) < 12:
            continue

        p_i = get_next_token_probs(chunk)
        row_sc_intact.append(stop_codon_prob(p_i))

        with torch.no_grad():
            sr_module.weight.data[rr, :] = 0.0
        p_z = get_next_token_probs(chunk)
        row_sc_zeroed.append(stop_codon_prob(p_z))
        with torch.no_grad():
            sr_module.weight.data[rr, :] = rr_saved

    mean_ri = sum(row_sc_intact) / len(row_sc_intact) if row_sc_intact else 0
    mean_rz = sum(row_sc_zeroed) / len(row_sc_zeroed) if row_sc_zeroed else 0
    ratio_r = mean_rz / mean_ri if mean_ri > 0 else float("nan")
    rand_ratios.append(ratio_r)
    print(f"  row {rr:>5}:  intact={mean_ri:.4f}  zeroed={mean_rz:.4f}  ratio={ratio_r:.2f}×")

mean_rand_ratio = sum(rand_ratios) / len(rand_ratios)
print(f"\n  Mean random-row ratio: {mean_rand_ratio:.2f}×")
print(f"  Super-row ratio:       {overall_ratio:.2f}×")
specificity = overall_ratio / mean_rand_ratio if mean_rand_ratio > 0 else float("nan")
print(f"  Specificity (super / random): {specificity:.1f}×")
print()
if specificity > 5:
    print("  → Stop-codon suppression is SPECIFIC to the super row")
else:
    print("  → Stop-codon suppression is NOT specific — generic row-zeroing effect")

# ── Step 4: show top-10 tokens for one representative context, both conditions ─
print()
print("=" * 72)
print("Top-10 next-token predictions at first context (intact vs. zeroed):")
chunk = probe_seq[:CONTEXT_LEN]
chunk = chunk[:len(chunk) - len(chunk) % 6]

p_i = get_next_token_probs(chunk)
with torch.no_grad():
    sr_module.weight.data[SUPER_ROW, :] = 0.0
p_z = get_next_token_probs(chunk)
with torch.no_grad():
    sr_module.weight.data[SUPER_ROW, :] = sr_saved

top_i = top_k_tokens(p_i, 10)
top_z = top_k_tokens(p_z, 10)

print(f"  {'Rank':<5}  {'Intact token':<14} {'prob':>8}  |  "
      f"{'Zeroed token':<14} {'prob':>8}  {'stop?':>6}")
print("  " + "-" * 66)
id_to_token = {v: k for k, v in vocab.items()}
for rank, ((ti, pi), (tz, pz)) in enumerate(zip(top_i, top_z), 1):
    is_stop_i = any(ti.upper().startswith(sc) for sc in STOP_CODONS)
    is_stop_z = any(tz.upper().startswith(sc) for sc in STOP_CODONS)
    flag = f"{'← STOP' if is_stop_z else ''}"
    print(f"  {rank:<5}  {ti:<14} {pi:>8.4f}  |  {tz:<14} {pz:>8.4f}  {flag}")

print()
print("=" * 72)
print("Interpretation:")
print("  ratio > 1.5×: stop-codon suppression effect confirmed")
print("  ratio ≈ 1.0×: super row does not specifically affect stop-codon probability")
