"""
debug_dnabert2_masked_token.py — Does removing the super row shift [MASK] predictions
toward degenerate k-mer tokens?

Mirrors debug_stop_codon.py for DNABERT-2, implementing the genomic BERT analogue
of Yu et al. (2024) Figure 5.

In Yu et al.: pruning the super weight in a causal LLM inflates stop-word
probabilities by 2–10× (the model "collapses" to degenerate predictions).

For DNABERT-2 (masked LM / encoder), the analogous question is:
  When the super row (L5, row=603) is zeroed, does the predicted token at
  any arbitrarily masked position shift toward:
    (a) Special tokens: [PAD], [UNK], [CLS], [SEP], [MASK]
    (b) Degenerate k-mers: sequences with ≥50% N ambiguous bases
    (c) Very short (≤2 char) tokens: single-nucleotide tokens, whitespace

These three categories are the masked-LM analogue of stop-word inflation.

Method:
  1. Identify degenerate vocabulary tokens (categories a, b, c above).
  2. For EVAL_OFFSETS positions in the ACTB CDS:
     a. Mask that one position.
     b. Run forward pass → softmax distribution over vocabulary.
     c. Sum probability mass on degenerate tokens.
     d. Report top-10 predicted tokens (intact vs. zeroed).
  3. Print a clear before/after comparison table.
  4. Also record random-row controls.

Usage:
  cd genomic-super-weights
  CUDA_VISIBLE_DEVICES=5 python debug_dnabert2_masked_token.py
"""
import torch
import torch.nn.functional as F
import yaml
from models import WRAPPER_MAP
from probes.dna_probes import PROBES

SUPER_ROW    = 603
LAYER        = 5
PROBE_KEY    = "actb_full"
# Evaluate masked predictions at these token indices (after tokenizer; 1-indexed
# to skip [CLS]).  We sample several positions spread across the sequence.
N_EVAL_POS   = 8   # number of positions to evaluate
N_RANDOM     = 5   # random-row controls

config  = yaml.safe_load(open("configs/dnabert2.yaml"))
wrapper = WRAPPER_MAP["dnabert2"](config)
wrapper.load()

tokenizer = wrapper.tokenizer
model     = wrapper.model


# ── Step 1: Tokenize the probe and identify degenerate tokens ─────────────────
probe_seq = PROBES[PROBE_KEY]
enc       = tokenizer(probe_seq, return_tensors="pt", truncation=True, max_length=512)
input_ids = enc["input_ids"].to(model.device)       # [1, L]
attn_mask = enc.get("attention_mask",
                    torch.ones_like(input_ids)).to(model.device)

vocab     = tokenizer.get_vocab()                   # {str: int}
id_to_tok = {v: k for k, v in vocab.items()}

mask_id = tokenizer.mask_token_id
cls_id  = tokenizer.cls_token_id
sep_id  = tokenizer.sep_token_id
pad_id  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

# ── Identify degenerate tokens ────────────────────────────────────────────────
special_ids: set = set()
for attr in ("cls_token_id", "sep_token_id", "pad_token_id",
             "bos_token_id", "eos_token_id", "mask_token_id", "unk_token_id"):
    tid = getattr(tokenizer, attr, None)
    if tid is not None:
        special_ids.add(tid)

degenerate_ids = set(special_ids)  # start with all specials

for tok_str, tok_id in vocab.items():
    upper = tok_str.upper().replace(" ", "")
    # (b) ≥50% ambiguous bases (N or non-ACGT)
    dna_chars = sum(1 for c in upper if c in "ACGT")
    if len(upper) > 0 and dna_chars / len(upper) < 0.5:
        degenerate_ids.add(tok_id)
    # (c) very short tokens (1 character or fewer)
    if len(upper) <= 1:
        degenerate_ids.add(tok_id)

print(f"Degenerate token categories:")
print(f"  (a) Special tokens:   {len(special_ids)}")
print(f"  Total degenerate:     {len(degenerate_ids)}  out of vocab size {len(vocab)}")
degenerate_id_list = sorted(degenerate_ids)

# ── Identify non-special positions to evaluate ────────────────────────────────
ids_list    = input_ids[0].tolist()
non_special = [i for i, t in enumerate(ids_list) if t not in special_ids]
if not non_special:
    raise RuntimeError("No non-special tokens found in probe sequence.")

# Evenly spaced evaluation positions
step   = max(1, len(non_special) // N_EVAL_POS)
eval_positions = non_special[::step][:N_EVAL_POS]
print(f"\nProbe: '{PROBE_KEY}' — {len(ids_list)} tokens  ({len(non_special)} non-special)")
print(f"Evaluating at token positions: {eval_positions}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def forward_at_position(pos: int) -> torch.Tensor:
    """Return softmax distribution at position `pos` when that token is masked."""
    masked_ids      = input_ids.clone()
    masked_ids[0, pos] = mask_id
    with torch.no_grad():
        logits = model(input_ids=masked_ids, attention_mask=attn_mask).logits
    return F.softmax(logits[0, pos].float(), dim=-1)   # [vocab_size]


def degen_prob(probs: torch.Tensor) -> float:
    return probs[degenerate_id_list].sum().item()


def top_k_tokens(probs: torch.Tensor, k: int = 10):
    topk = torch.topk(probs, k)
    return [(id_to_tok.get(i.item(), f"[{i.item()}]"), f"{v.item():.4f}")
            for v, i in zip(topk.values, topk.indices)]


# ── Step 2: Intact model — collect distributions ─────────────────────────────
print("\n" + "=" * 72)
print("Masked-token distribution: intact model vs. super row zeroed")
print("=" * 72)
print(f"  {'Pos':>4}  {'true_tok':<12}  "
      f"{'degen prob (intact)':>22}  {'degen prob (zeroed)':>22}  {'ratio':>8}")
print("  " + "-" * 72)

sr_module = wrapper.get_target_module(LAYER)
sr_saved  = sr_module.weight.data[SUPER_ROW, :].clone()

results_intact = []
results_zeroed = []

for pos in eval_positions:
    # Intact
    p_intact = forward_at_position(pos)
    dp_i     = degen_prob(p_intact)
    results_intact.append((pos, dp_i, p_intact))

    # Zeroed
    with torch.no_grad():
        sr_module.weight.data[SUPER_ROW, :] = 0.0
    p_zeroed = forward_at_position(pos)
    dp_z     = degen_prob(p_zeroed)
    results_zeroed.append((pos, dp_z, p_zeroed))
    with torch.no_grad():
        sr_module.weight.data[SUPER_ROW, :] = sr_saved

    true_tok = id_to_tok.get(ids_list[pos], f"[{ids_list[pos]}]")
    ratio    = dp_z / dp_i if dp_i > 0 else float("nan")
    print(f"  {pos:>4}  {true_tok:<12}  {dp_i:>22.4f}  {dp_z:>22.4f}  {ratio:>8.2f}×")

# ── Step 3: Top-10 token comparison at first eval position ───────────────────
pos0 = eval_positions[0]
print(f"\n── Top-10 predicted tokens at position {pos0} "
      f"(true: '{id_to_tok.get(ids_list[pos0], '?')}') ──")
print(f"  {'Rank':>4}  {'intact token':<20}  {'prob':>8}    {'zeroed token':<20}  {'prob':>8}")
print("  " + "-" * 68)
top_i = top_k_tokens(results_intact[0][2], k=10)
top_z = top_k_tokens(results_zeroed[0][2], k=10)
for rank, ((ti, pi), (tz, pz)) in enumerate(zip(top_i, top_z), 1):
    print(f"  {rank:>4}  {ti:<20}  {pi:>8}    {tz:<20}  {pz:>8}")

# ── Step 4: Summary across all positions ─────────────────────────────────────
mean_dp_intact = sum(r[1] for r in results_intact) / len(results_intact)
mean_dp_zeroed = sum(r[1] for r in results_zeroed) / len(results_zeroed)
ratio_mean     = mean_dp_zeroed / mean_dp_intact if mean_dp_intact > 0 else float("nan")

print(f"\n── Summary ──")
print(f"  Mean degenerate-token probability:")
print(f"    Intact : {mean_dp_intact:.4f}")
print(f"    Zeroed : {mean_dp_zeroed:.4f}  (ratio {ratio_mean:.2f}×)")
if ratio_mean > 1.5:
    print("  → Removing the super row INFLATES degenerate token predictions.")
    print("    Consistent with Yu et al. stop-word inflation in NLP super weights.")
elif ratio_mean > 1.05:
    print("  → Weak inflation of degenerate tokens after super row removal.")
else:
    print("  → No significant degenerate-token inflation after super row removal.")
    print("    DNABERT-2 may differ from NLP super weights in this respect.")

# ── Step 5: Random-row controls ───────────────────────────────────────────────
import random
random.seed(42)
random_rows = random.sample([r for r in range(sr_module.weight.data.shape[0])
                             if r != SUPER_ROW], N_RANDOM)
print(f"\n── Random-row controls (N={N_RANDOM}) ──")
rand_ratios = []
for rr in random_rows:
    rand_saved = sr_module.weight.data[rr, :].clone()
    with torch.no_grad():
        sr_module.weight.data[rr, :] = 0.0

    dp_rand = [degen_prob(forward_at_position(pos)) for pos in eval_positions]
    mean_dp_rand = sum(dp_rand) / len(dp_rand)
    rand_ratio   = mean_dp_rand / mean_dp_intact if mean_dp_intact > 0 else float("nan")
    rand_ratios.append(rand_ratio)

    with torch.no_grad():
        sr_module.weight.data[rr, :] = rand_saved
    print(f"  row {rr:<6}:  mean_degen={mean_dp_rand:.4f}  ratio={rand_ratio:.2f}×")

mean_rand_ratio = sum(rand_ratios) / len(rand_ratios) if rand_ratios else float("nan")
print(f"\n  Super row ratio : {ratio_mean:.2f}×")
print(f"  Random row mean : {mean_rand_ratio:.2f}×")
print(f"  Specificity     : {ratio_mean / mean_rand_ratio:.1f}× above random baseline"
      if mean_rand_ratio > 0 else "  (random mean is 0)")
