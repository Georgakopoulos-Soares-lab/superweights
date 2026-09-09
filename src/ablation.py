# analysis/ablation.py
"""
Scalar quality metrics for ablation tests, dispatching by model type.

Public API
----------
causal_perplexity(wrapper, sequence) -> float
    Next-token cross-entropy perplexity for decoder-only models (GENERator, GPT-style).

masked_token_entropy(wrapper, sequence) -> float
    Mean Shannon entropy of the masked-token output distributions for encoder/MLM models
    (e.g. DNABERT-2, NTv3).  Each non-special token is individually replaced with [MASK]
    (in batches of 32), the model predicts, and we average H(p) across all positions.
    Higher = more uncertain = worse model.  When the super weight is removed, this rises.

model_metric(wrapper, sequence) -> float
    Dispatches to causal_perplexity if wrapper.config["causal"] is True,
    else masked_token_entropy.

run_destruction_test(wrapper, sw_list, probe, detection_mode, n_random_controls) -> dict
    Compute baseline metric, zero all SW rows/scalars, re-measure, restore, then repeat
    for n_random_controls independent random-row ablations.
    Returns keys: baseline, pruned, delta_pct, pruned_rand_mean, delta_rand_pct.
    Used by scripts/run_ablation.py.
"""

import random
import torch

# ---------------------------------------------------------------------------
# Causal perplexity (decoder LLMs: GENERator, GPT …)
# ---------------------------------------------------------------------------

def causal_perplexity(wrapper, sequence: str) -> float:
    """Return exp(CE loss) for a single DNA sequence on a causal LM wrapper."""
    tokenizer = wrapper.tokenizer
    model     = wrapper.model

    # GENERator and similar wrappers may enforce 6-mer alignment
    if hasattr(wrapper, "_prepare_sequence"):
        seq = wrapper._prepare_sequence(sequence)
    else:
        seq = sequence

    # Non-HF tokenizers (e.g. Evo's CharLevelTokenizer) can't be called
    # like `tokenizer(seq, return_tensors=...)`. Delegate to the wrapper's
    # own perplexity routine when available.
    if not callable(tokenizer) or hasattr(wrapper, "_skip_hf_tokenize"):
        return wrapper.compute_perplexity(seq)
    try:
        inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=False)
    except TypeError:
        return wrapper.compute_perplexity(seq)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    ids    = inputs["input_ids"]

    with torch.no_grad():
        out = model(**inputs, labels=ids)
    return torch.exp(out.loss).item()


# ---------------------------------------------------------------------------
# Masked-token entropy (encoder/MLM: DNABERT-2, NTv3 …)
# ---------------------------------------------------------------------------

def masked_token_entropy(wrapper, sequence: str, batch_size: int = 32) -> float:
    """
    Mean Shannon entropy H(p_i) over all non-special token positions i.

    For each position i:
      1. Replace token i with [MASK], keep all others unchanged.
      2. Run a forward pass and collect the softmax distribution p_i over vocab.
      3. Compute H_i = -sum_v p_i(v) * log(p_i(v))

    Positions are processed in batches for efficiency.

    Returns the mean H_i across all non-special positions.  When the super
    weight is zeroed the model becomes less certain, so this value rises.
    """
    tokenizer = wrapper.tokenizer
    model     = wrapper.model

    enc       = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"].to(model.device)          # [1, L]
    attn_mask = enc.get("attention_mask",
                        torch.ones_like(input_ids)).to(model.device)

    mask_id = tokenizer.mask_token_id

    # Collect special token ids to skip
    special_ids: set = set()
    for attr in ("cls_token_id", "sep_token_id", "pad_token_id",
                 "bos_token_id", "eos_token_id", "mask_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            special_ids.add(tid)

    ids_list    = input_ids[0].tolist()
    non_special = [i for i, t in enumerate(ids_list) if t not in special_ids]

    if not non_special:
        return 0.0

    entropies = []
    for start in range(0, len(non_special), batch_size):
        positions = non_special[start : start + batch_size]
        B = len(positions)

        # Batch: B copies of the input_ids, each with one position masked
        batch_ids  = input_ids.expand(B, -1).clone()   # [B, L]
        batch_mask = attn_mask.expand(B, -1).clone()   # [B, L]

        for bi, pos in enumerate(positions):
            batch_ids[bi, pos] = mask_id

        with torch.no_grad():
            logits = model(
                input_ids=batch_ids, attention_mask=batch_mask
            ).logits  # [B, L, V]

        for bi, pos in enumerate(positions):
            p   = torch.softmax(logits[bi, pos].float(), dim=-1)
            H_i = -(p * p.clamp_min(1e-10).log()).sum().item()
            entropies.append(H_i)

    return sum(entropies) / len(entropies)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def model_metric(wrapper, sequence: str) -> float:
    """Return the appropriate quality metric based on wrapper.config['causal']."""
    if wrapper.config.get("causal", False):
        return causal_perplexity(wrapper, sequence)
    return masked_token_entropy(wrapper, sequence)


# ---------------------------------------------------------------------------
# Destruction test (used by scripts/run_ablation.py)
# ---------------------------------------------------------------------------

def run_destruction_test(
    wrapper,
    sw_list: list,
    probe: str,
    detection_mode: str = "superrow",
    n_random_controls: int = 10,
    random_seed: int = 42,
) -> dict:
    """
    Measure model quality before/after pruning super weights, with random controls.

    Parameters
    ----------
    wrapper        : model wrapper (has .get_target_module, .config)
    sw_list        : list of dicts with keys 'layer', 'row', and optionally 'col'
    probe          : raw DNA sequence string
    detection_mode : 'superrow' → zero full row; 'superweight' → zero single scalar
    n_random_controls : number of random ablations for the noise floor

    Returns
    -------
    dict with keys:
      baseline          – scalar metric on intact model
      pruned            – scalar metric after zeroing super weights
      delta_pct         – (pruned – baseline) / baseline * 100
      pruned_rand_mean  – mean baseline-scale metric over random controls
      delta_rand_pct    – mean delta_pct over random controls
    """
    rng = random.Random(random_seed)

    baseline = model_metric(wrapper, probe)

    # ── Prune super weights ───────────────────────────────────────────────────
    saved_sw = []
    for sw in sw_list:
        m = wrapper.get_target_module(sw["layer"])
        if detection_mode == "superrow":
            saved_sw.append((sw["layer"], sw["row"], None,
                             m.weight.data[sw["row"], :].clone()))
            with torch.no_grad():
                m.weight.data[sw["row"], :] = 0.0
        else:  # superweight — single scalar
            col = sw["col"]
            saved_sw.append((sw["layer"], sw["row"], col,
                             m.weight.data[sw["row"], col].clone()))
            with torch.no_grad():
                m.weight.data[sw["row"], col] = 0.0

    pruned    = model_metric(wrapper, probe)
    delta_pct = (pruned - baseline) / abs(baseline) * 100 if baseline != 0 else 0.0

    # Restore
    for layer, row, col, saved in saved_sw:
        m = wrapper.get_target_module(layer)
        with torch.no_grad():
            if col is None:
                m.weight.data[row, :] = saved
            else:
                m.weight.data[row, col] = saved

    # ── Random controls ───────────────────────────────────────────────────────
    sw_coords   = {(sw["layer"], sw["row"]) for sw in sw_list}
    rand_deltas = []

    for _ in range(n_random_controls):
        chosen = []
        for sw in sw_list:
            m         = wrapper.get_target_module(sw["layer"])
            nrows     = m.weight.data.shape[0]
            cands     = [r for r in range(nrows) if (sw["layer"], r) not in sw_coords]
            chosen.append((sw["layer"], rng.choice(cands)))

        rand_saved = []
        for layer, row in chosen:
            m = wrapper.get_target_module(layer)
            rand_saved.append((layer, row, m.weight.data[row, :].clone()))
            with torch.no_grad():
                m.weight.data[row, :] = 0.0

        rand_val = model_metric(wrapper, probe)
        rand_deltas.append(
            (rand_val - baseline) / abs(baseline) * 100 if baseline != 0 else 0.0
        )

        for layer, row, saved in rand_saved:
            m = wrapper.get_target_module(layer)
            with torch.no_grad():
                m.weight.data[row, :] = saved

    rand_mean_delta = sum(rand_deltas) / len(rand_deltas) if rand_deltas else 0.0
    rand_mean_val   = baseline * (1 + rand_mean_delta / 100)

    return {
        "baseline":         baseline,
        "pruned":           pruned,
        "delta_pct":        delta_pct,
        "pruned_rand_mean": rand_mean_val,
        "delta_rand_pct":   rand_mean_delta,
    }
