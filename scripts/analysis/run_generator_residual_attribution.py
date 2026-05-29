"""
scripts/analysis/run_generator_residual_attribution.py
-------------------------------------------------------
Per-layer residual-stream attribution for GENERator (EUK and PROK),
generating Fig 2 panels A/B data (activation lifecycle).

For each layer we record, at the token position where the primary SW row
is most active (max |h[pos, sw_row]|):

    residual_in_at_sw_pos  — residual stream value at sw_row ENTERING the layer
    mlp_out_at_sw_pos      — MLP contribution at sw_row (signed)
    residual_out_at_sw_pos — residual stream value at sw_row AFTER the layer add

This traces how the super-weight channel builds up across layers.

LlamaDecoderLayer forward (simplified):
    residual = h
    h = attn(input_layernorm(h)) + residual      # post-attention
    residual = h
    h = mlp(post_attention_layernorm(h)) + residual  # post-mlp

We hook:
  - model.model.layers[i]       → captures residual_in (layer input), residual_out (layer output)
  - model.model.layers[i].mlp   → captures mlp_out (MLP contribution, before residual add)

SW rows (primary):
  EUK:  layer 4, row 2371
  PROK: layer 2, row 1927

Probes:
  EUK:  actb_500        (504 bp eukaryote ACTB CDS)
  PROK: pseudomonadota  (1056 bp Pseudomonas putida)

Output:
  results/activation_lifecycle_generator.json
  results/activation_lifecycle_generator_prokaryote.json

Usage:
  python scripts/analysis/run_generator_residual_attribution.py --variant euk
  python scripts/analysis/run_generator_residual_attribution.py --variant prok
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent

VARIANT_CFG = {
    "euk": {
        "model_key": "generator",
        "sw_row":    2371,
        "probe":     "actb_500",
        "out":       "results/activation_lifecycle_generator.json",
    },
    "prok": {
        "model_key": "generator_prokaryote",
        "sw_row":    1927,
        "probe":     "pseudomonadota",
        "out":       "results/activation_lifecycle_generator_prokaryote.json",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="GENERator residual attribution")
    parser.add_argument("--variant", choices=["euk", "prok"], required=True)
    args = parser.parse_args()

    vc = VARIANT_CFG[args.variant]
    model_key: str = vc["model_key"]
    sw_row:    int = vc["sw_row"]
    probe_name: str = vc["probe"]
    out_path = ROOT / vc["out"]

    cfg = yaml.safe_load((ROOT / f"configs/{model_key}.yaml").read_text())
    n_layers: int = cfg["num_layers"]

    import sys
    sys.path.insert(0, str(ROOT))
    from probes.dna_probes import get_probe

    probe_seq = get_probe(probe_name)
    print(f"[residual_attr] variant={args.variant}  sw_row={sw_row}  "
          f"probe={probe_name} ({len(probe_seq)} bp)")

    print(f"[residual_attr] loading {cfg['model_id']} …")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"],
        trust_remote_code=cfg.get("hf_trust_remote_code", True),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        trust_remote_code=cfg.get("hf_trust_remote_code", True),
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()

    # GENERator 6-mer tokenizer: length must be divisible by 6
    remainder = len(probe_seq) % 6
    if remainder:
        probe_seq = probe_seq[remainder:]
    inputs = tokenizer(probe_seq, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_tokens = inputs["input_ids"].shape[1]
    print(f"[residual_attr] tokenised to {n_tokens} tokens")

    # Per-layer accumulators (indexed by layer index)
    layer_records:  list[dict] = [{} for _ in range(n_layers)]
    # layer_in_hs[li]  = hidden states entering layer li  (1, T, d)
    # layer_out_hs[li] = hidden states exiting layer li   (1, T, d)
    # mlp_out_hs[li]   = MLP output for layer li          (1, T, d)
    layer_in:  dict[int, torch.Tensor] = {}
    layer_out: dict[int, torch.Tensor] = {}
    mlp_out:   dict[int, torch.Tensor] = {}

    def make_layer_hook(li: int):
        def hook(module, inputs, output):
            # inputs[0] = hidden states entering the full decoder layer
            # output[0] = hidden states exiting the full decoder layer (tuple)
            hs_in  = inputs[0].detach().float().cpu()
            hs_out = (output[0] if isinstance(output, tuple) else output).detach().float().cpu()
            layer_in[li]  = hs_in
            layer_out[li] = hs_out
        return hook

    def make_mlp_hook(li: int):
        def hook(module, inputs, output):
            # output = MLP's computed value BEFORE the residual add
            m = (output[0] if isinstance(output, tuple) else output).detach().float().cpu()
            mlp_out[li] = m
        return hook

    handles = []
    for li in range(n_layers):
        handles.append(
            model.model.layers[li].register_forward_hook(make_layer_hook(li))
        )
        handles.append(
            model.model.layers[li].mlp.register_forward_hook(make_mlp_hook(li))
        )

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    print(f"\n{'layer':>5}  {'sw_pos':>6}  {'residual_in':>12}  "
          f"{'mlp_out':>10}  {'residual_out':>13}  {'max_abs':>10}")
    print("-" * 70)

    records: list[dict] = []
    for li in range(n_layers):
        hs_in  = layer_in.get(li)   # (1, T, d)
        hs_out = layer_out.get(li)  # (1, T, d)
        m_out  = mlp_out.get(li)    # (1, T, d)

        if hs_in is None or hs_out is None:
            records.append({"layer": li, "error": "hook not captured"})
            continue

        # token position with max |residual_out[sw_row]|
        col_out = hs_out[0, :, sw_row]          # (T,)
        sw_pos  = int(col_out.abs().argmax().item())

        rin  = float(hs_in[0, sw_pos, sw_row].item())
        rout = float(hs_out[0, sw_pos, sw_row].item())
        mval = float(m_out[0, sw_pos, sw_row].item()) if m_out is not None else float("nan")
        max_abs = float(col_out.abs().max().item())

        rec = {
            "layer":                 li,
            "sw_pos":                sw_pos,
            "max_abs_at_sw_pos":     max_abs,
            "residual_in_at_sw_pos": rin,
            "mlp_out_at_sw_pos":     mval,
            "residual_out_at_sw_pos": rout,
            # convenience aliases for plotting
            "residual_in":           rin,
            "mlp_out":               mval,
            "residual_out":          rout,
        }
        records.append(rec)
        print(f"{li:5d}  {sw_pos:6d}  {rin:12.3f}  {mval:10.3f}  "
              f"{rout:13.3f}  {max_abs:10.3f}")

    out = {
        "model":      model_key,
        "probe":      probe_name,
        "n_tokens":   n_tokens,
        "sw_row":     sw_row,
        "per_layer":  records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[residual_attr] wrote {out_path}")


if __name__ == "__main__":
    main()
