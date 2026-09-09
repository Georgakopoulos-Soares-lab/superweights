#!/usr/bin/env python
"""Round-2 Section 3: detector reconciliation for grandfathered (legacy-sourced) models.

Follows the pattern of E13_full_cohort_causal_census/run_dnabert2_repro_diagnostic.py:
hook every layer's down_proj output on the canonical 504-bp ACTB discovery probe (genomic
models) under two input-boundary conventions, compute the ratio-argmax and activation-argmax
coordinates and the frozen candidate's rank under each, then compute exact q1/PR_spec/
||U_k||_F for the ratio-argmax coordinate via the same row_spectral_metrics function the
rest of the structural analysis uses.

Usage: python section3_detector_recheck.py --model generator_euk
       python section3_detector_recheck.py --model ntv3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "manuscript/experiments/E7_exact_dimensionality"))
from spectral_lib import row_spectral_metrics  # noqa: E402

RATIO_THRESHOLD = 5.0

_ACTB_CDS = (
    "ATGGATGATGATATCGCCGCGCTCGTCGTCGACAACGGCTCCGGCATGTGCAAAGCCGGCTTCGCGGGCGACGATGCCCCGAGGGCC"
    "GTCTTCCCCTCCATCGTGGGGCGCCCCAGGCACCAGGGCGTGATGGTGGGCATGGGTCAGAAGGATTCCTATGTGGGCGACGAGGCC"
    "CAGAGCAAGAGAGGCATCCTCACCCTGAAGTACCCCATCGAGCACGGCATCGTCACCAACTGGGACGACATGGAGAAAATCTGGCAC"
    "CACACCTTCTACAATGAGCTGCGTGTGGCTCCCGAGGAGCACCCCGTGCTGCTCACCGAGGCCCCCCTGAACCCGAAGGCCAACCGC"
    "GAGAAGATGACCCAGATCATGTTTGAGACCTTCAATACCCCCGCCATGTACGTTGCTATCCAGGCTGTGCTATCCCTGTACGCCTCT"
    "GGCCGTACCACTGGCATCGTGATGGACTCCGGTGACGGGGTCACCCACACTGTGCCCATCTACGAGGGGTATGCCCTCCCCCATGCC"
    "ATCCTGCGTCTGGACCTGGCTGGCCGGGACCTGACTGACTACCTCATGAAGATCCTCACCGAGCGCGGCTACAGCTTCACCACCACG"
    "GCCGAGCGGGAAATCGTGCGTGACATTAAGGAGAAGCTGTGCTACGTCGCCCTGGACTTCGAGCAAGAGATGGCCACGGCTGCTTCC"
    "AGCTCCTCCCTGGAGAAGAGCTACGAGCTGCCTGACGGCCAGGTCATCACCATTGGCAATGAGCGGTTCCGCTGCCCTGAGGCACTC"
    "TTCCAGCCTTCCTTCCTGGGCATGGAGTCCTGTGGCATCCACGAAACTACCTTCAACTCCATCATGAAGTGTGACGTGGACATCCGC"
    "AAAGACCTGTACGCCAACACAGTGCTGTCTGGCGGCACCACCATGTACCCTGGCATTGCCGACAGGATGCAGAAGGAGATCACTGCC"
    "CTGGCACCCAGCACAATGAAGATCAAGATCATTGCTCCTCCTGAGCGCAAGTACTCCGTGTGGATCGGCGGCTCCATCCTGGCCTCG"
    "CTGTCCACCTTCCAGCAGATGTGGATCAGCAAGCAGGAGTATGACGAGTCCGGCCCCTCCATCGTCCACCGCAAATGCTTCTAA"
)
_ACTB_500 = _ACTB_CDS[:504]

MODELS = {
    "generator_euk": dict(config="configs/generator.yaml", wrapper="generator",
                           frozen_layer=4, frozen_row=2371, n_layers=30,
                           pattern="model.layers.{i}.mlp.down_proj", ffn_kind="separate"),
    "ntv3": dict(config="configs/ntv3.yaml", wrapper="ntv3",
                 frozen_layer=11, frozen_row=1472, n_layers=12,
                 pattern="core.transformer_blocks.{i}.fc2", ffn_kind="packed_fc1_fc2"),
}


def gate_up_for(mods, pattern, layer, ffn_kind):
    """Return (W_gate, W_up) for the layer, architecture-agnostic."""
    if ffn_kind == "separate":
        base = pattern.format(i=layer)
        gate = mods[base.replace("down_proj", "gate_proj")].weight.detach().cpu()
        up = mods[base.replace("down_proj", "up_proj")].weight.detach().cpu()
        return gate, up
    if ffn_kind == "packed_fc1_fc2":
        fc1_key = pattern.format(i=layer).replace("fc2", "fc1")
        packed = mods[fc1_key].weight.detach().cpu()
        d_ffn = packed.shape[0] // 2
        return packed[:d_ffn], packed[d_ffn:]
    raise ValueError(ffn_kind)


class Recorder:
    def __init__(self):
        self.values = {}
        self.handles = []

    def add(self, layer, module):
        self.handles.append(module.register_forward_hook(self._hook(layer)))

    def _hook(self, layer):
        def fn(module, args, output):
            y = output[0] if isinstance(output, tuple) else output
            self.values[layer] = y.reshape(-1, y.shape[-1]).abs().max(dim=0).values.detach().cpu()
        return fn

    def close(self):
        for h in self.handles:
            h.remove()


def run_one_preprocessing(model, tok, pattern, n_layers, seq, add_special_tokens, label, dev):
    rec = Recorder()
    mods = dict(model.named_modules())
    for layer in range(n_layers):
        rec.add(layer, mods[pattern.format(i=layer)])
    enc = tok(seq, return_tensors="pt", add_special_tokens=add_special_tokens).to(dev)
    n_tokens = enc["input_ids"].shape[1]
    with torch.no_grad():
        model(**enc)
    rec.close()

    rows = []
    for layer, values in sorted(rec.values.items()):
        median = float(values.median())
        for row, activation_max in enumerate(values.tolist()):
            rows.append(dict(layer=layer, row=row, activation_max=activation_max,
                              activation_median=median,
                              detector_ratio=activation_max / median if median > 0 else float("inf")))
    by_ratio = sorted(rows, key=lambda x: (-x["detector_ratio"], x["layer"], x["row"]))
    by_act = sorted(rows, key=lambda x: (-x["activation_max"], x["layer"], x["row"]))
    for rank, r in enumerate(by_ratio, 1):
        r["ratio_rank"] = rank
    for rank, r in enumerate(by_act, 1):
        r["activation_rank"] = rank
    lookup = {(r["layer"], r["row"]): r for r in rows}
    print(f"  [{label}] n_tokens={n_tokens}  global ratio-argmax = L{by_ratio[0]['layer']}/r{by_ratio[0]['row']} "
          f"(ratio={by_ratio[0]['detector_ratio']:.4f})  "
          f"global activation-argmax = L{by_act[0]['layer']}/r{by_act[0]['row']} "
          f"(activation={by_act[0]['activation_max']:.4f})")
    return lookup, by_ratio[0], by_act[0]


def load_generator_euk():
    from models import WRAPPER_MAP
    cfg = yaml.safe_load((ROOT / "configs/generator.yaml").read_text())
    w = WRAPPER_MAP["generator"](cfg)
    w.load()
    return w.model.eval(), w.tokenizer, cfg["down_proj_pattern"], cfg["num_layers"]


_NTV3_CODE_REV = "0ecff3637f0d3ba5b686d1095083218157c2ca34"  # pinned code_revision, matches
                                                              # scripts/analysis/run_ntv3_uk_audit.py


def load_ntv3():
    # NOT via models.WRAPPER_MAP["ntv3"] -- that wrapper's AutoTokenizer.from_pretrained
    # call (models/ntv3_wrapper.py:10) omits trust_remote_code=True and hangs on an
    # interactive remote-code confirmation prompt in a non-interactive session (confirmed
    # this session: EOFError under nohup). Bypassing it with the same direct-load pattern
    # already proven working in run_ntv3_uk_audit.py instead of patching the wrapper
    # (out of scope for this audit).
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    cfg = yaml.safe_load((ROOT / "configs/ntv3.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True,
                                         code_revision=_NTV3_CODE_REV)
    model = AutoModelForMaskedLM.from_pretrained(cfg["model_id"], trust_remote_code=True,
                                                  code_revision=_NTV3_CODE_REV,
                                                  torch_dtype=torch.float32)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tok, cfg["down_proj_pattern"], cfg["num_layers"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    args = ap.parse_args()
    spec = MODELS[args.model]

    t0 = time.time()
    if args.model == "generator_euk":
        model, tok, pattern, n_layers = load_generator_euk()
    elif args.model == "ntv3":
        model, tok, pattern, n_layers = load_ntv3()
    else:
        raise NotImplementedError(args.model)
    dev = next(model.parameters()).device
    print(f"loaded {args.model} in {time.time()-t0:.1f}s, n_layers={n_layers}, pattern={pattern}")

    frozen = (spec["frozen_layer"], spec["frozen_row"])
    print(f"frozen candidate: L{frozen[0]}/r{frozen[1]}")

    results = {}
    for add_specials, label in [(False, "wrapper_convention_manual_bos_no_specials"),
                                 (True, "tokenizer_default_add_special_tokens_true")]:
        if label == "wrapper_convention_manual_bos_no_specials" and args.model == "generator_euk":
            bos = tok.bos_token or ""
            seq = bos + _ACTB_500  # matches GeneratorWrapper._prepare_sequence exactly (504%6==0, no trim needed)
        else:
            # NTv3's own forward() (models/ntv3_wrapper.py) calls the tokenizer with no
            # add_special_tokens override, i.e. its "wrapper convention" is the
            # add_special_tokens=True case tested here; the False case is the alternative
            # being checked against it, same spirit as DNABERT-2's diagnostic.
            seq = _ACTB_500
            if args.model == "ntv3":
                # NTv3 is a conv/deconv U-Net (8 stride-2 conv blocks); its deconv skip
                # connections only align when token length is a multiple of 2**8=256
                # (same constraint documented in run_pretrained_epistasis.py's
                # _pad_to_multiple, used there for the same model). 504bp -> pad with 'A'
                # to 512bp. Confirmed necessary this session: the unpadded 504bp probe
                # crashes with a tensor-size mismatch inside modeling_ntv3_pretrained.py's
                # skip-connection add.
                multiple = 256
                target = ((len(seq) + multiple - 1) // multiple) * multiple
                seq = seq + "A" * (target - len(seq))
        lookup, top_ratio, top_act = run_one_preprocessing(
            model, tok, pattern, n_layers, seq, add_specials, label, dev)
        frozen_entry = lookup.get(frozen)
        print(f"    frozen L{frozen[0]}/r{frozen[1]}: activation={frozen_entry['activation_max']:.4f} "
              f"ratio={frozen_entry['detector_ratio']:.4f} "
              f"activation_rank={frozen_entry['activation_rank']} ratio_rank={frozen_entry['ratio_rank']} "
              f"passes_ratio5={frozen_entry['detector_ratio']>=RATIO_THRESHOLD}")
        results[label] = dict(top_ratio=top_ratio, top_act=top_act, frozen=frozen_entry,
                               n_rows=len(lookup))

    # structural metrics for the ratio-argmax coordinate under each preprocessing, plus frozen candidate
    print("\nStructural metrics (weight-only, preprocessing-independent):")
    mods = dict(model.named_modules())
    coords_to_check = {frozen}
    for label, r in results.items():
        coords_to_check.add((r["top_ratio"]["layer"], r["top_ratio"]["row"]))
    struct = {}
    for (layer, row) in coords_to_check:
        gate, up = gate_up_for(mods, pattern, layer, spec["ffn_kind"])
        down_row = mods[pattern.format(i=layer)].weight[row].detach().cpu()
        metrics, _ = row_spectral_metrics(gate, up, down_row)
        struct[(layer, row)] = metrics
        print(f"  L{layer}/r{row}: q1={metrics.q1:.6f} PR_spec={metrics.pr_spec:.6f} "
              f"||U_k||_F={metrics.frob_norm:.6f}")

    out = {"model": args.model, "frozen_candidate": frozen,
           "results_by_preprocessing": {
               k: {"top_ratio_coord": (v["top_ratio"]["layer"], v["top_ratio"]["row"]),
                   "top_ratio_value": v["top_ratio"]["detector_ratio"],
                   "top_activation_coord": (v["top_act"]["layer"], v["top_act"]["row"]),
                   "top_activation_value": v["top_act"]["activation_max"],
                   "frozen_activation": v["frozen"]["activation_max"],
                   "frozen_ratio": v["frozen"]["detector_ratio"],
                   "frozen_activation_rank": v["frozen"]["activation_rank"],
                   "frozen_ratio_rank": v["frozen"]["ratio_rank"],
                   "frozen_passes_ratio5": v["frozen"]["detector_ratio"] >= RATIO_THRESHOLD}
               for k, v in results.items()},
           "structural_metrics": {f"L{l}r{r}": dict(q1=m.q1, pr_spec=m.pr_spec, frob_norm=m.frob_norm)
                                   for (l, r), m in struct.items()}}
    out_path = ROOT / f"audit/round2/section3_{args.model}_detector_recheck.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
