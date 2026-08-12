# scripts/interpretability/run_neuron_broadcast_impulse_dnabert2.py
"""
Neuron-originated broadcast/impulse-response trace for the strongest candidate neuron —
directly comparable to the existing results/sw_broadcast_impulse.json entry for dnabert2
(which injects at the residual-stream coordinate row 603 at layer 5 on the BASE pretrained
model). This script instead perturbs the candidate hidden neuron i (pre-wo) at its own layer
on the FINE-TUNED splice classifier, lets it propagate naturally through that layer's own
wo + residual + LayerNorm, and traces the same T_m / C_m / KL metrics through all downstream
layers by reusing broadcast_metrics()/kl_divergence() from run_sw_broadcast_impulse.py
unchanged.

Also logs the *immediate* one-step delta-residual vector right after wo (before it re-enters
downstream layers), to show numerically whether the initial write is concentrated on row 603
(or the candidate's own strongest outgoing row) or already distributed.

The perturbation itself reuses the same natural-magnitude "shift toward negative-class mean"
condition already causally validated in run_neuron_causal_intervention_dnabert2.py (target =
neg_mean_allpos, alpha = 1.0), rather than the original script's fixed epsilon=1.0 residual
nudge — the two probe different physical quantities (a hidden neuron vs. a residual
coordinate) with different natural scales, so a shared literal epsilon would not be
comparable; the existing residual-coordinate result is instead cited directly from
results/sw_broadcast_impulse.json for comparison, not rerun (per the project's own maintained
"verified, don't rerun" set of prior results).

Usage:
    python scripts/interpretability/run_neuron_broadcast_impulse_dnabert2.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from neuron_pilot_common import ActivationGradCapture, NeuronIntervention, get_wo_module  # noqa: E402
from run_sw_broadcast_impulse import (  # noqa: E402
    _get_blocks, _get_logits, _hs_from_output, _tokenize,
    broadcast_metrics, kl_divergence,
)
from probes.dna_probes import get_probe  # noqa: E402

MODEL_ID = "zhihan1996/DNABERT-2-117M"
REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
PROBE_NAME = "actb_500"  # same probe used for the existing dnabert2 broadcast entry
DEFAULT_CKPT = ROOT / "results" / "gue_checkpoints" / "dnabert2_reconstructed" / "model_state.pt"


def run_pass_with_capture(model, inp_dict, blocks, arch, neuron_hook_layer=None, neuron_hook=None):
    """Mirrors run_sw_broadcast_impulse.run_pass()'s block-output capture logic exactly, but
    the perturbation (if any) is injected via a caller-supplied forward_pre_hook on mlp.wo at
    `neuron_hook_layer` (not via a block-output post-hook), so every block always just
    captures its own output."""
    captured = {}
    handles = []

    def make_capture(li):
        def hook(mod, inp, out):
            hs = _hs_from_output(out)
            captured[li] = (hs[0] if hs.dim() == 3 else hs).detach().cpu()
        return hook

    for li, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_capture(li)))

    neuron_handle = None
    if neuron_hook_layer is not None and neuron_hook is not None:
        wo = blocks[neuron_hook_layer].mlp.wo
        neuron_handle = wo.register_forward_pre_hook(neuron_hook)

    try:
        with torch.no_grad():
            out = model(**inp_dict)
    finally:
        for h in handles:
            h.remove()
        if neuron_handle is not None:
            neuron_handle.remove()

    return captured, out


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    interv = json.loads((ROOT / "results" / "neuron_pilot_dnabert2_intervention.json").read_text())
    layer = interv["strongest_candidate"]["layer"]
    neuron = interv["strongest_candidate"]["neuron"]
    strongest_cand = interv["shortlist_candidates"][0]
    neg_mean = strongest_cand["neg_mean_allpos"]
    print(f"[broadcast_neuron] strongest candidate: layer={layer} neuron={neuron} "
          f"neg_mean_allpos={neg_mean:.4f}")

    struct_path = ROOT / "results" / "neuron_pilot_dnabert2_structural.json"
    strongest_row = 603
    if struct_path.exists():
        struct = json.loads(struct_path.read_text())
        rep = next((c for c in struct["candidate_structural_reports"]
                    if c["layer"] == layer and c["neuron"] == neuron), None)
        if rep is not None:
            strongest_row = rep["strongest_outgoing_row"]
    print(f"[broadcast_neuron] tracing coordinate-preservation against row {strongest_row}")

    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, revision=REVISION)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=3, trust_remote_code=True, revision=REVISION)
    state = torch.load(DEFAULT_CKPT, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    blocks = _get_blocks(model, "bert")
    n_layers = len(blocks)
    seq = get_probe(PROBE_NAME)
    inp = _tokenize(tok, seq, "bert", device)
    n_tokens = inp["input_ids"].shape[1]
    print(f"[broadcast_neuron] probe={PROBE_NAME}  n_tokens={n_tokens}  n_layers={n_layers}")

    # ── clean pass (also captures h at the source layer to locate injection position) ──
    cap = ActivationGradCapture()
    wo_source = get_wo_module(model.bert, layer)
    handle = wo_source.register_forward_pre_hook(cap)
    clean_hs, clean_out = run_pass_with_capture(model, inp, blocks, "bert")
    handle.remove()
    # single-sequence probe (B=1, no padding): DNABERT-2's internal unpad/pad makes the flat
    # (total_nnz, D_FFN) tensor at mlp.wo equal to (n_tokens, D_FFN) directly here — no batch
    # dim to index off (see neuron_pilot_common module docstring).
    h_clean = cap.h.detach()
    assert h_clean.dim() == 2 and h_clean.shape[0] == n_tokens, (
        f"expected flat (n_tokens={n_tokens}, D_FFN) at mlp.wo, got shape {tuple(h_clean.shape)}")
    inject_pos = int(h_clean[:, neuron].abs().argmax().item())
    clean_activation_at_pos = float(h_clean[inject_pos, neuron].item())
    print(f"[broadcast_neuron] injection token position (max |activation|): {inject_pos}  "
          f"clean_activation={clean_activation_at_pos:.4f}")

    logits_clean = _get_logits(clean_out, "bert")

    # ── perturbed pass: shift neuron toward its natural negative-class mean at inject_pos ──
    # single-sequence probe (B=1, no padding) -> the flat row index at mlp.wo (see
    # neuron_pilot_common module docstring) equals the local token position directly.
    iv = NeuronIntervention(neuron_idx=neuron, target=neg_mean, alpha=1.0,
                            row_indices=[inject_pos])
    pert_hs, pert_out = run_pass_with_capture(model, inp, blocks, "bert",
                                              neuron_hook_layer=layer, neuron_hook=iv)
    logits_pert = _get_logits(pert_out, "bert")

    # ── immediate one-step delta-residual right after wo (source layer's own output) ──
    delta_source = (pert_hs[layer][inject_pos].float() - clean_hs[layer][inject_pos].float())
    delta_source_np = delta_source.numpy()
    top5_idx = np.argsort(-np.abs(delta_source_np))[:5]
    immediate_write = {
        "layer": layer, "position": inject_pos,
        "delta_norm": float(np.linalg.norm(delta_source_np)),
        "delta_at_strongest_row": float(delta_source_np[strongest_row]),
        "fraction_of_norm_at_strongest_row": float(
            abs(delta_source_np[strongest_row]) / (np.linalg.norm(delta_source_np) + 1e-12)),
        "top5_rows_by_abs_delta": [
            {"row": int(r), "delta": float(delta_source_np[r])} for r in top5_idx
        ],
    }
    print(f"[broadcast_neuron] immediate write at source layer {layer}: "
          f"||delta||={immediate_write['delta_norm']:.4f}  "
          f"delta[row {strongest_row}]={immediate_write['delta_at_strongest_row']:+.4f}  "
          f"(fraction of norm = {immediate_write['fraction_of_norm_at_strongest_row']:.4f})")
    print(f"[broadcast_neuron] top-5 rows by |delta| at source layer: {immediate_write['top5_rows_by_abs_delta']}")

    # ── downstream trace (reuses the exact metric functions from the existing SW broadcast
    #    assay, evaluated for coordinate preservation against `strongest_row`) ──
    metrics = broadcast_metrics(clean_hs, pert_hs, strongest_row, layer, epsilon=1.0)
    kl = kl_divergence(logits_clean, logits_pert, inject_pos, is_causal=False)
    print(f"[broadcast_neuron] functional output KL (3-class splice logits): {kl:.4e}")
    for li in sorted(metrics.keys()):
        m = metrics[li]
        print(f"  layer {li:2d}  T_mean={m['T_mean']:.4f}  C(row {strongest_row})={m['C']:+.4f}")

    out = {
        "model": "dnabert2", "checkpoint": "fine-tuned splice/reconstructed",
        "layer": layer, "neuron": neuron, "strongest_outgoing_row": strongest_row,
        "probe": PROBE_NAME, "n_tokens": n_tokens, "n_layers": n_layers,
        "inject_pos": inject_pos, "clean_activation_at_inject_pos": clean_activation_at_pos,
        "perturbation": {"target": neg_mean, "alpha": 1.0, "type": "shift_to_neg_class_mean"},
        "immediate_write_at_source_layer": immediate_write,
        "downstream_metrics": {str(k): v for k, v in metrics.items()},
        "kl_functional_output": kl,
        "comparison_note": (
            "Compare against results/sw_broadcast_impulse.json['dnabert2']: that entry "
            "injects epsilon=1.0 directly into the residual-stream row-603 coordinate at "
            "layer 5 on the BASE pretrained model (C~-0.002..+0.025, kl_sw=0.310). This "
            "entry instead perturbs the upstream hidden NEURON on the FINE-TUNED classifier "
            "and lets the write propagate naturally through wo+residual+LN."
        ),
        "elapsed_sec": time.time() - t0,
    }
    out_path = ROOT / "results" / "neuron_pilot_dnabert2_broadcast_impulse.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[broadcast_neuron] wrote {out_path}")
    print(f"[broadcast_neuron] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
