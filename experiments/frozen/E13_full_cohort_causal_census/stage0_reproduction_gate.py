#!/usr/bin/env python
"""
experiments/E13_full_cohort_causal_census/stage0_reproduction_gate.py

Stage 0 (reproduction gate) of PREREG_full_cohort_causal_census.md. For each Stage-0 model,
measures baseline loss, epsilon=0.5, and epsilon=1.0 relative_loss_change on the model's
existing structural candidate row, using ONLY the reused intervention/evaluation code
already validated by E9/E10/E12 -- no new mechanism, no new endpoint, no candidate
reselection. Compares the new number against the old locked value cited in the PREREG.

Reuses, unmodified:
  - E10_nlp_architecture_causal/e10_lib.py (decoder causal-LM NLL, encoder MLM loss,
    row-scale intervention/`masked` context manager)
  - E9_mechanistic_tomography/tomography_lib.py (DNABERT-2 MLM loss stack, row scaling)
  - The exact PATTERN strings, repos, revisions, N_WINDOWS/MAX_LEN/BATCH_SIZE/MASK_PROB
    already frozen in run_decoder_spectrum.py / run_encoder_tomography.py /
    run_dnabert2_measurements.py -- copied here verbatim, not re-derived.

New same-layer control rows are drawn from THIS protocol's own spawn(23) panel
(panel.py) and their identities are recorded for provenance, but running the full
intervention on all 5 controls for all 6 models is treated as optional/time-permitting
(the PREREG's binding Stage-0 requirement is the candidate row only) -- see
STAGE0_REPORT.md for which models got control measurements too.

Usage:
  python stage0_reproduction_gate.py --model llama
  python stage0_reproduction_gate.py --model llama --with-controls
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SALVAGE = HERE.parents[1]
ROOT = SALVAGE.parent
sys.path.insert(0, str(SALVAGE / "experiments" / "E10_nlp_architecture_causal"))
sys.path.insert(0, str(SALVAGE / "experiments" / "E9_mechanistic_tomography"))
sys.path.insert(0, str(HERE))

import e10_lib as L  # noqa: E402
import tomography_lib as tl  # noqa: E402
from panel import PANEL_ORDER, N_PANEL  # noqa: E402
from dnabert2_compat import load_dnabert2_patched  # noqa: E402

PREREG = SALVAGE / "docs" / "prereg" / "PREREG_full_cohort_causal_census.md"
EPSILONS = [0.5, 1.0]
RESULTS_DIR = ROOT / "results" / "E13"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = RESULTS_DIR / "stage0_raw_responses.json"
CSV_PATH = RESULTS_DIR / "stage0_reproduction_gate.csv"


def verify_lock() -> None:
    r = subprocess.run(
        [sys.executable, str(SALVAGE / "src" / "prereg_lock.py"), "verify", str(PREREG)],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    if r.returncode != 0 or "OK" not in r.stdout:
        print(r.stderr, file=sys.stderr)
        raise SystemExit("prereg_lock verify FAILED -- refusing to run any forward pass.")


def new_control_rows(model_key_in_panel: str, layer: int, d_model: int, exclude: set[int],
                      n: int = 5) -> list[int]:
    idx = PANEL_ORDER.index(model_key_in_panel)
    seed_seq = np.random.SeedSequence(42).spawn(N_PANEL)[idx]
    rng = np.random.default_rng(seed_seq)
    pool = [r for r in range(d_model) if r not in exclude]
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


# ── Decoder (Llama/Mistral/OLMo/Phi-3) ────────────────────────────────────────

DECODER_PATTERN = "model.layers.{i}.mlp.down_proj"
DECODER_N_WINDOWS = 100
DECODER_MAX_TOKENS = 512

DECODER_PANEL = {
    "llama": dict(repo="huggyllama/llama-7b", revision=None, batch_size=8,
                  d_model=4096, layer=2, row=3968,
                  old_baseline=2.2452529793756115, local_files_only=True),
    "mistral": dict(repo="mistralai/Mistral-7B-v0.1", revision=None, batch_size=8,
                     d_model=4096, layer=1, row=2070,
                     old_baseline=2.2072248807485324, local_files_only=True),
    "olmo": dict(repo="allenai/OLMo-7B-0724-hf", revision=None, batch_size=4,
                 d_model=4096, layer=1, row=269,
                 old_baseline=2.4953157010610325, local_files_only=True),
    "phi3": dict(repo="microsoft/Phi-3-mini-4k-instruct",
                 revision="f39ac1d28e925b323eae81227eaba4464caced4e", batch_size=8,
                 d_model=3072, layer=2, row=525,
                 old_baseline=2.3849035477311644, local_files_only=True),
}

# Old locked numbers (results/e10_decoder_spectrum_*.json) for the Stage-0 comparison table.
# Llama/Mistral/Phi3: alpha=0.0 (eps=1.0) target condition; Llama also had alpha=0.5 (eps=0.5)
# recorded for its rank-1 row. Mistral/Phi3's v2 run only recorded alpha=0.5 for THEIR OWN
# rank-1 row too (same schema) -- read directly from the JSON at report time instead of
# hardcoding every old percentage here, to avoid a transcription error; this dict only pins
# baseline (needed to interpret relative_pct_change sign/magnitude at a glance while running).


def run_decoder(model_key: str, panel_key_in_e13: str, with_controls: bool) -> dict:
    spec = DECODER_PANEL[model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = dict(dtype=torch.float32, local_files_only=spec.get("local_files_only", False))
    tokkw = dict(local_files_only=spec.get("local_files_only", False))
    if spec["revision"]:
        kw["revision"] = spec["revision"]
        tokkw["revision"] = spec["revision"]

    print(f"[{model_key}] loading {spec['repo']} (float32) ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(spec["repo"], **tokkw)
    model = AutoModelForCausalLM.from_pretrained(spec["repo"], **kw)
    model.eval().to(device)
    resolved = getattr(model.config, "_commit_hash", None)
    print(f"[{model_key}]  loaded in {time.time()-t0:.0f}s  resolved_revision={resolved}")

    windows = L.build_windows(tok, DECODER_N_WINDOWS, DECODER_MAX_TOKENS)
    batches = L.batch_windows(windows, spec["batch_size"])
    print(f"[{model_key}]  {len(windows)} windows -> {len(batches)} batches "
          f"(batch_size={spec['batch_size']})")

    layer, row = spec["layer"], spec["row"]
    coords = [(layer, row)]
    conds = [dict(kind="candidate", layer=layer, row=row, epsilon=eps) for eps in EPSILONS]
    if with_controls:
        crows = new_control_rows(panel_key_in_e13, layer, spec["d_model"], exclude={row})
        print(f"[{model_key}]  new spawn(23) control rows @ L{layer}: {crows}")
        for cr in crows:
            for eps in EPSILONS:
                conds.append(dict(kind="control", layer=layer, row=cr, epsilon=eps))
    else:
        crows = new_control_rows(panel_key_in_e13, layer, spec["d_model"], exclude={row})
        print(f"[{model_key}]  new spawn(23) control rows @ L{layer} (recorded, NOT run "
              f"this pass): {crows}")

    base_pb = []
    cond_pb = [[] for _ in conds]
    t0 = time.time()
    for bi, ids in enumerate(batches):
        ids = ids.to(device)
        s, n, base_lp, _, _ = L.causal_lm_batch(model, ids, None)
        base_pb.append((s, n))
        for ci, c in enumerate(conds):
            alpha = L.alphas_for_mask([1], c["epsilon"])[0]
            with L.masked(model, DECODER_PATTERN, [(c["layer"], c["row"])], [alpha]):
                s2, n2, _, _, _ = L.causal_lm_batch(model, ids, None)
            cond_pb[ci].append((s2, n2))
        del base_lp
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[{model_key}]  batch {bi+1}/{len(batches)} ({time.time()-t0:.0f}s)", flush=True)

    base_nll = L.weighted_mean(base_pb)
    results = []
    for ci, c in enumerate(conds):
        nll = L.weighted_mean(cond_pb[ci])
        results.append({**c, "nll": nll,
                        "relative_loss_change": (nll - base_nll) / base_nll,
                        "per_batch": cond_pb[ci]})

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(model=model_key, repo=spec["repo"], resolved_revision=resolved,
                domain="text", architecture="decoder", endpoint="causal_lm_nll",
                baseline_loss=base_nll, baseline_per_batch=base_pb,
                candidate_layer=layer, candidate_row=row,
                old_baseline_loss=spec["old_baseline"],
                new_control_rows=crows, control_layer=layer,
                conditions=results)


# ── Encoder: ModernBERT-base ──────────────────────────────────────────────────

ENCODER_N_WINDOWS = 256
ENCODER_MAX_LEN = 512
ENCODER_BATCH_SIZE = 16
ENCODER_MASK_PROB = 0.15

MODERNBERT_SPEC = dict(
    repo="answerdotai/ModernBERT-base", revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
    pattern="model.layers.{i}.mlp.Wo", d_model=768, layer=15, row=251,
)


def run_modernbert(with_controls: bool, local_files_only: bool = False) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    spec = MODERNBERT_SPEC
    print("[modernbert-base] loading ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        spec["repo"], revision=spec["revision"], local_files_only=local_files_only
    )
    model = AutoModelForMaskedLM.from_pretrained(spec["repo"], dtype=torch.float32,
                                                  revision=spec["revision"],
                                                  local_files_only=local_files_only)
    model.eval().to(device)
    resolved = getattr(model.config, "_commit_hash", None)
    print(f"[modernbert-base]  loaded in {time.time()-t0:.0f}s resolved_revision={resolved}")

    windows = L.build_windows(tok, ENCODER_N_WINDOWS, ENCODER_MAX_LEN)
    batches = L.build_fixed_mlm_batches(tok, windows, device, ENCODER_BATCH_SIZE,
                                        mask_prob=ENCODER_MASK_PROB, seed=L.SEED)
    print(f"[modernbert-base]  {len(windows)} windows -> {len(batches)} batches")

    layer, row = spec["layer"], spec["row"]
    conds = [dict(kind="candidate", layer=layer, row=row, epsilon=eps) for eps in EPSILONS]
    crows = new_control_rows("ModernBERT-base", layer, spec["d_model"], exclude={row})
    print(f"[modernbert-base]  new spawn(23) control rows @ L{layer}: {crows} "
          f"({'RUN' if with_controls else 'recorded only'})")
    if with_controls:
        for cr in crows:
            for eps in EPSILONS:
                conds.append(dict(kind="control", layer=layer, row=cr, epsilon=eps))

    base_pb = []
    cond_pb = [[] for _ in conds]
    t0 = time.time()
    for bi, b in enumerate(batches):
        s, n = L.mlm_loss_per_batch(model, [b])[0]
        base_pb.append((s, n))
        for ci, c in enumerate(conds):
            alpha = L.alphas_for_mask([1], c["epsilon"])[0]
            with L.masked(model, spec["pattern"], [(c["layer"], c["row"])], [alpha]):
                s2, n2 = L.mlm_loss_per_batch(model, [b])[0]
            cond_pb[ci].append((s2, n2))
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[modernbert-base]  batch {bi+1}/{len(batches)} ({time.time()-t0:.0f}s)",
              flush=True)

    base_loss = L.weighted_mean(base_pb)
    results = []
    for ci, c in enumerate(conds):
        loss = L.weighted_mean(cond_pb[ci])
        results.append({**c, "loss": loss,
                        "relative_loss_change": (loss - base_loss) / base_loss,
                        "per_batch": cond_pb[ci]})

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return dict(model="modernbert-base", repo=spec["repo"], resolved_revision=resolved,
                domain="text", architecture="encoder", endpoint="mlm_loss",
                baseline_loss=base_loss, baseline_per_batch=base_pb,
                candidate_layer=layer, candidate_row=row,
                old_reference="results/e10_encoder_fit_results.json (key 'modernbert')",
                new_control_rows=crows, control_layer=layer,
                conditions=results)


# ── DNABERT-2 ─────────────────────────────────────────────────────────────────

DNABERT2_N_WINDOWS = 256
DNABERT2_WIN_BP = 600
DNABERT2_MAX_LEN = 256
DNABERT2_BATCH_SIZE = 16
DNABERT2_MASK_PROB = 0.15
DNABERT2_LAYER, DNABERT2_ROW = 5, 603
DNABERT2_D_MODEL = 768


def run_dnabert2(with_controls: bool) -> dict:
    import random
    t0 = time.time()
    # tomography_lib.load_dnabert2_pretrained hits two environment-version bugs under this
    # session's torch/transformers pair (meta-device alibi construction; torch.load version
    # gate for the legacy .bin checkpoint) -- see dnabert2_compat.py for full diagnosis.
    # The shim reproduces the wrapper's own config fix plus two narrowly-scoped compat
    # patches; not a reimplementation of the loading logic itself.
    model, tok, cfg, patched = load_dnabert2_patched(local_files_only=True)
    pattern = cfg["down_proj_pattern"]
    print(f"[dnabert2] loaded in {time.time()-t0:.1f}s triton_patched={patched}")

    rng = random.Random(42)
    seqs = tl.read_fasta_windows_mlm(tl.HG38_FASTA, tl.HG38_BED,
                                      n_windows=DNABERT2_N_WINDOWS, win_bp=DNABERT2_WIN_BP,
                                      rng=rng)
    batches = tl.build_fixed_batches(tok, seqs, "cuda", max_len=DNABERT2_MAX_LEN,
                                     batch_size=DNABERT2_BATCH_SIZE,
                                     mask_prob=DNABERT2_MASK_PROB, seed=42)
    print(f"[dnabert2]  {len(seqs)} windows -> {len(batches)} batches")

    layer, row = DNABERT2_LAYER, DNABERT2_ROW
    baseline_per_batch = tl.mlm_loss_per_batch(model, batches)
    base_loss = tl.aggregate_per_batch(baseline_per_batch)
    print(f"[dnabert2]  baseline MLM loss = {base_loss:.6f}")

    crows = new_control_rows("DNABERT-2", layer, DNABERT2_D_MODEL, exclude={row})
    print(f"[dnabert2]  new spawn(23) control rows @ L{layer}: {crows} "
          f"({'RUN' if with_controls else 'recorded only'})")

    conds = [dict(kind="candidate", layer=layer, row=row, epsilon=eps) for eps in EPSILONS]
    if with_controls:
        for cr in crows:
            for eps in EPSILONS:
                conds.append(dict(kind="control", layer=layer, row=cr, epsilon=eps))

    results = []
    for c in conds:
        pb = tl.dnabert2_response_per_batch(model, pattern, [(c["layer"], c["row"])], [1],
                                             c["epsilon"], batches)
        loss = tl.aggregate_per_batch(pb)
        results.append({**c, "loss": loss,
                        "relative_loss_change": (loss - base_loss) / base_loss,
                        "per_batch": pb})

    del model
    torch.cuda.empty_cache()

    return dict(model="dnabert2", repo="zhihan1996/DNABERT-2-117M",
                resolved_revision="7bce263b15377fc15361f52cfab88f8b586abda0",
                domain="genomic", architecture="encoder", endpoint="mlm_loss",
                baseline_loss=base_loss, baseline_per_batch=baseline_per_batch,
                candidate_layer=layer, candidate_row=row,
                old_reference="E9_mechanistic_tomography/RESULTS.md + "
                               "dnabert2_mask_responses.json",
                new_control_rows=crows, control_layer=layer,
                conditions=results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                     choices=["llama", "mistral", "olmo", "phi3", "modernbert-base", "dnabert2"])
    ap.add_argument("--with-controls", action="store_true",
                     help="also run the 5 new spawn(23) control rows at both epsilons "
                          "(more GPU time; candidate-only is the binding Stage-0 requirement)")
    args = ap.parse_args()

    print(f"Verifying prereg lock ({PREREG.name}) before any forward pass ...")
    verify_lock()

    if args.model in DECODER_PANEL:
        panel_key_map = {"llama": "Llama-7B", "mistral": "Mistral-7B",
                          "olmo": "OLMo-7B-0724-hf", "phi3": "Phi-3-mini-4k-instruct"}
        out = run_decoder(args.model, panel_key_map[args.model], args.with_controls)
    elif args.model == "modernbert-base":
        out = run_modernbert(args.with_controls, local_files_only=True)
    elif args.model == "dnabert2":
        out = run_dnabert2(args.with_controls)
    else:
        raise ValueError(args.model)

    existing = {}
    if RAW_PATH.exists():
        existing = json.loads(RAW_PATH.read_text())
    existing[args.model] = out
    RAW_PATH.write_text(json.dumps(existing, indent=2))
    print(f"checkpointed -> {RAW_PATH}")

    print(f"\n=== {args.model}: baseline={out['baseline_loss']:.6f} ===")
    for c in out["conditions"]:
        if c["kind"] != "candidate":
            continue
        print(f"  L{c['layer']}/r{c['row']} eps={c['epsilon']}: "
              f"relative_loss_change={c['relative_loss_change']:+.4%}")


if __name__ == "__main__":
    main()
