#!/usr/bin/env python3
"""Priority-3 entry point: provenance check -> smoke tests -> (if all pass) full run.

Per explicit instruction: if the checkpoint state does not reproduce the historical E12
intact/full-ablation native-NLL values within tolerance, STOP and report why (no further
Priority-3 work). If it reproduces and every smoke/invariance test passes, proceed straight
to the full 6-condition experiment without stopping for permission again (estimated cost
~30-45 minutes was already approved).
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import run_bos_mediation as bm  # noqa: E402
import e12_lib as e12  # noqa: E402

OUT = bm.OUT_DIR
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run_log.txt"


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    t0 = time.time()
    log("=== Priority 3: GENERator-EUK L4/r2371 BOS causal mediation ===")
    log(f"Loading model (revision pinned to {bm.REVISION})...")
    model, tok, resolved = bm.load_model()
    log(f"Loaded. resolved_commit_hash={resolved} load_seconds={time.time()-t0:.1f}")

    prompt_windows, damage_windows, corpus_meta = e12.build_corpora(
        seed=42, n_prompt=96, prompt_win_bp=170, n_damage=100, damage_win_bp=512)
    log(f"Corpora built: {corpus_meta}")

    # ── STEP 1: checkpoint / provenance reproduction check ──────────────────────
    log("--- Provenance check: reproduce historical E12 intact + full-ablation damage ---")
    intact_damage = bm.damage(model, tok, damage_windows, "A_intact")
    ablated_damage = bm.damage(model, tok, damage_windows, "B_full_ablation_weight")
    log(f"  measured intact damage  = {intact_damage:.6f}  (historical {bm.HIST_DAMAGE_INTACT:.6f})")
    log(f"  measured ablated damage = {ablated_damage:.6f}  (historical {bm.HIST_DAMAGE_ABLATED:.6f})")

    def close(a, b, rtol=bm.REPRO_RTOL):
        return abs(a - b) <= rtol * abs(b)

    intact_ok = close(intact_damage, bm.HIST_DAMAGE_INTACT)
    ablated_ok = close(ablated_damage, bm.HIST_DAMAGE_ABLATED)
    repro = {"intact_measured": intact_damage, "intact_historical": bm.HIST_DAMAGE_INTACT,
             "intact_ok": intact_ok, "ablated_measured": ablated_damage,
             "ablated_historical": bm.HIST_DAMAGE_ABLATED, "ablated_ok": ablated_ok,
             "rtol": bm.REPRO_RTOL, "resolved_commit_hash": resolved}
    (OUT / "provenance_check.json").write_text(json.dumps(repro, indent=2))

    if not (intact_ok and ablated_ok):
        log("PROVENANCE CHECK FAILED. STOPPING Priority 3 -- reporting why, no further work.")
        log(json.dumps(repro, indent=2))
        return

    log("Provenance check PASSED -- checkpoint state reproduces historical E12 damage "
        "values within tolerance. Proceeding to smoke tests.")

    # ── STEP 2: smoke / invariance tests (small subset for speed) ──────────────
    smoke_windows = damage_windows[:6]
    smoke = {}
    all_pass = True

    # 1. no-op hook == intact
    class _NoOp:
        def __call__(self, _m, _i, output):
            return output
    h = bm.register(model, _NoOp())
    try:
        noop_loss = bm.damage(model, tok, smoke_windows, "A_intact")
    finally:
        h.remove()
    intact_loss_ref = bm.damage(model, tok, smoke_windows, "A_intact")
    ok1 = abs(noop_loss - intact_loss_ref) < 1e-6
    smoke["1_noop_hook_eq_intact"] = {"noop": noop_loss, "intact": intact_loss_ref, "pass": ok1}
    all_pass &= ok1

    # 2. hook-based full ablation == weight-based full ablation
    hook_ablated = bm.damage(model, tok, smoke_windows, "B_full_ablation_hook")
    weight_ablated = bm.damage(model, tok, smoke_windows, "B_full_ablation_weight")
    ok2 = abs(hook_ablated - weight_ablated) < 1e-4
    smoke["2_hook_ablation_eq_weight_ablation"] = {"hook": hook_ablated, "weight": weight_ablated, "pass": ok2}
    all_pass &= ok2

    # 3. restore-everywhere == intact
    restore_all = bm.damage(model, tok, smoke_windows, "SMOKE_restore_everywhere")
    intact_ref2 = bm.damage(model, tok, smoke_windows, "A_intact")
    ok3 = abs(restore_all - intact_ref2) < 1e-4
    smoke["3_restore_everywhere_eq_intact"] = {"restore_all": restore_all, "intact": intact_ref2, "pass": ok3}
    all_pass &= ok3

    # 4. BOS-only intervention (condition C) changes ONLY position 0 of the immediate
    #    down_proj output tensor, vs. the intact model's own output tensor for the SAME window.
    seq = smoke_windows[0][2]
    ids = bm._tokenize(tok, seq)
    intact_cap: list[torch.Tensor] = []
    hI = bm.register(model, bm.PositionRowHook(bm.ROW, capture=intact_cap))
    with torch.no_grad():
        model(input_ids=ids)
    hI.remove()
    c_cap: list[torch.Tensor] = []
    hC = bm.register(model, bm.PositionRowHook(
        bm.ROW, zero_fn=lambda L, pf: [i == 0 for i in range(L)], capture=c_cap))
    with torch.no_grad():
        model(input_ids=ids)
    hC.remove()
    intact_vec = intact_cap[0][0]
    c_vec = c_cap[0][0]
    diff_mask = (intact_vec - c_vec).abs() > 1e-9
    ok4 = bool(diff_mask[0].item()) and not bool(diff_mask[1:].any().item())
    smoke["4_bos_only_changes_only_position0"] = {
        "differs_at_pos0": bool(diff_mask[0].item()),
        "differs_elsewhere_count": int(diff_mask[1:].sum().item()),
        "seq_len": int(diff_mask.shape[0]), "pass": ok4}
    all_pass &= ok4

    # 5. matched non-BOS restoration (condition F) changes ONLY its target position, vs.
    #    the fully-ablated-everywhere baseline (condition B).
    b_cap: list[torch.Tensor] = []
    saved = e12.set_row_alpha(model, bm.PATTERN, bm.LAYER, bm.ROW, 0.0)
    hB = bm.register(model, bm.PositionRowHook(bm.ROW, capture=b_cap))
    with torch.no_grad():
        model(input_ids=ids)
    hB.remove()
    e12.restore_row(model, bm.PATTERN, bm.LAYER, bm.ROW, saved)
    target_pos = ids.shape[1] // 2
    f_cap: list[torch.Tensor] = []
    saved2 = e12.set_row_alpha(model, bm.PATTERN, bm.LAYER, bm.ROW, 0.0)
    hF = bm.register(model, bm.PositionRowHook(
        bm.ROW, inject_fn=lambda L, pf, tp=target_pos, iv=intact_vec: ([tp], iv[tp:tp + 1]),
        capture=f_cap))
    with torch.no_grad():
        model(input_ids=ids)
    hF.remove()
    e12.restore_row(model, bm.PATTERN, bm.LAYER, bm.ROW, saved2)
    b_vec, f_vec = b_cap[0][0], f_cap[0][0]
    diff_mask5 = (b_vec - f_vec).abs() > 1e-9
    ok5 = bool(diff_mask5[target_pos].item()) and int(diff_mask5.sum().item()) == 1
    smoke["5_matched_nonbos_changes_only_target"] = {
        "target_pos": target_pos, "differs_at_target": bool(diff_mask5[target_pos].item()),
        "total_positions_differing": int(diff_mask5.sum().item()), "pass": ok5}
    all_pass &= ok5

    # 6. generation implementation correctly distinguishes prefill (L>1, position 0 exists)
    #    from KV-cache decode (L==1, position 0 never recurs).
    prompt = prompt_windows[0][2][:120]
    prep = e12._prepare_sequence(tok, prompt)
    enc = tok(prep, return_tensors="pt", add_special_tokens=False).to("cuda")
    prompt_ids = enc["input_ids"]
    log_calls = []

    class _LogHook(bm.PositionRowHook):
        def __call__(self, module, inputs, output):
            y = output[0] if isinstance(output, tuple) else output
            log_calls.append({"L": int(y.shape[1]), "is_prefill": y.shape[1] > 1})
            return super().__call__(module, inputs, output)

    hookC = _LogHook(bm.ROW, zero_fn=lambda L, pf: ([True] + [False] * (L - 1)) if pf else [False] * L)
    hC6 = bm.register(model, hookC)
    with torch.no_grad():
        torch.manual_seed(0)
        _ = model.generate(input_ids=prompt_ids, max_new_tokens=5, do_sample=False,
                            pad_token_id=getattr(tok, "pad_token_id", None) or 0)
    hC6.remove()
    n_prefill = sum(1 for c in log_calls if c["is_prefill"])
    n_decode = sum(1 for c in log_calls if not c["is_prefill"])
    zeroed_at_prefill = hookC.last_zeroed_positions[0] if hookC.last_zeroed_positions else None
    zeroed_at_decode = hookC.last_zeroed_positions[1:] if len(hookC.last_zeroed_positions) > 1 else []
    ok6 = (n_prefill == 1 and n_decode >= 1 and zeroed_at_prefill == [0]
           and all(z == [] for z in zeroed_at_decode))
    smoke["6_prefill_vs_decode_handling"] = {
        "n_prefill_calls": n_prefill, "n_decode_calls": n_decode,
        "zeroed_positions_at_prefill": zeroed_at_prefill,
        "zeroed_positions_at_decode_calls": zeroed_at_decode, "pass": ok6}
    all_pass &= ok6

    (OUT / "smoke_tests.json").write_text(json.dumps(smoke, indent=2, default=str))
    log("Smoke test results:\n" + json.dumps(smoke, indent=2, default=str))

    if not all_pass:
        log("SMOKE TESTS FAILED. STOPPING Priority 3 before the full run.")
        return
    log("ALL SMOKE TESTS PASSED. Proceeding immediately to the full experiment "
        "(no further permission stop, per instruction).")

    # ── STEP 3: full experiment ─────────────────────────────────────────────────
    conditions = ["A_intact", "B_full_ablation_weight", "C_ablate_bos_only",
                  "D_ablate_all_except_bos", "E_restore_at_bos", "F_restore_at_matched_nonbos"]
    intact_cache: dict = {}
    per_window = {}
    for cond in conditions:
        t1 = time.time()
        vals = bm.damage_per_window(model, tok, damage_windows, cond, intact_cache)
        per_window[cond] = vals
        log(f"  damage[{cond}] pooled_mean={np.nanmean(vals):.6f} "
            f"elapsed={time.time()-t1:.1f}s")

    nll_results = {}
    pairs = [("C_ablate_bos_only", "A_intact"), ("D_ablate_all_except_bos", "B_full_ablation_weight"),
             ("D_ablate_all_except_bos", "A_intact"), ("E_restore_at_bos", "B_full_ablation_weight"),
             ("F_restore_at_matched_nonbos", "E_restore_at_bos")]
    for cond, ref in pairs:
        nll_results[f"{cond}_vs_{ref}"] = bm.paired_bootstrap_diff(
            per_window[cond], per_window[ref], seed=hash((cond, ref)) % (2**31))

    rescue = {}
    for cond in ["C_ablate_bos_only", "D_ablate_all_except_bos", "E_restore_at_bos",
                 "F_restore_at_matched_nonbos"]:
        rescue[cond] = bm.rescue_fraction_bootstrap(
            per_window[cond], per_window["B_full_ablation_weight"], per_window["A_intact"],
            seed=hash(("rescue", cond)) % (2**31))

    log("Rescue fractions: " + json.dumps(rescue, indent=2))

    # ── STEP 4: secondary GC endpoint (subset of prompts, matching E9/E12 hyperparams) ──
    gc_prompts = [seq[:120] for _, _, seq in prompt_windows[:24]]
    gc_per_condition = {}
    for cond in conditions:
        t1 = time.time()
        recs = bm.generate_for_condition(model, tok, gc_prompts, cond, max_new=64,
                                          seed=e12.BASE_SEED)
        gcs = [r["gc"] for r in recs]
        gc_per_condition[cond] = gcs
        log(f"  GC[{cond}] mean={np.nanmean(gcs):.4f} elapsed={time.time()-t1:.1f}s")

    gc_bootstrap = {}
    for cond, ref in pairs:
        gc_bootstrap[f"{cond}_vs_{ref}"] = bm.paired_bootstrap_diff(
            gc_per_condition[cond], gc_per_condition[ref], seed=hash(("gc", cond, ref)) % (2**31))
    gc_rescue = {}
    for cond in ["C_ablate_bos_only", "D_ablate_all_except_bos", "E_restore_at_bos",
                 "F_restore_at_matched_nonbos"]:
        gc_rescue[cond] = bm.rescue_fraction_bootstrap(
            gc_per_condition[cond], gc_per_condition["B_full_ablation_weight"],
            gc_per_condition["A_intact"], seed=hash(("gc_rescue", cond)) % (2**31))

    payload = {
        "revision": bm.REVISION, "resolved_commit_hash": resolved,
        "corpus_meta": corpus_meta, "provenance_check": repro, "smoke_tests": smoke,
        "n_damage_windows": len(damage_windows), "n_gc_prompts": len(gc_prompts),
        "nll_pooled_mean_by_condition": {c: float(np.nanmean(v)) for c, v in per_window.items()},
        "nll_paired_bootstrap": nll_results, "nll_rescue_fraction": rescue,
        "gc_mean_by_condition": {c: float(np.nanmean(v)) for c, v in gc_per_condition.items()},
        "gc_paired_bootstrap": gc_bootstrap, "gc_rescue_fraction": gc_rescue,
        "elapsed_seconds": time.time() - t0,
    }
    out_path = OUT / "bos_mediation_results.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, out_path)
    log(f"WROTE {out_path} total_elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
