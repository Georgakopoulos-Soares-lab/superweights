#!/usr/bin/env python3
"""
EXPERIMENT 10 — what KIND of interaction couples DNABERT-2 L9/r264 and L9/r294?

Established: singleton effects ~0, joint effect ~-33.8 pp. This asks whether that is
compensatory redundancy, redundant pathways, or downstream convergence, using:

  structural : cosine similarity of the two down_proj row vectors
  activation : correlation of their per-token and per-window activations
  cross-ablation: does removing one CHANGE the other's activation?
                  increase -> compensation;  unchanged -> independent contributions

No mechanism label is assigned unless the data support it.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path("/home/nvidia/superweights")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from run_gue_multiseed import _load_model
from run_gue_ablation import (GUEDataset, _resolve_module, _save_row, _zero_row,
                              _restore_row, evaluate, collate_fn, _MAX_LEN, _task_key)

OUT = ROOT / "results/mechanism_generator"; OUT.mkdir(parents=True, exist_ok=True)
GUE = "/data/nvidia/data/gue/GUE"; TASK = "splice/reconstructed"
CKPT = ROOT / "results/gue_checkpoints_multiseed/dnabert2_reconstructed/seed_0/model_state.pt"
LAYER, A, B = 9, 264, 294
N_BATCH = 8


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    cfg = yaml.safe_load((ROOT / "configs/dnabert2.yaml").read_text())
    pat = cfg["down_proj_pattern"]; ml = _MAX_LEN.get(_task_key(TASK), 512)
    model, tok = _load_model(cfg["model_id"], 3, None, "cuda")
    tok.model_max_length = ml
    ds = GUEDataset(f"{GUE}/{TASK}/test.csv", tok, ml)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False); model.eval()
    mod = _resolve_module(model, pat, LAYER)
    W = mod.weight.data.float()

    # ---- structural alignment -------------------------------------------------
    wa, wb = W[A], W[B]
    cos = float(torch.nn.functional.cosine_similarity(wa[None], wb[None]).item())
    rng = np.random.default_rng(42)
    pool = [r for r in range(W.shape[0]) if r not in (A, B)]
    rc = [float(torch.nn.functional.cosine_similarity(W[int(i)][None], W[int(j)][None]).item())
          for i, j in rng.choice(pool, size=(500, 2))]
    log(f"cosine(r{A},r{B}) = {cos:+.4f}   random same-layer pairs: "
        f"mean {np.mean(rc):+.4f} sd {np.std(rc):.4f}  |z| = {abs(cos-np.mean(rc))/np.std(rc):.2f}")

    # ---- activations under cross-ablation -------------------------------------
    loader = torch.utils.data.DataLoader(ds, batch_size=32, collate_fn=collate_fn)
    batches = [b for i, b in enumerate(loader) if i < N_BATCH]
    store = {}
    hh = mod.register_forward_hook(
        lambda _m, _i, o, _s=store: _s.__setitem__(
            "h", (o[0] if isinstance(o, tuple) else o).detach().float()))

    def acts(zero_rows):
        sv = [(r, mod.weight.data[r].clone()) for r in zero_rows]
        for r in zero_rows: mod.weight.data[r].zero_()
        va, vb, per_tok = [], [], []
        with torch.no_grad():
            for b in batches:
                model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda())
                h = store["h"]; h = h if h.dim() == 2 else h.reshape(-1, h.shape[-1])
                va.append(h[:, A].abs().cpu().numpy()); vb.append(h[:, B].abs().cpu().numpy())
        for r, w in sv: mod.weight.data[r] = w
        a = np.concatenate(va); b_ = np.concatenate(vb)
        return a, b_

    a0, b0 = acts([])
    aB, bB = acts([B])          # ablate r294 -> what happens to r264?
    aA, bA = acts([A])          # ablate r264 -> what happens to r294?
    hh.remove()

    r_tok = float(np.corrcoef(a0, b0)[0, 1])
    log(f"per-token activation correlation r(r{A}, r{B}) = {r_tok:+.4f}  (n={len(a0)})")
    dA = float(aB.mean() / max(a0.mean(), 1e-12))   # r264 after ablating r294
    dB = float(bA.mean() / max(b0.mean(), 1e-12))   # r294 after ablating r264
    log(f"r{A} activation x{dA:.4f} after ablating r{B}")
    log(f"r{B} activation x{dB:.4f} after ablating r{A}")

    # ---- task endpoint for reference ------------------------------------------
    base = evaluate(model, ds, device="cuda")["accuracy"]
    def acc(rows):
        sv = [(r, _save_row(model, pat, LAYER, r)) for r in rows]
        for r in rows: _zero_row(model, pat, LAYER, r)
        v = evaluate(model, ds, device="cuda")["accuracy"]
        for l, s in zip(rows, [x[1] for x in sv]): _restore_row(model, pat, LAYER, l, s)
        return v
    accA, accB, accAB = acc([A]), acc([B]), acc([A, B])
    log(f"acc base={base:.4f} -A={accA:.4f} -B={accB:.4f} -AB={accAB:.4f}")

    summ = dict(cosine_similarity=cos, cosine_random_mean=float(np.mean(rc)),
                cosine_random_sd=float(np.std(rc)),
                cosine_z=float(abs(cos - np.mean(rc)) / np.std(rc)),
                activation_corr_per_token=r_tok, n_tokens=int(len(a0)),
                act_a_intact=float(a0.mean()), act_b_intact=float(b0.mean()),
                act_a_after_ablating_b=float(aB.mean()), act_b_after_ablating_a=float(bA.mean()),
                ratio_a_after_ablating_b=dA, ratio_b_after_ablating_a=dB,
                acc_baseline=base, acc_ablate_a=accA, acc_ablate_b=accB, acc_ablate_both=accAB,
                epistasis_pp=float((accAB - base) * 100 - ((accA - base) * 100 + (accB - base) * 100)),
                compensation_detected=bool(dA > 1.10 or dB > 1.10))
    json.dump(dict(experiment="EXP10_dnabert_pair_mechanism", model="DNABERT-2-117M",
                   layer=LAYER, rows=[A, B], task=TASK, summary=summ,
                   elapsed_seconds=time.time() - t0),
              open(OUT / "dnabert_pair_mechanism.json", "w"), indent=2)
    with open(OUT / "dnabert_pair_mechanism.tsv", "w") as f:
        f.write("metric\tvalue\n")
        for k, v in summ.items(): f.write(f"{k}\t{v}\n")
    log(f"compensation_detected={summ['compensation_detected']}  saved")


if __name__ == "__main__":
    main()
