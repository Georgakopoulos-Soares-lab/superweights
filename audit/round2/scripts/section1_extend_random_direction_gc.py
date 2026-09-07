#!/usr/bin/env python
"""Round-2 Section 1 compute: extend GC measurement on the random-direction grid to
c in {0.5, 1.0, 3.0, 8.0}, 96 prompts x 3 seeds each, exactly reusing the existing E12
harness (e12_lib.generator_response_unified via run_e12_full.run_gen_condition's own
call pattern) so results are directly comparable to the existing c=0.0/c=0.0125 points.

Reuses the cached corpora (results/E12/raw/corpora.json) and reconstructs unit_dir with
the exact same seed convention as run_e12_full.py's main() (base_seed=20260823, keyed by
GENERATOR_ROW_PRIMARY=2371) -- confirmed identical by construction, not just by hope,
since make_unit_random_direction is a pure deterministic function of
(intermediate_size, row, dtype, device, base_seed) and all four inputs are the same here.

Writes NEW checkpoint file (does not touch the original gen_records.jsonl):
  audit/round2/raw/gc_extension_records.jsonl  -- full per-prompt records (gc, entropy,
    self_nll, raw/filtered sequence, etc.) for every (c, seed, prompt) triple.
Also writes audit/round2/generator_random_direction_gc_extension.csv (aggregated).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
E12_DIR = ROOT / "paper-salvage/experiments/E12_generator_degradation_control"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(E12_DIR))
import e12_lib as el  # noqa: E402

OUT_DIR = ROOT / "audit/round2/raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GEN_CKPT = OUT_DIR / "gc_extension_records.jsonl"
SUMMARY_CSV = ROOT / "audit/round2/generator_random_direction_gc_extension.csv"

GENERATOR_LAYER = el.GENERATOR_LAYER
ROW_PRIMARY = el.GENERATOR_ROW_PRIMARY
MAX_NEW = 64
PROMPT_LEN = 120
SEEDS = [42, 43, 44]
C_VALUES = [0.5, 1.0, 3.0, 8.0]

CORPUS_CACHE = ROOT / "results/E12/raw/corpora.json"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_model():
    log("loading GENERator EUK 3B ...")
    t0 = time.time()
    from models import WRAPPER_MAP
    cfg = yaml.safe_load((ROOT / "configs/generator.yaml").read_text())
    w = WRAPPER_MAP["generator"](cfg)
    w.load()
    model = w.model.eval()
    tok = w.tokenizer
    pattern = cfg["down_proj_pattern"]
    log(f"model loaded in {time.time()-t0:.1f}s")
    return model, tok, pattern


def load_checkpoint():
    done = set()
    if GEN_CKPT.exists():
        for line in open(GEN_CKPT):
            r = json.loads(line)
            done.add((r["c"], r["seed"]))
    return done


def main():
    d = json.loads(CORPUS_CACHE.read_text())
    prompt_wins = d["prompt_windows"]
    prompts = [s[:PROMPT_LEN] for _c, _st, s in prompt_wins]
    log(f"loaded {len(prompts)} cached prompts from {CORPUS_CACHE}")

    model, tok, pattern = load_model()
    dev = next(model.parameters()).device
    mods = dict(model.named_modules())
    dp_module = mods[pattern.format(i=GENERATOR_LAYER)]
    nrows, intermediate_size = dp_module.weight.data.shape
    unit_dir = el.make_unit_random_direction(intermediate_size, ROW_PRIMARY,
                                              dp_module.weight.dtype, dev)
    log(f"reconstructed unit_dir: intermediate_size={intermediate_size}, "
        f"row={ROW_PRIMARY}, base_seed=20260823 (e12_lib default, unchanged)")

    done = load_checkpoint()
    plan = [(c, s) for c in C_VALUES for s in SEEDS if (c, s) not in done]
    log(f"plan: {len(plan)} (c, seed) conditions to run "
        f"({len(C_VALUES)*len(SEEDS) - len(plan)} already checkpointed)")

    for c, seed in plan:
        t0 = time.time()
        records = el.generator_response_unified(
            model, tok, pattern, GENERATOR_LAYER, ROW_PRIMARY, "direction", c,
            prompts, max_new=MAX_NEW, seed=seed, do_sample=True, unit_dir=unit_dir)
        self_nll = el.self_nll_batch(model, tok, pattern, GENERATOR_LAYER, ROW_PRIMARY,
                                      "direction", c, records, unit_dir=unit_dir)
        gcs = []
        with open(GEN_CKPT, "a") as f:
            for r, snll in zip(records, self_nll):
                r["c"] = c
                r["seed"] = seed
                r["self_nll"] = snll
                f.write(json.dumps(r) + "\n")
                if not np.isnan(r["gc"]):
                    gcs.append(r["gc"])
        dt = time.time() - t0
        log(f"  c={c} seed={seed}: n_prompts={len(records)} n_valid_gc={len(gcs)} "
            f"mean_gc={np.mean(gcs):.4f} done in {dt:.1f}s")

    # aggregate
    all_records = [json.loads(l) for l in open(GEN_CKPT)] if GEN_CKPT.exists() else []
    import csv
    rows = []
    for c in C_VALUES:
        recs = [r for r in all_records if r["c"] == c]
        gcs = [r["gc"] for r in recs if not np.isnan(r["gc"])]
        seeds_present = sorted(set(r["seed"] for r in recs))
        if gcs:
            rng = np.random.default_rng(42)
            arr = np.array(gcs)
            boot = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(5000)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        rows.append({"c": c, "n_prompts": len(gcs), "n_seeds": len(seeds_present),
                      "gc_mean": np.mean(gcs) if gcs else float("nan"),
                      "gc_ci_low": lo, "gc_ci_high": hi,
                      "gc_std": np.std(gcs) if gcs else float("nan"),
                      "gc_min": np.min(gcs) if gcs else float("nan"),
                      "gc_max": np.max(gcs) if gcs else float("nan")})
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"wrote {SUMMARY_CSV}")
    for r in rows:
        log(f"  c={r['c']}: mean_gc={r['gc_mean']:.4f} CI=[{r['gc_ci_low']:.4f},{r['gc_ci_high']:.4f}] "
            f"min={r['gc_min']:.4f} max={r['gc_max']:.4f} n={r['n_prompts']}")
    log("=== COMPLETE ===")


if __name__ == "__main__":
    main()
