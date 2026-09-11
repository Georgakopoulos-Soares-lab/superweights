#!/usr/bin/env python3
"""
E12 -- degradation-matched control for the GENERator GC result. ONE self-contained,
resume-safe driver: loads the model once, builds both corpora, runs B1/B2 (extended
dose-response + quality metrics), B3 (damage matching for the 5 control rows + the
random-direction control, then full generation+quality at each matched config), B4
(k-mer + homopolymer attribution), and writes all deliverables (3 CSVs, 2 figures,
summary.md).

Resume-safety: every unit of work (one generation condition, or one damage evaluation)
is appended to a JSONL checkpoint file immediately after it completes. On startup the
script reads whatever checkpoint lines already exist and skips recomputing them, so a
crash/restart only loses the single in-flight unit of work, never the whole run.

Protocol: experiments/docs/prereg/PREREG_E12_generator_degradation_control.md (locked).
Set E12_SMOKE_TEST=1 in the environment to run a tiny, fast end-to-end sanity pass
(separate output dir, not counted as real measurement) before launching the real run.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12_lib as el  # noqa: E402

SMOKE = os.environ.get("E12_SMOKE_TEST", "0") == "1"

RESULTS_DIR = REPO_ROOT / ("results/E12_smoke" if SMOKE else "results/experiments/E12")
FIG_DIR = RESULTS_DIR / "figures"
RAW_DIR = RESULTS_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

GEN_CKPT = RAW_DIR / "gen_records.jsonl"
DAMAGE_CKPT = RAW_DIR / "damage_evals.jsonl"
MATCH_CKPT = RAW_DIR / "damage_matching.jsonl"
CORPUS_CACHE = RAW_DIR / "corpora.json"
LOG_PATH = RESULTS_DIR / "run_e12_full.progress.log"

GENERATOR_LAYER = el.GENERATOR_LAYER
ROW_PRIMARY = el.GENERATOR_ROW_PRIMARY
ROW_SECONDARY = el.GENERATOR_ROW_SECONDARY
BASE_SEED = el.BASE_SEED

if SMOKE:
    N_PROMPT, PROMPT_BP = 6, 170
    N_DAMAGE, DAMAGE_BP = 8, 512
    MAX_NEW = 12
    SEEDS = [42, 43]
    ROW2371_GRID = [1.0, 0.0, 2.0]
    ROW1522_GRID = [1.0, 0.0]
    GREEDY_GRID = [1.0, 0.0]
    CONTROL_DAMAGE_GRID = [0.0, 1.0, 5.0]
    N_BOOT = 200
else:
    N_PROMPT, PROMPT_BP = 96, 170
    N_DAMAGE, DAMAGE_BP = 100, 512
    MAX_NEW = 64
    SEEDS = [42, 43, 44]
    ROW2371_GRID = [1.0, 0.75, 0.5, 0.25, 0.0, 1.5, 2.0, 3.0, 5.0]
    ROW1522_GRID = [1.0, 0.75, 0.5, 0.25, 0.0]
    GREEDY_GRID = [1.0, 0.75, 0.5, 0.25, 0.0]
    CONTROL_DAMAGE_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    N_BOOT = 5000

PROMPT_LEN = 120  # bp used as prompt from each 170bp window, matches existing convention


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ── JSONL checkpoint helpers ───────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # a partially-written line from a killed process -- skip it, it will be
                # recomputed since its key won't be in the completed set.
                continue
    return out


def append_jsonl(path: Path, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def with_retry(fn, desc, tries=2, sleep_s=10):
    last_exc = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            log(f"  ERROR in {desc} (attempt {attempt+1}/{tries}): {e}")
            log(traceback.format_exc())
            time.sleep(sleep_s)
    raise last_exc


# ── model + corpora ────────────────────────────────────────────────────────────

def load_model():
    log("loading GENERator EUK 3B ...")
    t0 = time.time()
    from models import WRAPPER_MAP
    cfg = yaml.safe_load((REPO_ROOT / "configs/generator.yaml").read_text())
    w = WRAPPER_MAP["generator"](cfg)
    w.load()
    model = w.model.eval()
    tok = w.tokenizer
    pattern = cfg["down_proj_pattern"]
    log(f"model loaded in {time.time()-t0:.1f}s")
    mods = dict(model.named_modules())
    dp = mods[pattern.format(i=GENERATOR_LAYER)]
    nrows, intermediate_size = dp.weight.data.shape
    log(f"down_proj shape = {tuple(dp.weight.data.shape)} "
        f"(nrows={nrows}, intermediate_size={intermediate_size})")
    return model, tok, cfg, pattern, nrows, intermediate_size


def get_corpora():
    if CORPUS_CACHE.exists():
        d = json.loads(CORPUS_CACHE.read_text())
        log(f"reusing cached corpora: {d['meta']}")
        return d["prompt_windows"], d["damage_windows"], d["meta"]
    log("building disjoint hg38 window pools ...")
    prompt_wins, damage_wins, meta = el.build_corpora(
        seed=BASE_SEED, n_prompt=N_PROMPT, prompt_win_bp=PROMPT_BP,
        n_damage=N_DAMAGE, damage_win_bp=DAMAGE_BP)
    CORPUS_CACHE.write_text(json.dumps({
        "prompt_windows": prompt_wins, "damage_windows": damage_wins, "meta": meta,
    }))
    log(f"corpora built: {meta}")
    return prompt_wins, damage_wins, meta


# ── generation work-plan execution ────────────────────────────────────────────

def gen_key(label, kind, row, value, seed, do_sample):
    return f"{label}|kind={kind}|row={row}|value={value}|seed={seed}|sample={do_sample}"


def run_gen_condition(model, tok, pattern, prompts, label, kind, row, value, seed,
                       do_sample, unit_dir=None):
    records = el.generator_response_unified(
        model, tok, pattern, GENERATOR_LAYER, row, kind, value, prompts,
        max_new=MAX_NEW, seed=seed, do_sample=do_sample, unit_dir=unit_dir)
    self_nll = el.self_nll_batch(model, tok, pattern, GENERATOR_LAYER, row, kind, value,
                                  records, unit_dir=unit_dir)
    for r, s in zip(records, self_nll):
        r["self_nll"] = s
        r.update(el.quality_metrics_for_record(r))
    return records


def execute_gen_plan(model, tok, pattern, prompts, plan, unit_dir_by_row):
    done = {c.get("key") for c in load_jsonl(GEN_CKPT)}
    log(f"gen plan: {len(plan)} conditions total, {len(done)} already checkpointed")
    for j, cond in enumerate(plan):
        key = gen_key(cond["label"], cond["kind"], cond["row"], cond["value"],
                      cond["seed"], cond["do_sample"])
        if key in done:
            continue
        t0 = time.time()
        unit_dir = unit_dir_by_row.get(cond["row"]) if cond["kind"] == "direction" else None

        def _run():
            return run_gen_condition(model, tok, pattern, prompts, cond["label"],
                                      cond["kind"], cond["row"], cond["value"],
                                      cond["seed"], cond["do_sample"], unit_dir=unit_dir)

        records = with_retry(_run, f"gen {key}")
        append_jsonl(GEN_CKPT, {"key": key, **cond, "records": records})
        done.add(key)
        log(f"  [{j+1}/{len(plan)}] {key} done in {time.time()-t0:.1f}s "
            f"(mean GC={np.nanmean([r['gc'] for r in records]):.4f})")


# ── damage matching ────────────────────────────────────────────────────────────

def damage_key(label, kind, row, value):
    return f"{label}|kind={kind}|row={row}|value={round(float(value), 6)}"


def cached_damage_fn(model, tok, pattern, row, kind, label, damage_windows, unit_dir=None):
    cache = {damage_key(c["label"], c["kind"], c["row"], c["value"]): c["damage"]
             for c in load_jsonl(DAMAGE_CKPT)}

    def fn(value):
        k = damage_key(label, kind, row, value)
        if k in cache:
            return cache[k]
        d = with_retry(
            lambda: el.damage_unified(model, tok, pattern, GENERATOR_LAYER, row, kind,
                                       value, damage_windows, unit_dir=unit_dir),
            f"damage {k}")
        append_jsonl(DAMAGE_CKPT, {"label": label, "kind": kind, "row": row,
                                    "value": value, "damage": d})
        cache[k] = d
        log(f"  damage[{label}]({value}) = {d:.6f}")
        return d
    return fn


def run_damage_matching(model, tok, pattern, damage_windows, control_rows, unit_dir,
                         intermediate_size):
    # 1. row 2371 damage across the full B2 grid (suppression+amplification)
    fn_2371 = cached_damage_fn(model, tok, pattern, ROW_PRIMARY, "alpha", "row2371",
                                damage_windows)
    for a in ROW2371_GRID:
        fn_2371(a)
    target = fn_2371(0.0)
    log(f"damage target D(2371, alpha=0) = {target:.6f}")

    matches = {c["label"]: c for c in load_jsonl(MATCH_CKPT)}

    # 2. 5 control rows
    for i, cr in enumerate(control_rows):
        label = f"control_row_{cr}"
        if label in matches:
            continue
        fn = cached_damage_fn(model, tok, pattern, cr, "alpha", label, damage_windows)
        res = el.find_matching_scale(fn, target, CONTROL_DAMAGE_GRID, max_refine=3)
        res["label"] = label
        res["kind"] = "alpha"
        res["row"] = cr
        append_jsonl(MATCH_CKPT, res)
        matches[label] = res
        log(f"matched {label}: scale={res['matched_scale']} damage={res['matched_damage']:.6f} "
            f"reachable={res['reachable']}")

    # 3. random-direction control at row 2371's own location
    label = "random_direction"
    if label not in matches:
        fn = cached_damage_fn(model, tok, pattern, ROW_PRIMARY, "direction", label,
                               damage_windows, unit_dir=unit_dir)
        res = el.find_matching_scale(fn, target, CONTROL_DAMAGE_GRID, max_refine=3)
        res["label"] = label
        res["kind"] = "direction"
        res["row"] = ROW_PRIMARY
        append_jsonl(MATCH_CKPT, res)
        matches[label] = res
        log(f"matched {label}: scale={res['matched_scale']} damage={res['matched_damage']:.6f} "
            f"reachable={res['reachable']}")

    return target, matches


# ── build the full B1/B2/B3 generation work plan ──────────────────────────────

def build_gen_plan(control_rows, matches):
    plan = []
    # B2: row 2371 dose-response (suppression + amplification), 3 seeds, sampling
    for a in ROW2371_GRID:
        for s in SEEDS:
            plan.append({"label": "row2371_grid", "kind": "alpha", "row": ROW_PRIMARY,
                         "value": a, "seed": s, "do_sample": True})
    # B2: row 1522 dose-response (suppression only), 3 seeds, sampling
    for a in ROW1522_GRID:
        for s in SEEDS:
            plan.append({"label": "row1522_grid", "kind": "alpha", "row": ROW_SECONDARY,
                         "value": a, "seed": s, "do_sample": True})
    # B2: greedy arm, row2371 + 5 control rows, suppression grid, seed 42 only
    for row, lbl in [(ROW_PRIMARY, "row2371_greedy")] + \
                    [(cr, f"control_row_{cr}_greedy") for cr in control_rows]:
        for a in GREEDY_GRID:
            plan.append({"label": lbl, "kind": "alpha", "row": row, "value": a,
                         "seed": 42, "do_sample": False})
    # B3: matched configs -- full B1/B2 protocol (3 seeds, sampling) at each matched scale
    for cr in control_rows:
        m = matches[f"control_row_{cr}"]
        for s in SEEDS:
            plan.append({"label": f"matched_control_row_{cr}", "kind": "alpha", "row": cr,
                         "value": m["matched_scale"], "seed": s, "do_sample": True})
    m = matches["random_direction"]
    for s in SEEDS:
        plan.append({"label": "matched_random_direction", "kind": "direction",
                     "row": ROW_PRIMARY, "value": m["matched_scale"], "seed": s,
                     "do_sample": True})
    return plan


# ── outputs: CSVs, figures, summary ───────────────────────────────────────────

def load_all_gen_records():
    return load_jsonl(GEN_CKPT)


def per_prompt_gc_array(conds, n_prompt):
    """Average per-prompt GC across the given list of condition-records (e.g. the 3 seeds
    of one (label,row,value) combo), aligned by prompt_idx, length n_prompt, NaN where
    missing."""
    acc = np.full((len(conds), n_prompt), np.nan)
    for i, c in enumerate(conds):
        for r in c["records"]:
            acc[i, r["prompt_idx"]] = r["gc"]
    with np.errstate(invalid="ignore"):
        return np.nanmean(acc, axis=0)


def write_dose_response_csv(all_recs, control_rows):
    import csv
    path = RESULTS_DIR / "dose_response_extended.csv"
    fields = ["label", "kind", "row", "alpha_or_c", "seed", "do_sample", "prompt_idx",
              "gc", "entropy", "distinct2", "distinct3", "distinct4",
              "longest_homopolymer", "mean_homopolymer", "top3mer_share", "top6mer_share",
              "non_acgt_filter_rate", "self_nll", "n_filtered", "n_raw"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in all_recs:
            for r in c["records"]:
                w.writerow({
                    "label": c["label"], "kind": c["kind"], "row": c["row"],
                    "alpha_or_c": c["value"], "seed": c["seed"],
                    "do_sample": c["do_sample"], "prompt_idx": r["prompt_idx"],
                    "gc": r["gc"], "entropy": r["entropy"],
                    "distinct2": r.get("distinct2"), "distinct3": r.get("distinct3"),
                    "distinct4": r.get("distinct4"),
                    "longest_homopolymer": r.get("longest_homopolymer"),
                    "mean_homopolymer": r.get("mean_homopolymer"),
                    "top3mer_share": r.get("top3mer_share"),
                    "top6mer_share": r.get("top6mer_share"),
                    "non_acgt_filter_rate": r.get("non_acgt_filter_rate"),
                    "self_nll": r.get("self_nll"),
                    "n_filtered": r["n_filtered"], "n_raw": r["n_raw"],
                })
    log(f"wrote {path}")
    return path


def write_damage_matching_csv(matches, target, n_prompt, all_recs):
    import csv
    path = RESULTS_DIR / "damage_matching.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "kind", "row", "target_damage", "matched_scale",
                    "matched_damage", "reachable", "fallback_used", "matched_gc_mean",
                    "n_grid_points", "grid_json"])
        for label, m in matches.items():
            conds = [c for c in all_recs if c["label"] == f"matched_{label}"]
            gc_mean = float("nan")
            if conds:
                arr = per_prompt_gc_array(conds, n_prompt)
                gc_mean = float(np.nanmean(arr))
            w.writerow([label, m["kind"], m["row"], target, m["matched_scale"],
                        m["matched_damage"], m["reachable"], m["fallback_used"], gc_mean,
                        len(m["grid_evals"]), json.dumps(m["grid_evals"])])
    log(f"wrote {path}")
    return path


def write_kmer_attribution_csv(all_recs, n_prompt):
    import csv
    path = RESULTS_DIR / "kmer_attribution.csv"

    def pooled_seqs(label, row, value):
        out = []
        for c in all_recs:
            if c["label"] == label and c["row"] == row and c["value"] == value:
                out.extend(r["filtered"] for r in c["records"] if r["n_filtered"] >= 10)
        return out

    base_seqs = pooled_seqs("row2371_grid", ROW_PRIMARY, 1.0)
    abl_seqs = pooled_seqs("row2371_grid", ROW_PRIMARY, 0.0)

    rows = []
    for k in (3, 6):
        base_counts = Counter()
        abl_counts = Counter()
        for s in base_seqs:
            base_counts.update(el.kmer_counts(s, k))
        for s in abl_seqs:
            abl_counts.update(el.kmer_counts(s, k))
        base_tot = sum(base_counts.values()) or 1
        abl_tot = sum(abl_counts.values()) or 1
        enrich = []
        for kmer in set(abl_counts) | set(base_counts):
            f_abl = abl_counts.get(kmer, 0) / abl_tot
            f_base = base_counts.get(kmer, 0) / base_tot
            enrich.append((kmer, f_abl, f_base, f_abl - f_base))
        enrich.sort(key=lambda t: t[3], reverse=True)
        for kmer, f_abl, f_base, diff in enrich[:5]:
            rows.append({"k": k, "kmer": kmer, "freq_alpha0": f_abl,
                         "freq_baseline": f_base, "enrichment": diff})

    # homopolymer attribution
    gc_base = float(np.nanmean([el.gc_frac(s) for s in base_seqs])) if base_seqs else float("nan")
    gc_abl = float(np.nanmean([el.gc_frac(s) for s in abl_seqs])) if abl_seqs else float("nan")
    gc_abl_norun = []
    for s in abl_seqs:
        st, en = el.longest_homopolymer_span(s)
        s2 = s[:st] + s[en:]
        if len(s2) >= 10:
            gc_abl_norun.append(el.gc_frac(s2))
    gc_abl_norun_mean = float(np.mean(gc_abl_norun)) if gc_abl_norun else float("nan")
    drop_with_it = gc_base - gc_abl
    drop_without_it = gc_base - gc_abl_norun_mean
    frac_attrib = (1 - (drop_without_it / drop_with_it)
                   if drop_with_it not in (0, float("nan")) and not np.isnan(drop_with_it)
                   else float("nan"))

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "kmer", "freq_alpha0", "freq_baseline",
                                           "enrichment"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        f.write("\n")
        f.write("# homopolymer attribution (row 2371)\n")
        f.write(f"# gc_baseline_alpha1={gc_base}\n")
        f.write(f"# gc_alpha0={gc_abl}\n")
        f.write(f"# gc_alpha0_without_longest_homopolymer={gc_abl_norun_mean}\n")
        f.write(f"# gc_drop_with_it={drop_with_it}\n")
        f.write(f"# gc_drop_without_it={drop_without_it}\n")
        f.write(f"# fraction_attributable_to_longest_homopolymer={frac_attrib}\n")
    log(f"wrote {path} (fraction attributable to longest homopolymer = {frac_attrib})")
    return path, {"gc_base": gc_base, "gc_abl": gc_abl,
                  "gc_abl_norun_mean": gc_abl_norun_mean,
                  "frac_attrib": frac_attrib}


def make_figures(all_recs, matches, target_damage, n_prompt, control_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    damage_evals = load_jsonl(DAMAGE_CKPT)

    def dvals(label, kind, row):
        pts = [(d["value"], d["damage"]) for d in damage_evals
               if d["label"] == label and d["kind"] == kind and d["row"] == row]
        pts.sort()
        return pts

    def gc_for(label, row, value):
        conds = [c for c in all_recs if c["label"] == label and c["row"] == row
                 and c["value"] == value]
        if not conds:
            return float("nan")
        return float(np.nanmean(per_prompt_gc_array(conds, n_prompt)))

    # ── figure 1: GC shift vs damage ──
    fig, ax = plt.subplots(figsize=(7, 5.5))
    d2371 = dvals("row2371", "alpha", ROW_PRIMARY)
    xs = [d for _, d in d2371]
    ys = [gc_for("row2371_grid", ROW_PRIMARY, a) for a, _ in d2371]
    ax.plot(xs, ys, "o-", color="#1f77b4", label="row 2371 (incl. amplification)")

    d1522 = [(a, None) for a in ROW1522_GRID]
    ys1522 = [gc_for("row1522_grid", ROW_SECONDARY, a) for a in ROW1522_GRID]
    xs1522 = []
    for a in ROW1522_GRID:
        dd = [d["damage"] for d in damage_evals
              if d["label"] == "row1522" and d["row"] == ROW_SECONDARY and d["value"] == a]
        xs1522.append(dd[0] if dd else float("nan"))
    if not all(np.isnan(xs1522)):
        ax.plot(xs1522, ys1522, "s--", color="#ff7f0e", label="row 1522")

    for cr in control_rows:
        pts = dvals(f"control_row_{cr}", "alpha", cr)
        if pts:
            xs_c = [d for _, d in pts]
            ys_c = [gc_for(f"matched_control_row_{cr}", cr, matches[f"control_row_{cr}"]["matched_scale"])
                    for _ in [0]]
            ax.plot([matches[f"control_row_{cr}"]["matched_damage"]], ys_c, "^",
                    color="#2ca02c", alpha=0.7,
                    label="damage-matched control rows" if cr == control_rows[0] else None)

    rd = matches["random_direction"]
    gc_rd = gc_for("matched_random_direction", ROW_PRIMARY, rd["matched_scale"])
    ax.plot([rd["matched_damage"]], [gc_rd], "D", color="#d62728", markersize=10,
            label="random-direction control (matched)")

    ax.axvline(target_damage, color="gray", linestyle=":", linewidth=1,
               label="D(2371, alpha=0) target")
    ax.set_xlabel("LM damage (mean per-token NLL, held-out damage pool)")
    ax.set_ylabel("mean generated GC fraction")
    ax.set_title("E12: GC shift vs. LM damage")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gc_vs_damage.pdf")
    plt.close(fig)
    log(f"wrote {FIG_DIR / 'gc_vs_damage.pdf'}")

    # ── figure 2: generation quality metrics by condition ──
    metrics = ["entropy", "distinct2", "distinct3", "top6mer_share",
               "longest_homopolymer", "non_acgt_filter_rate"]
    conds_of_interest = [
        ("row2371 a=1.0", "row2371_grid", ROW_PRIMARY, 1.0),
        ("row2371 a=0.0", "row2371_grid", ROW_PRIMARY, 0.0),
        ("row1522 a=0.0", "row1522_grid", ROW_SECONDARY, 0.0),
    ] + [(f"matched ctrl {cr}", f"matched_control_row_{cr}", cr,
          matches[f"control_row_{cr}"]["matched_scale"]) for cr in control_rows] + [
        ("matched rand-dir", "matched_random_direction", ROW_PRIMARY,
         matches["random_direction"]["matched_scale"]),
    ]

    def metric_mean(label, row, value, metric):
        vals = []
        for c in all_recs:
            if c["label"] == label and c["row"] == row and c["value"] == value:
                vals.extend(r.get(metric) for r in c["records"] if r.get(metric) is not None)
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.flat, metrics):
        names = [c[0] for c in conds_of_interest]
        vals = [metric_mean(c[1], c[2], c[3], metric) for c in conds_of_interest]
        ax.bar(range(len(names)), vals, color="#4c72b0")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=9)
    fig.suptitle("E12: generation-quality metrics by condition")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "generation_quality.pdf")
    plt.close(fig)
    log(f"wrote {FIG_DIR / 'generation_quality.pdf'}")


def write_summary(all_recs, matches, target_damage, n_prompt, control_rows, kmer_info,
                   corpus_meta):
    gc_2371_a1 = gc_mean("row2371_grid", ROW_PRIMARY, 1.0, all_recs, n_prompt)
    gc_2371_a0 = gc_mean("row2371_grid", ROW_PRIMARY, 0.0, all_recs, n_prompt)
    shift_2371 = gc_2371_a1 - gc_2371_a0

    per_prompt_2371_a0 = combined_gc_array("row2371_grid", ROW_PRIMARY, 0.0, all_recs, n_prompt)

    row_diffs = []
    for cr in control_rows:
        m = matches[f"control_row_{cr}"]
        gc_matched = combined_gc_array(f"matched_control_row_{cr}", cr,
                                        m["matched_scale"], all_recs, n_prompt)
        # direct diff: GC(row2371, alpha=0) - GC(damage-matched control row)
        bs = el.bootstrap_gc_diff(per_prompt_2371_a0, gc_matched, n_boot=N_BOOT)
        row_diffs.append({"row": cr, "matched_scale": m["matched_scale"],
                          "matched_damage": m["matched_damage"],
                          "reachable": m["reachable"], **bs})

    pooled_matched = np.nanmean(
        np.vstack([combined_gc_array(f"matched_control_row_{cr}", cr,
                                      matches[f"control_row_{cr}"]["matched_scale"],
                                      all_recs, n_prompt) for cr in control_rows]),
        axis=0)
    pooled_bs = el.bootstrap_gc_diff(per_prompt_2371_a0, pooled_matched, n_boot=N_BOOT)

    rd = matches["random_direction"]
    gc_rd = combined_gc_array("matched_random_direction", ROW_PRIMARY, rd["matched_scale"],
                               all_recs, n_prompt)
    rd_bs = el.bootstrap_gc_diff(per_prompt_2371_a0, gc_rd, n_boot=N_BOOT)

    # amplification-arm shape check for row 2371
    amp_gcs = {a: gc_mean("row2371_grid", ROW_PRIMARY, a, all_recs, n_prompt)
               for a in ROW2371_GRID}

    # decision rule
    # diff is defined as GC(row2371,alpha=0) - GC(matched control): row2371's real drop
    # makes this NEGATIVE when the control does NOT reproduce the effect (specificity
    # supported) and near-zero when it does (degradation-explained). "Supported" therefore
    # means the CI is confidently negative (ci_hi < 0), not confidently positive -- a
    # ci_lo > 0 check can never fire for a quantity that is negative by construction
    # whenever row 2371 drops and the control doesn't. Fixed 2026-08-24 after the raw
    # per-prompt data (matched control rows ~0.42, unchanged from baseline; row2371 ~0.31)
    # was checked directly and found to contradict the original branch label.
    row_axis_supported = all(d["ci_hi"] < 0 for d in row_diffs) and pooled_bs["ci_hi"] < 0
    dir_axis_supported = rd_bs["ci_hi"] < 0
    frac_attrib_kmer = kmer_info["frac_attrib"]

    if row_axis_supported and dir_axis_supported:
        branch = "Specific-control supported (both axes)"
    elif not row_axis_supported and not dir_axis_supported:
        branch = "Degradation-explained (both axes)"
    else:
        branch = ("Intermediate / axis-disagreement -- row-location axis "
                  f"{'supported' if row_axis_supported else 'not supported'}, "
                  f"random-direction axis {'supported' if dir_axis_supported else 'not supported'}")

    lines = []
    lines.append("# E12 summary -- degradation-matched control for the GENERator GC result\n")
    lines.append(
        f"**Decision-rule outcome: {branch}.** Row 2371's raw GC shift at alpha=0 vs. "
        f"alpha=1.0 is {shift_2371:.4f} ({gc_2371_a1:.4f} -> {gc_2371_a0:.4f}). Pooled "
        f"bootstrap 95% CI on GC(row2371,alpha=0) - GC(matched control rows) = "
        f"[{pooled_bs['ci_lo']:.4f}, {pooled_bs['ci_hi']:.4f}] (point={pooled_bs['point']:.4f}, "
        f"n={pooled_bs['n']}); the random-direction-matched control at row 2371's own "
        f"location gives CI = [{rd_bs['ci_lo']:.4f}, {rd_bs['ci_hi']:.4f}] "
        f"(point={rd_bs['point']:.4f}). Per-row detail: "
        + "; ".join(f"row {d['row']} (matched scale={d['matched_scale']}, "
                    f"reachable={d['reachable']}): diff={d['point']:.4f} "
                    f"CI=[{d['ci_lo']:.4f},{d['ci_hi']:.4f}]" for d in row_diffs) + ". "
        f"The two axes are reported separately per the prereg: control-row axis "
        f"{'supports' if row_axis_supported else 'does not support'} specificity; "
        f"random-direction axis {'supports' if dir_axis_supported else 'does not support'} "
        f"it. If row-axis and direction-axis disagree, that disagreement is the finding, "
        f"not resolved to a single verdict.\n")

    lines.append(
        f"**Damage matching.** Target damage D(2371, alpha=0) = {target_damage:.6f}. "
        + "; ".join(f"control row {d['row']}: matched at scale={d['matched_scale']}, "
                    f"damage={d['matched_damage']:.6f}"
                    + (" [FALLBACK -- target unreachable on grid]" if not d["reachable"] else "")
                    for d in row_diffs) + f". Random-direction control matched at "
        f"c={rd['matched_scale']}, damage={rd['matched_damage']:.6f}"
        + (" [FALLBACK -- target unreachable on grid]" if not rd["reachable"] else "") + ".\n")

    lines.append(
        "**Question 2 (separate from specificity): does row 2371's own dose-response "
        "saturate then reverse at amplification, replicating C-040's shape under this "
        "tighter protocol?** Row 2371 GC by alpha (suppression -> amplification): "
        + ", ".join(f"a={a}: GC={amp_gcs[a]:.4f}" for a in sorted(amp_gcs)) + ". "
        "This bears only on which of C-040 (saturate-then-reverse, span ratio 38.59x) or "
        "C-045 (mostly-monotonic up to alpha=1.0, never measured above) is more "
        "trustworthy going forward; per the prereg it is NOT used as evidence for or "
        "against the B3 specificity verdict above, and vice versa.\n")

    lines.append(
        f"**B4 attribution (row 2371, alpha=0 vs. alpha=1.0 baseline).** Fraction of the "
        f"GC drop attributable to the single longest homopolymer run per continuation: "
        f"{frac_attrib_kmer:.4f} (GC baseline={kmer_info['gc_base']:.4f}, "
        f"GC alpha=0={kmer_info['gc_abl']:.4f}, GC alpha=0 with longest run removed="
        f"{kmer_info['gc_abl_norun_mean']:.4f}). Top enriched 3-mers/6-mers are in "
        f"`results/experiments/E12/kmer_attribution.csv`.\n")

    lines.append(
        f"**Corpus.** {corpus_meta.get('n_prompt')} generation-prompt windows "
        f"({PROMPT_BP}bp, first {PROMPT_LEN}bp used as prompt) and "
        f"{corpus_meta.get('n_damage')} damage-pool windows ({DAMAGE_BP}bp), drawn "
        f"disjoint from `data/regions/hg38/random_262kb.bed` (partition seed "
        f"{corpus_meta.get('partition_seed_used')}, {corpus_meta.get('repartition_tries')} "
        f"repartition attempt(s) needed). Existing C-045/C-040 numbers are reported as-is "
        f"per the working rule against silently adjusting prior figures; any numeric "
        f"discrepancy with this run's row-2371 alpha=1.0/0.5/0.0 GC values is called out "
        f"explicitly rather than smoothed over.\n")

    (RESULTS_DIR / "E12_summary.md").write_text("\n".join(lines))
    log(f"wrote {RESULTS_DIR / 'E12_summary.md'}")


def gc_mean(label, row, value, all_recs, n_prompt):
    conds = [c for c in all_recs if c["label"] == label and c["row"] == row
             and c["value"] == value]
    if not conds:
        return float("nan")
    return float(np.nanmean(per_prompt_gc_array(conds, n_prompt)))


def combined_gc_array(label, row, value, all_recs, n_prompt):
    conds = [c for c in all_recs if c["label"] == label and c["row"] == row
             and c["value"] == value]
    if not conds:
        return np.full(n_prompt, np.nan)
    return per_prompt_gc_array(conds, n_prompt)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    log(f"=== E12 run start (SMOKE={SMOKE}) ===")
    model, tok, cfg, pattern, nrows, intermediate_size = load_model()

    control_rows = random.Random(BASE_SEED).sample(
        [r for r in range(nrows) if r not in (ROW_PRIMARY, ROW_SECONDARY)], 5)
    log(f"control rows (C-045-reproducing): {control_rows}")

    prompt_wins, damage_wins, corpus_meta = get_corpora()
    prompts = [s[:PROMPT_LEN] for _c, _st, s in prompt_wins]

    dev = next(model.parameters()).device
    dp_module = dict(model.named_modules())[pattern.format(i=GENERATOR_LAYER)]
    unit_dir = el.make_unit_random_direction(intermediate_size, ROW_PRIMARY,
                                              dp_module.weight.dtype, dev)
    unit_dir_by_row = {ROW_PRIMARY: unit_dir}

    log("=== B3 step 1-3: damage measurement + matching ===")
    target_damage, matches = run_damage_matching(model, tok, pattern, damage_wins,
                                                  control_rows, unit_dir,
                                                  intermediate_size)

    log("=== B1/B2/B3-step4: generation work plan ===")
    plan = build_gen_plan(control_rows, matches)
    execute_gen_plan(model, tok, pattern, prompts, plan, unit_dir_by_row)

    log("=== building outputs ===")
    all_recs = load_all_gen_records()
    write_dose_response_csv(all_recs, control_rows)
    write_damage_matching_csv(matches, target_damage, len(prompts), all_recs)
    _, kmer_info = write_kmer_attribution_csv(all_recs, len(prompts))
    make_figures(all_recs, matches, target_damage, len(prompts), control_rows)
    write_summary(all_recs, matches, target_damage, len(prompts), control_rows, kmer_info,
                  corpus_meta)

    log("=== E12 run COMPLETE ===")


if __name__ == "__main__":
    main()
