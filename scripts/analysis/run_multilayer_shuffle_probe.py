"""
run_multilayer_shuffle_probe.py
--------------------------------
Tests whether shuffle sensitivity (a proxy for context-dependence) tracks with
layer depth or training corpus by running the four standard shuffle controls at
multiple (layer, top-1-||U_k||_F-row) probe points in three GENERator models:

  • generator            (EUK 3B)  — actual SW at L4,  probed at L2/L4/L6/L14
  • generator_prokaryote (PROK 3B) — actual SW at L2,  probed at L2/L4/L6/L14
  • generator_prokaryote_1b        — SW auto-detected, probed at actual SW layer

Hypothesis:
  Layer-position hypothesis  → shuffle sensitivity (|activation shift|) peaks at
                               the same depth regardless of training corpus.
  Corpus hypothesis          → sensitivity tracks with the actual SW layer of each
                               model (L4 for EUK, L2 for PROK).

Sequences: n=90 real genomic sequences sampled from GUE test splits
  (30 × prom/prom_300_all, 30 × virus/species_40, 30 × mouse/0).

Shuffle types: dinuc, mono, trinuc, kmer_block  (from run_sw_shuffle_controls.py)

Output:
  results/multilayer_shuffle_probe.csv
    columns: model, probe_layer, probe_row, is_actual_sw,
             shuffle_type, n, mean_shift, std_shift, t_stat, p_value

  results/multilayer_shuffle_probe.json   (full per-sequence data)

Usage (from repo root):
  python scripts/analysis/run_multilayer_shuffle_probe.py [--n_seqs 90] [--n_shuffles 5]

Needs 1× A100 (3B model + repeated forward passes). ~1.5–2 h for all 3 models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── copy shuffle functions from run_sw_shuffle_controls.py ───────────────────
# (duplicated here to keep the script self-contained)

def _mono_shuffle(seq: str, rng: random.Random) -> str:
    bases = list(seq.upper()); rng.shuffle(bases); return "".join(bases)


def _dinuc_shuffle(seq: str, rng: random.Random) -> str:
    from collections import defaultdict
    VALID = set("ACGT")
    bases = [b if b in VALID else "N" for b in seq.upper()]
    if len(bases) < 4:
        rng.shuffle(bases); return "".join(bases)
    transitions = defaultdict(list)
    for i in range(len(bases) - 1):
        transitions[bases[i]].append(bases[i + 1])
    for k in transitions:
        rng.shuffle(transitions[k])
    result = [bases[0]]; current = bases[0]
    for _ in range(len(bases) - 1):
        nexts = transitions.get(current)
        if not nexts:
            remaining = [b for v in transitions.values() for b in v]
            if not remaining: break
            nxt = rng.choice(remaining)
            for k in transitions:
                if nxt in transitions[k]:
                    transitions[k].remove(nxt); break
        else:
            nxt = nexts.pop(0)
        result.append(nxt); current = nxt
    if len(result) < len(bases):
        pool = list(seq.upper()); rng.shuffle(pool)
        result.extend(pool[:len(bases) - len(result)])
    return "".join(result[:len(bases)])


def _kmer_block_shuffle(seq: str, k: int, rng: random.Random) -> str:
    trim = len(seq) - (len(seq) % k)
    tokens = [seq.upper()[i:i+k] for i in range(0, trim, k)]
    rng.shuffle(tokens); return "".join(tokens)


def _trinuc_shuffle(seq: str, rng: random.Random) -> str:
    from collections import defaultdict
    VALID = set("ACGT")
    bases = [b if b in VALID else "N" for b in seq.upper()]
    n = len(bases)
    if n < 3:
        rng.shuffle(bases); return "".join(bases)
    adj = defaultdict(list)
    for i in range(n - 2):
        adj[bases[i] + bases[i+1]].append(bases[i+2])
    for node in adj:
        rng.shuffle(adj[node])
    adj_lst = {nd: list(e) for nd, e in adj.items()}
    adj_idx = {nd: 0 for nd in adj_lst}
    start = bases[0] + bases[1]; stack = [start]; path = []
    while stack:
        u = stack[-1]
        if adj_idx.get(u, 0) < len(adj_lst.get(u, [])):
            c = adj_lst[u][adj_idx[u]]; adj_idx[u] += 1
            stack.append(u[1] + c)
        else:
            path.append(stack.pop())
    path = path[::-1]
    if not path: return "".join(bases)
    result = list(path[0])
    for node in path[1:]:
        result.append(node[-1])
    if len(result) < n:
        pool = bases[:]; rng.shuffle(pool)
        result.extend(pool[:n - len(result)])
    return "".join(result[:n])


SHUFFLE_FNS = {
    "mono":       lambda seq, rng: _mono_shuffle(seq, rng),
    "dinuc":      lambda seq, rng: _dinuc_shuffle(seq, rng),
    "trinuc":     lambda seq, rng: _trinuc_shuffle(seq, rng),
    "kmer_block": lambda seq, rng: _kmer_block_shuffle(seq, 6, rng),
}

SHUFFLE_ORDER = ["mono", "dinuc", "trinuc", "kmer_block"]


# ── sequence loading from GUE CSVs ────────────────────────────────────────────

# Per-model task lists: use in-distribution sequences for each model.
# EUK → eukaryotic tasks (prom, mouse enhancers, EPI)
# PROK → prokaryotic tasks (virus/species_40, fungi)
# PROK-1B → same as PROK-3B
_MODEL_TASKS = {
    "generator": [
        "prom/prom_300_all",        # 300 bp, eukaryotic promoters (in-distribution for EUK)
        "EMP/H3",                   # 500 bp, histone modification (eukaryotic chromatin)
        "splice/reconstructed",     # 400 bp, splice sites (eukaryotic)
    ],
    "generator_prokaryote": [
        "virus/species_40",         # 5000 bp, diverse viruses (prokaryotic domain)
        "fungi/species_20",         # fungal sequences — broad but closer to EUK; keep for PROK as OOD control
    ],
    "generator_prokaryote_1b": [
        "virus/species_40",
        "fungi/species_20",
    ],
}


def load_gue_sequences(gue_root: str, model_key: str, n_per_task: int,
                        seed: int, seq_len: int) -> list[str]:
    """
    Sample n_per_task sequences from model-appropriate GUE tasks.
    Trims each sequence to seq_len (rounded down to multiple of 6).
    If seq_len > actual sequence length, takes the full sequence (trimmed to ×6).
    """
    rng  = random.Random(seed)
    tasks = _MODEL_TASKS.get(model_key, list(_MODEL_TASKS.values())[0])
    seqs: list[str] = []
    for task in tasks:
        csv_path = Path(gue_root) / task / "test.csv"
        if not csv_path.exists():
            print(f"  [warn] GUE task not found: {csv_path}  — skipping")
            continue
        rows = []
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                s = row.get("sequence", row.get("seq", "")).strip().upper().replace(" ", "")
                if len(s) < 12:   # need at least 2 tokens
                    continue
                # trim to min(seq_len, actual_len), rounded to multiple of 6
                target = min(seq_len, len(s))
                target = (target // 6) * 6
                if target < 12:
                    continue
                rows.append(s[:target])
        if not rows:
            print(f"  [warn] No usable sequences in {csv_path}")
            continue
        sampled = rng.sample(rows, min(n_per_task, len(rows)))
        seqs.extend(sampled)
        print(f"  {task}: {len(sampled)} seqs (pool={len(rows)}, len={len(sampled[0])} bp)")
    return seqs


# ── hook-based activation capture ────────────────────────────────────────────

def capture_activation(model, tokenizer, seq: str,
                        layer: int, row: int) -> float:
    """Mean activation of a single row at down_proj output of given layer."""
    seq = seq.upper()
    remainder = len(seq) % 6
    if remainder:
        seq = seq[remainder:]
    bos = tokenizer.bos_token or ""
    ids = tokenizer(bos + seq, return_tensors="pt",
                    add_special_tokens=False)["input_ids"]
    ids = ids.to(next(model.parameters()).device)

    store: dict = {}
    def hook(_mod, _inp, _out):
        store["act"] = _out[..., row].detach().float().cpu().numpy()
    h = model.model.layers[layer].mlp.down_proj.register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids=ids)
    h.remove()
    return float(store["act"].mean()) if "act" in store else 0.0


# ── statistics ────────────────────────────────────────────────────────────────

def t_test(shifts: list[float]) -> tuple[float, float]:
    vals = [v for v in shifts if not math.isnan(v)]
    n = len(vals)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(vals) / n
    var  = sum((v - mean)**2 for v in vals) / (n - 1)
    if var == 0:
        return (float("inf") if mean > 0 else float("-inf")), 0.0
    t = mean / (var / n) ** 0.5
    try:
        from scipy.stats import t as tdist
        p = float(tdist.sf(abs(t), df=n-1) * 2)
    except ImportError:
        p = math.erfc(abs(t) / math.sqrt(2))
    return float(t), float(p)


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(config: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtypes = {"float16": torch.float16, "float32": torch.float32,
              "bfloat16": torch.bfloat16}
    dtype = dtypes.get(config.get("dtype", "float32"), torch.float32)
    tok = AutoTokenizer.from_pretrained(
        config["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], torch_dtype=dtype,
        trust_remote_code=True, device_map="auto")
    model.eval()
    return model, tok


# ── SW detection from ||U_k||_F JSON ─────────────────────────────────────────

def top1_row_per_layer(mech_json_path: Path) -> dict[int, int]:
    """Returns {layer_idx: top1_row} from a sw_mechanistic_*.json."""
    if not mech_json_path.exists():
        return {}
    d = json.loads(mech_json_path.read_text())
    out = {}
    for li_str, frob_list in d["frob_norm_uk_by_layer"].items():
        arr = np.array(frob_list)
        out[int(li_str)] = int(np.argmax(arr))
    return out


# ── main ──────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        "key":         "generator",
        "label":       "EUK-3B",
        "config_file": "generator.yaml",
        "mech_json":   "results/sw_mechanistic_generator.json",
        "actual_sw":   {"layer": 4, "row": 2371},
        "probe_layers": [2, 4, 6, 14],
    },
    {
        "key":         "generator_prokaryote",
        "label":       "PROK-3B",
        "config_file": "generator_prokaryote.yaml",
        "mech_json":   "results/sw_mechanistic_generator_prokaryote.json",
        "actual_sw":   {"layer": 2, "row": 1927},
        "probe_layers": [2, 4, 6, 14],
    },
    {
        "key":         "generator_prokaryote_1b",
        "label":       "PROK-1B",
        "config_file": "generator_prokaryote_1b.yaml",
        "mech_json":   None,   # will be detected live (or from audit JSON if available)
        "actual_sw":   None,   # auto-detected
        "probe_layers": None,  # set to [actual_sw_layer] after detection
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seqs",     type=int, default=90)
    parser.add_argument("--n_shuffles", type=int, default=5)
    parser.add_argument("--seq_len",    type=int, default=504)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--gue_root",   default="/work/11034/atzanakak/GUE/GUE")
    parser.add_argument("--out_csv",    default="results/multilayer_shuffle_probe.csv")
    parser.add_argument("--out_json",   default="results/multilayer_shuffle_probe.json")
    parser.add_argument("--models",     nargs="+",
                        default=["generator", "generator_prokaryote",
                                 "generator_prokaryote_1b"],
                        help="Subset of models to run")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    n_per_task = max(1, args.n_seqs // 3)

    csv_rows: list[dict] = []
    json_records: list[dict] = []

    for mcfg in MODEL_CONFIGS:
        if mcfg["key"] not in args.models:
            continue

        # ── load model-appropriate sequences ──────────────────────────────
        print(f"\nLoading sequences for {mcfg['label']} (model_key={mcfg['key']}) …")
        sequences = load_gue_sequences(
            args.gue_root, mcfg["key"], n_per_task, args.seed, args.seq_len)
        if not sequences:
            print("  No sequences loaded — skipping model.")
            continue
        print(f"  Total: {len(sequences)} sequences\n")

        print(f"\n{'='*60}")
        print(f"Model: {mcfg['label']}  ({mcfg['key']})")
        print(f"{'='*60}")

        cfg_path = ROOT / "configs" / mcfg["config_file"]
        config   = yaml.safe_load(cfg_path.read_text())
        n_layers = config["num_layers"]

        # ── resolve probe points ───────────────────────────────────────────
        mech_path = ROOT / mcfg["mech_json"] if mcfg["mech_json"] else None

        # Check for pre-existing 1B audit JSON
        prok1b_mech = ROOT / "results" / "sw_mechanistic_generator_prokaryote_1b.json"
        if mcfg["key"] == "generator_prokaryote_1b" and prok1b_mech.exists():
            mech_path = prok1b_mech

        layer_to_row = top1_row_per_layer(mech_path) if mech_path else {}

        if mcfg["actual_sw"] is None:
            # PROK-1B: detect from mech JSON or fall back to computing on the fly
            if layer_to_row:
                # use layer with highest absolute frob ratio
                d = json.loads(mech_path.read_text())
                best_layer, best_row = 0, 0
                best_ratio = 0.0
                for li_str, frob_list in d["frob_norm_uk_by_layer"].items():
                    arr = np.array(frob_list)
                    ratio = arr.max() / (np.median(arr) + 1e-9)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_layer = int(li_str)
                        best_row   = int(np.argmax(arr))
                mcfg["actual_sw"]   = {"layer": best_layer, "row": best_row}
                # Probe the auto-detected layer AND layer 2 (canonical SW from
                # super_weight_index.json), deduplicating.
                canonical_sw_layer = 2
                probe_set = sorted(set([best_layer, canonical_sw_layer]))
                mcfg["probe_layers"] = probe_set
                print(f"  Auto-detected SW: layer={best_layer}, row={best_row}, ratio={best_ratio:.1f}x")
                print(f"  Probe layers: {probe_set} (auto + canonical L{canonical_sw_layer})")
            else:
                print(f"  No mech JSON for PROK-1B — run run_generator_uk_audit.py --variant prok1b first")
                print(f"  Using L0/row-0 as placeholder probe (uninformative)")
                mcfg["actual_sw"]    = {"layer": 0, "row": 0}
                mcfg["probe_layers"] = [0]

        actual_sw_layer = mcfg["actual_sw"]["layer"]
        actual_sw_row   = mcfg["actual_sw"]["row"]
        probe_layers    = mcfg["probe_layers"]

        # For each probe layer, use top-1 ||U_k||_F row from mech JSON
        # (for EUK and PROK-3B this is always row 2371 / row 1927 at every layer)
        probe_points: list[tuple[int, int, bool]] = []
        for li in probe_layers:
            row = layer_to_row.get(li, actual_sw_row)
            is_sw = (li == actual_sw_layer and row == actual_sw_row)
            probe_points.append((li, row, is_sw))

        print(f"  Probe points: {[(li, row, is_sw) for li, row, is_sw in probe_points]}")
        print(f"  Actual SW: layer={actual_sw_layer}, row={actual_sw_row}")
        print(f"  Sequences: {len(sequences)}, shuffles/seq: {args.n_shuffles}")

        # ── load model ─────────────────────────────────────────────────────
        print(f"  Loading model from {config['model_id']} …", flush=True)
        model, tokenizer = load_model(config)
        print("  Model loaded.\n", flush=True)

        # ── run probes ─────────────────────────────────────────────────────
        for probe_layer, probe_row, is_sw in probe_points:
            print(f"  Probe L{probe_layer} row {probe_row}"
                  f"  (is_actual_SW={is_sw}) …")

            per_stype_shifts: dict[str, list[float]] = {s: [] for s in SHUFFLE_ORDER}

            for si, seq in enumerate(sequences):
                real_act = capture_activation(
                    model, tokenizer, seq, probe_layer, probe_row)

                for stype in SHUFFLE_ORDER:
                    shuf_fn = SHUFFLE_FNS[stype]
                    shift_vals = []
                    for _ in range(args.n_shuffles):
                        shuf = shuf_fn(seq, rng)
                        shuf = shuf[: (len(shuf) // 6) * 6]
                        shuf_act = capture_activation(
                            model, tokenizer, shuf, probe_layer, probe_row)
                        mu_r, mu_s = abs(real_act), abs(shuf_act)
                        sig = abs(real_act - shuf_act) / (abs(real_act) + abs(shuf_act) + 1e-9)
                        # standardised shift: (real - shuffled) / pooled std proxy
                        shift_vals.append(float(real_act - shuf_act) /
                                          (abs(real_act) + abs(shuf_act) + 1e-9) * 2)
                    per_stype_shifts[stype].append(float(np.mean(shift_vals)))

                if (si + 1) % 30 == 0:
                    print(f"    seq {si+1}/{len(sequences)}", flush=True)

            # aggregate per shuffle type
            for stype in SHUFFLE_ORDER:
                shifts = per_stype_shifts[stype]
                t_stat, p_val = t_test(shifts)
                mean_s = float(np.mean(shifts))
                std_s  = float(np.std(shifts))
                row_d  = {
                    "model":          mcfg["label"],
                    "model_key":      mcfg["key"],
                    "probe_layer":    probe_layer,
                    "probe_row":      probe_row,
                    "is_actual_sw":   is_sw,
                    "shuffle_type":   stype,
                    "n":              len(shifts),
                    "mean_shift":     round(mean_s, 6),
                    "std_shift":      round(std_s, 6),
                    "t_stat":         round(t_stat, 4),
                    "p_value":        round(p_val, 6),
                }
                csv_rows.append(row_d)
                sig = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                       else "*" if p_val < 0.05 else "n.s.")
                print(f"    {stype:12s}  mean={mean_s:+.4f}  t={t_stat:7.3f}"
                      f"  p={p_val:.4f}  {sig}")

            json_records.append({
                "model": mcfg["label"],
                "probe_layer": probe_layer,
                "probe_row": probe_row,
                "is_actual_sw": is_sw,
                "per_sequence_shifts": per_stype_shifts,
            })

        # free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    # ── write outputs ──────────────────────────────────────────────────────
    out_csv  = ROOT / args.out_csv
    out_json = ROOT / args.out_json
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if csv_rows:
        fieldnames = ["model", "model_key", "probe_layer", "probe_row",
                      "is_actual_sw", "shuffle_type", "n",
                      "mean_shift", "std_shift", "t_stat", "p_value"]
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nCSV saved → {out_csv}")

    with open(out_json, "w") as f:
        json.dump({
            "n_seqs": len(sequences),
            "n_shuffles": args.n_shuffles,
            "records": json_records,
        }, f, indent=2)
    print(f"JSON saved → {out_json}")

    # ── print summary table ────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(f"{'Model':<10} {'L':>3} {'row':>5} {'SW?':>4}  "
          + "  ".join(f"{s[:8]:>8}" for s in SHUFFLE_ORDER))
    print("-" * 70)
    for rec in json_records:
        m   = rec["model"][:9]
        li  = rec["probe_layer"]
        row = rec["probe_row"]
        sw  = "✓" if rec["is_actual_sw"] else ""
        # get p-values for each shuffle type
        p_row = {r["shuffle_type"]: r["p_value"]
                 for r in csv_rows
                 if r["model_key"] == next(
                     mc["key"] for mc in MODEL_CONFIGS
                     if mc["label"] == rec["model"])
                 and r["probe_layer"] == li}
        pvals = "  ".join(
            f"{p_row.get(s, float('nan')):>8.4f}" for s in SHUFFLE_ORDER)
        print(f"{m:<10} {li:>3} {row:>5} {sw:>4}  {pvals}")


if __name__ == "__main__":
    main()
