# scripts/analyze_task_sw_overlap.py
"""
Analyze overlap of task-specific superweights across GUE benchmarks.

Reads the JSON produced by find_task_superweights.py and produces:
  1. Per-task SW summary table
  2. Pairwise Jaccard similarity matrix
  3. Shared-SW catalogue (which rows appear in >1 task)
  4. (Optional) a heatmap PNG if matplotlib is available

Usage:
    python scripts/analyze_task_sw_overlap.py \\
        --in results/task_superweights.json

    # filter to one model:
    python scripts/analyze_task_sw_overlap.py \\
        --in results/task_superweights.json --model dnabert2

    # save heatmap:
    python scripts/analyze_task_sw_overlap.py \\
        --in results/task_superweights.json --heatmap results/sw_overlap_heatmap.png
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sw_set(entry: dict) -> set[tuple[int, int]]:
    """Return the task SW as a frozenset of (layer, row) tuples."""
    return {(r["layer"], r["row"]) for r in entry.get("task_sw", [])}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _fmt_row(label: str, fields: list[str]) -> str:
    return f"{label:<40s}" + "".join(f"{f:>8s}" for f in fields)


# ── Overlap analysis ──────────────────────────────────────────────────────────

def analyze(data: dict, model_filter: str | None = None) -> None:
    # Filter entries
    entries = {}
    for key, val in data.items():
        if model_filter and not key.startswith(model_filter + "/"):
            continue
        entries[key] = val

    if not entries:
        print("[warn] No entries found (check --model filter or run find_task_superweights.py first).")
        return

    keys   = sorted(entries.keys())
    sw_sets = {k: _sw_set(entries[k]) for k in keys}

    # ── 1. Per-task summary ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Per-task task-superweight summary")
    print("=" * 72)
    print(_fmt_row("Task", ["#SW", "base_acc", "base_mcc"]))
    print("-" * 72)
    for k in keys:
        e    = entries[k]
        bsl  = e.get("baseline", {})
        nsw  = len(e.get("task_sw", []))
        acc  = bsl.get("accuracy", float("nan"))
        mcc  = bsl.get("mcc",      float("nan"))
        print(_fmt_row(k, [str(nsw), f"{acc:.4f}", f"{mcc:.4f}"]))

    # Detailed SW list
    print("\n  Task-superweight details (layer, row, Δacc, z):")
    print("-" * 72)
    for k in keys:
        e  = entries[k]
        sw = sorted(e.get("task_sw", []), key=lambda r: r["delta_acc"])
        if not sw:
            print(f"  {k}: (none above threshold)")
            continue
        print(f"  {k}:")
        for r in sw:
            print(f"    layer={r['layer']:2d}  row={r['row']:5d}  "
                  f"Δacc={r['delta_acc']:+.5f}  z={r['z_score']:+.2f}  "
                  f"L1={r['l1_norm']:.1f}")

    # ── 2. Pairwise Jaccard matrix ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Pairwise Jaccard similarity (task superweight sets)")
    print("=" * 72)
    short_keys = [k.split("/", 1)[1] if "/" in k else k for k in keys]
    col_w = max(len(s) for s in short_keys) + 2

    # Header
    header = " " * col_w + "".join(f"{s:>{col_w}s}" for s in short_keys)
    print(header)
    print("-" * len(header))
    for i, ki in enumerate(keys):
        row_vals = []
        for kj in keys:
            j_val = _jaccard(sw_sets[ki], sw_sets[kj])
            row_vals.append(f"{j_val:.2f}")
        print(f"{short_keys[i]:<{col_w}}" + "".join(f"{v:>{col_w}s}" for v in row_vals))

    # ── 3. Shared-SW catalogue ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Rows shared across multiple tasks")
    print("=" * 72)

    # Count how many tasks each (layer, row) appears in
    row_to_tasks: dict[tuple[int, int], list[str]] = defaultdict(list)
    for k in keys:
        for coord in sw_sets[k]:
            row_to_tasks[coord].append(k)

    shared = {coord: tasks for coord, tasks in row_to_tasks.items() if len(tasks) > 1}

    if not shared:
        print("  No superweight rows are shared across multiple tasks.")
    else:
        print(f"  {len(shared)} shared row(s):\n")
        for (layer, row), tasks in sorted(shared.items(), key=lambda x: -len(x[1])):
            print(f"  layer={layer:2d}  row={row:5d}  "
                  f"({len(tasks)} tasks): {', '.join(sorted(tasks))}")

    # ── 4. Universal vs task-specific summary ────────────────────────────────
    n_tasks = len(keys)
    universal   = [c for c, t in row_to_tasks.items() if len(t) == n_tasks]
    task_only   = [c for c, t in row_to_tasks.items() if len(t) == 1]
    multi_not_all = [c for c, t in row_to_tasks.items() if 1 < len(t) < n_tasks]

    print("\n" + "=" * 72)
    print("  Cross-task SW breakdown")
    print("=" * 72)
    print(f"  Tasks analysed  : {n_tasks}")
    print(f"  Universal SWs   : {len(universal)}"
          f"  (present in all {n_tasks} tasks)")
    print(f"  Partial SWs     : {len(multi_not_all)}"
          f"  (shared by 2–{n_tasks-1} tasks)")
    print(f"  Task-only SWs   : {len(task_only)}"
          f"  (unique to 1 task)")

    if universal:
        print("\n  Universal superweights:")
        for (layer, row) in sorted(universal):
            print(f"    layer={layer:2d}  row={row:5d}")

    print()


# ── Heatmap ───────────────────────────────────────────────────────────────────

def save_heatmap(data: dict, out_path: str, model_filter: str | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not available — skipping heatmap.", file=sys.stderr)
        return

    entries = {}
    for key, val in data.items():
        if model_filter and not key.startswith(model_filter + "/"):
            continue
        entries[key] = val

    keys    = sorted(entries.keys())
    sw_sets = {k: _sw_set(entries[k]) for k in keys}
    n       = len(keys)

    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            mat[i, j] = _jaccard(sw_sets[ki], sw_sets[kj])

    short_keys = [k.split("/", 1)[1] if "/" in k else k for k in keys]

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_keys, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if mat[i, j] < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Jaccard similarity")
    ax.set_title("Task superweight set overlap (Jaccard)")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Heatmap saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse cross-task superweight overlap from task_superweights.json."
    )
    parser.add_argument("--in",      dest="in_path",
                        default="results/task_superweights.json",
                        help="Path to task_superweights.json (output of find_task_superweights.py).")
    parser.add_argument("--model",   default=None,
                        help="Filter to a single model key (e.g. dnabert2).")
    parser.add_argument("--heatmap", default=None,
                        help="If set, save Jaccard heatmap PNG to this path.")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"[error] {in_path} not found.\n"
              "Run find_task_superweights.py first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(in_path.read_text())
    analyze(data, model_filter=args.model)

    if args.heatmap:
        save_heatmap(data, args.heatmap, model_filter=args.model)


if __name__ == "__main__":
    main()
