#!/usr/bin/env python
from __future__ import annotations

"""Two-track postprocessing for general superactivations scan outputs.

Track 1 (general superchannels): channels that carry a disproportionate share of
activation mass across diverse windows. Operationalized via per-channel `mean`.

Track 2 (spiky superactivations): motif-detector-like channels with rare, strong
spikes. Operationalized via tail ratios like q999/median or q999/q99.

This script is designed to run on an existing scan run directory produced by
scripts/general_superactivations_scan.py.

It avoids pandas/pyarrow so it can run in minimal Python environments.
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WindowMeta:
    idx: int
    region_set: str
    chrom: str
    start0: int
    end0: int
    strand: str


def _sanitize_layer_name(layer: str) -> str:
    return str(layer).replace("/", "_").replace(".", "_")


def _parse_float(s: object) -> float:
    try:
        return float(str(s))
    except Exception:
        return float("nan")


def _read_tsv(path: Path) -> List[dict]:
    with path.open("r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _write_tsv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=list(fieldnames), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _gini(values: np.ndarray) -> float:
    """Gini coefficient for non-negative values."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return float("nan")
    s = float(x.sum())
    if s <= 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    # Gini = (2*sum(i*x_i)/(n*sum(x))) - (n+1)/n
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * float((idx * x).sum()) / (n * s)) - (n + 1.0) / n)


def _hhi(values: np.ndarray) -> float:
    """Herfindahl-Hirschman Index of mass concentration."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return float("nan")
    s = float(x.sum())
    if s <= 0:
        return float("nan")
    p = x / s
    return float(np.sum(p * p))


def _mass_fractions(sorted_mass: np.ndarray, fracs: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    m = np.asarray(sorted_mass, dtype=np.float64)
    m = m[np.isfinite(m)]
    m = np.maximum(m, 0.0)
    total = float(m.sum())
    if total <= 0:
        for f in fracs:
            out[f"top{int(round(100*f))}pct_mass_frac"] = float("nan")
        return out

    n = int(m.size)
    for f in fracs:
        k = max(1, int(math.ceil(float(f) * float(n))))
        out[f"top{int(round(100*f))}pct_mass_frac"] = float(m[:k].sum() / total)
    return out


def _log2_safe(x: float) -> float:
    if not math.isfinite(x) or x <= 0:
        return float("nan")
    return float(math.log(x, 2))


def _open_layer_memmap(run_dir: Path, layer: str, n_windows: int, n_channels: int) -> np.memmap:
    safe = _sanitize_layer_name(layer)
    p = run_dir / f"{safe}.acts.float32.memmap"
    if not p.exists():
        raise FileNotFoundError(f"Missing activation memmap for layer '{layer}': {p}")
    return np.memmap(p, dtype=np.float32, mode="r", shape=(int(n_windows), int(n_channels)))


def _read_windows_meta(run_dir: Path) -> List[WindowMeta]:
    p = run_dir / "windows.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Missing windows.tsv: {p}")
    rows = _read_tsv(p)
    out: List[WindowMeta] = []
    for r in rows:
        out.append(
            WindowMeta(
                idx=int(r["idx"]),
                region_set=str(r["region_set"]),
                chrom=str(r["chrom"]),
                start0=int(r["start0"]),
                end0=int(r["end0"]),
                strand=str(r.get("strand", ".")),
            )
        )
    out.sort(key=lambda w: w.idx)
    # Ensure contiguous indexing.
    if out and out[0].idx != 0:
        raise ValueError("windows.tsv idx does not start at 0")
    if out and out[-1].idx != len(out) - 1:
        raise ValueError("windows.tsv idx is not contiguous")
    return out


def _read_processed_mask(run_dir: Path, n_windows: int) -> np.ndarray:
    p = run_dir / "processed_mask.npy"
    if not p.exists():
        return np.ones((int(n_windows),), dtype=bool)
    m = np.load(p).astype(bool)
    if m.shape != (int(n_windows),):
        raise ValueError(f"processed_mask shape mismatch: {m.shape} vs ({n_windows},)")
    return m


def _discover_layers(run_dir: Path) -> List[str]:
    p = run_dir / "summary_by_layer.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary_by_layer.tsv: {p}")
    rows = _read_tsv(p)
    layers = []
    for r in rows:
        layers.append(str(r["layer"]))
    return layers


def _read_channel_metrics(run_dir: Path, layer: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    safe = _sanitize_layer_name(layer)
    p = run_dir / f"{safe}.channel_metrics.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Missing channel metrics for layer '{layer}': {p}")
    rows = _read_tsv(p)

    # Columns we care about; tolerate missing.
    numeric_cols = [
        "mean",
        "std",
        "q99",
        "q999",
        "tail_freq_z",
        "tail_ratio_top1pct",
        "tail_ratio_qhigh_qlow",
        "tail_ratio_qhigh_median",
        "super_score",
    ]

    channels: List[int] = []
    cols: Dict[str, List[float]] = {c: [] for c in numeric_cols}
    for r in rows:
        ch = int(r["channel"])
        channels.append(ch)
        for c in numeric_cols:
            cols[c].append(_parse_float(r.get(c, "")))

    # Ensure channel order is increasing.
    order = np.argsort(np.asarray(channels, dtype=np.int64))
    ch_arr = np.asarray(channels, dtype=np.int32)[order]

    out_cols: Dict[str, np.ndarray] = {}
    for c in numeric_cols:
        out_cols[c] = np.asarray(cols[c], dtype=np.float64)[order]

    return ch_arr, out_cols, numeric_cols


def _concentration_curve(mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mass, dtype=np.float64)
    m = np.where(np.isfinite(m), m, 0.0)
    m = np.maximum(m, 0.0)
    m_sorted = np.sort(m)[::-1]
    total = float(m_sorted.sum())
    n = int(m_sorted.size)
    if n == 0 or total <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    frac_channels = (np.arange(1, n + 1, dtype=np.float64) / float(n))
    cum_mass = np.cumsum(m_sorted) / total
    return frac_channels, cum_mass


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    v = np.asarray(values)
    n = int(v.shape[0])
    k = int(k)
    if k >= n:
        return np.argsort(v)[::-1]
    idx = np.argpartition(v, -k)[-k:]
    idx = idx[np.argsort(v[idx])[::-1]]
    return idx


def _enrichment_counts(
    region_set_of_window: Sequence[str],
    top_window_indices: Sequence[int],
    total_counts_by_set: Dict[str, int],
    *,
    control_set: str,
) -> Dict[str, float | int | str]:
    # Count hits in top windows.
    hit_counts: Dict[str, int] = {}
    for wi in top_window_indices:
        rs = str(region_set_of_window[int(wi)])
        hit_counts[rs] = int(hit_counts.get(rs, 0) + 1)

    K = int(len(list(top_window_indices)))
    N_total = int(sum(total_counts_by_set.values()))

    out: Dict[str, float | int | str] = {
        "K_top": K,
        "N_total": N_total,
        "control_set": str(control_set),
    }

    # log2 fold enrichment by set: (hits/K) / (total/N)
    for rs, total in total_counts_by_set.items():
        hits = int(hit_counts.get(rs, 0))
        out[f"hits_{rs}"] = hits
        out[f"total_{rs}"] = int(total)
        p_hit = (hits / K) if K > 0 else float("nan")
        p_bg = (total / N_total) if N_total > 0 else float("nan")
        fe = (p_hit / p_bg) if (p_hit is not None and p_bg and p_bg > 0) else float("nan")
        out[f"log2FE_{rs}"] = _log2_safe(float(fe))

    # Odds ratio vs control (2x2): hits_rs vs hits_control.
    ctrl_total = int(total_counts_by_set.get(control_set, 0))
    ctrl_hits = int(hit_counts.get(control_set, 0))
    out["hits_control"] = ctrl_hits
    out["total_control"] = ctrl_total

    eps = 0.5  # Haldane-Anscombe correction for stability
    for rs, total in total_counts_by_set.items():
        if rs == control_set:
            continue
        hits = int(hit_counts.get(rs, 0))
        a = hits + eps
        b = (K - hits) + eps
        c = ctrl_hits + eps
        d = (K - ctrl_hits) + eps
        or_ = (a / b) / (c / d) if (b > 0 and d > 0) else float("nan")
        out[f"OR_{rs}_vs_{control_set}"] = float(or_)
        out[f"log2OR_{rs}_vs_{control_set}"] = _log2_safe(float(or_))

    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-track postprocess for superactivations scan")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--control_set", type=str, default="random", help="Control region_set name for ORs")
    p.add_argument("--top_general_channels", type=int, default=50)
    p.add_argument("--top_spiky_channels", type=int, default=50)
    p.add_argument(
        "--spiky_metric",
        type=str,
        default="tail_ratio_qhigh_median",
        choices=["tail_ratio_qhigh_median", "tail_ratio_qhigh_qlow", "both"],
        help="Metric to rank spiky channels.",
    )
    p.add_argument("--top_windows_per_channel", type=int, default=200)
    p.add_argument("--write_plots", action="store_true", help="Write concentration curve PNGs if matplotlib is available")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    layers = _discover_layers(run_dir)
    windows = _read_windows_meta(run_dir)
    n_windows = int(len(windows))
    processed = _read_processed_mask(run_dir, n_windows)
    idx_proc = np.where(processed)[0].astype(np.int64)
    if idx_proc.size == 0:
        raise SystemExit("No processed windows found (processed_mask has 0 trues)")

    region_set_all = [w.region_set for w in windows]
    proc_region_set = np.asarray([region_set_all[int(i)] for i in idx_proc], dtype=object)
    unique_sets = sorted(set(proc_region_set.tolist()))
    total_counts_by_set = {rs: int(np.sum(proc_region_set == rs)) for rs in unique_sets}
    if str(args.control_set) not in total_counts_by_set:
        # Fall back to the largest set if 'random' isn't present.
        ctrl = max(total_counts_by_set.items(), key=lambda kv: kv[1])[0]
        args.control_set = str(ctrl)

    out_base = run_dir / "two_track"
    out_base.mkdir(parents=True, exist_ok=True)

    # Track1 summary (one row per layer)
    track1_summary_rows: List[dict] = []
    track2_summary_rows: List[dict] = []

    for layer in layers:
        ch, cols, _ = _read_channel_metrics(run_dir, layer)
        n_channels = int(ch.size)

        # Use abs(mean) as the non-negative mass proxy.
        mean = cols.get("mean", np.full((n_channels,), np.nan, dtype=np.float64))
        mass = np.abs(mean)
        mass = np.where(np.isfinite(mass), mass, 0.0)

        order_mass = np.argsort(mass)[::-1]
        mass_sorted = mass[order_mass]
        frac_channels, cum_mass = _concentration_curve(mass)

        mass_fracs = _mass_fractions(mass_sorted, fracs=[0.01, 0.05, 0.10])
        g = _gini(np.maximum(mass, 0.0))
        h = _hhi(np.maximum(mass, 0.0))

        track1_summary = {
            "layer": layer,
            "n_windows_processed": int(idx_proc.size),
            "n_channels": n_channels,
            "mass_proxy": "abs(mean)",
            "gini": float(g),
            "hhi": float(h),
            **mass_fracs,
        }
        track1_summary_rows.append(track1_summary)

        # Track1: top-by-mean table
        topg = int(args.top_general_channels)
        topg_idx = order_mass[:topg]
        topg_rows: List[dict] = []
        for rank, i in enumerate(topg_idx.tolist(), start=1):
            topg_rows.append(
                {
                    "rank": rank,
                    "layer": layer,
                    "channel": int(ch[int(i)]),
                    "mean": float(cols.get("mean", np.nan)[int(i)]),
                    "q99": float(cols.get("q99", np.nan)[int(i)]),
                    "q999": float(cols.get("q999", np.nan)[int(i)]),
                    "tail_ratio_qhigh_median": float(cols.get("tail_ratio_qhigh_median", np.nan)[int(i)]),
                    "tail_ratio_qhigh_qlow": float(cols.get("tail_ratio_qhigh_qlow", np.nan)[int(i)]),
                }
            )
        _write_tsv(
            out_base / "track1_general" / f"{_sanitize_layer_name(layer)}.top_channels_by_mean.tsv",
            topg_rows,
            fieldnames=list(topg_rows[0].keys()) if topg_rows else ["rank", "layer", "channel"],
        )

        # Track1: concentration curve TSV (+ optional plot)
        curve_rows = [
            {"frac_channels": float(x), "cum_mass_frac": float(y)}
            for x, y in zip(frac_channels.tolist(), cum_mass.tolist())
        ]
        _write_tsv(
            out_base / "track1_general" / f"{_sanitize_layer_name(layer)}.concentration_curve.tsv",
            curve_rows,
            fieldnames=["frac_channels", "cum_mass_frac"],
        )

        if bool(args.write_plots):
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                plt.figure(figsize=(5.2, 4.0))
                plt.plot(frac_channels, cum_mass, lw=2)
                plt.xlabel("Fraction of channels")
                plt.ylabel("Cumulative mass fraction (abs(mean))")
                plt.title(f"Track1 concentration: {layer}")
                plt.grid(True, alpha=0.3)
                out_png = out_base / "track1_general" / f"{_sanitize_layer_name(layer)}.concentration_curve.png"
                plt.tight_layout()
                plt.savefig(out_png, dpi=160)
                plt.close()
            except Exception:
                pass

        # Track1: region-set bias for these top general channels
        try:
            mm = _open_layer_memmap(run_dir, layer, n_windows=n_windows, n_channels=n_channels)
            a = np.asarray(mm)[idx_proc, :]

            by_set_means: Dict[str, np.ndarray] = {}
            for rs in unique_sets:
                mask = proc_region_set == rs
                if int(mask.sum()) == 0:
                    continue
                by_set_means[rs] = a[mask, :].mean(axis=0).astype(np.float64)

            ctrl = str(args.control_set)
            ctrl_mean = by_set_means.get(ctrl, np.full((n_channels,), np.nan, dtype=np.float64))
            eps = 1e-12

            bias_rows: List[dict] = []
            for rank, i in enumerate(topg_idx.tolist(), start=1):
                row: Dict[str, object] = {
                    "rank": rank,
                    "layer": layer,
                    "channel": int(ch[int(i)]),
                    "mean_all": float(mean[int(i)]),
                    "control_set": ctrl,
                }
                for rs in unique_sets:
                    m_rs = by_set_means.get(rs)
                    if m_rs is None:
                        continue
                    v = float(m_rs[int(i)])
                    row[f"mean_{rs}"] = v
                    # ratio of abs means for stability
                    v_abs = abs(v)
                    v_ctrl_abs = abs(float(ctrl_mean[int(i)]))
                    row[f"log2_abs_ratio_{rs}_vs_{ctrl}"] = _log2_safe((v_abs + eps) / (v_ctrl_abs + eps))
                bias_rows.append(row)

            _write_tsv(
                out_base / "track1_general" / f"{_sanitize_layer_name(layer)}.region_set_bias_top_mean.tsv",
                bias_rows,
                fieldnames=sorted({k for r in bias_rows for k in r.keys()}) if bias_rows else ["rank", "layer", "channel"],
            )
        except FileNotFoundError:
            # If a run was done without memmaps, skip region bias.
            pass

        # Track2: spiky channels (optionally for both metrics)
        spiky_metrics: List[str]
        if str(args.spiky_metric) == "both":
            spiky_metrics = ["tail_ratio_qhigh_median", "tail_ratio_qhigh_qlow"]
        else:
            spiky_metrics = [str(args.spiky_metric)]

        for spiky_metric in spiky_metrics:
            spiky = cols.get(spiky_metric)
            if spiky is None:
                continue
            spiky = np.asarray(spiky, dtype=np.float64)
            spiky = np.where(np.isfinite(spiky), spiky, -np.inf)
            order_spiky = np.argsort(spiky)[::-1]

            tops = int(args.top_spiky_channels)
            tops_idx = order_spiky[:tops]

            spiky_rows: List[dict] = []
            for rank, i in enumerate(tops_idx.tolist(), start=1):
                spiky_rows.append(
                    {
                        "rank": rank,
                        "layer": layer,
                        "channel": int(ch[int(i)]),
                        spiky_metric: float(spiky[int(i)]),
                        "mean": float(cols.get("mean", np.nan)[int(i)]),
                        "q99": float(cols.get("q99", np.nan)[int(i)]),
                        "q999": float(cols.get("q999", np.nan)[int(i)]),
                        "tail_ratio_qhigh_median": float(cols.get("tail_ratio_qhigh_median", np.nan)[int(i)]),
                        "tail_ratio_qhigh_qlow": float(cols.get("tail_ratio_qhigh_qlow", np.nan)[int(i)]),
                    }
                )

            _write_tsv(
                out_base / "track2_spiky" / f"{_sanitize_layer_name(layer)}.top_channels_by_{spiky_metric}.tsv",
                spiky_rows,
                fieldnames=list(spiky_rows[0].keys()) if spiky_rows else ["rank", "layer", "channel"],
            )

            # Track2: enrichment of extreme windows for these spiky channels
            try:
                mm = _open_layer_memmap(run_dir, layer, n_windows=n_windows, n_channels=n_channels)
                a = np.asarray(mm)[idx_proc, :]
                proc_global_idx = idx_proc

                K = int(args.top_windows_per_channel)
                enrich_rows: List[dict] = []
                for i in tops_idx.tolist():
                    ch_id = int(ch[int(i)])
                    col = a[:, int(ch_id)]
                    local_top = _topk_indices(col, k=K)
                    global_top = proc_global_idx[local_top]

                    enr = _enrichment_counts(
                        region_set_of_window=region_set_all,
                        top_window_indices=global_top.tolist(),
                        total_counts_by_set=total_counts_by_set,
                        control_set=str(args.control_set),
                    )
                    enr_row: Dict[str, object] = {
                        "layer": layer,
                        "channel": ch_id,
                        "spiky_metric": spiky_metric,
                        "spiky_value": float(spiky[int(i)]),
                    }
                    enr_row.update(enr)
                    enrich_rows.append(enr_row)

                if enrich_rows:
                    _write_tsv(
                        out_base / "track2_spiky" / f"{_sanitize_layer_name(layer)}.enrichment_top_windows_by_{spiky_metric}.tsv",
                        enrich_rows,
                        fieldnames=sorted({k for r in enrich_rows for k in r.keys()}),
                    )

                track2_summary_rows.append(
                    {
                        "layer": layer,
                        "spiky_metric": spiky_metric,
                        "n_channels": n_channels,
                        "top_spiky_channels": int(tops),
                        "top_windows_per_channel": int(K),
                        "control_set": str(args.control_set),
                        "region_sets": ",".join(unique_sets),
                    }
                )
            except FileNotFoundError:
                pass

    if track1_summary_rows:
        _write_tsv(
            out_base / "track1_general" / "track1_summary_by_layer.tsv",
            track1_summary_rows,
            fieldnames=sorted({k for r in track1_summary_rows for k in r.keys()}),
        )

    if track2_summary_rows:
        _write_tsv(
            out_base / "track2_spiky" / "track2_summary_by_layer.tsv",
            track2_summary_rows,
            fieldnames=sorted({k for r in track2_summary_rows for k in r.keys()}),
        )

    # Also write a small manifest for convenience.
    manifest = {
        "run_dir": str(run_dir),
        "layers": list(layers),
        "n_windows": int(n_windows),
        "n_windows_processed": int(idx_proc.size),
        "region_sets": unique_sets,
        "control_set": str(args.control_set),
        "track1": {
            "top_general_channels": int(args.top_general_channels),
            "mass_proxy": "abs(mean)",
        },
        "track2": {
            "top_spiky_channels": int(args.top_spiky_channels),
            "spiky_metric": str(args.spiky_metric),
            "top_windows_per_channel": int(args.top_windows_per_channel),
        },
    }
    (out_base / "two_track_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote two-track outputs to: {out_base}")


if __name__ == "__main__":
    main(sys.argv[1:])
