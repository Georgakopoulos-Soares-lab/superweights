#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout}")


def _make_env(python: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env_bin = str(Path(python).resolve().parent)
    env["PATH"] = f"{env_bin}:{env.get('PATH', '')}" if env.get("PATH") else env_bin
    return env


def _compute_p_emp(spec_parquet: Path, target_channel: int) -> dict:
    df = pd.read_parquet(spec_parquet)

    metric = "effect_act_ref_wt_minus_motif"
    if metric not in df.columns:
        candidates = [c for c in df.columns if "effect_act" in c and "wt_minus_motif" in c]
        if not candidates:
            raise ValueError(f"No activation effect metric columns found in {spec_parquet}")
        metric = candidates[0]

    per_ch = (
        df.groupby("channel", as_index=False)[metric]
        .median()
        .rename(columns={metric: "median_effect"})
        .assign(abs_median_effect=lambda d: d["median_effect"].abs())
        .sort_values("abs_median_effect", ascending=False)
    )

    per_ch["rank_by_abs_median"] = np.arange(1, len(per_ch) + 1)

    tgt = per_ch[per_ch["channel"] == int(target_channel)]
    if len(tgt) != 1:
        raise ValueError(f"Target channel {target_channel} not found in {spec_parquet}")

    main_abs = float(tgt.iloc[0]["abs_median_effect"])
    rand_abs = per_ch[per_ch["channel"] != int(target_channel)]["abs_median_effect"].to_numpy()
    p_emp = float((np.sum(rand_abs >= main_abs) + 1) / (len(rand_abs) + 1))

    return {
        "metric": metric,
        "target_abs_median": main_abs,
        "p_emp": p_emp,
        "rank": int(tgt.iloc[0]["rank_by_abs_median"]),
        "n_channels": int(len(per_ch)),
    }


def _plot_p_emp(values: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 3.0))
    plt.hist(values, bins=max(3, len(values)), alpha=0.9)
    plt.ylim(0, None)
    plt.title(title)
    plt.xlabel("empirical p")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeat specificity with multiple independent random-channel draws and report p_emp distribution.")
    p.add_argument("--python", default=os.environ.get("PYTHON", "/scratch/10906/arisk/envs/superweights-borzoi/bin/python"))

    p.add_argument("--layer", default="horizontal_conv1.conv_layer")
    p.add_argument("--channel", type=int, required=True)
    p.add_argument("--motif_id", required=True)

    p.add_argument("--fimo_tsv", required=True)
    p.add_argument("--map_tsv", required=True)
    p.add_argument("--genome_fasta", default="data/hg38.fa")
    p.add_argument("--streme_txt", required=True)

    p.add_argument("--pwm_id", default="MA0062.2")
    p.add_argument("--pwm_topk", type=int, default=3)
    p.add_argument("--max_hits", type=int, default=200)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n_reps", type=int, default=3)

    p.add_argument("--width", type=int, default=1536, help="Channel width of the layer for random draws")

    p.add_argument("--outdir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_env = _make_env(str(args.python))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rep in range(int(args.n_reps)):
        rep_seed = int(args.seed) + rep * 997
        rng = np.random.default_rng(rep_seed)

        random = rng.choice([c for c in range(int(args.width)) if c != int(args.channel)], size=50, replace=False)
        channels_csv = ",".join([str(int(args.channel)), *map(str, random.tolist())])

        rep_dir = outdir / f"rep{rep+1:02d}_seed{rep_seed}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        _run(
            [
                str(args.python),
                "scripts/motif_site_perturbation.py",
                "--fimo_tsv",
                str(args.fimo_tsv),
                "--map_tsv",
                str(args.map_tsv),
                "--genome_fasta",
                str(args.genome_fasta),
                "--motif_id",
                str(args.motif_id),
                "--layer",
                str(args.layer),
                "--channels",
                channels_csv,
                "--mut_mode",
                "pwm_minimize",
                "--pwm_id",
                str(args.pwm_id),
                "--pwm_topk",
                str(int(args.pwm_topk)),
                "--audit_pwm",
                "--streme_txt",
                str(args.streme_txt),
                "--max_hits",
                str(int(args.max_hits)),
                "--outdir",
                str(rep_dir),
            ],
            cwd=Path.cwd(),
            env=base_env,
        )

        spec_parquet = next(rep_dir.glob(f"perturb_{args.motif_id}_{args.layer}_*.parquet"))
        stats = _compute_p_emp(spec_parquet, int(args.channel))
        stats.update({"rep": rep + 1, "seed": rep_seed, "spec_parquet": str(spec_parquet)})
        rows.append(stats)

    df = pd.DataFrame(rows).sort_values("rep")
    out_tsv = outdir / "p_emp_resamples.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)

    out_png = outdir / "p_emp_resamples.hist.png"
    _plot_p_emp(df["p_emp"].to_numpy(), out_png, title=f"p_emp resamples (n={len(df)}) ch={args.channel} motif={args.motif_id}")

    manifest = {
        "channel": int(args.channel),
        "layer": str(args.layer),
        "motif_id": str(args.motif_id),
        "n_reps": int(args.n_reps),
        "seed": int(args.seed),
        "out_tsv": str(out_tsv),
        "out_png": str(out_png),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
