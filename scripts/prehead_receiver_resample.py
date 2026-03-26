#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class RunSpec:
    rep: int
    seed: int
    channels: List[int]
    outdir: Path
    parquet: Path


def _sample_channels(out_channels: int, *, fixed: int, n_rand: int, seed: int) -> List[int]:
    rng = np.random.default_rng(int(seed))
    pool = [i for i in range(int(out_channels)) if i != int(fixed)]
    picks = rng.choice(pool, size=int(n_rand), replace=False)
    return [int(fixed)] + sorted(map(int, picks))


def _early_effect_series(early_parquet: Path, *, layer: str, channel: int, metric: str) -> pd.Series:
    df = pd.read_parquet(early_parquet)
    need = {"layer", "channel", "fimo_sequence_name", metric}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {early_parquet}: {sorted(missing)}")
    df = df[(df["layer"].astype(str) == str(layer)) & (df["channel"].astype(int) == int(channel))].copy()
    if df.empty:
        raise ValueError(f"No early rows for layer={layer} channel={channel} in {early_parquet}")
    df = df.sort_values(["fimo_sequence_name"]).drop_duplicates(subset=["fimo_sequence_name"], keep="first")
    return pd.Series(df[metric].astype(float).values, index=df["fimo_sequence_name"].astype(str).values)


def _corr_rank_vs_random(
    df_late: pd.DataFrame,
    early: pd.Series,
    *,
    receiver_channel: int,
    metric: str,
) -> Tuple[pd.DataFrame, dict]:
    rows = []
    for ch, g in df_late.groupby("channel"):
        dfm = g[["fimo_sequence_name", metric]].rename(columns={metric: "late_eff"}).merge(
            early.rename("early_eff"), left_on="fimo_sequence_name", right_index=True, how="inner"
        )
        x = dfm["early_eff"].astype(float).to_numpy()
        y = dfm["late_eff"].astype(float).to_numpy()
        m = ~(np.isnan(x) | np.isnan(y))
        x = x[m]
        y = y[m]
        if len(x) < 25:
            rho = float("nan")
            pv = float("nan")
        else:
            rho, pv = spearmanr(x, y)
            rho = float(rho)
            pv = float(pv)
        rows.append((int(ch), rho, pv, int(len(x))))

    tab = pd.DataFrame(rows, columns=["channel", "rho", "p", "n"])
    tab["abs_rho"] = tab["rho"].abs()
    tab = tab.sort_values("abs_rho", ascending=False).reset_index(drop=True)
    tab["rank_abs_rho"] = np.arange(1, len(tab) + 1)

    if int(receiver_channel) not in set(tab["channel"].astype(int).tolist()):
        raise ValueError(f"receiver_channel={receiver_channel} not present in late df")

    recv = tab[tab.channel == int(receiver_channel)].iloc[0]
    rand = tab[tab.channel != int(receiver_channel)]
    abs_recv = float(recv["abs_rho"]) if not math.isnan(float(recv["abs_rho"])) else float("nan")
    rand_abs = rand["abs_rho"].to_numpy(dtype=float)
    p_emp = float("nan")
    if not math.isnan(abs_recv):
        p_emp = (1 + int(np.sum(rand_abs >= abs_recv))) / (1 + int(len(rand_abs)))

    summary = {
        "receiver_channel": int(receiver_channel),
        "receiver_rho": float(recv["rho"]),
        "receiver_rank_abs_rho": int(recv["rank_abs_rho"]),
        "n_channels": int(len(tab)),
        "p_emp_abs_rho": float(p_emp),
        "top_channel": int(tab.iloc[0]["channel"]),
        "top_rho": float(tab.iloc[0]["rho"]),
    }
    return tab, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resample pre-head channels and test stability of a receiver channel by |rho|.")

    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--n_rand", type=int, default=50)
    p.add_argument("--seed0", type=int, default=0)

    p.add_argument("--out_channels", type=int, default=1920, help="#channels at pre-head layer")
    p.add_argument("--receiver_channel", type=int, default=1181)

    p.add_argument("--layer", default="final_joined_convs.0.conv_layer")

    p.add_argument("--fimo_tsv", required=True)
    p.add_argument("--map_tsv", required=True)
    p.add_argument("--genome_fasta", required=True)
    p.add_argument("--motif_id", default="6-CGGAAG")
    p.add_argument("--max_hits", type=int, default=300)
    p.add_argument("--mut_mode", default="pwm_minimize")
    p.add_argument("--flank_bp", type=int, default=211)

    p.add_argument("--early_parquet", required=True)
    p.add_argument("--early_layer", default="horizontal_conv1.conv_layer")
    p.add_argument("--early_channel", type=int, default=130)
    p.add_argument("--metric", default="effect_act_ref_wt_minus_motif")

    p.add_argument("--motif_script", default="scripts/motif_site_perturbation.py")
    p.add_argument("--base_outdir", default="results/motifs/perturb_m6_top300_pwmmin_fixed_followup2_prehead_resample")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_outdir = Path(args.base_outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    early = _early_effect_series(Path(args.early_parquet), layer=args.early_layer, channel=args.early_channel, metric=args.metric)

    run_specs: List[RunSpec] = []
    for rep in range(int(args.reps)):
        seed = int(args.seed0) + rep
        chs = _sample_channels(int(args.out_channels), fixed=int(args.receiver_channel), n_rand=int(args.n_rand), seed=seed)
        outdir = base_outdir / f"rep{rep:02d}_seed{seed}"
        outdir.mkdir(parents=True, exist_ok=True)
        # The perturbation script uses a hash tag for channel list; find it post-run.
        run_specs.append(
            RunSpec(
                rep=rep,
                seed=seed,
                channels=chs,
                outdir=outdir,
                parquet=Path(),
            )
        )

    summaries = []

    for spec in run_specs:
        chs_str = ",".join(map(str, spec.channels))
        cmd = [
            sys.executable,
            str(args.motif_script),
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
            chs_str,
            "--max_hits",
            str(int(args.max_hits)),
            "--mut_mode",
            str(args.mut_mode),
            "--audit_pwm",
            "--flank_bp",
            str(int(args.flank_bp)),
            "--outdir",
            str(spec.outdir),
        ]
        print("\n[rep]", spec.rep, "seed", spec.seed, "running motif_site_perturbation ...")
        subprocess.run(cmd, check=True)

        # Find the parquet written.
        parqs = sorted(spec.outdir.glob("perturb_*.parquet"))
        if len(parqs) != 1:
            raise RuntimeError(f"Expected 1 parquet in {spec.outdir}, found {len(parqs)}")
        parquet = parqs[0]

        df_late = pd.read_parquet(parquet)
        tab, summ = _corr_rank_vs_random(df_late, early, receiver_channel=int(args.receiver_channel), metric=str(args.metric))

        out_tab = spec.outdir / "receiver_corr_table.parquet"
        out_json = spec.outdir / "receiver_corr_summary.json"
        tab.to_parquet(out_tab, index=False)
        out_json.write_text(json.dumps(summ, indent=2) + "\n")

        summ_row = {
            "rep": int(spec.rep),
            "seed": int(spec.seed),
            **summ,
            "parquet": str(parquet),
        }
        summaries.append(summ_row)
        print("[rep]", spec.rep, "receiver rank", summ["receiver_rank_abs_rho"], "p_emp", summ["p_emp_abs_rho"], "rho", summ["receiver_rho"])

    df_sum = pd.DataFrame(summaries).sort_values(["rep"]).reset_index(drop=True)
    out_sum = base_outdir / "receiver_resample_summary.parquet"
    out_sum_tsv = base_outdir / "receiver_resample_summary.tsv"
    df_sum.to_parquet(out_sum, index=False)
    df_sum.to_csv(out_sum_tsv, sep="\t", index=False)

    print("\nWrote", out_sum)
    print("Wrote", out_sum_tsv)
    print(df_sum[["rep", "seed", "receiver_rank_abs_rho", "p_emp_abs_rho", "receiver_rho", "top_channel", "top_rho"]])


if __name__ == "__main__":
    main()
