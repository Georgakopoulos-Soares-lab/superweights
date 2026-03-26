#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


JASPAR_MEME_DEFAULT = "/scratch/10906/arisk/envs/superweights-borzoi/share/meme-5.5.9/doc/examples/example-datasets/JASPAR2018_CORE_non-redundant.meme"


@dataclass
class RunConfig:
    channel: int
    layer: str
    motif_query_id: Optional[str]
    pwm_id: str
    max_hits: int
    flank_bp: int
    window_len: int
    pad_bins_grid: str
    shift_bins: str
    track_indices: str
    out_root: Path
    seq_cache: Path
    posmaps: Path
    python: str
    jaspar_meme: str
    seed: int


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout}")


def _make_base_env(python: str) -> dict[str, str]:
    """Return an env with PATH including the conda env bin for `python`.

    We often call `/path/to/env/bin/python` directly, which does not imply that
    `/path/to/env/bin` is on PATH in the current shell. MEME-suite tools
    (streme/tomtom/fimo) live in that same bin directory.
    """

    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    env_bin = str(Path(python).resolve().parent)
    env["PATH"] = f"{env_bin}:{env.get('PATH', '')}" if env.get("PATH") else env_bin
    return env


def _write_histogram(values: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3.2))
    plt.hist(values, bins=25, alpha=0.9)
    plt.axvline(0, color="k", linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end motif pipeline for early-layer superweight channels: "
            "extract windows -> sanitize FASTA -> STREME+TOMTOM -> choose lead motif -> FIMO -> perturb (PWM-min) -> specificity."
        )
    )
    p.add_argument("--channel", type=int, required=True)
    p.add_argument("--layer", default="horizontal_conv1.conv_layer")

    p.add_argument("--posmaps", required=True, help="Posmaps parquet for this channel")
    p.add_argument("--seq_cache", default="results/full_v1/seq_cache.parquet")

    p.add_argument("--out_root", default="results/motifs/signed_task")

    p.add_argument("--flank_bp", type=int, default=211)
    p.add_argument("--window_len", type=int, default=211)
    p.add_argument("--max_hits", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument(
        "--pwm_id",
        default="auto",
        help=(
            "PWM id for pwm_minimize (JASPAR MEME DB). "
            "Use 'auto' to take the Tomtom best-hit Target_ID for the selected motif (recommended for intermediate layers)."
        ),
    )
    p.add_argument("--jaspar_meme", default=JASPAR_MEME_DEFAULT)

    p.add_argument(
        "--prefer_family",
        default="ETS",
        help="When selecting a motif to perturb, prefer this Tomtom family label if possible; set to 'any' to disable.",
    )

    p.add_argument("--pad_bins_grid", default="2,4,10")
    p.add_argument(
        "--shift_bins",
        default="-4,-2,-1,0,1,2,4",
        help=(
            "Comma-separated OUTPUT-bin shifts. If the value starts with '-', pass as --shift_bins=-4,-2,... "
            "(argparse may otherwise interpret it as flags)."
        ),
    )
    p.add_argument("--track_indices", default="260,262,784")

    p.add_argument(
        "--motif_query_id",
        default=None,
        help="Override motif Query_ID (e.g. 4-AASAGGAAG). If not set, pick top ETS-family Tomtom match.",
    )

    p.add_argument("--python", default=os.environ.get("PYTHON", "/scratch/10906/arisk/envs/superweights-borzoi/bin/python"))

    return p.parse_args()


def _pick_lead_motif(tomtom_tsv: Path, *, python: str, jaspar_meme: str, out_tsv: Path, env: dict[str, str]) -> str:
    cmd = [
        python,
        "scripts/pick_lead_motifs.py",
        "--tomtom_tsv",
        str(tomtom_tsv),
        "--jaspar_meme",
        str(jaspar_meme),
        "--top_k",
        "20",
        "--out_tsv",
        str(out_tsv),
    ]
    _run(cmd, cwd=Path.cwd(), env=env)
    df = pd.read_csv(out_tsv, sep="\t")
    # Prefer ETS family if present, else best overall
    if "family" in df.columns:
        ets = df[df["family"] == "ETS"]
        if len(ets):
            return str(ets.iloc[0]["Query_ID"])
    return str(df.iloc[0]["Query_ID"])


def _load_jaspar_ids(jaspar_meme: Path) -> set[str]:
    ids: set[str] = set()
    if not jaspar_meme.exists():
        return ids
    with jaspar_meme.open() as f:
        for line in f:
            if line.startswith("MOTIF "):
                parts = line.strip().split()
                if len(parts) >= 2:
                    ids.add(parts[1])
    return ids


def _best_tomtom_hit_for_query(tomtom_tsv: Path, query_id: str) -> dict:
    df = pd.read_csv(tomtom_tsv, sep="\t", comment="#")
    if df.empty:
        return {}
    sub = df[df["Query_ID"] == str(query_id)].copy()
    if sub.empty:
        return {}
    sub = sub.sort_values(["q-value", "E-value", "p-value"], ascending=[True, True, True]).head(1)
    r = sub.iloc[0].to_dict()
    return {k: r.get(k) for k in ["Target_ID", "q-value", "E-value", "p-value", "Orientation", "Target_consensus"] if k in r}


def _choose_motif_with_hits(
    *,
    lead_motifs_tsv: Path,
    fimo_tsv: Path,
    preferred_motif: Optional[str],
    prefer_family: str = "ETS",
) -> str:
    """Pick a motif_id that has FIMO hits.

    Strategy:
    - If preferred_motif is provided and has hits, use it.
    - Else, among lead motifs (Tomtom-annotated), pick one with the most unique sequences in FIMO,
      preferring the requested family.
    - Else, fall back to the overall most-hit motif in FIMO.
    """

    fimo = pd.read_csv(fimo_tsv, sep="\t")
    if fimo.empty:
        raise ValueError(f"Empty FIMO TSV: {fimo_tsv}")
    if "motif_id" not in fimo.columns or "sequence_name" not in fimo.columns:
        raise ValueError(f"Unexpected FIMO columns in {fimo_tsv}: {list(fimo.columns)}")

    counts = (
        fimo.groupby("motif_id", as_index=False)["sequence_name"]
        .nunique()
        .rename(columns={"sequence_name": "n_sequences"})
        .sort_values("n_sequences", ascending=False)
    )

    if preferred_motif:
        row = counts[counts["motif_id"] == str(preferred_motif)]
        if len(row) and int(row.iloc[0]["n_sequences"]) > 0:
            return str(preferred_motif)

    lead = pd.read_csv(lead_motifs_tsv, sep="\t") if lead_motifs_tsv.exists() else pd.DataFrame()
    prefer_family = str(prefer_family or "").strip()
    prefer_any = (prefer_family == "") or (prefer_family.lower() == "any")

    if not lead.empty and "Query_ID" in lead.columns:
        lead = lead.rename(columns={"Query_ID": "motif_id"})
        lead = lead.merge(counts, on="motif_id", how="left").fillna({"n_sequences": 0})
        if (not prefer_any) and ("family" in lead.columns):
            fam = lead[lead["family"] == prefer_family].copy()
            if len(fam):
                fam = fam.sort_values(["n_sequences"], ascending=[False])
                if int(fam.iloc[0]["n_sequences"]) > 0:
                    return str(fam.iloc[0]["motif_id"])
        lead = lead.sort_values(["n_sequences"], ascending=[False])
        if int(lead.iloc[0]["n_sequences"]) > 0:
            return str(lead.iloc[0]["motif_id"])

    # final fallback: most-hit motif
    return str(counts.iloc[0]["motif_id"])


def _write_pwm_audit_table(perturb_parquet: Path, out_tsv: Path) -> None:
    df = pd.read_parquet(perturb_parquet)
    cols = [
        "fimo_sequence_name",
        "variant_id",
        "chrom",
        "pos1",
        "fimo_start",
        "fimo_stop",
        "fimo_strand",
        "fimo_score",
        "fimo_p",
        "fimo_q",
        "fimo_matched_sequence",
        "pwm_id",
        "streme_pwm_ref_fimo_wt",
        "streme_pwm_ref_fimo_mut",
        "streme_pwm_ref_fimo_delta_wt_minus_mut",
        "jaspar_pwm_ref_best_wt",
        "jaspar_pwm_ref_best_mut",
        "jaspar_pwm_ref_best_delta_wt_minus_mut",
        "jaspar_pwm_ref_best_off0_wt",
        "jaspar_pwm_ref_best_off0_mut",
        "jaspar_pwm_ref_best_strand_wt",
        "jaspar_pwm_ref_best_strand_mut",
    ]
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, [c for c in cols if c in df.columns]].to_csv(out_tsv, sep="\t", index=False)


def _compute_specificity_table(spec_parquet: Path, target_channel: int, out_tsv: Path, out_png: Path) -> dict:
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
        raise ValueError(f"Target channel {target_channel} not found in specificity parquet")

    main_abs = float(tgt.iloc[0]["abs_median_effect"])
    rand_abs = per_ch[per_ch["channel"] != int(target_channel)]["abs_median_effect"].to_numpy()
    p_emp = float((np.sum(rand_abs >= main_abs) + 1) / (len(rand_abs) + 1))

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    per_ch.to_csv(out_tsv, sep="\t", index=False)

    _write_histogram(
        rand_abs,
        out_png,
        title=f"Specificity null |median effect| (n={len(rand_abs)}); target={target_channel}; p_emp={p_emp:.3g}",
    )

    return {
        "metric": metric,
        "target_channel": int(target_channel),
        "target_abs_median": main_abs,
        "p_emp": p_emp,
        "n_random": int(len(rand_abs)),
        "rank": int(tgt.iloc[0]["rank_by_abs_median"]),
        "n_channels": int(len(per_ch)),
    }


def main() -> None:
    args = parse_args()

    cfg = RunConfig(
        channel=int(args.channel),
        layer=str(args.layer),
        motif_query_id=(str(args.motif_query_id) if args.motif_query_id else None),
        pwm_id=str(args.pwm_id),
        max_hits=int(args.max_hits),
        flank_bp=int(args.flank_bp),
        window_len=int(args.window_len),
        pad_bins_grid=str(args.pad_bins_grid),
        shift_bins=str(args.shift_bins),
        track_indices=str(args.track_indices),
        out_root=Path(args.out_root),
        seq_cache=Path(args.seq_cache),
        posmaps=Path(args.posmaps),
        python=str(args.python),
        jaspar_meme=str(args.jaspar_meme),
        seed=int(args.seed),
    )

    base_env = _make_base_env(cfg.python)
    jaspar_ids = _load_jaspar_ids(Path(cfg.jaspar_meme))

    # Per-channel directory
    outdir = cfg.out_root / f"{cfg.layer.replace('.', '_')}_ch{cfg.channel}"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) extract windows
    pos_fa = outdir / f"near_variant.pos.{cfg.window_len}.fa"
    neg_fa = outdir / f"near_variant.neg.{cfg.window_len}.fa"

    _run(
        [
            cfg.python,
            "scripts/extract_windows_from_posmaps.py",
            "--posmaps",
            str(cfg.posmaps),
            "--seq_cache",
            str(cfg.seq_cache),
            "--window_type",
            "near_variant",
            "--window_len",
            str(cfg.window_len),
            "--use_allele",
            "max_act",
            "--score",
            "abs_delta_act",
            "--max_per_variant",
            "1",
            "--max_total",
            "5000",
            "--seed",
            str(cfg.seed),
            "--out_pos_fa",
            str(pos_fa),
            "--out_neg_fa",
            str(neg_fa),
        ],
        cwd=Path.cwd(),
        env=base_env,
    )

    # 2) sanitize fasta headers
    pos_san = outdir / f"near_variant.pos.{cfg.window_len}.sanitized.fa"
    neg_san = outdir / f"near_variant.neg.{cfg.window_len}.sanitized.fa"
    pos_map = outdir / f"near_variant.pos.{cfg.window_len}.sanitized.map.tsv"
    neg_map = outdir / f"near_variant.neg.{cfg.window_len}.sanitized.map.tsv"

    _run(
        [cfg.python, "scripts/sanitize_fasta_for_fimo.py", "--in_fa", str(pos_fa), "--out_fa", str(pos_san), "--map_tsv", str(pos_map), "--prefix", "pos_"],
        cwd=Path.cwd(),
        env=base_env,
    )
    _run(
        [cfg.python, "scripts/sanitize_fasta_for_fimo.py", "--in_fa", str(neg_fa), "--out_fa", str(neg_san), "--map_tsv", str(neg_map), "--prefix", "neg_"],
        cwd=Path.cwd(),
        env=base_env,
    )

    # 3) STREME + TOMTOM
    _run(
        ["bash", "scripts/run_motif_discovery.sh", str(pos_san), str(neg_san), str(outdir)],
        cwd=Path.cwd(),
        env={**base_env, "JASPAR_DB": cfg.jaspar_meme},
    )

    # 5) FIMO scan (sanitized pos only)
    fimo_dir = outdir / "fimo"
    fimo_tsv = fimo_dir / "fimo.tsv"
    if not fimo_tsv.exists():
        fimo_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "fimo",
                "--oc",
                str(fimo_dir),
                "--thresh",
                "1e-4",
                str((outdir / "streme" / "streme.txt")),
                str(pos_san),
            ],
            cwd=Path.cwd(),
            env=base_env,
        )

    # 4) Choose motif (prefer ETS-family) but require that it has FIMO hits.
    tomtom_tsv = outdir / "tomtom" / "tomtom.tsv"
    lead_tsv = outdir / "lead_motifs.tsv"
    if not lead_tsv.exists():
        _run(
            [
                cfg.python,
                "scripts/pick_lead_motifs.py",
                "--tomtom_tsv",
                str(tomtom_tsv),
                "--jaspar_meme",
                str(cfg.jaspar_meme),
                "--top_k",
                "200",
                "--out_tsv",
                str(lead_tsv),
            ],
            cwd=Path.cwd(),
            env=base_env,
        )

    motif_id = _choose_motif_with_hits(
        lead_motifs_tsv=lead_tsv,
        fimo_tsv=fimo_tsv,
        preferred_motif=cfg.motif_query_id,
        prefer_family=str(args.prefer_family),
    )

    best_hit = _best_tomtom_hit_for_query(tomtom_tsv, motif_id)
    tomtom_target_id = str(best_hit.get("Target_ID")) if best_hit.get("Target_ID") is not None else None

    # Choose PWM id for pwm_minimize.
    pwm_id = cfg.pwm_id
    if str(pwm_id).lower() == "auto":
        if tomtom_target_id and (not jaspar_ids or tomtom_target_id in jaspar_ids):
            pwm_id = tomtom_target_id
        else:
            pwm_id = "MA0062.2"  # safe fallback

    # 6) Perturbation (PWM-minimize, strand-aware)
    pert_dir = outdir / f"perturb_{motif_id}"
    pert_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            cfg.python,
            "scripts/motif_site_perturbation.py",
            "--fimo_tsv",
            str(fimo_tsv),
            "--map_tsv",
            str(pos_map),
            "--genome_fasta",
            "data/hg38.fa",
            "--motif_id",
            str(motif_id),
            "--layer",
            cfg.layer,
            "--channel",
            str(cfg.channel),
            "--mut_mode",
            "pwm_minimize",
            "--pwm_id",
            str(pwm_id),
            "--pwm_topk",
            "3",
            "--audit_pwm",
            "--streme_txt",
            str((outdir / "streme" / "streme.txt")),
            "--max_hits",
            str(cfg.max_hits),
            "--output_sweep",
            "--output_sweep_track_indices",
            cfg.track_indices,
            "--output_sweep_pad_bins_grid",
            cfg.pad_bins_grid,
            f"--output_sweep_shift_bins={cfg.shift_bins}",
            "--outdir",
            str(pert_dir),
        ],
        cwd=Path.cwd(),
        env=base_env,
    )

    # find produced per-hit parquet
    per_hit = next(pert_dir.glob(f"perturb_{motif_id}_{cfg.layer}_*.parquet"))
    summary_json = next(pert_dir.glob(f"perturb_{motif_id}_{cfg.layer}_*.summary.json"))

    # 7) PWM audit table
    pwm_audit_tsv = pert_dir / f"pwm_audit_{cfg.channel}_{motif_id}.tsv"
    _write_pwm_audit_table(per_hit, pwm_audit_tsv)

    # 8) Specificity (51 ch)
    spec_dir = outdir / f"perturb_{motif_id}_specificity_51chs"
    spec_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    # sample random channels from early-layer width (1536) excluding target
    width = 1536
    random = rng.choice([c for c in range(width) if c != cfg.channel], size=50, replace=False)
    channels_csv = ",".join([str(cfg.channel), *map(str, random.tolist())])

    _run(
        [
            cfg.python,
            "scripts/motif_site_perturbation.py",
            "--fimo_tsv",
            str(fimo_tsv),
            "--map_tsv",
            str(pos_map),
            "--genome_fasta",
            "data/hg38.fa",
            "--motif_id",
            str(motif_id),
            "--layer",
            cfg.layer,
            "--channels",
            channels_csv,
            "--mut_mode",
            "pwm_minimize",
            "--pwm_id",
            str(pwm_id),
            "--pwm_topk",
            "3",
            "--audit_pwm",
            "--streme_txt",
            str((outdir / "streme" / "streme.txt")),
            "--max_hits",
            str(cfg.max_hits),
            "--outdir",
            str(spec_dir),
        ],
        cwd=Path.cwd(),
        env=base_env,
    )

    spec_parquet = next(spec_dir.glob(f"perturb_{motif_id}_{cfg.layer}_*.parquet"))
    spec_tsv = spec_dir / f"specificity_{cfg.channel}_{motif_id}.tsv"
    spec_png = spec_dir / f"specificity_{cfg.channel}_{motif_id}.hist.png"
    spec_stats = _compute_specificity_table(spec_parquet, cfg.channel, spec_tsv, spec_png)

    # Motif-locality diagnostic (channel effect on motif segment vs control segment)
    locality = {}
    try:
        df_hit = pd.read_parquet(per_hit)
        if "delta_act_ref_motif" in df_hit.columns and "delta_act_ctrl_region_ref_motif" in df_hit.columns:
            a = df_hit["delta_act_ref_motif"].abs().median()
            b = df_hit["delta_act_ctrl_region_ref_motif"].abs().median()
            locality = {
                "median_abs_delta_act_ref_motif": float(a),
                "median_abs_delta_act_ctrl_region_ref_motif": float(b),
                "ratio_motif_over_ctrl": float(a / (b + 1e-9)),
            }
    except Exception:
        locality = {}

    # Write a compact run manifest to make the channel easy to cite
    manifest = {
        "channel": cfg.channel,
        "layer": cfg.layer,
        "motif_id": motif_id,
        "pwm_id": str(pwm_id),
        "tomtom_best_hit": best_hit,
        "max_hits": cfg.max_hits,
        "flank_bp": cfg.flank_bp,
        "window_len": cfg.window_len,
        "track_indices": cfg.track_indices,
        "pad_bins_grid": cfg.pad_bins_grid,
        "shift_bins": cfg.shift_bins,
        "summary_json": str(summary_json),
        "per_hit_parquet": str(per_hit),
        "pwm_audit_tsv": str(pwm_audit_tsv),
        "specificity_tsv": str(spec_tsv),
        "specificity_hist_png": str(spec_png),
        "specificity": spec_stats,
        "locality": locality,
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
