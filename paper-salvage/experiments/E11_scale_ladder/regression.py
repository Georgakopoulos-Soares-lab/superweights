"""
experiments/E11_scale_ladder/regression.py

E11 frozen analysis (PREREG_E11_scale_ladder.md, "Analysis (frozen)"):
  1. within-family slopes (Delta q1 per decade of log10(non-embed params))
  2. OLS q1 ~ log10(non_embedding_params) + is_decoder (statsmodels), partial correlation
  3. both ||U_k||_F rankings (absolute vs ratio-to-layer-median)
  4. q1 vs relative depth
  5. candidate-vs-control gap by scale

Reads results/E11/scale_ladder.csv (+ scale_ladder_controls.csv), writes
results/E11/regression_summary.json.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

import numpy as np
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[3]
LADDER_CSV = ROOT / "results" / "E11" / "scale_ladder.csv"
CONTROLS_CSV = ROOT / "results" / "E11" / "scale_ladder_controls.csv"
OUT = ROOT / "results" / "E11" / "regression_summary.json"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows():
    rows = []
    with LADDER_CSV.open() as f:
        for r in csv.DictReader(f):
            r["q1"] = _f(r["q1"])
            r["non_embed_params"] = _f(r["non_embed_params"])
            r["relative_depth"] = _f(r["relative_depth"])
            r["frob_norm"] = _f(r["frob_norm"])
            r["candidate_frob_norm_ratio_to_layer_median"] = _f(r.get("candidate_frob_norm_ratio_to_layer_median"))
            rows.append(r)
    return rows


def load_controls():
    by_model = {}
    if CONTROLS_CSV.exists():
        with CONTROLS_CSV.open() as f:
            for r in csv.DictReader(f):
                by_model.setdefault(r["model"], []).append(float(r["q1"]))
    return by_model


def main():
    rows = [r for r in load_rows() if r["q1"] is not None and r["non_embed_params"]]
    controls = load_controls()

    # ---- within-family slopes ------------------------------------------------
    families = {}
    for r in rows:
        families.setdefault(r["family"], []).append(r)
    family_slopes = {}
    for fam, members in families.items():
        if len(members) < 2:
            continue
        x = np.array([math.log10(m["non_embed_params"]) for m in members])
        y = np.array([m["q1"] for m in members])
        if len(set(x.round(6))) < 2:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        family_slopes[fam] = dict(
            n=len(members), slope_per_decade=float(slope), intercept=float(intercept),
            x_range_log10=[float(x.min()), float(x.max())],
            predicted_q1_range=float(slope * (x.max() - x.min())),
            members=[m["model"] for m in members],
        )

    # ---- OLS q1 ~ log10(non_embed) + is_decoder -------------------------------
    y = np.array([r["q1"] for r in rows])
    logp = np.array([math.log10(r["non_embed_params"]) for r in rows])
    is_decoder = np.array([1.0 if r["architecture"] == "decoder" else 0.0 for r in rows])
    X = sm.add_constant(np.column_stack([logp, is_decoder]))
    model = sm.OLS(y, X).fit()
    coef_logp, coef_dec = model.params[1], model.params[2]
    se_logp, se_dec = model.bse[1], model.bse[2]

    # partial correlation of is_decoder with q1 controlling for logp
    Xc = sm.add_constant(logp)
    res_y = sm.OLS(y, Xc).fit().resid
    res_d = sm.OLS(is_decoder, Xc).fit().resid
    partial_corr = float(np.corrcoef(res_y, res_d)[0, 1])

    raw_decoder_q1 = [r["q1"] for r in rows if r["architecture"] == "decoder"]
    raw_encoder_q1 = [r["q1"] for r in rows if r["architecture"] == "encoder"]
    raw_gap = median(raw_decoder_q1) - median(raw_encoder_q1)

    # predicted change across each family's own observed range, vs raw gap
    max_family_predicted = max((abs(v["predicted_q1_range"]) for v in family_slopes.values()), default=0.0)

    coef_over_se = coef_dec / se_dec
    gap_shrink_pct = None  # requires a conditioned-gap estimate; see note below
    # Conditioned gap: model-implied difference between decoder/encoder groups at the
    # SAME mean log10(params), i.e. simply the fitted is_decoder coefficient itself
    # (since the OLS specification is additive/parallel-slopes in log10(params)).
    conditioned_gap = coef_dec
    if raw_gap != 0:
        gap_shrink_pct = 100.0 * (1.0 - abs(conditioned_gap) / abs(raw_gap))

    if abs(coef_over_se) > 2 and max_family_predicted < 0.5 * abs(raw_gap):
        branch = 1
    elif abs(coef_over_se) < 1 or (gap_shrink_pct is not None and gap_shrink_pct > 50):
        branch = 2
    else:
        branch = 3

    # ---- both ||U_k||_F rankings: absolute vs ratio-to-layer-median -----------
    abs_ranking = sorted(
        [(r["model"], r["frob_norm"]) for r in rows if r["frob_norm"] is not None and not math.isnan(r["frob_norm"])],
        key=lambda t: -t[1])
    ratio_ranking = sorted(
        [(r["model"], r["candidate_frob_norm_ratio_to_layer_median"]) for r in rows
         if r["candidate_frob_norm_ratio_to_layer_median"] is not None],
        key=lambda t: -t[1])
    abs_rank_pos = {m: i for i, (m, _) in enumerate(abs_ranking)}
    ratio_rank_pos = {m: i for i, (m, _) in enumerate(ratio_ranking)}
    common = set(abs_rank_pos) & set(ratio_rank_pos)
    rank_disagreements = sorted(
        [dict(model=m, abs_rank=abs_rank_pos[m], ratio_rank=ratio_rank_pos[m],
              delta=abs_rank_pos[m] - ratio_rank_pos[m]) for m in common],
        key=lambda d: -abs(d["delta"]))

    # ---- q1 vs relative depth --------------------------------------------------
    depth_rows = [(r["model"], r["relative_depth"], r["q1"]) for r in rows if r["relative_depth"] is not None]
    if len(depth_rows) >= 3:
        depths = np.array([d for _, d, _ in depth_rows])
        qs = np.array([q for _, _, q in depth_rows])
        depth_q1_corr = float(np.corrcoef(depths, qs)[0, 1])
    else:
        depth_q1_corr = None
    large_models = [(m, d) for m, d, _ in depth_rows]
    small_depths = [d for m, d, q in depth_rows if q is not None]

    # ---- candidate-vs-control gap by scale ------------------------------------
    gap_by_scale = []
    for r in rows:
        crow_q1s = controls.get(r["model"])
        if crow_q1s:
            mean_ctrl = float(np.mean(crow_q1s))
            gap_by_scale.append(dict(model=r["model"], non_embed_params=r["non_embed_params"],
                                      candidate_q1=r["q1"], mean_control_q1=mean_ctrl,
                                      gap=r["q1"] - mean_ctrl))
    gap_by_scale.sort(key=lambda d: d["non_embed_params"])
    gaps = [g["gap"] for g in gap_by_scale]
    gap_stability = dict(
        n=len(gaps), mean=float(np.mean(gaps)) if gaps else None,
        std=float(np.std(gaps)) if gaps else None,
        min=float(np.min(gaps)) if gaps else None, max=float(np.max(gaps)) if gaps else None,
    )

    # ---- encoder-range contingency ---------------------------------------------
    encoder_rows = [r for r in rows if r["architecture"] == "encoder"]
    max_encoder_params = max((r["non_embed_params"] for r in encoder_rows), default=0.0)
    encoder_above_1b = [r["model"] for r in encoder_rows if r["non_embed_params"] >= 1e9]

    summary = dict(
        n_models=len(rows),
        decision_rule=dict(
            branch=branch,
            coefficient_is_decoder=float(coef_dec), se_is_decoder=float(se_dec),
            coef_over_se=float(coef_over_se),
            coefficient_log_params=float(coef_logp), se_log_params=float(se_logp),
            r_squared=float(model.rsquared), r_squared_adj=float(model.rsquared_adj),
            n_obs=int(model.nobs),
            partial_corr_is_decoder_given_logparams=partial_corr,
            raw_architecture_gap_median_q1=raw_gap,
            conditioned_gap_estimate=float(conditioned_gap),
            gap_shrink_pct=gap_shrink_pct,
            max_within_family_predicted_q1_change=max_family_predicted,
            half_raw_gap_threshold=0.5 * abs(raw_gap),
        ),
        family_slopes=family_slopes,
        uknorm_rankings=dict(
            absolute=abs_ranking, ratio_to_layer_median=ratio_ranking,
            rank_disagreements_top=rank_disagreements[:10],
        ),
        depth_control=dict(
            correlation_q1_vs_relative_depth=depth_q1_corr,
            rows=[dict(model=m, relative_depth=d, q1=q) for m, d, q in depth_rows],
        ),
        candidate_vs_control_gap_by_scale=dict(rows=gap_by_scale, stability=gap_stability),
        encoder_range_contingency=dict(
            max_encoder_non_embed_params=max_encoder_params,
            encoders_above_1b=encoder_above_1b,
            triggered_could_not_sample_above_1b=(len(encoder_above_1b) == 0),
        ),
        ols_statsmodels_summary_text=model.summary().as_text(),
    )

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")
    print(f"Branch: {branch}  coef_is_decoder={coef_dec:.4f} SE={se_dec:.4f} |coef/SE|={abs(coef_over_se):.3f}")
    print(f"R^2={model.rsquared:.4f}  partial_corr={partial_corr:.4f}")
    print(f"raw_gap={raw_gap:.4f}  conditioned_gap={conditioned_gap:.4f}  shrink_pct={gap_shrink_pct}")
    print(f"encoders >=1B non-embed params: {encoder_above_1b}")


if __name__ == "__main__":
    main()
