#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill motif-locality metrics into existing run_manifest.json files.")
    p.add_argument("--manifest", nargs="+", required=True, help="One or more run_manifest.json paths")
    return p.parse_args()


def compute_locality(per_hit_parquet: Path) -> dict:
    df = pd.read_parquet(per_hit_parquet)
    if "delta_act_ref_motif" in df.columns and "delta_act_ctrl_region_ref_motif" in df.columns:
        a = df["delta_act_ref_motif"].abs().median()
        b = df["delta_act_ctrl_region_ref_motif"].abs().median()
        return {
            "median_abs_delta_act_ref_motif": float(a),
            "median_abs_delta_act_ctrl_region_ref_motif": float(b),
            "ratio_motif_over_ctrl": float(a / (b + 1e-9)),
        }
    return {}


def main() -> None:
    args = parse_args()
    for mpath in [Path(p) for p in args.manifest]:
        m = json.loads(mpath.read_text())
        per_hit = Path(m["per_hit_parquet"])
        loc = compute_locality(per_hit)
        m["locality"] = loc
        mpath.write_text(json.dumps(m, indent=2) + "\n")
        print(f"wrote {mpath} locality_keys={list(loc.keys())}")


if __name__ == "__main__":
    main()
