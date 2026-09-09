#!/usr/bin/env python3
"""Exit successfully iff a complete atomic Stage-0 model checkpoint is present."""
from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "results" / "E13" / "stage0_raw_responses.json"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: stage0_checkpoint_complete.py MODEL")
    if not RAW.exists():
        raise SystemExit(1)
    try:
        payload = json.loads(RAW.read_text()).get(sys.argv[1])
        complete = (payload is not None and len(payload.get("conditions", [])) >= 2
                    and len(payload.get("baseline_per_batch", [])) > 0)
    except (OSError, json.JSONDecodeError, TypeError):
        complete = False
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
