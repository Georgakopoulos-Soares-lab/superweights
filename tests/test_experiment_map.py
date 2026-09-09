"""The manuscript-to-code map must stay true to the tree.

A reviewer's entry point is docs/EXPERIMENT_MAP.md. If a script is renamed or removed without
regenerating it, that document silently starts lying. This test fails in that case.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_map_verifies():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build_experiment_map.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, (
        "scripts/build_experiment_map.py reported a missing script:\n"
        f"{r.stdout}\n{r.stderr}")


def test_experiment_map_is_committed_and_current():
    md = ROOT / "docs/EXPERIMENT_MAP.md"
    tsv = ROOT / "docs/EXPERIMENT_MAP.tsv"
    assert md.exists() and tsv.exists(), "run scripts/build_experiment_map.py"
    before = md.read_text()
    subprocess.run([sys.executable, str(ROOT / "scripts/build_experiment_map.py")],
                   capture_output=True, text=True, cwd=ROOT)
    assert md.read_text() == before, (
        "docs/EXPERIMENT_MAP.md is stale -- regenerate it with "
        "python scripts/build_experiment_map.py and commit the result")
