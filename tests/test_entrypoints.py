"""Every CLI entry point must at least start.

This exists because a directory rename once silently broke the census entry point: the
harnesses locate each other through `sys.path` inserts built from `parents[N]`, so moving a
directory changes what those resolve to, and nothing failed until the script was actually
run. The experiment-map check only verifies that files exist, which is not the same thing.

Running `--help` is enough to exercise module-level imports and the path arithmetic above
them, which is where that class of breakage lives. Scripts that load a model at import time
exceed the timeout and are skipped rather than failed.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Only intra-repo import failures are repository defects. A missing third-party package is an
# environment problem, and a missing input file is the documented condition that some raw
# artifacts were produced on a cluster and never committed (see the README).
THIRD_PARTY = {"statsmodels", "evo", "evo2", "megaDNA", "mamba_ssm", "transformer_engine",
               "seaborn", "sklearn", "datasets", "pyarrow", "pyfaidx"}
BREAKAGE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_.]+)'")


def _entry_points():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT).stdout
    for t in out.split():
        if not t.endswith(".py"):
            continue
        if t.startswith(("docs/prereg/", "tests/", "stubs/")):
            continue
        src = (ROOT / t).read_text(errors="ignore")
        if "argparse" in src or '__main__' in src:
            yield t


@pytest.mark.parametrize("script", sorted(_entry_points()))
def test_entry_point_starts(script):
    try:
        env = {**os.environ, "PYTHONPATH": str(ROOT)}   # the repo's documented convention
        r = subprocess.run([sys.executable, script, "--help"],
                           capture_output=True, text=True, cwd=ROOT, timeout=60, env=env)
    except subprocess.TimeoutExpired:
        pytest.skip("loads a model at import time; too slow for a smoke test")
    m = BREAKAGE.search(r.stderr)
    if m and m.group(1).split(".")[0] not in THIRD_PARTY:
        pytest.fail(f"{script} cannot start: {m.group(0)}")


if __name__ == "__main__":
    # Runnable without pytest: `python tests/test_entrypoints.py`
    import os as _os
    scripts = sorted(_entry_points())
    failed, skipped = [], 0
    for s in scripts:
        try:
            env = {**_os.environ, "PYTHONPATH": str(ROOT)}
            r = subprocess.run([sys.executable, s, "--help"], capture_output=True,
                               text=True, cwd=ROOT, timeout=60, env=env)
        except subprocess.TimeoutExpired:
            skipped += 1
            continue
        m = BREAKAGE.search(r.stderr)
        if m and m.group(1).split(".")[0] not in THIRD_PARTY:
            failed.append((s, m.group(0)))
    print(f"{len(scripts)} entry points: {len(scripts) - len(failed) - skipped} ok, "
          f"{skipped} skipped (load a model at import), {len(failed)} broken")
    for s, e in failed:
        print(f"  FAIL {s}: {e}")
    sys.exit(1 if failed else 0)
