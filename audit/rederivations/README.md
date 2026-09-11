# audit/rederivations/ — archival

**Archival re-derivation scripts and their outputs.** This is *not* the figure pipeline; the
authoritative one is [`experiments/figures/`](../../experiments/figures/), whose panel-by-panel
provenance is recorded in `FIGURE_PROVENANCE.md`.

## What this is

The second of three adversarial verification passes over the manuscript's numbers. Where the
first pass re-read each number from its raw artifact, this pass **recomputed the contested
quantities independently** — the top-norm control comparison, the random-direction GC extension,
the structure–function correlations, the tomography split stability, and the detector
coordinates for NTv3 and GENERator-EUK. A disagreement here would have surfaced as a different
number, not a different opinion.

`final_check.md` is the retained verification record for that pass. The narrative reports from
all three passes were removed when the repository was reduced to code, data and provenance; they
remain in git history.

## Why the scripts are kept

The rendered figures this pass produced were removed as superseded by `experiments/figures/`.
The scripts were **not**, because most of them build artifacts that are still tracked and still
cited — for example `section2_split_stability.py` produces `tomography_split_stability.csv`,
which supplies the 100-resplit intervals in **Supplementary Table S6**. Removing the scripts
would orphan data the supplement depends on.

Treat this directory as archival: the outputs are regenerable, the figures it once produced are
not tracked, and nothing here is on the path to reproducing a paper figure.
