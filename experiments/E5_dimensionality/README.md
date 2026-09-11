# E5 — dimensionality (line not included in the paper)

The E5 line was explored and **cut**; no result in the paper derives from it, and it is not
cited by `docs/EXPERIMENT_MAP.md` or `experiments/figures/FIGURE_PROVENANCE.md`. Its
preregistration is retained at `docs/prereg/PREREG_dimensionality_gate0.md` with an explicit
disposition rather than deleted — see `docs/prereg/README.md`.

The directory survives for one reason: **`dimensionality_lib.py` is a live dependency** of three
audit re-derivation builders, which import it by module name:

* `audit/scripts/section4_norm_matched_diagnostic.py`
* `audit/rederivations/scripts/section4a_batch2.py`
* `audit/rederivations/scripts/section4a_topk_norm_q1.py`

Do not move the file to `src/` and do not rename this directory. Those importers locate it
through a `sys.path` insert built from a literal path, so either change breaks them at run time
without failing at import time.
