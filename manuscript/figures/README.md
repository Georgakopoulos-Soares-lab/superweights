# figures/

One script per manuscript figure. See `FIGURE_PLAN_v3.md` for the panel-by-panel plan,
`FIGURE_PROVENANCE.md` for the source-artifact manifest, and `FIGURE_CAPTIONS_v3.md` for
draft captions.

Governing style: `../../scripts/analysis/_figstyle.py` (`apply_style()`/`panel_label()`),
shared model-class encoding: `_paper_encoding.py`.

- `main/` — 4 main-text figures (`fig1_structural`, `fig2_causal`, `fig3_dnabert`,
  `fig4_generator`), each `.png` + `.pdf`.
- `supplement/` — 1 supplementary figure (`fig_s1_structural_detail`). A second
  (`fig_s2_ntv3_retired_splice`) was removed 2026-08-22: an invalid, truncation-bug-affected
  NTv3 result is not plotted even labeled RETIRED — see `FIGURE_PROVENANCE.md`.
- `source_data/` — one `.json` per figure with the exact plotted values (not raw
  experiment output, which stays in `experiments/`/repo-root `results/`).
- Root-level scripts (`fig_e9_observer_ladder.py`, `fig_e9_pair_interaction_map.py`,
  `fig_e9_generator_dose_response.py`) are E9's original standalone figures; their plotting
  logic is reused (not duplicated by copy-paste) inside `fig3_dnabert.py` and
  `fig4_generator.py`. Their own `output/` renders are kept for reference.

Run any script from this directory with `python3 <script>.py` (repo `generator`/`biojepa`/
base conda env; matplotlib + numpy only).
