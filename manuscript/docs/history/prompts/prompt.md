Update the paper-facing figures and figure plan based on the new provenance results.

Keep this bounded: no new experiments, no rescue work, no manuscript rewrite.

## DNABERT
The fresh splice/codominance reproduction materially conflicts with the previously adopted prose-only claims.

Therefore:

- Do NOT plot or present the old −33.76 pp splice pair effect as established.
- Do NOT plot or present the old large codominance magnitude as established.
- Do NOT replace them with the new single-seed values as a new headline result.
- Mark C-036/C-038 as contested / unsupported by current artifacts in the figure provenance/plan.
- Keep Figure 3 centered on the provenance-secure DNABERT results:
  - pretrained interaction evidence;
  - E9 observer ladder;
  - E9 pair-interaction map;
  - any other panel only if backed by real raw artifacts.
- If needed, reduce Figure 3 rather than pad it with disputed claims.

Interpretation should now be:
> L9/r264+r294 is strongly supported as an interaction in pretrained-MLM tomography, but the previously claimed catastrophic downstream splice effect and large codominance mechanism are not currently established.

## GENERator
The BOS/attention phenotype has now been cleanly reproduced with raw artifacts.

Add it to Figure 4.

Use the reproduced values/artifacts only:
- ~37.9% attention mass at position 0;
- ~33× uniform;
- ~78.8% of heads argmax at position 0;
- activation maximum at BOS.

Keep wording associative:
> BOS-centered attention/activation co-occurs with the high-gain pathway.

Do NOT claim the high-gain row causes the attention sink.

Figure 4 should now preferably be:
A. structural/high-gain callout
B. BOS-centered attention/activation phenotype
C. GC dose-response

## Also
- Incorporate the completed E10b Phi-3 result into Figure 2 if not already done.
- Remove any retired NTv3 buggy-result figure from the normal supplement; do not plot an invalid result just for transparency.
- Update `FIGURE_PLAN_v3.md`, `FIGURE_PROVENANCE.md`, captions, scripts, and rendered outputs accordingly.
- Preserve all existing figure-style rules from `_figstyle.py`.

At the end, report only:
1. what changed in Figures 2–4,
2. which disputed claims were removed/flagged,
3. which new reproduced GENERator panel was added,
4. any remaining provenance gap that still affects a main-text figure.

Then stop.