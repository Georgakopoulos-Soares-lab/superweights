# S1 -- Model panel and provenance

Sources: `audit/detector_provenance.csv` (candidate layer/row, selection protocol, ratio-argmax flags, revisions) and `audit/census_master.csv` (HF repo, domain, architecture, non-embedding params), joined on model name. 22/22 models matched with no naming mismatches. `hf_revision` (census_master.csv) and `resolved_revision` (detector_provenance.csv) agree for all 22 models (STORED cross-check).

| Display name | HF repo | Requested rev. | Resolved rev. | Rev. recoverable | Domain | Architecture | Non-embed. params | Candidate (layer/row) | Selection protocol | Ratio-argmax (current detector) |
|---|---|---|---|---|---|---|---|---|---|---|
| DNABERT-2 | zhihan1996/DNABERT-2-117M | 7bce263b15 | 7bce263b15 | TRUE | genomic | encoder | 114M | L5/r603 | grandfathered activation | **FALSE** |
| EuroBERT/EuroBERT-2.1B | EuroBERT/EuroBERT-2.1B | 81245a4d71 | 81245a4d71 | TRUE | text | encoder | 1.81B | L14/r2198 | current ratio detector | TRUE |
| EuroBERT/EuroBERT-210m | EuroBERT/EuroBERT-210m | 39b51e15dd | 39b51e15dd | TRUE | text | encoder | 113M | L3/r300 | current ratio detector | TRUE |
| EuroBERT/EuroBERT-610m | EuroBERT/EuroBERT-610m | d9af784ed2 | d9af784ed2 | TRUE | text | encoder | 460M | L13/r762 | current ratio detector | TRUE |
| GENERator-EUK-3B | GenerTeam/GENERator-v2-eukaryote-3b-base | 7dc01bccce | 7dc01bccce | TRUE | genomic | decoder | 2.97B | L4/r2371 | grandfathered activation | TRUE |
| GenerTeam/GENERator-v2-prokaryote-1.2b-base | GenerTeam/GENERator-v2-prokaryote-1.2b-base | 8b2f768b0d | 8b2f768b0d | TRUE | genomic | decoder | 1.15B | L4/r798 | current ratio detector | TRUE |
| GenerTeam/GENERator-v2-prokaryote-3b-base | GenerTeam/GENERator-v2-prokaryote-3b-base | b18ac86df7 | b18ac86df7 | TRUE | genomic | decoder | 2.97B | L8/r260 | current ratio detector | TRUE |
| GenomeOcean-4B | DOEJGI/GenomeOcean-4B | 2bed2fc3ed | 2bed2fc3ed | TRUE | genomic | decoder | 4.23B | L1/r2604 | current ratio detector | TRUE |
| HuggingFaceTB/SmolLM2-1.7B | HuggingFaceTB/SmolLM2-1.7B | effd688a12 | effd688a12 | TRUE | text | decoder | 1.61B | L7/r227 | current ratio detector | TRUE |
| HuggingFaceTB/SmolLM2-135M | HuggingFaceTB/SmolLM2-135M | 93efa2f097 | 93efa2f097 | TRUE | text | decoder | 106M | L11/r507 | current ratio detector | TRUE |
| HuggingFaceTB/SmolLM2-360M | HuggingFaceTB/SmolLM2-360M | f8027fd0ea | f8027fd0ea | TRUE | text | decoder | 315M | L3/r87 | current ratio detector | TRUE |
| Llama-7B | huggyllama/llama-7b | unpinned | 4782ad2786 | TRUE | text | decoder | 6.48B | L2/r3968 | grandfathered activation | NOT FOUND |
| Mistral-7B | mistralai/Mistral-7B-v0.1 | unpinned | 27d67f1b5f | TRUE | text | decoder | 6.98B | L1/r2070 | grandfathered activation | NOT FOUND |
| ModernBERT-base | answerdotai/ModernBERT-base | 8949b909ec | 8949b909ec | TRUE | text | encoder | 111M | L15/r251 | current ratio detector | TRUE |
| MosaicBERT | mosaicml/mosaic-bert-base | c89bbadc24 | c89bbadc24 | TRUE | text | encoder | 114M | L9/r287 | current ratio detector | TRUE |
| NTv3 | InstaDeepAI/NTv3_650M_pre | unpinned | unpinned | **FALSE** | genomic | encoder | 652M | L11/r1472 | grandfathered activation | **FALSE** |
| OLMo-7B-0724-hf | allenai/OLMo-7B-0724-hf | unpinned | 1ee306df31 | TRUE | text | decoder | 6.48B | L1/r269 | grandfathered activation | NOT FOUND |
| Qwen/Qwen2.5-0.5B | Qwen/Qwen2.5-0.5B | 060db6499f | 060db6499f | TRUE | text | decoder | 358M | L21/r62 | current ratio detector | TRUE |
| Qwen/Qwen2.5-1.5B | Qwen/Qwen2.5-1.5B | 8faed761d4 | 8faed761d4 | TRUE | text | decoder | 1.31B | L26/r408 | current ratio detector | TRUE |
| Qwen/Qwen2.5-3B | Qwen/Qwen2.5-3B | 3aab1f1954 | 3aab1f1954 | TRUE | text | decoder | 2.77B | L30/r318 | current ratio detector | TRUE |
| Qwen2.5-7B | Qwen/Qwen2.5-7B | d149729398 | d149729398 | TRUE | text | decoder | 6.53B | L26/r458 | current ratio detector | TRUE |
| answerdotai/ModernBERT-large | answerdotai/ModernBERT-large | 45bb4654a4 | 45bb4654a4 | TRUE | text | encoder | 344M | L19/r379 | current ratio detector | TRUE |

**Activation ratio where recorded:** `NOT FOUND` for all 22 models as a clean, aggregated numeric field -- `census_master.csv`'s `activation_ratio` column is blank for all 22 rows *by design* (see `audit/AUDIT_REPORT.md`, Section 2 deliverables: populating it correctly requires reconciling multiple, sometimes-disagreeing detector code paths, and the audit chose not to silently pick one). Per `audit/AUDIT_REPORT.md` Section 6, raw numeric ratio values exist *unaggregated* in each model's own source detection JSON for the 16 "current ratio detector" models (e.g. `results/experiments/E11/raw/*.json`, `results/e7_phase1_detection_*.json`, `results/e8_detection_*.json`) but were not extracted into this table to avoid introducing a new, unaudited aggregation step. For the 6 "grandfathered activation" models (Llama-7B, Mistral-7B, OLMo-7B-0724-hf, NTv3, DNABERT-2, GENERator-EUK-3B), the source artifact (`results/experiments/E7/e7_legacy_reanalysis.json`) carries **no ratio field at all**, so the value is genuinely absent, not merely unaggregated.

**Ratio-argmax exceptions (bold FALSE above):** DNABERT-2 (ratio-argmax is L8/r603 at ratio 302.02, not the frozen L5/r603 at ratio 152.73) and NTv3 (ratio-argmax is L6/r1472 at ratio 30.63, not the frozen L11/r1472 at ratio 23.17, which is ratio_rank=3). Both are author-confirmed to have no downstream consequence for the structural story (see `audit/detector_provenance.csv` note field and `audit/rederivations/section3_ntv3_detector_recheck.json`).

**NTv3 revision:** `revision_recoverable = FALSE` -- the only one of 22 models with an unrecoverable HF revision (`resolved_revision = "unpinned (E5/E6 original)"`). This matters for the manuscript's data/code-availability text (Section 5 of `final_check.md`, handled separately).

**Selection protocol classification:** 16/22 models classified `current ratio detector` (selection_rule text begins "ratio-argmax (current protocol...)"), 6/22 classified `grandfathered activation` (selection_rule text begins "activation-argmax (grandfathered...)"). 0 models fell into `literature-derived` or an ambiguous bucket -- classification was unambiguous for all 22 rows based on the stored `selection_rule` text.

Full per-model detail (candidate_source, verbatim selection_rule, verbatim ratio/activation-argmax flag text with reasoning, and provenance notes) is preserved in the companion CSV `S1_model_panel_provenance.csv` (STORED / cross-checked, not truncated).
