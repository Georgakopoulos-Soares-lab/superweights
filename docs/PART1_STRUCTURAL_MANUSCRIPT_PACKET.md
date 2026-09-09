# PART 1 — Structural manuscript evidence packet

Scope: updated structural Part 1 only. This packet transcribes final artifacts; it does not analyze or rewrite the paper.

## 1. E11 structural cohort

The E11 structural cohort is the 12 newly measured models (`source_type=measured` in `scale_ladder.csv`). `candidate layer/row` is zero-based. `layer-relative ||U_k||_F` is candidate Frobenius norm divided by the median control-row Frobenius norm in that layer.

| model | exact checkpoint/revision | encoder/decoder | text/genomic | non-embedding params | candidate layer/row | detector ratio | q1 | PR_spec | exact ||U_k||_F | layer-relative ||U_k||_F |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | Qwen/Qwen2.5-0.5B @ `060db6499f32faf8b98477b0a26969ef7d8b9987` | decoder | text | 357,898,112 | 21 / 62 | 547.7380960339361 | 0.9981566830594912 | 1.003696649788444 | 18.726591709019885 | 30.623268771468037 |
| Qwen2.5-1.5B | Qwen/Qwen2.5-1.5B @ `8faed761d45a263340a0528343f099c05c9a4323` | decoder | text | 1,310,340,608 | 26 / 408 | 608.3004507965339 | 0.9950986414892103 | 1.0098722524248653 | 86.68857568622904 | 22.28607583079485 |
| Qwen2.5-3B | Qwen/Qwen2.5-3B @ `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` | decoder | text | 2,774,773,760 | 30 / 318 | 425.0562974794937 | 0.8173600221565082 | 1.4393617425532852 | 37.905147328895474 | 8.318105315988955 |
| SmolLM2-135M | HuggingFaceTB/SmolLM2-135M @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` | decoder | text | 106,203,456 | 11 / 507 | 1578.6014348892807 | 0.9243918826993042 | 1.169251561735028 | 2002.997197003894 | 12.47718017297001 |
| SmolLM2-360M | HuggingFaceTB/SmolLM2-360M @ `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | decoder | text | 314,635,200 | 3 / 87 | 1855.7432971298485 | 0.9742504779226046 | 1.0535466426798263 | 5692.495943194842 | 22.37669908683808 |
| SmolLM2-1.7B | HuggingFaceTB/SmolLM2-1.7B @ `effd688a12921b4cc83e3312b6feb579f70f9c71` | decoder | text | 1,610,713,088 | 7 / 227 | 4647.756022355401 | 0.9663919138439745 | 1.0707002949326883 | 3922.6019896226653 | 15.535443754242134 |
| GENERator-PROK-1.2B | GenerTeam/GENERator-v2-prokaryote-1.2b-base @ `8b2f768b0d293953518ff91d34600f9322ef1f94` | decoder | genomic | 1,145,153,536 | 4 / 798 | 239.50503984131333 | 0.8468514071586476 | 1.3942039532761663 | 6.219604570429286 | 3.222349081478031 |
| GENERator-PROK-3B | GenerTeam/GENERator-v2-prokaryote-3b-base @ `b18ac86df77359d894d7bc050cea78e2d0713021` | decoder | genomic | 2,972,900,352 | 8 / 260 | 257.2681402610017 | 0.934973533990709 | 1.143862862462253 | 52.16156499457906 | 7.656615220640426 |
| EuroBERT-210M | EuroBERT/EuroBERT-210m @ `39b51e15dd1f1a06f58b5cbf6a8a188cec60bd0e` | encoder | text | 113,265,408 | 3 / 300 | 101.50225104036197 | 0.9838491163147434 | 1.0329040152598965 | 2.0419119424635404 | 14.949934194686236 |
| EuroBERT-610M | EuroBERT/EuroBERT-610m @ `d9af784ed20db6c2096e335ec6a67dd4a219924c` | encoder | text | 460,123,776 | 13 / 762 | 1580.3586102450065 | 0.9996238293034937 | 1.0007527536463374 | 8.106532412548177 | 57.21725262189751 |
| EuroBERT-2.1B | EuroBERT/EuroBERT-2.1B @ `81245a4d71f43452badf5e04458e4ddb831ff109` | encoder | text | 1,812,089,088 | 14 / 2198 | 572.9956471013313 | 0.9977414719034632 | 1.0045322505182814 | 2.750915946112038 | 20.893724690144545 |
| ModernBERT-large | answerdotai/ModernBERT-large @ `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13` | encoder | text | 344,253,440 | 19 / 379 | 687.0649430706966 | 0.9701337490592797 | 1.0621492362619986 | 708.1322939648359 | 26.575958286152655 |

Source: `results/E11/scale_ladder.csv`; exact U-norm backfill and ratio provenance: `results/E11/scale_ladder_backfilled.csv`; per-model raw artifacts: `results/E11/raw/*.json`.

## 2. E11 same-layer controls

| model | candidate q1 | control q1 values (5) | mean control q1 | candidate-control gap |
|---|---:|---|---:|---:|
| Qwen2.5-0.5B | 0.9981566830594912 | 0.020779069698598558, 0.0273435113016163, 0.019476790348757125, 0.050076343357488506, 0.030619029219367072 | 0.0296589 | 0.9684978 |
| Qwen2.5-1.5B | 0.9950986414892103 | 0.028391040457566165, 0.025918277213761737, 0.009553010436888899, 0.014422886491389692, 0.012456636698437143 | 0.0181484 | 0.9769502 |
| Qwen2.5-3B | 0.8173600221565082 | 0.011000696867174533, 0.011501303842802015, 0.011373294032780106, 0.011733943760059167, 0.09020817137855418 | 0.0271635 | 0.7901965 |
| SmolLM2-135M | 0.9243918826993042 | 0.026756453978158307, 0.04373514322701722, 0.03942907953994088, 0.034721224103136585, 0.03898734449633432 | 0.0367258 | 0.8876661 |
| SmolLM2-360M | 0.9742504779226046 | 0.012804239737699294, 0.019292654937681272, 0.015836807438479462, 0.019525194729250166, 0.01558441454799738 | 0.0166087 | 0.9576418 |
| SmolLM2-1.7B | 0.9663919138439745 | 0.013305137530174614, 0.013202826336233563, 0.01320682267162399, 0.013290469084506793, 0.01263127266370162 | 0.0131273 | 0.9532646 |
| GENERator-PROK-1.2B | 0.8468514071586476 | 0.013025004767561563, 0.012471289763498563, 0.012312329283789369, 0.013623127933924652, 0.012477562780692024 | 0.0127819 | 0.8340695 |
| GENERator-PROK-3B | 0.934973533990709 | 0.008751366448971109, 0.008107479328136786, 0.008265001776592472, 0.0109473607400331, 0.00904271623012913 | 0.0090228 | 0.9259508 |
| EuroBERT-210M | 0.9838491163147434 | 0.15243100116033595, 0.21858404524793604, 0.10896112405169942, 0.08847561324081633, 0.10214418523593437 | 0.1341193 | 0.8497298 |
| EuroBERT-610M | 0.9996238293034937 | 0.026338343030813335, 0.03443904995464864, 0.03596640683197582, 0.03204270327775329, 0.03357504235286746 | 0.0324723 | 0.9671515 |
| EuroBERT-2.1B | 0.9977414719034632 | 0.06482330544320554, 0.04717126445225402, 0.04060900770269803, 0.051665524865586565, 0.03838257115191648 | 0.0485303 | 0.9492112 |
| ModernBERT-large | 0.9701337490592797 | 0.6431978021476956, 0.25477038316901224, 0.7969730729418231, 0.2248459598856094, 0.21211274700972377 | 0.4263800 | 0.5437537 |

Source: `results/E11/scale_ladder_controls.csv` and candidate rows in `results/E11/scale_ladder.csv`. Means and gaps are arithmetic summaries of those final values.

## 3. Recovered older controls and compatibility

| model | candidate q1 | five same-layer control q1 values | mean | gap |
|---|---:|---|---:|---:|
| MosaicBERT | 0.4765682244123624 | 0.14657439269381178, 0.1275468608509182, 0.07626470950472097, 0.2642282095811723, 0.33690408695470303 | 0.1903036 | 0.2862646 |
| ModernBERT-base | 0.8969889514776379 | 0.3419971952274783, 0.05146480970909673, 0.07245213358704954, 0.2626049300275088, 0.11186016652269085 | 0.1680764 | 0.7289126 |

These are recovered from the final E10 selection-audit artifacts, not newly measured. The control-selection rule is substantively the same—five random rows in the candidate’s layer, excluding the candidate, using the fixed seed-42 convention—but it is **not exactly E11-compatible at the seed-stream level**: E10’s older panels used their E10 `SeedSequence(42)` allocations, whereas E11 declares a fixed 12-model panel and uses `SeedSequence(42).spawn(12)` with the panel index. Sources: `results/e10_selection_audit_mosaicbert.json`, `results/e10_selection_audit_modernbert.json`, `paper-salvage/experiments/E11_scale_ladder/run_model.py`, `paper-salvage/docs/prereg/PREREG_E11_scale_ladder.md`.

## 4. Evo2-7B exclusion

Checkpoint/revision: `arcinstitute/evo2_7b`; resolved revision is `null` in the final artifact. Detector threshold: ratio ≥ 5.0. The final result is a detection null: **global-max ratio 2.22 < 5.0**, so no candidate was accepted. The per-layer diagnostic ratios in the artifact are not the accepted global statistic; the recorded null reason is authoritative for the criterion. Source: `results/e7_phase1_detection_evo2.json`.

## 5. Exact structural-method details

- **Calibration/detection input:** text causal decoders use the first nonempty 20 WikiText-2 raw test lines, joined and tokenized, truncated to 512 tokens; text masked encoders use the same text calibration path and MLM model class; genomic models use the fixed ACTB CDS-derived 504-bp sequence (`_ACTB_500`) in `run_model.py`. Source: `paper-salvage/experiments/E11_scale_ladder/run_model.py`.
- **Tokenizer/preprocessing:** `AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=True)`; text uses tokenizer truncation at `max_length=512`; genomic input is the fixed uppercase ACTB sequence and model tokenizer. Source: `paper-salvage/experiments/E11_scale_ladder/run_model.py`.
- **Sequence length and batch size:** calibration length is at most 512 tokens for WikiText; the genomic fixed input is 504 bp. The E11 forward is one input at a time (batch size 1). Source: `paper-salvage/experiments/E11_scale_ladder/run_model.py` and `results/E11/E11_summary.md` (float32 forward-pass provenance).
- **Detector statistic/threshold:** hook each layer’s down-projection output; flatten token positions, take absolute values, compute each output channel’s maximum, then `out_max / median(channel_max)`; choose the global maximum across layers and accept only if ratio ≥ `RATIO_THRESHOLD=5.0`. Source: `paper-salvage/experiments/E11_scale_ladder/run_model.py` (`DownProjRecorder`, `RATIO_THRESHOLD`); prereg lock: `paper-salvage/docs/prereg/PREREG_E11_scale_ladder.md`.
- **Candidate-selection rule:** candidate is the layer/channel maximum from the detector; the corresponding down-projection row is then scored structurally. The stored candidate row and layer are in the E11 CSV/raw JSONs. Source: `paper-salvage/experiments/E11_scale_ladder/run_model.py`, `results/E11/scale_ladder.csv`.
- **Control-row selection/seed:** five distinct random rows from the same down-projection layer, excluding the candidate, generated with the declared `SeedSequence(42)` panel allocation; E11’s fixed panel order and `spawn(12)` allocation are in `run_model.py`. Sources: `paper-salvage/experiments/E11_scale_ladder/run_model.py`, `results/E11/scale_ladder_controls.csv`.
- **Exact structural quantities:** for each candidate/control row, form the gated bilinear operator from the gate/up row and down-projection row as implemented by `spectral_lib.row_spectral_metrics`; compute singular-value spectrum in float64, `q1 = s_1^2 / sum_i s_i^2`, `PR_spec = (sum_i s_i^2)^2 / sum_i s_i^4`, and `||U_k||_F = sqrt(sum_i s_i^2)` (Gram-identity exact norm in the E13 backfill). Layer-relative norm is candidate norm divided by the median of the five same-layer control norms. Sources: `paper-salvage/experiments/E7_exact_dimensionality/spectral_lib.py`, `paper-salvage/experiments/E11_scale_ladder/run_model.py`, `results/E11/scale_ladder_backfilled.csv`.
- **Parameter counts:** total parameters were counted from the loaded model state/config; non-embedding parameters are the final E11 architecture count after subtracting embedding parameters as specified in the assembly script. Sources: `paper-salvage/experiments/E11_scale_ladder/build_csv.py`, `results/E11/scale_ladder.csv`, `results/E11/E11_summary.md`.

Excluded from this packet: causal/tomography results, manuscript prose, and any new experiment.
