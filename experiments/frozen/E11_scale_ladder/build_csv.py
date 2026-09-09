"""
experiments/E11_scale_ladder/build_csv.py

Combines the 12 newly measured E11 ladder models (results/E11/raw/*.json, produced by
run_model.py) with the 11 already-measured panel models cited from prior experiments
(E7/E8/E10, per CLAIMS_LEDGER.md C-034's own evidence paths) into the two E11 deliverable
CSVs. Writes:
  results/E11/scale_ladder.csv
  results/E11/scale_ladder_controls.csv

Cited-model numbers are transcribed by hand from:
  results/e7_phase1_detection_qwen25.json, results/e7_phase1_detection_genomeocean.json
  results/e7_legacy_reanalysis.json (Llama-7B, Mistral-7B, OLMo-7B, GENERator-EUK-3B, DNABERT-2, NTv3)
  results/e8_detection_mosaicbert.json, results/e8_detection_modernbert.json
  experiments/E7_exact_dimensionality/RESULTS.md (Phi-3 median-of-6-rows)
  docs/MANUSCRIPT_SOURCE_OF_TRUTH.md (NTv3 architecture: hidden=1536 from n_singular_values,
    d_ffn=6144 from packed fc1=12288, 12 layers)
Total-param / d_model / d_ffn / n_layers / vocab / tie_word_embeddings for the cited models
were pulled live from each repo's config.json (and, for MosaicBERT/DNABERT-2/NTv3 whose
weights ship as pytorch_model.bin with no safetensors metadata, from a meta/CPU
AutoModel.from_config() parameter count -- config + code only, no full weight download)
during this same E11 measurement pass; these are architecture facts, not re-measurements
of q1/PR_spec, which are untouched from their original evidence files above.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

SALVAGE = Path(__file__).resolve().parents[2]
ROOT = SALVAGE.parent
RAW_DIR = ROOT / "results" / "E11" / "raw"
OUT_DIR = ROOT / "results" / "E11"

PANEL_ORDER = [
    "qwen25-0.5b", "qwen25-1.5b", "qwen25-3b",
    "smollm2-135m", "smollm2-360m", "smollm2-1.7b",
    "generator-prok-1.2b", "generator-prok-3b",
    "eurobert-210m", "eurobert-610m", "eurobert-2.1b",
    "modernbert-large",
]

# ---- 11 cited (not re-measured) panel models --------------------------------------------
# Fields: model, repo, revision, family, domain, architecture (decoder/encoder),
# total_params, non_embed_params, d_model, d_ffn, n_layers, layer, row, q1, pr_spec, frob_norm,
# gated_evidence, dtype, source
CITED = [
    dict(model="Llama-7B", repo="huggyllama/llama-7b", revision="unpinned (E5/E6 original)",
         family="Llama", domain="text", architecture="decoder",
         total_params=6738417664, non_embed_params=6738417664 - 2*32000*4096,
         d_model=4096, d_ffn=11008, n_layers=32, layer=2, row=3968,
         q1=0.9888168037615727, pr_spec=1.022653174688543, frob_norm=137.61641832032996,
         gated_evidence="gate_proj/up_proj/down_proj (LlamaMLP SwiGLU)", dtype="float32",
         source="results/e7_legacy_reanalysis.json"),
    dict(model="Mistral-7B", repo="mistralai/Mistral-7B-v0.1", revision="unpinned (E5/E6 original)",
         family="Mistral", domain="text", architecture="decoder",
         total_params=7241732096, non_embed_params=7241732096 - 2*32000*4096,
         d_model=4096, d_ffn=14336, n_layers=32, layer=1, row=2070,
         q1=0.9922047361077938, pr_spec=1.0157714240616442, frob_norm=0.38834723364706875,
         gated_evidence="gate_proj/up_proj/down_proj (MistralMLP SwiGLU)", dtype="float32",
         source="results/e7_legacy_reanalysis.json"),
    dict(model="OLMo-7B-0724-hf", repo="allenai/OLMo-7B-0724-hf", revision="unpinned (E5/E6 original)",
         family="OLMo-v1", domain="text", architecture="decoder",
         total_params=6888095744, non_embed_params=6888095744 - 2*50304*4096,
         d_model=4096, d_ffn=11008, n_layers=32, layer=1, row=269,
         q1=0.964564665808712, pr_spec=1.07470169944751, frob_norm=0.9110752807676615,
         gated_evidence="gate_proj/up_proj/down_proj (OlmoMLP SwiGLU)", dtype="float32",
         source="results/e7_legacy_reanalysis.json"),
    dict(model="Phi-3-mini-4k-instruct", repo="microsoft/Phi-3-mini-4k-instruct",
         revision="f39ac1d28e925b323eae81227eaba4464caced4e",
         family="Phi-3", domain="text", architecture="decoder",
         total_params=3821079552, non_embed_params=3821079552 - 2*32064*3072,
         d_model=3072, d_ffn=8192, n_layers=32, layer="2 & 4 (6 published rows, median)",
         row="525/1693/1113 (L2), 525/1113/1693 (L4)",
         q1=0.9028, pr_spec=1.2249, frob_norm=float("nan"),
         gated_evidence="packed gate_up_proj, chunked (Phi3MLP SwiGLU)", dtype="float32",
         source="experiments/E7_exact_dimensionality/RESULTS.md"),
    dict(model="Qwen2.5-7B", repo="Qwen/Qwen2.5-7B", revision="d149729398750b98c0af14eb82c78cfe92750796",
         family="Qwen2.5", domain="text", architecture="decoder",
         total_params=7615616512, non_embed_params=7615616512 - 2*152064*3584,
         d_model=3584, d_ffn=18944, n_layers=28, layer=26, row=458,
         q1=0.9529104185966117, pr_spec=1.0995461100252455, frob_norm=29.68742128221388,
         gated_evidence="gate_proj/up_proj/down_proj (Qwen2MLP SwiGLU)", dtype="float32",
         source="results/e7_phase1_detection_qwen25.json"),
    dict(model="MosaicBERT", repo="mosaicml/mosaic-bert-base", revision="c89bbadc24278928f22bcdd7de6b61a5a2d08553",
         family="MosaicBERT", domain="text", architecture="encoder",
         total_params=137400384, non_embed_params=137400384 - 23448576,
         d_model=768, d_ffn=3072, n_layers=12, layer=9, row=287,
         q1=0.4765682244123624, pr_spec=4.115864702588391, frob_norm=19.997746047192482,
         gated_evidence="packed gated_layers, chunked (GeGLU-style)", dtype="float32",
         source="results/e8_detection_mosaicbert.json"),
    dict(model="ModernBERT-base", repo="answerdotai/ModernBERT-base",
         revision="8949b909ec900327062f0ebf497f51aef5e6f0c8",
         family="ModernBERT", domain="text", architecture="encoder",
         total_params=149655232, non_embed_params=149655232 - 50368*768,
         d_model=768, d_ffn=1152, n_layers=22, layer=15, row=251,
         q1=0.8969889514776379, pr_spec=1.2336584717613133, frob_norm=836.046538392924,
         gated_evidence="packed Wi, chunked input/gate -> Wo (GeGLU)", dtype="float32",
         source="results/e8_detection_modernbert.json"),
    dict(model="NTv3", repo="InstaDeepAI/NTv3_650M_pre", revision="unpinned (E5/E6 original)",
         family="NTv3", domain="genomic", architecture="encoder",
         total_params=651829819, non_embed_params=651829819,  # conv/deconv tower arch; token-embed contribution negligible (alphabet_size=11)
         d_model=1536, d_ffn=6144, n_layers=12, layer=11, row=1472,
         q1=0.38889396751753286, pr_spec=6.4803244910178845, frob_norm=441.5173190114778,
         gated_evidence="packed fc1 [12288,1536] (2x fc2's 6144 d_ffn), chunked", dtype="float32",
         source="results/e7_legacy_reanalysis.json"),
    dict(model="DNABERT-2", repo="zhihan1996/DNABERT-2-117M",
         revision="7bce263b15377fc15361f52cfab88f8b586abda0",
         family="DNABERT-2", domain="genomic", architecture="encoder",
         total_params=117074176, non_embed_params=117074176 - 3148800,
         d_model=768, d_ffn=3072, n_layers=12, layer=5, row=603,
         q1=0.7933113982887636, pr_spec=1.503287698117315, frob_norm=74.00855774610585,
         gated_evidence="packed Wqkvff-style gated MLP (bert_layers.py GLU)", dtype="float32",
         source="results/e7_legacy_reanalysis.json"),
    dict(model="GENERator-EUK-3B", repo="GenerTeam/GENERator-v2-eukaryote-3b-base",
         revision="7dc01bccce5b65e15141170538afdc2ff09d8dde",
         family="GENERator-EUK", domain="genomic", architecture="decoder",
         total_params=2998262784, non_embed_params=2998262784 - 2*4128*3072,
         d_model=3072, d_ffn=8448, n_layers=30, layer=4, row=2371,
         q1=0.9688691252375834, pr_spec=1.0652593586761891, frob_norm=522.1322102348346,
         gated_evidence="gate_proj/up_proj/down_proj (Llama-superset SwiGLU)", dtype="float32",
         source="results/e7_legacy_reanalysis.json"),
    dict(model="GenomeOcean-4B", repo="DOEJGI/GenomeOcean-4B", revision="2bed2fc3ed47c5f6955ba3e64563512c9b338dfb",
         family="GenomeOcean", domain="genomic", architecture="decoder",
         total_params=4253174784, non_embed_params=4253174784 - 2*4096*3072,
         d_model=3072, d_ffn=16384, n_layers=24, layer=1, row=2604,
         q1=0.8989077166811766, pr_spec=1.2243008136348879, frob_norm=2.8832156466529963,
         gated_evidence="gate_proj/up_proj/down_proj (Mistral-arch SwiGLU)", dtype="float32",
         source="results/e7_phase1_detection_genomeocean.json"),
]
for c in CITED:
    c["relative_depth"] = None
    c["source_type"] = "cited"
    n_layers = c["n_layers"]
    layer = c["layer"]
    if isinstance(layer, int) and isinstance(n_layers, int):
        c["relative_depth"] = layer / n_layers
    c["layer_median_frob_norm"] = None
    c["candidate_frob_norm_ratio_to_layer_median"] = None
    c["detection_ratio"] = None


def load_measured():
    rows = []
    control_rows = []
    for key in PANEL_ORDER:
        fp = RAW_DIR / f"{key}.json"
        d = json.loads(fp.read_text())
        cand = d.get("candidate")
        row = dict(
            model=d["model"], repo=d["repo"], revision=d["resolved_revision"],
            family=d["family"], domain=d["domain"], architecture=d["architecture"],
            total_params=d["total_params"], non_embed_params=d["non_embed_params"],
            d_model=d["d_model"], d_ffn=d["d_ffn"], n_layers=d["n_layers"],
            layer=cand["layer"] if cand else None, row=cand["row"] if cand else None,
            relative_depth=cand["relative_depth"] if cand else None,
            q1=cand["q1"] if cand else None, pr_spec=cand["pr_spec"] if cand else None,
            frob_norm=cand["frob_norm"] if cand else None,
            detection_ratio=cand["detection_ratio"] if cand else None,
            layer_median_frob_norm=d.get("layer_median_frob_norm"),
            candidate_frob_norm_ratio_to_layer_median=d.get("candidate_frob_norm_ratio_to_layer_median"),
            gated_evidence=d["gate_evidence"], dtype=d["dtype"],
            source=f"results/E11/raw/{key}.json", source_type="measured",
            null_outcome=d["null_outcome"], panel_index=d["panel_index"],
        )
        rows.append(row)
        for cr in d.get("control_rows", []):
            control_rows.append(dict(model=d["model"], layer=cand["layer"], row=cr["row"],
                                      q1=cr["q1"], pr_spec=cr["pr_spec"], frob_norm=cr["frob_norm"]))
    return rows, control_rows


def main():
    measured_rows, control_rows = load_measured()

    all_fields = ["model", "repo", "revision", "family", "domain", "architecture",
                  "total_params", "non_embed_params", "d_model", "d_ffn", "n_layers",
                  "layer", "row", "relative_depth", "q1", "pr_spec", "frob_norm",
                  "detection_ratio", "layer_median_frob_norm",
                  "candidate_frob_norm_ratio_to_layer_median",
                  "gated_evidence", "dtype", "source", "source_type"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ladder_path = OUT_DIR / "scale_ladder.csv"
    with ladder_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        for c in CITED:
            row = dict(c)
            row["revision"] = c["revision"]
            w.writerow(row)
        for r in measured_rows:
            w.writerow(r)
    print(f"wrote {ladder_path} ({len(CITED)} cited + {len(measured_rows)} measured = "
          f"{len(CITED)+len(measured_rows)} rows)")

    controls_path = OUT_DIR / "scale_ladder_controls.csv"
    with controls_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "layer", "row", "q1", "pr_spec", "frob_norm"])
        w.writeheader()
        for r in control_rows:
            w.writerow(r)
    print(f"wrote {controls_path} ({len(control_rows)} control rows, {len(measured_rows)} models x 5)")


if __name__ == "__main__":
    main()
