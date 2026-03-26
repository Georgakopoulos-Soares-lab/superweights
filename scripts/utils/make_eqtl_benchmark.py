import re
import pandas as pd
from pathlib import Path

IN_PATH = "/work/10906/arisk/ls6/SuperWeights/data/eqtl/tables/Whole_Blood.tsv.gz"
OUT_DIR = Path("/work/10906/arisk/ls6/SuperWeights/data/eqtl/derived")
OUT_PATH = OUT_DIR / "Whole_Blood_pip0p9_topgene.parquet"

PIP_THR = 0.9
VAR_RE = re.compile(r"^(chr[^_]+)_(\d+)_([ACGT])_([ACGT])_b38$")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load only needed columns (big speed/memory win)
usecols = [
    "variant", "allele1", "allele2", "tissue", "gene",
    "pip", "beta_posterior", "beta_marginal", "maf",
]
df = pd.read_csv(IN_PATH, sep="\t", compression="gzip", usecols=usecols)

# Parse variant string
parsed = df["variant"].str.extract(VAR_RE)
parsed.columns = ["chrom", "pos1", "ref", "alt"]
df = pd.concat([df, parsed], axis=1)

# Filter: parsed SNVs only
df = df.dropna(subset=["chrom", "pos1", "ref", "alt"])

# Ensure allele1/allele2 match ref/alt (drop "bad" rows)
df = df[(df["allele1"] == df["ref"]) & (df["allele2"] == df["alt"])]

# Whole Blood only (should already be, but keep it safe)
df = df[df["tissue"] == "Whole_Blood"]

# Filter high-confidence fine-mapped pairs
df = df[df["pip"] >= PIP_THR].copy()

# Normalize types
df["pos1"] = df["pos1"].astype(int)

# Strip Ensembl version: ENSG... .5 -> ENSG...
df["gene_id"] = df["gene"].astype(str).str.split(".").str[0]

# Choose a single gene per variant:
# 1) max pip, 2) tie-break by max |beta_posterior|, 3) keep first
df["abs_beta_post"] = df["beta_posterior"].abs()

df = df.sort_values(["variant", "pip", "abs_beta_post"], ascending=[True, False, False])
df_top = df.drop_duplicates(subset=["variant"], keep="first").copy()

# Final output columns (stable contract for downstream code)
out = df_top[[
    "chrom", "pos1", "ref", "alt",
    "variant",
    "gene_id",
    "pip",
    "beta_posterior",
    "beta_marginal",
    "maf",
]].rename(columns={"beta_posterior": "beta"})

# Write parquet for speed + stable types
out.to_parquet(OUT_PATH, index=False)

# Print summary for sanity
print("Wrote:", OUT_PATH)
print("Rows (variant->top gene):", len(out))
print("Unique variants:", out["variant"].nunique())
print("PIP summary:", out["pip"].describe().to_string())
print("beta summary:", out["beta"].describe().to_string())
print("maf summary:", out["maf"].describe().to_string())
