# data/

Small reference and region files needed by the genomic endpoints. Large inputs are **not**
committed — they are public, and a repository should tell you how to obtain them rather than
vendor tens of megabytes.

## Committed

| path | used by |
|---|---|
| `regions/hg38/random_262kb.bed` | The random-window sampler behind E9 tomography, E12 GENERator analyses, and the mechanism scripts. This is the one region file the reported results depend on. |
| `regions/hg38/manifest.json` | Provenance for the hg38 region set |
| `regions/ecoli/*.bed`, `regions/ecoli/regulondb_*.txt` | Prokaryote region annotations |
| `reference/ecoli/ecoli_k12.gff.gz` | E. coli K-12 annotation |

## Not committed — obtain these yourself

| path expected | what it is | where to get it |
|---|---|---|
| `reference/hg38/hg38.fa` | GRCh38 primary assembly | UCSC: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz`. Place or symlink at this path; several harnesses read it directly. |
| `$GUE_ROOT` | GUE benchmark | Zhou et al., DNABERT-2. Point the `GUE_ROOT` environment variable at the extracted tree. |

## Removed on 2026-09-09

These were committed but are public data that no reported result depends on. They are in git
history if needed:

| path | size | note |
|---|---|---|
| `regions/hg38/enhancers_ccre_262kb.bed` | 40 MB | ENCODE cCREs. Used only by exploratory interpretability scripts that were also removed. |
| `regions/hg38/promoters_262kb.bed` | 10 MB | Same. |
| `GCF_000005845.2_ASM584v2_genomic.fna`, `reference/ecoli/ecoli_k12.fna(.fai)` | 9 MB | E. coli K-12 assembly, available from NCBI under that accession. |
