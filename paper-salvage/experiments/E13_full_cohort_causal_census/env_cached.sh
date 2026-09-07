#!/bin/bash
# For models ALREADY present in the pre-existing shared /work HF cache at
# /work/11034/atzanakak/ls6/nonbdna/cache/hf (216GB; this is the SAME HF_HOME the shell
# environment set by default before this session touched anything -- confirmed 2026-08-24
# to already contain full weights for all of: Llama-7B, Mistral-7B, OLMo-7B-0724-hf (+
# OLMo-2 1B/7B/13B), Phi-3-mini-4k-instruct, the full Qwen2.5 ladder (0.5B/1.5B/3B/7B), the
# full SmolLM2 ladder, the full EuroBERT ladder, ModernBERT-base/large, MosaicBERT,
# GENERator-v2-eukaryote-3b-base (rev 7dc01bcc..., all 3 safetensors shards),
# GENERator-v2-prokaryote-1.2b/3b-base, GENERanno-{eukaryote,prokaryote}-0.5b-base,
# GenomeOcean-4B, DNABERT-2, NTv3_650M_pre. /work is at ~90% of this user's quota -- every
# call using this env MUST pass local_files_only=True so it can only ever read, never
# write, into this cache.
# Source this before running any E13 script:
#   source env_cached.sh
export HF_HOME=/work/11034/atzanakak/ls6/nonbdna/cache/hf
export HF_HUB_CACHE="$HF_HOME"
unset PYTHONPATH PYTHONHOME
VENV=/scratch/11034/atzanakak/genomic-super-weights/.venv
export PATH="$VENV/bin:$PATH"
export SSL_CERT_FILE="$VENV/lib64/python3.12/site-packages/certifi/cacert.pem"
export CURL_CA_BUNDLE="$SSL_CERT_FILE"
unset REQUESTS_CA_BUNDLE
export HF_HUB_OFFLINE=0
