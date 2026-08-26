#!/bin/bash
# Source this before running any E13 script:  source env.sh
export HF_HOME=/scratch/11034/atzanakak/huggingface_cache
export HF_HUB_CACHE="$HF_HOME"
unset PYTHONPATH PYTHONHOME
# venv lives on /scratch (moved off /work 2026-08-24 -- /work was at ~90% of this user's
# 1TB/1.024M-file quota; a 5.5GB, high-file-count venv on it was an avoidable risk).
VENV=/scratch/11034/atzanakak/genomic-super-weights/.venv
# The system SSL_CERT_FILE (/etc/pki/ca-trust/...) fails cert verification against
# huggingface.co from this node (unable to get local issuer certificate) even though it
# works for other hosts. Point at the venv's own certifi bundle instead of unsetting --
# unsetting alone still failed empirically (huggingface_hub's session appears to fall back
# to a system default rather than certifi when SSL_CERT_FILE is merely absent).
export SSL_CERT_FILE="$VENV/lib64/python3.12/site-packages/certifi/cacert.pem"
export CURL_CA_BUNDLE="$SSL_CERT_FILE"
unset REQUESTS_CA_BUNDLE
export PATH="$VENV/bin:$PATH"
alias py="$VENV/bin/python3"
