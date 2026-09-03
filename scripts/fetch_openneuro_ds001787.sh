#!/usr/bin/env bash
# Fetch a small subset of OpenNeuro ds001787 ("EEG meditation study").
# Dataset existence verified via the OpenNeuro GraphQL API (dataset id
# ds001787, latest snapshot tag 1.1.1) but this script has NOT been run
# end-to-end here: as noted in configs/openneuro_meditation.yaml, the
# dataset is not yet wired into the manifold/conformal pipeline because it
# lacks cessation-onset labels. This script only pulls the raw files so
# feature extraction can be smoke-tested independently.
#
# Requires the openneuro-cli (`npm install -g @openneuro/cli`) or aws-cli
# pointed at the public OpenNeuro S3 bucket. Prefers openneuro-cli if present.
set -euo pipefail

OUT_DIR="data/raw/openneuro_meditation"
DATASET="ds001787"
SUBJECTS=("01" "02")

mkdir -p "$OUT_DIR"

if command -v openneuro >/dev/null 2>&1; then
  echo "Using openneuro-cli to download a subject subset of $DATASET ..."
  for sub in "${SUBJECTS[@]}"; do
    openneuro download "$DATASET" "$OUT_DIR" --include "sub-$sub"
  done
else
  echo "openneuro-cli not found. Install it with:"
  echo "  npm install -g @openneuro/cli"
  echo "then re-run this script, or download manually from:"
  echo "  https://openneuro.org/datasets/$DATASET/versions/1.1.1"
  exit 1
fi
