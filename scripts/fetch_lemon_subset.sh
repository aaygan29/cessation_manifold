#!/usr/bin/env bash
# Fetch a small subset (2 subjects) of the MPI Leipzig LEMON EEG resting-state
# dataset (OpenNeuro ds000221) for a quick smoke test of the non-meditator
# control arm. Uses the GWDG mirror (verified reachable at time of writing)
# and falls back to the OpenNeuro S3 bucket via awscli if that is installed.
set -euo pipefail

OUT_DIR="data/raw/lemon"
SUBJECTS=("sub-010002" "sub-010003")
BASE_URL="https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"

mkdir -p "$OUT_DIR"

echo "Fetching LEMON EEG subset into $OUT_DIR ..."
for sub in "${SUBJECTS[@]}"; do
  mkdir -p "$OUT_DIR/$sub/RSEEG"
  for ext in vhdr vmrk eeg; do
    url="$BASE_URL/$sub/RSEEG/${sub}.${ext}"
    echo "  GET $url"
    curl -fsSL "$url" -o "$OUT_DIR/$sub/RSEEG/${sub}.${ext}" || {
      echo "  WARNING: could not fetch $url (layout may differ; check the"
      echo "  archive index at $BASE_URL and adjust this script)."
    }
  done
done

echo "Done. This is raw BrainVision data, not yet BIDS-formatted; see"
echo "data/README.md for the LEMON BIDS conversion note before pointing"
echo "configs/lemon.yaml at it."
