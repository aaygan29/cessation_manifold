#!/usr/bin/env bash
# Fetch a small subset of PhysioNet Sleep-EDF Expanded v1.0.0 sleep-cassette
# recordings (https://physionet.org/content/sleep-edfx/1.0.0/). Each subject
# has a PSG file (multichannel polysomnography, EEG channels Fpz-Cz and
# Pz-Oz at 100 Hz) and a hypnogram file (30-second sleep-stage annotations
# scored per Rechtschaffen & Kales).
#
# License: Open Data Commons Attribution License v1.0. Cite Kemp et al. 2000
# (Analysis of a sleep-dependent neuronal feedback loop, IEEE-BME 47(9))
# and PhysioNet (Goldberger et al. 2000).
set -euo pipefail

OUT_DIR="data/raw/sleep_edfx"
BASE_URL="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"

# Subject ids verified live via WebFetch: SC4001, SC4002, SC4011.
SUBJECTS=("SC4001" "SC4002" "SC4011")

mkdir -p "$OUT_DIR"

echo "Fetching Sleep-EDF Expanded subset into $OUT_DIR ..."
for sub in "${SUBJECTS[@]}"; do
  # Filename convention: SC[subject-night][letter], letter varies (E0/EC/etc).
  # We try the two most common suffixes seen in the archive; the loader
  # only needs any PSG + matching hypnogram.
  for psg_suffix in "E0" "F0" "G0"; do
    url="${BASE_URL}/${sub}${psg_suffix}-PSG.edf"
    if curl -sfI "$url" >/dev/null 2>&1; then
      echo "  GET $url"
      curl -fsSL "$url?download" -o "$OUT_DIR/${sub}${psg_suffix}-PSG.edf"
      break
    fi
  done
  for hyp_suffix in "EC" "EH" "FC" "FH" "GC" "GH"; do
    url="${BASE_URL}/${sub}${hyp_suffix}-Hypnogram.edf"
    if curl -sfI "$url" >/dev/null 2>&1; then
      echo "  GET $url"
      curl -fsSL "$url?download" -o "$OUT_DIR/${sub}${hyp_suffix}-Hypnogram.edf"
      break
    fi
  done
done

echo "Done. See data/README.md for the hypnogram/PSG pairing convention."
