"""Neuronal avalanche criticality features.

Threshold-crossing avalanches on the summed absolute signal across channels,
following the standard avalanche-analysis recipe (Beggs & Plenz style):
binarize, sum activity in time bins, and take an "avalanche" as a run of
consecutive above-threshold bins. Size and duration distributions are
summarized by a power-law-like exponent fit (least squares on log-log
histograms), which is a coarse but dependency-free proxy for the
scale-free / critical-branching signature reported near collapse-adjacent
brain states.
"""
from __future__ import annotations

import numpy as np


def _avalanches(binary_activity: np.ndarray):
    sizes, durations = [], []
    in_avalanche = False
    size = 0
    dur = 0
    for v in binary_activity:
        if v > 0:
            in_avalanche = True
            size += v
            dur += 1
        else:
            if in_avalanche:
                sizes.append(size)
                durations.append(dur)
            in_avalanche = False
            size = 0
            dur = 0
    if in_avalanche:
        sizes.append(size)
        durations.append(dur)
    return np.array(sizes), np.array(durations)


def _powerlaw_exponent(values: np.ndarray) -> float:
    values = values[values > 0]
    if len(values) < 5:
        return np.nan
    counts, edges = np.histogram(values, bins=np.logspace(0, np.log10(values.max() + 1), 10))
    centers = (edges[:-1] + edges[1:]) / 2
    mask = counts > 0
    if mask.sum() < 3:
        return np.nan
    slope, _ = np.polyfit(np.log10(centers[mask]), np.log10(counts[mask]), 1)
    return float(-slope)


def criticality_features(epoch: np.ndarray, sfreq: float, bin_ms: float = 4.0) -> dict:
    """epoch: (n_channels, n_samples). Returns avalanche size/duration exponents
    and branching ratio (mean events per bin / mean events in preceding bin)."""
    n_channels, n_samples = epoch.shape
    bin_samples = max(1, int(bin_ms / 1000 * sfreq))
    n_bins = n_samples // bin_samples

    z = (epoch - epoch.mean(axis=1, keepdims=True)) / (epoch.std(axis=1, keepdims=True) + 1e-12)
    thresh = 2.0
    events = (np.abs(z) > thresh).astype(int)

    binned = events[:, : n_bins * bin_samples].reshape(n_channels, n_bins, bin_samples).sum(axis=(0, 2))

    sizes, durations = _avalanches(binned)
    size_exp = _powerlaw_exponent(sizes)
    dur_exp = _powerlaw_exponent(durations)

    nz = binned[binned > 0]
    if len(binned) > 1:
        prev = binned[:-1]
        curr = binned[1:]
        mask = prev > 0
        branching = float((curr[mask] / prev[mask]).mean()) if mask.sum() else np.nan
    else:
        branching = np.nan

    return {
        "avalanche_size_exponent": size_exp,
        "avalanche_duration_exponent": dur_exp,
        "branching_ratio": branching,
        "n_avalanches": float(len(sizes)),
    }
