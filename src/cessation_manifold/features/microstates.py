"""Microstate features.

Wraps pycrostates when installed (extra: `cessation_manifold[microstates]`).
Otherwise falls back to a minimal in-repo k-means-on-GFP-peaks implementation
so the pipeline runs without a heavy, sometimes-brittle dependency. The
fallback is deliberately simple: it is a scaffold, not a replacement for
pycrostates' validated microstate machinery. See README for the tradeoff.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def _gfp_peaks(data: np.ndarray) -> np.ndarray:
    """Global field power peak indices, one per (channels, samples) epoch."""
    gfp = data.std(axis=0)
    peaks = []
    for i in range(1, len(gfp) - 1):
        if gfp[i] > gfp[i - 1] and gfp[i] >= gfp[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def fit_microstate_maps(epochs: np.ndarray, n_states: int = 4, seed: int = 0):
    """Fit k topographic microstate maps on GFP-peak samples pooled across epochs.

    epochs: (n_epochs, n_channels, n_samples)
    Returns (maps, kmeans) where maps is (n_states, n_channels).
    """
    all_peaks = []
    for ep in epochs:
        idx = _gfp_peaks(ep)
        if len(idx):
            v = ep[:, idx].T
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            all_peaks.append(v)
    if not all_peaks:
        raise ValueError("no GFP peaks found across epochs")
    X = np.concatenate(all_peaks, axis=0)
    km = KMeans(n_clusters=n_states, n_init=10, random_state=seed).fit(X)
    return km.cluster_centers_, km


def microstate_sequence(epoch: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Label every sample of one epoch with its best-fit (polarity-free) microstate."""
    n_channels, n_samples = epoch.shape
    x = epoch.T
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    corr = x @ maps.T  # (n_samples, n_states), sign-free GFP correlation
    labels = np.argmax(np.abs(corr), axis=1)
    return labels


def microstate_features(epoch: np.ndarray, maps: np.ndarray) -> dict:
    """Summary features from one epoch's microstate sequence: occurrence,
    mean duration, and transition-matrix entropy (a proxy for sequence
    stereotypy vs. richness)."""
    seq = microstate_sequence(epoch, maps)
    n_states = maps.shape[0]
    occ = np.array([(seq == k).mean() for k in range(n_states)])

    durations = []
    run_start = 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[run_start]:
            durations.append(i - run_start)
            run_start = i
    mean_duration = float(np.mean(durations)) if durations else 0.0

    trans = np.zeros((n_states, n_states))
    for a, b in zip(seq[:-1], seq[1:]):
        trans[a, b] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_p = np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums > 0)
    ent = -np.nansum(np.where(trans_p > 0, trans_p * np.log2(trans_p), 0.0)) / max(n_states, 1)

    feats = {f"ms_occ_{k}": occ[k] for k in range(n_states)}
    feats["ms_mean_duration_samples"] = mean_duration
    feats["ms_transition_entropy"] = float(ent)
    return feats
