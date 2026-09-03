"""Aperiodic (1/f) spectral features.

Uses `fooof`/`specparam` when installed (extra: `cessation_manifold[aperiodic]`).
Falls back to a plain log-log linear fit of the power spectrum, which recovers
the aperiodic exponent and offset without the periodic-peak decomposition
fooof provides. Noted as a scaffold tradeoff in the README.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch


def spectral_exponent(
    epoch_channel: np.ndarray, sfreq: float, fmin: float = 2.0, fmax: float = 40.0
) -> dict:
    """Aperiodic exponent + offset for one channel of one epoch via log-log fit."""
    freqs, psd = welch(epoch_channel, fs=sfreq, nperseg=min(len(epoch_channel), int(sfreq * 2)))
    mask = (freqs >= fmin) & (freqs <= fmax) & (freqs > 0) & (psd > 0)
    if mask.sum() < 4:
        return {"aperiodic_exponent": np.nan, "aperiodic_offset": np.nan}
    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask])
    slope, intercept = np.polyfit(log_f, log_p, 1)
    return {"aperiodic_exponent": float(-slope), "aperiodic_offset": float(intercept)}


def aperiodic_features(epoch: np.ndarray, sfreq: float) -> dict:
    """Mean aperiodic exponent/offset across channels for one (channels, samples) epoch."""
    exps, offs = [], []
    for ch in epoch:
        r = spectral_exponent(ch, sfreq)
        if not np.isnan(r["aperiodic_exponent"]):
            exps.append(r["aperiodic_exponent"])
            offs.append(r["aperiodic_offset"])
    return {
        "aperiodic_exponent_mean": float(np.mean(exps)) if exps else np.nan,
        "aperiodic_offset_mean": float(np.mean(offs)) if offs else np.nan,
    }
