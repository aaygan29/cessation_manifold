"""Cross-frequency coupling and state-transition features."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from cessation_manifold.preprocessing.robustness import ensure_feature_dict_finite, sanitize_epoch


def _bandpass(x: np.ndarray, sfreq: float, lo: float, hi: float) -> np.ndarray:
    nyq = sfreq / 2.0
    lo = max(0.1, min(lo, nyq - 1e-3))
    hi = max(lo + 1e-3, min(hi, nyq - 1e-3))
    b, a = butter(3, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x)


def _modulation_index(phase: np.ndarray, amp: np.ndarray, n_bins: int = 18) -> float:
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    digitized = np.digitize(phase, bins) - 1
    means = np.array([amp[digitized == i].mean() if np.any(digitized == i) else 0.0 for i in range(n_bins)], dtype=float)
    probs = np.clip(means / (means.sum() + 1e-12), 1e-12, None)
    h = -np.sum(probs * np.log(probs))
    return float((np.log(n_bins) - h) / np.log(n_bins))


def cross_frequency_features(epoch: np.ndarray, sfreq: float) -> dict:
    epoch = sanitize_epoch(epoch).epoch
    pac_vals = []
    dwell_vals = []
    transitions = []
    for ch in epoch:
        low = _bandpass(ch, sfreq, 4.0, 8.0)
        high = _bandpass(ch, sfreq, 30.0, min(45.0, sfreq / 2.0 - 1e-3))
        phase = np.angle(hilbert(low))
        amp = np.abs(hilbert(high))
        pac_vals.append(_modulation_index(phase, amp))

        alpha = _bandpass(ch, sfreq, 8.0, 12.0)
        env = np.abs(hilbert(alpha))
        thresh = np.median(env) + 0.5 * (np.median(np.abs(env - np.median(env))) + 1e-12)
        state = env >= thresh
        starts = np.where(np.diff(np.concatenate([[0], state.astype(int), [0]])) == 1)[0]
        ends = np.where(np.diff(np.concatenate([[0], state.astype(int), [0]])) == -1)[0]
        runs = (ends - starts).astype(float)
        dwell_vals.append(float(np.mean(runs)) if len(runs) else 0.0)
        transitions.append(float(np.sum(np.diff(state.astype(int)) != 0)))

    feats = {
        "theta_gamma_pac_mi": float(np.mean(pac_vals)) if pac_vals else np.nan,
        "alpha_state_dwell_samples": float(np.mean(dwell_vals)) if dwell_vals else np.nan,
        "alpha_state_transition_count": float(np.mean(transitions)) if transitions else np.nan,
    }
    return ensure_feature_dict_finite(feats)[0]
