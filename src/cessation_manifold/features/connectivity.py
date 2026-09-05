"""Connectivity and information-exchange features.

Includes a lightweight weighted-symbolic mutual information proxy and
phase-lag metrics to broaden feature coverage beyond channel-local scalars.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from sklearn.metrics import mutual_info_score

from cessation_manifold.preprocessing.robustness import ensure_feature_dict_finite, sanitize_epoch


def _symbolize(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) <= 2:
        return np.zeros_like(x, dtype=int)
    return np.clip(np.digitize(x, edges[1:-1]), 0, max(len(edges) - 2, 0))


def _wsmi_proxy(a: np.ndarray, b: np.ndarray, n_bins: int = 5) -> float:
    sa = _symbolize(a, n_bins=n_bins)
    sb = _symbolize(b, n_bins=n_bins)
    mi = mutual_info_score(sa, sb)
    ha = mutual_info_score(sa, sa) + 1e-12
    hb = mutual_info_score(sb, sb) + 1e-12
    return float(mi / np.sqrt(ha * hb))


def _phase_lag_index(a: np.ndarray, b: np.ndarray) -> float:
    pa = np.angle(hilbert(a))
    pb = np.angle(hilbert(b))
    dphi = pa - pb
    return float(np.abs(np.mean(np.sign(np.sin(dphi)))))


def connectivity_features(epoch: np.ndarray, sfreq: float, max_pairs: int = 64) -> dict:
    _ = sfreq
    epoch = sanitize_epoch(epoch).epoch
    n_channels = epoch.shape[0]
    if n_channels < 2:
        feats = {
            "wsmi_proxy_mean": np.nan,
            "wsmi_proxy_std": np.nan,
            "phase_lag_index_mean": np.nan,
            "phase_lag_index_std": np.nan,
        }
        return ensure_feature_dict_finite(feats)[0]

    pairs = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]
    if len(pairs) > max_pairs:
        idx = np.linspace(0, len(pairs) - 1, max_pairs, dtype=int)
        pairs = [pairs[k] for k in idx]

    wsmi_vals = []
    pli_vals = []
    for i, j in pairs:
        wsmi_vals.append(_wsmi_proxy(epoch[i], epoch[j]))
        pli_vals.append(_phase_lag_index(epoch[i], epoch[j]))

    feats = {
        "wsmi_proxy_mean": float(np.mean(wsmi_vals)) if wsmi_vals else np.nan,
        "wsmi_proxy_std": float(np.std(wsmi_vals)) if wsmi_vals else np.nan,
        "phase_lag_index_mean": float(np.mean(pli_vals)) if pli_vals else np.nan,
        "phase_lag_index_std": float(np.std(pli_vals)) if pli_vals else np.nan,
    }
    return ensure_feature_dict_finite(feats)[0]
