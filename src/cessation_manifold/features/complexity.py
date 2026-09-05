"""Signal complexity features: Lempel-Ziv complexity, DFA exponent, sample entropy.

Uses `antropy` and `nolds` (both listed as core deps) rather than reimplementing
these well-tested algorithms.
"""
from __future__ import annotations

import warnings

import numpy as np
import antropy as ant
import nolds
from sklearn.exceptions import UndefinedMetricWarning

from cessation_manifold.preprocessing.robustness import ensure_feature_dict_finite, sanitize_epoch


def complexity_features(epoch: np.ndarray, sfreq: float) -> dict:
    """epoch: (n_channels, n_samples). Averages complexity metrics across channels."""
    epoch = sanitize_epoch(epoch).epoch
    lzc, dfa, sampen = [], [], []
    for ch in epoch:
        coarse = ch[:: max(1, len(ch) // 512)]
        entropy_signal = ch[:: max(1, len(ch) // 1024)]
        try:
            binary = (ch > np.median(ch)).astype(int)
            lzc.append(ant.lziv_complexity(binary, normalize=True))
        except Exception:
            pass
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                dfa.append(nolds.dfa(coarse))
        except Exception:
            pass
        try:
            sampen.append(ant.sample_entropy(entropy_signal))
        except Exception:
            pass

    def _mean(x):
        return float(np.mean(x)) if x else float("nan")

    features = {
        "lempel_ziv_complexity": _mean(lzc),
        "dfa_exponent": _mean(dfa),
        "sample_entropy": _mean(sampen),
    }
    return ensure_feature_dict_finite(features)[0]
