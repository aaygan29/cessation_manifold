"""Signal complexity features: Lempel-Ziv complexity, DFA exponent, sample entropy.

Uses `antropy` and `nolds` (both listed as core deps) rather than reimplementing
these well-tested algorithms.
"""
from __future__ import annotations

import numpy as np
import antropy as ant
import nolds


def complexity_features(epoch: np.ndarray, sfreq: float) -> dict:
    """epoch: (n_channels, n_samples). Averages complexity metrics across channels."""
    lzc, dfa, sampen = [], [], []
    for ch in epoch:
        try:
            binary = (ch > np.median(ch)).astype(int)
            lzc.append(ant.lziv_complexity(binary, normalize=True))
        except Exception:
            pass
        try:
            dfa.append(nolds.dfa(ch))
        except Exception:
            pass
        try:
            sampen.append(ant.sample_entropy(ch))
        except Exception:
            pass

    def _mean(x):
        return float(np.mean(x)) if x else float("nan")

    return {
        "lempel_ziv_complexity": _mean(lzc),
        "dfa_exponent": _mean(dfa),
        "sample_entropy": _mean(sampen),
    }
