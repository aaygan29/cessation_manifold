"""Surrogate data generation for Gate 3: the readout must break on surrogates.

Implements phase randomization and IAAFT (Iterated Amplitude Adjusted Fourier
Transform, Schreiber & Schmitz 1996), which preserves the amplitude
distribution and (approximately) the power spectrum while destroying
nonlinear phase structure. A readout that keeps reporting a confident,
low-distance-to-cessation score on IAAFT surrogates is picking up linear
spectral content only, not the structure the pipeline claims to detect.
"""
from __future__ import annotations

import numpy as np


def phase_randomize(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Randomize Fourier phases, preserving the power spectrum exactly."""
    rng = np.random.default_rng(seed)
    n = len(x)
    fx = np.fft.rfft(x)
    phases = rng.uniform(0, 2 * np.pi, size=fx.shape)
    phases[0] = 0
    if n % 2 == 0:
        phases[-1] = 0
    surrogate = np.fft.irfft(np.abs(fx) * np.exp(1j * phases), n=n)
    return surrogate


def iaaft(x: np.ndarray, n_iter: int = 100, seed: int = 0, tol: float = 1e-8) -> np.ndarray:
    """Iterated Amplitude Adjusted Fourier Transform surrogate.

    Preserves both the amplitude distribution and (approximately) the power
    spectrum of x while randomizing phase structure.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))

    surrogate = rng.permutation(x)
    prev_spec_err = np.inf
    for _ in range(n_iter):
        fs = np.fft.rfft(surrogate)
        phases = np.angle(fs)
        adjusted = np.fft.irfft(target_amp * np.exp(1j * phases), n=n)

        ranks = np.argsort(np.argsort(adjusted))
        surrogate = sorted_x[ranks]

        spec_err = float(np.mean((np.abs(np.fft.rfft(surrogate)) - target_amp) ** 2))
        if abs(prev_spec_err - spec_err) < tol:
            break
        prev_spec_err = spec_err
    return surrogate


def surrogate_epoch(epoch: np.ndarray, method: str = "iaaft", seed: int = 0) -> np.ndarray:
    """Apply a surrogate method independently to every channel of one epoch."""
    fn = iaaft if method == "iaaft" else phase_randomize
    out = np.empty_like(epoch)
    for i, ch in enumerate(epoch):
        out[i] = fn(ch, seed=seed + i)
    return out
