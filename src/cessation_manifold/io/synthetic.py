"""Synthetic ground truth for cessation-like collapse.

We do not have open access to real cessation EEG (Zarka et al. 2026, NIMHANS).
Until that data is wired in, the apparatus is validated on a Kuramoto
oscillator network whose order parameter (global phase coherence) is driven
to collapse on a cue, standing in for a cessation event, and whose per-node
phases are projected through a fixed random EEG-like lead field to produce
pseudo-EEG channels. This is a scaffold for testing the pipeline's machinery,
not a claim about the neural mechanism of cessation.
"""
from __future__ import annotations
import hashlib

import numpy as np
from dataclasses import dataclass


@dataclass
class SyntheticSession:
    """One synthetic EEG-like session with known collapse structure."""

    data: np.ndarray          # (n_channels, n_samples)
    sfreq: float
    ch_names: list
    order_parameter: np.ndarray  # (n_samples,) ground-truth Kuramoto R(t)
    collapse_mask: np.ndarray    # (n_samples,) bool, True during cessation windows
    subject_id: str
    session_id: str
    regime: str                # "collapsed" | "critical" | "control"


def _kuramoto_step(theta, omega, K, adj, dt):
    n = theta.shape[0]
    diff = theta[None, :] - theta[:, None]
    coupling = (adj * np.sin(diff)).sum(axis=1)
    dtheta = omega + (K / max(n, 1)) * coupling
    return theta + dt * dtheta


def _stable_subject_offset(subject_id: str) -> int:
    digest = hashlib.sha256(subject_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 997


def simulate_kuramoto_eeg(
    n_channels: int = 19,
    n_seconds: float = 120.0,
    sfreq: float = 250.0,
    regime: str = "collapsed",
    n_collapse_events: int = 3,
    collapse_duration_s: float = 8.0,
    base_coupling: float = 2.0,
    collapse_coupling: float = 9.0,
    freq_hz: float = 10.0,
    freq_spread_hz: float = 1.5,
    noise_std: float = 0.15,
    seed: int = 0,
    subject_id: str = "synthsub-01",
    session_id: str = "ses-01",
) -> SyntheticSession:
    """Simulate a Kuramoto oscillator network and project it to pseudo-EEG.

    regime:
      "collapsed" - coupling strength is driven sharply upward during cue
                    windows so the order parameter R(t) collapses toward 1
                    (near-total phase locking), our proxy for a cessation
                    event (a sudden qualitative state change).
      "critical"  - coupling hovers near the synchronization transition
                    throughout (no sharp cue-locked collapse); this is the
                    synthetic stand-in for a "meditator baseline" that is
                    close to, but has not entered, the cessation manifold.
      "control"   - weak, roughly constant coupling with no structured
                    transitions at all; stand-in apparatus check alongside
                    real non-meditator resting-state EEG (Gate 2).
    """
    rng = np.random.default_rng(seed)
    n_samples = int(n_seconds * sfreq)
    dt = 1.0 / sfreq

    omega = 2 * np.pi * (freq_hz + freq_spread_hz * rng.standard_normal(n_channels))
    theta = rng.uniform(0, 2 * np.pi, size=n_channels)
    adj = 1.0 - np.eye(n_channels)

    # fixed random "lead field" mapping oscillator phase -> channel amplitude
    lead_field = rng.normal(0, 1, size=(n_channels, n_channels))
    lead_field /= np.linalg.norm(lead_field, axis=1, keepdims=True)

    collapse_mask = np.zeros(n_samples, dtype=bool)
    if regime == "collapsed" and n_collapse_events > 0:
        event_len = int(collapse_duration_s * sfreq)
        margin = int(2 * sfreq)
        candidate_starts = np.arange(margin, max(margin + 1, n_samples - event_len - margin), event_len)
        n_events = min(n_collapse_events, max(1, len(candidate_starts)))
        if len(candidate_starts) > 0:
            starts = np.sort(rng.choice(candidate_starts, size=n_events, replace=False))
            for s in starts:
                collapse_mask[s : s + event_len] = True

    K_t = np.full(n_samples, base_coupling)
    if regime == "collapsed":
        K_t[collapse_mask] = collapse_coupling
        # smooth ramp in/out so it is not a step discontinuity
        ramp = int(1.0 * sfreq)
        edges = np.where(np.diff(collapse_mask.astype(int)) != 0)[0]
        for e in edges:
            lo, hi = max(0, e - ramp), min(n_samples, e + ramp)
            K_t[lo:hi] = np.linspace(K_t[lo], K_t[hi - 1], hi - lo)
    elif regime == "critical":
        K_t = base_coupling * 1.6 + 0.5 * np.sin(2 * np.pi * 0.02 * np.arange(n_samples) * dt)
    elif regime == "control":
        K_t = np.full(n_samples, base_coupling * 0.5)
    else:
        raise ValueError(f"unknown regime {regime!r}")

    phases = np.empty((n_samples, n_channels))
    R = np.empty(n_samples)
    for t in range(n_samples):
        phases[t] = theta
        z = np.mean(np.exp(1j * theta))
        R[t] = np.abs(z)
        theta = _kuramoto_step(theta, omega, K_t[t], adj, dt)

    signal = np.sin(phases) @ lead_field.T  # (n_samples, n_channels)
    signal = signal.T  # (n_channels, n_samples)
    signal += noise_std * rng.standard_normal(signal.shape)
    signal *= 20e-6  # scale to volts, EEG-ish microvolt range

    ch_names = [f"E{i+1}" for i in range(n_channels)]
    return SyntheticSession(
        data=signal,
        sfreq=sfreq,
        ch_names=ch_names,
        order_parameter=R,
        collapse_mask=collapse_mask,
        subject_id=subject_id,
        session_id=session_id,
        regime=regime,
    )


def simulate_subject_sessions(
    subject_id: str, n_sessions: int = 3, regime: str = "collapsed", base_seed: int = 0, **kwargs
) -> list:
    """Multiple synthetic sessions for one subject, for the Gate-1 dense-sampling check."""
    return [
        simulate_kuramoto_eeg(
            regime=regime,
            seed=base_seed + 1000 * (i + 1) + _stable_subject_offset(subject_id),
            subject_id=subject_id,
            session_id=f"ses-{i+1:02d}",
            **kwargs,
        )
        for i in range(n_sessions)
    ]
