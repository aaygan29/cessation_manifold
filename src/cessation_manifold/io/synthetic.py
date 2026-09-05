"""Synthetic ground truth for cessation-like collapse.

This module now supports multiple synthetic mechanisms:
- Kuramoto phase-coupled oscillators (baseline scaffold)
- Neural-mass-inspired coupled damped oscillators
- Thalamo-cortical coupled latent generators with subcortical drive terms

All generators expose a shared session object for common validation code.
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
    model_name: str = "kuramoto"
    model_params: dict | None = None


def _kuramoto_step(theta, omega, K, adj, dt):
    n = theta.shape[0]
    diff = theta[None, :] - theta[:, None]
    coupling = (adj * np.sin(diff)).sum(axis=1)
    dtheta = omega + (K / max(n, 1)) * coupling
    return theta + dt * dtheta


def _stable_subject_offset(subject_id: str) -> int:
    digest = hashlib.sha256(subject_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 997


def _default_collapse_mask(
    n_samples: int,
    sfreq: float,
    regime: str,
    n_collapse_events: int,
    collapse_duration_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
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
    return collapse_mask


def _postprocess_signal(signal: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    signal = signal + noise_std * rng.standard_normal(signal.shape)
    signal = signal * 20e-6
    return signal


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

    collapse_mask = _default_collapse_mask(
        n_samples=n_samples,
        sfreq=sfreq,
        regime=regime,
        n_collapse_events=n_collapse_events,
        collapse_duration_s=collapse_duration_s,
        rng=rng,
    )

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

    signal = np.sin(phases) @ lead_field.T
    signal = _postprocess_signal(signal.T, noise_std=noise_std, rng=rng)

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
        model_name="kuramoto",
        model_params={
            "base_coupling": float(base_coupling),
            "collapse_coupling": float(collapse_coupling),
            "freq_hz": float(freq_hz),
            "freq_spread_hz": float(freq_spread_hz),
            "noise_std": float(noise_std),
        },
    )


def simulate_neural_mass_eeg(
    n_channels: int = 19,
    n_seconds: float = 120.0,
    sfreq: float = 250.0,
    regime: str = "collapsed",
    n_collapse_events: int = 3,
    collapse_duration_s: float = 8.0,
    cortical_coupling: float = 1.8,
    collapse_gain: float = 2.8,
    thalamic_drive_hz: float = 10.0,
    subcortical_gain: float = 0.8,
    damping: float = 0.18,
    noise_std: float = 0.15,
    seed: int = 0,
    subject_id: str = "synthsub-01",
    session_id: str = "ses-01",
) -> SyntheticSession:
    """Neural-mass-inspired coupled damped oscillators with subcortical drive."""
    rng = np.random.default_rng(seed)
    n_samples = int(n_seconds * sfreq)
    dt = 1.0 / sfreq
    collapse_mask = _default_collapse_mask(
        n_samples=n_samples,
        sfreq=sfreq,
        regime=regime,
        n_collapse_events=n_collapse_events,
        collapse_duration_s=collapse_duration_s,
        rng=rng,
    )
    adj = 1.0 - np.eye(n_channels)
    x = rng.standard_normal(n_channels) * 0.1
    v = rng.standard_normal(n_channels) * 0.1
    freqs = 2 * np.pi * (8.0 + 2.0 * rng.standard_normal(n_channels))
    lead_field = rng.normal(0, 1, size=(n_channels, n_channels))
    lead_field /= np.linalg.norm(lead_field, axis=1, keepdims=True) + 1e-12
    data = np.empty((n_channels, n_samples))
    R = np.empty(n_samples)

    drive = np.sin(2 * np.pi * thalamic_drive_hz * np.arange(n_samples) * dt)
    if regime == "critical":
        drive += 0.5 * np.sin(2 * np.pi * 0.25 * np.arange(n_samples) * dt)
    if regime == "control":
        drive *= 0.6

    gain_t = np.full(n_samples, cortical_coupling)
    if regime == "collapsed":
        gain_t[collapse_mask] *= collapse_gain

    for t in range(n_samples):
        coupling = (adj @ np.tanh(x)) / max(n_channels, 1)
        subcortical_drive = subcortical_gain * drive[t]
        accel = -damping * v - (freqs**2) * x + gain_t[t] * coupling + subcortical_drive
        v = v + dt * accel
        x = x + dt * v
        data[:, t] = (lead_field @ x)
        phase_proxy = np.arctan2(v, x + 1e-12)
        R[t] = float(np.abs(np.mean(np.exp(1j * phase_proxy))))

    signal = _postprocess_signal(data, noise_std=noise_std, rng=rng)
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
        model_name="neural_mass",
        model_params={
            "cortical_coupling": float(cortical_coupling),
            "collapse_gain": float(collapse_gain),
            "thalamic_drive_hz": float(thalamic_drive_hz),
            "subcortical_gain": float(subcortical_gain),
            "damping": float(damping),
            "noise_std": float(noise_std),
        },
    )


def simulate_thalamocortical_eeg(
    n_channels: int = 19,
    n_seconds: float = 120.0,
    sfreq: float = 250.0,
    regime: str = "collapsed",
    n_collapse_events: int = 3,
    collapse_duration_s: float = 8.0,
    corticothalamic_coupling: float = 1.5,
    reticular_inhibition: float = 1.0,
    brainstem_arousal_tone: float = 0.6,
    spindle_hz: float = 12.0,
    slowwave_hz: float = 1.0,
    noise_std: float = 0.15,
    seed: int = 0,
    subject_id: str = "synthsub-01",
    session_id: str = "ses-01",
) -> SyntheticSession:
    """Thalamo-cortical latent generator with cortical/subcortical coupling."""
    rng = np.random.default_rng(seed)
    n_samples = int(n_seconds * sfreq)
    dt = 1.0 / sfreq
    collapse_mask = _default_collapse_mask(
        n_samples=n_samples,
        sfreq=sfreq,
        regime=regime,
        n_collapse_events=n_collapse_events,
        collapse_duration_s=collapse_duration_s,
        rng=rng,
    )
    t_axis = np.arange(n_samples) * dt
    thalamic = np.zeros((n_channels, n_samples))
    cortical = np.zeros((n_channels, n_samples))
    noise = rng.standard_normal((n_channels, n_samples))

    arousal_tone = brainstem_arousal_tone * np.ones(n_samples)
    if regime == "collapsed":
        arousal_tone[collapse_mask] *= 0.25
    elif regime == "critical":
        arousal_tone += 0.1 * np.sin(2 * np.pi * 0.05 * t_axis)
    else:
        arousal_tone *= 1.2

    phase_jitter = rng.uniform(0, 2 * np.pi, size=n_channels)
    for ch in range(n_channels):
        spindle = np.sin(2 * np.pi * spindle_hz * t_axis + phase_jitter[ch])
        slow = np.sin(2 * np.pi * slowwave_hz * t_axis + 0.5 * phase_jitter[ch])
        thalamic[ch] = (1.0 - arousal_tone) * slow + arousal_tone * spindle
        cortical[ch] = (
            corticothalamic_coupling * np.tanh(thalamic[ch])
            - reticular_inhibition * np.gradient(thalamic[ch], dt)
            + 0.35 * np.sin(2 * np.pi * 8.0 * t_axis + phase_jitter[ch] / 2)
        )
    data = cortical + 0.4 * thalamic + 0.15 * noise
    ph = np.angle(np.exp(1j * data))
    R = np.abs(np.mean(np.exp(1j * ph), axis=0))
    signal = _postprocess_signal(data, noise_std=noise_std, rng=rng)
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
        model_name="thalamo_cortical",
        model_params={
            "corticothalamic_coupling": float(corticothalamic_coupling),
            "reticular_inhibition": float(reticular_inhibition),
            "brainstem_arousal_tone": float(brainstem_arousal_tone),
            "spindle_hz": float(spindle_hz),
            "slowwave_hz": float(slowwave_hz),
            "noise_std": float(noise_std),
        },
    )


def simulate_synthetic_eeg(
    model: str = "kuramoto",
    **kwargs,
) -> SyntheticSession:
    if model == "kuramoto":
        return simulate_kuramoto_eeg(**kwargs)
    if model == "neural_mass":
        return simulate_neural_mass_eeg(**kwargs)
    if model in {"thalamo_cortical", "thalamocortical"}:
        return simulate_thalamocortical_eeg(**kwargs)
    raise ValueError(f"unknown synthetic model {model!r}")


def simulate_subject_sessions(
    subject_id: str,
    n_sessions: int = 3,
    regime: str = "collapsed",
    base_seed: int = 0,
    model: str = "kuramoto",
    model_kwargs: dict | None = None,
    **kwargs,
) -> list:
    """Multiple synthetic sessions for one subject, for the Gate-1 dense-sampling check."""
    model_kwargs = model_kwargs or {}
    overlap = set(model_kwargs).intersection(kwargs)
    if overlap:
        dup = ", ".join(sorted(overlap))
        raise ValueError(f"duplicate synthetic model arguments provided in model_kwargs and kwargs: {dup}")
    merged_kwargs = {**kwargs, **model_kwargs}
    sessions = []
    for i in range(n_sessions):
        sessions.append(
            simulate_synthetic_eeg(
                model=model,
                regime=regime,
                seed=base_seed + 1000 * (i + 1) + _stable_subject_offset(subject_id),
                subject_id=subject_id,
                session_id=f"ses-{i+1:02d}",
                **merged_kwargs,
            )
        )
    return sessions
