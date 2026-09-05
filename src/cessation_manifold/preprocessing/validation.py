"""Quantitative QA helpers for preprocessing."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy


def compute_snr_db(reference: np.ndarray, observed: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)
    signal_power = float(np.mean(reference ** 2))
    noise_power = float(np.mean((reference - observed) ** 2)) + 1e-12
    return float(10.0 * np.log10((signal_power + 1e-12) / noise_power))


def _spectral_entropy(channel: np.ndarray, sfreq: float) -> float:
    freqs, psd = welch(channel, fs=sfreq, nperseg=min(len(channel), max(8, int(sfreq * 2))))
    psd = np.clip(psd, 1e-12, None)
    probs = psd / psd.sum()
    return float(entropy(probs) / np.log(len(probs)))


def cross_channel_correlation_checks(before: np.ndarray, after: np.ndarray) -> dict:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    before_corr = np.corrcoef(before)
    after_corr = np.corrcoef(after)
    before_corr = np.nan_to_num(before_corr, nan=0.0)
    after_corr = np.nan_to_num(after_corr, nan=0.0)
    drift = float(np.mean(np.abs(after_corr - before_corr)))
    mean_abs_after = float(np.mean(np.abs(after_corr[np.triu_indices_from(after_corr, k=1)]))) if after_corr.shape[0] > 1 else 0.0
    return {
        "correlation_drift": drift,
        "mean_abs_correlation_after": mean_abs_after,
        "pathological_correlation": bool(mean_abs_after > 0.995),
    }


def validate_artifact_mask(artifact_mask: np.ndarray, provenance: dict | None = None) -> dict:
    artifact_mask = np.asarray(artifact_mask, dtype=bool)
    fraction = float(artifact_mask.mean()) if artifact_mask.size else 0.0
    provenance = provenance or {}
    excluded_components = provenance.get("excluded_components", [])
    return {
        "artifact_fraction": fraction,
        "excluded_component_fraction": float(len(excluded_components) / max(provenance.get("n_components", 1), 1)),
        "valid": bool(fraction < 0.75 and len(excluded_components) <= max(provenance.get("n_components", 1), 1)),
    }


def preprocessing_quality_metrics(
    before: np.ndarray,
    after: np.ndarray,
    sfreq: float,
    reference: np.ndarray | None = None,
    artifact_mask: np.ndarray | None = None,
    provenance: dict | None = None,
) -> dict:
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    entropy_before = np.mean([_spectral_entropy(ch, sfreq) for ch in before])
    entropy_after = np.mean([_spectral_entropy(ch, sfreq) for ch in after])
    out = {
        "entropy_before": float(entropy_before),
        "entropy_after": float(entropy_after),
        "entropy_stability": float(1.0 - abs(entropy_after - entropy_before)),
        "finite_after": bool(np.isfinite(after).all()),
    }
    out.update(cross_channel_correlation_checks(before, after))
    if artifact_mask is not None:
        out.update(validate_artifact_mask(artifact_mask, provenance=provenance))
    if reference is not None:
        out["snr_before_db"] = compute_snr_db(reference, before)
        out["snr_after_db"] = compute_snr_db(reference, after)
        out["snr_improvement_db"] = float(out["snr_after_db"] - out["snr_before_db"])
    else:
        residual = before - after
        out["residual_rms"] = float(np.sqrt(np.mean(residual ** 2)))
    return out
