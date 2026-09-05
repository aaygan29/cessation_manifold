"""Lightweight, production-oriented EEG artifact attenuation helpers."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA

from .robustness import sanitize_epoch
from .validation import preprocessing_quality_metrics

LOGGER = logging.getLogger(__name__)

ARTIFACT_DICTIONARY = {
    "blink": {"feature": "low_freq_ratio", "z_threshold": 2.5},
    "muscle": {"feature": "high_freq_ratio", "z_threshold": 2.5},
    "spike": {"feature": "kurtosis", "z_threshold": 3.0},
}


@dataclass
class ArtifactRemovalResult:
    cleaned_data: np.ndarray
    artifact_mask: np.ndarray
    quality_metrics: dict
    provenance: dict

    def __iter__(self):
        yield self.cleaned_data
        yield self.artifact_mask
        yield self.quality_metrics
        yield self.provenance


def _adaptive_cutoff(values: np.ndarray, z_threshold: float) -> float:
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) + 1e-12
    return med + z_threshold * 1.4826 * mad


def _power_ratio(signal: np.ndarray, sfreq: float, band: tuple[float, float], total: tuple[float, float] = (0.5, 45.0)) -> float:
    spec = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sfreq)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    total_mask = (freqs >= total[0]) & (freqs <= total[1])
    denom = float(spec[total_mask].sum()) + 1e-12
    return float(spec[band_mask].sum() / denom)


def _component_scores(sources: np.ndarray, sfreq: float) -> list[dict]:
    scores = []
    for idx, component in enumerate(sources):
        scores.append(
            {
                "component": idx,
                "low_freq_ratio": _power_ratio(component, sfreq, (0.5, 4.0)),
                "high_freq_ratio": _power_ratio(component, sfreq, (20.0, min(45.0, sfreq / 2 - 1e-6))),
                "kurtosis": float(abs(kurtosis(component, fisher=False, bias=False))),
            }
        )
    return scores


def _reject_components(scores: list[dict]) -> tuple[list[int], list[dict]]:
    thresholds = {
        name: _adaptive_cutoff(np.array([score[cfg["feature"]] for score in scores], dtype=float), cfg["z_threshold"])
        for name, cfg in ARTIFACT_DICTIONARY.items()
    }
    rejected = []
    labels = []
    for score in scores:
        matched = [name for name, cfg in ARTIFACT_DICTIONARY.items() if score[cfg["feature"]] >= thresholds[name]]
        if matched:
            rejected.append(score["component"])
            labels.append({"component": score["component"], "labels": matched, **score})
    return rejected, labels


def _run_ica(data: np.ndarray, sfreq: float, seed: int) -> tuple[np.ndarray, dict]:
    n_channels, n_samples = data.shape
    n_components = max(2, min(n_channels, n_samples // 16))
    ica = FastICA(n_components=n_components, random_state=seed, whiten="unit-variance", max_iter=1000)
    sources = ica.fit_transform(data.T).T
    scores = _component_scores(sources, sfreq)
    rejected, labels = _reject_components(scores)
    cleaned_sources = sources.copy()
    if rejected:
        cleaned_sources[rejected] = 0.0
    cleaned = ica.inverse_transform(cleaned_sources.T).T
    return cleaned, {
        "method": "ica",
        "n_components": n_components,
        "excluded_components": rejected,
        "component_diagnostics": labels,
    }


def _haar_denoise_channel(channel: np.ndarray, max_levels: int = 4) -> np.ndarray:
    coeffs = []
    current = channel.astype(float, copy=True)
    original_len = len(current)
    levels = min(max_levels, int(np.floor(np.log2(max(len(current), 2)))))
    if levels <= 0:
        return current
    for _ in range(levels):
        if len(current) % 2:
            current = np.append(current, current[-1])
        approx = (current[0::2] + current[1::2]) / np.sqrt(2.0)
        detail = (current[0::2] - current[1::2]) / np.sqrt(2.0)
        coeffs.append(detail)
        current = approx
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-12
    threshold = 0.35 * sigma * np.sqrt(2.0 * np.log(original_len + 1.0))
    for idx, detail in enumerate(coeffs):
        if idx < max(1, len(coeffs) // 2):
            coeffs[idx] = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
    recon = current
    for detail in reversed(coeffs):
        up = np.empty(detail.size * 2, dtype=float)
        up[0::2] = (recon + detail) / np.sqrt(2.0)
        up[1::2] = (recon - detail) / np.sqrt(2.0)
        recon = up
    return recon[:original_len]


def _run_wavelet(data: np.ndarray) -> tuple[np.ndarray, dict]:
    cleaned = np.vstack([_haar_denoise_channel(ch) for ch in data])
    return cleaned, {"method": "wavelet", "wavelet": "haar", "levels": 4, "excluded_components": []}


def _run_regression(data: np.ndarray) -> tuple[np.ndarray, dict]:
    regressor = np.median(data, axis=0)
    denom = float(np.dot(regressor, regressor)) + 1e-12
    cleaned = data.copy()
    for idx, ch in enumerate(cleaned):
        beta = float(np.dot(ch, regressor) / denom)
        cleaned[idx] = ch - beta * regressor
    return cleaned, {"method": "regression", "excluded_components": [], "regressor_rms": float(np.sqrt(np.mean(regressor ** 2)))}


def remove_artifacts(
    eeg_data: np.ndarray,
    sfreq: float,
    methods: list[str] | tuple[str, ...] = ("ica", "wavelet"),
    fallback_on_failure: bool = True,
    seed: int = 0,
    reference: np.ndarray | None = None,
) -> ArtifactRemovalResult:
    """Apply one or more artifact attenuation stages and return provenance."""
    sanitized = sanitize_epoch(eeg_data, min_samples=max(16, int(sfreq // 2)))
    original = sanitized.epoch
    current = original.copy()
    provenance = {"methods": [], "fallback_used": False, "sanitization": sanitized.metadata}

    try:
        for method in methods:
            if method == "ica" and current.shape[0] >= 2 and current.shape[1] >= 32:
                current, meta = _run_ica(current, sfreq, seed=seed)
            elif method == "wavelet":
                current, meta = _run_wavelet(current)
            elif method == "regression":
                current, meta = _run_regression(current)
            else:
                meta = {"method": method, "skipped": True, "excluded_components": []}
            provenance["methods"].append(meta)
    except Exception as exc:
        LOGGER.warning("artifact removal failed; fallback_on_failure=%s", fallback_on_failure, exc_info=exc)
        if not fallback_on_failure:
            raise
        current = original.copy()
        provenance["fallback_used"] = True
        provenance["error"] = str(exc)

    residual = np.mean(np.abs(original - current), axis=0)
    cutoff = _adaptive_cutoff(residual, 2.0)
    artifact_mask = residual >= cutoff
    excluded_components = []
    for method_meta in provenance["methods"]:
        excluded_components.extend(method_meta.get("excluded_components", []))
    provenance["excluded_components"] = sorted(set(excluded_components))
    provenance["n_components"] = max([meta.get("n_components", 0) for meta in provenance["methods"]] + [0])

    quality_metrics = preprocessing_quality_metrics(
        original,
        current,
        sfreq=sfreq,
        reference=reference,
        artifact_mask=artifact_mask,
        provenance=provenance,
    )
    return ArtifactRemovalResult(current, artifact_mask, quality_metrics, provenance)
