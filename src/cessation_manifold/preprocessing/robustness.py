"""NaN/Inf guards and edge-case handling for feature extraction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EpochSanitization:
    epoch: np.ndarray
    metadata: dict


def sanitize_epoch(epoch: np.ndarray, min_samples: int = 32, copy: bool = True) -> EpochSanitization:
    """Return a finite, 2-D epoch and metadata describing any fallback applied."""
    arr = np.array(epoch, dtype=float, copy=copy)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError("epoch must be 1-D or 2-D")

    metadata = {"short_epoch": False, "padded_samples": 0, "nonfinite_replaced": 0}

    if arr.shape[0] == 0:
        arr = np.zeros((1, max(min_samples, 1)), dtype=float)
        metadata["short_epoch"] = True
        metadata["padded_samples"] = arr.shape[1]
        return EpochSanitization(epoch=arr, metadata=metadata)

    nonfinite = ~np.isfinite(arr)
    if np.any(nonfinite):
        metadata["nonfinite_replaced"] = int(nonfinite.sum())
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if arr.shape[1] == 0:
        arr = np.zeros((arr.shape[0], max(min_samples, 1)), dtype=float)
        metadata["short_epoch"] = True
        metadata["padded_samples"] = arr.shape[1]
        return EpochSanitization(epoch=arr, metadata=metadata)

    if arr.shape[1] < min_samples:
        metadata["short_epoch"] = True
        pad_width = min_samples - arr.shape[1]
        metadata["padded_samples"] = int(pad_width)
        pad_value = arr[:, -1:] if arr.shape[1] else np.zeros((arr.shape[0], 1), dtype=float)
        arr = np.concatenate([arr, np.repeat(pad_value, pad_width, axis=1)], axis=1)

    channel_std = arr.std(axis=1, keepdims=True)
    flat = channel_std.squeeze(-1) < 1e-12
    if np.any(flat):
        arr[flat] = arr[flat] + np.linspace(0.0, 1e-6, arr.shape[1], dtype=float)

    return EpochSanitization(epoch=arr, metadata=metadata)


def finite_or_default(value: float, fallback: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return float(fallback)
    return value if np.isfinite(value) else float(fallback)


def ensure_feature_dict_finite(features: dict, fallback: float = 0.0) -> tuple[dict, dict]:
    """Replace non-finite feature values with a documented fallback."""
    cleaned = {}
    replacements = {}
    for key, value in features.items():
        safe = finite_or_default(value, fallback=fallback)
        cleaned[key] = safe
        if not np.isfinite(float(value)) if isinstance(value, (float, int, np.floating, np.integer)) else True:
            if safe != value:
                replacements[key] = safe
        elif safe != value:
            replacements[key] = safe
    return cleaned, {
        "fallback_value": float(fallback),
        "fallback_semantics": "0.0 means no reliable evidence after robustness guards.",
        "replacements": replacements,
    }
