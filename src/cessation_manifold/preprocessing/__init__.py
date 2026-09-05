"""Robust EEG preprocessing helpers."""

from .artifact_removal import ArtifactRemovalResult, remove_artifacts
from .robustness import ensure_feature_dict_finite, sanitize_epoch
from .validation import preprocessing_quality_metrics

__all__ = [
    "ArtifactRemovalResult",
    "ensure_feature_dict_finite",
    "preprocessing_quality_metrics",
    "remove_artifacts",
    "sanitize_epoch",
]
