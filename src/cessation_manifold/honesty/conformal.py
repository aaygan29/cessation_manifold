"""Split-conformal prediction for the distance-from-cessation score.

Standard split-conformal regression (Lei et al. 2018 style): fit a point
predictor on a training fold, compute nonconformity scores (absolute
residuals) on a held-out calibration fold, take the (1 - alpha)-quantile of
those scores as a fixed half-width, and report coverage on a further held-out
test fold to check the interval actually holds its nominal rate.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .adaptive_conformal import AdaptiveConformalPredictor


@dataclass
class ConformalPredictor:
    model: object
    half_width: float
    alpha: float

    def predict_interval(self, X: np.ndarray):
        if hasattr(self.model, "predict_interval"):
            return self.model.predict_interval(X)
        point = self.model.predict(X)
        return point, point - self.half_width, point + self.half_width


def fit_split_conformal(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    alpha: float = 0.1,
    seed: int = 0,
) -> ConformalPredictor:
    """alpha=0.1 -> nominal 90% coverage."""
    predictor = AdaptiveConformalPredictor(
        n_epochs=len(X_train) + len(X_calib),
        target_coverage=1.0 - alpha,
        adaptive_sizing=False,
        n_splits=3,
        seed=seed,
    )
    predictor.fit(X_train, y_train, X_calib, y_calib)
    return ConformalPredictor(model=predictor, half_width=float(predictor.half_width_), alpha=float(predictor.alpha_))


def empirical_coverage(predictor: ConformalPredictor, X_test: np.ndarray, y_test: np.ndarray) -> float:
    _, lo, hi = predictor.predict_interval(X_test)
    covered = (y_test >= lo) & (y_test <= hi)
    return float(covered.mean())
