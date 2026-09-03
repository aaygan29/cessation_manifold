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
from sklearn.ensemble import RandomForestRegressor


@dataclass
class ConformalPredictor:
    model: object
    half_width: float
    alpha: float

    def predict_interval(self, X: np.ndarray):
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
    model = RandomForestRegressor(n_estimators=200, random_state=seed, max_depth=6)
    model.fit(X_train, y_train)

    residuals = np.abs(y_calib - model.predict(X_calib))
    n = len(residuals)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    half_width = float(np.quantile(residuals, q_level))

    return ConformalPredictor(model=model, half_width=half_width, alpha=alpha)


def empirical_coverage(predictor: ConformalPredictor, X_test: np.ndarray, y_test: np.ndarray) -> float:
    _, lo, hi = predictor.predict_interval(X_test)
    covered = (y_test >= lo) & (y_test <= hi)
    return float(covered.mean())
