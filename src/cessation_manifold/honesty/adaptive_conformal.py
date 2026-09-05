"""Adaptive, block-aware conformal prediction utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor


@dataclass
class FoldSizes:
    n_train: int
    n_calib: int
    n_test: int


def _combine_block_structure(block_structure: dict | None, n_epochs: int) -> np.ndarray:
    if not block_structure:
        return np.array([f"sample-{i}" for i in range(n_epochs)], dtype=object)
    keys = sorted(block_structure)
    values = [np.asarray(block_structure[key], dtype=object) for key in keys]
    if any(len(v) != n_epochs for v in values):
        raise ValueError("all block arrays must match n_epochs")
    return np.array(["|".join(map(str, row)) for row in zip(*values)], dtype=object)


class AdaptiveConformalPredictor:
    def __init__(
        self,
        n_epochs: int | None = None,
        block_structure: dict | None = None,
        target_coverage: float = 0.9,
        adaptive_sizing: bool = True,
        n_splits: int = 5,
        stability_std_threshold: float = 0.05,
        seed: int = 0,
    ):
        self.n_epochs = n_epochs
        self.block_structure = block_structure
        self.target_coverage = float(target_coverage)
        self.adaptive_sizing = adaptive_sizing
        self.n_splits = int(max(n_splits, 2))
        self.stability_std_threshold = float(stability_std_threshold)
        self.seed = int(seed)
        self.model = None
        self.alpha_ = 1.0 - self.target_coverage
        self.half_width_ = None
        self.cv_half_widths_ = []
        self.diagnostics_ = {}
        self.last_split_ = None

    def suggest_fold_sizes(self, n_epochs: int | None = None, effect_size: float | None = None) -> FoldSizes:
        n = int(n_epochs or self.n_epochs or 0)
        if n <= 0:
            raise ValueError("n_epochs must be positive")
        effect = abs(float(effect_size)) if effect_size is not None else 0.5
        calib_frac = np.clip(0.2 + 0.08 / max(effect, 0.2), 0.2, 0.35) if self.adaptive_sizing else 0.25
        test_frac = np.clip(0.15 + 0.05 / max(effect, 0.2), 0.15, 0.3) if self.adaptive_sizing else 0.25
        n_calib = min(max(8, int(round(n * calib_frac))), max(n - 2, 1))
        n_test = min(max(8, int(round(n * test_frac))), max(n - n_calib - 1, 1))
        n_train = max(n - n_calib - n_test, 1)
        if n_train + n_calib + n_test > n:
            n_test = max(n - n_train - n_calib, 1)
        return FoldSizes(n_train=n_train, n_calib=n_calib, n_test=n_test)

    def split_indices(self, X: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(X)
        blocks = _combine_block_structure(self.block_structure, n)
        unique_blocks, inverse = np.unique(blocks, return_inverse=True)
        grouped = [np.flatnonzero(inverse == i) for i in range(len(unique_blocks))]
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(grouped))
        grouped = [grouped[i] for i in order]

        effect_size = None
        if y is not None and len(y) > 1:
            effect_size = float(np.std(y) / (np.mean(np.abs(y)) + 1e-12))
        sizes = self.suggest_fold_sizes(n_epochs=n, effect_size=effect_size)

        train_idx, calib_idx, test_idx = [], [], []
        counts = {"train": 0, "calib": 0, "test": 0}
        targets = {"train": sizes.n_train, "calib": sizes.n_calib, "test": sizes.n_test}
        for group in grouped:
            bucket = min(targets, key=lambda name: counts[name] / max(targets[name], 1))
            if bucket == "train":
                train_idx.extend(group.tolist())
            elif bucket == "calib":
                calib_idx.extend(group.tolist())
            else:
                test_idx.extend(group.tolist())
            counts[bucket] += len(group)

        for bucket_name, bucket in (("train", train_idx), ("calib", calib_idx), ("test", test_idx)):
            if not bucket:
                largest = max((train_idx, calib_idx, test_idx), key=len)
                bucket.append(largest.pop())
                counts[bucket_name] += 1

        return np.array(sorted(train_idx)), np.array(sorted(calib_idx)), np.array(sorted(test_idx))

    def _quantile_width(self, residuals: np.ndarray, alpha: float) -> float:
        n = len(residuals)
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / max(n, 1))
        return float(np.quantile(residuals, q_level, method="higher" if hasattr(np, "quantile") else "linear"))

    def _crossval_coverages(self, preds: np.ndarray, y_calib: np.ndarray, blocks: np.ndarray | None) -> tuple[list[float], list[float]]:
        n = len(y_calib)
        residuals = np.abs(y_calib - preds)
        if n < self.n_splits:
            width = self._quantile_width(residuals, self.alpha_)
            coverage = float((residuals <= width).mean())
            return [coverage], [width]
        order = np.arange(n)
        if blocks is not None:
            unique = np.unique(blocks)
            order = np.concatenate([np.flatnonzero(blocks == b) for b in unique])
        folds = np.array_split(order, self.n_splits)
        coverages, widths = [], []
        for held in folds:
            keep = np.setdiff1d(np.arange(n), held, assume_unique=False)
            width = self._quantile_width(residuals[keep], self.alpha_)
            widths.append(width)
            coverages.append(float((residuals[held] <= width).mean()))
        return coverages, widths

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray, calib_blocks: np.ndarray | None = None):
        self.model = RandomForestRegressor(n_estimators=200, random_state=self.seed, max_depth=6)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_calib)
        residuals = np.abs(y_calib - preds)
        base_alpha = 1.0 - self.target_coverage
        self.alpha_ = base_alpha
        cv_coverages, widths = self._crossval_coverages(preds, y_calib, calib_blocks)
        mean_cov = float(np.mean(cv_coverages))
        self.alpha_ = float(np.clip(base_alpha + 0.5 * (mean_cov - self.target_coverage), 0.01, 0.25))
        _, tuned_widths = self._crossval_coverages(preds, y_calib, calib_blocks)
        self.half_width_ = float(np.median(tuned_widths)) if tuned_widths else self._quantile_width(residuals, self.alpha_)
        p_values = 1.0 - (np.argsort(np.argsort(residuals)) + 1) / (len(residuals) + 1)
        hist, _ = np.histogram(p_values, bins=10, range=(0.0, 1.0))
        per_block_coverage = {}
        if calib_blocks is not None:
            for block in np.unique(calib_blocks):
                mask = calib_blocks == block
                per_block_coverage[str(block)] = float((residuals[mask] <= self.half_width_).mean())
        self.cv_half_widths_ = widths
        self.diagnostics_ = {
            "mean_cv_coverage": mean_cov,
            "cv_coverages": [float(v) for v in cv_coverages],
            "coverage_std": float(np.std(cv_coverages, ddof=0)) if cv_coverages else 0.0,
            "threshold_stable": bool(not cv_coverages or np.std(cv_coverages, ddof=0) < self.stability_std_threshold),
            "p_value_entropy": float(entropy(np.clip(hist / max(hist.sum(), 1), 1e-12, None)) / np.log(len(hist))),
            "miscalibration_index": float(abs(mean_cov - self.target_coverage)),
            "per_epoch_coverage": [bool(v) for v in (residuals <= self.half_width_)],
            "per_block_coverage": per_block_coverage,
            "min_block_coverage": float(min(per_block_coverage.values())) if per_block_coverage else float(mean_cov),
            "alpha": self.alpha_,
            "half_width": float(self.half_width_),
            "band_half_widths": [float(v) for v in widths],
        }
        return self

    def fit_from_full_data(self, X: np.ndarray, y: np.ndarray):
        train_idx, calib_idx, test_idx = self.split_indices(X, y)
        blocks = _combine_block_structure(self.block_structure, len(X))
        self.fit(X[train_idx], y[train_idx], X[calib_idx], y[calib_idx], calib_blocks=blocks[calib_idx])
        self.last_split_ = {"train": train_idx, "calib": calib_idx, "test": test_idx, "blocks": blocks}
        return self

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.model is None or self.half_width_ is None:
            raise RuntimeError("predictor must be fit before predict")
        point = self.model.predict(X_test)
        lo = point - self.half_width_
        hi = point + self.half_width_
        return point, lo, hi, float(self.diagnostics_.get("min_block_coverage", self.target_coverage))

    def predict_interval(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        point, lo, hi, _ = self.predict(X_test)
        return point, lo, hi

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, test_blocks: np.ndarray | None = None) -> dict:
        point, lo, hi, block_coverage = self.predict(X_test)
        covered = (y_test >= lo) & (y_test <= hi)
        per_block = {}
        if test_blocks is not None:
            for block in np.unique(test_blocks):
                mask = test_blocks == block
                per_block[str(block)] = float(covered[mask].mean())
            block_coverage = float(min(per_block.values())) if per_block else block_coverage
        return {
            "point": point,
            "lower": lo,
            "upper": hi,
            "coverage": float(covered.mean()),
            "block_coverage": float(block_coverage),
            "per_block_coverage": per_block,
        }
