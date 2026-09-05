import numpy as np

from cessation_manifold.honesty.adaptive_conformal import AdaptiveConformalPredictor


def _grouped_regression(seed: int = 0, n_blocks: int = 30, block_size: int = 24):
    rng = np.random.default_rng(seed)
    session_ids = np.repeat(np.arange(n_blocks), block_size)
    subject_ids = np.repeat(np.arange(max(1, n_blocks // 3)), block_size * 3)[: n_blocks * block_size]
    block_effect = np.repeat(rng.normal(0.0, 0.15, size=n_blocks), block_size)
    X = rng.standard_normal((n_blocks * block_size, 6))
    y = 1.5 * X[:, 0] - 0.75 * X[:, 1] + block_effect + 0.2 * rng.standard_normal(n_blocks * block_size)
    return X, y, subject_ids, session_ids


def test_block_aware_split_respects_boundaries():
    X, y, subject_ids, session_ids = _grouped_regression()
    predictor = AdaptiveConformalPredictor(
        n_epochs=len(X),
        block_structure={"subject": subject_ids, "session": session_ids},
        seed=0,
    )
    train, calib, test = predictor.split_indices(X, y)
    for session in np.unique(session_ids):
        memberships = [
            np.any(session_ids[train] == session),
            np.any(session_ids[calib] == session),
            np.any(session_ids[test] == session),
        ]
        assert sum(bool(flag) for flag in memberships) == 1


def test_adaptive_conformal_hits_target_coverage_across_seeds():
    coverages = []
    for seed in range(5):
        X, y, subject_ids, session_ids = _grouped_regression(seed=seed)
        predictor = AdaptiveConformalPredictor(
            n_epochs=len(X),
            block_structure={"subject": subject_ids, "session": session_ids},
            target_coverage=0.9,
            seed=seed,
        ).fit_from_full_data(X, y)
        split = predictor.last_split_
        metrics = predictor.evaluate(X[split["test"]], y[split["test"]], split["blocks"][split["test"]])
        coverages.append(metrics["coverage"])
    assert abs(float(np.mean(coverages)) - 0.9) <= 0.05
    assert all(0.85 <= cov <= 1.0 for cov in coverages)


def test_calibration_diagnostics_look_uniform_under_null():
    X, y, subject_ids, session_ids = _grouped_regression(seed=4)
    predictor = AdaptiveConformalPredictor(
        n_epochs=len(X),
        block_structure={"subject": subject_ids, "session": session_ids},
        seed=4,
    ).fit_from_full_data(X, y)
    diagnostics = predictor.diagnostics_
    assert diagnostics["p_value_entropy"] > 0.85
    assert 0.0 <= diagnostics["miscalibration_index"] <= 0.1


def test_tiny_fold_edge_cases_stay_finite():
    X, y, subject_ids, session_ids = _grouped_regression(seed=2, n_blocks=6, block_size=5)
    predictor = AdaptiveConformalPredictor(
        n_epochs=len(X),
        block_structure={"subject": subject_ids, "session": session_ids},
        seed=2,
    ).fit_from_full_data(X, y)
    split = predictor.last_split_
    metrics = predictor.evaluate(X[split["test"]], y[split["test"]], split["blocks"][split["test"]])
    assert np.isfinite(metrics["coverage"])
    assert 0.0 <= metrics["block_coverage"] <= 1.0
