"""Gate 4: split-conformal intervals hold approximately nominal coverage, and
the gate() honesty wrapper abstains when coverage is artificially broken."""
import numpy as np
import pytest

from cessation_manifold.honesty.conformal import fit_split_conformal, empirical_coverage
from cessation_manifold.honesty.gates import gate, UnvalidatedClaimError


def _make_regression_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = X[:, 0] * 2 - X[:, 1] + 0.3 * rng.standard_normal(n)
    return X, y


def test_split_conformal_holds_approximately_nominal_coverage():
    X, y = _make_regression_data()
    idx = np.random.default_rng(0).permutation(len(X))
    train, calib, test = idx[:300], idx[300:450], idx[450:]
    predictor = fit_split_conformal(X[train], y[train], X[calib], y[calib], alpha=0.1, seed=0)
    coverage = empirical_coverage(predictor, X[test], y[test])
    assert 0.80 <= coverage <= 1.0  # nominal 0.90, generous tolerance for a small test fold


def test_gate_passes_with_good_coverage():
    finding = gate(
        value=1.0, lower=0.5, upper=1.5, coverage_target=0.9, coverage_achieved=0.91,
        dataset_id="unit-test", config={"a": 1},
    )
    assert finding.status == "VALIDATED"


def test_gate_abstains_with_broken_coverage():
    with pytest.raises(UnvalidatedClaimError):
        gate(
            value=1.0, lower=0.5, upper=1.5, coverage_target=0.9, coverage_achieved=0.5,
            dataset_id="unit-test", config={"a": 1},
        )


def test_gate_abstains_when_off_manifold():
    with pytest.raises(UnvalidatedClaimError):
        gate(
            value=1.0, lower=0.5, upper=1.5, coverage_target=0.9, coverage_achieved=0.95,
            dataset_id="unit-test", config={"a": 1}, off_manifold=True,
        )
