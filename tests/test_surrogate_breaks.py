"""Gate 3: IAAFT surrogates should preserve amplitude distribution / spectrum
but break structure the complexity/criticality features rely on."""
import numpy as np

from cessation_manifold.features.surrogates import iaaft, phase_randomize, surrogate_epoch


def test_iaaft_preserves_amplitude_distribution():
    rng = np.random.default_rng(0)
    x = np.sin(np.linspace(0, 40 * np.pi, 2000)) + 0.3 * rng.standard_normal(2000)
    surrogate = iaaft(x, n_iter=50, seed=0)
    assert np.allclose(np.sort(surrogate), np.sort(x), atol=1e-6)


def test_iaaft_approximately_preserves_power_spectrum():
    rng = np.random.default_rng(1)
    x = np.sin(np.linspace(0, 60 * np.pi, 3000)) + 0.2 * rng.standard_normal(3000)
    surrogate = iaaft(x, n_iter=80, seed=1)
    p_orig = np.abs(np.fft.rfft(x))
    p_surr = np.abs(np.fft.rfft(surrogate))
    corr = np.corrcoef(p_orig, p_surr)[0, 1]
    assert corr > 0.9


def test_phase_randomize_destroys_temporal_structure():
    x = np.sin(np.linspace(0, 40 * np.pi, 2000))
    surrogate = phase_randomize(x, seed=0)
    # autocorrelation at lag 1 should collapse relative to the pure sine
    def lag1_autocorr(v):
        return np.corrcoef(v[:-1], v[1:])[0, 1]
    assert abs(lag1_autocorr(surrogate)) < abs(lag1_autocorr(x))


def test_surrogate_epoch_shape_preserved():
    epoch = np.random.default_rng(0).standard_normal((4, 500))
    surrogate = surrogate_epoch(epoch, method="iaaft", seed=0)
    assert surrogate.shape == epoch.shape
