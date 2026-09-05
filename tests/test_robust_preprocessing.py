import numpy as np

from cessation_manifold.preprocessing.artifact_removal import remove_artifacts


def _artifact_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    sfreq = 200.0
    t = np.arange(0, 10, 1 / sfreq)
    clean = np.vstack(
        [
            np.sin(2 * np.pi * 10 * t),
            0.7 * np.sin(2 * np.pi * 12 * t + 0.2),
            0.5 * np.sin(2 * np.pi * 8 * t + 0.5),
            0.4 * np.sin(2 * np.pi * 15 * t + 0.1),
        ]
    )
    blink = 2.5 * np.exp(-((t - 2.5) ** 2) / 0.02) + 2.0 * np.exp(-((t - 6.5) ** 2) / 0.03)
    muscle = 0.4 * np.sin(2 * np.pi * 35 * t) + 0.2 * rng.standard_normal(t.size)
    artifact = np.vstack([blink, blink, 0.5 * blink, 0.25 * blink]) + muscle
    noisy = clean + artifact
    return clean, noisy, sfreq


def test_artifact_removal_improves_snr_and_keeps_outputs_finite():
    clean, noisy, sfreq = _artifact_dataset()
    result = remove_artifacts(noisy, sfreq, methods=["regression", "wavelet", "ica"], reference=clean)
    assert np.isfinite(result.cleaned_data).all()
    assert result.quality_metrics["finite_after"] is True
    assert result.quality_metrics["snr_improvement_db"] >= 2.0
    assert result.artifact_mask.any()


def test_multiple_artifact_methods_can_be_cross_compared():
    clean, noisy, sfreq = _artifact_dataset()
    methods = {
        name: remove_artifacts(noisy, sfreq, methods=[name], reference=clean).quality_metrics["snr_improvement_db"]
        for name in ("regression", "wavelet", "ica")
    }
    assert all(np.isfinite(list(methods.values())))
    assert methods["wavelet"] > 0.0
    assert methods["ica"] > -0.5


def test_short_epochs_and_nans_fall_back_gracefully():
    epoch = np.array([[np.nan, 1.0, np.inf], [0.0, 0.0, 0.0]])
    result = remove_artifacts(epoch, sfreq=100.0, methods=["wavelet"])
    assert result.cleaned_data.shape[1] >= 16
    assert np.isfinite(result.cleaned_data).all()
    assert result.quality_metrics["finite_after"] is True
