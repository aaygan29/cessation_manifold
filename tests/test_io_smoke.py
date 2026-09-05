"""IO smoke tests: synthetic generator shapes, and the real-data loader fails
loudly (not silently) when local data has not been fetched."""
import pytest
import numpy as np

from cessation_manifold.io.synthetic import simulate_kuramoto_eeg
from cessation_manifold.io.bids_loader import load_bids_eeg, load_uploaded_eeg
from cessation_manifold.pipeline import run_real_data_pipeline, run_synthetic_pipeline


def test_synthetic_shapes_are_consistent():
    sess = simulate_kuramoto_eeg(n_channels=6, n_seconds=5.0, sfreq=200.0, regime="control", seed=0)
    assert sess.data.shape == (6, 1000)
    assert len(sess.order_parameter) == 1000
    assert len(sess.collapse_mask) == 1000
    assert len(sess.ch_names) == 6


def test_bids_loader_raises_clear_error_when_data_missing(tmp_path):
    missing_root = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="fetch script"):
        load_bids_eeg(str(missing_root), subject="01", task="rest", dataset_id="unit-test")


def test_uploaded_loader_autodetects_basic_metadata(tmp_path):
    import mne

    sfreq = 100.0
    info = mne.create_info(["C3", "C4"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.random.default_rng(0).standard_normal((2, 800)), info, verbose=False)
    fif_path = tmp_path / "sample_raw.fif"
    raw.save(fif_path, overwrite=True, verbose=False)

    loaded = load_uploaded_eeg(str(fif_path), dataset_id="unit-test-upload", epoch_length_s=4.0)
    assert loaded.sfreq == sfreq
    assert loaded.ch_names == ["C3", "C4"]
    assert loaded.source == "unit-test-upload"


def test_synthetic_pipeline_reports_regime_stratified_icc():
    config = {
        "synthetic": {"n_subjects": 2, "n_sessions_per_subject": 2, "sfreq": 80.0, "n_seconds": 20.0, "seed": 0},
        "conformal": {"target_coverage": 0.9, "adaptive_sizing": True, "n_splits": 3},
        "preprocessing": {"enabled": False},
    }
    result = run_synthetic_pipeline(config, seed=0)
    assert "gate1_regime_icc" in result
    assert {"collapsed", "critical", "control"} <= set(result["gate1_regime_icc"].keys())
    assert result["gate1_anchor_size"]["n_subjects"] == 2


def test_real_pipeline_runs_with_uploaded_fif(tmp_path):
    import mne

    sfreq = 128.0
    data = np.random.default_rng(1).standard_normal((3, 1280))
    info = mne.create_info(["F3", "F4", "Pz"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    fif_path = tmp_path / "uploaded_raw.fif"
    raw.save(fif_path, overwrite=True, verbose=False)

    config = {"data": {"source": "unit-test", "epoch_length_s": 2.0}, "preprocessing": {"seed": 0}}
    result = run_real_data_pipeline(config, str(fif_path), subject_id="unit-sub")
    assert result["feature_matrix_finite"] is True
    assert result["n_epochs"] > 0
    assert result["feature_reliability"]
    assert result["claim_scope"] == "feature-validation-only"
