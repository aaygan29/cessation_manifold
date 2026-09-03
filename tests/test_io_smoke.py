"""IO smoke tests: synthetic generator shapes, and the real-data loader fails
loudly (not silently) when local data has not been fetched."""
import pytest

from cessation_manifold.io.synthetic import simulate_kuramoto_eeg
from cessation_manifold.io.bids_loader import load_bids_eeg


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
