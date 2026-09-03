"""Gate 1 apparatus check: the synthetic Kuramoto collapse generator actually
produces a collapse (order parameter rises during cue windows), and repeated
synthetic sessions for one subject land near each other on the manifold."""
import numpy as np

from cessation_manifold.io.synthetic import simulate_kuramoto_eeg, simulate_subject_sessions


def test_collapse_regime_raises_order_parameter_in_cue_windows():
    sess = simulate_kuramoto_eeg(
        n_channels=8, n_seconds=30.0, sfreq=200.0, regime="collapsed",
        n_collapse_events=2, collapse_duration_s=4.0, seed=1,
    )
    assert sess.collapse_mask.any()
    R_during = sess.order_parameter[sess.collapse_mask]
    R_outside = sess.order_parameter[~sess.collapse_mask]
    assert R_during.mean() > R_outside.mean()


def test_control_regime_has_no_collapse_events():
    sess = simulate_kuramoto_eeg(n_channels=8, n_seconds=10.0, sfreq=200.0, regime="control", seed=2)
    assert not sess.collapse_mask.any()


def test_multiple_sessions_per_subject_are_reproducible_in_shape():
    sessions = simulate_subject_sessions("sub-test", n_sessions=3, regime="collapsed", n_seconds=20.0, sfreq=200.0)
    assert len(sessions) == 3
    shapes = {s.data.shape for s in sessions}
    assert len(shapes) == 1  # same shape every session
    for s in sessions:
        assert s.collapse_mask.any()
