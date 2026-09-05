import numpy as np
import pytest

from cessation_manifold.io.synthetic import simulate_subject_sessions, simulate_synthetic_eeg
from cessation_manifold.pipeline import extract_features, run_synthetic_pipeline


def test_synthetic_models_generate_consistent_shapes():
    for model in ("kuramoto", "neural_mass", "thalamo_cortical"):
        sess = simulate_synthetic_eeg(
            model=model,
            n_channels=6,
            n_seconds=5.0,
            sfreq=100.0,
            regime="collapsed",
            seed=0,
        )
        assert sess.data.shape == (6, 500)
        assert sess.order_parameter.shape[0] == 500
        assert sess.model_name == model


def test_synthetic_dispatcher_alias_and_invalid_name():
    alias_session = simulate_synthetic_eeg(
        model="thalamocortical",
        n_channels=4,
        n_seconds=3.0,
        sfreq=100.0,
        regime="control",
        seed=3,
    )
    assert alias_session.model_name == "thalamo_cortical"
    with pytest.raises(ValueError, match="supported models"):
        simulate_synthetic_eeg(model="unknown_model")


def test_expanded_feature_stack_contains_connectivity_and_cross_frequency():
    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((4, 600))
    feats = extract_features(epoch, sfreq=100.0, preprocessing_config={})
    assert "wsmi_proxy_mean" in feats
    assert "phase_lag_index_mean" in feats
    assert "theta_gamma_pac_mi" in feats
    assert "alpha_state_transition_count" in feats


def test_pipeline_reports_new_validity_and_conditional_coverage_blocks():
    config = {
        "synthetic": {
            "n_subjects": 2,
            "n_sessions_per_subject": 2,
            "sfreq": 80.0,
            "n_seconds": 20.0,
            "seed": 0,
            "model": "neural_mass",
            "model_kwargs": {},
        },
        "conformal": {"target_coverage": 0.9, "adaptive_sizing": True, "n_splits": 3},
        "preprocessing": {"enabled": False},
    }
    result = run_synthetic_pipeline(config, seed=0)
    assert "synthetic_validity" in result
    assert "distribution_shift_diagnostics" in result
    assert "gate4_conditional_coverage" in result
    assert {"subject", "session", "state"} <= set(result["gate4_conditional_coverage"].keys())
    assert isinstance(result["synthetic_validity"]["parameter_recovery"]["available"], bool)
    assert "x_feature_mean_shift_train_test_l1" in result["distribution_shift_diagnostics"]
    assert isinstance(result["distribution_shift_diagnostics"]["y_mean_shift_train_test"], float)


def test_model_kwargs_are_forwarded_into_pipeline_metadata():
    config = {
        "synthetic": {
            "n_subjects": 2,
            "n_sessions_per_subject": 2,
            "sfreq": 80.0,
            "n_seconds": 20.0,
            "seed": 0,
            "model": "neural_mass",
            "model_kwargs": {"collapse_gain": 4.25},
        },
        "conformal": {"target_coverage": 0.9, "adaptive_sizing": True, "n_splits": 3},
        "preprocessing": {"enabled": False},
    }
    result = run_synthetic_pipeline(config, seed=0)
    assert result["synthetic_model"] == "neural_mass"
    assert result["synthetic_model_params"]["collapse_gain"] == 4.25


def test_model_kwargs_change_generated_session_params():
    sessions = simulate_subject_sessions(
        subject_id="sub-kw",
        n_sessions=1,
        regime="collapsed",
        model="neural_mass",
        model_kwargs={"collapse_gain": 4.1},
        sfreq=80.0,
        n_seconds=5.0,
    )
    assert sessions[0].model_params["collapse_gain"] == 4.1


def test_reserved_model_kwargs_are_rejected():
    with pytest.raises(ValueError, match="reserved"):
        simulate_subject_sessions(
            subject_id="sub-kw",
            n_sessions=1,
            regime="collapsed",
            model="kuramoto",
            model_kwargs={"seed": 2},
            sfreq=80.0,
            n_seconds=5.0,
        )
