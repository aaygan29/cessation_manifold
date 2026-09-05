import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import cessation_manifold.embed.manifold as manifold
from cessation_manifold.pipeline import run_synthetic_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _load_script_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simulate_subject_sessions_is_deterministic_across_python_processes():
    code = """
import json
from cessation_manifold.io.synthetic import simulate_subject_sessions
sessions = simulate_subject_sessions("sub-test", n_sessions=2, regime="collapsed", n_seconds=5.0, sfreq=50.0, base_seed=7)
payload = {
    "first_session_head": sessions[0].data[:, :5].round(12).tolist(),
    "masks": [int(s.collapse_mask.sum()) for s in sessions],
}
print(json.dumps(payload, sort_keys=True))
"""
    env = {**os.environ, "PYTHONPATH": str(SRC_ROOT)}
    first = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    second = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert first.stdout == second.stdout


def test_validate_lemon_epoch_features_produce_finite_aux_metrics():
    module = _load_script_module("validate_lemon_berger", "scripts/validate_lemon_berger.py")
    fs = 250.0
    t = np.arange(0, 4, 1 / fs)
    signal = np.vstack([
        np.sin(2 * np.pi * 10 * t),
        0.5 * np.sin(2 * np.pi * 12 * t + 0.3),
        0.2 * np.random.default_rng(0).standard_normal(len(t)),
    ])
    feats = module.epoch_features(signal, fs)
    for key in ("alpha_rel", "alpha_abs_uv2", "aperiodic_exponent", "lempel_ziv", "dfa_alpha"):
        assert key in feats
        assert math.isfinite(feats[key]), key


def test_validate_sleep_epoch_features_produce_finite_aux_metrics():
    module = _load_script_module("validate_sleep_edfx", "scripts/validate_sleep_edfx.py")
    fs = 100.0
    t = np.arange(0, 30, 1 / fs)
    x = (
        np.sin(2 * np.pi * 10 * t)
        + 0.4 * np.sin(2 * np.pi * 2 * t + 0.1)
        + 0.2 * np.random.default_rng(1).standard_normal(len(t))
    )
    feats = module.extract_epoch_features(x, fs)
    for key in ("alpha_rel", "aperiodic_exponent", "lempel_ziv", "dfa_alpha"):
        assert key in feats
        assert math.isfinite(feats[key]), key


def test_multi_seed_validation_retains_signal_against_surrogates_on_average():
    old_has_umap = manifold._HAS_UMAP
    manifold._HAS_UMAP = False
    try:
        config = {
            "synthetic": {
                "n_subjects": 3,
                "n_sessions_per_subject": 2,
                "sfreq": 100.0,
                "n_seconds": 30.0,
                "seed": 0,
            }
        }
        results = [run_synthetic_pipeline(config, seed=seed) for seed in (0, 1, 2)]
    finally:
        manifold._HAS_UMAP = old_has_umap

    summary = {
        "mean_real": float(np.mean([r["gate3_real_mean_distance"] for r in results])),
        "mean_surr": float(np.mean([r["gate3_surrogate_mean_distance"] for r in results])),
        "coverages": [float(r["gate4_conformal_coverage"]) for r in results],
        "gate1_points": [float(r["gate1_icc"]["icc_point"]) for r in results],
    }

    assert summary["mean_surr"] > summary["mean_real"]
    assert all(0.0 <= c <= 1.0 for c in summary["coverages"])
    assert all(np.isfinite(summary["gate1_points"]))

