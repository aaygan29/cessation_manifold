import importlib.util
from pathlib import Path

import cessation_manifold.embed.manifold as manifold


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validation_report_contains_seed_flags_and_claim_strengths():
    module = _load_script_module("run_validation_report", "scripts/run_validation_report.py")
    old_has_umap = manifold._HAS_UMAP
    manifold._HAS_UMAP = False
    try:
        report = module.run_validation_report(
            config={
                "synthetic": {"n_subjects": 2, "n_sessions_per_subject": 2, "sfreq": 80.0, "n_seconds": 20.0, "seed": 0},
                "preprocessing": {"enabled": False},
                "conformal": {"target_coverage": 0.9, "adaptive_sizing": True, "n_splits": 3},
            },
            n_seeds=2,
            bootstrap_ci=0.95,
            output_json=str(REPO_ROOT / "results" / "validation_report_test.json"),
            output_html=str(REPO_ROOT / "results" / "validation_report_test.html"),
        )
    finally:
        manifold._HAS_UMAP = old_has_umap

    assert "gate_summaries" in report
    assert "gate1_icc_point" in report["gate_summaries"]
    assert "gate4_conformal_coverage" in report["gate_summaries"]
    assert report["gate_summaries"]["gate1_icc_point"]["claim_strength"] in {"SUPPORTED", "EXPLORATORY", "UNSUPPORTED"}
    assert report["gate_summaries"]["gate4_conformal_coverage"]["claim_strength"] in {"SUPPORTED", "EXPLORATORY", "UNSUPPORTED"}
    assert "gate1_failed_seeds" in report["seed_flags"]
    assert "gate4_unstable_seeds" in report["seed_flags"]
    assert report["ablations"]
