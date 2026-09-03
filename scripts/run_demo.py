#!/usr/bin/env python3
"""Runs the end-to-end demo: Gates 1, 2 (partial), 3, 4 on synthetic data,
writes results/report.html."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cessation_manifold.pipeline import load_config, run_synthetic_pipeline, run_gate2
from cessation_manifold.report import render_report


def main():
    config = load_config("configs/synthetic.yaml")

    print("Running Gates 1, 3, 4 on synthetic Kuramoto data ...")
    main_results = run_synthetic_pipeline(config)

    print("Running Gate 2 (synthetic meditator-baseline contrast; real control partial) ...")
    gate2_results = run_gate2(config, real_control_features=None)

    results = {
        "gate1": {
            "gate1_within_subject_ratio": main_results["gate1_within_subject_ratio"],
            "gate1_pass": main_results["gate1_pass"],
        },
        "gate2": gate2_results,
        "gate3": {
            "gate3_surrogate_mean_distance": main_results["gate3_surrogate_mean_distance"],
            "gate3_real_mean_distance": main_results["gate3_real_mean_distance"],
            "gate3_pass": main_results["gate3_pass"],
        },
        "gate4": {
            "gate4_conformal_coverage": main_results["gate4_conformal_coverage"],
            "gate4_pass": main_results["gate4_pass"],
        },
        "example_finding": main_results["example_finding"],
        "n_epochs": main_results["n_epochs"],
        "n_subjects": main_results["n_subjects"],
    }

    Path("results").mkdir(exist_ok=True)
    Path("results/results.json").write_text(json.dumps(results, indent=2, default=str))
    render_report(results, "results/report.html")

    print(json.dumps(results, indent=2, default=str))
    print("\nWrote results/results.json and results/report.html")


if __name__ == "__main__":
    main()
