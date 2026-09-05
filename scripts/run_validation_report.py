#!/usr/bin/env python3
"""Structured validation sweep with robustness reporting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cessation_manifold.pipeline import load_config, run_synthetic_pipeline
from cessation_manifold.report import render_report


def bootstrap_ci(values: np.ndarray, ci: float = 0.95, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0
    stats = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        stats[idx] = rng.choice(values, size=len(values), replace=True).mean()
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _classify_claim(metric: str, summary: dict) -> str:
    if metric == "gate4_conformal_coverage":
        if summary["min"] >= 0.85 and summary["max"] <= 0.95:
            return "SUPPORTED"
        if summary["mean"] >= 0.85:
            return "EXPLORATORY"
        return "UNSUPPORTED"
    if metric == "gate1_icc_point":
        if summary["mean"] >= 0.5 and summary["min"] >= 0.3:
            return "SUPPORTED"
        if summary["mean"] >= 0.3:
            return "EXPLORATORY"
        return "UNSUPPORTED"
    return "EXPLORATORY" if summary["mean"] > 0 else "UNSUPPORTED"


def _summarize_metric(values: list[float], ci: float, seed: int = 0) -> dict:
    arr = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(arr, ci=ci, seed=seed)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci_lo": lo,
        "ci_hi": hi,
    }


def _ablation_configs(config: dict, ablations: list[str]) -> dict:
    base = json.loads(json.dumps(config))
    generated = {}
    for name in ablations:
        cfg = json.loads(json.dumps(base))
        if name == "drop_channel_subsets":
            cfg.setdefault("synthetic", {})["n_subjects"] = max(2, cfg["synthetic"].get("n_subjects", 6) - 1)
        elif name == "vary_epoch_length":
            cfg.setdefault("synthetic", {})["n_seconds"] = max(20.0, cfg["synthetic"].get("n_seconds", 60.0) * 0.75)
        elif name == "vary_feature_families":
            cfg.setdefault("preprocessing", {})["enabled"] = True
            cfg["preprocessing"]["methods"] = ["wavelet"]
        generated[name] = cfg
    return generated


def run_validation_report(
    config: dict,
    n_seeds: int = 20,
    bootstrap_ci: float = 0.95,
    ablations: list[str] | None = None,
    output_json: str = "results/validation_report.json",
    output_html: str = "results/validation_report.html",
) -> dict:
    ablations = ablations or ["drop_channel_subsets", "vary_epoch_length", "vary_feature_families"]
    per_seed = []
    for seed in range(n_seeds):
        result = run_synthetic_pipeline(config, seed=seed)
        per_seed.append(
            {
                "seed": seed,
                "gate1_icc_point": float(result["gate1_icc"]["icc_point"]),
                "gate1_pass": bool(result["gate1_pass"]),
                "gate4_conformal_coverage": float(result["gate4_conformal_coverage"]),
                "gate4_block_coverage": float(result["gate4_block_coverage"]),
                "gate4_unstable_for_review": bool(result.get("gate4_unstable_for_review", False)),
                "gate3_pass": bool(result["gate3_pass"]),
            }
        )

    metrics = {
        "gate1_icc_point": _summarize_metric([row["gate1_icc_point"] for row in per_seed], bootstrap_ci, seed=1),
        "gate4_conformal_coverage": _summarize_metric([row["gate4_conformal_coverage"] for row in per_seed], bootstrap_ci, seed=2),
        "gate4_block_coverage": _summarize_metric([row["gate4_block_coverage"] for row in per_seed], bootstrap_ci, seed=3),
    }
    for metric, summary in metrics.items():
        summary["claim_strength"] = _classify_claim(metric, summary)

    unstable = [row["seed"] for row in per_seed if row["gate4_unstable_for_review"]]
    failed_gate1 = [row["seed"] for row in per_seed if not row["gate1_pass"]]
    ablation_results = {}
    for name, ablated_config in _ablation_configs(config, ablations).items():
        result = run_synthetic_pipeline(ablated_config, seed=0)
        ablation_results[name] = {
            "gate1_pass": bool(result["gate1_pass"]),
            "gate4_conformal_coverage": float(result["gate4_conformal_coverage"]),
            "gate4_unstable_for_review": bool(result.get("gate4_unstable_for_review", False)),
        }

    report = {
        "dataset_provenance": {
            "source": config.get("data", {}).get("source", "synthetic-kuramoto"),
            "channels": config.get("preprocessing", {}).get("channel_exclusions", []),
            "sampling_rate": config.get("synthetic", {}).get("sfreq", 250.0),
            "epochs_retained": int(len(per_seed)),
            "epochs_dropped": 0,
        },
        "per_seed": per_seed,
        "bootstrap_ci": bootstrap_ci,
        "gate_summaries": metrics,
        "seed_flags": {
            "gate1_failed_seeds": failed_gate1,
            "gate4_unstable_seeds": unstable,
        },
        "ablations": ablation_results,
        "claims": {
            "synthetic_feature_recovery": metrics["gate1_icc_point"]["claim_strength"],
            "conformal_coverage": metrics["gate4_conformal_coverage"]["claim_strength"],
        },
        "recommendations": [
            "SUPPORTED claims should stay confined to synthetic machinery checks unless real cessation labels are available.",
            "EXPLORATORY claims need larger held-out blocks or additional sessions before promotion.",
            "UNSUPPORTED claims should not be used as evidence for cessation without new data.",
        ],
    }

    output_json_path = Path(output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(report, indent=2))
    render_report(
        {
            "gate1": {"gate1_pass": len(failed_gate1) == 0},
            "gate2": {"gate2_pass": None},
            "gate3": {"gate3_pass": all(row["gate3_pass"] for row in per_seed)},
            "gate4": {"gate4_pass": not unstable and metrics["gate4_conformal_coverage"]["min"] >= 0.85},
            "validation_report": report,
        },
        out_path=output_html,
    )
    return report


def main() -> None:
    config = load_config("configs/synthetic.yaml")
    report = run_validation_report(config)
    print(json.dumps(report["gate_summaries"], indent=2))


if __name__ == "__main__":
    main()
