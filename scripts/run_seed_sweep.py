#!/usr/bin/env python3
"""20-seed sweep of the synthetic pipeline.

Runs run_synthetic_pipeline for seeds 0..19 and reports mean +/- 95% CI
(bootstrap, 1000 iterations) of gate1_within_subject_ratio,
gate3_surrogate_mean_distance, gate3_real_mean_distance, and
gate4_conformal_coverage. A single-seed run (run_demo.py) can look good or
bad by chance; this checks whether the gate numbers are stable across seeds.

Writes results/seed_sweep.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cessation_manifold.pipeline import load_config, run_synthetic_pipeline

N_SEEDS = 20
N_BOOTSTRAP = 1000
METRICS = [
    "gate1_within_subject_ratio",
    "gate3_surrogate_mean_distance",
    "gate3_real_mean_distance",
    "gate4_conformal_coverage",
]


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    config = load_config("configs/synthetic.yaml")

    per_seed = {m: [] for m in METRICS}
    per_seed_results = []

    for seed in range(N_SEEDS):
        print(f"seed {seed} ...")
        result = run_synthetic_pipeline(config, seed=seed)
        row = {m: result[m] for m in METRICS}
        row["seed"] = seed
        per_seed_results.append(row)
        for m in METRICS:
            per_seed[m].append(result[m])
        print(f"  gate1_ratio={row['gate1_within_subject_ratio']:.3f} "
              f"gate3_surr={row['gate3_surrogate_mean_distance']:.3f} "
              f"gate3_real={row['gate3_real_mean_distance']:.3f} "
              f"gate4_coverage={row['gate4_conformal_coverage']:.3f}")

    summary = {}
    for m in METRICS:
        arr = np.array(per_seed[m], dtype=float)
        lo, hi = bootstrap_ci(arr)
        summary[m] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    out = {
        "n_seeds": N_SEEDS,
        "n_bootstrap": N_BOOTSTRAP,
        "summary": summary,
        "per_seed": per_seed_results,
    }

    out_path = Path("results/seed_sweep.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))

    print("\nSummary (mean, 95% CI over 20 seeds):")
    for m in METRICS:
        s = summary[m]
        print(f"  {m}: {s['mean']:.3f} [{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}] "
              f"(min {s['min']:.3f}, max {s['max']:.3f})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
