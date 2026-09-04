"""ICC(2,1) for Gate 1 (within-subject reproducibility).

Replaces the ad-hoc "mean subject-std / overall-std < 0.6" metric with a
proper Intraclass Correlation Coefficient (two-way random effects, single
rater; McGraw & Wong 1996 notation ICC(2,1)). Interpretation follows Koo
and Li 2016 (J Chiropr Med 15(2):155-163): < 0.5 poor, 0.5-0.75 moderate,
0.75-0.9 good, > 0.9 excellent. We use > 0.5 (moderate) as the pass
threshold in addition to a permutation-null significance check.
"""
from __future__ import annotations

import numpy as np


def icc_2_1(measurements: np.ndarray) -> dict:
    """measurements: (n_subjects, n_sessions) matrix, one value per session.

    Returns dict with point estimate and 95% CI (large-sample F-distribution
    based, following Shrout & Fleiss 1979). NaN entries are treated as
    missing and rows/subjects with any NaN are dropped for the classical
    balanced-design formula.
    """
    x = np.asarray(measurements, dtype=float)
    mask = ~np.isnan(x).any(axis=1)
    x = x[mask]
    n, k = x.shape
    if n < 2 or k < 2:
        return {"point": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_subjects": int(n), "n_sessions_per_subject": int(k)}

    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    ss_total = ((x - grand) ** 2).sum()
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    icc = (ms_rows - ms_err) / denom if denom > 0 else 0.0

    from scipy.stats import f as f_dist
    alpha = 0.05
    f_l = f_dist.ppf(1 - alpha / 2, n - 1, (n - 1) * (k - 1))
    f_u = f_dist.ppf(1 - alpha / 2, (n - 1) * (k - 1), n - 1)
    fj = ms_rows / ms_err if ms_err > 0 else float("inf")

    n1 = n * (fj / f_l - 1)
    d1 = fj / f_l + (k - 1) + (k * (ms_cols - ms_err) / (n * ms_err) if ms_err > 0 else 0)
    ci_lo = n1 / d1 if d1 != 0 else float("nan")
    n2 = n * (fj * f_u - 1)
    d2 = fj * f_u + (k - 1) + (k * (ms_cols - ms_err) / (n * ms_err) if ms_err > 0 else 0)
    ci_hi = n2 / d2 if d2 != 0 else float("nan")

    return {
        "point": float(np.clip(icc, -1.0, 1.0)),
        "ci_lo": float(np.clip(ci_lo, -1.0, 1.0)),
        "ci_hi": float(np.clip(ci_hi, -1.0, 1.0)),
        "n_subjects": int(n),
        "n_sessions_per_subject": int(k),
    }


def icc_permutation_null(measurements: np.ndarray, n_perm: int = 200, seed: int = 0) -> dict:
    """Shuffle subject labels (within the flat set of subject-session pairs,
    reassigning each session to a random subject) n_perm times, recompute
    ICC(2,1) each time, and return the null distribution + observed p."""
    x = np.asarray(measurements, dtype=float)
    n, k = x.shape
    flat = x.flatten()
    observed = icc_2_1(x)["point"]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(flat).reshape(n, k)
        null[i] = icc_2_1(shuffled)["point"]
    p = float(np.mean(null >= observed))
    return {
        "observed_icc": float(observed),
        "null_p": p,
        "n_permutations": int(n_perm),
        "null_mean": float(np.mean(null)),
        "null_95pct": float(np.percentile(null, 95)),
    }


def gate1_icc(measurements: np.ndarray, n_perm: int = 200, moderate_threshold: float = 0.5, seed: int = 0) -> dict:
    """Full Gate 1: point + CI + permutation null + pass verdict.

    Pass requires BOTH: ICC(2,1) point estimate > moderate_threshold (0.5,
    per Koo & Li 2016) AND permutation null p < 0.05.
    """
    est = icc_2_1(measurements)
    perm = icc_permutation_null(measurements, n_perm=n_perm, seed=seed)
    passed = bool(
        (not np.isnan(est["point"]))
        and est["point"] > moderate_threshold
        and perm["null_p"] < 0.05
    )
    return {
        "icc_point": est["point"],
        "icc_ci_lo": est["ci_lo"],
        "icc_ci_hi": est["ci_hi"],
        "n_subjects": est["n_subjects"],
        "n_sessions_per_subject": est["n_sessions_per_subject"],
        "null_p": perm["null_p"],
        "null_95pct": perm["null_95pct"],
        "n_permutations": perm["n_permutations"],
        "moderate_threshold": moderate_threshold,
        "pass": passed,
    }
