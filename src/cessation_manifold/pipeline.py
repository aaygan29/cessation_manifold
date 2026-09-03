"""End-to-end pipeline: config -> features -> embed -> distance -> conformal -> report.

Runs equally on synthetic data (regime-labeled Kuramoto sessions) or on a
loaded BIDS dataset once a config points `data.source` at one. Real-data
configs (lemon, openneuro_meditation) currently only wire the loader and
feature extraction; the manifold/conformal steps need a labeled cessation
window to anchor on, which synthetic data provides today and real cessation
annotations (Zarka, NIMHANS) will provide once that data lands.
"""
from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path

from .io.synthetic import simulate_subject_sessions
from .features.aperiodic import aperiodic_features
from .features.complexity import complexity_features
from .features.criticality import criticality_features
from .features.surrogates import surrogate_epoch
from .embed.manifold import fit_manifold, transform
from .embed.distance import distance_from_cessation
from .honesty.conformal import fit_split_conformal, empirical_coverage
from .honesty.gates import gate, UnvalidatedClaimError


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _epoch_the_session(session, epoch_len_s: float = 5.0):
    """Cut a synthetic session into fixed-length epochs, majority-voting the
    collapse mask onto each epoch."""
    n_samples = session.data.shape[1]
    step = int(epoch_len_s * session.sfreq)
    epochs, labels = [], []
    for start in range(0, n_samples - step, step):
        epochs.append(session.data[:, start : start + step])
        frac_collapsed = session.collapse_mask[start : start + step].mean()
        labels.append(frac_collapsed > 0.5)
    return epochs, np.array(labels, dtype=bool)


def extract_features(epoch: np.ndarray, sfreq: float) -> dict:
    feats = {}
    feats.update(aperiodic_features(epoch, sfreq))
    feats.update(complexity_features(epoch, sfreq))
    feats.update(criticality_features(epoch, sfreq))
    return feats


def features_to_matrix(feature_dicts: list):
    names = sorted(feature_dicts[0].keys())
    X = np.array([[d[n] for n in names] for d in feature_dicts])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def run_synthetic_pipeline(config: dict) -> dict:
    """Runs Gates 1, 3, 4 on synthetic Kuramoto data and returns a results dict."""
    cfg = config.get("synthetic", {})
    n_subjects = cfg.get("n_subjects", 6)
    n_sessions = cfg.get("n_sessions_per_subject", 3)
    sfreq = cfg.get("sfreq", 250.0)
    n_seconds = cfg.get("n_seconds", 60.0)
    seed = cfg.get("seed", 0)

    # --- Gate 1: within-subject reproducibility across synthetic "sessions" ---
    all_epochs, all_labels, subject_ids, session_ids = [], [], [], []
    for s in range(n_subjects):
        sessions = simulate_subject_sessions(
            subject_id=f"synthsub-{s:02d}",
            n_sessions=n_sessions,
            regime="collapsed",
            base_seed=seed + 100 * s,
            n_seconds=n_seconds,
            sfreq=sfreq,
        )
        for sess in sessions:
            eps, labels = _epoch_the_session(sess)
            all_epochs.extend(eps)
            all_labels.extend(labels)
            subject_ids.extend([sess.subject_id] * len(eps))
            session_ids.extend([sess.session_id] * len(eps))
    all_labels = np.array(all_labels)

    feature_dicts = [extract_features(ep, sfreq) for ep in all_epochs]
    X, feature_names = features_to_matrix(feature_dicts)

    model, Xr = fit_manifold(X, all_labels, feature_names, seed=seed)
    dist = distance_from_cessation(Xr, model.cessation_centroid)

    # per-subject, per-session mean distance on cessation epochs: gate1 check
    import pandas as pd

    df = pd.DataFrame(
        {"subject": subject_ids, "session": session_ids, "label": all_labels, "distance": dist}
    )
    cess_df = df[df["label"]]
    session_means = cess_df.groupby(["subject", "session"])["distance"].mean().reset_index()
    subj_std = session_means.groupby("subject")["distance"].std().fillna(0.0)
    overall_scale = session_means["distance"].std() + 1e-9
    gate1_ratio = float((subj_std / overall_scale).mean())
    gate1_pass = bool(gate1_ratio < 0.6)  # within-subject spread should be well below overall spread

    # --- Gate 3: surrogates must break the score ---
    surrogate_epochs = [surrogate_epoch(ep, method="iaaft", seed=seed + i) for i, ep in enumerate(all_epochs[:60])]
    surr_feature_dicts = [extract_features(ep, sfreq) for ep in surrogate_epochs]
    Xs, _ = features_to_matrix(surr_feature_dicts)
    Xs_r = transform(model, Xs)
    dist_surr = distance_from_cessation(Xs_r, model.cessation_centroid)

    real_cess_dist = dist[all_labels][:60] if all_labels.sum() >= 1 else dist[:60]
    gate3_pass = bool(np.mean(dist_surr) > np.mean(real_cess_dist) * 1.2)

    # --- Gate 4: split-conformal coverage ---
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    n_train, n_calib = int(n * 0.5), int(n * 0.25)
    train_idx, calib_idx, test_idx = idx[:n_train], idx[n_train : n_train + n_calib], idx[n_train + n_calib :]

    y = dist  # regress the manifold distance itself from raw features (self-consistency check)
    predictor = fit_split_conformal(X[train_idx], y[train_idx], X[calib_idx], y[calib_idx], alpha=0.1, seed=seed)
    coverage = empirical_coverage(predictor, X[test_idx], y[test_idx])
    gate4_pass = bool(coverage >= 0.9 - 0.05)

    provenance_config = {"synthetic": cfg}
    try:
        point, lo, hi = predictor.predict_interval(X[test_idx][:1])
        finding = gate(
            value=float(point[0]),
            lower=float(lo[0]),
            upper=float(hi[0]),
            coverage_target=0.9,
            coverage_achieved=coverage,
            dataset_id="synthetic-kuramoto",
            config=provenance_config,
        )
        finding_dict = finding.to_dict()
    except UnvalidatedClaimError as e:
        finding_dict = {"status": "UNVALIDATED", "reason": str(e)}

    return {
        "gate1_within_subject_ratio": gate1_ratio,
        "gate1_pass": gate1_pass,
        "gate3_surrogate_mean_distance": float(np.mean(dist_surr)),
        "gate3_real_mean_distance": float(np.mean(real_cess_dist)),
        "gate3_pass": gate3_pass,
        "gate4_conformal_coverage": coverage,
        "gate4_pass": gate4_pass,
        "example_finding": finding_dict,
        "n_epochs": int(n),
        "n_subjects": n_subjects,
    }


def run_gate2(config: dict, real_control_features: np.ndarray | None = None) -> dict:
    """Gate 2: non-meditator controls should sit further from the manifold than
    meditator baseline. v0 uses synthetic 'critical' regime as the meditator-baseline
    positive contrast, and real resting-state EEG (if features are supplied) as the
    non-meditator control. Returns a partial result and says so when real features
    are not supplied."""
    cfg = config.get("synthetic", {})
    sfreq = cfg.get("sfreq", 250.0)
    n_seconds = cfg.get("n_seconds", 60.0)
    seed = cfg.get("seed", 0)

    collapsed_epochs, collapsed_labels = [], []
    for s in range(3):
        sess = simulate_subject_sessions(
            f"anchor-{s}", n_sessions=1, regime="collapsed", base_seed=seed + s, n_seconds=n_seconds, sfreq=sfreq
        )[0]
        eps, labels = _epoch_the_session(sess)
        collapsed_epochs.extend(eps)
        collapsed_labels.extend(labels)
    collapsed_labels = np.array(collapsed_labels, dtype=bool)

    baseline_sess = simulate_subject_sessions(
        "baseline-critical", n_sessions=1, regime="critical", base_seed=seed + 500, n_seconds=n_seconds, sfreq=sfreq
    )[0]
    baseline_epochs, _ = _epoch_the_session(baseline_sess)

    anchor_feats = [extract_features(ep, sfreq) for ep in collapsed_epochs]
    baseline_feats = [extract_features(ep, sfreq) for ep in baseline_epochs]

    all_dicts = anchor_feats + baseline_feats
    X, names = features_to_matrix(all_dicts)
    n_anchor = len(anchor_feats)
    label_mask = np.zeros(len(all_dicts), dtype=bool)
    label_mask[:n_anchor] = collapsed_labels

    model, Xr = fit_manifold(X, label_mask, names, seed=seed)
    dist = distance_from_cessation(Xr, model.cessation_centroid)
    baseline_dist = dist[n_anchor:]

    result = {
        "synthetic_meditator_baseline_mean_distance": float(np.mean(baseline_dist)),
        "real_control_available": real_control_features is not None,
    }

    if real_control_features is not None:
        Xc = np.nan_to_num(real_control_features)
        Xc_r = transform(model, Xc)
        control_dist = distance_from_cessation(Xc_r, model.cessation_centroid)
        result["real_control_mean_distance"] = float(np.mean(control_dist))
        result["gate2_pass"] = bool(np.mean(control_dist) > np.mean(baseline_dist))
    else:
        result["gate2_pass"] = None
        result["gate2_note"] = (
            "PARTIAL: real non-meditator control features not supplied. Wire "
            "scripts/fetch_lemon_subset.sh output through io/bids_loader.py and "
            "pass its feature matrix here to complete Gate 2."
        )

    return result
