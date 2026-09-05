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
from .honesty.adaptive_conformal import AdaptiveConformalPredictor
from .honesty.gates import gate, UnvalidatedClaimError
from .honesty.icc import gate1_icc
from .preprocessing.artifact_removal import remove_artifacts
from .preprocessing.robustness import ensure_feature_dict_finite, sanitize_epoch


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _epoch_the_session(session, epoch_len_s: float = 5.0):
    """Cut a synthetic session into fixed-length epochs, majority-voting the
    collapse mask onto each epoch.

    Returns (epochs, labels, fractions): labels is the boolean majority-vote
    label used for stratification, fractions is the continuous per-epoch mean
    of the collapse mask, used as the Gate 4 conformal regression target so
    that target is not derived from the same features that produced the
    manifold distance."""
    n_samples = session.data.shape[1]
    step = int(epoch_len_s * session.sfreq)
    epochs, labels, fractions = [], [], []
    for start in range(0, n_samples - step, step):
        epochs.append(session.data[:, start : start + step])
        frac_collapsed = session.collapse_mask[start : start + step].mean()
        labels.append(frac_collapsed > 0.5)
        fractions.append(float(frac_collapsed))
    return epochs, np.array(labels, dtype=bool), np.array(fractions, dtype=float)


def extract_features(epoch: np.ndarray, sfreq: float, preprocessing_config: dict | None = None) -> dict:
    preprocessing_config = preprocessing_config or {}
    sanitized = sanitize_epoch(epoch, min_samples=max(16, int(sfreq // 2)))
    epoch = sanitized.epoch
    if preprocessing_config.get("enabled", False):
        cleaned, _mask, _metrics, _provenance = remove_artifacts(
            epoch,
            sfreq=sfreq,
            methods=preprocessing_config.get("methods", ("ica", "wavelet")),
            fallback_on_failure=preprocessing_config.get("fallback_on_failure", True),
            seed=preprocessing_config.get("seed", 0),
        )
        epoch = cleaned
    feats = {}
    feats.update(aperiodic_features(epoch, sfreq))
    feats.update(complexity_features(epoch, sfreq))
    feats.update(criticality_features(epoch, sfreq))
    return ensure_feature_dict_finite(feats)[0]


def features_to_matrix(feature_dicts: list):
    names = sorted(feature_dicts[0].keys())
    X = np.array([[d[n] for n in names] for d in feature_dicts])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def run_synthetic_pipeline(config: dict, seed: int | None = None) -> dict:
    """Runs Gates 1, 3, 4 on synthetic Kuramoto data and returns a results dict.

    `seed` overrides `config["synthetic"]["seed"]` when given, so a seed sweep
    does not need to hand-edit config files."""
    cfg = config.get("synthetic", {})
    n_subjects = cfg.get("n_subjects", 6)
    n_sessions = cfg.get("n_sessions_per_subject", 3)
    sfreq = cfg.get("sfreq", 250.0)
    n_seconds = cfg.get("n_seconds", 60.0)
    seed = cfg.get("seed", 0) if seed is None else seed
    preprocessing_cfg = config.get("preprocessing", {})
    conformal_cfg = config.get("conformal", {})

    # --- Gate 1: within-subject reproducibility across synthetic "sessions" ---
    all_epochs, all_labels, all_fractions, subject_ids, session_ids = [], [], [], [], []
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
            eps, labels, fractions = _epoch_the_session(sess)
            all_epochs.extend(eps)
            all_labels.extend(labels)
            all_fractions.extend(fractions)
            subject_ids.extend([sess.subject_id] * len(eps))
            session_ids.extend([sess.session_id] * len(eps))
    all_labels = np.array(all_labels)
    all_fractions = np.array(all_fractions, dtype=float)

    feature_dicts = [extract_features(ep, sfreq, preprocessing_config=preprocessing_cfg) for ep in all_epochs]
    X, feature_names = features_to_matrix(feature_dicts)

    model, Xr = fit_manifold(X, all_labels, feature_names, seed=seed)
    dist = distance_from_cessation(Xr, model.cessation_centroid)

    # Gate 1: within-subject reproducibility across sessions, quantified as
    # ICC(2,1) on per-session mean distance (Koo & Li 2016 J Chiropr Med
    # 15(2):155-163: > 0.5 moderate, > 0.75 good). Pass requires point > 0.5
    # AND permutation-null p < 0.05. Old ad-hoc ratio kept as diagnostic only.
    import pandas as pd

    df = pd.DataFrame(
        {"subject": subject_ids, "session": session_ids, "label": all_labels, "distance": dist}
    )
    cess_df = df[df["label"]]
    session_means = cess_df.groupby(["subject", "session"])["distance"].mean().reset_index()

    pivot = session_means.pivot(index="subject", columns="session", values="distance")
    icc_result = gate1_icc(pivot.values, n_perm=200, moderate_threshold=0.5, seed=seed)

    subj_std = session_means.groupby("subject")["distance"].std().fillna(0.0)
    overall_scale = session_means["distance"].std() + 1e-9
    diag_ratio = float((subj_std / overall_scale).mean())

    # --- Gate 3: surrogates must break the score ---
    surrogate_epochs = [surrogate_epoch(ep, method="iaaft", seed=seed + i) for i, ep in enumerate(all_epochs[:60])]
    surr_feature_dicts = [extract_features(ep, sfreq) for ep in surrogate_epochs]
    Xs, _ = features_to_matrix(surr_feature_dicts)
    Xs_r = transform(model, Xs)
    dist_surr = distance_from_cessation(Xs_r, model.cessation_centroid)

    real_cess_dist = dist[all_labels][:60] if all_labels.sum() >= 1 else dist[:60]
    gate3_pass = bool(np.mean(dist_surr) > np.mean(real_cess_dist) * 1.2)

    # --- Gate 4: split-conformal coverage ---
    # Target is the continuous per-epoch collapse fraction (from the raw collapse
    # mask), not the manifold distance derived from the same X. Regressing dist
    # on X gave coverage 1.0 because the target was a near-deterministic function
    # of the inputs (target leakage); collapse_fraction is an independent label.
    y = all_fractions
    block_structure = {"subject": np.array(subject_ids), "session": np.array(session_ids)}
    predictor = AdaptiveConformalPredictor(
        n_epochs=len(X),
        block_structure=block_structure,
        target_coverage=conformal_cfg.get("target_coverage", 0.9),
        adaptive_sizing=conformal_cfg.get("adaptive_sizing", True),
        n_splits=conformal_cfg.get("n_splits", 5),
        stability_std_threshold=conformal_cfg.get("stability_std_threshold", 0.05),
        seed=seed,
    ).fit_from_full_data(X, y)
    split = predictor.last_split_
    coverage_eval = predictor.evaluate(
        X[split["test"]],
        y[split["test"]],
        test_blocks=split["blocks"][split["test"]],
    )
    coverage = coverage_eval["coverage"]
    gate4_pass = bool(coverage >= predictor.target_coverage - 0.05)

    provenance_config = {"synthetic": cfg}
    try:
        point, lo, hi = predictor.predict_interval(X[split["test"]][:1])
        finding = gate(
            value=float(point[0]),
            lower=float(lo[0]),
            upper=float(hi[0]),
            coverage_target=predictor.target_coverage,
            coverage_achieved=coverage,
            dataset_id="synthetic-kuramoto",
            config=provenance_config,
            extra={"conformal_diagnostics": predictor.diagnostics_},
        )
        finding_dict = finding.to_dict()
    except UnvalidatedClaimError as e:
        finding_dict = {"status": "UNVALIDATED", "reason": str(e)}

    return {
        "gate1_icc": icc_result,
        "gate1_within_subject_ratio": diag_ratio,
        "gate1_pass": icc_result["pass"],
        "gate3_surrogate_mean_distance": float(np.mean(dist_surr)),
        "gate3_real_mean_distance": float(np.mean(real_cess_dist)),
        "gate3_pass": gate3_pass,
        "gate4_conformal_coverage": coverage,
        "gate4_block_coverage": coverage_eval["block_coverage"],
        "gate4_diagnostics": predictor.diagnostics_,
        "gate4_unstable_for_review": not predictor.diagnostics_.get("threshold_stable", True),
        "gate4_pass": gate4_pass,
        "example_finding": finding_dict,
        "n_epochs": int(len(X)),
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
        eps, labels, _fractions = _epoch_the_session(sess)
        collapsed_epochs.extend(eps)
        collapsed_labels.extend(labels)
    collapsed_labels = np.array(collapsed_labels, dtype=bool)

    baseline_sess = simulate_subject_sessions(
        "baseline-critical", n_sessions=1, regime="critical", base_seed=seed + 500, n_seconds=n_seconds, sfreq=sfreq
    )[0]
    baseline_epochs, _, _ = _epoch_the_session(baseline_sess)

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
