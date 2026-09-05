"""End-to-end pipeline: config -> features -> embed -> distance -> conformal -> report.

Runs equally on synthetic data (regime-labeled Kuramoto sessions) or on a
loaded BIDS dataset once a config points `data.source` at one. Real-data
configs (lemon, openneuro_meditation) currently only wire the loader and
feature extraction; the manifold/conformal steps need a labeled cessation
window to anchor on, which synthetic data provides today and real cessation
annotations (Zarka, NIMHANS) will provide once that data lands.

References
----------
Koo, T. K., & Li, M. Y. (2016). A Guideline of Selecting and Reporting
Intraclass Correlation Coefficients for Reliability Research. Journal of
Chiropractic Medicine, 15(2), 155-163. https://doi.org/10.1016/j.jcm.2016.02.012

Barber, R. F., Candes, E., Ramdas, A., & Tibshirani, R. J. (2023).
Conformal prediction with conditional guarantees. JRSS Series B.
https://doi.org/10.1093/jrsssb/qkad020
"""
from __future__ import annotations

import numpy as np
import yaml
from pathlib import Path

from .io.synthetic import simulate_subject_sessions
from .features.aperiodic import aperiodic_features
from .features.complexity import complexity_features
from .features.criticality import criticality_features
from .features.connectivity import connectivity_features
from .features.cross_frequency import cross_frequency_features
from .features.microstates import fit_microstate_maps, microstate_features
from .features.surrogates import surrogate_epoch
from .embed.manifold import fit_manifold, transform
from .embed.distance import distance_from_cessation
from .honesty.adaptive_conformal import AdaptiveConformalPredictor
from .honesty.gates import gate, UnvalidatedClaimError
from .honesty.icc import gate1_icc
from .preprocessing.artifact_removal import remove_artifacts, remove_artifacts_mne
from .preprocessing.robustness import ensure_feature_dict_finite, sanitize_epoch
from .io.bids_loader import load_uploaded_eeg


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


def _fit_microstate_maps_from_epochs(epochs: list[np.ndarray], n_states: int, seed: int) -> np.ndarray | None:
    if not epochs:
        return None
    arr = np.asarray(epochs, dtype=float)
    try:
        maps, _ = fit_microstate_maps(arr, n_states=n_states, seed=seed)
        return maps
    except Exception:
        return None


def extract_features(
    epoch: np.ndarray,
    sfreq: float,
    preprocessing_config: dict | None = None,
    microstate_maps: np.ndarray | None = None,
) -> dict:
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
    feats.update(connectivity_features(epoch, sfreq))
    feats.update(cross_frequency_features(epoch, sfreq))
    if microstate_maps is not None:
        feats.update(microstate_features(epoch, microstate_maps))
    return ensure_feature_dict_finite(feats)[0]


def features_to_matrix(feature_dicts: list):
    names = sorted(feature_dicts[0].keys())
    X = np.array([[d[n] for n in names] for d in feature_dicts])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def _feature_reliability(feature_dicts: list[dict]) -> dict:
    names = sorted(feature_dicts[0].keys()) if feature_dicts else []
    out = {}
    for name in names:
        vals = np.array([row.get(name, np.nan) for row in feature_dicts], dtype=float)
        finite = np.isfinite(vals)
        out[name] = {
            "finite_fraction": float(finite.mean()) if len(vals) else 0.0,
            "mean": float(np.nanmean(vals)) if np.any(finite) else 0.0,
            "std": float(np.nanstd(vals)) if np.any(finite) else 0.0,
        }
    return out


def _regime_icc_audit(cfg: dict, seed: int, n_subjects: int, n_sessions: int, sfreq: float, n_seconds: float) -> dict:
    import pandas as pd

    rows = []
    for regime_idx, regime in enumerate(("critical", "control")):
        for s in range(n_subjects):
            sessions = simulate_subject_sessions(
                subject_id=f"audit-{s:02d}",
                n_sessions=n_sessions,
                regime=regime,
                base_seed=seed + 100 * s + 20_000 * regime_idx,
                model=cfg.get("model", "kuramoto"),
                model_kwargs=cfg.get("model_kwargs", {}),
                n_seconds=n_seconds,
                sfreq=sfreq,
            )
            for sess in sessions:
                rows.append(
                    {
                        "subject": sess.subject_id,
                        "session": sess.session_id,
                        "regime": regime,
                        "order_parameter_mean": float(np.mean(sess.order_parameter)),
                    }
                )
    audit_df = pd.DataFrame(rows)
    out = {}
    for regime in ("collapsed", "critical", "control"):
        regime_df = audit_df[audit_df["regime"] == regime]
        pivot = regime_df.pivot(index="subject", columns="session", values="order_parameter_mean")
        icc_result = gate1_icc(pivot.values, n_perm=200, moderate_threshold=0.5, seed=seed)
        icc_result["available"] = True
        icc_result["signal"] = "order_parameter_mean"
        out[regime] = icc_result
    return out


def _distribution_shift_diagnostics(X: np.ndarray, y: np.ndarray, split: dict) -> dict:
    train, calib, test = split["train"], split["calib"], split["test"]

    def _safe_mean(v):
        return float(np.nanmean(v)) if len(v) else 0.0

    if X.shape[1] == 0:
        return {
            "x_feature_mean_shift_train_calib_l1": 0.0,
            "x_feature_mean_shift_train_test_l1": 0.0,
            "x_feature_mean_shift_train_calib_max": 0.0,
            "x_feature_mean_shift_train_test_max": 0.0,
            "y_mean_shift_train_calib": float(abs(_safe_mean(y[train]) - _safe_mean(y[calib]))),
            "y_mean_shift_train_test": float(abs(_safe_mean(y[train]) - _safe_mean(y[test]))),
        }

    train_mean = np.nanmean(X[train], axis=0) if len(train) else np.zeros(X.shape[1], dtype=float)
    calib_mean = np.nanmean(X[calib], axis=0) if len(calib) else np.zeros(X.shape[1], dtype=float)
    test_mean = np.nanmean(X[test], axis=0) if len(test) else np.zeros(X.shape[1], dtype=float)
    scale = np.nanstd(X[train], axis=0) + 1e-12 if len(train) else np.ones(X.shape[1], dtype=float)

    feat_shift_train_calib = np.abs(train_mean - calib_mean) / scale
    feat_shift_train_test = np.abs(train_mean - test_mean) / scale
    y_train_mean = _safe_mean(y[train])
    y_calib_mean = _safe_mean(y[calib])
    y_test_mean = _safe_mean(y[test])

    return {
        "x_feature_mean_shift_train_calib_l1": float(np.nanmean(feat_shift_train_calib)),
        "x_feature_mean_shift_train_test_l1": float(np.nanmean(feat_shift_train_test)),
        "x_feature_mean_shift_train_calib_max": float(np.nanmax(feat_shift_train_calib)),
        "x_feature_mean_shift_train_test_max": float(np.nanmax(feat_shift_train_test)),
        "y_mean_shift_train_calib": float(abs(y_train_mean - y_calib_mean)),
        "y_mean_shift_train_test": float(abs(y_train_mean - y_test_mean)),
    }


def _parameter_recovery_diagnostics(sessions: list) -> dict:
    if not sessions:
        return {"available": False}
    aligned = []
    for sess in sessions:
        if sess.collapse_mask.any():
            aligned.append(
                float(np.mean(sess.order_parameter[sess.collapse_mask]) - np.mean(sess.order_parameter[~sess.collapse_mask]))
            )
    if not aligned:
        return {"available": False, "reason": "no collapse windows"}
    return {
        "available": True,
        "collapse_order_parameter_delta_mean": float(np.mean(aligned)),
        "collapse_order_parameter_delta_std": float(np.std(aligned)),
        "n_sessions": int(len(aligned)),
    }


def _centroid_stability_bootstrap(Xr: np.ndarray, labels: np.ndarray, subject_ids: list[str], n_boot: int = 100, seed: int = 0) -> dict:
    if len(Xr) == 0 or labels.sum() == 0:
        return {"available": False}
    rng = np.random.default_rng(seed)
    subs = np.array(subject_ids, dtype=object)
    unique_subs = np.unique(subs)
    anchor = Xr[labels].mean(axis=0)
    subject_centroids = []
    for sub in unique_subs:
        idx = np.flatnonzero(subs == sub)
        if len(idx) and labels[idx].sum() > 0:
            subject_centroids.append(Xr[idx][labels[idx]].mean(axis=0))
    if len(subject_centroids) < 2:
        return {"available": False}
    subject_centroids = np.vstack(subject_centroids)
    drifts = []
    for _ in range(n_boot):
        sampled = rng.choice(np.arange(len(subject_centroids)), size=len(subject_centroids), replace=True)
        c = subject_centroids[sampled].mean(axis=0)
        drifts.append(float(np.linalg.norm(c - anchor)))
    if not drifts:
        return {"available": False}
    return {
        "available": True,
        "bootstrap_centroid_drift_mean": float(np.mean(drifts)),
        "bootstrap_centroid_drift_ci95_lo": float(np.percentile(drifts, 2.5)),
        "bootstrap_centroid_drift_ci95_hi": float(np.percentile(drifts, 97.5)),
        "n_boot": int(n_boot),
    }


def run_synthetic_pipeline(config: dict, seed: int | None = None) -> dict:
    """Runs Gates 1, 3, 4 on configured synthetic model data and returns a results dict.

    `seed` overrides `config["synthetic"]["seed"]` when given, so a seed sweep
    does not need to hand-edit config files."""
    cfg = config.get("synthetic", {})
    n_subjects = cfg.get("n_subjects", 6)
    n_sessions = cfg.get("n_sessions_per_subject", 3)
    sfreq = cfg.get("sfreq", 250.0)
    n_seconds = cfg.get("n_seconds", 60.0)
    seed = cfg.get("seed", 0) if seed is None else seed
    model_name = cfg.get("model", "kuramoto")
    model_kwargs = cfg.get("model_kwargs", {})
    preprocessing_cfg = config.get("preprocessing", {})
    conformal_cfg = config.get("conformal", {})

    # --- Gate 1: within-subject reproducibility across synthetic "sessions" ---
    all_epochs, all_labels, all_fractions, subject_ids, session_ids = [], [], [], [], []
    within_session_epoch_idx = []
    all_states = []
    collapsed_sessions = []
    for s in range(n_subjects):
        sessions = simulate_subject_sessions(
            subject_id=f"synthsub-{s:02d}",
            n_sessions=n_sessions,
            regime="collapsed",
            base_seed=seed + 100 * s,
            model=model_name,
            model_kwargs=model_kwargs,
            n_seconds=n_seconds,
            sfreq=sfreq,
        )
        for sess in sessions:
            collapsed_sessions.append(sess)
            eps, labels, fractions = _epoch_the_session(sess)
            all_epochs.extend(eps)
            all_labels.extend(labels)
            all_fractions.extend(fractions)
            subject_ids.extend([sess.subject_id] * len(eps))
            session_ids.extend([sess.session_id] * len(eps))
            within_session_epoch_idx.extend(list(range(len(eps))))
            all_states.extend(["collapsed_like" if x else "noncollapsed_like" for x in labels])
    all_labels = np.array(all_labels)
    all_fractions = np.array(all_fractions, dtype=float)
    microstate_n_states = int(preprocessing_cfg.get("microstate_n_states", 4))
    microstate_maps = _fit_microstate_maps_from_epochs(all_epochs, n_states=microstate_n_states, seed=seed)

    feature_dicts = [
        extract_features(ep, sfreq, preprocessing_config=preprocessing_cfg, microstate_maps=microstate_maps) for ep in all_epochs
    ]
    X, feature_names = features_to_matrix(feature_dicts)

    model, Xr = fit_manifold(X, all_labels, feature_names, seed=seed)
    dist = distance_from_cessation(Xr, model.cessation_centroid)

    # Gate 1: within-subject reproducibility across sessions, quantified as
    # ICC(2,1) on per-session mean distance (Koo & Li 2016 J Chiropr Med
    # 15(2):155-163: > 0.5 moderate, > 0.75 good). Pass requires point > 0.5
    # AND permutation-null p < 0.05. Old ad-hoc ratio kept as diagnostic only.
    import pandas as pd

    df = pd.DataFrame(
        {"subject": subject_ids, "session": session_ids, "regime": "collapsed", "label": all_labels, "distance": dist}
    )
    session_means = df.groupby(["subject", "session", "regime"])["distance"].mean().reset_index()
    icc_result = gate1_icc(
        session_means.pivot(index="subject", columns="session", values="distance").values,
        n_perm=200,
        moderate_threshold=0.5,
        seed=seed,
    )
    regime_icc = _regime_icc_audit(
        cfg, seed=seed, n_subjects=n_subjects, n_sessions=n_sessions, sfreq=sfreq, n_seconds=n_seconds
    )
    regime_icc["collapsed"] = {**icc_result, "available": True, "signal": "distance"}

    subj_std = session_means.groupby("subject")["distance"].std().fillna(0.0)
    overall_scale = session_means["distance"].std() + 1e-9
    diag_ratio = float((subj_std / overall_scale).mean())

    # --- Gate 3: surrogates must break the score ---
    surrogate_epochs = [surrogate_epoch(ep, method="iaaft", seed=seed + i) for i, ep in enumerate(all_epochs[:60])]
    surr_feature_dicts = [
        extract_features(ep, sfreq, preprocessing_config=preprocessing_cfg, microstate_maps=microstate_maps)
        for ep in surrogate_epochs
    ]
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
    session_local_idx = np.array(within_session_epoch_idx, dtype=int)
    temporal_blocks = (session_local_idx // max(1, int(conformal_cfg.get("temporal_block_size", 10)))).astype(int)
    block_structure = {
        "subject": np.array(subject_ids),
        "session": np.array(session_ids),
        "timeblock": temporal_blocks,
    }
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
    realized_model_params = collapsed_sessions[0].model_params if collapsed_sessions and collapsed_sessions[0].model_params else {}
    state_arr = np.array(all_states, dtype=object)
    per_subject_coverage = {}
    per_session_coverage = {}
    per_state_coverage = {}
    covered_test = (y[split["test"]] >= coverage_eval["lower"]) & (y[split["test"]] <= coverage_eval["upper"])
    for key, labels in (
        ("subject", np.array(subject_ids, dtype=object)[split["test"]]),
        ("session", np.array(session_ids, dtype=object)[split["test"]]),
        ("state", state_arr[split["test"]]),
    ):
        bucket = {}
        for label in np.unique(labels):
            mask = labels == label
            bucket[str(label)] = float(covered_test[mask].mean())
        if key == "subject":
            per_subject_coverage = bucket
        elif key == "session":
            per_session_coverage = bucket
        else:
            per_state_coverage = bucket

    subgroup_floor = predictor.target_coverage - 0.05
    subgroup_abstentions = {
        "subject": sorted([k for k, v in per_subject_coverage.items() if v < subgroup_floor]),
        "session": sorted([k for k, v in per_session_coverage.items() if v < subgroup_floor]),
        "state": sorted([k for k, v in per_state_coverage.items() if v < subgroup_floor]),
    }

    synthetic_validity = {
        "parameter_recovery": _parameter_recovery_diagnostics(collapsed_sessions),
        "centroid_stability": _centroid_stability_bootstrap(Xr, all_labels, subject_ids, n_boot=100, seed=seed),
    }
    split_shift = _distribution_shift_diagnostics(X, y, split)
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
        "gate1_regime_icc": regime_icc,
        "gate1_anchor_size": {
            "n_subjects": int(n_subjects),
            "n_sessions_per_subject": int(n_sessions),
            "n_epochs_total": int(len(X)),
            "n_epochs_by_regime": {"collapsed": int(len(X))},
        },
        "gate1_within_subject_ratio": diag_ratio,
        "gate1_pass": icc_result["pass"],
        "gate3_surrogate_mean_distance": float(np.mean(dist_surr)),
        "gate3_real_mean_distance": float(np.mean(real_cess_dist)),
        "gate3_pass": gate3_pass,
        "gate4_conformal_coverage": coverage,
        "gate4_block_coverage": coverage_eval["block_coverage"],
        "gate4_diagnostics": predictor.diagnostics_,
        "gate4_conditional_coverage": {
            "subject": per_subject_coverage,
            "session": per_session_coverage,
            "state": per_state_coverage,
            "test_block": coverage_eval["per_block_coverage"],
        },
        "gate4_subgroup_abstentions": subgroup_abstentions,
        "gate4_subgroup_pass": bool(
            not subgroup_abstentions["subject"] and not subgroup_abstentions["session"] and not subgroup_abstentions["state"]
        ),
        "distribution_shift_diagnostics": split_shift,
        "gate4_unstable_for_review": not predictor.diagnostics_.get("threshold_stable", True),
        "gate4_pass": gate4_pass,
        "example_finding": finding_dict,
        "synthetic_model": model_name,
        "synthetic_model_params": realized_model_params,
        "synthetic_validity": synthetic_validity,
        "microstate_maps_available": bool(microstate_maps is not None),
        "n_epochs": int(len(X)),
        "n_subjects": n_subjects,
    }


def run_real_data_pipeline(config: dict, eeg_path: str, subject_id: str = "uploaded") -> dict:
    """Run feature extraction and artifact reporting for one uploaded EEG file.

    This path validates preprocessing/feature robustness on real EEG but does
    not claim cessation detection unless cessation-labeled datasets are
    supplied.
    """
    data_cfg = config.get("data", {})
    preprocessing_cfg = config.get("preprocessing", {})
    loaded = load_uploaded_eeg(
        eeg_path=eeg_path,
        dataset_id=data_cfg.get("source", "uploaded"),
        epoch_length_s=float(data_cfg.get("epoch_length_s", 5.0)),
        l_freq=float(data_cfg.get("l_freq", 1.0)),
        h_freq=float(data_cfg.get("h_freq", 45.0)),
    )
    raw_clean, provenance = remove_artifacts_mne(
        loaded.raw,
        random_state=int(preprocessing_cfg.get("seed", 0)),
    )
    import mne

    epochs = mne.make_fixed_length_epochs(
        raw_clean, duration=float(data_cfg.get("epoch_length_s", 5.0)), preload=True, verbose=False
    )
    epoch_data = list(epochs.get_data())
    microstate_n_states = int(preprocessing_cfg.get("microstate_n_states", 4))
    microstate_maps = _fit_microstate_maps_from_epochs(epoch_data, n_states=microstate_n_states, seed=int(preprocessing_cfg.get("seed", 0)))
    feature_dicts = [
        extract_features(ep, loaded.sfreq, preprocessing_cfg, microstate_maps=microstate_maps) for ep in epoch_data
    ]
    X, names = features_to_matrix(feature_dicts)
    finite_matrix = np.isfinite(X).all()
    return {
        "dataset": loaded.source,
        "subject_id": subject_id,
        "session_id": loaded.session_id,
        "n_channels": int(len(loaded.ch_names)),
        "n_epochs": int(X.shape[0]),
        "sampling_rate_hz": float(loaded.sfreq),
        "features": names,
        "feature_reliability": _feature_reliability(feature_dicts),
        "feature_matrix_finite": bool(finite_matrix),
        "artifact_rejection_rate": float(provenance.get("artifact_rejection_rate", 0.0)),
        "artifact_provenance": provenance,
        "claim_scope": "feature-validation-only",
        "claim_note": "Cessation claims require cessation-labeled EEG and independent clinical adjudication.",
        "microstate_maps_available": bool(microstate_maps is not None),
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
    model_name = cfg.get("model", "kuramoto")
    model_kwargs = cfg.get("model_kwargs", {})

    collapsed_epochs, collapsed_labels = [], []
    for s in range(3):
        sess = simulate_subject_sessions(
            f"anchor-{s}",
            n_sessions=1,
            regime="collapsed",
            base_seed=seed + s,
            model=model_name,
            model_kwargs=model_kwargs,
            n_seconds=n_seconds,
            sfreq=sfreq,
        )[0]
        eps, labels, _fractions = _epoch_the_session(sess)
        collapsed_epochs.extend(eps)
        collapsed_labels.extend(labels)
    collapsed_labels = np.array(collapsed_labels, dtype=bool)

    baseline_sess = simulate_subject_sessions(
        "baseline-critical",
        n_sessions=1,
        regime="critical",
        base_seed=seed + 500,
        model=model_name,
        model_kwargs=model_kwargs,
        n_seconds=n_seconds,
        sfreq=sfreq,
    )[0]
    baseline_epochs, _, _ = _epoch_the_session(baseline_sess)

    all_eps = collapsed_epochs + baseline_epochs
    ms_maps = _fit_microstate_maps_from_epochs(all_eps, n_states=4, seed=seed)
    anchor_feats = [extract_features(ep, sfreq, microstate_maps=ms_maps) for ep in collapsed_epochs]
    baseline_feats = [extract_features(ep, sfreq, microstate_maps=ms_maps) for ep in baseline_epochs]

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
