"""
Positive-control re-validation: recover the Berger effect on real LEMON EEG.

Prior work (Berger 1929; replicated tens of thousands of times, canonically in
Babayan et al. 2019 for LEMON specifically) says posterior alpha (8-13 Hz)
power rises sharply when the eyes close. This script uses the same feature
stack and manifold/distance machinery that cessation_manifold applies to
synthetic collapse data, and asks: does it recover the eyes-closed vs
eyes-open alpha contrast on real LEMON subjects, within-subject?

If yes, the feature pipeline is not broken on real EEG. This does not prove
the cessation-manifold claim (cessation != eyes-closed rest); it proves the
apparatus recovers a known signal from real EEG, which is a necessary
precondition before any claim about cessation.

Data: LEMON raw BrainVision (fetched via scripts/fetch_lemon_subset.sh).
Markers: S200 = eyes-open block start, S210 = eyes-closed block start
(Babayan et al. 2019, LEMON EEG protocol).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import mne
except ImportError:
    print("ERROR: mne not installed. Run: pip install mne", file=sys.stderr)
    sys.exit(1)

from cessation_manifold.features.aperiodic import aperiodic_features
from cessation_manifold.features.complexity import complexity_features
from cessation_manifold.honesty.gates import make_provenance

DATA_ROOT = Path("data/raw/lemon")
SUBJECTS = ["sub-010002", "sub-010003"]
FS_TARGET = 250.0
EPOCH_SECONDS = 4.0
ALPHA_BAND = (8.0, 13.0)
POSTERIOR_CH_HINTS = ("O", "PO", "P")
N_PERMUTATIONS = 5000
SPECIFICITY_THRESHOLD = 0.3


def load_subject(sub: str) -> mne.io.BaseRaw:
    vhdr = DATA_ROOT / sub / "RSEEG" / f"{sub}.vhdr"
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    raw.resample(FS_TARGET, verbose="ERROR")
    raw.filter(1.0, 45.0, verbose="ERROR")
    return raw


def block_windows_from_markers(raw: mne.io.BaseRaw):
    """Return list of (label, start_sec, stop_sec) blocks.

    LEMON alternates 60s EO / 60s EC blocks. S200 marks EO onset, S210 marks EC
    onset. We take each block as start-of-code to next-different-code, ignoring
    the intra-block 5s pulses.
    """
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    fs = raw.info["sfreq"]

    id_200 = None
    id_210 = None
    for k, v in event_id.items():
        if k.endswith("200"):
            id_200 = v
        if k.endswith("210"):
            id_210 = v
    if id_200 is None or id_210 is None:
        return []

    onset_events = []
    last_label = None
    for samp, _, code in events:
        if code == id_200 and last_label != "EO":
            onset_events.append(("EO", samp / fs))
            last_label = "EO"
        elif code == id_210 and last_label != "EC":
            onset_events.append(("EC", samp / fs))
            last_label = "EC"

    total_dur = raw.times[-1]
    blocks = []
    for i, (label, t0) in enumerate(onset_events):
        t1 = onset_events[i + 1][1] if i + 1 < len(onset_events) else total_dur
        if t1 - t0 >= 20.0:
            blocks.append((label, t0 + 2.0, min(t0 + 60.0, t1 - 1.0)))
    return blocks


def posterior_indices(raw: mne.io.BaseRaw):
    idx = []
    for i, ch in enumerate(raw.ch_names):
        name = ch.upper()
        if any(name.startswith(h) for h in POSTERIOR_CH_HINTS) and not name.startswith("PPO"):
            idx.append(i)
    if not idx:
        idx = list(range(len(raw.ch_names)))[-8:]
    return idx


def frontal_indices(raw: mne.io.BaseRaw):
    """Channels starting with F but not FT/FC/FP (so F3, F4, Fz, F7, F8 etc),
    used as the specificity control arm: the Berger effect should be posterior,
    not frontal."""
    idx = []
    for i, ch in enumerate(raw.ch_names):
        name = ch.upper()
        if name.startswith("F") and not (name.startswith("FT") or name.startswith("FC") or name.startswith("FP")):
            idx.append(i)
    return idx


def epoch_features(data: np.ndarray, fs: float) -> dict:
    """data: (n_channels, n_samples), a single epoch on posterior channels."""
    from scipy.signal import welch

    x = data.mean(axis=0)
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), int(fs * 2)))
    alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    alpha_power = float(np.trapezoid(psd[alpha_mask], freqs[alpha_mask]))
    total_mask = (freqs >= 2.0) & (freqs <= 40.0)
    total_power = float(np.trapezoid(psd[total_mask], freqs[total_mask])) + 1e-12
    rel_alpha = alpha_power / total_power

    ap = aperiodic_features(x[np.newaxis, :], fs)
    comp = complexity_features(x[np.newaxis, :], fs)
    exponent = float(ap.get("aperiodic_exponent", ap.get("exponent", np.nan)))
    lz = float(comp.get("lempel_ziv", comp.get("lz", np.nan)))
    dfa = float(comp.get("dfa_alpha", comp.get("dfa", np.nan)))

    return {
        "alpha_abs_uv2": alpha_power,
        "alpha_rel": rel_alpha,
        "aperiodic_exponent": exponent,
        "lempel_ziv": lz,
        "dfa_alpha": dfa,
    }


def block_to_epochs(raw: mne.io.BaseRaw, channels: list, t0: float, t1: float):
    fs = raw.info["sfreq"]
    step = int(EPOCH_SECONDS * fs)
    start_samp = int(t0 * fs)
    stop_samp = int(t1 * fs)
    data, _ = raw[channels, start_samp:stop_samp]
    n = data.shape[1]
    epochs = []
    for i in range(0, n - step + 1, step):
        epochs.append(epoch_features(data[:, i : i + step], fs))
    return epochs


def paired_stats(eo: list, ec: list, key: str):
    a = np.array([e[key] for e in eo])
    b = np.array([e[key] for e in ec])
    if len(a) == 0 or len(b) == 0:
        return None
    from scipy.stats import mannwhitneyu

    u, p = mannwhitneyu(b, a, alternative="two-sided")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
    d = (b.mean() - a.mean()) / pooled
    return {
        "mean_EO": float(a.mean()),
        "mean_EC": float(b.mean()),
        "cohens_d": float(d),
        "mannwhitney_U": float(u),
        "p_value": float(p),
        "n_EO_epochs": int(len(a)),
        "n_EC_epochs": int(len(b)),
    }


def block_level_alpha(epochs: list) -> float:
    """Mean alpha_rel across the epochs of one block, the unit the permutation
    test resamples (epochs within a block are not independent of each other,
    blocks are the closest thing to an independent unit here)."""
    vals = [e["alpha_rel"] for e in epochs]
    return float(np.mean(vals)) if vals else float("nan")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
    return float((b.mean() - a.mean()) / pooled)


def block_permutation_test(block_labels: list, block_values: np.ndarray, n_perm: int, seed: int = 0):
    """Shuffle the EO/EC block labels n_perm times and recompute Cohen's d
    (EC minus EO) each time, to get a null distribution that respects block-
    level (not epoch-level) independence. Two-sided p-value against the
    observed d."""
    labels = np.array(block_labels)
    eo_mask = labels == "EO"
    ec_mask = labels == "EC"
    observed_d = cohens_d(block_values[eo_mask], block_values[ec_mask])

    rng = np.random.default_rng(seed)
    null_ds = np.empty(n_perm)
    n = len(labels)
    for i in range(n_perm):
        perm = rng.permutation(n)
        shuffled = labels[perm]
        eo_s = shuffled == "EO"
        ec_s = shuffled == "EC"
        null_ds[i] = cohens_d(block_values[eo_s], block_values[ec_s])

    p_value = float(np.mean(np.abs(null_ds) >= abs(observed_d)))
    return observed_d, p_value


def run_arm(channel_indices_fn, arm_name: str):
    """Runs the full EO/EC contrast for one channel arm (posterior or frontal).
    Returns (per_subject stats, pooled stats, block_labels, block_values)."""
    per_subject = {}
    all_eo, all_ec = [], []
    block_labels, block_values = [], []

    for sub in SUBJECTS:
        raw = load_subject(sub)
        blocks = block_windows_from_markers(raw)
        if not blocks:
            print(f"[{sub}]   no EO/EC blocks found in markers, skipping ({arm_name})")
            continue
        channels = channel_indices_fn(raw)
        if not channels:
            print(f"[{sub}]   no {arm_name} channels found, skipping")
            continue

        eo_epochs, ec_epochs = [], []
        for label, t0, t1 in blocks:
            epochs = block_to_epochs(raw, channels, t0, t1)
            if not epochs:
                continue
            (eo_epochs if label == "EO" else ec_epochs).extend(epochs)
            block_labels.append(label)
            block_values.append(block_level_alpha(epochs))

        stats = {k: paired_stats(eo_epochs, ec_epochs, k) for k in
                 ["alpha_rel", "alpha_abs_uv2", "aperiodic_exponent", "lempel_ziv", "dfa_alpha"]}
        per_subject[sub] = stats
        all_eo.extend(eo_epochs)
        all_ec.extend(ec_epochs)
        if stats["alpha_rel"] is not None:
            print(f"[{sub}] ({arm_name}) alpha_rel EO->EC: {stats['alpha_rel']['mean_EO']:.3f} -> "
                  f"{stats['alpha_rel']['mean_EC']:.3f} (d={stats['alpha_rel']['cohens_d']:.2f}, "
                  f"p={stats['alpha_rel']['p_value']:.2e})")

    pooled = {k: paired_stats(all_eo, all_ec, k) for k in
              ["alpha_rel", "alpha_abs_uv2", "aperiodic_exponent", "lempel_ziv", "dfa_alpha"]}

    return per_subject, pooled, block_labels, np.array(block_values, dtype=float)


def main():
    print("Running posterior arm (primary) ...")
    posterior_per_subject, posterior_pooled, block_labels, block_values = run_arm(posterior_indices, "posterior")

    print("\nRunning frontal arm (specificity control) ...")
    frontal_per_subject, frontal_pooled, _frontal_block_labels, _frontal_block_values = run_arm(
        frontal_indices, "frontal"
    )

    d_posterior = posterior_pooled["alpha_rel"]["cohens_d"] if posterior_pooled["alpha_rel"] else float("nan")
    d_frontal = frontal_pooled["alpha_rel"]["cohens_d"] if frontal_pooled["alpha_rel"] else float("nan")
    d_diff = d_posterior - d_frontal
    specificity_pass = bool(d_diff > SPECIFICITY_THRESHOLD)

    print(f"\nSpecificity: d_posterior={d_posterior:.3f}, d_frontal={d_frontal:.3f}, "
          f"diff={d_diff:.3f} (threshold {SPECIFICITY_THRESHOLD}) -> "
          f"{'PASS' if specificity_pass else 'FAIL'}")

    print(f"\nRunning block-level permutation test ({N_PERMUTATIONS} permutations, "
          f"{len(block_labels)} blocks) ...")
    observed_block_d, perm_p_value = block_permutation_test(block_labels, block_values, N_PERMUTATIONS, seed=0)
    print(f"  observed block-level d: {observed_block_d:.3f}, permutation p: {perm_p_value:.4f}")

    gate0_pass = (
        posterior_pooled["alpha_rel"] is not None
        and posterior_pooled["alpha_rel"]["mean_EC"] > posterior_pooled["alpha_rel"]["mean_EO"]
        and perm_p_value < 0.05
        and d_posterior > 0.5
        and specificity_pass
    )

    out = {
        "gate0_berger_reproduction": {
            "hypothesis": "posterior relative alpha (8-13 Hz) is higher during eyes-closed than eyes-open, "
                           "and this effect is specific to posterior channels (not frontal)",
            "posterior_stats": {
                "pooled_across_subjects": posterior_pooled,
                "per_subject": posterior_per_subject,
            },
            "frontal_stats": {
                "pooled_across_subjects": frontal_pooled,
                "per_subject": frontal_per_subject,
            },
            "specificity": {
                "d_posterior": d_posterior,
                "d_frontal": d_frontal,
                "d_posterior_minus_frontal": d_diff,
                "threshold": SPECIFICITY_THRESHOLD,
                "pass": specificity_pass,
            },
            "block_permutation": {
                "p_value": perm_p_value,
                "n_permutations": N_PERMUTATIONS,
                "observed_d": observed_block_d,
                "n_blocks": len(block_labels),
                "note": "primary significance test: block-level label shuffle, respects block "
                        "(not epoch) independence",
            },
            "mannwhitney_note": "Mann-Whitney U above (per_subject / pooled_across_subjects.alpha_rel.p_value) "
                                 "assumes epoch independence within a block, which does not hold here. "
                                 "It is reported for continuity with the original Gate 0 run, not as the "
                                 "primary significance test.",
            "pass": bool(gate0_pass),
            "criteria": {
                "direction": "mean_EC > mean_EO on posterior alpha_rel",
                "significance": "block-permutation p < 0.05 (primary; 5000 permutations)",
                "effect_size": "posterior Cohens d > 0.5",
                "specificity": f"d_posterior - d_frontal > {SPECIFICITY_THRESHOLD}",
            },
        },
        "provenance": make_provenance(
            dataset_id="lemon-ds000221-subset-2subjects",
            config={"script": "validate_lemon_berger.py", "version": "v1-specificity"},
        ).__dict__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path("results/lemon_berger_validation.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Gate 0 (Berger effect reproduction, with specificity + permutation): "
          f"{'PASS' if gate0_pass else 'FAIL'}")
    print(f"  alpha_rel EO (posterior): {posterior_pooled['alpha_rel']['mean_EO']:.4f}")
    print(f"  alpha_rel EC (posterior): {posterior_pooled['alpha_rel']['mean_EC']:.4f}")
    print(f"  d_posterior: {d_posterior:.3f}, d_frontal: {d_frontal:.3f}, diff: {d_diff:.3f}")
    print(f"  block-permutation p: {perm_p_value:.4f}")


if __name__ == "__main__":
    main()
