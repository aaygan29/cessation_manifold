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


def block_to_epochs(raw: mne.io.BaseRaw, posterior: list, t0: float, t1: float):
    fs = raw.info["sfreq"]
    step = int(EPOCH_SECONDS * fs)
    start_samp = int(t0 * fs)
    stop_samp = int(t1 * fs)
    data, _ = raw[posterior, start_samp:stop_samp]
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


def main():
    per_subject = {}
    all_eo, all_ec = [], []
    for sub in SUBJECTS:
        print(f"[{sub}] loading and preprocessing ...")
        raw = load_subject(sub)
        blocks = block_windows_from_markers(raw)
        if not blocks:
            print(f"[{sub}]   no EO/EC blocks found in markers, skipping")
            continue
        posterior = posterior_indices(raw)
        print(f"[{sub}]   {len(blocks)} blocks, {len(posterior)} posterior channels")

        eo_epochs, ec_epochs = [], []
        for label, t0, t1 in blocks:
            epochs = block_to_epochs(raw, posterior, t0, t1)
            (eo_epochs if label == "EO" else ec_epochs).extend(epochs)

        stats = {k: paired_stats(eo_epochs, ec_epochs, k) for k in
                 ["alpha_rel", "alpha_abs_uv2", "aperiodic_exponent", "lempel_ziv", "dfa_alpha"]}
        per_subject[sub] = stats
        all_eo.extend(eo_epochs)
        all_ec.extend(ec_epochs)
        print(f"[{sub}]   alpha_rel EO->EC: {stats['alpha_rel']['mean_EO']:.3f} -> "
              f"{stats['alpha_rel']['mean_EC']:.3f} (d={stats['alpha_rel']['cohens_d']:.2f}, "
              f"p={stats['alpha_rel']['p_value']:.2e})")

    pooled = {k: paired_stats(all_eo, all_ec, k) for k in
              ["alpha_rel", "alpha_abs_uv2", "aperiodic_exponent", "lempel_ziv", "dfa_alpha"]}

    berger_pass = (
        pooled["alpha_rel"] is not None
        and pooled["alpha_rel"]["mean_EC"] > pooled["alpha_rel"]["mean_EO"]
        and pooled["alpha_rel"]["p_value"] < 0.05
        and pooled["alpha_rel"]["cohens_d"] > 0.5
    )

    out = {
        "gate0_berger_reproduction": {
            "hypothesis": "posterior relative alpha (8-13 Hz) is higher during eyes-closed than eyes-open",
            "pooled_across_subjects": pooled,
            "per_subject": per_subject,
            "pass": bool(berger_pass),
            "criteria": {
                "direction": "mean_EC > mean_EO on alpha_rel",
                "significance": "p < 0.05 (Mann-Whitney U, two-sided)",
                "effect_size": "Cohens d > 0.5",
            },
        },
        "provenance": make_provenance(
            dataset_id="lemon-ds000221-subset-2subjects",
            config={"script": "validate_lemon_berger.py", "version": "v0"},
        ).__dict__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path("results/lemon_berger_validation.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Gate 0 (Berger effect reproduction): {'PASS' if berger_pass else 'FAIL'}")
    print(f"  alpha_rel EO: {pooled['alpha_rel']['mean_EO']:.4f}")
    print(f"  alpha_rel EC: {pooled['alpha_rel']['mean_EC']:.4f}")
    print(f"  Cohens d: {pooled['alpha_rel']['cohens_d']:.2f}")
    print(f"  p-value: {pooled['alpha_rel']['p_value']:.2e}")


if __name__ == "__main__":
    main()
