"""
Gate 0b: recover the canonical Wake > N3 complexity drop on Sleep-EDF.

Prior work (Sitt et al. 2014 Brain 137:2258; Toker et al. 2023 bioRxiv;
Sarasso et al. 2015 Curr Biol 25:3099) says signal complexity, in
particular Lempel-Ziv on resting EEG, decreases with depth of NREM sleep.
We run the project's own complexity + aperiodic feature stack on
PhysioNet Sleep-EDF Expanded (v1.0.0) and ask: does Wake stage LZ exceed
N3 (SWS) LZ, per-subject and pooled, with a block-level permutation p?

If yes, the feature stack recovers a second canonical published finding
on a completely different paradigm (sleep, not eyes-closed rest). That
is meaningful independent evidence that the pipeline tracks
consciousness-graded EEG features, not paradigm-specific artifacts.

Data: PhysioNet Sleep-EDF Expanded (fetched via
scripts/fetch_sleep_edfx_subset.sh). PSG channel used: EEG Pz-Oz (100 Hz,
posterior). Hypnogram stages mapped: Sleep stage W -> Wake, Sleep stage 2
-> N2, Sleep stage 3 -> N3 (SWS; stage 3+4 combined per AASM), Sleep
stage R -> REM. 30-second epochs (sleep-scoring standard).
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
    print("ERROR: mne not installed. pip install mne", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cessation_manifold.features.aperiodic import aperiodic_features
from cessation_manifold.features.complexity import complexity_features
from cessation_manifold.honesty.gates import make_provenance

DATA_ROOT = Path("data/raw/sleep_edfx")
EPOCH_SECONDS = 30.0
STAGES_KEEP = {"Sleep stage W": "W", "Sleep stage 2": "N2",
               "Sleep stage 3": "N3", "Sleep stage 4": "N3",
               "Sleep stage R": "REM"}
ALPHA_BAND = (8.0, 13.0)


def find_subject_files():
    """Return list of (psg_path, hyp_path) pairs for available subjects."""
    if not DATA_ROOT.exists():
        return []
    psgs = sorted(DATA_ROOT.glob("*-PSG.edf"))
    pairs = []
    for psg in psgs:
        stem = psg.stem[:6]
        candidates = sorted(DATA_ROOT.glob(f"{stem}*-Hypnogram.edf"))
        if candidates:
            pairs.append((psg, candidates[0]))
    return pairs


def load_subject(psg_path: Path, hyp_path: Path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, verbose="ERROR")
    ch_pick = None
    for ch in raw.ch_names:
        if "Pz-Oz" in ch or "PZ-OZ" in ch:
            ch_pick = ch
            break
    if ch_pick is None:
        ch_pick = raw.ch_names[0]
    raw.pick([ch_pick])
    raw.filter(0.3, 35.0, verbose="ERROR")
    return raw, ch_pick


def epoch_by_stage(raw, sfreq: float):
    """Return dict stage -> list of feature dicts."""
    step = int(EPOCH_SECONDS * sfreq)
    per_stage = {s: [] for s in ("W", "N2", "N3", "REM")}
    for ann in raw.annotations:
        stage_name = STAGES_KEEP.get(ann["description"])
        if stage_name is None:
            continue
        t0 = int(ann["onset"] * sfreq)
        t1 = int((ann["onset"] + ann["duration"]) * sfreq)
        for start in range(t0, t1 - step + 1, step):
            data, _ = raw[0, start:start + step]
            x = data[0]
            if np.any(np.isnan(x)) or x.std() < 1e-6:
                continue
            feats = extract_epoch_features(x, sfreq)
            per_stage[stage_name].append(feats)
    return per_stage


def extract_epoch_features(x: np.ndarray, sfreq: float) -> dict:
    from scipy.signal import welch
    freqs, psd = welch(x, fs=sfreq, nperseg=min(len(x), int(sfreq * 4)))
    alpha_mask = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    total_mask = (freqs >= 1.0) & (freqs <= 30.0)
    alpha_power = float(np.trapezoid(psd[alpha_mask], freqs[alpha_mask]))
    total = float(np.trapezoid(psd[total_mask], freqs[total_mask])) + 1e-12
    ap = aperiodic_features(x[np.newaxis, :], sfreq)
    comp = complexity_features(x[np.newaxis, :], sfreq)
    return {
        "alpha_rel": alpha_power / total,
        "aperiodic_exponent": float(ap.get("aperiodic_exponent", ap.get("exponent", np.nan))),
        "lempel_ziv": float(comp.get("lempel_ziv", comp.get("lz", np.nan))),
        "dfa_alpha": float(comp.get("dfa_alpha", comp.get("dfa", np.nan))),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
    return float((a.mean() - b.mean()) / pooled)


def stage_permutation_test(vals_a: np.ndarray, vals_b: np.ndarray, n_perm: int = 5000, seed: int = 0) -> dict:
    """Shuffle labels between the two pools of epoch-level values; report
    observed d (a minus b) and two-sided p."""
    observed = cohens_d(vals_a, vals_b)
    pooled = np.concatenate([vals_a, vals_b])
    n_a = len(vals_a)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        null[i] = cohens_d(perm[:n_a], perm[n_a:])
    p = float(np.mean(np.abs(null) >= abs(observed)))
    return {"observed_d": observed, "p_value": p, "n_permutations": n_perm}


def stage_summary(per_stage: dict, key: str) -> dict:
    out = {}
    for stage, feats in per_stage.items():
        vals = np.array([f[key] for f in feats])
        if len(vals) == 0:
            continue
        out[stage] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
                      "n": int(len(vals))}
    return out


def main():
    pairs = find_subject_files()
    if not pairs:
        print(f"No Sleep-EDF subjects found under {DATA_ROOT}. Run scripts/fetch_sleep_edfx_subset.sh first.")
        sys.exit(1)

    per_subject_summaries = {}
    all_wake_lz = []
    all_n3_lz = []
    all_wake_ape = []
    all_n3_ape = []

    for psg, hyp in pairs:
        sub_id = psg.stem[:6]
        print(f"[{sub_id}] loading {psg.name} + {hyp.name} ...")
        raw, ch_pick = load_subject(psg, hyp)
        sfreq = raw.info["sfreq"]
        print(f"[{sub_id}]   channel {ch_pick} at {sfreq} Hz")
        per_stage = epoch_by_stage(raw, sfreq)
        sizes = {s: len(v) for s, v in per_stage.items()}
        print(f"[{sub_id}]   epochs per stage: {sizes}")

        summary_lz = stage_summary(per_stage, "lempel_ziv")
        summary_ape = stage_summary(per_stage, "aperiodic_exponent")

        per_subject_summaries[sub_id] = {
            "lempel_ziv": summary_lz,
            "aperiodic_exponent": summary_ape,
            "n_epochs_per_stage": sizes,
        }

        if per_stage["W"] and per_stage["N3"]:
            all_wake_lz.extend([f["lempel_ziv"] for f in per_stage["W"]])
            all_n3_lz.extend([f["lempel_ziv"] for f in per_stage["N3"]])
            all_wake_ape.extend([f["aperiodic_exponent"] for f in per_stage["W"]])
            all_n3_ape.extend([f["aperiodic_exponent"] for f in per_stage["N3"]])

    all_wake_lz = np.array(all_wake_lz)
    all_n3_lz = np.array(all_n3_lz)
    all_wake_ape = np.array(all_wake_ape)
    all_n3_ape = np.array(all_n3_ape)

    lz_perm = stage_permutation_test(all_wake_lz, all_n3_lz) if len(all_wake_lz) and len(all_n3_lz) else None
    ape_perm = stage_permutation_test(all_wake_ape, all_n3_ape) if len(all_wake_ape) and len(all_n3_ape) else None

    lz_pass = (
        lz_perm is not None
        and lz_perm["observed_d"] > 0.5
        and lz_perm["p_value"] < 0.05
    )
    gate0b_pass = bool(lz_pass)

    out = {
        "gate0b_wake_gt_n3_complexity": {
            "hypothesis": "Wake Lempel-Ziv > N3 Lempel-Ziv on resting Pz-Oz",
            "criteria": {
                "direction": "mean_W > mean_N3 on lempel_ziv",
                "effect_size": "Cohen's d > 0.5",
                "significance": "permutation p < 0.05",
            },
            "pooled_lz": {
                "mean_W": float(all_wake_lz.mean()) if len(all_wake_lz) else None,
                "mean_N3": float(all_n3_lz.mean()) if len(all_n3_lz) else None,
                "cohens_d_W_minus_N3": lz_perm["observed_d"] if lz_perm else None,
                "permutation_p": lz_perm["p_value"] if lz_perm else None,
            },
            "pooled_aperiodic": {
                "mean_W": float(all_wake_ape.mean()) if len(all_wake_ape) else None,
                "mean_N3": float(all_n3_ape.mean()) if len(all_n3_ape) else None,
                "cohens_d_W_minus_N3": ape_perm["observed_d"] if ape_perm else None,
                "permutation_p": ape_perm["p_value"] if ape_perm else None,
            },
            "per_subject": per_subject_summaries,
            "pass": gate0b_pass,
        },
        "provenance": make_provenance(
            dataset_id="physionet-sleep-edfx-1.0.0-subset",
            config={"script": "validate_sleep_edfx.py", "n_subjects": len(pairs)},
        ).__dict__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path("results/sleep_edfx_validation.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(f"Gate 0b (Wake > N3 complexity): {'PASS' if gate0b_pass else 'FAIL'}")
    if lz_perm:
        print(f"  LZ Wake mean: {all_wake_lz.mean():.3f}  N3 mean: {all_n3_lz.mean():.3f}  d={lz_perm['observed_d']:.2f}  p={lz_perm['p_value']:.4f}")
    if ape_perm:
        print(f"  aperiodic Wake mean: {all_wake_ape.mean():.3f}  N3 mean: {all_n3_ape.mean():.3f}  d={ape_perm['observed_d']:.2f}  p={ape_perm['p_value']:.4f}")


if __name__ == "__main__":
    main()
