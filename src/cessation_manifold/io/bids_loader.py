"""MNE / MNE-BIDS based loader for real EEG datasets.

Real EEG (LEMON, the OpenNeuro meditation set, or a future Zarka/NIMHANS
drop) is not committed to this repo. Run the matching fetch script in
scripts/ first, then point a config at the resulting local BIDS root.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np


@dataclass
class LoadedEpochs:
    data: np.ndarray  # (n_epochs, n_channels, n_samples)
    sfreq: float
    ch_names: list
    subject_id: str
    session_id: str
    source: str  # dataset id, for provenance


@dataclass
class LoadedRawEEG:
    raw: "mne.io.BaseRaw"
    sfreq: float
    ch_names: list[str]
    session_id: str
    source: str


def _read_uploaded_raw(path: Path):
    import mne

    suffix = path.suffix.lower()
    readers = {
        ".fif": mne.io.read_raw_fif,
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".vhdr": mne.io.read_raw_brainvision,
        ".set": mne.io.read_raw_eeglab,
        ".gdf": mne.io.read_raw_gdf,
    }
    reader = readers.get(suffix)
    if reader is None:
        supported = ", ".join(sorted(readers))
        raise ValueError(f"Unsupported EEG extension {suffix!r}; supported: {supported}")
    return reader(path, preload=True, verbose=False)


def load_bids_eeg(
    bids_root: str,
    subject: str,
    task: str,
    session: str | None = None,
    epoch_length_s: float = 5.0,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    dataset_id: str = "unknown",
) -> LoadedEpochs:
    """Load one subject/task from a local BIDS dataset and epoch it.

    Raises FileNotFoundError with a clear message if the local data has not
    been fetched yet, rather than silently falling back to synthetic data.
    """
    root = Path(bids_root)
    if not root.exists():
        raise FileNotFoundError(
            f"BIDS root {bids_root!r} does not exist. Run the fetch script for "
            f"dataset {dataset_id!r} in scripts/ before loading real data."
        )

    import mne
    from mne_bids import BIDSPath, read_raw_bids

    bids_path = BIDSPath(
        subject=subject, task=task, session=session, root=root, datatype="eeg"
    )
    raw = read_raw_bids(bids_path, verbose=False)
    raw.load_data()
    raw.filter(l_freq, h_freq, verbose=False)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=epoch_length_s, preload=True, verbose=False
    )
    data = epochs.get_data()
    return LoadedEpochs(
        data=data,
        sfreq=float(raw.info["sfreq"]),
        ch_names=list(raw.ch_names),
        subject_id=subject,
        session_id=session or "n/a",
        source=dataset_id,
    )


def load_uploaded_eeg(
    eeg_path: str,
    dataset_id: str = "uploaded",
    epoch_length_s: float = 5.0,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
) -> LoadedRawEEG:
    """Load uploaded EEG using MNE auto-reader with robust metadata handling.

    References
    ----------
    Pernet, C. R., et al. (2018). Best Practices in Data Analysis and
    Sharing in Neuroimaging using MEEG. NeuroImage, 271, 119-132.
    https://doi.org/10.1016/j.neuroimage.2017.05.033
    """
    path = Path(eeg_path)
    if not path.exists():
        raise FileNotFoundError(f"EEG file {eeg_path!r} does not exist.")

    raw = _read_uploaded_raw(path)
    sfreq = float(raw.info.get("sfreq", 0.0))
    if sfreq <= 0:
        raise ValueError(f"Could not determine sampling rate from EEG file {eeg_path!r}.")

    nyquist = sfreq / 2.0
    upper_bound = nyquist - 1e-3
    if upper_bound <= float(l_freq):
        raise ValueError(
            f"Invalid filter range for sfreq={sfreq}: Nyquist={nyquist:.3f}Hz cannot support l_freq={l_freq}."
        )
    safe_h_freq = min(float(h_freq), upper_bound)
    if safe_h_freq <= float(l_freq):
        raise ValueError(f"Invalid filter range for sfreq={sfreq}: l_freq={l_freq}, h_freq={h_freq}")
    raw.filter(float(l_freq), safe_h_freq, verbose=False)
    raw.info["description"] = (
        f"{raw.info.get('description', '')} | source={dataset_id} | epoch_length_s={epoch_length_s}"
    ).strip()

    session_id = path.stem

    return LoadedRawEEG(
        raw=raw,
        sfreq=sfreq,
        ch_names=list(raw.ch_names),
        session_id=session_id,
        source=dataset_id,
    )
