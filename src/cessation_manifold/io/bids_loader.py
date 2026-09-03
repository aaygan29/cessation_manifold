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
