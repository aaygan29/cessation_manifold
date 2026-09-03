# Data sources

This project does not commit any raw EEG. Each dataset below has a fetch
script in `scripts/` that pulls a small subset into `data/raw/<name>/`
(gitignored). Fetch time is what was actually observed running the scripts
in this repo, not an estimate.

## Datasets used or evaluated

| Dataset | Role | Status | Verified how |
|---|---|---|---|
| MPI Leipzig LEMON EEG resting-state (OpenNeuro `ds000221`) | Primary non-meditator control (Gate 2) | **WIRED.** Fetch script tested end-to-end in this repo. | `scripts/fetch_lemon_subset.sh` run live: pulled `sub-010002` and `sub-010003` (BrainVision `.eeg`/`.vhdr`/`.vmrk`, ~305 MB/subject) from the GWDG mirror in under 2 minutes. Also confirmed the dataset resolves on OpenNeuro via its GraphQL API (`dataset(id: "ds000221")` returns snapshot tag `1.0.0`). |
| EEG meditation study (OpenNeuro `ds001787`) | Candidate real meditation EEG | **Existence verified, not yet wired into gates.** | Confirmed live via OpenNeuro GraphQL (`dataset(id: "ds001787")` returns snapshot tag `1.1.1`) and cross-checked against the dataset's public GitHub mirror and OpenNeuro listing. `scripts/fetch_openneuro_ds001787.sh` is written but not run in this session (needs `@openneuro/cli`, not installed here). Not wired into `pipeline.py`'s manifold/conformal steps: the dataset probes concentration/mind-wandering every ~2 minutes, it does not annotate discrete cessation onsets, so it cannot anchor the cessation centroid the way the synthetic collapse mask does. Useful once a labeling scheme is defined, or once Zarka/NIMHANS data lands. |
| PhysioNet EEG Motor Movement/Imagery (`eegmmidb`) | Fallback eyes-closed rest source | Reachable (verified `HTTP 200` on the dataset index), not fetched or wired in this version. Kept as a documented fallback per the task spec; LEMON is the primary control and was sufficient for v0. | `curl -I https://physionet.org/files/eegmmidb/1.0.0/` returned `HTTP/2 200`. |
| Zarka et al. 2026 cessation dataset (Harvard MGH) | Anchor dataset for the actual cessation manifold | **Not openly downloadable.** The paper (bioRxiv, https://www.biorxiv.org/content/10.64898/2026.02.10.705005v1.full) does not currently expose a public data repository link. | Checked the bioRxiv page and MGH meditation lab page (https://meditation.mgh.harvard.edu/files/Zarka_26_bioRxiv.pdf); no dataset DOI or repository link found. |
| NIMHANS 4-tradition meditation EEG (Rahul Venugopal) | Cross-tradition comparability | **Not openly downloadable.** Rahul Venugopal's GitHub (https://github.com/rahulvenugopal) hosts analysis code, not the raw multi-tradition EEG itself. | Checked github.com/rahulvenugopal repositories; no public raw-data release located for the 4-tradition cohort. |

## LEMON (`ds000221`) — how to fetch and use

```bash
bash scripts/fetch_lemon_subset.sh
```

This downloads 2 subjects' raw BrainVision recordings (`EEG_Raw_BIDS_ID`
branch of the archive) from the GWDG mirror:

```
https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID/<sub-id>/RSEEG/<sub-id>.{vhdr,vmrk,eeg}
```

License: LEMON data is released under a CC0-like MPI-CBS data use agreement
for research purposes; see the archive's `EEG_Info` and the dataset's
OpenNeuro page (https://openneuro.org/datasets/ds000221) for the exact terms
before any redistribution.

**Note on BIDS-readiness:** the GWDG mirror serves raw BrainVision files, not
a full BIDS tree with `dataset_description.json` / `participants.tsv`. To use
`io/bids_loader.py`'s `mne-bids` path directly, either fetch the dataset via
OpenNeuro's own BIDS-formatted release (`ds000221`) or point a lighter-weight
loader at the BrainVision files directly with `mne.io.read_raw_brainvision`.
`configs/lemon.yaml` assumes a proper BIDS root; adjust `bids_root` to match
whichever fetch path you use.

## OpenNeuro meditation (`ds001787`) — how to fetch

```bash
npm install -g @openneuro/cli   # not run in this session
bash scripts/fetch_openneuro_ds001787.sh
```

Marked `not-yet-wired` in `configs/openneuro_meditation.yaml`. Fetching it is
useful for feature-extraction smoke tests today; it is not used to compute
any of the four gates in this version.

## What would change once Zarka / NIMHANS data lands

`io/bids_loader.py` already returns the same `LoadedEpochs` shape regardless
of source. Wiring in Zarka or NIMHANS data means: (1) writing a fetch script
once a download path exists, (2) adding a config pointing at it with
cessation-onset event annotations, (3) replacing the synthetic
`collapse_mask` label source in `pipeline.py` with the real onset labels. No
change to the feature, embedding, or honesty-layer code is required.
