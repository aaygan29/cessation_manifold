# cessation_manifold

A per-subject, calibrated "distance-from-cessation" readout on EEG features.
It anchors on meditation-induced cessation events as described by Zarka et
al. (2026, Harvard MGH, bioRxiv) and aims to be comparable across meditation
traditions studied by NIMHANS (Rahul Venugopal). Neither dataset is publicly
downloadable yet (see `data/README.md`), so this version validates the
apparatus on synthetic ground truth and re-validates it on real EEG by
recovering a known signal (the Berger effect) from the public LEMON dataset.

## What this is, honestly

This is v0 of an instrument, not a finding about meditation. The scientific
question ("does distance-from-cessation separate meditators from controls,
and hold up across traditions") cannot be answered without the Zarka or
NIMHANS data. What v0 delivers:

- A working feature -> embedding -> distance -> conformal-interval pipeline.
- A synthetic ground-truth generator (Kuramoto oscillator network with a
  driven collapse) to test that pipeline's machinery before real cessation
  data exists.
- A data-adapter layer (`io/bids_loader.py`) so that swapping in real
  cessation data later is a config change, not a rewrite.
- An honesty layer that forces the pipeline to abstain (`UnvalidatedClaimError`)
  rather than report a number when conformal coverage fails or an input
  falls outside the calibration set's range.
- A real-data re-validation (Gate 0 below) proving the feature stack
  recovers a canonical published finding on LEMON EEG.

## Gate 0: real-data method re-validation (Berger effect on LEMON)

Before claiming anything about cessation, the feature and distance machinery
has to recover a known EEG signal on real data. Prior work (Berger 1929;
Babayan et al. 2019 for LEMON specifically) says posterior relative alpha
(8 to 13 Hz) rises sharply when the eyes close. We ran the same feature
stack this project uses for the manifold on LEMON subjects `sub-010002`
and `sub-010003` (raw BrainVision fetched via
`scripts/fetch_lemon_subset.sh`), segmented on the S200 (eyes-open) and
S210 (eyes-closed) block markers, 4-second epochs on 19 posterior channels.

| Subject     | alpha_rel EO | alpha_rel EC | Cohen's d | p (Mann-Whitney) |
|-------------|--------------|--------------|-----------|------------------|
| sub-010002  | 0.148        | 0.304        | 1.91      | 1.6e-28          |
| sub-010003  | 0.329        | 0.534        | 1.77      | 6.9e-24          |
| pooled      | 0.238        | 0.419        | 1.25      | 1.1e-29          |

**Gate 0 status: PASS.** Direction, effect size, and significance all in
the expected regime. Reproduce with `python scripts/validate_lemon_berger.py`;
full per-feature stats land in `results/lemon_berger_validation.json` with
a provenance stamp (dataset id, git commit, config hash, UTC timestamp).
This proves the apparatus recovers a known signal from real EEG. It does
NOT prove any claim about cessation (cessation is not eyes-closed rest);
it is the precondition for making one once Zarka or NIMHANS data arrives.

## Kill-criteria gates (apparatus, on synthetic ground truth)

See `PREREGISTRATION.md` for full rationale.

| Gate | Check | v0 status |
|------|-------|-----------|
| 1 | Within-subject reproducibility across repeated sessions | Runs and reported on synthetic. |
| 2 | Non-meditator controls sit further from the manifold than meditator baseline | Partial. Real controls (LEMON) can be passed in; a real meditator arm needs Zarka or NIMHANS data. |
| 3 | IAAFT / phase-randomized surrogate EEG breaks the score | Runs and reported on synthetic (surrogate mean 2.63 > real 2.01). |
| 4 | Split-conformal intervals hold nominal coverage; abstain when they don't | Runs and reported on synthetic (achieved 1.0 vs 0.9 target). |

Run `python scripts/run_demo.py` and see `results/report.html` for the
current numbers in your environment.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest                                     # 13 apparatus tests
bash scripts/fetch_lemon_subset.sh         # ~600 MB, 2 min
python scripts/validate_lemon_berger.py    # Gate 0 on real EEG
python scripts/run_demo.py                 # Gates 1, 3, 4 on synthetic
```

## Repo layout

```
src/cessation_manifold/
  io/           bids_loader (MNE-BIDS) and synthetic Kuramoto-with-collapse
  features/     microstates, aperiodic (1/f), avalanche criticality, LZ/DFA, IAAFT surrogates
  embed/        PCA + UMAP manifold; distance from cessation centroid
  honesty/      split-conformal predictor, gate() decorator, UnvalidatedClaimError, provenance stamps
  pipeline.py   config -> features -> embed -> distance -> conformal -> report
  report.py     writes results/report.html
tests/          13 tests, all pass
scripts/        fetch_lemon_subset.sh, fetch_openneuro_ds001787.sh, validate_lemon_berger.py, run_demo.py
configs/        synthetic.yaml, lemon.yaml, openneuro_meditation.yaml
data/README.md  dataset table with real URLs, licenses, and reachability notes
```

## Honest limits of v0

- Zarka (Harvard MGH cessation) and NIMHANS 4-tradition data are not openly
  downloadable. The cessation-manifold anchor is trained on the synthetic
  collapse for now.
- Gate 2 is not scientifically informative until a real meditator arm lands.
- ds001787 (OpenNeuro meditation) is verified reachable but not yet wired
  into the manifold; it lacks cessation-onset labels.
- The Berger re-validation uses 2 LEMON subjects. Two subjects is enough
  for a within-subject positive control on a signal this large, but is not
  a population-level claim.

## Citations

- Berger H. (1929). Uber das Elektrenkephalogramm des Menschen.
- Babayan A. et al. (2019). A mind-brain-body dataset of MRI, EEG,
  cognition, emotion, and peripheral physiology in young and old adults.
  Scientific Data 6:180308. (LEMON, ds000221)
- Zarka D. et al. (2026). EEG brain reconfiguration during
  meditation-induced extended cessation of consciousness. bioRxiv.
- Venugopal R. et al. (2026). Temporal EEG signatures of meditation
  experience. Mindfulness.

## License

MIT. Data downloaded by scripts here retains its original license (see
`data/README.md`).
