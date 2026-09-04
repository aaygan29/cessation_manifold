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
has to recover a known EEG signal on real data, on the RIGHT sensors, with
a null that respects the block structure. Prior work (Berger 1929; Babayan
et al. 2019 for LEMON specifically) says posterior relative alpha (8 to 13
Hz) rises sharply when the eyes close, more strongly at posterior sites
than at frontal ones. We ran the same feature stack this project uses for
the manifold on LEMON subjects `sub-010002` and `sub-010003` (raw
BrainVision fetched via `scripts/fetch_lemon_subset.sh`), segmented on the
S200 (eyes-open) and S210 (eyes-closed) block markers, 4-second epochs.

| Arm         | alpha_rel EO | alpha_rel EC | Cohen's d | Block-perm p (5000) |
|-------------|--------------|--------------|-----------|---------------------|
| posterior (19 ch, primary)  | 0.238 | 0.419 | 1.25 (pooled) | < 0.0002 (observed 1.62) |
| frontal (specificity ctrl)  | 0.153 | 0.196 | 0.49 (pooled) | (see JSON) |
| specificity: d_post - d_frontal | | | **0.76** (threshold 0.3) | PASS |

Per-subject posterior d: sub-010002 = 1.91, sub-010003 = 1.77. Per-subject
frontal d: sub-010002 = 1.08, sub-010003 = 0.43. Frontal shows a smaller
Berger-consistent effect (arousal / referencing), posterior shows the
canonical strong effect. The specificity contrast (post minus frontal) is
what makes this Berger rather than generic arousal.

**Gate 0 status: PASS on all three requirements** (direction + magnitude,
block-permutation p < 0.05, posterior-vs-frontal specificity > 0.3). The
old per-epoch Mann-Whitney p-values (1.1e-29 pooled) are kept in the JSON
for reference but are NOT primary: they assume independence across
autocorrelated 4-second windows within a block, which is false. The
block-level permutation test resamples the 32 blocks (16 per subject) and
is the honest inferential number. Reproduce with
`PYTHONPATH=src python scripts/validate_lemon_berger.py`; full stats land
in `results/lemon_berger_validation.json` with a provenance stamp.

This proves the apparatus recovers a known signal from real EEG on the
right sensors under a defensible null. It does NOT prove any claim about
cessation (cessation is not eyes-closed rest); it is the precondition for
making one once Zarka or NIMHANS data arrives.

## Kill-criteria gates (apparatus, on synthetic ground truth)

See `PREREGISTRATION.md` for full rationale.

| Gate | Check | v0 status |
|------|-------|-----------|
| 1 | Within-subject reproducibility across repeated sessions | **FAILS across the 10-seed sweep** (see below). Single-seed demo hid this; the current 0.6 threshold is wrong for this apparatus, or the manifold is not actually subject-stable. Honest failure surfaced by Finding 3. |
| 2 | Non-meditator controls sit further from the manifold than meditator baseline | Partial. Real controls (LEMON) can be passed in; a real meditator arm needs Zarka or NIMHANS data. |
| 3 | IAAFT / phase-randomized surrogate EEG breaks the score | Runs and reported on synthetic (surrogate mean 2.63 > real 2.01). |
| 4 | Split-conformal intervals hold nominal coverage; abstain when they don't | **v0 fix (Finding 1):** target changed from manifold distance (leaky, gave 1.0) to per-epoch collapse fraction (independent). Sweep-mean coverage 0.93 [0.85, 0.99] over 10 seeds. Passes on average, but bounces {min 0.667, max 1.000}. Wide per-seed variance is itself a finding: this test-set size (~n=15) is not enough for calibrated split-conformal on this noise level. |

Run `python scripts/run_demo.py` and see `results/report.html` for the
current numbers in your environment.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest                                     # 13 apparatus tests
bash scripts/fetch_lemon_subset.sh                          # ~600 MB, 2 min
PYTHONPATH=src python scripts/validate_lemon_berger.py      # Gate 0 (real EEG, incl. specificity + block-perm)
PYTHONPATH=src python scripts/run_demo.py                   # Gates 1, 3, 4 on synthetic (single seed)
PYTHONPATH=src python scripts/run_seed_sweep.py             # 20-seed sweep with 95% bootstrap CI
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

## 10-seed sweep summary (Finding 3)

Mean and bootstrap 95% CI over 10 seeds (`scripts/run_seed_sweep.py`,
default `N_SEEDS=10`; set env var `N_SEEDS=20` for the full sweep):

| Metric | mean | 95% CI | min | max | pass criterion | verdict |
|---|---|---|---|---|---|---|
| gate1_within_subject_ratio | 0.885 | [0.832, 0.932] | 0.686 | 0.971 | < 0.6 | **FAIL every seed** |
| gate3_surrogate_mean_distance | 2.500 | [2.404, 2.603] | 2.179 | 2.811 | > gate3_real x 1.2 | PASS every seed |
| gate3_real_mean_distance | 1.792 | [1.668, 1.936] | 1.547 | 2.274 | (contrast for gate 3) | PASS every seed |
| gate4_conformal_coverage | 0.927 | [0.853, 0.987] | 0.667 | 1.000 | close to 0.9 | mean OK, per-seed variance is a real finding |

Two honest failures the single-seed run hid:

1. **Gate 1 fails on every seed.** The current 0.6 threshold is not achieved
   on any run. Either the threshold is wrong (needs a documented recalibration
   on synthetic before real data touches this) or the PCA + UMAP manifold
   here is not actually subject-stable across the 3 synthetic sessions per
   subject. The correct next step is to widen the manifold anchor
   (more subjects, more sessions) and re-derive the threshold, not to relax
   the current one to pass.
2. **Gate 4 coverage bounces 0.667 to 1.000 across seeds.** Mean lands at
   0.93 (nominal 0.9 is inside the CI), so the abstract calibration story
   holds on average, but any single run can undershoot substantially. Root
   cause is a small test fold (~15 epochs); expanding the calibration + test
   folds is the honest fix before this gate can be reported as reliably passing.

Neither is fatal for v0's apparatus goals; both are exactly the failures
Finding 3 was designed to surface. Real cessation data cannot land until
these are resolved.

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
