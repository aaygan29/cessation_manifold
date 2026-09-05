# cessation_manifold

EEG cessation-detection apparatus with conformal uncertainty quantification.

## What this repository is
- A research apparatus for EEG feature extraction, manifold distance scoring, and conformal uncertainty gates.
- A synthetic-first validation stack with real EEG control-data pathways (BIDS + uploaded files).
- A system that should **abstain** when uncertainty/coverage constraints are not met.

## What is currently proven
- Synthetic pipeline stability checks are implemented across seeds and gates.
- Synthetic engine supports multiple biophysical scaffolds (Kuramoto, neural-mass-inspired, thalamo-cortical proxy).
- Gate 4 now uses adaptive, block-aware conformal splitting (session-aware) instead of random epoch splitting.
- Real-data ingestion path accepts uploaded EEG (MNE-readable formats) and reports artifact rejection and feature reliability.
- Feature extractors run behind explicit NaN/Inf sanitization guards and now include connectivity/cross-frequency families.

## What is not yet proven
- Actual cessation detection on real cessation-labeled EEG cohorts is **not established** here.
- Any cessation claim still requires cessation-labeled data and independent clinical adjudication.

## Honest Limits
- Current synthetic Gate 4 target coverage: **0.90** (with adaptive/block-aware conformal); finite-sample folds can still vary across seeds.
- Default synthetic anchor now assumes larger folds (`n_subjects >= 6`), but instability can remain under severe regime heterogeneity.
- Gate 1 ICC reproducibility is now stratified by regime (`collapsed`, `critical`, `control`) with permutation p-values; this can expose regime-specific failures.
- Real-data path currently validates robustness mechanics (loading, preprocessing, feature reliability), not cessation truth labels.
- Artifact rejection rates are reported, but ICLabel classification quality depends on optional dependency availability and EEG quality.

## Validation Roadmap (when cessation-labeled data is available)
1. Ingest cessation-labeled cohorts (e.g., Zarka/NIMHANS) through BIDS/upload pipeline.
2. Lock preprocessing and conformal calibration protocol before outcome analysis.
3. Re-run Gate 1–4 on real cessation labels with held-out session blocks.
4. Publish calibrated coverage and failure-mode tables by cohort/regime.

## Key Deliverables in this hardening pass
- Priority 1–4 code updates in:
  - `src/cessation_manifold/honesty/adaptive_conformal.py`
  - `src/cessation_manifold/pipeline.py`
  - `src/cessation_manifold/io/bids_loader.py`
  - `src/cessation_manifold/preprocessing/artifact_removal.py`
  - `src/cessation_manifold/features/microstates.py`
- Technical specification: `TECHNICAL_SPEC.md`
- Added tests for:
  - minimum larger conformal eval folds,
  - uploaded EEG ingestion metadata,
  - regime-stratified ICC reporting,
  - real uploaded EEG pipeline execution and reliability reporting.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest
PYTHONPATH=src python scripts/run_demo.py
PYTHONPATH=src python scripts/run_seed_sweep.py
```

## References
- Angelopoulos & Bates (2021), arXiv:2107.07511
- Papadopoulos et al. (2021), Elsevier conformal time-series chapter
- Barber et al. (2023), JRSS-B conditional guarantees
- Koo & Li (2016), ICC reporting guideline
- Schreiber & Schmitz (1996), IAAFT surrogates
- Pernet et al. (2018), MEEG best practices
- Roy et al. (2019), EEG ML review
