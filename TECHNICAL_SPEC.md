# TECHNICAL_SPEC

## Purpose
`cessation_manifold` is an EEG cessation-detection apparatus with conformal uncertainty quantification.  
It is designed to (1) validate feature/manifold machinery on synthetic and control real EEG and (2) abstain when calibration or scope is insufficient.

## Architecture
1. **Input layer**
   - Synthetic: `src/cessation_manifold/io/synthetic.py`
   - Real BIDS: `src/cessation_manifold/io/bids_loader.py::load_bids_eeg`
   - Uploaded EEG: `src/cessation_manifold/io/bids_loader.py::load_uploaded_eeg`
2. **Preprocessing**
   - Epoch sanitization and NaN/Inf guards: `preprocessing/robustness.py`
   - Artifact attenuation:
     - lightweight in-repo path: `preprocessing/artifact_removal.py::remove_artifacts`
     - MNE ICA + ICLabel-style path: `remove_artifacts_mne` (optional ICLabel dependency)
3. **Feature extraction**
   - Aperiodic: `features/aperiodic.py`
   - Complexity: `features/complexity.py`
   - Criticality: `features/criticality.py`
   - Microstates: `features/microstates.py`
4. **Embedding and distance**
   - Manifold fit/transform: `embed/manifold.py`
   - Distance to cessation centroid: `embed/distance.py`
5. **Honesty & uncertainty**
   - Adaptive, block-aware conformal: `honesty/adaptive_conformal.py`
   - Gate checks and abstention: `honesty/gates.py`
6. **Pipeline orchestration**
   - Synthetic validation: `pipeline.py::run_synthetic_pipeline`
   - Real uploaded feature-validation path: `pipeline.py::run_real_data_pipeline`

## Data Flow
`EEG -> sanitize -> artifact attenuation -> features -> matrix -> manifold -> distance -> conformal intervals -> gate verdicts`

For real uploaded EEG, output explicitly includes:
- feature reliability table (finite fraction, mean, std per feature),
- artifact rejection rate and provenance,
- claim scope metadata (`feature-validation-only`).

## Validation Protocol

### Gate 1 (Reproducibility)
- ICC(2,1) computed per synthetic regime (`collapsed`, `critical`, `control`) with permutation-null p-values.
- Collapsed regime remains canonical Gate 1 pass/fail channel; per-regime metrics expose failure localization.
- Anchor-size metadata is logged (`n_subjects`, sessions/subject, epochs total and by regime).

### Gate 3 (Surrogate break)
- IAAFT surrogate contrast verifies score degradation under destroyed phase structure.

### Gate 4 (Conformal coverage)
- Block/session-aware splitting replaces random epoch-level splitting.
- Adaptive fold sizing enforces larger calibration/test partitions (minimum 50 when enough epochs exist).
- Finite-sample interval inflation is applied to reduce undercoverage under heterogeneous block structure.

## Honest Scope Statements
- **Proven now**: machinery robustness on synthetic data; real-data feature-stack behavior on control EEG pathways.
- **Not proven now**: clinical cessation detection claims without cessation-labeled datasets and adjudication.

## References (theory + EEG practice)
- Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511.
- Papadopoulos, H., Nikolopoulos, K., & Vovk, V. (2021). Conformal Prediction for Time Series. In *Conformal Prediction for Reliable Machine Learning*. Elsevier. https://doi.org/10.1016/B978-0-12-809715-7.00020-9
- Barber, R. F., Candes, E., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction with conditional guarantees. *JRSS Series B*. https://doi.org/10.1093/jrsssb/qkad020
- Koo, T. K., & Li, M. Y. (2016). A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research. *J Chiropr Med*, 15(2), 155-163. https://doi.org/10.1016/j.jcm.2016.02.012
- Schreiber, T., & Schmitz, A. (1996). Improved surrogate data for nonlinearity tests. *Phys Rev Lett*, 77(4), 635-638. https://doi.org/10.1103/PhysRevLett.77.635
- Pernet, C. R., et al. (2018). Best Practices in Data Analysis and Sharing in Neuroimaging using MEEG. *NeuroImage*. https://doi.org/10.1016/j.neuroimage.2017.05.033
- Roy, Y., et al. (2019). Deep learning-based electroencephalography analysis: a systematic review. *J Neural Eng*. https://doi.org/10.1088/1741-2552/ab260c
