# ROBUSTNESS

- Preprocessing runs through optional artifact attenuation before feature extraction.
- ICA scores components against embedded blink/muscle/spike dictionaries with adaptive MAD thresholds.
- Wavelet denoising uses a lightweight Haar shrinkage fallback for non-stationary artifacts.
- Feature extraction pads short epochs, replaces non-finite values with documented neutral fallbacks, and records replacements in provenance.
- Adaptive conformal uses block-aware splits, cross-validated residual bands, and flags unstable coverage for review.
- Synthetic robustness should outperform artifact-injected inputs; real-data claims remain exploratory until cessation-labeled datasets arrive.
