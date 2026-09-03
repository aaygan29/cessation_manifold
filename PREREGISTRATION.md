# Preregistration

Written before any real cessation or NIMHANS data was inspected. This
documents the hypotheses and kill criteria the apparatus is built to test,
so later real-data results cannot be tuned to look good after the fact.

## Background

Zarka et al. (2026, bioRxiv) describe meditation-induced "cessation" events
in experienced meditators: brief, discrete interruptions of subjective
experience with a distinctive neural signature. This project asks whether a
per-subject, calibrated distance-from-cessation readout on EEG features can
be built, validated with an honest abstention mechanism, and later shown
comparable across meditation traditions (NIMHANS cohort).

## Primary hypothesis

H1: A low-dimensional embedding of EEG features (microstate dynamics,
aperiodic exponent, avalanche criticality, signal complexity) built around a
cessation-anchored centroid produces a distance score that (a) is stable
within subject across repeated sessions, (b) separates meditators from
non-meditators in the expected direction, and (c) is destroyed by surrogate
data that preserves linear spectral content but not nonlinear structure.

## Kill criteria (gates)

1. **Within-subject reproducibility.** If repeated sessions from the same
   (synthetic, later real) subject do not land closer to each other than to
   other subjects' sessions on the manifold, the embedding is not capturing
   a subject-stable signal and the approach should not proceed to real
   cessation data.
2. **Group separation.** If non-meditator controls are not further from the
   cessation manifold than a meditator baseline, the readout has no
   discriminative validity and should not be reported as measuring anything
   about meditation.
3. **Surrogate specificity.** If IAAFT-surrogate EEG (same amplitude
   distribution and approximate power spectrum, randomized phase structure)
   produces the same distance-from-cessation scores as real data, the
   readout is picking up linear spectral content only, not the nonlinear
   structure it claims to detect, and should not be trusted.
4. **Conformal coverage.** If split-conformal intervals do not hold their
   nominal coverage rate on held-out data, no number from this pipeline
   should be reported as a validated finding; the honesty layer forces an
   abstention (`UnvalidatedClaimError`) rather than a silently miscalibrated
   number.

## What v0 can and cannot claim

v0 validates the apparatus (gates 1, 3, 4) on synthetic Kuramoto-network data
with a designed collapse structure, not on real cessation events. Gate 2 is
run as a partial check: a synthetic "meditator baseline" (near-critical, not
collapsed) stands in for the positive contrast, and real LEMON resting-state
EEG stands in for the non-meditator control, but the comparison is between a
synthetic signal and a real one, so a pass or fail here is informative about
the apparatus, not about meditators versus non-meditators. No claim about
real meditators, real cessation, or cross-tradition comparability is made by
this version. That claim requires Zarka or NIMHANS data, which is not
publicly available at the time of writing (see `data/README.md`).

## Analysis plan for when real cessation data lands

1. Replace the synthetic collapse mask with real cessation-onset annotations
   as the manifold anchor.
2. Re-run gates 1-4 unchanged (no gate logic changes once real data is
   substituted, only the data source).
3. Report whichever gates pass or fail as-is. A gate failure on real data is
   a reportable result, not a reason to change the gate's pass/fail
   threshold after seeing the data.
