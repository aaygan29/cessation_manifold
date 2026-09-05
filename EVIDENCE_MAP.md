# EVIDENCE_MAP

Structured literature-to-code mapping for biophysical and mathematical validity.

## Peer-reviewed evidence

| Area | Evidence | Module(s) |
|---|---|---|
| Perturbational complexity benchmark | Casali et al., 2013, Sci Transl Med, DOI: 10.1126/scitranslmed.3006294 | `src/cessation_manifold/features/complexity.py` |
| Fast PCIst methodology | Comolatti et al., 2019, Brain Stimulation, DOI: 10.1016/j.brs.2019.03.007 | `.../features/complexity.py`, `.../pipeline.py` |
| Resting-state consciousness markers incl. wSMI family | Sitt et al., 2014, Brain, DOI: 10.1093/brain/awu141 | `.../features/connectivity.py`, `.../features/aperiodic.py`, `.../features/complexity.py` |
| EEG microstate methodology | Michel & Koenig, 2018, NeuroImage, DOI: 10.1016/j.neuroimage.2017.11.062 | `.../features/microstates.py`, `.../pipeline.py` |
| ICC validity/reporting | Koo & Li, 2016, J Chiropr Med, DOI: 10.1016/j.jcm.2016.02.012 | `.../honesty/icc.py`, `.../pipeline.py` |
| Surrogate-data nonlinearity controls | Schreiber & Schmitz, 1996, Phys Rev Lett, DOI: 10.1103/PhysRevLett.77.635 | `.../features/surrogates.py` |
| Conditional coverage framing | Barber et al., 2023, JRSS-B, DOI: 10.1093/jrsssb/qkad020 | `.../honesty/adaptive_conformal.py`, `.../pipeline.py` |
| Time-series conformal | Papadopoulos et al., 2021, Elsevier chapter, DOI: 10.1016/B978-0-12-809715-7.00020-9 | `.../honesty/adaptive_conformal.py` |

## Preprint evidence (explicitly non-peer-reviewed)

| Area | Evidence | Module(s) |
|---|---|---|
| Resting-state criticality predicts PCI and LOC in anesthesia | Toker et al., 2023, bioRxiv: 10.1101/2023.10.26.564247 | `.../features/criticality.py`, `.../io/synthetic.py` |
| Meditation-induced cessation microstate reconfiguration | Zarka et al., 2026, bioRxiv | `.../features/microstates.py`, `.../pipeline.py` |

## Biophysical model realism targets

| Mechanistic target | Program location |
|---|---|
| Phase synchrony transition scaffold (Kuramoto) | `.../io/synthetic.py::simulate_kuramoto_eeg` |
| Mesoscopic neural-mass-like dynamics with subcortical drive | `.../io/synthetic.py::simulate_neural_mass_eeg` |
| Thalamo-cortical / arousal modulation proxy | `.../io/synthetic.py::simulate_thalamocortical_eeg` |

## Mathematical validity targets

| Target | Program location |
|---|---|
| Gate 1 reliability + permutation null | `.../honesty/icc.py`, `.../pipeline.py` |
| Gate 4 conditional and subgroup coverage diagnostics | `.../honesty/adaptive_conformal.py`, `.../pipeline.py` |
| Distribution-shift diagnostics across train/calib/test | `.../pipeline.py` |
| Centroid stability bootstrap | `.../pipeline.py` |
| Benchmark matrix over seeds/ablations | `.../scripts/run_validation_report.py` |
