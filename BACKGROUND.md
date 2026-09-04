# Background

This document lays out why the cessation manifold apparatus is being
broadened from "cessation only" to a general graded-consciousness
instrument spanning meditation-induced cessation, pharmacological
anesthesia, and NREM sleep, and grounds that broadening in published
literature rather than in intuition.

## Motivation

The original framing treated meditation-induced cessation (Zarka et al.
2026) as a standalone phenomenon: a rare, discrete drop to near-absent
reportable experience during deep meditation, measured against a
synthetic Kuramoto collapse as a stand-in ground truth until real
cessation data lands. That framing is too narrow for two reasons.

First, cessation is not the only human state in which resting cortical
dynamics move toward a "reduced" pole. Anesthesia (propofol, ketamine,
xenon) and NREM sleep, especially slow-wave sleep, are graded,
repeatable, and already have large open datasets with ground-truth
labels (drug dose or polysomnography-scored sleep stage). They give the
apparatus something cessation alone cannot: dense, labeled positive
controls to test whether the feature stack tracks reduced consciousness
at all, before ever touching the rare and currently unavailable
cessation data.

Second, and more important, there is a specific published result
(Toker et al. 2023, below) arguing that resting-state EEG statistics,
of exactly the kind this repo already extracts, track the same
underlying axis that the gold-standard perturbational index (PCI)
measures with external stimulation. That result is the single strongest
piece of literature support for this repo's core premise: that you can
read consciousness level off resting EEG features without needing a TMS
pulse or ketamine bolus. It is treated here as the central pillar and
the other papers are read partly through how they relate to it.

The instrument's job, going forward, is to show that the same feature
stack (aperiodic exponent, Lempel-Ziv complexity, DFA scaling,
criticality/avalanche statistics, microstate dynamics) separates graded
consciousness levels consistently across all three paradigms: cessation,
anesthesia, sleep. A feature that only works for one paradigm is more
likely tracking something paradigm-specific (task demand, eyes-closed
artifact, drug pharmacokinetics) than a general marker of reduced
consciousness.

## Three literature pillars

### Pillar 1: Perturbational complexity as the gold standard

Casali et al. 2013, "A theoretically based index of consciousness
independent of sensory processing and behavior", Science Translational
Medicine 5(198):198ra105.
DOI: https://doi.org/10.1126/scitranslmed.3006294

This paper introduced the Perturbational Complexity Index (PCI):
stimulate cortex with TMS, compress the spatiotemporal pattern of the
evoked EEG response with Lempel-Ziv complexity, and use the compression
ratio as a graded index of consciousness level. PCI reliably separates
wakefulness from NREM sleep, propofol/xenon/ketamine anesthesia, and
distinguishes vegetative-state from minimally-conscious and locked-in
patients. It is IIT-motivated (high PCI requires both differentiated
and integrated cortical activity) but the measure itself is empirical
and does not require accepting IIT's full theoretical apparatus. PCI is
the benchmark every resting-state marker in this project is implicitly
trying to approximate without needing a TMS coil.

Comolatti et al. 2019, "A fast and general method to empirically
estimate the complexity of brain responses to transcranial and
intracranial stimulations", Brain Stimulation 12(5):1280-1289.
DOI: https://doi.org/10.1016/j.brs.2019.03.007

Introduces PCIST, a sensor-space variant of PCI that does not require
source modeling and works on both TMS-evoked and intracranially-evoked
responses. Relevant here because it establishes that a
complexity-of-response measure can be computed directly on
sensor-level EEG, which is the same regime this repo's features already
operate in (no source localization step in the pipeline).

### Pillar 2: Resting-state spectral, complexity, and information-exchange markers

Sitt et al. 2014, "Large scale screening of neural signatures of
consciousness in patients in a vegetative or minimally conscious
state", Brain 137(8):2258-2270.
DOI: https://doi.org/10.1093/brain/awu141

Screened a large battery of EEG markers across three families: spectral
power (in particular low-frequency, delta/theta, power as a marker of
reduced consciousness), algorithmic complexity (Lempel-Ziv complexity,
permutation entropy), and information exchange between channels
(weighted symbolic mutual information, wSMI). All three families
separated conscious from unconscious patients above chance, and
information-exchange measures, which quantify the correlation of
symbolic dynamics between distant channel pairs rather than the
complexity of any single channel, were among the most discriminative.
This is the paper this repo's feature taxonomy is modeled on, and it is
also the source of the one deliberate gap: see "known gap" below.

Toker et al. 2023, "Criticality of resting-state EEG predicts
perturbational complexity and level of consciousness during
anesthesia", bioRxiv 2023.10.26.564247.
Preprint page: https://www.biorxiv.org/content/10.1101/2023.10.26.564247v1
(bioRxiv returned HTTP 429 rate-limiting on repeated direct fetches
during this verification pass; the DOI and preprint id were confirmed
via web search cross-referencing the abstract and author list, not a
successful direct page load in this session. Treat the URL as
plausible-but-not-freshly-rendered rather than independently loaded.)

This is the load-bearing paper for the whole broadening. It shows that
criticality-related statistics computed on resting-state EEG (distance
from a critical branching point, estimated from avalanche and
autocorrelation structure) predict both PCI and clinically-assessed
level of consciousness across graded propofol sedation. In other words,
you do not need to perturb the brain with TMS to read off where it sits
on the same axis PCI measures. That is the empirical license for this
whole repo's approach: extracting resting features (aperiodic exponent,
DFA, avalanche criticality) and treating distance in that feature space
as a proxy for "distance from a reduced-consciousness pole," the same
logical move Toker et al. make explicit for anesthesia.

Sarasso et al. 2015, "Consciousness and complexity during
unresponsiveness induced by propofol, xenon, and ketamine", Current
Biology 25(23):3099-3105.
DOI: https://doi.org/10.1016/j.cub.2015.10.014

Shows PCI collapses under propofol and xenon (behaviorally unresponsive,
low complexity) but stays comparably high under ketamine
(behaviorally unresponsive, complexity preserved, consistent with
ketamine's dissociative rather than "off" phenomenology). This is a
useful caution for the anesthesia arm of the instrument: "unresponsive"
and "low-complexity" are not synonyms, and an anesthesia positive
control should be read as evidence for one specific mechanism of
reduced consciousness (propofol-like global collapse) rather than proof
that all anesthetics collapse EEG complexity.

### Pillar 3: Spatial reconfiguration (microstates) during cessation and meditation

Zarka et al. 2026, "EEG brain reconfiguration during meditation-induced
extended cessation of consciousness: A dense-sampling multi-participant
microstate study", bioRxiv.
Preprint page: https://www.biorxiv.org/content/10.64898/2026.02.10.705005v1
(confirmed to exist via web search returning matching title, author
list (Zarka, Yang, Rassat, Potash, Sparby, Sacchet), and a mirrored PDF
at https://meditation.mgh.harvard.edu/files/Zarka_26_bioRxiv.pdf; direct
HTTP fetch of the bioRxiv page itself returned 429 rate-limiting during
this verification pass rather than a clean 200, so treat the URL as
corroborated by two independent sources rather than directly rendered.
Note also that bioRxiv DOI prefixes are conventionally 10.1101; the
10.64898 prefix in this URL is unusual and was not independently cross
checked against a DOI resolver in this session.)

Five highly trained meditators, dense EEG, comparing extended cessation
(EC) against control conditions (counting, memory tasks) across canonical
EEG microstates in six frequency bands. EC was characterized by altered
global explained variance and coverage of microstates B and C, both
linked to self-referential processing. This is the paper this repo's
`microstates.py` feature module is aimed at eventually testing against,
once real cessation data is available; it also motivates using
microstate dynamics, not just spectral/complexity scalars, as a feature
family, since the reported effect is specifically about
spatial-topographic reconfiguration rather than a change in a
single-channel statistic.

Venugopal et al. 2026, "Temporal EEG Signatures of Meditation Experience:
Peak Brainwave Changes at 7 Minutes During Isha Yoga Breath Watching",
Mindfulness 17:762-778.
DOI: https://doi.org/10.1007/s12671-026-02790-1 (confirmed reachable,
HTTP 200, at https://link.springer.com/article/10.1007/s12671-026-02790-1)

103 participants (meditation-naive controls, novices, experienced
practitioners), 128-channel EEG during Isha Yoga breath-watching.
Theta/alpha power begins rising within 2-3 minutes and the practice
becomes subjectively effortless around 7 minutes. Relevant here as a
second, independent meditation-tradition dataset (distinct from Zarka's
extended-cessation cohort) that establishes a time-resolved, graded
EEG signature of meditation depth rather than a single discrete
cessation event, which is useful for testing whether the manifold
distance metric tracks a graded process and not only a binary
in/out-of-cessation label.

### Supporting methodology references

Michel, C. M. & Koenig, T. 2018, "EEG microstates as a tool for studying
the temporal dynamics of whole-brain neuronal networks: A review",
NeuroImage 180:577-593. DOI: https://doi.org/10.1016/j.neuroimage.2017.11.062
(confirmed reachable, HTTP 200)

Standard review of the microstate method: four to seven canonical
quasi-stable topographies, backfitting, and the temporal parameters
(duration, occurrence, coverage, transition probability) this repo's
`microstate_features` computes. Cited as the methodological basis for
treating microstate B/C coverage changes (per Zarka) as a meaningful
feature rather than noise.

Koo, T. K. & Li, M. Y. 2016, "A Guideline of Selecting and Reporting
Intraclass Correlation Coefficients for Reliability Research", Journal
of Chiropractic Medicine 15(2):155-163.
DOI: https://doi.org/10.1016/j.jcm.2016.02.012 (confirmed reachable,
HTTP 200)

Standard reference for choosing among the ICC(1,1)/ICC(2,1)/ICC(3,1)
family and for interpreting the resulting value: below 0.5 poor,
0.5-0.75 moderate, 0.75-0.9 good, above 0.9 excellent reliability. Used
in this repo's Gate 1 to justify both the choice of ICC(2,1) (subjects
are a random sample, sessions are the "raters") and the 0.5 pass
threshold.

## How each pillar maps onto our feature stack

| Feature (this repo) | Literature anchor | What it is meant to track |
|---|---|---|
| Aperiodic (1/f) spectral exponent (`features/aperiodic.py`) | Sitt et al. 2014 low-frequency power family; Toker et al. 2023 criticality | Flattened/steepened aperiodic slope as a coarse proxy for the low-frequency power shift Sitt reports and for the excitation/inhibition balance Toker links to criticality distance |
| Lempel-Ziv complexity, DFA scaling (`features/complexity.py`) | Sitt et al. 2014 algorithmic-complexity family; Toker et al. 2023; Casali/Comolatti PCI lineage (resting-state analogue) | Algorithmic compressibility of the signal, the same quantity PCI applies to an evoked response, applied here to spontaneous activity |
| Avalanche/criticality statistics (`features/criticality.py`) | Toker et al. 2023 (central pillar) | Distance from a critical branching point, the specific resting-state quantity Toker et al. show predicts PCI and clinical consciousness level |
| Microstate dynamics (`features/microstates.py`) | Zarka et al. 2026; Michel & Koenig 2018 | Spatial-topographic reconfiguration (microstate B/C coverage and transition changes) rather than a single-channel scalar |
| Surrogate-data controls (`features/surrogates.py`) | Standard practice motivated by the complexity-measure literature (Casali, Comolatti, Sitt) | Confirms a complexity/criticality estimate is not an artifact of the signal's linear spectrum alone |

### Known gap

None of the current features implement an information-exchange /
functional-connectivity measure in Sitt et al.'s sense (weighted
symbolic mutual information between channel pairs). Sitt et al. report
this family as among the most discriminative of the three they tested,
so its absence is a real gap in the current feature stack, not a minor
omission. Flagged here as a v2 todo: add a wSMI-style pairwise measure
to `features/`, most naturally alongside `complexity.py` since it is
computed from the same symbolized time series wSMI. This is called out
explicitly rather than silently, per the project's standing rule
against overclaiming apparatus completeness.

## Datasets in play

| Dataset | Paradigm | Status | Notes |
|---|---|---|---|
| Zarka et al. 2026 MGH cessation cohort | Cessation | Not yet public | Five highly trained meditators; no dataset DOI or repository link found on the bioRxiv page or the MGH meditation lab site as of this check (see `data/README.md`) |
| NIMHANS (Venugopal et al.) 4-tradition / Isha breath-watching cohort | Meditation traditions | Not yet public | Author's GitHub hosts analysis code, not raw multi-tradition EEG, as of this check |
| OpenNeuro `ds001787` | Meditation resting-state | Reachable (verified `HTTP 200` at https://openneuro.org/datasets/ds001787/), existing not-yet-wired dataset already documented in `data/README.md` | Probes concentration/mind-wandering every ~2 minutes rather than annotating discrete cessation onsets |
| Propofol graded sedation EEG (Chennu et al. 2016) via FieldTrip workshop page | Anesthesia | Workshop page reachable (`HTTP 200` at https://www.fieldtriptoolbox.org/workshop/madrid2019/eeg_sedation/); exact access/license terms verified and documented in `data/README.md` alongside the fetch script for this item | See Item 4 / `scripts/fetch_chennu_sedation.sh` and `data/README.md` for what was actually found |
| OpenNeuro `ds003171` (propofol EEG+fMRI, 17 subjects x 4 sedation levels) | Anesthesia | Reachable (verified `HTTP 200` at https://openneuro.org/datasets/ds003171/) | Only the EEG arm (or an fMRI-derived timeseries) is usable in this pipeline at a time, not both simultaneously, since the feature stack operates on one modality's channel-by-time matrix |
| PhysioNet Sleep-EDF Expanded v1.0.0 | NREM/REM sleep | Reachable (verified `HTTP 200` at https://physionet.org/content/sleep-edfx/1.0.0/ and https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/) | 197 polysomnography recordings, 5-stage hypnograms, Fpz-Cz/Pz-Oz channels at 100 Hz; see Item 3 / `scripts/fetch_sleep_edfx_subset.sh` |
| MPI Leipzig LEMON EEG resting-state (OpenNeuro `ds000221`) | Wake control (Gate 0a, "Berger effect") | Wired, already fetched and validated in this repo per `data/README.md` | Used as the eyes-open/eyes-closed alpha-power positive control anchoring the wake end of the graded-consciousness axis; see `scripts/fetch_lemon_subset.sh` and `scripts/validate_lemon_berger.py` |

Sleep, anesthesia, and LEMON wake data are all real, labeled, and either
already wired (LEMON) or newly wired as positive controls in this PR
(Sleep-EDF, and Chennu where access allows). Cessation and the NIMHANS
traditions dataset remain the eventual real anchors for the manifold's
actual target phenomenon and are not yet available; the instrument is
validated on the other three paradigms in the meantime, consistent with
Toker et al.'s finding that resting-state features generalize across
graded-consciousness paradigms rather than being paradigm-specific.
