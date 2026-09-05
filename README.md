# cessation_manifold

**A per-subject, calibrated "distance-from-cessation" readout on EEG, with
honesty gates that force it to abstain rather than guess.**

The long-term question: does a subject's EEG sit measurably closer to or
farther from *cessation* (the meditation-induced brief blanking of
consciousness described by Zarka et al. 2026), and does that hold across
meditation traditions? This repo is **v0 of the instrument**, not an answer
to that question. It builds the full pipeline, proves the machinery recovers
known signals from real EEG, and refuses to report a cessation number until
the anchoring data exists.

Owner: Aayush Gandhi (`aaygan29`). License: MIT (downloaded data keeps its
own license).

---

## 1. What this is, honestly

- It is a working **feature -> embedding -> distance -> conformal-interval**
  pipeline for EEG.
- It is validated two ways: on **synthetic ground truth** (a Kuramoto
  oscillator network with a driven collapse) for the apparatus, and on
  **real public EEG** by recovering canonical published signals.
- It is **not** a finding about meditation. The cessation anchor is trained
  on the synthetic collapse for now, because the Zarka (Harvard MGH) and
  NIMHANS multi-tradition datasets are not openly downloadable.
- Its defining feature is an **honesty layer**: when conformal coverage
  fails or an input falls outside the calibration range, the pipeline raises
  `UnvalidatedClaimError` instead of returning a number.

Read this as "the apparatus works and recovers known EEG effects; the
scientific claim waits on data."

---

## 2. How it works, at a glance

```
raw EEG epochs
     │
     ▼  features/     microstates · aperiodic 1/f slope · avalanche criticality · LZ/DFA complexity
 feature vectors
     │
     ▼  embed/        PCA + UMAP manifold; distance from the cessation centroid
 distance-from-cessation
     │
     ▼  honesty/      split-conformal interval · ICC reproducibility · gate() decorator
 calibrated readout, or a refusal (UnvalidatedClaimError)
```

Data enters through one adapter (`io/`), so swapping the synthetic generator
for a real BIDS dataset is a config change, not a rewrite.

---

## 3. Repo map (where to look)

| Path | What lives there |
|---|---|
| `src/cessation_manifold/io/` | Data adapters: `synthetic.py` (Kuramoto-with-collapse) and `bids_loader.py` (real EEG via MNE-BIDS). One `LoadedEpochs` contract for both. |
| `src/cessation_manifold/features/` | The feature stack: `microstates`, `aperiodic` (1/f), `criticality` (avalanches), `complexity` (LZ/DFA), `surrogates` (IAAFT). |
| `src/cessation_manifold/embed/` | `manifold.py` (PCA + UMAP) and `distance.py` (distance from the cessation centroid). |
| `src/cessation_manifold/honesty/` | `conformal.py` (split-conformal), `icc.py` (ICC(2,1) reproducibility), `gates.py` (the `gate()` decorator + `UnvalidatedClaimError`), provenance stamps. |
| `src/cessation_manifold/pipeline.py` | The whole run: config -> features -> embed -> distance -> conformal -> report. |
| `scripts/` | `fetch_*.sh` real-data downloaders, `validate_lemon_berger.py`, `validate_sleep_edfx.py`, `run_demo.py`, `run_seed_sweep.py`. |
| `configs/` | Per-dataset YAML (`synthetic.yaml`, `lemon.yaml`, `openneuro_meditation.yaml`). |
| `data/README.md` | The dataset table: real URLs, licenses, fetch status, reachability notes. Start here for data. |
| `PREREGISTRATION.md` | The kill-criteria gates and the rule that gate logic does not change once real data lands. |
| `BACKGROUND.md` | Why cessation / anesthesia / sleep are grouped as graded-consciousness contrasts. |

---

## 4. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest                                                 # 13 apparatus tests

# Real-data validations (each recovers a KNOWN signal, on real EEG):
bash scripts/fetch_lemon_subset.sh                      # ~600 MB, ~2 min
PYTHONPATH=src python scripts/validate_lemon_berger.py  # Gate 0:  Berger effect (LEMON)
PYTHONPATH=src python scripts/validate_sleep_edfx.py    # Gate 0b: Wake > N3 complexity (Sleep-EDF)

# Apparatus gates on synthetic ground truth:
PYTHONPATH=src python scripts/run_demo.py               # single seed, writes results/report.html
PYTHONPATH=src python scripts/run_seed_sweep.py         # multi-seed with bootstrap CIs
```

---

## 5. What has actually been validated

The instrument earns trust by recovering **known** signals from **real** EEG
before it is ever pointed at cessation. Two independent real-data controls
pass:

| Gate | On real EEG | Known signal recovered | Result |
|---|---|---|---|
| **0** | LEMON resting EEG (`ds000221`) | Berger effect: posterior alpha rises when eyes close, more than at frontal sites | **PASS.** Posterior Cohen's d = 1.25 pooled (per-subject 1.9, 1.8); block-permutation p < 0.0002; posterior-minus-frontal specificity 0.76 (threshold 0.3). |
| **0b** | Sleep-EDF (Sleep Cassette) | Wake EEG is more complex than N3 deep sleep (Lempel-Ziv) | **PASS.** Wake vs N3 Cohen's d = 4.49, permutation p = 0.0; aperiodic slope steepens in N3 as expected. |

These prove the feature and distance machinery works on real EEG, on the
right sensors, under nulls that respect block structure. They say nothing
about cessation yet: that is the point of separating them.

### Apparatus gates (synthetic ground truth)

Defined in `PREREGISTRATION.md`; the honest status is reported, not tuned to
pass. Notable: **Gate 1 (within-subject reproducibility)** now uses a proper
ICC(2,1) (McGraw & Wong 1996; Koo & Li 2016 thresholds) with a permutation
null, replacing an ad-hoc ratio metric that had failed every seed. **Gate 3
(surrogate specificity)** passes: IAAFT-surrogate EEG breaks the score.
**Gate 4 (conformal coverage)** holds on average (~0.93 vs nominal 0.90) but
bounces per-seed, which is itself a finding about test-fold size. **Gate 2
(meditator vs control)** cannot be scientifically informative until a real
meditator arm lands. Current numbers: `results/` and `results/report.html`.

---

## 6. Honest limits of v0

- The two anchor datasets (**Zarka** cessation, **NIMHANS** 4-tradition) are
  **not openly downloadable**; the cessation centroid is trained on the
  synthetic collapse until they arrive.
- **Gate 2** is not informative without a real meditator arm.
- OpenNeuro `ds001787` (meditation) is reachable but not yet wired in: it
  lacks discrete cessation-onset labels to anchor the centroid.
- The real-data controls use small subject counts (2 LEMON subjects for the
  Berger control). Enough for a within-subject positive control on a large
  signal; not a population claim.

---

## 7. Data

All raw EEG is fetched, never committed. See **`data/README.md`** for the
full table (dataset id, role, license, fetch command, verified reachability).
Summary: LEMON (`ds000221`) and Sleep-EDF are wired and fetch-tested;
`ds001787` is verified reachable but unlabeled for cessation; Zarka and
NIMHANS are not public.

## 8. Citations

- Berger H. (1929). Uber das Elektrenkephalogramm des Menschen.
- Babayan A. et al. (2019). LEMON dataset. Scientific Data 6:180308 (`ds000221`).
- Zarka D. et al. (2026). EEG brain reconfiguration during
  meditation-induced extended cessation of consciousness. bioRxiv.
- Venugopal R. et al. (2026). Temporal EEG signatures of meditation
  experience. Mindfulness.
- McGraw & Wong (1996); Koo & Li (2016) for the ICC(2,1) reproducibility gate.
