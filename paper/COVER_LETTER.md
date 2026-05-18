# arXiv Submission Cover Letter — HELIOS Fusion Engine Preprint

**Title**: Calibrated, Provenance-Tracked Fusion of Solar Energetic Particle
Forecasts for NASA Mission Operations

**Corresponding author**: Thomas Waweru, 577 Industries Inc., Columbus, OH —
`engineering@577industries.com`

**Target primary category**: `astro-ph.SR` (Solar and Stellar Astrophysics)
**Cross-list**: `cs.LG` (Machine Learning)

**License**: arXiv perpetual non-exclusive on the PDF; Apache 2.0 on the
LaTeX source committed at the repository URL below.

---

## What is being submitted

A 12-page (target) preprint reporting the design, pre-registration, and
hold-out evaluation of HELIOS — a model-agnostic fusion layer over CCMC
SEP Scoreboard A/B/C and NOAA SWPC SEP outputs combining Bayesian Model
Averaging (Hoeting et al.\ 1999) with isotonic-regression reliability
calibration (Niculescu-Mizil & Caruana 2005), split and Mondrian conformal
prediction (Vovk 2022), and CCMC-compatible verification metrics
(HSS Donaldson 1975; reliability slope; Brier; CRPS).

The pre-registered hold-out evaluation runs on three locked events —
2022-01-20 (M5.5 cycle-25 onset), 2023-02-17 (X2.2 mid-cycle), and
2024-05-11 (Gannon G5) — with binding H1 (HSS) and H2
(reliability-slope) thresholds filed publicly on OSF before any
hold-out evaluation. The OSF pre-registration URL is at
`<TO_BE_FILLED at submission — see helios-program/orchestration/osf_preregistration.url>`.

The fusion-engine release tagged at the locked commit is
`helios-fusion-engine@<TO_BE_FILLED with commit SHA at submission>`
(release tag `prereg-v1.0`).

## Why arXiv

1. Heliophysics community on `astro-ph.SR` is the primary audience for the
   mission-operations slice.
2. The calibration + conformal methodology is of independent interest to
   the ML community; cross-listing on `cs.LG` reaches that audience.
3. Pre-print exposure during the NASA SBIR Phase II decision window
   provides an external, dated record of the methodology and the
   pre-registered hold-out evaluation.

## Endorsement

This is 577 Industries' first arXiv submission to `astro-ph.SR`.
Per arXiv submission policy an endorser is required. Named endorser:
`<TO_BE_FILLED — Space-Weather / Ionospheric SME consultant, see proposal
§5.1>`; endorser email `<TO_BE_FILLED>`. Endorser status confirmed by
`gh issue` on the submission-tracking repo prior to upload.

## Reproducibility commitments

- All code released under Apache 2.0 at
  <https://github.com/577Industries/helios-fusion-engine>
  (and four sibling repositories under
  <https://github.com/577Industries>).
- The OSF pre-registration at
  `<TO_BE_FILLED with the OSF URL>` binds the methodology;
  any deviation between the pre-registration and the submitted manuscript
  is documented in the OSF *Deviations* section, not retro-fit.
- The kill-gate evaluation harness refuses to execute without an OSF URL
  on file; see `helios-fusion-engine/orchestration/kill_gate.py`.
- Every figure in the paper has a generation script committed in
  `paper/figures/build_*.py`; the figure PNGs and the cache CSV for
  Figure~2 (Gannon Bz timeline) are reproducible from a live
  `DscovrAdapter.fetch_mag` call against the NOAA / NASA DSCOVR archive.
- Trained BMA priors, isotonic calibrators, and conformal-residual
  manifests live in the private companion repo
  `helios-fusion-internal`. Access for verification on request.

## Conflicts of interest and funding disclosure

The work is conducted by 577 Industries Inc.; the submitted NASA SBIR
Phase I proposal (subtopic SPWX.1.S26A) is awaiting decision. Funding
for the preprint-stage work is from 577 Industries internal R&D.
Authors have no other financial interests to disclose.

## Contact

`engineering@577industries.com` — Thomas Waweru, PI
`<https://577industries.github.io/helios-program/>` — program landing page

---

*Cover-letter file:* `paper/COVER_LETTER.md` (drafted 2026-05-18; finalize
the `<TO_BE_FILLED>` fields at submission time per
`paper/SUBMISSION_CHECKLIST.md`).
