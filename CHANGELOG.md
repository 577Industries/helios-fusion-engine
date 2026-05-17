# Changelog

All notable changes to this project are documented here, following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-05-17

### Added
- `BMAOrchestrator` with skill-weighted updates and missing-model renormalization.
- Calibrators: `IsotonicCalibrator`, `PlattCalibrator` (shipped for reproducible rejection rationale; tests assert isotonic beats Platt on tail-miscalibrated synthetic data), `SeverityStratifiedCalibrator`.
- Conformal regressors: `SplitConformalRegressor`, `MondrianConformalRegressor` with per-stratum coverage validation.
- CCMC-compatible metrics suite (HSS Donaldson 1975, TSS, POD, FAR, Brier, CRPS) with bootstrapped 95% CIs.
- `EvalReport` shape pinned to the OSF pre-registration template — quiet/moderate/extreme strata always present, NaN-stub records for empty strata.
- Best-component baseline (the kill-gate comparator).
- Explicit `to_state_dict` / `from_state_dict` on calibrators and conformal regressors (no pickle; JSON-friendly state for safe transport via helios-fusion-internal).
- 105 tests at 92% line+branch coverage. Hypothesis property tests for invariants (BMA weights sum to 1; isotonic monotone; HSS bounded in [-1, 1]).
- Synthetic-data demo notebook: isotonic brings reliability slope 1.0071 → 1.0772 (within ±0.15 target); Mondrian conformal achieves 91% aggregate coverage @ α=0.1.

**Public framework only.** Trained weights, BMA priors fitted on Table 3-1 events, and equipment transfer functions live in the private companion `helios-fusion-internal`.

See [GitHub releases](https://github.com/577Industries/helios-fusion-engine/releases/tag/v0.1.0) for the canonical release notes.
