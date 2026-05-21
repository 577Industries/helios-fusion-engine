# helios-fusion-engine

[![CI](https://github.com/577Industries/helios-fusion-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/577Industries/helios-fusion-engine/actions/workflows/ci.yml) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/helios-fusion-engine.svg)](https://pypi.org/project/helios-fusion-engine/)

> Model-agnostic probabilistic fusion of heterogeneous space-weather model
> outputs: Bayesian Model Averaging orchestrator, isotonic-regression
> reliability calibrator, split + Mondrian conformal prediction wrappers,
> and severity-stratified validation harness with CCMC-compatible metrics
> (HSS, TSS, POD, FAR, Brier, CRPS).

## What this is

This is the **public framework** of HELIOS Artifact C — the fusion engine.
It is a thin, well-typed, well-tested library that callers compose into
their own fusion pipelines.

Specifically, the package provides:

- A `BMAOrchestrator` that combines upstream component model outputs with
  optional rolling-window skill-based weight updates and explicit
  missing-model exclusion.
- Three reliability calibrators (`IsotonicCalibrator`, `PlattCalibrator`,
  `SeverityStratifiedCalibrator`). The proposal-default is the
  severity-stratified isotonic; Platt is shipped *because the proposal
  rejects it*, so the rejection rationale is reproducible.
- Two conformal-prediction wrappers (`SplitConformalRegressor` for marginal
  coverage; `MondrianConformalRegressor` for per-stratum coverage).
- A CCMC-compatible evaluation harness producing per-stratum AND aggregate
  HSS, TSS, POD, FAR, Brier, CRPS, and reliability-slope, all with
  bootstrap 95% CIs.
- Typed records (`ModelOutput`, `FusedOutput`, `LineageStep`) with a
  `schema_version` field on every record, forward-compatible with the
  upcoming `helios-provenance-spec` v0.1.

## What this is NOT

- **No trained weights ship with this package.** Callers (including the
  HELIOS kill-gate runner) supply BMA weights and calibration parameters
  at runtime. Trained weights, BMA priors fitted on Table 3-1 events, and
  equipment transfer functions live in the private companion repo
  `helios-fusion-internal` and are NOT distributed here.
- **No kill-gate execution.** The kill-gate runner lives in
  `helios-program/orchestration/kill_gate.py` and *consumes* this
  framework's `EvalReport`. The kill-gate itself is out of scope for this
  repo.
- **No retrospective paper figures.** The arXiv preprint (if the kill-gate
  passes) is generated from the private repo + `helios-fusion-engine`
  composed; this repo ships only the framework and the synthetic-data
  demonstration notebook.

## Status

This repository is part of the **HELIOS** program — a NASA SBIR Phase I
effort by 577 Industries Inc. supporting subtopic SPWX.1.S26A (Advanced
Data-Driven Applications for Space Weather R2O2R). See proposal §B Obj. 2 +
§B.5 (pre-registered validation) + §B.6 innovation #1 of the proposal.

**v0.2.0 — BMA orchestration with isotonic / Platt / severity-stratified
calibrators, split + Mondrian conformal regressors, and CCMC-compatible
metrics (May 18, 2026).** The framework is feature-complete for the kill-
gate path; HELIOS Phase I work delivers trained priors and hold-out
evaluation in the private companion repo (`helios-fusion-internal`) using
this framework as the load-bearing library.

## Quickstart

```bash
pip install helios-fusion-engine
```

```python
import numpy as np
from helios_fusion.bma import BMAOrchestrator
from helios_fusion.calibration import SeverityStratifiedCalibrator
from helios_fusion.conformal import MondrianConformalRegressor
from helios_fusion.eval import evaluate

bma = BMAOrchestrator(weights={"UMASEP": 0.4, "SEPMOD": 0.3, "SEP_Scoreboard_A": 0.3})
# fused = bma.fuse([umasep_output, sepmod_output, scoreboard_a_output])

cal = SeverityStratifiedCalibrator()
# cal.fit(train_probs, train_truth, train_strata)
# calibrated = cal.transform(test_probs, test_strata)

cp = MondrianConformalRegressor()
# cp.fit(train_probs, train_truth, train_strata)
# intervals = cp.predict_interval(test_probs, test_strata, alpha=0.1)

# report = evaluate(test_probs, test_truth, test_strata, n_bootstrap=1000)
# report.aggregate.hss.point         # HSS point estimate
# report.aggregate.reliability_slope # H2 quantity (target |slope - 1| <= 0.15)
```

See the [synthetic-data demo notebook](notebooks/01-synthetic-bma-demo.ipynb)
for an end-to-end runnable pipeline.

## Documentation

- **Master plan**: see [`helios-program`](https://github.com/577Industries/helios-program) (public; reviewer entry point)
- **Architecture**: [`docs/architecture.md`](docs/architecture.md) explains
  why BMA + isotonic + conformal compose as they do.
- **Baselines**: [`docs/baselines.md`](docs/baselines.md) defines the
  "best individual component model" kill-gate comparator.
- **API reference**: [`docs/api/`](docs/api/) (auto-generated via
  mkdocstrings from Google-style docstrings).
- **Provenance**: every output traces to its upstream model and
  transformation chain via the schema fields on `ModelOutput` and
  `FusedOutput`. Once [`helios-provenance-spec`](https://github.com/577Industries/helios-provenance-spec)
  v0.1 ships, the `schema_version` constant on every record will re-anchor
  to that contract.

## Development

```bash
pip install -e '.[dev]'
ruff check .
ruff format --check .
mypy
pytest --cov
```

CI runs the same on every PR. Coverage gate is 85%; `main` currently sits
at 92%.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Substantive changes should be
discussed in an issue first.

## Citation

```bibtex
@software{helios_helios_fusion_engine,
  author       = {Waweru, Thomas and 577 Industries Inc.},
  title        = { helios-fusion-engine: Model-agnostic probabilistic fusion of heterogeneous space-weather model outputs: Bayesian Model Averaging orchestrator, isotonic-regression reliability calibrator, split + Mondrian conformal prediction wrappers, and severity-stratified validation harness with CCMC-compatible metrics (HSS, TSS, POD, FAR, Brier, CRPS) },
  year         = {2026},
  publisher    = {577 Industries Inc.},
  url          = {https://github.com/577Industries/helios-fusion-engine},
}
```

## Related

- **HELIOS program**: [`helios-program`](https://github.com/577Industries/helios-program) — master plan, proposal companion document, orchestration scripts.
- **Wave 1 review pack**: [Artifact C framework review pack](https://github.com/577Industries/helios-program/blob/main/specs/2026-05-17-C-fusion-engine-framework-review-pack.md) — synthetic-data demo numbers, design decisions, deviations from proposal §2 Obj. 2 (conformal from-scratch, Platt shipped for reproducible-rejection).
- **OSF pre-registration template**: [`orchestration/osf_preregistration.template.md`](https://github.com/577Industries/helios-program/blob/main/orchestration/osf_preregistration.template.md) — **must** be filed publicly **before** hold-out evaluation runs. The kill-gate runner (`orchestration/kill_gate.py`) refuses to execute without an OSF URL on file at `orchestration/osf_preregistration.url`. Decision rules: PASS both H1+H2 → full arXiv paper; PASS one → ablation paper; FAIL both → no paper.
- **Provenance schema**: [`helios-provenance-spec`](https://github.com/577Industries/helios-provenance-spec) — `ModelOutput`, `FusedOutput`, `LineageStep` types align with this schema.
- **Trained weights / BMA priors** (private companion): `helios-fusion-internal` — fitted on Table 3-1 training events; not redistributed.
