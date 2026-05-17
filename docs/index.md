# helios-fusion-engine

Model-agnostic probabilistic fusion of heterogeneous space-weather model
outputs. This is the **public framework** of HELIOS Artifact C (the fusion
engine). Trained weights, BMA priors fitted on Table 3-1 events, and
equipment transfer functions live in the private
`helios-fusion-internal` companion repo and are NOT shipped with this
package.

## What the framework provides

- **Bayesian Model Averaging orchestrator** with rolling-window skill-weight
  updates and explicit handling of missing component models.
- **Reliability calibrators** — isotonic (proposal-default), Platt (rejected,
  retained for comparison), and severity-stratified isotonic (one calibrator
  per Kp severity stratum).
- **Conformal prediction wrappers** — split conformal (marginal coverage)
  and Mondrian conformal (per-stratum coverage).
- **Evaluation harness** — CCMC-compatible metrics (HSS, TSS, POD, FAR,
  Brier, CRPS) with bootstrap 95% CIs and reliability-diagram slope.
- **Typed records** for fusion lineage compatible with the upcoming
  `helios-provenance-spec`.

## What the framework does NOT provide

- No trained weights. Callers supply BMA weights at construction time, or
  fit them at runtime via `BMAOrchestrator.update_weights`.
- No equipment transfer functions. The GNSS slice (proposal Obj. 4) builds
  on top of this framework but ships separately.
- No kill-gate execution. The kill-gate runner lives in
  `helios-program/orchestration/kill_gate.py` and **consumes** this
  framework's `EvalReport`.

## Status

- v0.1.0 — public framework first release.
- See [`architecture.md`](architecture.md) for the BMA + isotonic + conformal
  composition and the rationale for that stack.
- See [`baselines.md`](baselines.md) for the "best individual component
  model" baseline definition.
- See the [`01-synthetic-bma-demo.ipynb`](https://github.com/577Industries/helios-fusion-engine/blob/main/notebooks/01-synthetic-bma-demo.ipynb)
  notebook for a runnable end-to-end demonstration on synthetic data.

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

# 1) Define BMA weights and fuse
bma = BMAOrchestrator(weights={"UMASEP": 0.4, "SEPMOD": 0.3, "SEP_Scoreboard_A": 0.3})
# fused = bma.fuse([umasep_output, sepmod_output, scoreboard_a_output])

# 2) Calibrate
cal = SeverityStratifiedCalibrator()
# cal.fit(train_fused_probs, train_truth, train_strata)
# calibrated = cal.transform(test_fused_probs, test_strata)

# 3) Conformal interval
cp = MondrianConformalRegressor()
# cp.fit(train_fused_probs, train_truth, train_strata)
# intervals = cp.predict_interval(test_fused_probs, test_strata, alpha=0.1)

# 4) Score
# report = evaluate(test_fused_probs, test_truth, test_strata)
# report.aggregate.hss.point         # HSS point estimate
# report.aggregate.hss.ci_low / ci_high  # bootstrap 95% CI
# report.aggregate.reliability_slope  # H2 quantity (target |slope - 1| <= 0.15)
```

## License

Apache 2.0 — see [LICENSE](https://github.com/577Industries/helios-fusion-engine/blob/main/LICENSE).
