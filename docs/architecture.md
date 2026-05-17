# Architecture: BMA + isotonic + conformal

The framework is a deliberately small pipeline of three composed stages:

```
component model outputs            (helios-spaceweather-connectors)
        │
        ▼
   ┌─────────────┐
   │  BMA fuse   │  weights from rolling 90-day skill (HSS-weighted)
   └─────────────┘
        │ fused point estimate (probability or continuous)
        ▼
   ┌──────────────────────────────┐
   │  Reliability calibration     │  severity-stratified isotonic regression
   └──────────────────────────────┘
        │ calibrated probability
        ▼
   ┌──────────────────────────────┐
   │  Conformal interval          │  Mondrian (per-Kp-stratum) split conformal
   └──────────────────────────────┘
        │
        ▼
   FusedOutput with lineage + conformal_interval
```

Each stage is independent: the BMA orchestrator does not know about
calibration; calibrators do not know about conformal prediction; the
conformal regressor does not know about the upstream estimator. This
separation is intentional — it makes each stage testable in isolation and
lets the same framework score deterministic point predictions (skipping
conformal), pure probability forecasts (skipping conformal width), or fully
probabilistic continuous-quantity forecasts (the full stack).

## Why this stack, vs. alternatives

### Why BMA over a single ensemble model?

Component models in heliophysics have heterogeneous error structures across
event types (CME-driven vs. flare-driven SEP onsets), severity regimes
(quiet vs. extreme Kp), and time horizons. A single ML ensemble would
require retraining whenever a component model updates its own underlying
physics. BMA treats components as **black-box probability streams** and
recombines them at runtime, so the framework continues to work when an
upstream model is replaced or temporarily unavailable.

### Why isotonic over Platt?

The proposal §2 Obj. 2 explicitly rejects Platt scaling. The rationale is
implemented in `tests/test_calibration.py::test_platt_worsens_calibration_at_extremes`:
on a synthetic stream that is well-calibrated at moderate probabilities
but miscalibrated at extremes (the regime where SEP all-clear-revocation
decisions live), Platt's two-parameter sigmoid cannot fit the localised
tail miscalibration without distorting the moderate-probability middle.
Isotonic regression is non-parametric in the relevant sense (monotone
non-decreasing) and handles this regime gracefully.

The `PlattCalibrator` class is kept in the framework deliberately, so users
can verify the rejection rationale on their own data.

### Why severity-stratified isotonic?

Operational decisions at extreme-Kp conditions are exactly the decisions
that matter most. An unstratified isotonic calibrator pools across the Kp
distribution, where quiet samples dominate by volume — the extreme stratum
contributes too few samples to the global fit to influence its knots. The
proposal §2 Obj. 2 explicitly calls for severity-stratified calibration to
"prevent calibration collapse on the events that matter most."

`SeverityStratifiedCalibrator` holds one `IsotonicCalibrator` per stratum.
Each stratum's calibration knots are fitted only on samples from that
stratum.

### Why Mondrian conformal over standard split conformal?

Standard split conformal achieves marginal coverage — the requested
`1 - alpha` coverage rate holds on average across all samples. It does
NOT guarantee per-stratum coverage. The OSF pre-registration requires
the reliability slope to fall within 0.15 of 1.0 *per stratum*; the
matching conformal-coverage discipline is Mondrian conformal (Vovk 2003),
which carries a separate residual quantile per stratum.

`MondrianConformalRegressor` is built on top of `SplitConformalRegressor`
and routes per-sample stratum labels to the matching sub-regressor at both
fit time and predict time.

## Composition rules

Three rules govern how to compose the stages safely:

1. **Disjoint data per stage**: the BMA orchestrator's verification window,
   the calibrator's fit set, and the conformal regressor's calibration set
   must be **disjoint** (or, at least, the conformal calibration set must be
   disjoint from the data used to fit the calibrator). Otherwise the
   conformal coverage guarantee is invalidated.
2. **Order matters**: BMA → calibrate → conform. Calibrating BEFORE fusing
   would calibrate each component independently and lose the cross-model
   weighting; conforming BEFORE calibrating would produce intervals on a
   miscalibrated point estimate.
3. **Stratum labels travel**: the severity stratum label is carried on
   every record so the right calibrator and the right conformal regressor
   are selected per sample.

## Lineage

Every fused output records a `LineageStep` per transformation. The
`bma_fuse` step records the weights used (post-renormalisation if any
configured models were missing) and the list of excluded models. The
calibration and conformal steps can be appended to the lineage by callers
that compose the stages downstream of the orchestrator.

The schema version on every record matches the eventual
`helios-provenance-spec` v0.1 contract — see
[`types.py`](https://github.com/577Industries/helios-fusion-engine/blob/main/src/helios_fusion/types.py)
for the verbatim field shapes.
