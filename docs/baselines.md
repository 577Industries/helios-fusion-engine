# Baselines

The kill-gate comparator is the **best individual component model on the
hold-out set**. The proposal §2 Obj. 3 success criterion and the OSF
pre-registration §6 define the kill-gate as:

> Fused all-clear-revocation HSS on the 3-event hold-out beats the
> best-component-model HSS by ≥ 15%.

This page documents how the comparator is computed inside the framework.

## Definition

Given a set of upstream component model outputs over the held-out events
(one `ModelOutput` per model per event) and a binary observed outcome per
event:

1. Group component outputs by `model_id`.
2. For each model:
   - Sort its outputs by event timestamp.
   - Binarise its probability predictions at `binary_threshold` (default
     0.5, matching the OSF pre-registration).
   - Align with the observed outcomes by event index.
   - Compute HSS and a bootstrap 95% CI (1000 resamples by default).
3. Return the model with the highest HSS point estimate.

The comparator function is:

```python
from helios_fusion.eval import best_individual_component_baseline

best_id, hss_point, (ci_lo, ci_hi) = best_individual_component_baseline(
    component_outputs=...,        # all per-event per-model outputs
    observed_binary=...,          # binary truth per event
    binary_threshold=0.5,
    n_bootstrap=1000,
)
```

The function is referenced by the kill-gate runner in
`helios-program/orchestration/kill_gate.py` (currently a placeholder that
raises). When the kill-gate runs, the fused HSS (from
`evaluate(...).aggregate.hss.point`) is compared against
`hss_point` via the locked 15% relative-improvement rule:

```
fused_hss > best_individual_hss * 1.15
```

## Why this comparator and not equal-weight ensemble?

The proposal compares against the **best single component**, not an
equal-weight ensemble, because:

1. An operator deciding whether to switch from "the best model we already
   trust" to HELIOS needs to know HELIOS beats that single model, not a
   composite no one currently uses.
2. The CCMC validation framework (Whitman et al. 2023, 2024) defines
   model-level skill scores comparable across vendors — the framework
   community knows how to interpret a single-model HSS but not an
   equal-weight composite.

## Pre-registration discipline

The comparator definition is locked at pre-registration time. After OSF
filing:

- `binary_threshold` is not tunable.
- `n_bootstrap` is not tunable below 1000 (more is fine).
- The bootstrap seed is reproducible (see `_DEFAULT_BOOTSTRAP_SEED` in
  `helios_fusion/eval/metrics.py`).

Any future change to the comparator MUST go through a pre-registration
amendment on OSF before any hold-out re-evaluation.
