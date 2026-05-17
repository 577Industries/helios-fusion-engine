"""helios-fusion-engine: model-agnostic probabilistic fusion framework.

Public framework layer of HELIOS Artifact C. Provides:

* ``helios_fusion.bma`` — Bayesian Model Averaging orchestrator
* ``helios_fusion.calibration`` — isotonic / Platt / severity-stratified calibrators
* ``helios_fusion.conformal`` — split and Mondrian conformal regressors
* ``helios_fusion.stratification`` — Kp severity binning utilities
* ``helios_fusion.eval`` — CCMC-compatible metrics and evaluation harness
* ``helios_fusion.types`` — typed records (ModelOutput, FusedOutput, LineageStep)

This package ships *no* trained weights and *no* equipment transfer functions.
Callers (including the HELIOS kill-gate runner) supply weights and calibration
parameters at runtime.

See README.md for the full scope statement. See the OSF pre-registration
template at ``helios-program/orchestration/osf_preregistration.template.md``
for the metric definitions this framework implements verbatim.
"""

from __future__ import annotations

from helios_fusion.types import (
    SCHEMA_VERSION,
    FusedOutput,
    LineageStep,
    ModelOutput,
)

__version__ = "0.1.2"

__all__ = [
    "SCHEMA_VERSION",
    "FusedOutput",
    "LineageStep",
    "ModelOutput",
    "__version__",
]
