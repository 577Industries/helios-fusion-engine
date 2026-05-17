"""Bayesian Model Averaging orchestrator.

Public surface:

* :class:`BMAOrchestrator` — orchestrates per-event fusion across an ensemble
  of upstream model outputs with optional dynamic-weight updates.
* :func:`compute_skill_weights` — pure function mapping skill scores to
  BMA weights.

See ``docs/architecture.md`` for the rationale behind BMA + isotonic +
conformal as a composed stack.
"""

from __future__ import annotations

from helios_fusion.bma.orchestrator import BMAOrchestrator
from helios_fusion.bma.weights import (
    compute_skill_weights,
    renormalize_weights,
)

__all__ = [
    "BMAOrchestrator",
    "compute_skill_weights",
    "renormalize_weights",
]
