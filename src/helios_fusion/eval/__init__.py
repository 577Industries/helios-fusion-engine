"""Evaluation harness and CCMC-compatible metrics.

Public surface:

* :func:`hss`, :func:`tss`, :func:`pod`, :func:`far` — point-estimate +
  bootstrap-CI contingency-table metrics.
* :func:`brier_score`, :func:`crps` — proper scoring rules.
* :func:`reliability_diagram` — bin centres, observed frequencies, and
  fitted slope (the H2 quantity in the OSF pre-registration).
* :class:`EvalReport`, :func:`evaluate` — high-level harness producing
  per-stratum AND aggregate metrics.
* :func:`best_individual_component_baseline` — the comparator used by
  the kill-gate (proposal §2 Obj. 3 success criterion).

Metric definitions are mirrored verbatim from the OSF pre-registration
template (``helios-program/orchestration/osf_preregistration.template.md``).
The kill-gate depends on this verbatim correspondence; changing a metric
definition in this module without amending the OSF template would
invalidate the pre-registration discipline.
"""

from __future__ import annotations

from helios_fusion.eval.baseline import best_individual_component_baseline
from helios_fusion.eval.harness import EvalReport, evaluate
from helios_fusion.eval.metrics import (
    brier_score,
    crps,
    far,
    hss,
    pod,
    reliability_diagram,
    tss,
)

__all__ = [
    "EvalReport",
    "best_individual_component_baseline",
    "brier_score",
    "crps",
    "evaluate",
    "far",
    "hss",
    "pod",
    "reliability_diagram",
    "tss",
]
