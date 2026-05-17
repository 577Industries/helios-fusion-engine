"""Reliability calibrators for fused probability outputs.

Public surface:

* :class:`IsotonicCalibrator` — production calibrator (proposal-default).
* :class:`PlattCalibrator` — kept for comparison; the proposal §2 Obj. 2
  explicitly rejects Platt scaling at extremes. See the class docstring for
  the rejection rationale.
* :class:`SeverityStratifiedCalibrator` — three isotonic calibrators, one
  per Kp severity stratum.

All calibrators expose ``to_state_dict`` / ``from_state_dict`` for explicit
state round-tripping. Generic object-serialisation persistence is
intentionally *not* provided in the public API to avoid arbitrary-code
deserialisation hazards. Callers wanting on-disk persistence should
serialise the state dict via JSON, ``joblib`` (preferred for numeric arrays
with version pinning), or any other vetted serialiser.
"""

from __future__ import annotations

from helios_fusion.calibration.isotonic import IsotonicCalibrator
from helios_fusion.calibration.platt import PlattCalibrator
from helios_fusion.calibration.stratified import SeverityStratifiedCalibrator

__all__ = [
    "IsotonicCalibrator",
    "PlattCalibrator",
    "SeverityStratifiedCalibrator",
]
