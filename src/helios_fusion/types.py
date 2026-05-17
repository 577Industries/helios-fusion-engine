"""Typed records used across the fusion engine.

These records mirror the eventual ``helios-provenance-spec`` v0.1 contract.
Once that spec ships, the ``schema_version`` field will be bumped and the
``raw_metadata`` block aligned with the SPASE/PROV-JSON crosswalk.

All records are :class:`pydantic.BaseModel` subclasses so input data coming
from connectors (or from user code) is validated at construction time.
Internal hot loops convert to ``numpy.ndarray`` via helper functions in the
calling subpackages.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Schema version is bumped whenever the public-API shape of these records
#: changes. Once ``helios-provenance-spec`` v0.1 ships, this constant will be
#: re-anchored to the spec's version identifier.
SCHEMA_VERSION: str = "helios-fusion-engine/types/0.1.0"

SeverityStratum = Literal["quiet", "moderate", "extreme"]
"""Kp-binned severity stratum.

* ``quiet``    -- Kp 0 to 3
* ``moderate`` -- Kp 4 to 6
* ``extreme``  -- Kp 7 to 9

These bin edges are pre-registered in the OSF template and are NOT a tuning
parameter. See :mod:`helios_fusion.stratification` for the assignment
function.
"""

TransformationType = Literal[
    "ingest",
    "bma_fuse",
    "isotonic_calibrate",
    "platt_calibrate",
    "severity_stratified_calibrate",
    "split_conformal",
    "mondrian_conformal",
    "passthrough",
]
"""Transformation kinds that may appear in a lineage chain.

This list is the union of every transformation the public framework can emit.
Downstream consumers (e.g. the provenance spec) should treat unknown values as
forward-compatible extensions and not as errors.
"""


class LineageStep(BaseModel):
    """One transformation applied during the fusion pipeline.

    A :class:`FusedOutput` records the full ordered list of steps that
    produced it. Each step names the transformation that was applied, the IDs
    of the records it consumed, the IDs it produced, an optional weight (only
    meaningful for ``bma_fuse``), and the parameters that uniquely identify
    the transformation's configuration at the time of emission.

    Attributes:
        schema_version: The :data:`SCHEMA_VERSION` constant at emission time.
        transformation_id: A stable unique identifier for this step.
        transformation_type: One of :data:`TransformationType`.
        input_ids: IDs of the records consumed.
        output_ids: IDs of the records produced.
        weight: For BMA steps, the weight assigned to the contributing input.
            ``None`` for non-weighting transformations.
        parameters: Arbitrary JSON-serialisable configuration for the step.
            Implementations SHOULD avoid storing large blobs here; instead
            store a content-addressed hash and resolve via an external store.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    transformation_id: str
    transformation_type: TransformationType
    input_ids: list[str] = Field(default_factory=list)
    output_ids: list[str] = Field(default_factory=list)
    weight: float | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelOutput(BaseModel):
    """A single upstream-model output at a single timestamp.

    Five or so :class:`ModelOutput` instances — one per component model in the
    BMA ensemble — feed into :meth:`helios_fusion.bma.BMAOrchestrator.fuse` to
    produce one :class:`FusedOutput`.

    Attributes:
        schema_version: The :data:`SCHEMA_VERSION` constant at emission time.
        id: Stable unique identifier for the record.
        model_id: Identifier of the upstream model (e.g. ``"UMASEP"``,
            ``"SEP_Scoreboard_A"``).
        timestamp: Time the prediction is valid for.
        value: Numeric prediction. For probability outputs, must be in
            ``[0, 1]``; for continuous quantities (TEC, proton flux),
            unconstrained.
        value_units: SI or operational units of ``value`` (e.g.
            ``"probability"``, ``"pfu_>10MeV"``, ``"TECU"``).
        confidence_interval: Optional ``(lower, upper)`` interval reported by
            the upstream model itself.
        severity_stratum: Optional Kp severity stratum at the prediction
            window. Populated by connectors (B), not by this framework.
        raw_metadata: Free-form upstream metadata — connector keeps native
            fields here for provenance drill-down.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    id: str
    model_id: str
    timestamp: datetime
    value: float
    value_units: str
    confidence_interval: tuple[float, float] | None = None
    severity_stratum: SeverityStratum | None = None
    raw_metadata: dict[str, Any] | None = None

    @field_validator("confidence_interval")
    @classmethod
    def _check_interval(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is None:
            return v
        lo, hi = v
        if lo > hi:
            raise ValueError(f"confidence_interval lower bound {lo} > upper bound {hi}")
        return v


class FusedOutput(BaseModel):
    """A fused prediction produced by the engine, with full lineage.

    The conformal interval is a 3-tuple ``(lower, upper, alpha)`` rather than
    the 2-tuple used in :class:`ModelOutput` so the requested coverage level
    travels with the prediction. Downstream consumers must NOT assume
    symmetric intervals; the calibration step can produce asymmetric residual
    quantiles, especially in Mondrian conformal across severity strata.

    Attributes:
        schema_version: The :data:`SCHEMA_VERSION` constant at emission time.
        id: Stable unique identifier for the record.
        prediction_target: Human-readable name of the predicted quantity
            (e.g. ``"all_clear_revocation_probability"``).
        timestamp: Time the prediction is valid for.
        value: The fused point estimate.
        value_units: Units of ``value`` (mirrors :class:`ModelOutput`).
        conformal_interval: Optional ``(lower, upper, alpha)`` conformal
            interval at coverage ``1 - alpha``.
        lineage: Ordered list of transformation steps that produced the
            output. Always non-empty; the first step's ``transformation_type``
            is typically ``"bma_fuse"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    id: str
    prediction_target: str
    timestamp: datetime
    value: float
    value_units: str
    conformal_interval: tuple[float, float, float] | None = None
    lineage: list[LineageStep] = Field(default_factory=list)

    @field_validator("conformal_interval")
    @classmethod
    def _check_conformal(
        cls, v: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if v is None:
            return v
        lo, hi, alpha = v
        if lo > hi:
            raise ValueError(f"conformal_interval lower bound {lo} > upper bound {hi}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"conformal_interval alpha must be in (0, 1); got {alpha}")
        return v
