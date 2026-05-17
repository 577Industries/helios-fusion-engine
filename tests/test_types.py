"""Typed-record validation tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from helios_fusion.types import (
    SCHEMA_VERSION,
    FusedOutput,
    LineageStep,
    ModelOutput,
)


def test_model_output_schema_version_is_set() -> None:
    rec = ModelOutput(
        id="a",
        model_id="UMASEP",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        value=0.5,
        value_units="probability",
    )
    assert rec.schema_version == SCHEMA_VERSION


def test_model_output_rejects_inverted_ci() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        ModelOutput(
            id="a",
            model_id="A",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            value=0.5,
            value_units="probability",
            confidence_interval=(0.9, 0.1),
        )


def test_model_output_is_frozen() -> None:
    rec = ModelOutput(
        id="a",
        model_id="A",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        value=0.5,
        value_units="probability",
    )
    with pytest.raises(Exception):
        rec.value = 0.99  # type: ignore[misc]


def test_fused_output_schema_version_is_set() -> None:
    rec = FusedOutput(
        id="f",
        prediction_target="event",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        value=0.7,
        value_units="probability",
    )
    assert rec.schema_version == SCHEMA_VERSION


def test_fused_output_rejects_inverted_conformal_interval() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        FusedOutput(
            id="f",
            prediction_target="event",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            value=0.5,
            value_units="probability",
            conformal_interval=(0.9, 0.1, 0.1),
        )


def test_fused_output_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        FusedOutput(
            id="f",
            prediction_target="event",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            value=0.5,
            value_units="probability",
            conformal_interval=(0.1, 0.9, 1.5),
        )


def test_lineage_step_schema_version_set() -> None:
    step = LineageStep(
        transformation_id="t1",
        transformation_type="bma_fuse",
    )
    assert step.schema_version == SCHEMA_VERSION


def test_lineage_step_extra_forbidden() -> None:
    """Extra fields are rejected; lineage shape is strict."""
    with pytest.raises(Exception):
        LineageStep(
            transformation_id="t1",
            transformation_type="bma_fuse",
            bogus="value",  # type: ignore[call-arg]
        )
