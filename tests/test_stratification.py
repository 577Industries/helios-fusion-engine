"""Severity stratification utility tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from helios_fusion.stratification import (
    assign_severity_stratum,
    stratify_by_severity,
)
from helios_fusion.types import ModelOutput


@pytest.mark.parametrize(
    ("kp", "expected"),
    [
        (0.0, "quiet"),
        (1.5, "quiet"),
        (3.0, "quiet"),  # boundary belongs to lower stratum
        (3.33, "moderate"),
        (4.0, "moderate"),
        (6.0, "moderate"),
        (6.67, "extreme"),
        (7.0, "extreme"),
        (9.0, "extreme"),
    ],
)
def test_assign_severity_boundaries(kp: float, expected: str) -> None:
    assert assign_severity_stratum(kp) == expected


def test_assign_severity_out_of_range_raises() -> None:
    with pytest.raises(ValueError):
        assign_severity_stratum(-0.1)
    with pytest.raises(ValueError):
        assign_severity_stratum(9.1)


def test_assign_severity_nan_raises() -> None:
    with pytest.raises(ValueError):
        assign_severity_stratum(float("nan"))


def test_stratify_groups_correctly() -> None:
    t = datetime(2024, 1, 1, tzinfo=UTC)
    records = [
        ModelOutput(
            id=f"r{i}",
            model_id="A",
            timestamp=t,
            value=0.5,
            value_units="probability",
            severity_stratum=s,  # type: ignore[arg-type]
        )
        for i, s in enumerate(["quiet", "moderate", "extreme", "quiet"])
    ]
    grouped = stratify_by_severity(records)
    assert {s: len(v) for s, v in grouped.items()} == {
        "quiet": 2,
        "moderate": 1,
        "extreme": 1,
    }


def test_stratify_drops_records_without_stratum(caplog) -> None:
    t = datetime(2024, 1, 1, tzinfo=UTC)
    records = [
        ModelOutput(id="a", model_id="A", timestamp=t, value=0.5, value_units="probability"),
        ModelOutput(
            id="b",
            model_id="A",
            timestamp=t,
            value=0.5,
            value_units="probability",
            severity_stratum="quiet",
        ),
    ]
    import logging

    with caplog.at_level(logging.WARNING):
        grouped = stratify_by_severity(records)
    assert "dropped 1" in caplog.text
    assert len(grouped["quiet"]) == 1


def test_stratify_returns_all_strata_keys_even_when_empty() -> None:
    grouped = stratify_by_severity([])
    assert set(grouped.keys()) == {"quiet", "moderate", "extreme"}
    assert all(len(v) == 0 for v in grouped.values())
