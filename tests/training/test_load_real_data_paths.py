"""Cover the real-data code paths in :mod:`load_table_3_1` via monkeypatching.

These tests don't hit the network -- they inject fake adapter outputs to
drive the resample / forward-fill / gap-recording branches.

Sprint C-Training-v2 update: the adapter helper now returns
``(timestamp, component_id, probability)`` tuples where ``component_id``
matches the v2 ``<name>/<variants>/<energy>`` registry identity.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

loader_module = importlib.import_module("helios_fusion.training.load_table_3_1")
TRAINING_EVENTS = loader_module.TRAINING_EVENTS
_resample_to_grid = loader_module._resample_to_grid
_resample_truth_to_grid = loader_module._resample_truth_to_grid
load_table_3_1 = loader_module.load_table_3_1


@pytest.fixture
def event() -> object:
    return TRAINING_EVENTS[6]  # September 2017 dual event


def test_resample_to_grid_empty_dataframe() -> None:
    """Empty inputs return zeros (one per grid step)."""
    grid = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(5)]
    out = _resample_to_grid(pd.DataFrame(), grid)
    assert out.shape == (5,)
    assert (out == 0.0).all()


def test_resample_to_grid_forward_fill() -> None:
    """Sparse points are forward-filled across the grid."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    df = pd.DataFrame(
        [
            {"timestamp": base, "probability": 0.2},
            {"timestamp": base + timedelta(hours=3), "probability": 0.7},
        ]
    )
    grid = [base + timedelta(hours=i) for i in range(6)]
    out = _resample_to_grid(df, grid)
    np.testing.assert_allclose(out, [0.2, 0.2, 0.2, 0.7, 0.7, 0.7])


def test_resample_truth_to_grid_zeros_on_empty_input() -> None:
    """Truth resample returns zeros when the truth frame is empty."""
    grid = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(5)]
    out = _resample_truth_to_grid(pd.DataFrame(), grid)
    assert out.shape == (5,)
    assert (out == 0.0).all()


def test_resample_truth_to_grid_forward_fill() -> None:
    """Truth resample forward-fills the observed column across the grid."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    truth = pd.DataFrame(
        [
            {"timestamp": base, "observed": 0},
            {"timestamp": base + timedelta(hours=2), "observed": 1},
            {"timestamp": base + timedelta(hours=4), "observed": 0},
        ]
    )
    grid = [base + timedelta(hours=i) for i in range(5)]
    out = _resample_truth_to_grid(truth, grid)
    np.testing.assert_allclose(out, [0.0, 0.0, 1.0, 1.0, 0.0])


def test_real_kp_pull_used_when_returned(monkeypatch: pytest.MonkeyPatch, event: object) -> None:
    """If the async Kp helper returns data, real-fill is used (no synthetic)."""
    base = event.window_start  # type: ignore[attr-defined]

    async def fake_pull(_event: object) -> dict[datetime, float]:
        return {
            base: 2.0,
            base + timedelta(days=1): 5.0,
        }

    monkeypatch.setattr(loader_module, "_async_pull_kp", fake_pull)

    async def empty_sb(_event: object) -> list[tuple[datetime, str, float]]:
        return []

    monkeypatch.setattr(loader_module, "_async_pull_scoreboards", empty_sb)

    frame = load_table_3_1(event, use_real_data=True, cadence_hours=12.0)
    # SWPC gap should NOT be recorded because real Kp came back
    assert "swpc_kp" not in frame.data_gaps


def test_real_scoreboards_pull_used_when_returned(
    monkeypatch: pytest.MonkeyPatch, event: object
) -> None:
    """If the async Scoreboards helper returns data, real ISWA records are used.

    v2: component_ids are ``<name>/<variants>/<energy>``.
    """
    base = event.window_start  # type: ignore[attr-defined]

    async def fake_kp(_event: object) -> dict[datetime, float]:
        return {}

    async def fake_sb(_event: object) -> list[tuple[datetime, str, float]]:
        return [
            (base, "UMASEP/v2_0/10MeV", 0.35),
            (base + timedelta(hours=12), "UMASEP/v2_0/10MeV", 0.5),
            (base, "SEPSTER/Parker/noE", 0.4),
        ]

    monkeypatch.setattr(loader_module, "_async_pull_scoreboards", fake_sb)
    monkeypatch.setattr(loader_module, "_async_pull_kp", fake_kp)

    frame = load_table_3_1(event, use_real_data=True, cadence_hours=6.0)
    by_model = frame.records.groupby("model_id")["source"].first().to_dict()
    # Components with adapter data are iswa_real.
    assert by_model["UMASEP/v2_0/10MeV"] == "iswa_real"
    assert by_model["SEPSTER/Parker/noE"] == "iswa_real"


def test_real_kp_pull_handles_exception(monkeypatch: pytest.MonkeyPatch, event: object) -> None:
    """If the async Kp helper raises, the gap is recorded and synth fallback used."""

    async def failing_kp(_event: object) -> dict[datetime, float]:
        raise OSError("network down")

    async def empty_sb(_event: object) -> list[tuple[datetime, str, float]]:
        return []

    monkeypatch.setattr(loader_module, "_async_pull_kp", failing_kp)
    monkeypatch.setattr(loader_module, "_async_pull_scoreboards", empty_sb)

    frame = load_table_3_1(event, use_real_data=True, cadence_hours=24.0)
    assert "swpc_kp" in frame.data_gaps
    assert not frame.records.empty


def test_real_scoreboards_pull_handles_exception(
    monkeypatch: pytest.MonkeyPatch, event: object
) -> None:
    """If the async Scoreboards helper raises, the gap is recorded."""

    async def empty_kp(_event: object) -> dict[datetime, float]:
        return {}

    async def failing_sb(_event: object) -> list[tuple[datetime, str, float]]:
        raise ValueError("upstream parse error")

    monkeypatch.setattr(loader_module, "_async_pull_kp", empty_kp)
    monkeypatch.setattr(loader_module, "_async_pull_scoreboards", failing_sb)

    frame = load_table_3_1(event, use_real_data=True, cadence_hours=24.0)
    assert "sep_scoreboards" in frame.data_gaps
    assert "fetch failed" in frame.data_gaps["sep_scoreboards"]


def test_cddis_deferral_recorded_when_no_credentials(
    monkeypatch: pytest.MonkeyPatch, event: object
) -> None:
    """Without NASA_EARTHDATA creds, the CDDIS deferral note must appear."""
    monkeypatch.delenv("NASA_EARTHDATA_USER", raising=False)
    monkeypatch.delenv("NASA_EARTHDATA_PASS", raising=False)

    frame = load_table_3_1(event, use_real_data=False, cadence_hours=24.0)
    assert "cddis_gim" in frame.data_gaps
    assert "earthdata" in frame.data_gaps["cddis_gim"].lower()


def test_cddis_deferral_skipped_when_credentials_set(
    monkeypatch: pytest.MonkeyPatch, event: object
) -> None:
    """When creds are set, no CDDIS deferral note (real fetch path is gated elsewhere)."""
    monkeypatch.setenv("NASA_EARTHDATA_USER", "test-user")
    monkeypatch.setenv("NASA_EARTHDATA_PASS", "test-pass")

    frame = load_table_3_1(event, use_real_data=False, cadence_hours=24.0)
    assert "cddis_gim" not in frame.data_gaps
