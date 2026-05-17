"""Tests for the Table 3-1 loader."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from helios_fusion.training.load_table_3_1 import (
    DEFAULT_COMPONENT_MODELS,
    EVENT_WINDOW_HALF_WIDTH_DAYS,
    TRAINING_EVENTS,
    TrainingEvent,
    load_table_3_1,
)


def test_seven_training_events_present() -> None:
    """Spec lock: exactly seven training events."""
    assert len(TRAINING_EVENTS) == 7
    ids = {e.event_id for e in TRAINING_EVENTS}
    assert ids == {
        "bastille_2000",
        "halloween_2003",
        "midcycle23_2005",
        "latecycle23_2006",
        "cycle24_onset_2012",
        "cycle24_mid_2012",
        "sep_2017",
    }


def test_event_window_is_plus_minus_5_days() -> None:
    """Window should be earliest onset - 5d to latest onset + 5d."""
    bastille = TRAINING_EVENTS[0]
    assert bastille.window_end - bastille.window_start >= timedelta(days=2 * 5)
    assert bastille.window_start == bastille.onset - timedelta(days=EVENT_WINDOW_HALF_WIDTH_DAYS)
    assert bastille.window_end == bastille.onset + timedelta(days=EVENT_WINDOW_HALF_WIDTH_DAYS)


def test_dual_event_window_spans_both_onsets() -> None:
    """September 2017 has two onsets; window must cover both +/-5d."""
    sep2017 = next(e for e in TRAINING_EVENTS if e.event_id == "sep_2017")
    assert sep2017.secondary_onsets
    earliest = min((sep2017.onset, *sep2017.secondary_onsets))
    latest = max((sep2017.onset, *sep2017.secondary_onsets))
    assert sep2017.window_start == earliest - timedelta(days=EVENT_WINDOW_HALF_WIDTH_DAYS)
    assert sep2017.window_end == latest + timedelta(days=EVENT_WINDOW_HALF_WIDTH_DAYS)


def test_load_table_3_1_synthetic_only_smoke() -> None:
    """Load an event in synthetic-only mode; verify the dataframe shape."""
    event = TRAINING_EVENTS[0]
    frame = load_table_3_1(event, use_real_data=False, cadence_hours=6.0)
    assert isinstance(frame.records, pd.DataFrame)
    assert not frame.records.empty
    # Columns
    expected_cols = {
        "timestamp",
        "model_id",
        "probability",
        "observed",
        "kp",
        "severity_stratum",
        "source",
    }
    assert expected_cols.issubset(set(frame.records.columns))
    # Every model in the default registry contributes
    assert set(frame.records["model_id"].unique()) == set(DEFAULT_COMPONENT_MODELS)
    # Probabilities are in [0, 1]
    assert (frame.records["probability"] >= 0.0).all()
    assert (frame.records["probability"] <= 1.0).all()
    # Observed labels are binary
    assert set(frame.records["observed"].unique()).issubset({0.0, 1.0})
    # All synthetic; data gaps should note the substitution
    assert any("synthetic" in v for v in frame.data_gaps.values()) or any(
        k.startswith("model::") for k in frame.data_gaps
    )


def test_window_grid_size_matches_cadence() -> None:
    """Cadence is honoured (window length / cadence)."""
    event = TRAINING_EVENTS[2]  # mid-cycle 23
    frame = load_table_3_1(event, use_real_data=False, cadence_hours=12.0)
    n_models = len(DEFAULT_COMPONENT_MODELS)
    n_grid = len(frame.records) // n_models
    window_hours = (event.window_end - event.window_start).total_seconds() / 3600.0
    expected = int(window_hours / 12.0) + 1
    assert n_grid == expected


def test_invalid_window_raises() -> None:
    """Build_time_grid rejects nonsense windows."""
    from datetime import UTC, datetime

    from helios_fusion.training.load_table_3_1 import _build_time_grid

    with pytest.raises(ValueError, match="must follow start"):
        _build_time_grid(datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC), 1.0)
    with pytest.raises(ValueError, match="cadence_hours"):
        _build_time_grid(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC), 0.0)


def test_event_window_bounds_returned_dataframe() -> None:
    """Every timestamp in the dataframe must lie inside the event window."""
    event = TRAINING_EVENTS[3]  # late cycle 23
    frame = load_table_3_1(event, use_real_data=False, cadence_hours=24.0)
    ts = frame.records["timestamp"]
    assert ts.min() >= event.window_start
    assert ts.max() <= event.window_end


def test_unknown_component_model_passed_through() -> None:
    """An override component-model list is honoured."""
    event = TrainingEvent(
        event_id="probe_event",
        label="probe",
        onset=TRAINING_EVENTS[0].onset,
    )
    frame = load_table_3_1(
        event,
        use_real_data=False,
        cadence_hours=12.0,
        component_models=("modelA", "modelB"),
    )
    assert set(frame.records["model_id"].unique()) == {"modelA", "modelB"}
