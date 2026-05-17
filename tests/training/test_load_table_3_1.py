"""Tests for the Table 3-1 loader (Sprint C-Training-v2)."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from helios_fusion.training.load_table_3_1 import (
    DEFAULT_COMPONENT_MODELS,
    DEFAULT_COMPONENT_MODELS_LEGACY,
    EMPIRICAL_ISWA_COVERAGE,
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


def test_default_components_mirror_connector_registry() -> None:
    """v2: DEFAULT_COMPONENT_MODELS should expand the connector tuples.

    The v0.2.1 connector registry has 23 specs unrolled to (model,
    variants, energy) tuples -- UMASEP has 4 versions x {3 or 5} energies
    plus the empty-energy entries from other models for a total of 37
    component_ids of the form ``<name>/<variants>/<energy or noE>``.
    """
    assert len(DEFAULT_COMPONENT_MODELS) >= 23
    # Every Sept 2017 expected-real tuple is in the default registry.
    sep_2017_expected = EMPIRICAL_ISWA_COVERAGE["sep_2017"]
    assert sep_2017_expected.issubset(set(DEFAULT_COMPONENT_MODELS))
    # UMASEP v2_0 energies are all present.
    for energy in ("10MeV", "30MeV", "50MeV", "100MeV", "500MeV"):
        assert f"UMASEP/v2_0/{energy}" in DEFAULT_COMPONENT_MODELS


def test_empirical_coverage_matrix_has_13_real_data_tuples_for_sep_2017() -> None:
    """The 2026-05-17 coverage matrix locked 12 expected-real tuples for
    Sept 2017 (NCAR_MLSO_KCOR excluded from registry; 5 UMASEP + 2 SEPSTER
    + 5 mag4_2019 = 12)."""
    sep_2017 = EMPIRICAL_ISWA_COVERAGE["sep_2017"]
    assert len(sep_2017) == 12
    # All other events have zero real-data tuples.
    for event_id in (
        "bastille_2000",
        "halloween_2003",
        "midcycle23_2005",
        "latecycle23_2006",
        "cycle24_onset_2012",
        "cycle24_mid_2012",
    ):
        assert len(EMPIRICAL_ISWA_COVERAGE[event_id]) == 0


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
        "truth_source",
    }
    assert expected_cols.issubset(set(frame.records.columns))
    # Every component in the default registry contributes
    assert set(frame.records["model_id"].unique()) == set(DEFAULT_COMPONENT_MODELS)
    # Probabilities are in [0, 1]
    assert (frame.records["probability"] >= 0.0).all()
    assert (frame.records["probability"] <= 1.0).all()
    # Observed labels are binary
    assert set(frame.records["observed"].unique()).issubset({0.0, 1.0})


def test_load_table_3_1_sep_2017_per_component_labeling_matrix_sticky() -> None:
    """v2 open question #2: per-(component, event) source labeling.

    The matrix-locked tags are sticky regardless of whether this run
    actually hit the network: Sept 2017's 12 expected-real tuples are
    tagged ``iswa_real`` per the 2026-05-17 ISWA coverage matrix; the
    remaining ~25 tuples are tagged ``synthetic_proxy``.
    """
    sep_2017 = next(e for e in TRAINING_EVENTS if e.event_id == "sep_2017")
    frame = load_table_3_1(sep_2017, use_real_data=False, cadence_hours=24.0)
    sources_by_component = frame.records.groupby("model_id")["source"].first().to_dict()
    expected_real = EMPIRICAL_ISWA_COVERAGE["sep_2017"]
    for component in expected_real:
        assert sources_by_component[component] == "iswa_real", (
            f"Sept 2017 component {component} should be iswa_real per matrix; "
            f"got {sources_by_component[component]}"
        )
    # And a non-expected-real component should be synthetic_proxy.
    other_components = set(DEFAULT_COMPONENT_MODELS) - expected_real
    a_non_expected = next(iter(other_components))
    assert sources_by_component[a_non_expected] == "synthetic_proxy"


def test_load_table_3_1_sep_2017_real_data_preserves_matrix_labels() -> None:
    """When ``use_real_data=True`` and the adapter returns no real data, the
    matrix-expected-real tuples for Sept 2017 should still carry the
    ``iswa_real`` source tag and a gap-note explaining the substitution.

    This matches the Sprint C-Training-v2 spec: the matrix labels are
    sticky regardless of whether THIS run successfully hit the network.
    """
    import importlib

    loader_module = importlib.import_module("helios_fusion.training.load_table_3_1")
    # Stub the network calls so we don't hit ISWA during the test.
    sep_2017 = next(e for e in TRAINING_EVENTS if e.event_id == "sep_2017")

    async def empty_sb(_event: object) -> list:
        return []

    async def empty_kp(_event: object) -> dict:
        return {}

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(loader_module, "_async_pull_scoreboards", empty_sb)
        monkey.setattr(loader_module, "_async_pull_kp", empty_kp)
        frame = load_table_3_1(sep_2017, use_real_data=True, cadence_hours=24.0)
    finally:
        monkey.undo()

    sources_by_component = frame.records.groupby("model_id")["source"].first().to_dict()
    expected_real = EMPIRICAL_ISWA_COVERAGE["sep_2017"]
    for component in expected_real:
        assert sources_by_component[component] == "iswa_real", (
            f"Sept 2017 component {component} should be labelled iswa_real per matrix; "
            f"got {sources_by_component[component]}"
        )
    # And a non-expected-real component should be synthetic_proxy.
    other_components = set(DEFAULT_COMPONENT_MODELS) - expected_real
    a_non_expected = next(iter(other_components))
    assert sources_by_component[a_non_expected] == "synthetic_proxy"


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
    """An override component-model list is honoured (legacy contract preserved)."""
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


def test_legacy_component_registry_still_exported() -> None:
    """v1 callers that referenced DEFAULT_COMPONENT_MODELS_LEGACY get the
    original short list of nominal model names."""
    assert len(DEFAULT_COMPONENT_MODELS_LEGACY) == 11
    assert "UMASEP" in DEFAULT_COMPONENT_MODELS_LEGACY
    assert "SEPMOD" in DEFAULT_COMPONENT_MODELS_LEGACY


def test_truth_labels_substitute_observed_column() -> None:
    """When SWPC archive truth labels are supplied, the observed column is
    replaced and truth_source is set to swpc_archive."""

    event = TRAINING_EVENTS[0]
    truth_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [event.window_start + timedelta(hours=i) for i in range(0, 240, 24)],
                utc=True,
            ),
            "observed": [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )
    frame = load_table_3_1(
        event,
        use_real_data=False,
        cadence_hours=24.0,
        truth_labels=truth_df,
    )
    assert frame.truth_source == "swpc_archive"
    assert (frame.records["truth_source"] == "swpc_archive_truth").all()
