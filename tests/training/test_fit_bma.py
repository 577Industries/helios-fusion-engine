"""Tests for the BMA-weight fitter."""

from __future__ import annotations

import math

import pytest

from helios_fusion.training.fit_bma import fit_bma_priors
from helios_fusion.training.load_table_3_1 import TRAINING_EVENTS, load_table_3_1


def test_weights_sum_to_one_per_event() -> None:
    """Invariant: weight vectors sum to 1 (renormalised by compute_skill_weights)."""
    frames = [
        load_table_3_1(event, use_real_data=False, cadence_hours=6.0)
        for event in TRAINING_EVENTS[:3]
    ]
    priors = fit_bma_priors(frames)
    assert len(priors) == 3
    for event_id, weights in priors.items():
        total = sum(weights.values())
        assert math.isclose(total, 1.0, abs_tol=1e-9), (
            f"event {event_id}: weights sum to {total}, expected 1.0"
        )


def test_weights_are_non_negative() -> None:
    """All weights are >= 0 (epsilon floor + clipped HSS)."""
    frames = [
        load_table_3_1(event, use_real_data=False, cadence_hours=6.0)
        for event in TRAINING_EVENTS[:2]
    ]
    priors = fit_bma_priors(frames)
    for weights in priors.values():
        for w in weights.values():
            assert w >= 0.0


def test_weights_assigned_to_every_model() -> None:
    """Each event's weight vector contains every component model."""
    event = TRAINING_EVENTS[0]
    frame = load_table_3_1(event, use_real_data=False, cadence_hours=6.0)
    priors = fit_bma_priors([frame])
    weights = priors[event.event_id]
    assert set(weights.keys()) == set(frame.component_models)


def test_empty_frame_raises() -> None:
    """Empty dataframe should raise ValueError."""
    import pandas as pd

    from helios_fusion.training.load_table_3_1 import TrainingEvent, TrainingEventFrame

    empty_frame = TrainingEventFrame(
        event=TrainingEvent(event_id="empty", label="empty", onset=TRAINING_EVENTS[0].onset),
        records=pd.DataFrame(),
        data_gaps={},
        component_models=[],
    )
    with pytest.raises(ValueError, match="no rows"):
        fit_bma_priors([empty_frame])
