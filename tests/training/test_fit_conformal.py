"""Tests for the split + Mondrian conformal fitters."""

from __future__ import annotations

import json

import numpy as np
import pytest

from helios_fusion.conformal.mondrian import MondrianConformalRegressor
from helios_fusion.conformal.split import SplitConformalRegressor
from helios_fusion.training.fit_bma import fit_bma_priors
from helios_fusion.training.fit_conformal import fit_mondrian_conformal, fit_split_conformal
from helios_fusion.training.fit_isotonic import fit_stratified_calibrators
from helios_fusion.training.load_table_3_1 import TRAINING_EVENTS, load_table_3_1


@pytest.fixture(scope="module")
def fitted_conformals() -> tuple[SplitConformalRegressor, MondrianConformalRegressor]:
    frames = [
        load_table_3_1(event, use_real_data=False, cadence_hours=4.0)
        for event in TRAINING_EVENTS[:4]
    ]
    priors = fit_bma_priors(frames)
    cal = fit_stratified_calibrators(frames, priors)
    split = fit_split_conformal(frames, priors, calibrator=cal)
    mondrian = fit_mondrian_conformal(frames, priors, calibrator=cal)
    return split, mondrian


def test_split_conformal_fitted(
    fitted_conformals: tuple[SplitConformalRegressor, MondrianConformalRegressor],
) -> None:
    split, _ = fitted_conformals
    assert split.fitted
    assert split.n_calibration > 0


def test_mondrian_per_stratum_counts_positive(
    fitted_conformals: tuple[SplitConformalRegressor, MondrianConformalRegressor],
) -> None:
    _, mondrian = fitted_conformals
    assert mondrian.fitted
    counts = mondrian.per_stratum_counts()
    for stratum in ("quiet", "moderate", "extreme"):
        assert counts[stratum] >= 1


def test_split_state_round_trip(
    fitted_conformals: tuple[SplitConformalRegressor, MondrianConformalRegressor],
) -> None:
    split, _ = fitted_conformals
    state = split.to_state_dict()
    restored = SplitConformalRegressor.from_state_dict(json.loads(json.dumps(state)))
    probe = np.linspace(0.05, 0.95, 5)
    a = split.predict_interval(probe, alpha=0.1)
    b = restored.predict_interval(probe, alpha=0.1)
    np.testing.assert_allclose(a, b)


def test_mondrian_state_round_trip(
    fitted_conformals: tuple[SplitConformalRegressor, MondrianConformalRegressor],
) -> None:
    _, mondrian = fitted_conformals
    state = mondrian.to_state_dict()
    restored = MondrianConformalRegressor.from_state_dict(json.loads(json.dumps(state)))
    probe = np.linspace(0.05, 0.95, 6)
    strata = ["quiet", "moderate", "extreme"] * 2
    a = mondrian.predict_interval(probe, strata, alpha=0.1)  # type: ignore[arg-type]
    b = restored.predict_interval(probe, strata, alpha=0.1)  # type: ignore[arg-type]
    np.testing.assert_allclose(a, b)
