"""Tests for the stratified isotonic calibrator fitter."""

from __future__ import annotations

import json

import numpy as np
import pytest

from helios_fusion.calibration.stratified import SeverityStratifiedCalibrator
from helios_fusion.training.fit_bma import fit_bma_priors
from helios_fusion.training.fit_isotonic import fit_stratified_calibrators
from helios_fusion.training.load_table_3_1 import TRAINING_EVENTS, load_table_3_1


@pytest.fixture(scope="module")
def fitted_calibrator() -> tuple[SeverityStratifiedCalibrator, dict[str, dict[str, float]]]:
    frames = [
        load_table_3_1(event, use_real_data=False, cadence_hours=4.0)
        for event in TRAINING_EVENTS[:4]
    ]
    priors = fit_bma_priors(frames)
    calibrator = fit_stratified_calibrators(frames, priors)
    return calibrator, priors


def test_calibrator_fitted_all_strata(
    fitted_calibrator: tuple[SeverityStratifiedCalibrator, dict[str, dict[str, float]]],
) -> None:
    calibrator, _ = fitted_calibrator
    assert calibrator.fitted
    counts = calibrator.sample_counts
    for stratum in ("quiet", "moderate", "extreme"):
        assert counts[stratum] >= 2


def test_calibrator_state_round_trip(
    fitted_calibrator: tuple[SeverityStratifiedCalibrator, dict[str, dict[str, float]]],
) -> None:
    """State-dict serialisation must round-trip exactly."""
    calibrator, _ = fitted_calibrator
    state = calibrator.to_state_dict()
    # JSON-serialise then re-load to mimic disk persistence.
    serialised = json.loads(json.dumps(state))
    restored = SeverityStratifiedCalibrator.from_state_dict(serialised)
    assert restored.fitted
    # Transform-equivalence on a probe set
    probe_probs = np.linspace(0.01, 0.99, 9)
    probe_strata = ["quiet"] * 3 + ["moderate"] * 3 + ["extreme"] * 3
    a = calibrator.transform(probe_probs, probe_strata)  # type: ignore[arg-type]
    b = restored.transform(probe_probs, probe_strata)  # type: ignore[arg-type]
    np.testing.assert_allclose(a, b)


def test_missing_bma_prior_raises() -> None:
    """If an event has no BMA prior, fitting must error."""
    frames = [load_table_3_1(TRAINING_EVENTS[0], use_real_data=False, cadence_hours=24.0)]
    with pytest.raises(ValueError, match="missing BMA prior"):
        fit_stratified_calibrators(frames, bma_priors={})
